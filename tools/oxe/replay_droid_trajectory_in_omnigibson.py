#!/usr/bin/env python3
"""Replay an extracted OXE DROID joint trajectory in an OmniGibson Franka scene.

This is a bridge / smoke-test utility, not automatic real-to-sim registration:

* With no ``--scene-json``, it creates a floor + Franka environment.
* With ``--scene-json``, it loads a SimFoundry-generated scene.
* It commands the seven Franka joints with DROID ``action_joint_position`` and
  converts the DROID gripper convention (0=open, 1=closed) to OmniGibson's
  convention (0=closed, 1=open).
* It records a compact HDF5 rollout, NPZ arrays, and an RGB video.

Geometric alignment of an arbitrary OXE scene, objects, cameras, and robot base
is outside the scope of this bridge and must be calibrated for task-level replay.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _as_uint8_rgb(frame: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
    except ImportError:
        pass
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = frame * 255 if frame.size and frame.max() <= 1.0 else frame
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    if frame.shape[0] == target_height:
        return frame
    from PIL import Image

    target_width = max(1, round(frame.shape[1] * target_height / frame.shape[0]))
    return np.asarray(
        Image.fromarray(frame).resize(
            (target_width, target_height), Image.Resampling.BILINEAR
        )
    )


class FfmpegWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float):
        self.path = path
        self.width = width
        self.height = height
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def append(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        code = self.process.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed with code {code}: {self.path}")


def _collect_frame(obs: dict[str, Any], robot: Any) -> np.ndarray | None:
    frames: list[np.ndarray] = []
    robot_obs = obs.get(robot.name, {})
    for camera_obs in robot_obs.values():
        if isinstance(camera_obs, dict) and "rgb" in camera_obs:
            frames.append(_as_uint8_rgb(camera_obs["rgb"]))
    for camera_obs in obs.get("external", {}).values():
        if isinstance(camera_obs, dict) and "rgb" in camera_obs:
            frames.append(_as_uint8_rgb(camera_obs["rgb"]))
    if not frames:
        return None
    height = max(frame.shape[0] for frame in frames)
    return np.concatenate([_resize_to_height(frame, height) for frame in frames], axis=1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="trajectory.npz produced by the extractor")
    parser.add_argument(
        "--scene-json",
        default=None,
        help="Optional SimFoundry s13_og/reconstructed_og_scene.json.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--external-sensors-config", default="nv_franka_droid")
    parser.add_argument("--action-frequency", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument(
        "--robot-position",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.gui:
        os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

    import torch as th

    from simfoundry import CFG_DIR, configure_omnigibson_data_path, import_og_dependencies

    configure_omnigibson_data_path(force=True)
    import_og_dependencies()

    import omnigibson as og
    from omnigibson.utils.config_utils import parse_config
    from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros
    from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

    trajectory_path = Path(args.trajectory).expanduser().resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    trajectory = np.load(trajectory_path)
    joint_targets = np.asarray(trajectory["action_joint_position"], dtype=np.float32)
    droid_gripper = np.asarray(
        trajectory["action_gripper_position"], dtype=np.float32
    ).reshape(-1)
    initial_joint = np.asarray(
        trajectory["observation_joint_position"][0], dtype=np.float32
    )

    n_steps = min(len(joint_targets), len(droid_gripper))
    if args.max_steps is not None:
        n_steps = min(n_steps, args.max_steps)
    if n_steps <= 0:
        raise ValueError("Trajectory has no replayable steps.")

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else trajectory_path.parent / "omnigibson_replay"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    external_cfg_path = (
        Path(CFG_DIR) / "external_sensors" / f"{args.external_sensors_config}.yaml"
    )
    external_sensors = parse_config(str(external_cfg_path))["external_sensors"]
    env_cfg = {
        "external_sensors": external_sensors,
        "action_frequency": args.action_frequency,
        "rendering_frequency": args.action_frequency,
        "physics_frequency": 120,
    }

    if args.scene_json:
        scene_path = Path(args.scene_json).expanduser().resolve()
        if not scene_path.is_file():
            raise FileNotFoundError(scene_path)
        scene_cfg = {
            "type": "Scene",
            "scene_file": load_json_with_absolute_usd_paths(str(scene_path)),
            "use_floor_plane": True,
            "floor_plane_visible": True,
            "use_skybox": True,
            "include_robots": True,
        }
        og_cfg = {"env": env_cfg, "scene": scene_cfg}
    else:
        scene_cfg = {
            "type": "Scene",
            "use_floor_plane": True,
            "floor_plane_visible": True,
            "use_skybox": True,
        }
        robot_cfg = {
            "type": "FrankaPanda",
            "name": "robot0",
            "end_effector": "gripper",
            "self_collisions": False,
            "obs_modalities": ["rgb", "depth_linear", "proprio"],
            "action_normalize": False,
            "grasping_mode": "assisted",
            "position": list(args.robot_position),
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "controller_config": {
                "arm_0": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "smoothing_filter_size": 5,
                },
                "gripper_0": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": [0.0, 1.0],
                    "mode": "smooth",
                    "inverted": True,
                },
            },
        }
        og_cfg = {"env": env_cfg, "scene": scene_cfg, "robots": [robot_cfg]}

    apply_teleop_omnigibson_macros(enable_tr=False)
    env = og.Environment(configs=og_cfg)
    obs, _ = env.reset()
    if len(env.robots) != 1:
        raise RuntimeError(f"Expected one robot, found {len(env.robots)}")
    robot = env.robots[0]

    # A loaded scene may carry an IK controller. Force absolute joint-position
    # control, matching the SimFoundry policy evaluation stage.
    controller_cfg = {}
    for arm in robot.arm_names:
        controller_cfg[f"arm_{arm}"] = {
            "name": "JointController",
            "motor_type": "position",
            "command_input_limits": None,
            "use_delta_commands": False,
            "smoothing_filter_size": 5,
        }
        controller_cfg[f"gripper_{arm}"] = {
            "name": "MultiFingerGripperController",
            "command_input_limits": (0.0, 1.0),
            "mode": "smooth",
            "inverted": True,
        }
    robot.reload_controllers(controller_cfg)

    arm_indices = robot.arm_control_idx[robot.default_arm]
    robot.set_joint_positions(th.as_tensor(initial_joint), indices=arm_indices)
    robot.keep_still()
    for _ in range(args.settle_steps):
        og.sim.step()
    env.scene.update_initial_file()
    obs, _ = env.reset()

    command_rows: list[np.ndarray] = []
    qpos_rows: list[np.ndarray] = []
    gripper_rows: list[float] = []
    reward_rows: list[float] = []
    terminated_rows: list[bool] = []
    video_writer: FfmpegWriter | None = None

    try:
        for index in range(n_steps):
            action = np.zeros(robot.action_dim, dtype=np.float32)
            action[:7] = joint_targets[index]
            # DROID: 0=open, 1=closed. OmniGibson: 0=closed, 1=open.
            action[7] = np.clip(1.0 - droid_gripper[index], 0.0, 1.0)

            obs, reward, terminated, truncated, _ = env.step(th.from_numpy(action))
            qpos = robot.get_joint_positions()
            qpos = qpos.detach().cpu().numpy() if hasattr(qpos, "detach") else np.asarray(qpos)
            command_rows.append(action.copy())
            qpos_rows.append(np.asarray(qpos[:7], dtype=np.float32))
            gripper_rows.append(float(np.asarray(qpos[-2:]).mean()))
            reward_rows.append(float(reward))
            terminated_rows.append(bool(terminated or truncated))

            frame = _collect_frame(obs, robot)
            if frame is not None:
                if video_writer is None:
                    video_writer = FfmpegWriter(
                        output_dir / "sim_replay.mp4",
                        width=frame.shape[1],
                        height=frame.shape[0],
                        fps=args.action_frequency,
                    )
                video_writer.append(frame)

            if (index + 1) % 50 == 0 or index + 1 == n_steps:
                print(f"Replayed {index + 1}/{n_steps} steps", flush=True)
            if terminated or truncated:
                print(f"Environment ended at step {index + 1}; stopping replay.")
                break
    except BaseException:
        if video_writer is not None:
            video_writer.close()
        og.shutdown(due_to_signal=True)
        raise

    if video_writer is not None:
        video_writer.close()

    commands = np.asarray(command_rows, dtype=np.float32)
    sim_qpos = np.asarray(qpos_rows, dtype=np.float32)
    sim_gripper = np.asarray(gripper_rows, dtype=np.float32)
    rewards = np.asarray(reward_rows, dtype=np.float32)
    terminated = np.asarray(terminated_rows, dtype=np.bool_)
    np.savez_compressed(
        output_dir / "sim_rollout.npz",
        actions=commands,
        observation_joint_position=sim_qpos,
        observation_gripper_position=sim_gripper,
        reward=rewards,
        terminated=terminated,
        source_action_joint_position=joint_targets[: len(commands)],
        source_action_gripper_position=droid_gripper[: len(commands)],
    )

    hdf5_path = output_dir / "sim_rollout.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["n_episodes"] = 1
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = len(commands)
        demo.attrs["source"] = "OXE DROID"
        demo.attrs["source_trajectory"] = str(trajectory_path)
        demo.create_dataset("actions", data=commands, compression="gzip")
        demo.create_dataset("rewards", data=rewards, compression="gzip")
        demo.create_dataset("dones", data=terminated, compression="gzip")
        obs_group = demo.create_group("obs")
        obs_group.create_dataset(
            "joint_position", data=sim_qpos, compression="gzip"
        )
        obs_group.create_dataset(
            "gripper_position", data=sim_gripper[:, None], compression="gzip"
        )
        source = demo.create_group("source_oxe")
        source.create_dataset(
            "action_joint_position",
            data=joint_targets[: len(commands)],
            compression="gzip",
        )
        source.create_dataset(
            "action_gripper_position",
            data=droid_gripper[: len(commands), None],
            compression="gzip",
        )

    max_joint_error = (
        float(np.max(np.abs(sim_qpos - joint_targets[: len(sim_qpos)])))
        if len(sim_qpos)
        else None
    )
    summary = {
        "source_trajectory": str(trajectory_path),
        "scene_json": str(Path(args.scene_json).resolve()) if args.scene_json else None,
        "num_steps_requested": n_steps,
        "num_steps_replayed": int(len(commands)),
        "action_frequency": args.action_frequency,
        "max_abs_joint_tracking_error_rad": max_joint_error,
        "outputs": {
            "hdf5": str(hdf5_path),
            "npz": str(output_dir / "sim_rollout.npz"),
            "video": str(output_dir / "sim_replay.mp4"),
        },
        "limitations": [
            "This replays robot joint commands; it does not infer or register OXE objects.",
            "Task-level interaction requires a matching reconstructed scene and calibrated transforms.",
        ],
    }
    (output_dir / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    og.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
