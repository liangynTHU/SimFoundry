#!/usr/bin/env python3
"""Replay an extracted OXE DROID trajectory in a lightweight PyBullet sim.

This fallback is useful on compute GPUs that cannot run Isaac Sim's RTX renderer
(for example NVIDIA H20).  It creates a Franka Panda simulation, commands the
seven arm joints from ``action_joint_position``, maps the DROID gripper signal,
renders an RGB camera with PyBullet's software renderer, and writes a compact
HDF5 rollout suitable for downstream inspection.

It intentionally validates the data/control bridge only.  The simple table,
marker, and cup are not registered to the original DROID camera geometry, so
this is not a task-faithful replay of the real episode.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pybullet as p
import pybullet_data


PANDA_ARM_JOINTS = tuple(range(7))
PANDA_FINGER_JOINTS = (9, 10)
PANDA_EEF_LINK = 11


class FfmpegWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float, ffmpeg: str):
        self.path = path
        self.width = width
        self.height = height
        self.process = subprocess.Popen(
            [
                ffmpeg,
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
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Expected frame {(self.height, self.width, 3)}, got {frame.shape}"
            )
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        code = self.process.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed with code {code}: {self.path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="trajectory.npz from extract_droid_trajectory.py")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--action-frequency", type=float, default=15.0)
    parser.add_argument("--physics-frequency", type=float, default=240.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-demo-props", action="store_true")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    return parser.parse_args()


def _create_demo_props(client_id: int) -> dict[str, int]:
    """Create a table, cup-like cylinder, and marker-like cylinder."""

    table_id = p.loadURDF(
        "table/table.urdf",
        basePosition=[0.55, 0.0, -0.65],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    cup_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=0.06,
        height=0.12,
        physicsClientId=client_id,
    )
    cup_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.06,
        length=0.12,
        rgbaColor=[0.9, 0.9, 0.95, 1.0],
        physicsClientId=client_id,
    )
    cup_id = p.createMultiBody(
        baseMass=0.2,
        baseCollisionShapeIndex=cup_collision,
        baseVisualShapeIndex=cup_visual,
        basePosition=[0.52, -0.16, 0.06],
        physicsClientId=client_id,
    )

    marker_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=0.012,
        height=0.14,
        physicsClientId=client_id,
    )
    marker_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.012,
        length=0.14,
        rgbaColor=[0.05, 0.05, 0.05, 1.0],
        physicsClientId=client_id,
    )
    marker_id = p.createMultiBody(
        baseMass=0.03,
        baseCollisionShapeIndex=marker_collision,
        baseVisualShapeIndex=marker_visual,
        basePosition=[0.52, -0.16, 0.18],
        baseOrientation=p.getQuaternionFromEuler([0.0, math.pi / 2.0, 0.0]),
        physicsClientId=client_id,
    )
    return {"table": table_id, "cup": cup_id, "marker": marker_id}


def _render_frame(client_id: int, width: int, height: int) -> np.ndarray:
    view = p.computeViewMatrix(
        cameraEyePosition=[1.25, -1.15, 0.95],
        cameraTargetPosition=[0.45, 0.0, 0.35],
        cameraUpVector=[0.0, 0.0, 1.0],
    )
    projection = p.computeProjectionMatrixFOV(
        fov=55.0,
        aspect=width / height,
        nearVal=0.02,
        farVal=5.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]


def main() -> int:
    args = _parse_args()
    trajectory_path = Path(args.trajectory).expanduser().resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)

    trajectory = np.load(trajectory_path)
    source_metadata_path = trajectory_path.parent / "metadata.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    language_instructions = source_metadata.get("trajectory", {}).get(
        "language_instructions", []
    )
    joint_targets = np.asarray(trajectory["action_joint_position"], dtype=np.float32)
    initial_joint = np.asarray(
        trajectory["observation_joint_position"][0], dtype=np.float32
    )
    droid_gripper = np.asarray(
        trajectory["action_gripper_position"], dtype=np.float32
    ).reshape(-1)
    n_steps = min(len(joint_targets), len(droid_gripper))
    if args.max_steps is not None:
        n_steps = min(n_steps, args.max_steps)
    if n_steps <= 0:
        raise ValueError("Trajectory contains no replayable steps.")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else trajectory_path.parent / "pybullet_replay"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = p.GUI if args.gui else p.DIRECT
    client_id = p.connect(mode)
    if client_id < 0:
        raise RuntimeError("Could not connect to PyBullet.")

    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)
    p.setTimeStep(1.0 / args.physics_frequency, physicsClientId=client_id)
    # Put the room floor below the tabletop. The Panda base and props are
    # expressed in the table-top frame (z=0), matching common DROID setups.
    plane_id = p.loadURDF(
        "plane.urdf",
        basePosition=[0.0, 0.0, -0.65],
        physicsClientId=client_id,
    )
    panda_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=client_id,
    )
    props = {} if args.no_demo_props else _create_demo_props(client_id)

    for joint_index, value in zip(PANDA_ARM_JOINTS, initial_joint):
        p.resetJointState(
            panda_id, joint_index, float(value), physicsClientId=client_id
        )
    initial_opening = float(np.clip(1.0 - droid_gripper[0], 0.0, 1.0) * 0.04)
    for finger_index in PANDA_FINGER_JOINTS:
        p.resetJointState(
            panda_id, finger_index, initial_opening, physicsClientId=client_id
        )

    substeps = max(1, round(args.physics_frequency / args.action_frequency))
    writer = (
        None
        if args.no_video
        else FfmpegWriter(
            output_dir / "sim_replay.mp4",
            args.width,
            args.height,
            args.action_frequency,
            args.ffmpeg,
        )
    )

    command_rows: list[np.ndarray] = []
    actual_joint_rows: list[np.ndarray] = []
    actual_gripper_rows: list[np.ndarray] = []
    eef_position_rows: list[np.ndarray] = []
    eef_quaternion_rows: list[np.ndarray] = []
    rgb_rows: list[np.ndarray] = []

    try:
        for step_index in range(n_steps):
            target = joint_targets[step_index]
            droid_grip = float(np.clip(droid_gripper[step_index], 0.0, 1.0))
            sim_gripper_open = 1.0 - droid_grip
            finger_target = sim_gripper_open * 0.04

            p.setJointMotorControlArray(
                panda_id,
                PANDA_ARM_JOINTS,
                p.POSITION_CONTROL,
                targetPositions=target.tolist(),
                forces=[240.0] * 7,
                positionGains=[0.35] * 7,
                velocityGains=[1.0] * 7,
                physicsClientId=client_id,
            )
            p.setJointMotorControlArray(
                panda_id,
                PANDA_FINGER_JOINTS,
                p.POSITION_CONTROL,
                targetPositions=[finger_target, finger_target],
                forces=[40.0, 40.0],
                positionGains=[0.6, 0.6],
                physicsClientId=client_id,
            )
            for _ in range(substeps):
                p.stepSimulation(physicsClientId=client_id)

            joint_states = p.getJointStates(
                panda_id, PANDA_ARM_JOINTS, physicsClientId=client_id
            )
            finger_states = p.getJointStates(
                panda_id, PANDA_FINGER_JOINTS, physicsClientId=client_id
            )
            eef_state = p.getLinkState(
                panda_id,
                PANDA_EEF_LINK,
                computeForwardKinematics=True,
                physicsClientId=client_id,
            )

            command_rows.append(
                np.concatenate([target, [sim_gripper_open]]).astype(np.float32)
            )
            actual_joint_rows.append(
                np.asarray([state[0] for state in joint_states], dtype=np.float32)
            )
            actual_gripper_rows.append(
                np.asarray([state[0] for state in finger_states], dtype=np.float32)
            )
            eef_position_rows.append(np.asarray(eef_state[4], dtype=np.float32))
            eef_quaternion_rows.append(np.asarray(eef_state[5], dtype=np.float32))

            frame = _render_frame(client_id, args.width, args.height)
            rgb_rows.append(frame)
            if writer is not None:
                writer.append(frame)
            if (step_index + 1) % 50 == 0 or step_index + 1 == n_steps:
                print(f"Replayed {step_index + 1}/{n_steps} steps", flush=True)
    finally:
        if writer is not None:
            writer.close()
        p.disconnect(client_id)

    commands = np.asarray(command_rows, dtype=np.float32)
    actual_joints = np.asarray(actual_joint_rows, dtype=np.float32)
    actual_gripper = np.asarray(actual_gripper_rows, dtype=np.float32)
    eef_position = np.asarray(eef_position_rows, dtype=np.float32)
    eef_quaternion = np.asarray(eef_quaternion_rows, dtype=np.float32)
    rgb = np.asarray(rgb_rows, dtype=np.uint8)
    timestamps = (
        np.arange(len(commands), dtype=np.float32) / float(args.action_frequency)
    )
    rewards = np.zeros(len(commands), dtype=np.float32)
    dones = np.zeros(len(commands), dtype=np.bool_)
    if len(dones):
        dones[-1] = True

    np.savez_compressed(
        output_dir / "sim_rollout.npz",
        actions=commands,
        observation_joint_position=actual_joints,
        observation_finger_position=actual_gripper,
        observation_eef_position=eef_position,
        observation_eef_quaternion_xyzw=eef_quaternion,
        observation_rgb=rgb,
        timestamp=timestamps,
        reward=rewards,
        done=dones,
        source_action_joint_position=joint_targets[: len(commands)],
        source_action_gripper_position=droid_gripper[: len(commands), None],
    )

    hdf5_path = output_dir / "sim_rollout.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["n_episodes"] = 1
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = len(commands)
        demo.attrs["source"] = "OXE DROID"
        demo.attrs["source_trajectory"] = str(trajectory_path)
        demo.attrs["language_instructions_json"] = json.dumps(
            language_instructions, ensure_ascii=False
        )
        demo.attrs["action_convention"] = (
            "7 Franka joint-position targets + normalized gripper opening "
            "(0=closed, 1=open)"
        )
        demo.create_dataset("actions", data=commands, compression="gzip")
        demo.create_dataset("timestamps", data=timestamps, compression="gzip")
        demo.create_dataset("rewards", data=rewards, compression="gzip")
        demo.create_dataset("dones", data=dones, compression="gzip")
        obs = demo.create_group("obs")
        obs.create_dataset("joint_position", data=actual_joints, compression="gzip")
        obs.create_dataset("finger_position", data=actual_gripper, compression="gzip")
        obs.create_dataset("eef_position", data=eef_position, compression="gzip")
        obs.create_dataset("eef_quaternion_xyzw", data=eef_quaternion, compression="gzip")
        obs.create_dataset(
            "rgb",
            data=rgb,
            compression="gzip",
            chunks=(1, args.height, args.width, 3),
        )
        source = demo.create_group("source_oxe")
        for key in (
            "legacy_action",
            "observation_joint_position",
            "observation_cartesian_position",
            "observation_gripper_position",
            "action_joint_position",
            "action_gripper_position",
        ):
            if key in trajectory:
                source.create_dataset(
                    key,
                    data=np.asarray(trajectory[key])[: len(commands)],
                    compression="gzip",
                )

    max_error = float(
        np.max(np.abs(actual_joints - joint_targets[: len(actual_joints)]))
    )
    summary = {
        "backend": "pybullet",
        "source_trajectory": str(trajectory_path),
        "language_instructions": language_instructions,
        "num_steps": int(len(commands)),
        "action_frequency": args.action_frequency,
        "physics_frequency": args.physics_frequency,
        "max_abs_joint_tracking_error_rad": max_error,
        "objects": {"plane": plane_id, "panda": panda_id, **props},
        "outputs": {
            "hdf5": str(hdf5_path),
            "npz": str(output_dir / "sim_rollout.npz"),
            "video": None if args.no_video else str(output_dir / "sim_replay.mp4"),
        },
        "limitations": [
            "The scene props are illustrative and are not geometrically registered to the OXE recording.",
            "This validates executable trajectory control and data collection on the current H20 node.",
            "Use the OmniGibson bridge on an RTX-capable node for the repository-native simulator.",
        ],
    }
    (output_dir / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
