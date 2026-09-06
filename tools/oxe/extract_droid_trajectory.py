#!/usr/bin/env python3
"""Extract one DROID trajectory from a local OXE/RLDS TFDS directory.

The extractor writes:

* ``trajectory.npz``: robot observations and actions
* one MP4 per camera plus ``all_cameras.mp4``
* ``reconstruction_input.mp4``: a symlink/copy to the selected exterior camera
* ``metadata.json``: provenance, task text, shapes, and action conventions

Run this script in an environment containing TensorFlow and tensorflow-datasets.
The existing ``rlds_env`` Conda environment on this machine is sufficient.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET_ROOT = os.environ.get("OXE_DROID_ROOT")
DEFAULT_CAMERAS = (
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
)


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _decode(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_decode(val) for val in value]
    return value


class FfmpegWriter:
    """Small raw-RGB-to-MP4 writer with no Python video dependency."""

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
        frame = np.asarray(frame)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"{self.path.name}: expected {(self.height, self.width, 3)}, "
                f"got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path} with code {return_code}")


def _unique_nonempty(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in output:
            output.append(value)
    return output


def _stack(rows: list[np.ndarray], *, dtype: np.dtype | None = None) -> np.ndarray:
    array = np.stack(rows, axis=0)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        required=DEFAULT_DATASET_ROOT is None,
        help="Local TFDS builder directory. Can also be set with OXE_DROID_ROOT.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: artifacts/oxe/droid_episode_<index> under the current checkout.",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help="Observation image keys to export.",
    )
    parser.add_argument(
        "--reconstruction-camera",
        default="exterior_image_1_left",
        help="Camera MP4 exposed as reconstruction_input.mp4.",
    )
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise SystemExit(
            "tensorflow-datasets is required. On this machine run with:\n"
            "  conda run -n rlds_env python tools/oxe/extract_droid_trajectory.py ..."
        ) from exc

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"OXE dataset directory does not exist: {dataset_root}")

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else Path.cwd() / "artifacts" / "oxe" / f"droid_episode_{args.episode_index:06d}"
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Pass --overwrite to replace its generated files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TFDS metadata from: {dataset_root}", flush=True)
    builder = tfds.builder_from_directory(str(dataset_root))
    split_expr = f"{args.split}[{args.episode_index}:{args.episode_index + 1}]"
    dataset = builder.as_dataset(
        split=split_expr,
        shuffle_files=False,
        read_config=tfds.ReadConfig(try_autocache=False),
    )
    iterator = iter(tfds.as_numpy(dataset))
    try:
        episode = next(iterator)
    except StopIteration as exc:
        raise IndexError(
            f"No episode {args.episode_index} in split {args.split!r}"
        ) from exc

    arrays: dict[str, list[np.ndarray]] = {
        "observation_joint_position": [],
        "observation_cartesian_position": [],
        "observation_gripper_position": [],
        "action_joint_position": [],
        "action_joint_velocity": [],
        "action_cartesian_position": [],
        "action_cartesian_velocity": [],
        "action_gripper_position": [],
        "action_gripper_velocity": [],
        "legacy_action": [],
        "reward": [],
        "discount": [],
        "is_first": [],
        "is_last": [],
        "is_terminal": [],
    }
    instructions: list[str] = []
    video_writers: dict[str, FfmpegWriter] = {}
    combined_writer: FfmpegWriter | None = None
    camera_shapes: dict[str, list[int]] = {}

    try:
        for step_index, step in enumerate(episode["steps"]):
            if args.max_steps is not None and step_index >= args.max_steps:
                break

            observation = step["observation"]
            action_dict = step["action_dict"]

            arrays["observation_joint_position"].append(
                np.asarray(observation["joint_position"], dtype=np.float32)
            )
            arrays["observation_cartesian_position"].append(
                np.asarray(observation["cartesian_position"], dtype=np.float32)
            )
            arrays["observation_gripper_position"].append(
                np.asarray(observation["gripper_position"], dtype=np.float32)
            )
            arrays["action_joint_position"].append(
                np.asarray(action_dict["joint_position"], dtype=np.float32)
            )
            arrays["action_joint_velocity"].append(
                np.asarray(action_dict["joint_velocity"], dtype=np.float32)
            )
            arrays["action_cartesian_position"].append(
                np.asarray(action_dict["cartesian_position"], dtype=np.float32)
            )
            arrays["action_cartesian_velocity"].append(
                np.asarray(action_dict["cartesian_velocity"], dtype=np.float32)
            )
            arrays["action_gripper_position"].append(
                np.asarray(action_dict["gripper_position"], dtype=np.float32)
            )
            arrays["action_gripper_velocity"].append(
                np.asarray(action_dict["gripper_velocity"], dtype=np.float32)
            )
            arrays["legacy_action"].append(np.asarray(step["action"], dtype=np.float32))
            arrays["reward"].append(np.asarray(step["reward"], dtype=np.float32))
            arrays["discount"].append(np.asarray(step["discount"], dtype=np.float32))
            arrays["is_first"].append(np.asarray(step["is_first"], dtype=np.bool_))
            arrays["is_last"].append(np.asarray(step["is_last"], dtype=np.bool_))
            arrays["is_terminal"].append(np.asarray(step["is_terminal"], dtype=np.bool_))

            for language_key in (
                "language_instruction",
                "language_instruction_2",
                "language_instruction_3",
            ):
                instructions.append(_decode(step[language_key]))

            frames: list[np.ndarray] = []
            for camera in args.cameras:
                if camera not in observation:
                    if step_index == 0:
                        print(f"WARNING: camera key not present: {camera}", file=sys.stderr)
                    continue
                frame = np.asarray(observation[camera], dtype=np.uint8)
                if frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError(f"Unexpected frame shape for {camera}: {frame.shape}")
                height, width = frame.shape[:2]
                camera_shapes[camera] = [height, width, 3]
                if camera not in video_writers:
                    video_writers[camera] = FfmpegWriter(
                        output_dir / f"{camera}.mp4",
                        width=width,
                        height=height,
                        fps=args.fps,
                        ffmpeg=args.ffmpeg,
                    )
                video_writers[camera].append(frame)
                frames.append(frame)

            if frames:
                common_height = frames[0].shape[0]
                if any(frame.shape[0] != common_height for frame in frames):
                    raise ValueError("Camera heights differ; combined video cannot be made.")
                combined = np.concatenate(frames, axis=1)
                if combined_writer is None:
                    combined_writer = FfmpegWriter(
                        output_dir / "all_cameras.mp4",
                        width=combined.shape[1],
                        height=combined.shape[0],
                        fps=args.fps,
                        ffmpeg=args.ffmpeg,
                    )
                combined_writer.append(combined)

            if (step_index + 1) % 100 == 0:
                print(f"Extracted {step_index + 1} steps...", flush=True)
    finally:
        for writer in video_writers.values():
            writer.close()
        if combined_writer is not None:
            combined_writer.close()

    if not arrays["legacy_action"]:
        raise RuntimeError("Selected trajectory contained zero steps.")

    packed: dict[str, np.ndarray] = {}
    for key, rows in arrays.items():
        if key.startswith("is_"):
            packed[key] = _stack(rows, dtype=np.bool_)
        elif key in {"reward", "discount"}:
            packed[key] = np.asarray(rows, dtype=np.float32).reshape(-1)
        else:
            packed[key] = _stack(rows, dtype=np.float32)
    np.savez_compressed(output_dir / "trajectory.npz", **packed)

    instructions = _unique_nonempty(instructions)
    metadata = {
        "source": {
            "format": "OXE / RLDS / TFDS",
            "dataset_name": builder.name,
            "dataset_version": str(builder.version),
            "dataset_root": str(dataset_root),
            "split": args.split,
            "episode_index": args.episode_index,
            "episode_metadata": _decode(episode.get("episode_metadata", {})),
        },
        "trajectory": {
            "num_steps": int(packed["legacy_action"].shape[0]),
            "fps": args.fps,
            "duration_seconds": float(packed["legacy_action"].shape[0] / args.fps),
            "language_instructions": instructions,
            "camera_shapes": camera_shapes,
        },
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in packed.items()
        },
        "action_conventions": {
            "legacy_action": (
                "DROID canonical action from this OXE build: Cartesian pose "
                "[x,y,z,roll,pitch,yaw] + gripper."
            ),
            "action_joint_position": (
                "Commanded seven-DoF Franka joint positions from action_dict; "
                "used by the included OmniGibson replay bridge."
            ),
            "action_gripper_position": (
                "DROID convention used by SimFoundry policy code: 0=open, 1=closed. "
                "The replay bridge inverts it for OmniGibson."
            ),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    selected_video = output_dir / f"{args.reconstruction_camera}.mp4"
    reconstruction_video = output_dir / "reconstruction_input.mp4"
    if selected_video.exists():
        reconstruction_video.unlink(missing_ok=True)
        try:
            reconstruction_video.symlink_to(selected_video.name)
        except OSError:
            shutil.copy2(selected_video, reconstruction_video)

    print("\nExtraction complete")
    print(f"  Output:      {output_dir}")
    print(f"  Steps:       {metadata['trajectory']['num_steps']}")
    print(f"  Instruction: {instructions[0] if instructions else '<empty>'}")
    print(f"  Arrays:      {output_dir / 'trajectory.npz'}")
    print(f"  Video:       {output_dir / 'all_cameras.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
