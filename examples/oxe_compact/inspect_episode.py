#!/usr/bin/env python3
"""Inspect one compact OXE/DROID episode without TensorFlow or RLDS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--show-first", type=int, default=0, help="Print the first N rows of each trajectory array.")
    args = parser.parse_args()

    episode_dir = args.episode_dir.expanduser().resolve()
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / "trajectory.npz", allow_pickle=False) as trajectory:
        print(json.dumps(metadata["source"], indent=2, ensure_ascii=False))
        print("\nTrajectory:")
        for key in sorted(trajectory.files):
            array = trajectory[key]
            print(f"  {key:34s} shape={str(array.shape):16s} dtype={array.dtype}")
            if args.show_first:
                print(array[: args.show_first])

    print("\nVideos:")
    for video in sorted(episode_dir.glob("*.mp4")):
        kind = "symlink" if video.is_symlink() else "file"
        print(f"  {video.name:32s} {kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
