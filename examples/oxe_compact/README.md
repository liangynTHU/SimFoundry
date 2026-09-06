# Compact OXE / DROID examples

This directory contains **five self-contained DROID episodes exported from the
Open X-Embodiment RLDS/TFDS dataset**. They are deliberately small enough to be
cloned with the repository and read without access to the original multi-terabyte
TFDS directory.

## Included episodes

| Directory | Steps | Duration | Primary instruction |
|---|---:|---:|---|
| `droid_episode_000000` | 213 | 14.2 s | Push the faucet to the right |
| `droid_episode_000003` | 137 | 9.1 s | Close the cupboard door |
| `droid_episode_000004` | 254 | 16.9 s | Remove the lid from the tall shaker and put it on the table |
| `droid_episode_000005` | 154 | 10.3 s | Push the chair forward |
| `droid_episode_000008` | 302 | 20.1 s | Pour some of the things in the jug into the bowl |

Each episode contains:

```text
metadata.json                 provenance, instructions, shapes, conventions
trajectory.npz                observations, actions, reward and boundaries
exterior_image_1_left.mp4     first external camera
exterior_image_2_left.mp4     second external camera
wrist_image_left.mp4          wrist camera
all_cameras.mp4               three cameras concatenated horizontally
reconstruction_input.mp4      portable copy of exterior_image_1_left.mp4
```

`manifest.json` inventories all episodes and files. `SHA256SUMS` can be used to
verify that the copied data are intact.

## Clone and verify

```bash
git clone git@github.com:liangynTHU/SimFoundry.git
cd SimFoundry
(cd examples/oxe_compact && sha256sum -c SHA256SUMS)
```

The media files in this directory are ordinary Git objects; Git LFS is not
required for these compact examples.

## Read an episode directly

Only NumPy is needed for trajectory inspection:

```bash
python examples/oxe_compact/inspect_episode.py \
  examples/oxe_compact/droid_episode_000003
```

Or from Python:

```python
import json
from pathlib import Path
import numpy as np

root = Path("examples/oxe_compact/droid_episode_000003")
metadata = json.loads((root / "metadata.json").read_text())
with np.load(root / "trajectory.npz", allow_pickle=False) as episode:
    joint_targets = episode["action_joint_position"]   # [T, 7]
    gripper = episode["action_gripper_position"]       # [T, 1]
    images_are_in = root / "all_cameras.mp4"
```

The full key list and array shapes are stored in every `metadata.json`. In the
DROID convention used here, gripper position is `0=open, 1=closed`.

## Use a video as SimFoundry reconstruction input

```bash
bash examples/oxe_compact/run_reconstruction.sh 3
```

To test only video decoding and frame extraction:

```bash
bash examples/oxe_compact/run_reconstruction.sh 3 \
  --include 1b --no-stream
```

These videos generally come from fixed robot cameras while the robot and objects move,
whereas high-quality SimFoundry reconstruction expects a moving camera and an
approximately static scene. They are therefore interface/reproducibility samples,
not a guarantee of task-faithful 3D reconstruction.

## Replay the trajectory

The replay utilities are documented in [`tools/oxe/README_ZH.md`](../../tools/oxe/README_ZH.md).
For example, after installing the SimFoundry/PyBullet environment:

```bash
python tools/oxe/replay_droid_trajectory_in_pybullet.py \
  examples/oxe_compact/droid_episode_000003/trajectory.npz \
  --output-dir /tmp/droid_episode_000003_replay
```

## Re-extract from a local DROID TFDS build

If the full dataset is available locally, use
`tools/oxe/extract_droid_trajectory.py`. The compact files were exported from
DROID dataset version `1.4.0`, split `train`, at 15 FPS. The five indices are
`0`, `3`, `4`, `5`, and `8`.

The upstream dataset's terms, privacy obligations, and attribution requirements
still apply to these derived samples. Review them before redistributing this
repository or using the data beyond research and evaluation.

See [`DATA_NOTICE.md`](DATA_NOTICE.md) for redistribution and upstream-terms guidance.
