# OXE trajectory → 仿真 → 数据采集

本目录是为当前机器补充的 OXE 数据桥接工具。

## 文件

| 文件 | 作用 |
|---|---|
| `extract_droid_trajectory.py` | 从本地 OXE/DROID RLDS（TFDS）目录提取一条 episode，并导出状态、动作和三路相机视频。 |
| `replay_droid_trajectory_in_pybullet.py` | 在当前 H20 节点可运行的 PyBullet Franka 环境中回放关节轨迹，保存 RGB、状态、动作、HDF5、NPZ 和视频。 |
| `replay_droid_trajectory_in_omnigibson.py` | 在支持 Isaac Sim 的 RTX 节点上，把同一轨迹回放到 OmniGibson Franka 环境；可加载 SimFoundry 场景 JSON。 |
| `run_oxe_bridge.sh` | 一键执行“提取 trajectory + 回放 + 数据采集”。当前机器默认使用 PyBullet。 |
| `environment_rlds.yml` | OXE/RLDS 读取环境的最小可复现规格。当前机器已有可用环境 `rlds_env`。 |

## 仓库内置的 5 条紧凑样例

`examples/oxe_compact/` 已包含 episode `0`、`3`、`4`、`5`、`8` 的
`trajectory.npz`、元数据和三路相机视频。读取或回放这些样例不需要挂载原始
TFDS 数据目录：

```bash
python examples/oxe_compact/inspect_episode.py \
  examples/oxe_compact/droid_episode_000003

python tools/oxe/replay_droid_trajectory_in_pybullet.py \
  examples/oxe_compact/droid_episode_000003/trajectory.npz \
  --output-dir /tmp/droid_episode_000003_replay
```

要把样例视频送入 SimFoundry 重建流水线：

```bash
bash examples/oxe_compact/run_reconstruction.sh 3
```

详见 `examples/oxe_compact/README.md`。

## 一键运行

```bash
cd /mnt/lyn/workspace/SimFoundry

# 当前 H20 节点：默认 PyBullet 后端
EPISODE_INDEX=28 BACKEND=pybullet \
  bash tools/oxe/run_oxe_bridge.sh
```

默认数据目录：

```text
/apdcephfs_gy5/share_303588738/peterrao/data/vla-data/OXE/droid/1.4.0
```

默认输出：

```text
artifacts/oxe/droid_episode_000028/
├── metadata.json
├── trajectory.npz
├── exterior_image_1_left.mp4
├── exterior_image_2_left.mp4
├── wrist_image_left.mp4
├── all_cameras.mp4
├── reconstruction_input.mp4
└── pybullet_replay/
    ├── replay_summary.json
    ├── sim_replay.mp4
    ├── sim_rollout.hdf5
    └── sim_rollout.npz
```

只做快速测试：

```bash
OUTPUT_DIR=/tmp/oxe_smoke EPISODE_INDEX=28 BACKEND=pybullet \
  bash tools/oxe/run_oxe_bridge.sh --max-steps 5
```

如果自行创建了 OXE 环境：

```bash
source /jizhicfs/peterrao/miniconda3/bin/activate
conda env create -f tools/oxe/environment_rlds.yml

RLDS_ENV=simfoundry_oxe EPISODE_INDEX=28 BACKEND=pybullet \
  bash tools/oxe/run_oxe_bridge.sh
```

## 只提取 trajectory

```bash
source /jizhicfs/peterrao/miniconda3/bin/activate

CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 \
conda run -n rlds_env python tools/oxe/extract_droid_trajectory.py \
  --dataset-root \
    /apdcephfs_gy5/share_303588738/peterrao/data/vla-data/OXE/droid/1.4.0 \
  --episode-index 28 \
  --output-dir artifacts/oxe/droid_episode_000028 \
  --overwrite
```

`trajectory.npz` 包含：

- `observation_joint_position`: `[T, 7]`
- `observation_cartesian_position`: `[T, 6]`
- `observation_gripper_position`: `[T, 1]`
- `action_joint_position`: `[T, 7]`
- `action_joint_velocity`: `[T, 7]`
- `action_cartesian_position`: `[T, 6]`
- `action_cartesian_velocity`: `[T, 6]`
- `action_gripper_position`: `[T, 1]`
- `action_gripper_velocity`: `[T, 1]`
- `legacy_action`: `[T, 7]`
- reward / discount / episode 边界标志

回放使用 `action_joint_position`。DROID 的夹爪约定是 `0=open,
1=closed`，脚本会转换成各仿真后端需要的控制量。

## PyBullet 数据集结构

输出 HDF5 的主要结构：

```text
data/
└── demo_0/
    ├── actions                       [T, 8]
    ├── timestamps                    [T]
    ├── rewards                       [T]
    ├── dones                         [T]
    ├── obs/
    │   ├── joint_position            [T, 7]
    │   ├── finger_position           [T, 2]
    │   ├── eef_position              [T, 3]
    │   ├── eef_quaternion_xyzw       [T, 4]
    │   └── rgb                       [T, H, W, 3]
    └── source_oxe/
        ├── action_joint_position
        ├── action_gripper_position
        ├── legacy_action
        └── original observations/actions
```

## OmniGibson 后端

在 **RTX-capable GPU** 节点上：

```bash
source scripts/local/activate_simfoundry.sh

BACKEND=omnigibson EPISODE_INDEX=28 \
  bash tools/oxe/run_oxe_bridge.sh \
  --scene-json /path/to/reconstructed_og_scene.json
```

如果不传 `--scene-json`，脚本创建 floor + Franka 的最小环境。

当前 H20 节点会主动拒绝启动该后端。原因不是 CUDA 计算不可用：
PyTorch 和 FAISS GPU 均已验证可用；问题在于 Isaac Sim 的渲染器要求
带 RT Cores 的 RTX GPU。当前节点上即使补齐 Vulkan ICD 后，Isaac Sim
仍在 renderer 初始化阶段触发 GPU crash。

## 重要限制

1. PyBullet 场景里的桌子、杯子和 marker 是示意物体，未与原始 DROID
   录制做 3D 配准，因此目前验证的是**机器人轨迹控制与数据采集闭环**，
   不是原任务的物理成功复现。
2. OXE/DROID 的外部相机通常是固定相机，且 episode 中机器人和物体在动；
   SimFoundry A 流水线需要带视差的移动相机扫拍和近似静态场景。因此不能
   把任意 DROID episode 当作高质量 real-to-sim 重建输入。
3. 真正做任务级 replay，需要：
   - 对同一场景做符合 SimFoundry 要求的移动相机扫描，或手工准备场景；
   - 标定真实 Franka base / 相机 / 物体坐标到仿真坐标；
   - 在 RTX 节点加载 `reconstructed_og_scene.json`；
   - 再回放或重定向 OXE 动作并验证接触和成功条件。
