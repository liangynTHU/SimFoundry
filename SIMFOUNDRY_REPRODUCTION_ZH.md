# SimFoundry 复现与 OXE 仿真数据采集报告

> 记录时区：Asia/Shanghai
> 仓库：`/mnt/lyn/workspace/SimFoundry`
> 上游提交：`617faccd36b07017def30db84ea897c91c577eae`

## 1. 结论

本次已经完成一个可执行的最小闭环：

```text
本地 OXE/DROID RLDS
        │
        ├─ 提取一条 trajectory、三路相机视频、语言指令
        │
        ├─ 将外部相机视频送入 SimFoundry A-1b，成功生成 1056 帧和 15 个采样帧
        │
        └─ 将 7-DoF Franka 轨迹送入 PyBullet 仿真
               └─ 采集 RGB + joint + gripper + EEF + action
                  并保存 HDF5 / NPZ / MP4
```

当前机器上的最终状态：

| 项目 | 状态 |
|---|---|
| Clone 到 `/mnt/lyn/workspace/SimFoundry` | ✅ |
| `simfoundry` Conda 核心环境 | ✅ |
| PyTorch CUDA / H20 | ✅ |
| OmniGibson 3.8.0 安装与 import | ✅ |
| Franka Panda / Franka Robotiq 资产 | ✅ |
| FAISS GPU | ✅ |
| `da3`（Depth Anything 3）环境 | ✅ |
| `any6d`（姿态估计）环境 | ✅ |
| `hunyuan`（Hunyuan3D-2.1）环境与 CUDA 扩展 | ✅ |
| 仓库完整单元测试 | ✅ `240 passed, 29 skipped` |
| OXE/DROID trajectory 提取 | ✅ |
| SimFoundry Stage 1b 接收 OXE 视频 | ✅ |
| 当前机器上的可运行仿真与数据采集 | ✅ PyBullet |
| 当前机器上的原生 OmniGibson rollout | ❌ H20 不满足 Isaac Sim RTX 渲染要求 |
| SimFoundry A 1–13 全流程 | ⏸ 未运行；缺少模型凭据/checkpoint，且最终 OG 阶段受当前 H20/RTX 渲染兼容性限制 |

因此：

- 如果目标是“**从 OXE 取一条 trajectory，在当前节点的 sim 里执行并收集数据**”，
  已经可以直接使用。
- 如果目标是“**在 SimFoundry 原生 OmniGibson 场景中完成任务级复现**”，
  需要换到 RTX-capable GPU 节点，并完成场景配准和模型凭据配置。

---

## 2. 仓库与机器信息

### 2.1 仓库

```text
路径：/mnt/lyn/workspace/SimFoundry
remote：https://github.com/NVlabs/SimFoundry.git
commit：617faccd36b07017def30db84ea897c91c577eae
提交说明：Fix casting issue in decompose step
```

GitHub 直连最初超时，最终使用机器已有内部代理完成 clone。

### 2.2 硬件

```text
GPU：8 × NVIDIA H20
显存：每卡约 97,871 MiB
Compute capability：9.0
Driver：535.161.08
系统 CUDA toolkit：13.2
```

PyTorch 使用自身的 CUDA 12.8 runtime：

```text
torch 2.7.0+cu128
torch.cuda.is_available() == True
```

### 2.3 Conda

用户指定的 Conda：

```bash
source /jizhicfs/peterrao/miniconda3/bin/activate
```

本机 Conda 有 libmamba solver，但没有独立 `mamba` 可执行文件。为了兼容
上游脚本，在仓库中加入了：

```text
scripts/local/mamba
```

它将上游的 `mamba create/install/run/activate` 调用委托给当前 Conda。

---

## 3. 已配置环境

### 3.1 核心环境

环境名：

```text
simfoundry
```

关键版本：

```text
Python       3.11.6
PyTorch      2.7.0+cu128
OmniGibson   3.8.0
LeRobot      0.3.4
OpenPI client 0.1.2
TorchCodec   0.5
Gymnasium    0.29.1
NumPy        1.26.4
FAISS GPU    1.12.0
```

已验证：

- `simfoundry` 从当前 checkout 导入；
- CUDA 能识别 H20；
- FAISS GPU 能创建 GPU index 并执行 search；
- OmniGibson、LeRobot、OmniGibson data wrapper 均可 import；
- Panda 和 Robotiq USD 机器人资产存在。

说明：上游安装过程混合了 Isaac Sim、OmniGibson、SAM3、Google GenAI 等多个
依赖集合，`pip check` 仍会报告若干声明层面的版本冲突（例如 Isaac Sim 与
Google GenAI 对 `websockets` 的约束不同）。关键运行模块均已实际 import，
OpenPI/LeRobot 的缺失依赖也已补齐，完整仓库测试已通过；本报告不把
`pip check` 声明为全绿。

激活方式：

```bash
cd /mnt/lyn/workspace/SimFoundry
source scripts/local/activate_simfoundry.sh
```

重新收尾或在相同机器重建核心环境：

```bash
cd /mnt/lyn/workspace/SimFoundry
SIMFOUNDRY_HTTP_PROXY=http://star-proxy.oa.com:3128 \
  bash scripts/local/install_core_this_machine.sh
```

### 3.2 OXE 读取环境

这台机器已经有可用环境：

```text
rlds_env
Python 3.9
TensorFlow 2.13.0
tensorflow-datasets 4.9.2
```

也提供了最小环境规格：

```text
tools/oxe/environment_rlds.yml
```

如需新建：

```bash
source /jizhicfs/peterrao/miniconda3/bin/activate
conda env create -f tools/oxe/environment_rlds.yml
```

之后运行桥接脚本时指定：

```bash
RLDS_ENV=simfoundry_oxe bash tools/oxe/run_oxe_bridge.sh
```

### 3.3 重建辅助环境

除核心环境外，已经完成并验证以下三个官方独立环境：

| 环境 | 用途 | 已验证组件 |
|---|---|---|
| `da3` | Stage 2 深度估计 | PyTorch `2.7.0+cu128`、Depth Anything 3、xFormers、gsplat、pycolmap、FAISS GPU |
| `any6d` | 姿态估计相关组件 | nvdiffrast、Kaolin `0.18.0`、PyTorch3D `0.7.8`、FoundationPose `common` / `gridencoder` CUDA 扩展、SAM2、BOP Toolkit、FAISS GPU |
| `hunyuan` | Stage 7 物体网格生成 | Hunyuan3D-2.1、Blender Python `4.0.0`、shape/texture pipeline、custom rasterizer、mesh painter |

源码固定版本：

```text
Depth-Anything-3  3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
Any6D             80eb4866a1c96ecb18be18836aba4f4bd6e80e9e
Hunyuan3D-2.1     82920d643c0dc2f7bfd7255f45f62d386edfe60c
```

`hunyuan` 环境的关键版本：

```text
Python       3.10.12
PyTorch      2.7.0+cu128
NumPy        2.2.6
SciPy        1.14.1
Open3D       0.19.0
ONNXRuntime  1.19.2
OpenCV       4.11.0
Transformers 4.46.0
Diffusers    0.30.0
bpy          4.0.0
```

为使上游 Hunyuan 安装器能在本机完成，已处理：

1. `tb_nightly==2.18.0a20240726` 已从包索引下架：改用
   `tensorboard==2.18.0`，并安装仅用于依赖解析的同版本
   `tb-nightly` 元数据 shim；
2. 为 `bpy` 补齐 Conda 内的 X11/OpenGL 动态库，并写入环境激活钩子；
3. 不再依赖缺失的 `python3-config`，改由 Python `sysconfig` 得到扩展后缀；
4. 编译 `custom_rasterizer_kernel` 与 `mesh_inpaint_processor`；
5. 对齐 NumPy 2 / Open3D / ONNXRuntime / OpenCV，并固定
   `google-genai==1.67.0` 与 `pydantic==2.10.6`。OpenCV 使用 headless
   wheel 作为实际运行库，并用元数据 shim 满足 BasicSR/RealESRGAN 对
   `opencv-python` 包名的旧依赖。

H20 上已实际执行 custom rasterizer 的 CPU/GPU 小样例，两端均得到
`72` 个覆盖像素，说明编译产物不仅能 import，也能真正调用 CUDA kernel。

统一验证日志：

```text
logs/final_multi_env_validation.log
```

---

## 4. OXE 数据检查与选择

OXE 根目录：

```text
/apdcephfs_gy5/share_303588738/peterrao/data/vla-data/OXE
```

该目录下检查到 27 个数据集。最终选择：

```text
/apdcephfs_gy5/share_303588738/peterrao/data/vla-data/OXE/droid/1.4.0
```

选择 DROID 的原因：

1. 机器人是 Franka，与 SimFoundry 默认 Franka 配置最接近；
2. 包含 7 维关节位置、关节速度、笛卡尔位姿、笛卡尔速度和夹爪；
3. 包含两路外部相机和一路腕部相机；
4. 数据频率为 15 Hz，与 SimFoundry 默认应用频率一致；
5. SimFoundry 的 policy adapter 本身也包含 DROID/OpenPI/GR00T 约定。

DROID 本地 TFDS 元数据：

```text
版本：1.4.0
训练 episodes：92,233
TFRecord shards：2,048
```

### 4.1 选定 trajectory

选定：

```text
split=train
episode_index=28
steps=1056
fps=15
duration=70.4 秒
```

语言描述：

```text
Pick the marker and place it in the pot
Place the marker inside the pot
Put the sharpie in the pot
```

三个语言标注语义一致，而且该任务与仓库已有的 marker/container 示例较接近。

提取产物：

```text
artifacts/oxe/droid_episode_000028/
├── metadata.json
├── trajectory.npz
├── exterior_image_1_left.mp4
├── exterior_image_2_left.mp4
├── wrist_image_left.mp4
├── all_cameras.mp4
└── reconstruction_input.mp4
```

其中 `reconstruction_input.mp4` 指向第一路外部相机。

---

## 5. 已完成的 OXE → SimFoundry 接口验证

已把该 episode 的外部相机视频实际送入 SimFoundry Stage 1b：

```bash
cd /mnt/lyn/workspace/SimFoundry
source scripts/local/activate_simfoundry.sh

bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name oxe_droid_episode_000028 \
  --root-dir /mnt/lyn/workspace/SimFoundry/Data \
  --video-fpath \
    /mnt/lyn/workspace/SimFoundry/artifacts/oxe/droid_episode_000028/reconstruction_input.mp4 \
  --include 1b \
  --no-stream \
  --exec-mode direct \
  --python-bin "$CONDA_PREFIX/bin/python"
```

结果：

```text
1056 个全量 PNG 帧
15 个均匀抽样帧
stage_info.json: success=true
Pipeline 1b stage time: 60.33 秒
Pipeline 1b wall time: 61.75 秒
```

输出：

```text
Data/oxe_droid_episode_000028/s1_video/
Data/oxe_droid_episode_000028/pipeline_manifest.json
Data/oxe_droid_episode_000028/pipeline_run_report.json
```

注意：这只证明格式接口已经打通。DROID episode 的相机通常固定，而且机器人
与物体在运动；SimFoundry A 流水线需要移动相机扫拍、视差和近似静态场景。
因此 OXE episode 视频通常不适合作为高质量 3D 场景重建输入。

---

## 6. 当前机器可直接运行的仿真与数据采集

### 6.1 一键运行

```bash
cd /mnt/lyn/workspace/SimFoundry

EPISODE_INDEX=28 BACKEND=pybullet \
  bash tools/oxe/run_oxe_bridge.sh
```

脚本会：

1. 从 OXE/DROID TFDS 中读取 episode；
2. 导出 trajectory 和相机视频；
3. 创建 PyBullet Franka Panda 环境；
4. 按 15 Hz 发送 7 维目标关节位置；
5. 把 DROID 夹爪约定转换为 Panda finger opening；
6. 以 240 Hz 执行物理仿真；
7. 保存 RGB、关节、夹爪、EEF、动作和来源 OXE 字段。

### 6.2 已生成数据

```text
artifacts/oxe/droid_episode_000028/pybullet_replay/
├── replay_summary.json
├── sim_replay.mp4
├── sim_rollout.hdf5
└── sim_rollout.npz
```

验证结果：

```text
回放步数：1056
仿真动作频率：15 Hz
物理频率：240 Hz
最大关节跟踪绝对误差：0.0157640 rad
视频：H.264，640×480，15 FPS，1056 帧，70.4 秒
```

HDF5 内容：

```text
data/demo_0/
├── actions                         (1056, 8) float32
├── timestamps                      (1056,) float32
├── rewards                         (1056,) float32
├── dones                           (1056,) bool
├── obs/
│   ├── joint_position              (1056, 7) float32
│   ├── finger_position             (1056, 2) float32
│   ├── eef_position                (1056, 3) float32
│   ├── eef_quaternion_xyzw         (1056, 4) float32
│   └── rgb                         (1056, 480, 640, 3) uint8
└── source_oxe/
    ├── action_joint_position       (1056, 7)
    ├── action_gripper_position     (1056, 1)
    ├── legacy_action               (1056, 7)
    ├── observation_joint_position  (1056, 7)
    ├── observation_cartesian_position (1056, 6)
    └── observation_gripper_position    (1056, 1)
```

这满足了当前机器上的“取 OXE trajectory → 有一个 sim → 在其中执行并收集新数据”
的最小可运行要求。

### 6.3 改用其他 episode

```bash
EPISODE_INDEX=1 BACKEND=pybullet \
  bash tools/oxe/run_oxe_bridge.sh
```

也可以更换 DROID 根目录：

```bash
OXE_DROID_ROOT=/path/to/droid/1.4.0 \
EPISODE_INDEX=28 \
BACKEND=pybullet \
  bash tools/oxe/run_oxe_bridge.sh
```

快速冒烟测试：

```bash
OUTPUT_DIR=/tmp/oxe_smoke \
EPISODE_INDEX=28 \
BACKEND=pybullet \
  bash tools/oxe/run_oxe_bridge.sh --max-steps 5
```

---

## 7. 为什么当前节点不能运行原生 OmniGibson rollout

原生桥已经实现：

```text
tools/oxe/replay_droid_trajectory_in_omnigibson.py
```

它支持：

- floor + Franka 最小 OmniGibson 环境；
- `--scene-json` 加载 SimFoundry 生成的场景；
- 7 维 Franka joint-position replay；
- DROID → OmniGibson 夹爪约定转换；
- HDF5 / NPZ / 相机视频采集。

但当前节点是 NVIDIA H20。实际测试过程：

1. 最初容器只有 CUDA compute 库，没有 NVIDIA Vulkan ICD；
2. 使用与宿主机内核驱动完全匹配的 535.161.08 官方用户态库后，
   `vulkaninfo` 能正确枚举 8 张 H20；
3. Isaac Sim 5.1 / OmniGibson 随后仍在 renderer 初始化阶段报告 GPU crash；
4. 去掉相机并尝试 PXR / physics-only 最小环境也仍在同一阶段失败。

NVIDIA Isaac Sim 5.1 的官方系统要求明确说明：没有 RT Cores 的数据中心 GPU
（文档举例 A100、H100）不受支持；官方测试的 Linux 驱动为 580.65.06。
当前 H20 + 535.161.08 组合在实测中无法完成该原生渲染路径。

因此 `tools/oxe/run_oxe_bridge.sh` 在 H20 上会对
`BACKEND=omnigibson` 做安全拦截，避免再次触发 GPU crash。

### 7.1 在 RTX 节点上运行

建议：

- RTX 4090 / RTX 6000 Ada / RTX PRO 系列等带 RT Cores 的 GPU；
- 使用 Isaac Sim 5.1 官方支持或推荐的较新驱动；
- 容器需暴露 compute、utility、graphics、display/Vulkan 能力；
- `vulkaninfo --summary` 能看到 NVIDIA GPU。

命令：

```bash
cd /mnt/lyn/workspace/SimFoundry
source scripts/local/activate_simfoundry.sh

BACKEND=omnigibson EPISODE_INDEX=28 \
  bash tools/oxe/run_oxe_bridge.sh \
  --scene-json /path/to/reconstructed_og_scene.json
```

---

## 8. 完整 SimFoundry A 流水线的剩余条件

核心的 `simfoundry`、`da3`、`any6d`、`hunyuan` 环境已经安装并通过
import/GPU/原生扩展验证。现在阻止 A 1–13 真正端到端运行的主要因素不是这
四个环境，而是外部模型资产、服务凭据以及当前 H20 的渲染限制。

尚未下载的代表性 checkpoint：

```text
FoundationStereo model_best_bp2_serialize.pth
FoundationPose refiner / scorer
SAM2.1 hiera-large
Hunyuan RealESRGAN_x4plus.pth
```

当前凭据状态：

```text
gcloud executable：缺失
Google ADC：缺失
GEMINI_API_KEY：缺失
repo api_keys.txt：缺失
Hugging Face token/login：缺失
```

另外，若选择 auto-background 路线，还需要额外构建：

```text
void
nerfstudio_simfoundry
3dgrut
```

这些不是本次“OXE trajectory → 仿真 → 采集”的必要依赖。

此外，上游 README 在本次使用的提交中仍把完整的 robotics data generation /
training 标为 “Coming Soon”。仓库里已经有 C pipeline 的 teleop、evaluation、
demo/replay 代码，但本文提供的 OXE bridge 是本次复现新增的本地适配，不应等同
于论文所述生产级训练/数据生成系统已经完整开源。

如果补齐凭据和 checkpoint，并换到 RTX 节点，可继续：

```bash
cd /mnt/lyn/workspace/SimFoundry
source scripts/local/activate_simfoundry.sh

# 用户自行配置 HF_TOKEN 与 GEMINI_API_KEY / Vertex AI ADC 后：
bash scripts/installation/download_checkpoints.sh --default
```

然后用符合 SimFoundry 拍摄要求的移动相机视频运行 A：

```bash
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name my_scene \
  --video-fpath /path/to/moving_camera_scene_video.mp4
```

不要默认用 DROID 固定相机 episode 代替场景扫拍视频。

---

## 9. 测试记录

### 9.1 完整测试

```text
240 passed, 29 skipped, 6 warnings
最终回归耗时：334.22 秒
日志：logs/pytest_full_post_hunyuan.log
```

此前核心环境完成后的同结果日志仍保留为
`logs/pytest_full_final.log`（317.73 秒）。

### 9.2 文档推荐核心测试子集

```text
20 passed, 1 skipped
耗时：9.91 秒
日志：logs/pytest_core_subset.log
```

### 9.3 Pipeline dry-run

A、B、C 的统一 runner 均成功打印计划：

```text
logs/pipeline_dry_runs.log
```

### 9.4 其他日志

| 日志 | 内容 |
|---|---|
| `logs/install_simfoundry_core.log` | 上游核心安装；最后的 charset-normalizer 问题已在收尾脚本修复。 |
| `logs/install_core_finalize.log` | 幂等收尾验证成功。 |
| `logs/install_da3.log` | DA3 / gsplat / pycolmap / FAISS GPU 完整安装记录。 |
| `logs/install_any6d.log` | Any6D / FoundationPose / SAM2 / BOP / CUDA 扩展安装记录。 |
| `logs/install_hunyuan.log` | Hunyuan 初始安装及原始 `tb_nightly` 失败记录。 |
| `logs/install_hunyuan_resume.log` | Hunyuan 修复、依赖补齐、原生扩展编译和 Blender 运行库安装记录。 |
| `logs/hunyuan_extension_validation.log` | Hunyuan custom rasterizer CPU/H20 CUDA 功能测试。 |
| `logs/final_multi_env_validation.log` | 四环境及 OXE/Stage 1b/HDF5 产物统一最终验证。 |
| `logs/oxe_episode_28_stage1b.log` | OXE 视频进入 SimFoundry Stage 1b 的成功记录。 |
| `logs/oxe_omnigibson_replay_vulkan.log` | H20 上原生 OmniGibson renderer GPU crash 记录。 |
| `logs/min_og_physics.log` | 无相机最小 physics-only 尝试仍触发 renderer crash。 |

---

## 10. 新增文件说明

```text
scripts/local/
├── activate_simfoundry.sh
├── install_core_this_machine.sh
└── mamba

tools/oxe/
├── README_ZH.md
├── environment_rlds.yml
├── extract_droid_trajectory.py
├── replay_droid_trajectory_in_pybullet.py
├── replay_droid_trajectory_in_omnigibson.py
└── run_oxe_bridge.sh
```

同时修改了：

```text
scripts/installation/install_hunyuan.sh
requirements_hunyuan.txt
.gitignore
```

这些改动分别用于本机可重建的 Hunyuan 安装、兼容依赖固定，以及忽略大型
`artifacts/` 输出。OXE 工具没有改动 SimFoundry 的核心 pipeline 算法。

---

## 11. 建议的后续路线

### 路线 A：当前 H20 节点继续做算法和数据格式开发

使用已经可运行的 PyBullet 闭环：

```bash
EPISODE_INDEX=28 BACKEND=pybullet \
  bash tools/oxe/run_oxe_bridge.sh
```

适合：

- action/state 数据转换；
- VLA 数据格式实验；
- trajectory filtering；
- HDF5/NPZ 数据采集；
- controller 和 action convention 验证。

### 路线 B：在 RTX 节点做 SimFoundry 原生场景

1. 用移动相机扫拍场景；
2. 跑 SimFoundry A 1–13；
3. 得到 `s13_og/reconstructed_og_scene.json`；
4. 对真实 Franka base、相机和物体坐标做配准；
5. 使用 OmniGibson bridge 回放 OXE；
6. 用 SimFoundry C pipeline 继续 teleop / replay / LeRobot 数据导出。

这是达到“原场景、原任务、原生 OmniGibson、可训练数据”的正确路线。
