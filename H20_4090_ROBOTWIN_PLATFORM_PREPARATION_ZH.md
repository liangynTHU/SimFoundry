# H20 + RTX 4090：SimFoundry / RoboTwin 风格数据平台准备清单

> 更新时间：2026-08-25（Asia/Shanghai）
> 仓库：`/mnt/lyn/workspace/SimFoundry`
> 目标：尽可能使用 8 张 H20 承担模型推理、资产生成、数据处理和策略训练，仅将必须依赖 RTX 光追能力的 Isaac Sim / OmniGibson 仿真与渲染放到 RTX 4090 集群。

---

## 1. 一页结论

最终希望建立的不是单条 OXE trajectory 回放，而是如下数据生产平台：

```text
大量机器人与物体资产
        ↓
场景、物体、相机、材质和光照随机化
        ↓
任务 YAML + 成功条件 predicates
        ↓
脚本控制器 / 少量人工示教 / OXE seed demo
        ↓
4090 上运行 OmniGibson 物理仿真与多相机渲染
        ↓
自动生成大量成功 trajectories
        ↓
自动 assert、失败过滤、确定性回放
        ↓
HDF5 + LeRobot + MP4 数据集
        ↓
H20 上做数据处理、训练和评估
```

推荐分工：

```text
H20：模型计算、资产生成、策略训练、数据处理
4090：Isaac Sim / OmniGibson 物理仿真和 RTX 渲染
```

当前最优先需要准备：

1. 可直接使用的 4090 SSH alias；
2. Slurm 申请 4090 的命令；
3. 4090 远端工作和数据目录；
4. 确认 H20 与 4090 是否共享 `/mnt/lyn`、`/apdcephfs_gy5`、`/jizhicfs`；
5. 第一批 3 个任务及每个任务期望生成的成功 episode 数量；
6. 一批稳定的物体资产，或允许下载 BEHAVIOR-1K 资产；
7. 确定机器人、控制方式和数据格式。

---

## 2. 当前已经完成的基础

### 2.1 已配置环境

当前 H20 节点已经配置并验证：

```text
simfoundry
hunyuan
da3
any6d
rlds_env
```

已验证组件包括：

- PyTorch `2.7.0+cu128`；
- 8 张 NVIDIA H20 可用于 CUDA 计算；
- OmniGibson `3.8.0` 可以 import；
- DA3、gsplat、pycolmap；
- Any6D、FoundationPose CUDA 扩展、SAM2、BOP Toolkit；
- Hunyuan3D shape/texture pipeline；
- Hunyuan custom rasterizer CPU/GPU 功能调用；
- 完整测试：`240 passed, 29 skipped`。

### 2.2 已完成 OXE 最小闭环

```text
OXE/DROID episode 28
→ 提取 trajectory 和三路相机视频
→ PyBullet Franka 回放
→ 采集 RGB、joint、gripper、EEF、action
→ 保存 HDF5 / NPZ / MP4
```

原始 OXE 三相机视频：

```text
artifacts/oxe/droid_episode_000028/all_cameras.mp4
```

PyBullet 回放视频：

```text
artifacts/oxe/droid_episode_000028/pybullet_replay/sim_replay.mp4
```

采集数据：

```text
artifacts/oxe/droid_episode_000028/pybullet_replay/sim_rollout.hdf5
artifacts/oxe/droid_episode_000028/pybullet_replay/sim_rollout.npz
```

这条轨迹可以作为：

- 动作格式测试；
- `marker in cup` 的参考任务；
- 第一条 seed demonstration；
- sim/real 对比样例。

但平台建成后，不应依赖单一 OXE episode。

---

## 3. H20 与 4090 的明确分工

## 3.1 尽可能交给 H20 的工作

H20 应作为模型计算中心、资产处理中心和训练中心。

| 工作 | H20 是否承担 | 说明 |
|---|---:|---|
| OXE/RLDS 读取和转换 | 是 | TensorFlow 读取主要使用 CPU，输出统一格式 |
| DA3 深度估计 | 是 | GPU 推理 |
| SAM3 地面和物体分割 | 是 | GPU 推理 |
| DINOv3 特征提取 | 是 | GPU 推理 |
| RMBG 背景移除 | 是 | GPU/ONNX 推理 |
| RealESRGAN 图像增强 | 是 | GPU 推理 |
| Hunyuan3D shape generation | 是 | H20 大显存很适合 |
| Hunyuan3D texture generation | 是 | H20 承担 |
| Any6D / FoundationPose | 是 | 姿态估计和配准 |
| Digital cousin 生成 | 是 | 图像、网格和纹理生成 |
| 碰撞网格和资产预处理 | 是 | CPU/GPU 混合 |
| 任务 YAML / manifest 生成 | 是 | CPU 为主 |
| Policy inference server | 是 | OpenPI、GR00T 或自定义策略 |
| VLA / policy 训练 | 是 | 主要训练算力 |
| HDF5 / LeRobot 转换 | 是 | CPU 为主 |
| 数据检查、过滤和统计 | 是 | CPU/GPU 混合 |
| 训练/验证/测试切分 | 是 | 数据管理 |
| 视频对比和指标计算 | 是 | CPU/GPU 混合 |

### H20 多 GPU 初始分配建议

```text
GPU 0：DA3 / 深度估计
GPU 1：SAM3 / segmentation
GPU 2：DINOv3 / RMBG
GPU 3：Any6D / FoundationPose
GPU 4：Hunyuan shape worker 1
GPU 5：Hunyuan shape worker 2
GPU 6：Hunyuan texture worker
GPU 7：policy server / 备用 worker
```

后续应改成队列式调度，而不是永久绑定：

```text
asset_manifest.jsonl
        ↓
H20 model worker pool
        ↓
sim-ready asset packages
        ↓
4090 simulation worker pool
```

### H20 节点输出给 4090 的内容

每个场景或任务应打包为：

```text
scene_package/
├── scene.json
├── task.yaml
├── assets/
│   ├── object_001.usd
│   ├── object_001_collision.obj
│   └── textures/
├── robot.yaml
├── cameras.yaml
├── randomization.yaml
└── manifest.json
```

## 3.2 必须交给 4090 的工作

4090 只承担需要 RTX 图形能力的工作：

- Isaac Sim headless 启动；
- OmniGibson 场景加载；
- Stage 11–13 的原生仿真阶段；
- 物体 physics settle；
- 场景和物体随机化；
- 材质、光照、HDRI 和相机随机化；
- 多相机 RGB/depth/segmentation 渲染；
- 机器人任务执行；
- success predicates 检查；
- episode HDF5、视频和状态采集；
- 保存后 trajectory 的回放验证。

4090 一般为 24 GB 显存，建议：

```text
每张 4090 同时运行 1 个 OmniGibson 进程
```

第一版不要单卡多进程。Hunyuan 在 4090 上运行时必须设置：

```yaml
s7_mesh:
  low_vram: true
```

但更推荐让 Hunyuan 始终运行在 H20。

---

## 4. 4090 集群需要准备的详细内容

### 4.1 SSH 接入

请先在当前 H20 节点配置无需手工输入密码的 SSH alias：

```bash
ssh sim4090
```

不要在聊天中发送密码或私钥。需要准备：

```text
SSH alias
hostname
SSH port
username
是否需要跳板机
```

目标是确保下面命令可直接执行：

```bash
ssh sim4090 'hostname && nvidia-smi'
```

### 4.2 Slurm 或其他调度器

如果使用 Slurm，需要提供：

```text
GPU partition / queue 名称
申请 1 张 4090 的命令
CPU 数量
内存大小
最大运行时间
每个节点有几张 4090
是否允许 job array
是否允许节点独占
```

最好提供一条已知可用的命令，例如：

```bash
srun \
  --partition=4090 \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=04:00:00 \
  --pty bash
```

或提供 `sbatch` 模板：

```bash
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
```

### 4.3 GPU、驱动和图形环境

需要确认：

```text
操作系统版本
NVIDIA Driver 版本
CUDA 可用性
Vulkan 可用性
Docker 是否可用
Apptainer/Singularity 是否可用
Conda 路径
是否允许安装软件包
```

第一次登录后需要运行：

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
vulkaninfo --summary
```

随后依次验证：

```text
1. PyTorch CUDA
2. Isaac Sim headless
3. OmniGibson headless
4. 加载 Franka
5. 加载一个简单物体
6. 执行随机动作
7. 保存 RGB 视频
8. 加载 task YAML
9. 检查 success predicate
```

### 4.4 Headless 图形能力

集群至少需要支持一种：

```text
EGL headless
Vulkan headless
Xorg virtual display
TurboVNC / VirtualGL
```

纯自动采集优先使用：

```text
Isaac Sim headless + EGL/Vulkan
```

只有人工 SpaceMouse/Oculus teleop 时，才需要 VNC、VirtualGL 和输入设备转发。

### 4.5 远端目录

需要准备：

```text
REMOTE_WORKDIR
REMOTE_DATA_DIR
REMOTE_CACHE_DIR
REMOTE_SCRATCH_DIR
```

示例：

```text
REMOTE_WORKDIR=/data/<user>/SimFoundry
REMOTE_DATA_DIR=/data/<user>/simfoundry_datasets
REMOTE_CACHE_DIR=/data/<user>/.cache
REMOTE_SCRATCH_DIR=/local_scratch/<user>
```

### 4.6 文件系统共享情况

需要确认 4090 是否能访问：

```text
/mnt/lyn
/apdcephfs_gy5
/jizhicfs
```

重点路径：

```text
/mnt/lyn/workspace/SimFoundry
/apdcephfs_gy5/share_303588738/peterrao/data/vla-data
/jizhicfs/peterrao/miniconda3
```

如果不共享，需要增加：

```text
scripts/cluster/sync_to_4090.sh
scripts/cluster/sync_results_back.sh
```

并采用：

```text
H20 生成资产包
→ rsync 到 4090
→ 4090 生成数据分片
→ rsync 数据和日志回 H20
```

### 4.7 网络与代理

需要确认 4090 是否能访问：

```text
GitHub
Hugging Face
PyPI
Google Drive
NVIDIA 下载服务
```

如果需要代理，请提供：

```text
HTTP_PROXY
HTTPS_PROXY
内部镜像或代理地址
```

如果 4090 无公网访问，则在 H20 下载：

- 模型权重；
- Hugging Face snapshots；
- Conda/Pip wheel；
- OmniGibson 资产；
- 材质和纹理库；

然后同步到 4090。

---

## 5. 资产库需要准备的内容

### 5.1 机器人选择

第一版建议：

```text
Franka Panda + 原生 Panda gripper
```

需要确定：

```text
机器人：Panda / Panda + Robotiq
控制方式：EEF delta / joint position
抓取方式：assisted / physical
```

推荐第一版：

```text
控制：EEF delta pose
同时保存：joint position action
抓取：assisted
```

等采集系统稳定后，再增加 physical grasping。

### 5.2 通用物体资产库

当前主要已有机器人资产，仍需准备：

```text
BEHAVIOR-1K 物体资产
custom-assets
SimFoundry 重建资产
Hunyuan digital cousins
用户已有 URDF/USD/OBJ/GLB 资产
```

需要确认：

- 是否接受 BEHAVIOR-1K / OmniGibson 数据条款；
- 集群是否已有 `behavior-1k-assets`；
- 是否允许从官方地址下载；
- 是否有内部镜像；
- 是否有用户自有资产目录或 URL。

### 5.3 第一版资产规模

建议先准备：

```text
5 个杯子或容器
5 个碗或盘子
5 个 marker、pen 等长条物体
5 个水果或积木
2～3 个桌面或工作台
```

总量先控制在：

```text
20～30 个稳定资产
```

第一步不要直接加入几千个未经验证的资产。

### 5.4 资产 manifest

每个资产至少记录：

```yaml
asset_id: marker_001
category: marker
visual_mesh: marker.glb
collision_mesh: marker_collision.obj
usd_path: marker.usd

scale_meters: 1.0
mass_kg: 0.02
friction: 0.8
restitution: 0.0

fixed_base: false
graspable: true
container: false

dimensions:
  x: 0.015
  y: 0.015
  z: 0.140

split: train
```

容器还需要：

```yaml
container: true
container_link: null
interior_aabb:
  lower: [-0.04, -0.04, 0.01]
  upper: [0.04, 0.04, 0.10]
```

### 5.5 资产自动检查

每个资产必须通过：

```text
[ ] USD 可加载
[ ] visual mesh 可见
[ ] collision mesh 存在
[ ] 单位为米
[ ] AABB 尺寸合理
[ ] 质量大于 0
[ ] 摩擦和恢复系数合理
[ ] 放到桌面后可以稳定 settle
[ ] 不会穿过桌面
[ ] 不会因碰撞网格异常而飞出
[ ] 可抓宽度不超过夹爪范围
[ ] 容器内部区域有效
[ ] 纹理引用完整
```

---

## 6. 第一批任务需要准备的内容

### 6.1 推荐任务

第一版建议：

```text
1. put_marker_in_cup
2. put_apple_in_bowl
3. put_cup_on_plate
```

后续增加：

```text
stack_objects
move_object_near_target
lift_object
open_drawer
close_drawer
put_object_in_drawer
多阶段组合任务
```

### 6.2 每个任务需要描述

```text
任务名称
自然语言指令
源物体类别
目标物体/容器类别
初始摆放范围
成功条件
最大步数
目标成功 episode 数量
```

示例：

```yaml
task_name: put_marker_in_cup

language:
  - Put the marker in the cup.
  - Place the marker inside the cup.
  - Pick up the marker and put it into the cup.

objects:
  source_category: marker
  target_category: cup

initialization:
  marker_xy_range: 0.15
  cup_xy_range: 0.10
  minimum_distance: 0.10

success:
  predicate: InsideAABB
  volume_threshold: 0.7
  robot_not_touching: true
  hold_steps: 20

max_steps: 500
target_success_episodes: 1000
```

### 6.3 成功条件建议

放入容器不应只检查瞬时位置，应检查：

```text
物体主要体积位于容器内部
物体质心低于容器上沿
机器人已经释放物体
物体线速度和角速度低于阈值
条件连续保持一定 physics steps
```

### 6.4 初始状态分布

每个任务需明确：

```text
物体 XY 范围
物体 Z 轴旋转范围
源物体和目标物体的最小距离
是否允许遮挡
是否加入 distractors
机器人初始关节扰动
相机位置和朝向扰动
```

### 6.5 训练、验证和测试资产切分

建议：

```text
训练资产：70%
验证资产：15%
测试资产：15%
```

测试集应包含：

- 未见过的物体实例；
- 未见过的尺寸和纹理；
- 未见过的场景摆放；
- 未见过的 digital cousins。

---

## 7. 示教和自动控制需要准备的内容

### 7.1 推荐：脚本控制器

第一版优先实现：

```text
采样 grasp pose
→ IK 可达性检查
→ approach
→ close gripper
→ lift
→ move above target
→ descend
→ release
→ retreat
→ success assert
```

优势：

- 适合 SSH 和无人值守；
- 不依赖 SpaceMouse USB 转发；
- 易于批量生成；
- 失败可以自动重试；
- 容易记录失败原因。

### 7.2 Seed demonstrations

每个任务建议先获得：

```text
5～20 条成功 seed demonstrations
```

然后使用 SimFoundry 已有流程：

```text
示教标注
→ object-centric waypoint 提取
→ KNN waypoint 选择
→ 场景随机化
→ 自动生成数百或数千条 demonstrations
```

### 7.3 SpaceMouse / Oculus

如果后续使用人工 teleop，还需：

- SpaceMouse 或 Oculus；
- TeleMoMa 使用许可和安装；
- TurboVNC / VirtualGL；
- USB 转发，或开发 WebSocket/browser teleop。

第一版不建议将 teleop 作为阻塞条件。

### 7.4 OXE trajectory

OXE 可以作为 seed，但精确任务复现还需要：

- 相机内参；
- 相机到机器人基座的外参；
- 腕部相机手眼标定；
- 机器人基座和场景坐标变换；
- 物体初始姿态；
- 夹爪尺寸和动作约定；
- 图像、状态和动作时间同步。

当前 RLDS episode 没有完整保存上述标定，因此 OXE 更适合作为动作先验，而不是
平台启动的必要条件。

---

## 8. 推荐数据规格

### 8.1 输出格式

```text
原始采集：HDF5
训练格式：LeRobot v2.1
视频：MP4
元数据：JSON / YAML
```

### 8.2 频率

```text
physics_frequency: 120 Hz
action_frequency: 15 Hz
rendering_frequency: 15 Hz
```

### 8.3 相机

建议保持 DROID 风格：

```text
外部相机 1：RGB
外部相机 2：RGB
腕部相机：RGB
```

第一版分辨率建议：

```text
320 × 180 或 256 × 256
```

可选模态：

```text
depth_linear
seg_semantic
seg_instance
normal
```

### 8.4 机器人状态

保存：

```text
joint_position
joint_velocity
eef_position
eef_quaternion
gripper_position
gripper_normalized
```

### 8.5 动作表示

建议同时保存：

```text
joint_position_action
eef_delta_position
eef_delta_rotation
gripper_action
```

### 8.6 任务信息

```text
task_name
language_instruction
success
milestones
failure_reason
reward
episode_seed
```

### 8.7 场景和复现信息

```text
scene_id
asset_ids
object_initial_poses
object_final_poses
camera_intrinsics
camera_extrinsics
randomization_parameters
simulator_version
git_commit
config_hash
```

---

## 9. 必须实现的自动 asserts

### 9.1 场景初始化

```text
assert asset_exists
assert usd_references_valid
assert object_scale_valid
assert no_initial_penetration
assert object_on_support_surface
assert target_reachable
assert initial_state_not_success
assert camera_sees_target
```

### 9.2 轨迹执行

```text
assert action_is_finite
assert observation_is_finite
assert joint_limits_valid
assert joint_velocity_valid
assert image_not_blank
assert timestamps_monotonic
assert frame_counts_match
```

### 9.3 任务成功

```text
assert success_predicate
assert milestone_order_valid
assert object_stable
assert object_released
assert success_held_for_n_steps
```

### 9.4 数据保存和回放

```text
assert hdf5_readable
assert lerobot_readable
assert action_state_length_match
assert video_frame_count_valid
assert replay_reproduces_success
assert config_and_seed_recorded
```

### 9.5 失败标签

```text
invalid_scene
ik_failure
planning_failure
grasp_failure
object_dropped
collision_failure
timeout
predicate_failure
invalid_observation
replay_mismatch
```

失败 episode 应单独保存或只保存元数据，不能混入成功训练集。

---

## 10. 磁盘和数据规模准备

当前 70.4 秒、单路 640×480 RGB 的 PyBullet HDF5 约为：

```text
169 MB
```

三路相机、depth、segmentation 和大量 episode 会快速增加到 TB 规模。

第一阶段建议：

```text
3 个任务
每任务 100 条成功数据
每条 10～30 秒
3 路 320×180 RGB
HDF5 + MP4 + LeRobot
```

建议预留：

```text
环境、源码和模型：200～500 GB
资产库：100 GB～数 TB
首期数据集：2～10 TB
每个 4090 worker 本地 scratch：100～300 GB
```

最终容量应在完成 100 条试采后，根据实测的单 episode 大小重新估算。

---

## 11. 模型、权重和凭据

### 11.1 最小仿真平台暂时不需要 Gemini

如果第一版使用：

```text
现成资产
+ 人工定义任务
+ 脚本控制器
```

则可以先不把 Gemini 和完整 real-to-sim 模型链作为阻塞条件。

最小条件：

```text
4090 可运行 OmniGibson
+ 一批稳定资产
+ 任务 YAML
+ 自动控制器
+ 批量采集和验证脚本
```

### 11.2 自动从视频生成资产时需要

需要：

```text
HF_TOKEN
GEMINI_API_KEY 或 Google Vertex ADC
SAM3
DINOv3
RMBG
Hunyuan3D
DA3
FoundationPose
RealESRGAN
```

公开 checkpoint 的下载入口：

```text
scripts/installation/download_checkpoints.sh
```

Hugging Face gated 模型可能包括：

```text
facebook/sam3
facebook/dinov3-vitl16-pretrain-lvd1689m
briaai/RMBG-2.0
```

VOID 自动背景路线还需要：

```text
netflix/void-model
```

不要在聊天里发送 token，可放到：

```text
/mnt/lyn/workspace/SimFoundry/scripts/installation/api_keys.txt
```

Gemini key 可放到：

```text
/mnt/lyn/workspace/SimFoundry/api_keys.txt
```

---

## 12. 推荐实施阶段

### 阶段 1：4090 可用性验证

目标：

```text
Isaac Sim headless 能启动
OmniGibson 能加载 Franka
能够 step、渲染并保存视频
能够加载任务并检查 predicate
```

该阶段不下载大型资产，也不批量采集。

### 阶段 2：最小平台闭环

```text
Franka Panda
+ 10～20 个资产
+ 3 个任务
+ 每任务 10 条成功 trajectory
+ RGB/proprio/action
+ HDF5/LeRobot
+ 自动 assert
```

### 阶段 3：首批数据

```text
20～30 个资产
+ 每任务 100 条成功数据
+ 场景和相机随机化
+ 失败自动重试
+ 回放验证
```

### 阶段 4：规模化

```text
50～200 个稳定资产
+ 10～20 个任务
+ 每任务 1000+ 成功 episodes
+ Slurm job array
+ 数据分片和合并
+ 自动统计报表
```

### 阶段 5：real-to-sim

```text
真实移动相机视频
→ H20 上运行 SimFoundry 重建
→ H20 生成 digital cousins
→ 导入资产库
→ 4090 批量仿真采集
→ H20 训练 VLA / policy
```

---

## 13. 最终产物

每个数据分片至少输出：

```text
scene_preview.png
initial_state.png
episode_<id>.mp4
episode_<id>_success_overlay.mp4
rollouts_<shard>.hdf5
lerobot_dataset/
generation_summary.json
failure_report.json
```

完整场景输出：

```text
Data/<scene>/s13_og/reconstructed_og_scene.json
Data/<scene>/s13_og/reconstructed_scene.png
Data/<scene>/s13_og/auto_generation/scene_*.json
Data/<scene>/s13_og/auto_generation/scene_*.png
```

统计报表建议包含：

```text
成功率
平均尝试次数
各失败原因占比
各资产成功率
各随机化范围覆盖率
episode 时长分布
轨迹长度分布
数据磁盘占用
回放一致率
```

---

## 14. 需要用户填写的完整模板

不知道的项目可以留空，登录集群后再检查。

```text
# 4090 SSH
SSH_ALIAS=
SSH_HOST=
SSH_PORT=
SSH_USER=
NEED_PROXY_JUMP=yes/no
PROXY_JUMP_ALIAS=

# 调度器
SCHEDULER=slurm/none
GPU_PARTITION=
SRUN_EXAMPLE=
SBATCH_EXAMPLE=
GPUS_PER_NODE=
CPU_PER_GPU=
RAM_PER_NODE=
MAX_JOB_TIME=
JOB_ARRAY_AVAILABLE=yes/no

# 远端目录
REMOTE_WORKDIR=
REMOTE_DATA_DIR=
REMOTE_CACHE_DIR=
REMOTE_SCRATCH_DIR=

# 文件系统
MNT_LYN_SHARED=yes/no
APDCEPHFS_SHARED=yes/no
JIZHICFS_SHARED=yes/no

# 网络
GITHUB_ACCESS=yes/no
HUGGINGFACE_ACCESS=yes/no
PYPI_ACCESS=yes/no
GOOGLE_DRIVE_ACCESS=yes/no
HTTP_PROXY=
HTTPS_PROXY=

# 软件
OS_VERSION=
NVIDIA_DRIVER=
DOCKER_AVAILABLE=yes/no
APPTAINER_AVAILABLE=yes/no
CONDA_PATH=
CAN_INSTALL_PACKAGES=yes/no

# 资产
ACCEPT_BEHAVIOR1K_TERMS=yes/no
EXISTING_BEHAVIOR1K_ASSET_PATH=
CUSTOM_ASSET_PATH=
CUSTOM_ASSET_URL=

# 机器人
ROBOT=Franka Panda/Franka Robotiq
CONTROL_MODE=EEF delta/joint position
GRASPING_MODE=assisted/physical

# 第一批任务
TASK_1=
TASK_2=
TASK_3=
SUCCESS_EPISODES_PER_TASK=

# 数据
OUTPUT_FORMAT=HDF5 + LeRobot v2.1
CAMERA_RESOLUTION=320x180
SAVE_DEPTH=yes/no
SAVE_SEGMENTATION=yes/no
SAVE_FAILURE_EPISODES=yes/no
ACTION_FREQUENCY=15
PHYSICS_FREQUENCY=120
MAX_EPISODE_SECONDS=

# 示教
DEMO_METHOD=scripted/SpaceMouse/OXE
SEED_DEMOS_PER_TASK=

# 磁盘预算
AVAILABLE_DATA_STORAGE_TB=
AVAILABLE_SCRATCH_GB_PER_NODE=

# 凭据
HF_TOKEN_CONFIGURED=yes/no
GEMINI_CONFIGURED=yes/no
```

---

## 15. 最优先需要提供的 5 项

如果暂时不填写全部，只需先提供：

```text
1. SSH_ALIAS
2. 申请 1 张 4090 的 Slurm 命令
3. REMOTE_WORKDIR
4. /mnt/lyn、/apdcephfs_gy5、/jizhicfs 是否共享
5. 第一批 3 个任务和每个任务的成功 episode 数量
```

下一步应先运行 4090 的 Isaac Sim / OmniGibson smoke test。测试通过后，再同步完整
环境、资产和任务，避免提前下载大量文件却发现图形环境不可用。
