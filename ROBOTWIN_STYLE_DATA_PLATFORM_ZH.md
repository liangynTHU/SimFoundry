# SimFoundry → RoboTwin 风格数据采集平台建设方案

> 更新时间：2026-08-24
> 仓库：`/mnt/lyn/workspace/SimFoundry`
> 目标：基于 SimFoundry 和 OmniGibson，构建一个支持大量资产、任务随机化、自动成功判定和批量数据采集的平台。

## 1. 目标定义

目标不是只复现某一条 OXE trajectory，而是构建如下闭环：

```text
大量机器人与物体资产
        ↓
随机生成场景和初始状态
        ↓
任务 YAML + success predicates
        ↓
人工示教 / OXE 示教 / 脚本规划
        ↓
自动生成大量成功轨迹
        ↓
质量检查、失败过滤、可复现回放
        ↓
HDF5 / LeRobot 数据集 + 视频可视化
```

当前选取的 OXE/DROID episode 28 可以作为：

- 动作和状态格式验证样例；
- `marker in cup` 任务的参考；
- 第一条 seed trajectory；
- sim/real 可视化对比样例。

但平台建成后，数据生成不应依赖单一 OXE episode。

---

## 2. 当前已经具备的能力

### 2.1 已配置环境

以下环境已经安装并验证：

```text
simfoundry
da3
any6d
hunyuan
rlds_env
```

已经验证：

- PyTorch CUDA 可识别 NVIDIA H20；
- OmniGibson `3.8.0` 可以 import；
- Depth Anything 3、gsplat、pycolmap 可用；
- Any6D、FoundationPose CUDA 扩展、SAM2、BOP Toolkit 可用；
- Hunyuan3D shape/texture pipeline 可导入；
- Hunyuan custom rasterizer 已在 CPU 和 H20 GPU 上实际执行；
- 仓库测试结果为 `240 passed, 29 skipped`。

### 2.2 OXE → 仿真 → 数据采集

当前已经完成：

```text
OXE/DROID trajectory
→ PyBullet Franka 回放
→ RGB、joint、gripper、EEF、action 数据采集
→ HDF5 / NPZ / MP4
```

原始 OXE 三相机可视化：

```text
artifacts/oxe/droid_episode_000028/all_cameras.mp4
```

PyBullet 回放结果：

```text
artifacts/oxe/droid_episode_000028/pybullet_replay/sim_replay.mp4
```

采集数据：

```text
artifacts/oxe/droid_episode_000028/pybullet_replay/sim_rollout.hdf5
artifacts/oxe/droid_episode_000028/pybullet_replay/sim_rollout.npz
```

### 2.3 SimFoundry 已有的平台骨架

#### 任务定义与成功判定

相关代码：

```text
scripts/cfg/task/
simfoundry/tasks/predicates.py
simfoundry/tasks/pick_place_task.py
```

现有 predicate 包括：

- `OnTop`
- `Touching`
- `InsideAABB`
- `OnTopAABB`
- `AboveAABB`
- `Lifted`
- `IsGrasping`
- milestone 顺序条件
- `all / any / specific` 条件组合

例如，把 marker 放进 cup 可以定义为：

```yaml
goal_predicates_all:
  - state: InsideAABB
    state_kwargs:
      volume_threshold: 0.7
    value: true
    group: marker
    other_group: cup
```

#### 随机化能力

已有：

- 物体 XYZ 位置随机化；
- 物体 Z 轴旋转随机化；
- 基于空间关系的初始化，例如 `inside`、`on_top`、`near`；
- 机器人关节随机化；
- 相机位姿随机化；
- 材质随机化；
- 光照和 HDRI 随机化；
- 动作噪声；
- distractor 物体；
- digital cousin 替换；
- 场景物理 settle 和有效性检查。

#### 数据生成流水线

```text
scripts/pipeline/C_application/
```

主要阶段：

| 阶段 | 功能 |
|---|---|
| `smoke` | 随机动作加载和渲染测试 |
| `1` | 策略评估 |
| `2` | 人工 teleoperation 数据采集 |
| `3/3b` | 示教标注和修订 |
| `4` | 提取 object-centric waypoints |
| `5` | 基于 waypoint 自动生成大量 demonstrations |
| `6` | 回放并导出 LeRobot 数据集 |

仓库当前默认支持将批量生成数量设为：

```yaml
s17_generate:
  n_demos: 1000
  max_attempts: 2000
```

---

## 3. 4090 集群是否可用

可以，而且比当前 H20 更适合运行原生 Isaac Sim / OmniGibson。

RTX 4090 具备 RT Cores，可以作为：

- OmniGibson 场景执行节点；
- Isaac Sim 渲染节点；
- 多相机 RGB/depth/segmentation 数据采集节点；
- 批量 episode 生成节点；
- 最终场景和轨迹可视化节点。

但 SSH 可登录不等于环境一定可用。节点还需要：

1. 正确安装 NVIDIA 驱动；
2. 容器或作业环境暴露 GPU compute、graphics 和 Vulkan；
3. `vulkaninfo --summary` 能识别 4090；
4. Isaac Sim 能 headless 启动；
5. OmniGibson 能加载 Franka 和场景；
6. 有足够磁盘空间保存环境、资产和数据。

4090 一般为 24 GB 显存，Hunyuan 网格生成建议使用：

```yaml
s7_mesh:
  low_vram: true
```

初期建议：

```text
每张 GPU 运行一个 OmniGibson 进程
```

确认显存和稳定性后，再考虑单卡多进程。

---

## 4. 需要提供的 4090 集群信息

不要在聊天中发送 SSH 密码或私钥。最好先在当前机器配置一个可以直接使用的
SSH alias，例如：

```bash
ssh sim4090
```

需要提供：

```text
SSH alias 或 hostname：
SSH 端口：
是否需要跳板机：
是否使用 Slurm：
4090 partition / queue：
srun 或 sbatch 示例命令：
远端可写工作目录：
单个作业时间限制：
每个节点有几张 4090：
```

还需要确认远端是否能访问：

```text
/mnt/lyn/workspace/SimFoundry
/apdcephfs_gy5/share_303588738/peterrao/data/vla-data
/jizhicfs/peterrao/miniconda3
```

如果文件系统不共享，将采用：

```text
rsync 仓库和必要资产
→ 远端重新安装 Conda 环境
→ 数据写入远端大容量目录
→ 只同步结果和日志回来
```

不建议直接复制整个 Conda 环境，因为 Isaac Sim、GPU 驱动和动态库与宿主机环境
关联较强。

---

## 5. 还缺少的核心内容

### 5.1 大规模资产库

当前主要只有：

```text
omnigibson-robot-assets
```

还需要选择和准备物体资产来源：

```text
BEHAVIOR-1K 物体资产
+ SimFoundry 重建的 custom assets
+ Hunyuan 生成的 digital cousins
+ 用户已有的 URDF / USD / OBJ / GLB 资产
```

需要用户确认：

- 是否接受 BEHAVIOR-1K / OmniGibson 数据条款；
- 集群上是否已有 `behavior-1k-assets`；
- 是否有内部镜像或可下载 URL；
- 是否有自有资产目录。

建议先完成一个稳定的小规模版本：

```text
20～50 个桌面物体
3～5 个容器
2～3 种桌面或工作台
```

资产必须经过以下检查：

- 尺度正确；
- 质心和坐标轴正确；
- visual mesh 正常；
- collision mesh 不过于复杂；
- 质量、摩擦、恢复系数合理；
- 容器有可靠的内部区域或 AABB；
- 可抓物体尺寸不超过夹爪能力；
- 仿真 settle 后不会穿模、飞出或掉落。

### 5.2 第一批任务

建议第一版支持：

```text
1. pick_and_place
2. put_object_in_container
3. stack_objects
4. move_object_near_target
5. lift_object
```

后续可增加：

```text
6. open_drawer
7. close_drawer
8. open_lid
9. put_object_in_drawer
10. 多阶段组合任务
```

用户只需给自然语言任务列表，例如：

```text
- 把 marker 放进 cup
- 把 apple 放进 bowl
- 把 cup 放到 plate 上
- 把两个 bowl 堆起来
- 把物体抬高 10 cm
```

这些任务可以进一步转换为：

- Task YAML；
- semantic group；
- 初始摆放条件；
- success predicates；
- milestone；
- timeout；
- 失败原因标签。

### 5.3 Seed demonstrations 或自动控制器

批量生成之前通常需要少量成功示教。

#### 推荐：脚本化控制

第一版优先实现：

```text
grasp pose 采样
→ IK
→ approach
→ grasp
→ lift
→ move
→ place
→ release
→ success 检查
```

优点：

- 适合 SSH 和无人值守集群；
- 不依赖本地 USB 设备；
- 易于批量生成；
- 失败后可以自动重试。

#### Teleoperation

仓库当前 teleop 依赖：

```text
TeleMoMa
SpaceMouse 或 Oculus
```

远程集群还需要：

- TurboVNC / VirtualGL 或类似远程桌面；
- USB 转发，或者开发 WebSocket/browser teleop；
- 用户自行确认 TeleMoMa 的使用许可并安装。

#### OXE trajectory

OXE 可以作为 seed，但要做任务级复现还需要：

- 相机内参和外参；
- 机器人基座与场景坐标变换；
- 场景物体初始姿态；
- 夹爪尺度和控制约定；
- 时间同步。

当前 OXE RLDS episode 不包含完整相机标定，因此精确复刻原场景仍需原始 DROID
episode 或额外标定数据。

### 5.4 数据规格

推荐第一版默认规格：

```text
格式：
  - 原始 HDF5
  - LeRobot v2.1

频率：
  - action: 15 Hz
  - physics: 120 Hz

相机：
  - 两路外部 RGB
  - 一路腕部 RGB
  - 可选 depth
  - 可选 semantic segmentation

机器人：
  - joint position
  - joint velocity
  - EEF position / quaternion
  - gripper state
  - action

任务：
  - language instruction
  - task id
  - success
  - milestone progress
  - object initial/final poses

复现元数据：
  - random seed
  - scene ID
  - asset IDs
  - task config
  - randomization parameters
  - camera intrinsics/extrinsics
  - simulator和代码版本
```

如果目标是训练 VLA，建议保持和 DROID 接近的：

```text
三相机 + 15 Hz + joint/EEF/gripper/action
```

---

## 6. 自动检查与数据质量控制

如果“很多 assert”也指程序断言和自动检查，批量采集器应至少包含：

### 6.1 场景初始化检查

- 所有 USD 和纹理引用存在；
- 资产尺度位于合理范围；
- 初始状态不能已经满足目标；
- 物体不穿模；
- 物体 settle 后没有飞走或掉出工作区；
- 目标物体在机器人工作空间内；
- 目标区域没有被其他物体完全阻塞；
- 相机能够看到任务相关物体。

### 6.2 轨迹执行检查

- IK 有解；
- 关节位置和速度不超限；
- action/state 无 NaN 或 Inf；
- 夹爪动作约定正确；
- 碰撞和接触合理；
- 图像不是全黑、全白或冻结；
- 时间戳严格单调；
- action、state、图像帧数一致。

### 6.3 成功与回放检查

- episode 结束时满足 success predicate；
- 必要 milestone 已按顺序满足；
- 保存后的 trajectory 能确定性回放；
- 回放后仍满足 success；
- 失败 episode 不混入成功训练集；
- 保存失败原因，例如：
  - `ik_failure`
  - `grasp_failure`
  - `object_dropped`
  - `collision`
  - `timeout`
  - `predicate_failure`
  - `invalid_observation`

### 6.4 数据完整性检查

- HDF5/LeRobot schema 正确；
- 每个 episode 有唯一 ID；
- 图像、状态、动作长度一致；
- 配置、seed 和资产版本可追踪；
- 训练、验证和测试资产集合互不泄漏；
- 数据分片可独立读取；
- 生成过程中支持断点续跑。

---

## 7. 模型和凭据

### 7.1 仅做仿真数据平台

如果使用现成资产、人工定义任务和脚本控制器，第一版可以暂时不使用 Gemini，
也不需要先完成整个 real-to-sim 重建模型链。

最低条件是：

```text
4090 上可运行 OmniGibson
+ 一批稳定资产
+ 任务 YAML
+ 自动控制器或少量示教
+ 批量采集和验证脚本
```

这是最快形成 RoboTwin 风格平台的路线。

### 7.2 自动从视频生成新场景和资产

需要：

- `HF_TOKEN`；
- Gemini API key 或 Google Vertex ADC；
- SAM3、DINOv3、RMBG、Hunyuan、DA3 等模型；
- FoundationPose、RealESRGAN 等 checkpoint；
- 符合 SimFoundry 要求的移动相机扫描视频。

公开 checkpoint 的下载地址已经在：

```text
scripts/installation/download_checkpoints.sh
```

需要 Hugging Face 权限的模型主要包括：

```text
facebook/sam3
facebook/dinov3-vitl16-pretrain-lvd1689m
briaai/RMBG-2.0
```

如果使用 VOID 自动背景路线，还需要：

```text
netflix/void-model
```

不要在聊天中发送 token。可以写入：

```text
/mnt/lyn/workspace/SimFoundry/scripts/installation/api_keys.txt
```

Gemini API key 可放在：

```text
/mnt/lyn/workspace/SimFoundry/api_keys.txt
```

---

## 8. 推荐实施阶段

### 阶段 1：4090 可用性验证

在 4090 节点依次验证：

```text
1. nvidia-smi
2. Vulkan
3. PyTorch CUDA
4. Isaac Sim headless
5. OmniGibson headless
6. 加载 Franka
7. 加载一个物体
8. 执行随机动作
9. 保存 RGB 视频
10. 加载 task YAML 并检查 success predicate
```

只有这一阶段通过后，才传输大型资产和搭建批量任务。

### 阶段 2：最小 RoboTwin 风格闭环

目标：

```text
Franka Panda
+ 10～20 个物体
+ 3 个任务
+ 每个任务 10 条成功 trajectory
+ RGB/proprio/action
+ HDF5/LeRobot
+ 自动 success 和数据完整性检查
```

建议任务：

```text
marker → cup
apple → bowl
cup → plate
```

### 阶段 3：规模化采集

目标：

```text
50～200 个稳定资产
+ 更多容器和桌面
+ scene/object/camera/light/material randomization
+ 每任务数百至数千条成功数据
+ Slurm job array
+ 自动失败重试
+ 数据分片、合并和统计
```

### 阶段 4：real-to-sim 扩展

```text
真实移动相机视频
→ SimFoundry 场景重建
→ 新资产导入资产库
→ digital cousins
→ 自动任务生成
→ 4090 集群批量采集
```

---

## 9. 最终可视化和数据产物

平台完成后，每个任务或数据分片至少输出：

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

可视化内容包括：

- 原始场景与随机化场景；
- 多相机 RGB；
- 可选 depth 和 semantic segmentation；
- 机器人轨迹；
- 目标物体和目标区域；
- milestone 状态；
- 当前 success predicate；
- 最终成功/失败原因；
- 原始 OXE 与仿真左右对比。

完整 SimFoundry 场景还会输出：

```text
Data/<scene>/s13_og/reconstructed_og_scene.json
Data/<scene>/s13_og/reconstructed_scene.png
Data/<scene>/s13_og/auto_generation/scene_*.json
Data/<scene>/s13_og/auto_generation/scene_*.png
```

---

## 10. 用户下一步需要提供

请按以下模板提供信息：

```text
SSH_ALIAS=
SSH_PORT=
是否需要跳板机=

是否使用 Slurm=
4090_PARTITION=
SBATCH或SRUN示例=
每节点GPU数量=
单任务时间限制=

REMOTE_WORKDIR=
/mnt/lyn 是否共享=
/apdcephfs_gy5 是否共享=
/jizhicfs 是否共享=
外网是否需要代理=

是否允许下载 BEHAVIOR-1K 资产=
是否已有 behavior-1k-assets 路径=
是否有自有资产目录或下载 URL=

机器人=Franka Panda / Franka Robotiq
第一批任务=
每个任务目标成功 episode 数量=
输出格式=HDF5 + LeRobot v2.1（推荐）
示教方式=脚本化控制（推荐）/ SpaceMouse / OXE

HF_TOKEN是否已在机器配置=
GEMINI_API_KEY或Vertex ADC是否已配置=
```

最优先只需要提供：

```text
1. 可直接使用的 SSH alias；
2. 4090 作业申请命令；
3. 远端工作目录；
4. 三个第一批任务；
5. 每个任务希望生成多少条成功数据。
```

拿到这些信息后，应先执行 4090 的 Isaac Sim / OmniGibson smoke test，再决定
是否同步完整环境和资产。
