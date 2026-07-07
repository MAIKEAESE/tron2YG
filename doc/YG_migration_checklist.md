# YG TRON2A 迁移验证清单

本文档用于验证 `feature/yg-flat-registration` 分支中的 SFYG/WFYG 任务迁移。当前分支只完成了 Flat 任务注册和 YG 资产配置接入；后续云端验证时按本清单逐步排错。

## 当前分支状态

- 分支：`feature/yg-flat-registration`
- 远端：`maike/feature/yg-flat-registration`
- 已注册任务：
  - `Isaac-Limx-SFYG-TRON2A-Blind-Flat-v0`
  - `Isaac-Limx-SFYG-TRON2A-Blind-Flat-Play-v0`
  - `Isaac-Limx-WFYG-TRON2A-Blind-Flat-v0`
  - `Isaac-Limx-WFYG-TRON2A-Blind-Flat-Play-v0`

## 资产要求

本分支把 `limx-tron2/robot-description` 作为 `robot_description` submodule 接入，并固定到：

```text
6ef32a33b9405e84a6551bc19d380f6747ea9611
```

云端拉取本分支后，先初始化资产 submodule：

```bash
git submodule update --init --recursive robot_description
```

如果云端是第一次克隆仓库，也可以直接递归克隆：

```bash
git clone --recursive git@github.com:MAIKEAESE/tron2YG.git
```

YG 资产配置会按顺序查找以下路径：

```text
exts/bipedal_locomotion/bipedal_locomotion/assets/usd/SFYG_TRON2A/usd/robot.usd
robot_description/tron2/SFYG_TRON2A/usd/robot.usd

exts/bipedal_locomotion/bipedal_locomotion/assets/usd/WFYG_TRON2A/usd/robot.usd
robot_description/tron2/WFYG_TRON2A/usd/robot.usd
```

推荐云端先使用 `robot_description` 目录方式，保持与官方 YG 仓一致：

```text
tron2_rl_lab/
  robot_description/
    tron2/
      SFYG_TRON2A/
        usd/robot.usd
      WFYG_TRON2A/
        usd/robot.usd
```

需要确认 meshes/configuration 等 USD 引用的相对文件也完整存在，不能只复制顶层 `robot.usd`。使用 submodule 初始化时会自动保留完整目录。

## 云端拉取分支

```bash
git fetch maike
git checkout feature/yg-flat-registration
git pull
git submodule update --init --recursive robot_description
```

如果云端还没有 `maike` 远端：

```bash
git remote add maike git@github.com:MAIKEAESE/tron2YG.git
git fetch maike
git checkout -b feature/yg-flat-registration maike/feature/yg-flat-registration
git submodule update --init --recursive robot_description
```

## 第一阶段：只验证 Play

先不要训练。Play 的目标是验证任务注册、USD 加载、关节名匹配、机械臂锁定和基础仿真稳定。

```bash
python scripts/rsl_rl/play.py --task Isaac-Limx-SFYG-TRON2A-Blind-Flat-Play-v0 --num_envs 1
python scripts/rsl_rl/play.py --task Isaac-Limx-WFYG-TRON2A-Blind-Flat-Play-v0 --num_envs 1
```

预期：

- Gym task 能找到，不报 `Environment ... doesn't exist`。
- USD 能加载，不报 `Could not open asset` 或 `Failed to open layer`。
- 关节名能匹配，不报找不到 `arm1_Joint`、`gripper1_Joint`、`wheel_L_Joint` 等。
- 机器人初始姿态稳定，机械臂/夹爪被 `arm_lock` 锁定，不出现明显抖动或爆炸。

## 第二阶段：小规模 Train Smoke Test

Play 通过后再做小规模训练冒烟测试：

```bash
python scripts/rsl_rl/train.py --task Isaac-Limx-SFYG-TRON2A-Blind-Flat-v0 --num_envs 64 --headless --max_iterations 10
python scripts/rsl_rl/train.py --task Isaac-Limx-WFYG-TRON2A-Blind-Flat-v0 --num_envs 64 --headless --max_iterations 10
```

预期：

- 能完成 10 个 iteration。
- observation/action 维度不报错。
- reset 不会把 arm/gripper 随机到异常位置。
- reward 计算不因为 arm/gripper 限位产生异常惩罚。

## 常见错误定位

### Task 找不到

检查：

- `bipedal_locomotion.tasks.locomotion.robots.__init__` 是否被导入。
- `scripts/rsl_rl/train.py` / `play.py` 是否正常 import extension。
- task 名称是否完全一致，尤其是 `SFYG` / `WFYG`。

### USD 找不到

检查：

- `robot_description/tron2/SFYG_TRON2A/usd/robot.usd`
- `robot_description/tron2/WFYG_TRON2A/usd/robot.usd`
- USD 内部引用的 meshes/configuration 是否也存在。

### 关节名找不到

检查资产中的关节名是否与代码一致：

SFYG：

```text
proximal_pitch_L_Joint, proximal_roll_L_Joint, proximal_yaw_L_Joint, knee_L_Joint, ankle_pitch_L_Joint
proximal_pitch_R_Joint, proximal_roll_R_Joint, proximal_yaw_R_Joint, knee_R_Joint, ankle_pitch_R_Joint
arm1_Joint ... arm6_Joint, gripper1_Joint, gripper2_Joint
```

WFYG：

```text
proximal_pitch_L_Joint, proximal_roll_L_Joint, proximal_yaw_L_Joint, knee_L_Joint, wheel_L_Joint
proximal_pitch_R_Joint, proximal_roll_R_Joint, proximal_yaw_R_Joint, knee_R_Joint, wheel_R_Joint
arm1_Joint ... arm6_Joint, gripper1_Joint, gripper2_Joint
```

### 机械臂抖动或爆炸

优先检查：

- `arm_lock` 是否匹配到 `arm[1-6]_Joint` 和 `gripper[12]_Joint`。
- reset 是否只作用于腿/轮关节。
- actuator gain randomization 是否误作用到 arm/gripper。

当前基础仓主要使用 `add_base_mass` 和 `reset_robot_joints`，YG env cfg 已限制 reset 范围；如果云端分支后来合入更多随机化项，需要同步把 arm/gripper 排除。

## 与云端分支合并策略

云端排队结束后，将云端改动作为单独分支推送，例如：

```text
cloud/train-play-terrain-fix
```

本地合并顺序建议：

1. 先确认 `feature/yg-flat-registration` 可 play。
2. 再 merge/rebase 云端 terrain/train/play 分支。
3. 冲突优先保留云端已验证的 train/play 改动，同时保留本分支新增的 YG task 注册和 asset cfg。
4. 合并后重新执行 Play smoke test，再执行 Train smoke test。

不要同时引入 YG、地形、训练脚本大改并直接长训；先分层验证，否则日志很难定位问题来源。
