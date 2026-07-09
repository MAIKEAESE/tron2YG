# SFYG Training Lineage

本文档记录 `Isaac-Limx-SFYG-TRON2A` 当前训练模型的传导路线、各阶段目标、观察结论和后续选型依据。重点用于避免 checkpoint 来源混乱，以及解释为什么某些轮次适合继续训练、某些轮次只适合提交或废弃。

## 当前主线

当前主线只使用 SFYG，不继续投入 WFYG。

```text
官方/基础可运行站立模型
  -> TronCamp official terrain 验证
  -> Forward task 取得可提交基线
  -> Gait task 修小跳和步态
  -> 大步高抬腿探索
  -> 宽松脚步对称约束
```

## 关键任务

| 任务 | 用途 |
| --- | --- |
| `Isaac-Limx-SFYG-TRON2A-Blind-Flat-v0` | SFYG 基础平地冒烟测试 |
| `Isaac-Limx-SFYG-TRON2A-Blind-TronCamp-v0` | 官方训练地图验证 |
| `Isaac-Limx-SFYG-TRON2A-Blind-TronCamp-Forward-v0` | 快速取得可提交的前进策略 |
| `Isaac-Limx-SFYG-TRON2A-Blind-Gait-v0` | 修步态、反小跳、提高动作可控性 |

## 已知有效基线

### TronCamp Forward 早期模型

用途：快速提交、测评拿分。

现象：

- 能在官方地图上向前推进。
- 曾取得约 `28` 分测评结果。
- 但步态存在明显问题：偏航、前后脚、小跳或弓步倾向。

结论：

- 可作为提交基线。
- 不建议作为长期步态优化的唯一来源，因为它已经学到一些坏习惯。

### Gait 早期模型

用途：继续修步态。

观察结论：

- `model_100`：脚步更接近垂直，但节奏和稳定性不足。
- `model_300`：步频较快，膝盖更灵活，是当前较好的继续训练来源。
- `model_500`：节奏较稳定，但前后脚姿态更明显。
- `model_600/1000`：更容易收敛到保守、膝盖僵硬的稳定解。

结论：

- 继续训练优先从 `model_300.pt` 或更早的灵活 checkpoint 开始。
- 不建议从 `model_600.pt` / `model_1000.pt` 继续修步态，除非目标是稳定性而不是动作质量。

## 奖励演化路线

### 1. 解决不动问题

问题：

```text
模型站在原地不动，稳定奖励收益高于前进收益。
```

处理：

- 增加 `lin_vel_x_shortfall_penalty`
- 提高 `track_lin_vel_x_exp`
- 降低 `keep_balance` 和部分站立收益

结果：

- 模型开始向前走。
- 问题从“不动”转为“如何走得像迈步”。

### 2. 解决同步小跳

问题：

```text
模型通过双脚小跳向前移动，而不是交替迈步。
```

处理：

- 增加 `single_stance_reward`
- 增加 `swing_foot_clearance_reward`
- 增加 `double_flight_penalty`
- 加重竖直跳动相关惩罚

结果：

- 小跳减少。
- 开始出现左右脚先后动作。

### 3. 解决落地站相歪

问题：

```text
落地后两脚仍是一前一后，脚连线不垂直于前进方向。
```

处理：

- 增加 `double_stance_feet_x_alignment_reward`
- 只在双脚都接触地面时奖励两脚 body-x 对齐
- 降低全程 `same_feet_x_position`，避免摆腿过程被强行拉齐

结果：

- 坏姿态有一定好转。
- 但约束过强时会使步态保守，甚至过早收敛。

### 4. 降低膝盖僵硬

问题：

```text
训练后期膝盖越来越僵硬，步态稳定但保守。
```

处理：

- 降低 `action_rate_l2`
- 降低 `action_smoothness`
- 降低 `knee_joint_orientation`
- 提高 `gait_reward`
- 提高 `swing_foot_clearance`

结果：

- 膝盖灵活度有所改善。
- 但仍存在前后脚走的问题。

### 5. 大步高抬腿探索

问题：

```text
模型过早收敛到小步保守策略。
```

处理：

- 提高 `lin_vel_x` 命令范围
- 提高 `gait_command.frequencies`
- 降低 `gait_command.durations`
- 提高 `gait_command.swing_height`
- 增加 `single_stance_feet_x_separation_reward`

结果：

- 速度明显提高。
- 步态略有改善。
- 但 `single_stance_feet_x_separation_reward` 可能固化了前后脚走。

### 6. 宽松对称替代强约束

问题：

```text
不希望严格横向对齐，也不希望继续奖励固定前后脚。
目标是在身体平衡前提下，让前一步和后一步动作相对均衡。
```

处理：

- 大幅降低 `single_stance_feet_x_separation_reward`
- 降低 `double_stance_feet_x_alignment_reward`
- 降低 `same_feet_x_position`
- 增加 `balanced_feet_x_symmetry_reward`

当前意图：

```text
摆腿时允许自然前后运动；
身体平衡时，鼓励两脚前后位置围绕身体相对均衡；
不要强制严格对称；
不要让一只脚长期固定在前方。
```

## 当前推荐继续训练方式

优先从上一次表现“速度快、膝盖还灵活”的 checkpoint 继续。当前经验上优先选择：

```text
logs/rsl_rl/sfyg_tron_2a_gait/<run>/model_300.pt
```

训练命令模板：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SFYG-TRON2A-Blind-Gait-v0 \
  --num_envs 1024 \
  --headless \
  --resume True \
  --checkpoint_path logs/rsl_rl/sfyg_tron_2a_gait/<run>/model_300.pt
```

短训观察优先级：

```text
model_50
model_100
model_200
model_300
```

不建议一开始就等到 1000 轮，因为当前问题经常在训练后期收敛成保守坏姿态。

## TensorBoard 观察指标

### 前进

```text
Episode_Reward/track_lin_vel_x_exp
Episode_Reward/forward_speed_shortfall
```

目标：

- `track_lin_vel_x_exp` 上升。
- `forward_speed_shortfall` 接近 0。

### 小跳

```text
Episode_Reward/double_flight
Episode_Reward/lin_vel_z_l2
Episode_Reward/body_lin_acc
```

目标：

- `double_flight` 接近 0。
- 竖直速度和加速度惩罚不要明显恶化。

### 迈步

```text
Episode_Reward/single_stance
Episode_Reward/swing_foot_clearance
Episode_Reward/gait_reward
```

目标：

- 单脚支撑有上升，但不要异常过高。
- 摆脚高度有上升。
- gait reward 稳定提升。

### 脚步对称

```text
Episode_Reward/balanced_feet_x_symmetry
Episode_Reward/double_stance_feet_x_alignment
```

目标：

- `balanced_feet_x_symmetry` 上升。
- `double_stance_feet_x_alignment` 只作为弱参考，不要求过高。

### 僵硬程度

```text
Episode_Reward/action_rate_l2
Episode_Reward/action_smoothness
Episode_Reward/knee_joint_orientation
```

目标：

- 不要快速恶化。
- 如果视频中膝盖僵硬，而这些项表现很“好”，说明惩罚仍然过强，模型在用少动作换奖励。

## 选 checkpoint 原则

优先选择：

```text
速度正常
膝盖灵活
已经有左右脚先后动作
没有明显双脚小跳
前后脚坏姿态尚未完全固化
```

避免选择：

```text
训练后期虽然稳定但膝盖僵硬
长期一只脚在前、一只脚在后
靠身体倾斜或滑步前进
TensorBoard 快速收敛但视频动作保守
```

## 后续方向

短期：

- 继续在 Gait 任务中修步态。
- 先取得“能快走、膝盖灵活、不是小跳”的模型。
- 再回到 TronCamp 地图做地形适配。

中期：

- 将较好的 Gait checkpoint 作为前置动作基础。
- 在官方地图上微调通过坡、台阶等结构。
- 加入感知前，先保证盲走基础动作不要过差。

长期：

- 如果规则型 reward 无法进一步改善步态，再考虑引入参考轨迹或步态数据集。
- 数据集路线需要解决关节映射、机械臂固定、动作频率对齐和 imitation reward 设计，成本高于当前 reward 调参路线。
