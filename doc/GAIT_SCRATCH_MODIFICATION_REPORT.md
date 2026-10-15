# GaitScratch 从零训练线 — 修改审查报告（v2.2）

状态：**已修改工作区，未 commit、未 push**。
v2 依据审查意见修正：① 补齐基类继承奖励项的清单与权重（原"21 项"表述不准确）；
② 新增课程覆盖，去掉 bumps 出生权重；③ 修正"台阶 day-0"表述。
v2.1：④ PLAY 类补挂 `_use_scratch_curriculum_resets`；⑤ docstring 项数更正。
v2.2：⑥ 新增 `scripts/rsl_rl/reward_audit.py` 奖励尺度审计工具（分位数/激活率/
episode 贡献/摔倒前峰值 + 触发条件）；⑦ 修正奖励有界性评估的三处绝对化表述
（见 §7）；⑧ `foot_contact_force` 的处置从"减半是第一候选"改为"**待审计数据决定**"
（静态承重 vs 落地冲击需分开看，直接减半会同时削弱对冲击的约束）。
v2.3：⑨ 新增 `scripts/rsl_rl/eval_protocol.py` 固定评估协议——项目此前缺失
"收敛 vs 有效"的判据（训练 reward 只能判收敛）。固定 seed=42、确定性策略
（均值动作无探索噪声）、分段通过率（坡/下坡/坎/台阶/平台）+ 到达距离 +
失败率，结果追加 `eval/eval_history.csv`，每 ~200 轮跑一次，D2 决策点
同时看训练曲线与 eval 曲线。

---

## 1. 改动动机

翻查 `记录.md` 全程与当前配置，结论：站立模型本身的姿势没问题（3000 轮版"站姿稳定没有外八"），
问题在**"站立优先"课程结构**给网络烙进"站着最安全"先验，后续每个阶段都与之搏斗，
奖励栈膨胀到 ~40 项生效项并互相打架（内八、偏移、台阶不抬腿）。

对策：新增**从零训练线** `GaitScratch`（不 resume 任何 checkpoint），站立降级为 0 速命令模式，
台阶由地形课程 + 单一地形自适应摆脚间隙奖励学出。原 PerceptiveGait / StandLift / RandomDir
等任务**一律不动**，新旧线并存，随时可回退。

---

## 2. 文件级改动清单

| 文件 | 改动 | 性质 |
|---|---|---|
| `.../robots/limx_solefoot_yg_tron2a_env_cfg.py` | 新增 `_use_scratch_gait_rewards()`、`_use_scratch_curriculum_resets()`；新增 Env 类 `SFYG_TRON2A_PerceptiveGaitScratchTronCampEnvCfg(_PLAY)` | 纯新增，不改任何既有函数 |
| `.../agents/limx_rsl_rl_ppo_cfg.py` | 新增 `SFYG_TRON2APerceptiveGaitScratchPPORunnerCfg` | 纯新增 |
| `.../robots/__init__.py` | import +1、实例化 +1、gym.register +2（GaitScratch v0 / Play-v0） | 纯新增 |
| `doc/GAIT_SCRATCH_RUNPLAN.md` | 启动命令、监控、决策点、上线前验证清单 | 新文档（未跟踪） |

继承结构：`GaitScratch ──> PerceptiveGaitTronCampEnvCfg`（原任务零改动，含 height_scan 观测、
课程 resets、30s episode、宽松 termination）。PLAY 变体与现有 PerceptiveGait PLAY 同构
（BlindGait PLAY + 手动挂 perceptive helpers + scratch 奖励），**并额外挂
`_use_scratch_curriculum_resets`**——v2.1 修正：PLAY 链原本使用 track-aligned
`reset_root_state_uniform` 复位（无课程），补挂后 Play 冒烟与训练走同一个
无 bumps 课程 reset 函数，冒烟才真正覆盖课程参数。

---

## 3. 奖励栈（v2：完整清单）

生效奖励 = **27 项非零**（21 项任务级设计项 + 6 项继承基础项），另有 2 项基类姿态先验被
新线归零、2 项父链已归零项被显式钉住。老线（PerceptiveGait）生效约 39 项。

### 3.1 任务级设计项（21 项，`_use_scratch_gait_rewards` 显式赋值）

| 奖励项 | 老权重 | 新权重 | 说明 |
|---|---|---|---|
| track_lin_vel_x_exp (std 0.25) | 4.0 | 4.0 | 主驱动力 |
| track_lin_vel_y_exp (std 0.18) | 3.5 | **1.5** | 直线赛道降权 |
| track_ang_vel_z_exp (std 0.20) | 4.0 | **2.0** | 航向固定 0 |
| forward_speed_shortfall | 0.0* | **-1.5** | 恢复温和版，防"启动即站住" |
| keep_balance | 0.85 | **0.6** | 压低防"站着拿分" |
| base_height_exp (target 0.74) | 0.95 | **0.9** | |
| flat_orientation_l2 | 0.0* | **-1.0** | 恢复标准姿态项 |
| single_stance | 0.90 | **0.8** | |
| walking_single_leg_air_time | 5.0 | **3.0** | 仍是迈步最强驱动；防硬甩腿 |
| single_leg_air_time_overrun | -3.0 | **-2.0** | 配套微降 |
| swing_foot_clearance（terrain_adaptive，yaw 前向 + height_bands，参数继承 d55d262） | 3.0 | 3.0 | **台阶学习唯一通道** |
| swing_foot_liftoff | 0.40 | **0.35** | 从零早期的离地引子 |
| double_flight | -1.5 | -1.5 | 防跳步 |
| feet_slide | -0.24 | **-0.25** | |
| feet_ang_slide_penalty | -0.28 | **-0.25** | |
| undesired_contacts (threshold 5.0) | -3.5 | **-1.5** | 老值过度惩罚台阶边缘接触；摔倒另有 termination |
| world_pos_y_abs | -0.5 (db 0.20) | -0.5 (**db 0.25**) | 路线纪律主项 |
| heading_error_abs | -1.0 (10°) | **-0.8** (10°) | 路线纪律第二项 |
| action_rate_l2 | -0.05 | **-0.02** | |
| dof_vel_l2 | -2.0e-4 | -2.0e-4 | |
| dof_acc_l2 | -2.0e-6 | **-1.0e-6** | |

*0.0 = 老线中被 `_use_perceptive_stair_clearance` 置零。

### 3.2 继承基础项（6 项，v2 新增：显式钉住基类值，纳入清单）

| 奖励项 | 基类值 | 新线处理 | 作用 |
|---|---|---|---|
| ang_vel_xy_l2 | -0.15 | **保留 -0.15** | 躯干横滚/俯仰角速度抑制，标准基础项 |
| dof_pos_limits | -0.2 | **保留 -0.2** | 关节位置限位（安全，必需） |
| dof_vel_limits | -0.025 (soft 0.92) | **保留 -0.025** | 关节速度限位（安全，必需） |
| applied_torque_limits | -1e-2 | **保留 -1e-2** | 力矩限位（安全，必需） |
| dof_power_l1 | -2.5e-7 | **保留 -2.5e-7** | 按关节加权的功率正则 |
| foot_contact_force | -0.002 | **保留 -0.002** | 脚底接触力惩罚；若台阶段出现踮脚/不敢踩，第一候选是减半 |

### 3.3 v2 新增归零的基类姿态先验（2 项）

| 奖励项 | 基类值 | 处理 | 理由 |
|---|---|---|---|
| joint_deviation_l1 | -1.0（×0.005 关节权重） | **归零** | 把关节拉向默认站立姿态 —— 正是本次要移除的"站立先验"家族 |
| orientation_exp_knee | +0.1 | **归零** | 膝部 yaw 朝向整形，与已归零的 knee_joint_orientation 同族 |

### 3.4 显式钉住的父链已零项（2 项，行为不变）

`feet_air_time`（基类 0.05，**gait 父链已置 0**，老线有效值即 0）、
`dof_torques_l2`（基类 -4e-7，**gait 父链已置 0**，老线有效值即 0）。
审查意见中列出的这两项实际在老线就已是零，v2 代码中显式写 0 以自文档化。

### 3.5 归零项总表（与 v1 相同的 18 项 + §3.3 的 2 项 = 20 项）

- **A. 脚姿整形器互殴（内八/偏移嫌疑人）**：feet_distance 0.4、foot_joint_orientation -0.11、
  single_stance_feet_x_separation 0.45、balanced_feet_x_symmetry 0.80、
  double_stance_feet_x_alignment 0.25、stance_foot_stability 2.00、stance_knee_extension 0.75、
  swing_knee_flexion 0.08、forward_swing_knee_lift 0.45、knee_joint_orientation -0.8 → 全部归零
- **B. 逃站立横向补丁**：centerline_y 2.5、heading_alignment 0.3、world_lin_vel_y_abs -0.1、
  predicted_path_lateral_error -0.2、lateral_correction_velocity 2.5 → 全部归零
- **C. 障碍救援项**：obstacle_forward_swing 4.0、obstacle_approach_progress 1.0、
  obstacle_heading_error -1.0、obstacle_heading_alignment 2.2、high_obstacle_approach 0.65 → 全部归零
- **D. 冗余平滑/重复项**：action_smoothness、torques_smoothness_penalty、body_lin_acc、lin_vel_z_l2、
  track_world_lin_vel_x_exp 4.5、forward_overspeed -3.0、gait_reward 0.45 → 全部归零

### 3.6 命令分布（老 → 新）

| 参数 | 老值 | 新值 |
|---|---|---|
| heading_command | True | **False**（直线赛道） |
| rel_standing_envs | 0.22 | **0.20**（站立 = 命令模式） |
| rel_heading_envs | 1.0 | 0.0 |
| lin_vel_x | (0.12, 0.50) | 不变 |
| gait frequencies | (0.45, 0.75) | **(0.50, 0.80)** |
| gait durations | (0.58, 0.72) | **(0.58, 0.70)** |
| gait swing_height | (0.13, 0.23) | **(0.15, 0.24)** |

### 3.7 课程覆盖（v2 新增：`_use_scratch_curriculum_resets`）

先调用原 `_use_tron_camp_curriculum_resets`（继承全部参数），然后：

- `curriculum_segments` 中 `bumps_entry` 权重 0.30/0.16→**0.0**（基础段权重）
- `phase_schedule` 六个阶段中所有 `bumps_entry` 键**删除**（reset 函数按段名查表，
  缺键 = 0；采样内部自动归一化，剩余段保持原比例 —— 即比例重归一化；
  阶段间线性插值机制保留）

去 bumps 后各阶段等效出生比例：

| 阶段 | start_flat / 上坡 / 下坡 / **台阶** / 平台 |
|---|---|
| step 0（day-0） | 80% / 10% / 10% / 0 / 0（**平地+坡道 day-0，台阶未进入**） |
| 300 ep | 46% / 20% / 17% / **17%** / 0 |
| 600 ep | 34% / 14% / 14% / **36%** / 1% |
| 1000 ep+ | 32% / 13% / 13% / **40%** / 1% |

（台阶占比因 bumps 退出而由 25–30% 升至 36–40%，符合"以台阶为训练重心"的目标。）

**措辞修正（审查点 3）**：原报告"台阶 day-0 进课程"不准确。准确表述：
**平地/坡道 day-0 进入课程，台阶自 300 episode 起以 17% 引入、逐步升至 40%。**

### 3.8 PPO/Runner

entropy 0.003→**0.008**、lr 3e-4→**1e-3**（与 StandLift 从零线同款，adaptive+desired_kl 0.01 自动下调），
max_iterations 6000、save_interval 100，其余不变。

---

## 4. 风险与待拍板项

1. **（待拍板 D）bumps 地形仍在赛道中部**。训练地形结构为
   `起始平地 → 上斜坡 → 高台 → 下斜坡 → 坎(bumps) → 上台阶 → 平台`（mixed_terrain.py），
   bumps 位于下坡与台阶之间：**零出生权重 ≠ 不穿越**——从坡段出生走向台阶仍要跨越 bumps。
   彻底隔离需地形变体（`n_bumps_range=(0, 0)`，一行改动，但云端首次启动要重建地形缓存，
   且地形文件注释称训练地形"覆盖评测的全部地形要素"——**请确认评测图确实无 bumps**，
   若有，则不应去掉地形，只需保持本次的课程权重改动）。
2. 本地未跑 Isaac。py_compile + 全量静态核对（基类 30 项 + gait 17 项 + perceptive 13 项，
   共 60 项定义全部对账，v2 清单见 §3）。
3. 零权重项仍会被 RewardManager 计算（沿项目既有模式），轻微性能开销。
4. 保留项内部门槛未动（min_stance_force=80、min_base_height=0.58、min_forward_speed=0.12）：
   从零早期不满足 → 前 ~200 轮主要靠速度跟踪 + 姿态 + 基础安全项驱动，总 reward 曲线前段难看是预期。
5. 一次改了三个变量（从零初始化 + 奖励栈 + 命令/课程），非严格消融；若需归因干净，
   可加"原配置从零"对照跑（多花 ~半天 GPU）。
6. PLAY 变体手动组装 helper，冒烟失败优先查它。

## 5. 上线前验证清单（对应审查意见 3/4/5 步）

1. 云端 Play 冒烟：`GaitScratch-TronCamp-Play-v0`，1 env，确认任务注册、资产加载、
   课程参数无 KeyError。
2. **奖励尺度审计**（新，先于短训）：
   `python scripts/rsl_rl/reward_audit.py --task Isaac-Limx-SFYG-TRON2A-Perceptive-GaitScratch-TronCamp-v0 --num_envs 256 --steps 2500 --output reward_audit_scratch.md`
   随机动作 ≈ 从零初期的最坏情形。逐项输出 P50/P95/P99/max、激活率、episode 贡献、
   摔倒前 25 步峰值，并打 DOMINANT / SPIKY / PRE-FALL-BURST / DEAD 标记。
   任何标记命中 → 按触发条件调该项权重/deadband/归一化（**不做全局有界化**），
   重点盯 `foot_contact_force`、`dof_acc_l2`、`applied_torque_limits` 与接触/摔倒瞬间。
3. 短训冒烟（~50 轮）：控制台 Reward 字典 / TensorBoard `Episode rew:*` 的**非零项**应恰好为
   §3.1 的 21 项 + §3.2 的 6 项 = 27 项，多一项少一项都回来改报告。
4. 全部通过 → commit + push `feature/gait-scratch-retrain` → 正式开训。
5. D2 中午决策点与回退方案 C 详见 `GAIT_SCRATCH_RUNPLAN.md`。

## 7. 奖励有界性结论（v2.2 修正版）

- 无界奖励**在理论上**不是问题，但"PPO 只要求 episode 回报有限"的表述不完整：
  实际训练还需考虑数值稳定性——value loss 离群点、梯度方差、跨环境回报尺度。
  从零初期的随机策略审计（§5.2）就是用来实测这些的，不靠权重相加推算。
- "早死逃生"病理**不能由权重估算排除**：速度跟踪奖励在站立命令与行走命令下含义不同，
  存活价值必须看实际 episode 分项回报统计（审计表的 ep-contrib 列），开训头 200 轮复核。
- **不做全局有界化**（tanh / 1−exp 会在大误差区丢失梯度，机器人会停留在坏动作区）。
  设计原则：目标项（跟踪/高度/净空）用有界相似度核；滑移/偏航/超限用 deadband + 线性/平方；
  扭矩/加速度/动作变化用小权重 L1/L2；**接触力先扣除正常承重区间或按 mg 归一化，只罚超出部分**
  （候选形式 `p_F = -w·max(0, F_z − F_support)²`）——是否落地等审计数据说话。
- `foot_contact_force` 处置变更：从"减半是第一候选"改为"待审计"。静态承重（近似常数，
  不改变策略排序但拉低 value 基线）、双脚常量、落地冲击峰值是三回事，审计表的
  P50/P95 分离度会告诉我们它在惩罚哪一个。

## 6. 回滚方式

均未提交。全量撤销：
`git checkout -- exts/bipedal_locomotion/` + 删除两份 doc；部分回滚直接改
`_use_scratch_gait_rewards` / `_use_scratch_curriculum_resets` 内对应行，不影响任何既有任务线。
