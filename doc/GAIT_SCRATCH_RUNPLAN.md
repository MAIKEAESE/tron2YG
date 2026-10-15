# GaitScratch 从零重训计划（5 天窗口）

背景：历次训练都从"站立先验"checkpoint resume，反复落入站立/小跳局部最优，
奖励栈膨胀到 ~40 项并互相打架（内八、偏移、台阶不抬腿）。
本计划废弃站立先验：**从零直接训 Perceptive Gait**，站立作为 0 速命令模式，
台阶靠 day-0 课程 + 单一地形自适应摆脚间隙奖励学出。

新任务 ID：`Isaac-Limx-SFYG-TRON2A-Perceptive-GaitScratch-TronCamp-v0`（+ `-Play-v0`）

## 设计要点

- **不 resume 任何 checkpoint**。训练命令里绝不带 `--resume True`。
- 站立 = 命令：`rel_standing_envs = 0.20`（20% 环境练 0 速站立），不单独成阶段。
- 课程：**平地/坡道 day-0 进入；台阶自 300 episode 起引入（等效 17%），600 ep 后升至 36%，
  1000 ep 后 40%**。`_use_scratch_curriculum_resets` 已把 bumps 出生权重全程置零
  （剩余段比例重归一化）。注意：训练地形中 bumps 仍在下坡与台阶之间，坡段出生走向台阶
  仍会穿越 bumps —— 是否需要无 bumps 地形变体待拍板（见修改报告 §4.1）。
- 奖励生效 **27 项非零**（原 ~39）：21 项任务级设计项 + 6 项继承基础项
  （ang_vel_xy_l2 / dof_pos_limits / dof_vel_limits / applied_torque_limits / dof_power_l1 /
  foot_contact_force）；归零 20 项（脚姿整形器×10、横向补丁×5、障碍救援×5 冗余项×…，
  以及基类姿态先验 joint_deviation_l1、orientation_exp_knee）。完整对照表见
  `GAIT_SCRATCH_MODIFICATION_REPORT.md` §3。
- PPO：entropy 0.008、lr 1e-3（adaptive, desired_kl 0.01），save_interval 100。

## 云端启动（服务器开机后）

```bash
cd /root/code/tron1_rl_lab   # 或实际仓库目录
git fetch maike
git checkout feature/gait-scratch-retrain
git pull --ff-only maike feature/gait-scratch-retrain

# 1) Play 冒烟（1 env，确认任务能注册、资产能加载、课程参数无 KeyError）
python scripts/rsl_rl/play.py \
  --task Isaac-Limx-SFYG-TRON2A-Perceptive-GaitScratch-TronCamp-Play-v0 \
  --num_envs 1

# 2) 奖励尺度审计（随机动作 ≈ 从零初期最坏情形，~10 分钟）
python scripts/rsl_rl/reward_audit.py \
  --task Isaac-Limx-SFYG-TRON2A-Perceptive-GaitScratch-TronCamp-v0 \
  --num_envs 256 --steps 2500 --output reward_audit_scratch.md
#   逐项看 P50/P95/P99/max、激活率、episode 贡献、摔倒前峰值；
#   DOMINANT / SPIKY / PRE-FALL-BURST / DEAD 任一命中 → 调该项权重/deadband
#   （不做全局有界化），重点：foot_contact_force、dof_acc_l2、applied_torque_limits。

# 3) 短训冒烟（~50 轮，验证实际生效的奖励项与报告一致）
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SFYG-TRON2A-Perceptive-GaitScratch-TronCamp-v0 \
  --num_envs 1024 --max_iterations 50 --run_name scratch_smoke
#   然后看控制台 Reward 字典或 TB 的 Episode rew:* 曲线：
#   非零项应恰好 27 个（清单见修改报告 §5.3），多/少任何一项都先回来对账再开正式训练。

# 4) 正式训练（nohup + 独立日志；num_envs 按 GPU 显存调，A100/4090 建议 4096 起步）
nohup python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SFYG-TRON2A-Perceptive-GaitScratch-TronCamp-v0 \
  --num_envs 4096 \
  --run_name scratch_a --seed 1 \
  > scratch_a.log 2>&1 &

# 5) 监控（两条曲线：训练 reward 只回答"收敛没"，固定评估才回答"有效没"）
tail -f scratch_a.log
# tensorboard: logs/rsl_rl/sfyg_tron_2a_perceptive_gait_scratch/scratch_a_*/

# 每 ~200 轮对最新 checkpoint 跑一次固定评估（同一把尺子，seed 恒定 42）：
python scripts/rsl_rl/eval_protocol.py \
  --task Isaac-Limx-SFYG-TRON2A-Perceptive-GaitScratch-TronCamp-Play-v0 \
  --checkpoint_path logs/rsl_rl/sfyg_tron_2a_perceptive_gait_scratch/scratch_a_*/model_<N>.pt \
  --num_envs 50 --repeat 2 --eval_seed 42
#   结果追加进 eval/eval_history.csv —— 画 progress_mean 和 stairs_pass_pct
#   随 checkpoint 的曲线：这才是"是否过于保守"的判据（progress 卡在坡前=保守局部最优）。
#   D2 决策点同时看两条曲线：训练 reward 平了 + eval progress 涨 = 真有效；
#   训练 reward 平了 + eval progress 也平 = 收敛到保守策略，按决策点处理。
```

## 监控指标（重点看 reward 分项，不是总 reward）

```
Episode/episode_length      # 变长 = 不再秒摔
Episode/mean_reward          # 0-300 轮应明显上升
Policy/mean_noise_std        # 应缓慢下降
reward/track_lin_vel_x_exp   # 主要驱动力，应持续走高
reward/walking_single_leg_air_time + swing_foot_clearance   # 抬腿是否学出来
```

## 决策点（Day 2 中午，约 1200-1500 轮）

- **通过**：能稳定平地行走 + 坡段不摔 + 摆脚间隙分项在涨 → 继续训到 3000+，
  选点后按需接 random-dir 或全图精修。
- **不通过**（仍站住不动 / 秒摔 / 完全不抬腿）→ 回退方案 C：
  从大班 best checkpoint 精修（台阶强化配置 d55d262 已在分支里）：

```bash
nohup python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SFYG-TRON2A-Perceptive-Gait-TronCamp-v0 \
  --num_envs 4096 \
  --resume True \
  --checkpoint_path logs/rsl_rl/sfyg_tron_2a_perceptive_random_dir/<run>/model_<best>.pt \
  --max_iterations 500 \
  --run_name refine_c1 --seed 1 \
  > refine_c1.log 2>&1 &
```

## 提交物（Day 5 前）

导出 `policy.pt` + `solution.py` 打包（SFYG 18 维动作 = 腿 10 + 臂/夹爪 8，
locomotion-only 策略需按既定 padding/重映射方案输出）。
