# TronCamp (TRON2) 训练地形 —— 随机化、宽阔开放，覆盖评测的全部地形要素。
#
# 用它来训练（例如在 tron2_rl_lab 内）。你最终按固定的评测赛道评分，而不是这条训练地形。
# 接入方式见 README.md。
#
#   from tron_camp_training_terrain import TRON_CAMP_TRAINING_TERRAIN_CFG, TRON2_SPAWN_Z
#   env_cfg.scene.terrain = TRON_CAMP_TRAINING_TERRAIN_CFG

from __future__ import annotations

from .mixed_terrain import MixedTrackCfg, make_training_terrain_cfg

# TRON2 站高略低于 1 m；把 base 放在距地格原点上方约 0.97 m（与评测环境一致）。
TRON2_SPAWN_Z = 0.966

# 比 3 m 的评测走廊更宽，让学习中的策略有横向回旋余地。
TRACK_WIDTH = 8.0

# TRON2（腿式 / 轮足）能应对完整的障碍范围 —— 从 builder 默认值起步。
_track = MixedTrackCfg()
_track.step_height_range = (0.10, 0.18)
_track.platform_height_range = (0.15, 0.40)   # 上界贴齐评测 S8 最高台 0.40 m（评测已去掉原 0.50 m 台）

# 10 x 10 = 100 条各自独立随机化的赛道（难度随行递增）。按你的 GPU 调整。
TRON_CAMP_TRAINING_TERRAIN_CFG = make_training_terrain_cfg(
    width=TRACK_WIDTH,
    cell_length=80.0,
    num_rows=10,
    num_cols=10,
    base_seed=0,
    track_cfg=_track,
)
