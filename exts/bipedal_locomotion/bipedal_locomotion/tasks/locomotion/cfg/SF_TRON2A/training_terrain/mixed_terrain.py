# 用于 TronCamp / HumanoidCamp 基准的随机化混合地形训练赛道。
#
# 为何存在
# ---------------
# 评测赛道是唯一确定性的赛道 —— 这对公平计分非常理想，但如果你直接在其上训练，你的策略会
# 记住那个确切的布局并过拟合。本构建器产生与评测赛道相同的特征类型，但对每个地形格的每个参数
# 进行程序化随机化，并运行在更宽 / 更开阔的走廊上，从而让学习中的策略有恢复的空间。在此训练，
# 在固定的评测赛道上接受评分。
#
# 它包含 TronCamp 与 HumanoidCamp 全部地形要素的并集，使一个策略能够学会一切：
#   起始平地 -> 上斜坡 -> 山顶高台 -> 下斜坡 -> 坎 (bumps) -> 上台阶
#   -> 带沟壑 (gullies) 的桥 -> 下台阶 -> 抬升的高台 (踏石) -> 结束平地
#
# 每个特征的尺寸/数量/间距都按地形格采样，将 IsaacLab 课程式 `difficulty`
# (0->1，更难的行) 与按地形格的随机性混合。使用 num_rows x num_cols > 1 可一次性得到许多不同的
# 赛道。所有贯穿地面的特征都跨越整个 mesh 宽度 (使其无法被绕行)；斜坡是经 fix_normals() 挤出的
# prism，从而让 PhysX 单面 collider 朝外。

from __future__ import annotations

import math

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from .terrain_base import BetterTerrainGenerator, BetterTerrainImporter

# 一个按进程计数的计数器，使相继的地形格得到不同的 (但可复现的) 随机布局。
_CALL_COUNTER = 0


def _next_rng(base_seed: int) -> np.random.Generator:
    global _CALL_COUNTER
    seed = (base_seed * 1_000_003 + _CALL_COUNTER) & 0x7FFFFFFF
    _CALL_COUNTER += 1
    return np.random.default_rng(seed)


def _blend(rng: np.random.Generator, lo: float, hi: float, difficulty: float, d_weight: float) -> float:
    """在 [lo, hi] 内采样，将课程式 difficulty 与随机性混合。
    d_weight=0 -> 纯随机 (用于几何多样性)；d_weight~0.6 -> 更难的地形格趋向更大。"""
    frac = d_weight * float(difficulty) + (1.0 - d_weight) * float(rng.random())
    return lo + (hi - lo) * max(0.0, min(1.0, frac))


def make_mixed_track(difficulty: float, cfg: "MixedTrackCfg"):
    """构建一条随机化混合地形赛道。按 IsaacLab 的约定返回 (mesh_list, origin)。

    地形格局部坐标：x in [0, total_len], y in [0, cfg.size[1]]；生成器将地形格居中于世界原点。
    每次调用都用新的随机参数重建赛道 (见 _next_rng)。"""
    rng = _next_rng(cfg.base_seed)
    W = cfg.width
    cy = cfg.size[1] / 2.0
    y0, y1 = cy - W / 2.0, cy + W / 2.0
    meshes: list[trimesh.Trimesh] = []

    def _box(x_center, z_center, ext_x, ext_z, ext_y=W, y_center=cy):
        return trimesh.creation.box(
            (ext_x, ext_y, ext_z),
            trimesh.transformations.translation_matrix((x_center, y_center, z_center)),
        )

    def _ramp_prism(x_lo, x_hi, z_lo, z_hi):
        """一个在 (x,z) 平面上的直角三角形 prism，跨整个宽度挤出。fix_normals() -> 朝外的面。"""
        profile = [(x_lo, z_lo), (x_hi, z_lo), (x_hi, z_hi)] if z_hi >= z_lo \
            else [(x_lo, z_lo), (x_hi, z_hi), (x_lo, z_hi)]
        verts = np.array([[x, y0, z] for (x, z) in profile] + [[x, y1, z] for (x, z) in profile], dtype=float)
        n = len(profile)
        faces = []
        for i in range(n):
            a, b = i, (i + 1) % n
            faces.append([a, b, b + n])
            faces.append([a, b + n, a + n])
        for i in range(1, n - 1):
            faces.append([0, i, i + 1])
            faces.append([n, n + i + 1, n + i])
        m = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
        m.fix_normals()
        return m

    x = 0.0  # 沿赛道推进的游标

    # --- 地面底板 (顶面在 z = 0)，待最终长度确定后再定尺寸 ---
    base_idx = len(meshes)
    meshes.append(None)  # 占位符

    # --- 1. 起始平地 ---
    x += _blend(rng, *cfg.start_flat_len_range, difficulty, 0.0)

    # --- 2. 上斜坡 (+角度) ---
    a_up = math.radians(_blend(rng, *cfg.ramp_angle_deg_range, difficulty, 0.6))
    run_up = _blend(rng, *cfg.ramp_run_range, difficulty, 0.0)
    rise = run_up * math.tan(a_up)
    meshes.append(_ramp_prism(x, x + run_up, 0.0, rise))
    x += run_up

    # --- 3. 山顶高台 (平面在 z = rise) ---
    hill_len = _blend(rng, *cfg.hill_len_range, difficulty, 0.0)
    meshes.append(_box(x + hill_len / 2.0, rise / 2.0, hill_len, rise))
    x += hill_len

    # --- 4. 下斜坡 (回到 z = 0) ---
    a_dn = math.radians(_blend(rng, *cfg.ramp_angle_deg_range, difficulty, 0.6))
    run_dn = rise / math.tan(a_dn)
    meshes.append(_ramp_prism(x, x + run_dn, rise, 0.0))
    x += run_dn

    x += _blend(rng, *cfg.mid_flat_len_range, difficulty, 0.0)

    # --- 5. 坎 (bumps)：地面上贯穿全宽的脊 ---
    n_bumps = int(round(_blend(rng, *cfg.n_bumps_range, difficulty, 0.5)))
    for _ in range(max(0, n_bumps)):
        h = _blend(rng, *cfg.bump_height_range, difficulty, 0.6)
        t = _blend(rng, *cfg.bump_thickness_range, difficulty, 0.0)
        meshes.append(_box(x + t / 2.0, h / 2.0, t, h))
        x += t + _blend(rng, *cfg.bump_spacing_range, difficulty, 0.0)

    x += _blend(rng, *cfg.mid_flat_len_range, difficulty, 0.0)

    # --- 6. 上台阶 ---
    n_steps = int(round(_blend(rng, *cfg.n_stairs_range, difficulty, 0.6)))
    step_h = _blend(rng, *cfg.step_height_range, difficulty, 0.4)
    step_run = _blend(rng, *cfg.step_run_range, difficulty, 0.0)
    for i in range(n_steps):
        top_z = (i + 1) * step_h
        meshes.append(_box(x + (i + 0.5) * step_run, top_z / 2.0, step_run, top_z))
    x += n_steps * step_run
    top_z = n_steps * step_h

    # --- 7. 带沟壑 (gullies) 的桥：位于 top_z 的贯穿全宽地板段，由间隙隔开 ---
    n_gaps = int(round(_blend(rng, *cfg.n_gaps_range, difficulty, 0.5)))
    for i in range(n_gaps + 1):
        seg = _blend(rng, *cfg.gap_floor_seg_range, difficulty, 0.0)
        meshes.append(_box(x + seg / 2.0, top_z / 2.0, seg, top_z))
        x += seg
        if i < n_gaps:
            x += _blend(rng, *cfg.gap_width_range, difficulty, 0.6)  # 间隙 (无地板)

    # --- 8. 下台阶 (镜像上台阶的高度，使其重新落回 z = 0) ---
    for i in range(n_steps):
        z = (n_steps - i) * step_h
        meshes.append(_box(x + (i + 0.5) * step_run, z / 2.0, step_run, z))
    x += n_steps * step_run

    x += _blend(rng, *cfg.mid_flat_len_range, difficulty, 0.0)

    # --- 9. 抬升的高台 (踏石，在 y 方向窄) ---
    n_plat = int(round(_blend(rng, *cfg.n_platforms_range, difficulty, 0.5)))
    for _ in range(max(0, n_plat)):
        h = _blend(rng, *cfg.platform_height_range, difficulty, 0.6)
        fx = _blend(rng, *cfg.platform_face_x_range, difficulty, 0.0)
        meshes.append(_box(x + fx / 2.0, h / 2.0, fx, h, ext_y=cfg.platform_face_y))
        x += fx + _blend(rng, *cfg.platform_gap_range, difficulty, 0.0)

    # --- 结束平地 ---
    x += _blend(rng, *cfg.finish_flat_len_range, difficulty, 0.0)
    total_len = x

    # 现在长度已知，填充底板 (顶面在 z = 0)。
    meshes[base_idx] = _box(total_len / 2.0, -0.1, total_len, 0.2)

    origin = np.array([1.0, cy, 0.05])  # 在起始平地内约 1 m 处放置
    return meshes, origin


@configclass
class MixedTrackCfg(SubTerrainBaseCfg):
    """随机化混合地形训练赛道的子地形配置。

    所有范围均为 (low, high)。编辑这些以让训练更容易/更难，或匹配某个机器人。
    """

    function = make_mixed_track

    width: float = 8.0                       # mesh 的 y 跨度 (相比 3 m / 5 m 评测走廊更宽更开阔)
    base_seed: int = 0                       # 与按地形格的计数器组合以产生布局多样性

    start_flat_len_range: tuple = (3.0, 7.0)
    mid_flat_len_range: tuple = (1.0, 3.0)   # 各特征之间的小块平地
    finish_flat_len_range: tuple = (3.0, 6.0)

    ramp_angle_deg_range: tuple = (6.0, 16.0)
    ramp_run_range: tuple = (2.5, 4.5)
    hill_len_range: tuple = (2.0, 5.0)

    n_bumps_range: tuple = (2, 4)
    bump_height_range: tuple = (0.08, 0.30)
    bump_thickness_range: tuple = (0.12, 0.30)
    bump_spacing_range: tuple = (2.0, 3.5)

    n_stairs_range: tuple = (6, 15)
    step_height_range: tuple = (0.10, 0.18)
    step_run_range: tuple = (0.28, 0.34)

    n_gaps_range: tuple = (2, 4)
    gap_width_range: tuple = (0.15, 0.35)
    gap_floor_seg_range: tuple = (0.8, 1.2)

    n_platforms_range: tuple = (3, 5)
    platform_height_range: tuple = (0.15, 0.55)
    platform_face_x_range: tuple = (0.6, 1.0)
    platform_face_y: float = 1.0             # 窄踏石的深度 (不跨越宽度)
    platform_gap_range: tuple = (1.5, 2.8)


def make_training_terrain_cfg(
    *,
    width: float = 8.0,
    cell_length: float = 80.0,
    num_rows: int = 10,
    num_cols: int = 10,
    base_seed: int = 0,
    track_cfg: "MixedTrackCfg | None" = None,
) -> TerrainImporterCfg:
    """构建一个由随机化混合地形格组成的 TerrainImporterCfg，用于训练。

    生成 num_rows x num_cols 个不同的地形格；`difficulty` 随行递增 (课程式)，且
    每个地形格独立随机化。`cell_length` 应充分超过这些范围所能产生的最长赛道
    (超出的部分是无害的平坦尾段)。把返回的 cfg 接入你的
    训练环境的 scene.terrain。
    """
    track = track_cfg or MixedTrackCfg()
    track.width = width
    track.base_seed = base_seed
    return TerrainImporterCfg(
        class_type=BetterTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            class_type=BetterTerrainGenerator,
            seed=base_seed,
            size=(cell_length, max(width, 8.0)),
            border_width=0.0,
            num_rows=num_rows,
            num_cols=num_cols,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={"track": track},
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        # 纯色视觉：不依赖外部 MDL 材质资源 (训练通常是 headless 的)。
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
        debug_vis=False,
    )
