"""Short straight-stair terrain builder + per-cell geometry manifest (DESIGN.md).

Layout (local x, cell 20 x 8 m):
  bottom platform 2 m -> N up steps -> top platform 1.2 m -> N down steps
  -> bottom platform 2 m
Up tread k at z = k*h; down tread j at z = (N-j)*h. Manifest JSON per cell:
edges (x, z_low, z_high), platform intervals, spawn zones (bottom/top roles),
up/down goals, seed derived from deterministic cell order (row/col), and the
cell origin transform used by the manifest consumer.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import trimesh

MANIFEST_DIR = None  # set by env cfg before terrain generation


def make_straight_stairs(difficulty: float, cfg) -> tuple[list, np.ndarray]:
    """SubTerrain function: stairs with height/tread/steps from cfg ranges."""
    # NOTE: IsaacLab may pass a copy of the sub-terrain cfg, dropping attrs
    # preset by IndexedStairGenerator; fall back to a fixed seed (smoke
    # geometry uses single-valued ranges, so per-cell seeds are moot here).
    # Full per-cell seeding must instead be derived inside this function from
    # the terrain grid when wiring the real curriculum.
    rng = np.random.default_rng(getattr(cfg, "cell_seed", 0xF00D))
    h = float(rng.uniform(*cfg.step_height_range))
    N = int(round(rng.integers(cfg.num_steps_range[0], cfg.num_steps_range[1] + 1)))
    d = float(rng.uniform(*cfg.tread_range))
    return build_stair_meshes(h, N, d, cfg), np.array([1.0, cfg.size[1] / 2.0, 0.05])


def build_stair_meshes(h: float, N: int, d: float, cfg) -> list:
    W = cfg.width
    meshes: list[trimesh.Trimesh] = []

    def box(xc, zc, ext_x, ext_z, yc=None):
        yc = cfg.size[1] / 2.0 if yc is None else yc
        m = trimesh.creation.box(
            (ext_x, W, ext_z),
            trimesh.transformations.translation_matrix((xc, yc, zc)))
        m.fix_normals()
        return m

    x = 0.0
    x0_bottom = x                       # bottom platform 2 m
    meshes.append(box(1.0, -0.1, 2.0, 0.2))
    x += 2.0
    edges = []
    for k in range(1, N + 1):           # up steps: tread k at z = k*h
        top_z = k * h
        meshes.append(box(x + d / 2.0, top_z / 2.0, d, top_z))
        edges.append({"x_local": round(x, 4), "z_low": round((k - 1) * h, 4),
                      "z_high": round(top_z, 4), "dir": "up", "k": k})
        x += d
    top_z = N * h
    x_top_start = x                     # top platform 1.2 m
    meshes.append(box(x + 0.6, top_z / 2.0, 1.2, top_z))
    x += 1.2
    x_top_end = x
    for j in range(1, N + 1):           # down steps: tread j at z = (N-j)*h
        z = (N - j) * h
        if z > 1e-6:                     # raised treads need a box; the final
            meshes.append(box(x + d / 2.0, z / 2.0, d, z))
        # P1 fix 2026-10-25: register EVERY down edge. The z=0 tread previously
        # skipped BOTH box and edge -> manifest had N-1 treads, goals/targets
        # misderived, and (pre-heightfield) a 34 cm mesh gap. The final tread
        # is now covered by the trailing platform below.
        edges.append({"x_local": round(x, 4), "z_low": round(z, 4),
                      "z_high": round(z + h, 4), "dir": "down", "j": j})
        x += d
    trail_start = x - d                  # trailing platform spans the final
    trail_len = 2.0 + d                  # z=0 tread plus the 2 m exit apron
    total = trail_start + trail_len
    meshes.append(box(trail_start + trail_len / 2.0, -0.1, trail_len, 0.2))
    dn_last = max(e["x_local"] for e in edges if e["dir"] == "down")

    cfg._manifest = {
        "step_height_m": h, "num_steps": N, "tread_m": round(d, 4),
        "edges": edges,
        "platforms": {"bottom_pre": [0.0, 2.0], "top": [round(x_top_start, 4), round(x_top_end, 4)],
                      "bottom_post": [round(trail_start, 4), round(total, 4)]},
        "spawn_zones": {
            "bottom": {"x_local": [0.8, 1.2], "ground_z": 0.0},
            "top": {"x_local": [round(x_top_start + 0.3, 4),
                                 round(x_top_start + 0.3 + 0.4, 4)],
                    "ground_z": round(N * h, 4)}},
        "goals": {"up_goal_local_x": round(x_top_end - 0.1, 4),
                  # P2 fix 2026-10-25: last registered EDGE + 0.6 (was
                  # loop-end + d + 0.6 = one tread too far). Training spawn
                  # and screening both derive from this -> one shared line.
                  "down_goal_local_x": round(dn_last + 0.6, 4)},
        "cell_total_len_m": round(total, 4),
        "seed": getattr(cfg, "cell_seed", None),
    }
    if MANIFEST_DIR:
        Path(MANIFEST_DIR).mkdir(parents=True, exist_ok=True)
        (Path(MANIFEST_DIR) / f"cell_{getattr(cfg, '_row', 'x')}_{getattr(cfg, '_col', 'x')}.json").write_text(
            json.dumps(cfg._manifest, indent=1))
    return meshes


from isaaclab.terrains import SubTerrainBaseCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from terrain_base import BetterTerrainGenerator, BetterTerrainImporter


class IndexedStairGenerator(BetterTerrainGenerator):
    """Deterministic per-cell seed from (row, col); no opaque global counter."""
    def _get_terrain_mesh(self, difficulty, cfg):
        idx = self._cell_counter                     # deterministic cell order
        cfg._row, cfg._col = divmod(idx, self.cfg.num_cols)
        cfg.cell_seed = (self.cfg.seed * 1_000_003 + idx * 7919) & 0x7FFFFFFF
        return super()._get_terrain_mesh(difficulty, cfg)


@configclass
class StraightStairsCfg(SubTerrainBaseCfg):
    function = make_straight_stairs
    width: float = 8.0
    step_height_range: tuple = (0.04, 0.04)     # smoke: single 4 cm step
    num_steps_range: tuple = (1, 1)             # smoke: N = 1
    tread_range: tuple = (0.34, 0.34)


def make_stair_terrain_cfg(num_rows=2, num_cols=2, base_seed=0,
                           track_cfg=None) -> TerrainImporterCfg:
    from isaaclab.sim import spawners
    from isaaclab.terrains import TerrainGeneratorCfg
    track = track_cfg or StraightStairsCfg()
    track.width = 8.0
    return TerrainImporterCfg(
        class_type=BetterTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            class_type=IndexedStairGenerator,
            seed=base_seed,
            size=(20.0, 8.0),
            border_width=0.0,
            num_rows=num_rows,
            num_cols=num_cols,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={"stairs": track},
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=spawners.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )


# ---------------- training-grid cell types (dual-arm S1, DESIGN v1.1) ---------

def flat_cell(difficulty: float, cfg):
    import numpy as np, trimesh
    W = cfg.width
    m = trimesh.creation.box(
        (20.0, W, 0.2),
        trimesh.transformations.translation_matrix((10.0, cfg.size[1] / 2.0, -0.1)))
    m.fix_normals()
    return [m], np.array([1.0, cfg.size[1] / 2.0, 0.05])


def low_rough_cell(difficulty: float, cfg):
    """Fixed-pattern low roughness (deterministic; replay purpose, not
    difficulty progression). Heights 1-4 cm, wavelength ~0.5 m."""
    import numpy as np, trimesh
    W = cfg.width
    xs = np.linspace(0.0, 20.0, 200)
    ys = np.linspace(0.0, cfg.size[1], int(cfg.size[1] / 0.1))
    mesh = trimesh.creation.box(
        (20.0, W, 0.2),
        trimesh.transformations.translation_matrix((10.0, cfg.size[1] / 2.0, -0.1)))
    mesh.fix_normals()
    meshes = [mesh]
    rng = np.random.default_rng(1234)          # FIXED: same pattern every cell
    for _ in range(40):
        cx, cy = rng.uniform(2.0, 18.0), rng.uniform(1.0, cfg.size[1] - 1.0)
        h = rng.uniform(0.01, 0.04)
        r = rng.uniform(0.15, 0.35)
        m = trimesh.creation.box(
            (2 * r, 2 * r, h),
            trimesh.transformations.translation_matrix((cx, cy, h / 2.0)))
        m.fix_normals()
        meshes.append(m)
    return meshes, np.array([1.0, cfg.size[1] / 2.0, 0.05])


def make_dualarm_terrain(base_seed=0):
    """S1 training grid: 6 stairs / 2 flat / 2 low-rough per 10-cell cycle,
    deterministic via terrain_sequence (no proportion guesswork). Stair
    heights are TYPED (04/06/08) so every cell is exact and manifests are
    per-type, immune to cfg-copy attr loss."""
    from isaaclab.sim import spawners
    from isaaclab.terrains import TerrainGeneratorCfg, SubTerrainBaseCfg
    from isaaclab.utils import configclass
    from terrain_base import BetterTerrainGenerator, BetterTerrainGeneratorCfg

    def stair_cfg(h):
        c = StraightStairsCfg()
        c.step_height_range = (h, h)
        c.num_steps_range = (3, 3)
        c.tread_range = (0.34, 0.34)
        return c

    @configclass
    class _FlatCell(SubTerrainBaseCfg):
        function = flat_cell
        width: float = 8.0

    @configclass
    class _RoughCell(SubTerrainBaseCfg):
        function = low_rough_cell
        width: float = 8.0

    @configclass
    class _SeqGenerator(BetterTerrainGeneratorCfg):
        class_type = BetterTerrainGenerator   # cfg base does not carry it
        terrain_sequence = ["stairs04", "flat", "stairs06", "rough",
                            "stairs08", "stairs04", "flat", "stairs06",
                            "rough", "stairs08"]

    return TerrainImporterCfg(
        class_type=BetterTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=_SeqGenerator(
            seed=base_seed, size=(20.0, 8.0), border_width=0.0,
            num_rows=2, num_cols=10, horizontal_scale=0.1,
            vertical_scale=0.005, slope_threshold=0.75, use_cache=False,
            sub_terrains={"stairs04": stair_cfg(0.04), "stairs06": stair_cfg(0.06),
                          "stairs08": stair_cfg(0.08),
                          "flat": _FlatCell(), "rough": _RoughCell()}),
        max_init_terrain_level=0, collision_group=-1,
        physics_material=spawners.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False)


# ---------------- v1.3 difficulty-row grid (DESIGN.md v1.3) ----------------
# Rows = difficulty (row 0 buffer: flat/rough; rows 1.. stair heights), cols =
# variants (identical typed cells this round; per-col variation deferred).
# Cell 8 x 8 m: the S1 route (N=3, tread 0.34) is 7.24 m and fits with exit
# margin. 20 m cells would let an episode end before any completion
# semantics could trigger inside a 25 s episode.

def flat_cell_sized(difficulty: float, cfg):
    L = float(cfg.size[0])
    m = trimesh.creation.box(
        (L, cfg.width, 0.2),
        trimesh.transformations.translation_matrix((L / 2.0, cfg.size[1] / 2.0, -0.1)))
    m.fix_normals()
    return [m], np.array([1.0, cfg.size[1] / 2.0, 0.05])


def low_rough_cell_sized(difficulty: float, cfg):
    L = float(cfg.size[0])
    base = trimesh.creation.box(
        (L, cfg.width, 0.2),
        trimesh.transformations.translation_matrix((L / 2.0, cfg.size[1] / 2.0, -0.1)))
    base.fix_normals()
    meshes = [base]
    rng = np.random.default_rng(1234)          # FIXED pattern: replay cells identical
    for _ in range(40):
        cx, cy = rng.uniform(1.0, L - 1.0), rng.uniform(1.0, cfg.size[1] - 1.0)
        h = rng.uniform(0.01, 0.04)
        r = rng.uniform(0.15, 0.35)
        m = trimesh.creation.box(
            (2 * r, 2 * r, h),
            trimesh.transformations.translation_matrix((cx, cy, h / 2.0)))
        m.fix_normals()
        meshes.append(m)
    return meshes, np.array([1.0, cfg.size[1] / 2.0, 0.05])


def rowgrid_sequence(step_heights=(0.04, 0.06, 0.08), num_cols=4):
    """Row-major generation order: row 0 buffer (flat/rough alternating),
    then one stair height per row, num_cols copies each."""
    seq = ["flat" if c % 2 == 0 else "rough" for c in range(num_cols)]
    for h in step_heights:
        seq.extend([f"stairs{int(round(h * 100)):02d}"] * num_cols)
    return seq


def make_rowgrid_terrain(base_seed=0, cell_len=8.0,
                         step_heights=(0.04, 0.06, 0.08), num_cols=4):
    from isaaclab.sim import spawners
    from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
    from isaaclab.utils import configclass
    from terrain_base import BetterTerrainGenerator, BetterTerrainGeneratorCfg

    def stair_cfg(h):
        c = StraightStairsCfg()
        c.step_height_range = (h, h)
        c.num_steps_range = (3, 3)
        c.tread_range = (0.34, 0.34)
        return c

    @configclass
    class _FlatSized(SubTerrainBaseCfg):
        function = flat_cell_sized
        width: float = 8.0

    @configclass
    class _RoughSized(SubTerrainBaseCfg):
        function = low_rough_cell_sized
        width: float = 8.0

    seq = list(rowgrid_sequence(step_heights, num_cols))
    num_rows = 1 + len(step_heights)

    @configclass
    class _RowGridGenerator(BetterTerrainGeneratorCfg):
        class_type = BetterTerrainGenerator
        terrain_sequence = seq

    sub = {"flat": _FlatSized(), "rough": _RoughSized()}
    for h in step_heights:
        sub[f"stairs{int(round(h * 100)):02d}"] = stair_cfg(h)
    return TerrainImporterCfg(
        class_type=BetterTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=_RowGridGenerator(
            seed=base_seed, size=(cell_len, 8.0), border_width=0.0,
            num_rows=num_rows, num_cols=num_cols, horizontal_scale=0.1,
            vertical_scale=0.005, slope_threshold=0.75, use_cache=False,
            sub_terrains=sub),
        max_init_terrain_level=0, collision_group=-1,
        physics_material=spawners.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False)
