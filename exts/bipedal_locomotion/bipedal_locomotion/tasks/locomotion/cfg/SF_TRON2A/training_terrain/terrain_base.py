# BetterTerrainGenerator / BetterTerrainImporter —— 对 IsaacLab 的 TerrainGenerator /
# TerrainImporter 的轻量封装，它 (a) 让你能驱动一个固定的子地形键 `terrain_sequence`，且 (b) 加入一道
# 永久的安全网：每个子地形 primitive 在成为 PhysX collider 之前都会被校验为具有正体积 (朝外的
# 面法线)。PhysX 三角网格 collider 是单面的，因此
# 反向/不一致的缠绕 (负体积) 会在其内侧面发生碰撞，机器人会从中
# 穿透 —— 在地形构建器中调用 `mesh.fix_normals()` 以避免此问题。
#
# 自包含：仅需 IsaacLab。
from __future__ import annotations

from typing import Callable

import trimesh
import numpy as np
from isaaclab.terrains import (
    SubTerrainBaseCfg,
    TerrainImporterCfg,
    TerrainImporter,
    TerrainGenerator,
)
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.spawners.from_files import spawn_ground_plane
from isaaclab.sim.spawners import materials


class BetterTerrainImporter(TerrainImporter):
    def __init__(self, cfg: TerrainImporterCfg):
        """初始化 terrain importer。与 IsaacLab 的 TerrainImporter 不同，
        本类持有 terrain generator 对象。"""
        cfg.validate()
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None
        self._terrain_flat_patches = dict()

        self.terrain_generator = None
        if self.cfg.terrain_type == "generator":
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            terrain_generator = self.cfg.terrain_generator.class_type(
                cfg=self.cfg.terrain_generator, device=self.device
            )
            self.import_mesh("terrain", terrain_generator.terrain_mesh)
            self.configure_env_origins(terrain_generator.terrain_origins)
            self._terrain_flat_patches = terrain_generator.flat_patches
            self.terrain_generator = terrain_generator
        elif self.cfg.terrain_type == "usd":
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")
            self.import_usd("terrain", self.cfg.usd_path)
            self.configure_env_origins()
        elif self.cfg.terrain_type == "plane":
            self.import_ground_plane("terrain")
            self.configure_env_origins()
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        self.set_debug_vis(self.cfg.debug_vis)

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
        """向 terrain importer 添加一个 plane。"""
        prim_path = self.cfg.prim_path + f"/{name}"
        if prim_path in self.terrain_prim_paths:
            raise ValueError(f"A terrain with the name '{name}' already exists.")
        self.terrain_prim_paths.append(prim_path)

        color = (0.0, 0.0, 0.0)
        if self.cfg.visual_material is not None:
            material = self.cfg.visual_material.to_dict()
            if "diffuse_color" in material:
                color = material["diffuse_color"]

        ground_plane_cfg = GroundPlaneCfg(physics_material=self.cfg.physics_material, size=size, color=color)
        ground_plane_cfg.func(prim_path, ground_plane_cfg)


@configclass
class GroundPlaneCfg(SpawnerCfg):
    """创建一个 ground plane prim (仅由 'plane' 地形类型使用)。"""

    func: Callable = spawn_ground_plane

    usd_path: str = ""  # 仅在使用 "plane" 地形类型时设置；generator 不使用它
    color: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    size: tuple[float, float] = (100.0, 100.0)
    physics_material: materials.RigidBodyMaterialCfg = materials.RigidBodyMaterialCfg()


@configclass
class BetterTerrainGeneratorCfg(TerrainGeneratorCfg):
    terrain_sequence: list[str] | None = None


class BetterTerrainGenerator(TerrainGenerator):
    sub_terrain_types = []
    _cell_counter = 0

    def _get_terrain_mesh(
        self, difficulty: float, cfg: SubTerrainBaseCfg
    ) -> tuple[trimesh.Trimesh, np.ndarray]:

        seq = getattr(self.cfg, "terrain_sequence", None)
        if seq is not None:
            key = seq[self._cell_counter % len(seq)]
            if key not in self.cfg.sub_terrains:
                raise KeyError(
                    f"terrain_sequence key '{key}' not in cfg.sub_terrains: {list(self.cfg.sub_terrains.keys())}")
            cfg = self.cfg.sub_terrains[key]
        else:
            key = [k for k, v in self.cfg.sub_terrains.items() if v == cfg][0]

        self._cell_counter += 1
        self.sub_terrain_types.append(key)

        # 永久的安全网：在每个 primitive 成为 collider 之前校验其面法线。
        orig_fn = cfg.function

        def _validate_outward_normals(diff, c, _fn=orig_fn):
            meshes, origin = _fn(diff, c)
            for i, m in enumerate(meshes):
                if getattr(m, "is_watertight", False) and m.volume <= 0.0:
                    raise ValueError(
                        f"[terrain] sub-mesh #{i} from '{getattr(_fn, '__name__', '?')}' has non-positive "
                        f"volume ({m.volume:.3f}): inverted/inconsistent face normals. PhysX triangle "
                        f"colliders are single-sided, so the robot would fall through it — call "
                        f"mesh.fix_normals() in the terrain builder."
                    )
            return meshes, origin

        cfg.function = _validate_outward_normals
        try:
            return super()._get_terrain_mesh(difficulty, cfg)
        finally:
            cfg.function = orig_fn
