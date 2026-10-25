"""Percept-Stair: fix3 perception + short straight stairs + role-based legal spawn.

Per stair_curriculum DESIGN.md (2026-10-24) round 1:
- inherits the CURRENT Percept wiring (fix3 cfg, hash f87af52b) verbatim:
  scanner, miss-safe heights, valid-hit base_height, TerrainGaitReward
- terrain: straight stairs (smoke defaults: single 4 cm step, tread 34 cm)
- reset: 75% bottom-platform / 25% top-platform spawn; tight ranges
  (yaw +-0.05, lateral +-0.10 m, lin/ang vel +-0.05, joints +-0.05 rad);
  root z = platform ground z + USD default root height + 0.01 m clearance
  (read from robot.data.default_root_state at first reset; nothing hardcoded)
- distance curriculum OFF; success/failure stats NOT yet wired (deferred to
  the training round per design; smoke validates terrain+spawn+obs+reward)
"""
from __future__ import annotations

import math

import torch

from bipedal_locomotion.tasks.locomotion.robots.limx_solefoot_yg_tron2a_env_cfg import (
    SFYG_TRON2A_BlindRoughEnvCfg,
    SFYG_TRON2A_BlindRoughEnvCfg_PLAY,
)
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.tasks.locomotion.robots.percept_yg_tron2a_env_cfg import (
    _apply_perception,  # fix3 wiring (miss-safe heights, valid-hit rewards)
)
try:
    import stair_terrain  # terrain_kit sys.path entry set by the launcher
    _STAIR_TERRAIN_AVAILABLE = True
except ImportError:
    # P1 fix 2026-10-25: terrain_kit is launcher-provided, not vendored. A
    # bare import here broke module load on machines without it — which
    # took down the Flat/Rough entries too (they share robots/__init__).
    # With the guard, Flat/Rough boot fine; Stair cfgs raise on use only.
    stair_terrain = None
    _STAIR_TERRAIN_AVAILABLE = False


def _require_stair_terrain():
    if stair_terrain is None:
        raise ImportError(
            "Percept-Stair requires the terrain_kit package (stair_terrain) "
            "on sys.path — launchers from /root/autodl-tmp/terrain_kit set "
            "this. Flat/Rough tasks remain usable without it.")


def spawn_roled_stairs(env, robot_cfg: SceneEntityCfg, manifest_geom,
                       prob_bottom: float = 0.75, env_ids=None):
    """Role-based legal spawn on the straight-stair cell.

    z = spawn ground z + USD-default standing height + 0.01 m sole clearance.
    Manifest geometry (bottom/top zone local x, ground z) is passed by the
    event closure built in the cfg below.
    """
    robot = env.scene[robot_cfg.name]
    n = robot.data.root_pos_w.shape[0]
    dev = robot.device
    if env_ids is not None:
        env_ids = env_ids.tolist()
    else:
        env_ids = list(range(n))
    # USD-derived standing offset, measured once (NOT a hardcoded constant)
    if not hasattr(env, "_stair_default_root_offset"):
        dz = robot.data.default_root_state[0, 2]
        env._stair_default_root_offset = float(dz)
    off = env._stair_default_root_offset

    bottom = manifest_geom["spawn_zones"]["bottom"]
    top = manifest_geom["spawn_zones"]["top"]
    origin_local_x = 1.0                     # sub-terrain origin x (see builder)
    for i in env_ids:
        eorg = env.scene.env_origins[i]
        take_bottom = bool(torch.rand(1, device=dev).item() < prob_bottom)
        zone = bottom if take_bottom else top
        x_local = float(torch.empty(1).uniform_(zone["x_local"][0], zone["x_local"][1]).item())
        x_world = float(eorg[0]) + (x_local - origin_local_x)
        y_world = float(eorg[1]) + float(torch.empty(1).uniform_(-0.10, 0.10).item())
        z_world = zone["ground_z"] + off + 0.01
        yaw = float(torch.empty(1).uniform_(-0.05, 0.05).item())
        robot.write_root_pose_to_sim(
            torch.tensor([[x_world, y_world, z_world,
                           math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
                         dtype=torch.float, device=dev),
            torch.tensor([i], dtype=torch.long, device=dev))
        robot.write_root_velocity_to_sim(
            torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float, device=dev),
            torch.tensor([i], dtype=torch.long, device=dev))
    return


def _stage_params_from_env(defaults):
    """Per-process stage injection: STAIR_GEOMETRY (json) and STAIR_SPAWN_ROLE
    (bottom|top|mixed) let eval/train launchers select curriculum geometry and
    spawn role without touching the registered cfg class."""
    import json as _json
    import os as _os
    params = dict(defaults)
    geo = _os.environ.get("STAIR_GEOMETRY")
    if geo:
        params.update(_json.loads(geo))
    print(f"[STAIR_GEOMETRY_EFFECTIVE] {params}", flush=True)
    return params


def _apply_stairs(cfg, manifest_dir, episode_length_s, stair_params):
    _require_stair_terrain()
    _apply_perception(cfg)                    # fix3 verbatim
    stair_terrain.MANIFEST_DIR = manifest_dir
    track = stair_terrain.StraightStairsCfg()
    track.step_height_range = stair_params["step_height_range"]
    track.num_steps_range = stair_params["num_steps_range"]
    track.tread_range = stair_params["tread_range"]
    cfg.scene.terrain = stair_terrain.make_stair_terrain_cfg(
        num_rows=stair_params.get("num_rows", 2),
        num_cols=stair_params.get("num_cols", 2),
        base_seed=stair_params.get("base_seed", 0), track_cfg=track)
    cfg.curriculum.terrain_levels = None      # distance curriculum OFF
    cfg.episode_length_s = episode_length_s
    vel = cfg.commands.base_velocity
    vel.heading_command = False
    vel.ranges.lin_vel_x = (0.3, 0.5)         # design table S1 forward command
    vel.ranges.lin_vel_y = (0.0, 0.0)
    vel.ranges.ang_vel_z = (0.0, 0.0)
    # role-based spawn replaces the default randomized root reset
    track.cell_seed = stair_params.get("base_seed", 0)
    stair_terrain.build_stair_meshes(
        stair_params["step_height_range"][0], stair_params["num_steps_range"][0],
        stair_params["tread_range"][0], track)
    manifest = track._manifest
    import os as _os
    _role = _os.environ.get("STAIR_SPAWN_ROLE", "mixed")
    _pb = {"bottom": 1.0, "top": 0.0, "mixed": 0.75}[_role]
    # self-documenting: every run logs its effective spawn role (the screening
    # bug hunt showed env-var wiring must never be silent)
    print(f"[STAIR_SPAWN_EFFECTIVE] role={_role!r} prob_bottom={_pb}", flush=True)

    def _spawn(env, env_ids, manifest):
        spawn_roled_stairs(env, SceneEntityCfg("robot"), manifest,
                           env_ids=env_ids, prob_bottom=_pb)
    cfg.events.reset_robot_base = EventTermCfg(
        func=_spawn, mode="reset", params={"manifest": manifest})
    # tight joint reset (design: +-0.05 rad)
    cfg.events.reset_robot_joints.params["position_range"] = (-0.05, 0.05)
    return manifest


@configclass
class SFYG_TRON2A_PerceptStairEnvCfg(SFYG_TRON2A_BlindRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _params = _stage_params_from_env(
            {"step_height_range": (0.04, 0.04), "num_steps_range": (1, 1),
             "tread_range": (0.34, 0.34)})
        self._stair_manifest = _apply_stairs(
            self, manifest_dir="/root/autodl-tmp/stair_manifests",
            episode_length_s=25.0, stair_params=_params)


@configclass
class SFYG_TRON2A_PerceptStairEnvCfg_PLAY(SFYG_TRON2A_BlindRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _params = _stage_params_from_env(
            {"step_height_range": (0.04, 0.04), "num_steps_range": (1, 1),
             "tread_range": (0.34, 0.34)})
        self._stair_manifest = _apply_stairs(
            self, manifest_dir="/root/autodl-tmp/stair_manifests",
            episode_length_s=25.0, stair_params=_params)
