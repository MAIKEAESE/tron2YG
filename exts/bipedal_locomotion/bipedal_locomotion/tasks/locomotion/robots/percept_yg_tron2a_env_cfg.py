"""SFYG TRON2A Percept-Rough environment: Blind-Rough + grid height scan in obs.

2026-10-23 fix round 2 (ray-miss NaN chain, per nan_capture fault analysis):
The official mdp.height_scan is `pos_w_z - hit_z - offset` with NO clip, so a
single missed ray (IsaacLab convention: hit = ±Inf) propagated -Inf into
policy/critic heights (action mean NaN -> Normal(loc) error) and into the
base_height scan mean. Fixes, all preserving finite-ray values bit-exactly:

  1. height_scan_miss_safe: same formula, result clipped to [-1, 1].
     Finite in-range rays are unchanged (measured 0.35 on flat @ root 0.85);
     a miss maps to the -1.0 bound instead of -Inf.
  2. base_com_height_valid_hits: terrain datum = mean over FINITE hits only
     (fallback to flat assumption 0.0 for an all-miss grid).
  3. TerrainGaitReward foot lookup: XY-nearest ray (no grid-flatten-order
     assumption, after the handoff candidate) with all-miss fallback to the
     finite-hit mean; misses never selected as nearest (Inf xy -> Inf dist).

obsHistory group stays 42-dim x 10 (encoder input 420, runner patch).
Heights at policy offset 42 / critic offset 45 (as measured).
"""

from __future__ import annotations

import torch

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.mdp.rewards import GaitReward
from bipedal_locomotion.tasks.locomotion.robots.limx_solefoot_yg_tron2a_env_cfg import (
    SFYG_TRON2A_BlindRoughEnvCfg,
    SFYG_TRON2A_BlindRoughEnvCfg_PLAY,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

HEIGHT_SCAN_RES = 0.05
HEIGHT_SCAN_SIZE = [0.5, 0.5]          # 11x11 = 121 rays, same as in-repo pattern
HEIGHT_SCAN_DIM = 121


def height_scan_miss_safe(env, sensor_cfg, offset: float = 0.5):
    """Official mdp.height_scan semantics + explicit miss handling.

    Finite rays: pos_w_z - hit_z - offset, unchanged (no clip was applied
    before, and in-range values are clip-invariant). Missed rays (hit = Inf):
    would yield -Inf; the clip maps them to the -1.0 bound ("no ground within
    ~1.5 m below the sensor").
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return torch.clip(height, -1.0, 1.0)


def base_com_height_valid_hits(env, target_height: float, sensor_cfg,
                               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """base_com_height with a valid-hits-only terrain datum.

    Same L1-abs form as the in-repo base_com_height; the scan mean uses only
    finite hits per env. All-miss grid falls back to the flat assumption 0.0.
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w[..., 2]
    finite = torch.isfinite(hits)
    # replacement masking, NOT hits*finite: Inf*0 = NaN in IEEE would
    # re-introduce the very nonfinite this term is supposed to remove
    mean_z = torch.where(finite, hits, torch.zeros_like(hits)).sum(dim=1)         / finite.sum(dim=1).clamp(min=1)
    mean_z = torch.where(finite.any(dim=1), mean_z, torch.zeros_like(mean_z))
    return torch.abs(asset.data.root_pos_w[:, 2] - (target_height + mean_z))


def make_height_scanner_cfg(update_dt: float) -> RayCasterCfg:
    cfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_Link",
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=HEIGHT_SCAN_RES, size=HEIGHT_SCAN_SIZE),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    cfg.update_period = update_dt
    return cfg


class TerrainGaitReward(GaitReward):
    """GaitReward fork with terrain-relative foot heights; only the height
    term changes. Foot lookup: XY-nearest ray over the grid; missed rays
    (Inf xyz) yield Inf distance and are never selected; if the selected or
    entire grid is degenerate, fall back to the finite-hit mean."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.height_sensor_name = cfg.params.get("height_sensor_name", "height_scanner")

    def _local_ground_under_feet(self, env) -> torch.Tensor:
        hits = env.scene.sensors[self.height_sensor_name].data.ray_hits_w
        feet = env.scene["robot"].data.body_pos_w[:, self.asset_cfg.body_ids]
        distance2 = (feet[..., :2].unsqueeze(2) - hits[:, None, :, :2]).square().sum(-1)
        minimum, index = distance2.min(-1)
        ground = hits[..., 2].gather(1, index)
        finite_z = torch.isfinite(hits[..., 2])
        # replacement masking (Inf*0 = NaN hazard, see base_com_height_valid_hits)
        fallback = (torch.where(finite_z, hits[..., 2], torch.zeros_like(hits[..., 2]))
                    .sum(1) / finite_z.sum(1).clamp(min=1)).unsqueeze(1).expand_as(ground)
        ground = torch.where(torch.isfinite(ground), ground, fallback)
        # coverage diagnostics (not a fault): nearest-sample distance + support
        self.last_ground_ray_distance_m = minimum.sqrt().detach()
        self.last_ground_lookup_supported = minimum <= (HEIGHT_SCAN_RES ** 2 / 2 + 1e-8)
        return ground

    def __call__(
        self,
        env,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        tracking_contacts_shaped_height,
        gait_force_sigma,
        gait_vel_sigma,
        gait_height_sigma,
        touch_down_vel,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
        use_reference_motion,
        height_sensor_name="height_scanner",
    ) -> torch.Tensor:
        gait_params = env.command_manager.get_command(self.command_name)  # type: ignore
        gait_indices = env.command_manager.get_term(self.command_name).gait_indices  # type: ignore

        desired_contact_states = self.compute_contact_targets(gait_params)
        self.compute_desired_foot_height(gait_params, gait_indices)

        foot_forces = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1
        )
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)
        total_reward = force_reward

        if self.vel_scale != 0:
            foot_velocities = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids]
            velocity_reward = self._compute_velocity_reward(
                foot_velocities, self.des_foot_velocity_z, desired_contact_states
            )
            total_reward += velocity_reward

        if self.height_scale != 0:
            feet_world = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids]
            foot_heights = feet_world[..., 2] - self._local_ground_under_feet(env)
            height_reward = self._compute_height_reward(
                foot_heights, self.des_foot_height, desired_contact_states
            )
            total_reward += height_reward

        is_standing_env = env.command_manager.get_term("base_velocity").is_standing_env  # type: ignore
        no_gait_env_ids = is_standing_env.nonzero(as_tuple=False).flatten()
        total_reward[no_gait_env_ids] = 0.0
        return total_reward


def _apply_perception(cfg):
    """Shared wiring for train and PLAY variants."""
    cfg.scene.height_scanner = make_height_scanner_cfg(cfg.decimation * cfg.sim.dt)
    cfg.observations.policy.heights = ObsTerm(
        func=height_scan_miss_safe,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=GaussianNoise(mean=0.0, std=0.01),
    )
    cfg.observations.critic.heights = ObsTerm(
        func=height_scan_miss_safe,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    cfg.rewards.base_height.func = base_com_height_valid_hits
    cfg.rewards.base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
    cfg.rewards.gait_reward.func = TerrainGaitReward
    cfg.rewards.gait_reward.params["height_sensor_name"] = "height_scanner"


@configclass
class SFYG_TRON2A_PerceptRoughEnvCfg(SFYG_TRON2A_BlindRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_perception(self)


@configclass
class SFYG_TRON2A_PerceptRoughEnvCfg_PLAY(SFYG_TRON2A_BlindRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_perception(self)
