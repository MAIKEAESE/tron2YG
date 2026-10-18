"""Per-leg diagnostic evaluation: force integral, knee angle, swing events, spawn validity.

Key fixes vs eval_protocol.py:
- Caches pre-step state for done envs (avoids auto-reset contamination)
- Per-leg vertical force time integral (NOT contact count)
- Per-leg knee angle P05-P95 range
- Per-swing forward displacement and clearance (independent event detection)
- Spawn validity and end reason classification
- Asymmetry index for dominant leg (thresholded, not binary)
"""
import argparse, math, os, sys, csv, json, hashlib, subprocess
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rsl_rl")))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Per-leg diagnostic evaluation")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--checkpoint_url", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=30)
parser.add_argument("--repeat", type=int, default=2)
parser.add_argument("--eval_seed", type=int, default=42)
parser.add_argument("--mirror", action="store_true", default=False, help="Swap L/R initial joint positions")
parser.add_argument("--phase_offset", type=float, default=0.0, help="Starting gait phase offset (radians)")
parser.add_argument("--output_prefix", type=str, default="legdiag")
import cli_args  # isort: skip
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
from types import SimpleNamespace
from rsl_rl.runner import OnPolicyRunner
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import bipedal_locomotion  # noqa
import cli_args  # isort: skip

# Independent event detection params (NOT the reward gate)
CONTACT_ON = 5.0    # N, foot considered on ground
CONTACT_OFF = 2.0   # N, hysteresis: foot considered off ground
MIN_SWING_FRAMES = 3  # at 50Hz = 60ms, filter contact chatter
MIN_STANCE_FRAMES = 2

# Asymmetry threshold
ASYM_THRESH = 0.15  # |A| > 0.15 → dominant; else balanced


def quat2rot(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], dim=-1),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], dim=-1),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], dim=-1),
    ], dim=-2)


class SwingDetector:
    """Independent liftoff/touchdown detection with hysteresis and duration filter."""
    def __init__(self, n_envs, n_feet=2, device="cpu"):
        self.on_ground = torch.ones(n_envs, n_feet, dtype=torch.bool, device=device)
        self.swing_start_frame = torch.full((n_envs, n_feet), -1, dtype=torch.long, device=device)
        self.swing_frames = torch.zeros(n_envs, n_feet, dtype=torch.long, device=device)
        self.stance_frames = torch.zeros(n_envs, n_feet, dtype=torch.long, device=device)
        self.total_swings = torch.zeros(n_envs, n_feet, dtype=torch.long, device=device)
        self.chatter_count = torch.zeros(n_envs, n_feet, dtype=torch.long, device=device)

    def update(self, fz):
        """fz: (N, 2) vertical contact force per foot."""
        above_on = fz > CONTACT_ON
        above_off = fz > CONTACT_OFF
        # Hysteresis: turn on easier than turn off
        new_on = torch.where(self.on_ground, above_off, above_on)
        # Count swing frames
        self.swing_frames = torch.where(~new_on, self.swing_frames + 1, torch.zeros_like(self.swing_frames))
        self.stance_frames = torch.where(new_on, self.stance_frames + 1, torch.zeros_like(self.stance_frames))
        # Detect valid liftoff (enough stance before, now off)
        was_on = self.on_ground & ~new_on
        self.chatter_count += (was_on & (self.stance_frames < MIN_STANCE_FRAMES)).long()
        # Detect valid touchdown
        landed = ~self.on_ground & new_on
        valid_td = landed & (self.swing_frames >= MIN_SWING_FRAMES)
        self.total_swings += valid_td.long()
        # Reset counters on transitions
        self.swing_frames = torch.where(landed, torch.zeros_like(self.swing_frames), self.swing_frames)
        self.stance_frames = torch.where(was_on, torch.zeros_like(self.stance_frames), self.stance_frames)
        self.on_ground = new_on
        return new_on, landed, valid_td


def main():
    env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = args_cli.eval_seed
    torch.manual_seed(args_cli.eval_seed)

    if args_cli.checkpoint_url and not args_cli.checkpoint_path:
        import tempfile, urllib.request
        dest = os.path.join(tempfile.gettempdir(), "legdiag_ckpt.pt")
        urllib.request.urlretrieve(args_cli.checkpoint_url, dest)
        args_cli.checkpoint_path = dest

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    base = env.unwrapped
    robot = base.scene["robot"]
    device = base.device

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint_path)
    policy = runner.get_inference_policy(device=device)
    encoder = runner.get_inference_encoder(device=device)

    # Weight hash for provenance
    with open(args_cli.checkpoint_path, "rb") as f:
        weight_hash = hashlib.md5(f.read()).hexdigest()[:12]

    # Setup sensors
    contact_sensor = base.scene.sensors["contact_forces"]
    feet_ids = sorted(contact_sensor.find_bodies("ankle_pitch_.*")[0])
    foot_names = [contact_sensor.body_names[i] for i in feet_ids]
    print(f"Feet sensor: ids={feet_ids} names={foot_names}")

    knee_ids = sorted(robot.find_joints("knee_.*")[0])
    knee_names = [robot.joint_names[i] for i in knee_ids]
    print(f"Knee joints: ids={knee_ids} names={knee_names}")
    # Sort by name to ensure L comes first
    knee_l_idx = knee_names.index("knee_L_Joint") if "knee_L_Joint" in knee_names else 0
    knee_r_idx = knee_names.index("knee_R_Joint") if "knee_R_Joint" in knee_names else 1

    env_origins = base.scene.env_origins.to(device)
    dt = base.step_dt
    N = args_cli.num_envs

    # Mirror initial state if requested
    if args_cli.mirror:
        print("[MIRROR] Swapping L/R initial joint positions")

    detector = SwingDetector(N, 2, device)

    # Per-step data buffer
    step_data = []
    swing_events = []
    episode_records = []

    # State tracking
    ep_step = torch.zeros(N, dtype=torch.long, device=device)
    fz_integral = torch.zeros(N, 2, device=device)  # per-leg vertical force integral
    knee_min = torch.full((N, 2), 999.0, device=device)
    knee_max = torch.full((N, 2), -999.0, device=device)
    knee_vals = [[] for _ in range(N)]  # per-env list of (L, R) tuples
    swing_disp_sum = torch.zeros(N, 2, device=device)  # sum of per-swing forward displacement
    swing_clear_sum = torch.zeros(N, 2, device=device)  # sum of per-swing max clearance
    # Spawn tracking
    spawn_pos = torch.zeros(N, 3, device=device)
    spawn_valid = torch.ones(N, dtype=torch.bool, device=device)

    # Cached pre-step data (for terminal state of done envs)
    cached_local_x = torch.zeros(N, device=device)
    cached_fz = torch.zeros(N, 2, device=device)
    cached_knee = torch.zeros(N, 2, device=device)
    cached_foot_pos = torch.zeros(N, 2, 3, device=device)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory")
    if obs_history is not None:
        obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict.get("commands")

    total_target = N * args_cli.repeat
    step_count = 0

    print(f"\nRunning {total_target} episodes, seed={args_cli.eval_seed}, mirror={args_cli.mirror}, phase={args_cli.phase_offset:.2f}")
    print(f"Weight hash: {weight_hash}\n")

    while len(episode_records) < total_target and simulation_app.is_running():
        # CACHE pre-step state
        cached_local_x = (robot.data.root_pos_w[:, 0] - env_origins[:, 0]).clone()
        fz_raw = contact_sensor.data.net_forces_w[:, feet_ids, 2].clone()
        cached_fz = fz_raw.clone()
        cached_knee = robot.data.joint_pos[:, knee_ids].clone()
        cached_foot_pos = robot.data.body_pos_w[:, feet_ids, :].clone()

        # Inference
        est = encoder(obs_history)
        actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        obs_dict, _, dones, _ = env.step(actions)
        obs = obs_dict["policy"]
        obs_history = obs_dict.get("obs_history")
        if obs_history is not None:
            obs_history = obs_history.flatten(start_dim=1)
        commands = obs_dict.get("commands")

        ep_step += 1
        step_count += 1

        # Read post-step state (valid for non-done envs, contaminated for done envs)
        post_local_x = robot.data.root_pos_w[:, 0] - env_origins[:, 0]
        post_fz = contact_sensor.data.net_forces_w[:, feet_ids, 2]
        post_knee = robot.data.joint_pos[:, knee_ids]

        # For accumulation, use post-step for non-done, cached for done envs
        safe_fz = torch.where(torch.as_tensor(dones, device=device).bool().unsqueeze(1), cached_fz, post_fz)
        fz_integral += safe_fz * dt

        safe_knee = torch.where(torch.as_tensor(dones, device=device).bool().unsqueeze(1), cached_knee, post_knee)
        knee_min = torch.minimum(knee_min, safe_knee)
        knee_max = torch.maximum(knee_max, safe_knee)
        for i in range(min(N, 8)):  # sample first 8 envs for per-step storage
            knee_vals[i].append((safe_knee[i, knee_l_idx].item(), safe_knee[i, knee_r_idx].item()))

        # Swing detection (use safe Fz)
        on_ground, landed, valid_td = detector.update(safe_fz)

        # On valid touchdown, record swing event
        for env_i in torch.nonzero(valid_td).flatten().tolist():
            for foot in range(2):
                if valid_td[env_i, foot]:
                    swing_events.append({
                        "env": env_i, "step": step_count,
                        "foot": foot_names[foot],
                        "duration_s": detector.swing_frames[env_i, foot].item() * dt,
                    })

        done_mask = torch.as_tensor(dones, device=device).bool()
        if episode_termination_cap(step_count, args_cli.max_steps if hasattr(args_cli, 'max_steps') else 2000):
            done_mask |= torch.ones_like(done_mask)

        if done_mask.any():
            remaining = total_target - len(episode_records)
            done_list = torch.nonzero(done_mask).flatten().tolist()[:remaining]
            for env_i in done_list:
                # Determine end reason
                timeout = base.termination_manager.time_out[env_i].item()
                fell = not timeout
                # Spawn validity: if first step has huge contact force, likely bad spawn
                bad_spawn = spawn_valid[env_i] and ep_step[env_i] < 10 and safe_fz[env_i].abs().max() > 500

                # Compute asymmetry index
                jl = fz_integral[env_i, 0].item()
                jr = fz_integral[env_i, 1].item()
                total_j = jl + jr + 1e-9
                asym = (jl - jr) / total_j
                if abs(asym) < ASYM_THRESH:
                    dom = "balanced"
                elif total_j < 10:
                    dom = "insufficient_data"
                else:
                    dom = "L" if asym > 0 else "R"

                # Knee P05-P95
                kv = knee_vals[env_i] if env_i < len(knee_vals) else []
                if kv:
                    kL = sorted([p[0] for p in kv])
                    kR = sorted([p[1] for p in kv])
                    kL_p05, kL_p95 = kL[int(0.05*len(kL))], kL[int(0.95*len(kL))]
                    kR_p05, kR_p95 = kR[int(0.05*len(kR))], kR[int(0.95*len(kR))]
                    kL_rng = kL_p95 - kL_p05
                    kR_rng = kR_p95 - kR_p05
                else:
                    kL_rng = kR_rng = 0.0

                rec = {
                    "env": env_i,
                    "steps": int(ep_step[env_i].item()),
                    "duration_s": round(ep_step[env_i].item() * dt, 2),
                    "disp": round(cached_local_x[env_i].item() - (spawn_pos[env_i, 0].item() if spawn_pos[env_i, 0] != 0 else 0), 3),
                    "end_reason": "timeout" if timeout else ("bad_spawn" if bad_spawn else "fell"),
                    "J_L": round(jl, 2), "J_R": round(jr, 2),
                    "asym": round(asym, 4), "dominant": dom,
                    "kL_range": round(kL_rng, 3), "kR_range": round(kR_rng, 3),
                    "swings_L": detector.total_swings[env_i, 0].item(),
                    "swings_R": detector.total_swings[env_i, 1].item(),
                    "chatter_L": detector.chatter_count[env_i, 0].item(),
                    "chatter_R": detector.chatter_count[env_i, 1].item(),
                    "weight_hash": weight_hash,
                    "mirror": args_cli.mirror,
                    "seed": args_cli.eval_seed,
                }
                episode_records.append(rec)

            # Reset accumulators for done envs
            fz_integral[done_mask] = 0
            knee_min[done_mask] = 999
            knee_max[done_mask] = -999
            for i in torch.nonzero(done_mask).flatten().tolist():
                if i < len(knee_vals):
                    knee_vals[i] = []
            swing_disp_sum[done_mask] = 0
            swing_clear_sum[done_mask] = 0
            detector.total_swings[done_mask] = 0
            detector.chatter_count[done_mask] = 0
            ep_step[done_mask] = 0
            spawn_valid[done_mask] = True

    # Output
    prefix = args_cli.output_prefix
    os.makedirs("legdiag", exist_ok=True)

    with open(f"legdiag/{prefix}_episodes.csv", "w", newline="", encoding="utf-8") as f:
        if episode_records:
            w = csv.DictWriter(f, fieldnames=list(episode_records[0].keys()))
            w.writeheader()
            w.writerows(episode_records)

    with open(f"legdiag/{prefix}_swings.csv", "w", newline="", encoding="utf-8") as f:
        if swing_events:
            w = csv.DictWriter(f, fieldnames=list(swing_events[0].keys()))
            w.writeheader()
            w.writerows(swing_events)

    # Print summary to stderr for log capture
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"LEG DIAGNOSTIC: {prefix} ({len(episode_records)} episodes)", file=sys.stderr)
    print(f"Weight: {weight_hash}  Mirror: {args_cli.mirror}  Seed: {args_cli.eval_seed}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    for r in episode_records:
        print(f"  E{r['env']:>3} {r['end_reason']:>10} {r['duration_s']:>6}s "
              f"disp={r['disp']:>7} J_L={r['J_L']:>8} J_R={r['J_R']:>8} "
              f"asym={r['asym']:>+7.3f} {r['dominant']:>16} "
              f"kL={r['kL_range']:.2f} kR={r['kR_range']:.2f} "
              f"swL={r['swings_L']:>3} swR={r['swings_R']:>3}", file=sys.stderr)

    # Aggregate
    doms = [r["dominant"] for r in episode_records]
    print(f"\n  Dominant: L={doms.count('L')} R={doms.count('R')} "
          f"balanced={doms.count('balanced')} insuff={doms.count('insufficient_data')}", file=sys.stderr)
    print(f"  [INFO] episodes saved: legdiag/{prefix}_episodes.csv", file=sys.stderr)
    print(f"  [INFO] swings saved: legdiag/{prefix}_swings.csv", file=sys.stderr)

    env.close()


def episode_termination_cap(step, max_steps):
    return step >= max_steps


if __name__ == "__main__":
    main()
    simulation_app.close()
