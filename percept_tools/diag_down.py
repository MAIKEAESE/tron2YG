"""Down-stall numeric verification (replaces video on this server: its
Replicator/Vulkan stack segfaults). Top-spawn 4 cm episodes, sampling root
x/z/vx every N steps. The z(t) profile identifies WHICH tread the robot
stands on (top 0.12 / step2 0.08 / step1 0.04 / ground 0) and for how long —
a rigorous form of the 'sustained mid-stair stall' check."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path


def main(args, simulation_app):
    import torch
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import bipedal_locomotion  # noqa: F401
    from rsl_rl.runner import OnPolicyRunner

    out = Path(args.output_dir)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    vel = env_cfg.commands.base_velocity
    vel.heading_command = False
    vel.ranges.lin_vel_x = (args.vx, args.vx)
    vel.ranges.lin_vel_y = (0.0, 0.0)
    vel.ranges.ang_vel_z = (0.0, 0.0)
    vel.resampling_time_range = (1000000.0, 1000000.0)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    unw = env.unwrapped
    robot = unw.scene["robot"]
    tm = unw.termination_manager
    manifest = env_cfg._stair_manifest
    print("[DG_ENV_OK]", json.dumps({k: manifest[k] for k in
          ("step_height_m", "num_steps", "tread_m")}), flush=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(out), device=args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    runner.alg.actor_critic.load_state_dict(ckpt["model_state_dict"], strict=True)
    runner.alg.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    policy = runner.get_inference_policy(device=unw.device)
    encoder = runner.get_inference_encoder(device=unw.device)

    dt = unw.step_dt
    max_steps = int(round(args.episode_s / dt))
    # reference-audit sensors: same formulas as fix3 reward terms
    scanner = unw.scene.sensors["height_scanner"]
    try:
        bh_cfg = tm.cfg.terminations if False else unw.reward_manager.get_term_cfg("base_height")
        bh_target = float(bh_cfg.params.get("target_height", float("nan")))
        bh_weight = float(bh_cfg.weight)
    except Exception:
        bh_target = float("nan")
        bh_weight = float("nan")
    print(f"[DG_BASEHEIGHT] target={bh_target} weight={bh_weight}", flush=True)
    feet_body = robot.find_bodies("ankle_pitch_.*")[0]
    treads = {"top(0.12)": 0.12, "step2(0.08)": 0.08, "step1(0.04)": 0.04,
              "ground(0)": 0.0}

    def scan_mean_z():
        hits = scanner.data.ray_hits_w[..., 2]
        finite = torch.isfinite(hits)
        mz = torch.where(finite, hits, torch.zeros_like(hits)).sum(1) \
            / finite.sum(1).clamp(min=1)
        return float(mz[0])

    def foot_ground():
        hits = scanner.data.ray_hits_w
        feet = robot.data.body_pos_w[:, feet_body]
        d2 = (feet[..., :2].unsqueeze(2) - hits[:, None, :, :2]).square().sum(-1)
        minimum, index = d2.min(-1)
        g = hits[..., 2].gather(1, index)
        return [round(float(g[0, i]), 3) for i in range(g.shape[1])]

    def tread_of(z):
        best, bd = None, 9.9
        for name, zv in treads.items():
            d = abs(z - zv)
            if d < bd:
                best, bd = name, d
        return best if bd < 0.02 else f"z={z:.3f}"

    rows = []
    summaries = []
    for ep in range(args.episodes):
        obs_dict = env.get_observations()
        x0 = z0 = None
        samples = []
        fell = False
        with torch.inference_mode():
            for step in range(max_steps):
                obs = obs_dict["policy"]
                hist = obs_dict["obsHistory"].flatten(start_dim=1)
                cm = obs_dict["commands"]
                actions = policy(torch.cat((encoder(hist), obs, cm), dim=-1))
                x = float(robot.data.root_pos_w[0, 0])
                z = float(robot.data.root_pos_w[0, 2]) - 0.85   # root minus stance height
                vx = float(robot.data.root_lin_vel_b[0, 0])
                if x0 is None:
                    x0, z0 = x, z
                if step % 25 == 0:
                    samples.append({"step": step, "t": round(step * dt, 2),
                                    "dx": round(x - x0, 3), "z": round(z, 3),
                                    "vx": round(vx, 3), "tread": tread_of(z),
                                    "scan_mz": round(scan_mean_z(), 3),
                                    "base_tgt": round(bh_target + scan_mean_z(), 3)
                                    if bh_target == bh_target else None,
                                    "foot_gz": foot_ground()})
                obs_dict, _, dones, _ = env.step(actions)
                if dones[0]:
                    fell = not bool(tm.get_term("time_out")[0])
                    break
        rows.extend({"ep": ep, **s} for s in samples)
        # stall analysis: windows of 3 s (150 steps) from samples
        win_s = 3.0
        lows = []
        for i in range(len(samples)):
            t_i = samples[i]["t"]
            j = i
            while j + 1 < len(samples) and samples[j + 1]["t"] <= t_i + win_s:
                j += 1
            prog = samples[j]["dx"] - samples[i]["dx"]
            if t_i >= 6.0:                       # after grace
                lows.append((t_i, round(prog, 3)))
        sustained = [t for t, p in lows if p < 0.15]
        z_last = samples[-2]["tread"] if len(samples) > 1 else "?"
        summaries.append({"ep": ep, "fell": fell, "end_dx": samples[-1]["dx"],
                          "end_tread": samples[-1]["tread"], "final_steps": samples[-1]["step"],
                          "sustained_stall_windows": len(sustained),
                          "first_sustained_t": sustained[0] if sustained else None})
        print(f"[DG_EP] {ep} fell={fell} end_dx={samples[-1]['dx']} "
              f"end_tread={samples[-1]['tread']} stall_windows={len(sustained)}", flush=True)
        print("[DG_TRACE]", json.dumps(samples), flush=True)

    with open(out / "trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    print("[DG_SUMMARY]", json.dumps(summaries), flush=True)
    env.close()
    print("[DG_DONE]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Limx-SFYG-TRON2A-Percept-Stair-Play-v0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode_s", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    from isaaclab.app import AppLauncher  # noqa: E402
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    args.output_dir = out
    repo = Path("/root/official_tron2_lab")
    sys.path.insert(0, str(repo / "rsl_rl"))
    sys.path.insert(0, str(repo / "exts" / "bipedal_locomotion"))
    sys.path.insert(0, "/root/autodl-tmp/terrain_kit")
    args.enable_cameras = False
    app = AppLauncher(args)
    try:
        main(args, app.app)
    except Exception:
        print("[DG_MAIN_FAIL]", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        import os
        os._exit(1)
    finally:
        app.app.close(wait_for_replicator=False)
