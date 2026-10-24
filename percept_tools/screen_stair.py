"""Frozen-model stair screening: passage determination per DESIGN.md.

STAIR_GEOMETRY + STAIR_SPAWN_ROLE env vars select stage geometry and role.
Passage (up): root AND both feet past the last up edge; each foot had a
debounced contact (>=5 frames) while past that edge; 0.5 s confirmation
window with forward motion and no termination. Down: analogous vs the last
down edge +0.6 m. Timeout without passage or any true failure = fail.
"""
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
    contact = unw.scene.sensors["contact_forces"]
    feet_ids = contact.find_bodies("ankle_pitch_.*")[0]
    feet_body = robot.find_bodies("ankle_pitch_.*")[0]
    tm = unw.termination_manager
    manifest = env_cfg._stair_manifest
    print("[SC_ENV_OK]", json.dumps({k: manifest[k] for k in
          ("step_height_m", "num_steps", "tread_m")}), flush=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(out), device=args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    runner.alg.actor_critic.load_state_dict(ckpt["model_state_dict"], strict=True)
    runner.alg.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    policy = runner.get_inference_policy(device=unw.device)
    encoder = runner.get_inference_encoder(device=unw.device)
    print("[SC_MODEL_OK]", flush=True)

    dt = unw.step_dt
    max_steps = int(round(args.episode_s / dt))
    # world-space passage targets from manifest (origin local x = 1.0)
    def to_world(x_local):
        return float(unw.scene.env_origins[0][0]) + (x_local - 1.0)
    up_role = args.role == "bottom"
    last_edge_local = ([e for e in manifest["edges"] if e["dir"] == "up"][-1]["x_local"]
                       if up_role else
                       [e for e in manifest["edges"] if e["dir"] == "down"][-1]["x_local"])
    goal_world = to_world(last_edge_local + (0.0 if up_role else 0.0) +
                          (0.6 if not up_role else 0.0))
    if up_role:  # up goal: past last up edge into top platform by 0.1 m
        goal_world = to_world(manifest["goals"]["up_goal_local_x"])

    rows = []
    need_reset = False   # 2026-10-24 fix: success-break leaves the env
    # mid-episode (no auto-reset); without a forced reset the next episode
    # starts already past the goal -> instant false-positive success chain.
    for ep in range(args.episodes):
        if need_reset:
            # must run under inference_mode: the env's state tensors were
            # created inside the episode loop's inference context, and
            # out-of-mode inplace updates on them raise RuntimeError
            with torch.inference_mode():
                unw._reset_idx(torch.tensor([0], device=unw.device))
            need_reset = False
        obs_dict = env.get_observations()
        success = False
        fell = timeout = False
        confirm_start = None
        contact_deb = [0, 0]
        contacted_past = [False, False]
        x0 = None
        x_last_pre = None
        with torch.inference_mode():
            for step in range(max_steps):
                obs = obs_dict["policy"]
                hist = obs_dict["obsHistory"].flatten(start_dim=1)
                cm = obs_dict["commands"]
                actions = policy(torch.cat((encoder(hist), obs, cm), dim=-1))
                if x0 is None:
                    x0 = float(robot.data.root_pos_w[0, 0])
                    e0 = float(unw.scene.env_origins[0][0])
                    lo, hi = ((e0 - 0.35, e0 + 0.35) if up_role else
                              (e0 + 2.17, e0 + 2.87))
                    assert lo < x0 < hi, (
                        f"[SC_SPAWN_OFFZONE] role={args.role} x0={x0:.3f} "
                        f"e0={e0:.3f} expect=({lo:.2f},{hi:.2f})")
                root_x = float(robot.data.root_pos_w[0, 0])
                x_last_pre = root_x
                feet_x = robot.data.body_pos_w[0, feet_body, 0]
                fz = contact.data.net_forces_w_history[:, -1][:, feet_ids].norm(dim=-1)[0]
                past = root_x > goal_world and bool((feet_x > goal_world).all())
                for i in (0, 1):
                    if float(fz[i]) >= 1.0:
                        contact_deb[i] += 1
                        if past and contact_deb[i] >= 5:
                            contacted_past[i] = True
                    else:
                        contact_deb[i] = 0
                if past and all(contacted_past):
                    if confirm_start is None:
                        confirm_start = step
                    if step - confirm_start >= int(0.5 / dt):
                        vx = float(robot.data.root_lin_vel_b[0, 0])
                        if vx > 0.05:
                            success = True
                            need_reset = True
                            break
                else:
                    confirm_start = None
                obs_dict, _, dones, _ = env.step(actions)
                if dones[0]:
                    t_out = bool(tm.get_term("time_out")[0])
                    timeout, fell = t_out, not t_out
                    break
        rows.append({"ep": ep, "role": args.role, "success": success,
                     "fell": fell, "timeout": timeout,
                     "steps": step + 1,
                     "x_pre_reset": (x_last_pre - x0) if x_last_pre is not None else 0.0,
                     "vx": args.vx})
        print(f"[SC_EP] {ep} success={success} fell={fell} timeout={timeout} "
              f"steps={step+1}", flush=True)

    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {"stage_geometry": {k: manifest[k] for k in
                                  ("step_height_m", "num_steps", "tread_m")},
               "role": args.role, "episodes": len(rows),
               "pass_rate": sum(r["success"] for r in rows) / len(rows),
               "falls": sum(r["fell"] for r in rows),
               "timeout_no_pass": sum(r["timeout"] and not r["success"] for r in rows),
               "goal_world_local": round(goal_world - float(unw.scene.env_origins[0][0]) + 1.0, 3),
               "checkpoint": str(Path(args.checkpoint).resolve())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[SC_SUMMARY]", json.dumps(summary), flush=True)
    env.close()
    print("[SC_DONE]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Limx-SFYG-TRON2A-Percept-Stair-Play-v0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--role", choices=("bottom", "top"), default="bottom")
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--episodes", type=int, default=10)
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
        print("[SC_MAIN_FAIL]", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        import os
        os._exit(1)
    finally:
        print("[SC_APP_CLOSE_BEGIN] wait_for_replicator=False", flush=True)
        app.app.close(wait_for_replicator=False)
        print("[SC_APP_CLOSE_OK]", flush=True)
