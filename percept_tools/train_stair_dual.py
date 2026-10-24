"""S1 dual-arm trainer (DESIGN v1.1): A=heights-zeroed blind vs B=perception.

Mixed grid 6 stairs/2 flat/2 rough per 10-cell cycle (typed cells, terrain_
sequence allocation, actual env counts logged). Cell-aware role spawn
(75% bottom / 25% top on stair cells; origin spawn on flat/rough). Chunks of
50 updates; per chunk: checkpoint (dual optimizers + parent 401 + arm label)
+ utilization monitor (heights norm, zero/random perturbation action deltas).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def build_type_manifests():
    import stair_terrain as st
    mans = {}
    for h in (0.04, 0.06, 0.08):
        t = st.StraightStairsCfg()
        t.step_height_range = (h, h)
        t.num_steps_range = (3, 3)
        t.tread_range = (0.34, 0.34)
        st.build_stair_meshes(h, 3, 0.34, t)
        mans[f"stairs{int(h*100):02d}"] = t._manifest
    return mans


def make_cell_aware_spawn(mans):
    import math
    import torch

    def _spawn(env, env_ids, manifests):
        robot = env.scene["robot"]
        dev = robot.device
        if not hasattr(env, "_dual_off"):
            env._dual_off = float(robot.data.default_root_state[0, 2])
        off = env._dual_off
        eorig = env.scene.env_origins
        # unique cell anchors -> cell type via generator's per-cell key list
        gen = env.scene.terrain.terrain_generator
        types = list(gen.sub_terrain_types)
        anchors = {}
        for i in range(eorig.shape[0]):
            key = (round(float(eorig[i, 0]), 3), round(float(eorg_y := eorig[i, 1]), 3))
            anchors.setdefault(key, []).append(i)
        anchor_list = sorted(anchors.keys())
        type_of = {}
        for ai, key in enumerate(anchor_list):
            type_of[key] = types[ai] if ai < len(types) else "flat"
        env._dual_type_counts = {}
        ids = env_ids.tolist() if env_ids is not None else list(range(eorig.shape[0]))
        for i in ids:
            key = (round(float(eorig[i, 0]), 3), round(float(eorig[i, 1]), 3))
            ctype = type_of.get(key, "flat")
            env._dual_type_counts[ctype] = env._dual_type_counts.get(ctype, 0) + 1
            man = manifests.get(ctype)
            if man is not None:  # stair cell: roled spawn
                take_bottom = bool(torch.rand(1, device=dev).item() < 0.75)
                zone = man["spawn_zones"]["bottom" if take_bottom else "top"]
                xl = float(torch.empty(1).uniform_(*zone["x_local"]).item())
                x = float(eorig[i, 0]) + (xl - 1.0)
                gz = zone["ground_z"]
            else:                # flat/rough cell: origin spawn
                x = float(eorig[i, 0]) + float(torch.empty(1).uniform_(-0.10, 0.10).item())
                gz = 0.0
            y = float(eorig[i, 1]) + float(torch.empty(1).uniform_(-0.10, 0.10).item())
            z = gz + off + 0.01
            yaw = float(torch.empty(1).uniform_(-0.05, 0.05).item())
            robot.write_root_pose_to_sim(
                torch.tensor([[x, y, z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
                             dtype=torch.float, device=dev),
                torch.tensor([i], dtype=torch.long, device=dev))
            robot.write_root_velocity_to_sim(
                torch.zeros((1, 6), dtype=torch.float, device=dev),
                torch.tensor([i], dtype=torch.long, device=dev))
    return _spawn


def main(args, simulation_app):
    import torch
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.managers import EventTermCfg
    import bipedal_locomotion  # noqa: F401
    from rsl_rl.runner import OnPolicyRunner
    import stair_terrain as st

    out = Path(args.output_dir)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = 50          # per-chunk
    agent_cfg.algorithm.learning_rate = 1.0e-5
    agent_cfg.algorithm.est_learning_rate = 1.0e-5
    agent_cfg.algorithm.desired_kl = None
    agent_cfg.save_interval = 500          # saves handled by chunk loop
    if args.track_w is not None:
        env_cfg.rewards.track_lin_vel_xy.weight = args.track_w
        print(f"[DUAL_TRACK_W] track_lin_vel_xy.weight={args.track_w}", flush=True)
    if args.still_w:  # phase-2 knob, default OFF per staging decision
        import torch as _t
        def _still_penalty(env, command_name, cruise_frac):
            cmd = env.command_manager.get_command(command_name)
            vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
            thr = cruise_frac * cmd[:, 0].abs().clamp(min=0.1)
            stuck = (vx.abs() < thr) & (cmd[:, 0].abs() > 0.1)
            return stuck.float()
        from isaaclab.managers import RewardTermCfg as _RW
        env_cfg.rewards.still_penalty = _RW(
            func=_still_penalty, weight=-abs(args.still_w),
            params={"command_name": "base_velocity", "cruise_frac": args.still_thr})
        print(f"[DUAL_STILL_PENALTY] w={-abs(args.still_w)} thr_frac={args.still_thr}", flush=True)

    # mixed grid + cell-aware spawn replace the task cfg's own wiring
    env_cfg.scene.terrain = st.make_dualarm_terrain()
    mans = build_type_manifests()
    env_cfg.events.reset_robot_base = EventTermCfg(
        func=make_cell_aware_spawn(mans), mode="reset", params={"manifests": mans})

    print("[DUAL_ENV_BEGIN]", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    unw = env.unwrapped
    print("[DUAL_ENV_OK]", flush=True)

    log_dir = out / "train"; log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    runner.alg.actor_critic.load_state_dict(ckpt["model_state_dict"], strict=True)
    runner.alg.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    if "optimizer_state_dict" in ckpt:
        try:
            runner.alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            pass
    print("[DUAL_LOAD_OK]", f"parent_iter={ckpt.get('iter')}", flush=True)

    # ---- arm A: zero heights in policy[42:163] / critic[45:166] observations
    obs_dict_holder = [None]
    if args.arm == "A":
        def _zero(d):
            for key, lo, hi in (("policy", 42, 163), ("critic", 45, 166)):
                if key in d and torch.is_tensor(d[key]):
                    d[key] = d[key].clone()
                    d[key][..., lo:hi] = 0.0
            return d
        orig_obs = env.get_observations
        def zeroed_obs(*a, **k):
            return _zero(orig_obs(*a, **k))
        env.get_observations = zeroed_obs
        orig_step = env.step
        def zeroed_step(actions):
            r = orig_step(actions)
            obs_dict_holder[0] = _zero(r[0])
            return (obs_dict_holder[0], r[1], r[2], r[3])
        env.step = zeroed_step
        print("[DUAL_ARM_A] heights zeroed on get_observations AND step returns", flush=True)

    policy = runner.get_inference_policy(device=unw.device)
    encoder = runner.get_inference_encoder(device=unw.device)

    def monitor(tag):
        with torch.no_grad():
            d = env.get_observations()
            obs, hist, cm = d["policy"], d["obsHistory"].flatten(start_dim=1), d["commands"]
            base = torch.cat((encoder(hist), obs, cm), dim=-1)
            a0 = policy(base)
            z = obs.clone(); z[..., 42:163] = 0.0
            a1 = policy(torch.cat((encoder(hist), z, cm), dim=-1))
            r = obs.clone(); r[..., 42:163] = torch.rand_like(r[..., 42:163]) * 0.2 - 0.1
            a2 = policy(torch.cat((encoder(hist), r, cm), dim=-1))
            d_zero = (a1 - a0).abs().amax(dim=-1)
            d_rand = (a2 - a0).abs().amax(dim=-1)
            W = runner.alg.actor_critic.state_dict()["actor.0.weight"]
            C = runner.alg.actor_critic.state_dict()["critic.0.weight"]
            m = {
                "tag": tag, "iteration": int(runner.current_learning_iteration),
                "heights_norm_actor": float(W[:, 45:166].norm()),
                "heights_norm_critic": float(C[:, 45:166].norm()),
                "perturb_zero_max_P50": float(d_zero.median()),
                "perturb_zero_max_P95": float(d_zero.quantile(0.95)),
                "perturb_rand_max_P50": float(d_rand.median()),
                "perturb_rand_max_P95": float(d_rand.quantile(0.95)),
            }
        return m

    mons = [monitor("pre")]
    type_counts = getattr(unw, "_dual_type_counts", None)
    if type_counts is None:
        type_counts = getattr(env.unwrapped, "_dual_type_counts", {})
    print("[DUAL_TYPE_COUNTS]", json.dumps(type_counts), flush=True)

    chunks = args.iters // 50
    for ci in range(chunks):
        runner.learn(num_learning_iterations=50, init_at_random_ep_len=(ci == 0))
        it = int(runner.current_learning_iteration)
        params_finite = all(torch.isfinite(p).all() for p in
                            runner.alg.actor_critic.parameters())
        mon = monitor(f"chunk{ci+1}")
        mons.append(mon)
        torch.save({
            "model_state_dict": runner.alg.actor_critic.state_dict(),
            "encoder_state_dict": runner.alg.encoder.state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "encoder_optimizer_state_dict": (
                runner.alg.extra_optimizer.state_dict()
                if getattr(runner.alg, "extra_optimizer", None) is not None else None),
            "iter": it,
            "parent_checkpoint_updates": 401,
            "adaptation_updates": (ci + 1) * 50,
            "arm": args.arm, "lr": {"ppo": 1e-5, "encoder": 1e-5},
            "stage": "S1-dual", "params_finite": params_finite,
        }, out / f"model_{args.arm}_{(ci+1)*50:03d}.pt")
        print(f"[DUAL_CHUNK_OK] {ci+1}/{chunks} it={it} finite={params_finite} "
              f"mon={json.dumps(mon)}", flush=True)
        if not params_finite:
            print("[DUAL_NONFINITE_ABORT]", flush=True)
            break

    (out / f"monitors_{args.arm}.json").write_text(json.dumps(mons, indent=1))
    (out / "run_meta.json").write_text(json.dumps({
        "arm": args.arm, "chunks": chunks, "iters": args.iters,
        "num_envs": args.num_envs, "type_counts": type_counts,
        "checkpoint": str(Path(args.checkpoint).resolve())}, indent=1))
    env.close()
    print("[DUAL_DONE]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Limx-SFYG-TRON2A-Percept-Stair-v0")
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--track_w", type=float, default=None,
                        help="override track_lin_vel_xy weight (this round: 2.5)")
    parser.add_argument("--still_w", type=float, default=0.0,
                        help="still-penalty weight (phase 2, default OFF)")
    parser.add_argument("--still_thr", type=float, default=0.175,
                        help="stuck threshold as fraction of cruise speed (0.15-0.20)")
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
        print("[DUAL_MAIN_FAIL]", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        import os
        os._exit(1)
    finally:
        print("[DUAL_APP_CLOSE_BEGIN] wait_for_replicator=False", flush=True)
        app.app.close(wait_for_replicator=False)
        print("[DUAL_APP_CLOSE_OK]", flush=True)
