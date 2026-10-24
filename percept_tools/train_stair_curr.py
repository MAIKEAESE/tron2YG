"""S1 direction-pool curriculum trainer (DESIGN v1.3.2, 2026-10-24 user spec).

ONE shared policy. Stair envs split into FIXED direction pools with
independent level ladders (rows 1-3, pool floor at row 1):
  up-pool   -> always spawns bottom, goal = TOP platform (up_goal)
  down-pool -> always spawns top,    goal = BOTTOM platform (down_goal)
Verdict LOCKS at passage: a pass promotes even if the env falls afterwards
(failures beyond the pool's own goal must not veto promotion — the up pool
is never judged on the descent). Any non-passage outcome demotes.
Pinned replay pools (flat/rough, never in curriculum) provide regression
samples. Round 1 keeps single-pass promotion / demote-on-fail thresholds
(streak recorded per env as telemetry only; loosening deferred).
Full-episode per-pool reward/length/displacement statistics — accumulated
across the WHOLE episode and finalized only at true episode end, immune to
rollout-window truncation.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

POOL_NAMES = ("flat", "rough", "up", "down")


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


def main(args, simulation_app):
    import math
    import torch
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.managers import EventTermCfg
    import bipedal_locomotion  # noqa: F401
    from rsl_rl.runner import OnPolicyRunner
    import stair_terrain as st
    import curriculum_core as cc

    out = Path(args.output_dir)
    N = args.num_envs
    assert args.pin_flat + args.pin_rough + args.pool_up + args.pool_down == N, \
        f"pool sizes {args.pin_flat}+{args.pin_rough}+{args.pool_up}+{args.pool_down} != num_envs {N}"
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=N)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = 50            # per chunk
    agent_cfg.algorithm.learning_rate = 1.0e-5
    agent_cfg.algorithm.est_learning_rate = 1.0e-5
    agent_cfg.algorithm.desired_kl = None
    agent_cfg.algorithm.save_interval = 500  # saves handled by chunk loop
    if args.track_w is not None:
        env_cfg.rewards.track_lin_vel_xy.weight = args.track_w
        print(f"[CURR_TRACK_W] track_lin_vel_xy.weight={args.track_w}", flush=True)

    step_heights = (0.04, 0.06, 0.08)
    num_rows, num_cols = 1 + len(step_heights), 4
    env_cfg.scene.terrain = st.make_rowgrid_terrain(step_heights=step_heights,
                                                    num_cols=num_cols)
    mans = build_type_manifests()
    seq = st.rowgrid_sequence(step_heights, num_cols)
    kind_of_key = {"flat": 0, "rough": 1, "stairs04": 2, "stairs06": 3, "stairs08": 4}
    stair_key_of_kind = {2: "stairs04", 3: "stairs06", 4: "stairs08"}

    def make_pool_spawn():
        def _spawn(env, env_ids, manifests):
            import torch
            robot = env.scene["robot"]
            dev = robot.device
            if not hasattr(env, "_stair_default_root_offset"):
                env._stair_default_root_offset = float(robot.data.default_root_state[0, 2])
            off = env._stair_default_root_offset
            eorig = env.scene.env_origins
            if not hasattr(env, "_curr_pool"):         # lazy init (startup reset)
                n = eorig.shape[0]
                env._curr_pool = torch.zeros(n, dtype=torch.long, device=dev)
                env._curr_kind = torch.zeros(n, dtype=torch.long, device=dev)
                env._curr_goal_x = torch.full((n,), float("inf"), device=dev)
                env._curr_stairs = torch.zeros(n, dtype=torch.bool, device=dev)
                env._curr_x0 = torch.zeros(n, device=dev)
                env._role_tally = {}
            ids = env_ids.tolist() if env_ids is not None else list(range(eorig.shape[0]))
            z_samples = []
            for i in ids:
                pool = int(env._curr_pool[i])          # 0 flat / 1 rough / 2 up / 3 down
                if pool >= 2:                          # stair cell: FIXED role per pool
                    man = manifests[stair_key_of_kind[int(env._curr_kind[i])]]
                    take_bottom = pool == 2           # up-pool: bottom; down-pool: top
                    zone = man["spawn_zones"]["bottom" if take_bottom else "top"]
                    xl = float(torch.empty(1).uniform_(*zone["x_local"]).item())
                    x = float(eorig[i, 0]) + (xl - 1.0)
                    gz = zone["ground_z"]
                    goal_local = (man["goals"]["up_goal_local_x"] if take_bottom
                                  else man["goals"]["down_goal_local_x"])
                    env._curr_goal_x[i] = float(eorig[i, 0]) + (goal_local - 1.0)
                    env._curr_stairs[i] = True
                    env._role_tally[POOL_NAMES[pool]] = \
                        env._role_tally.get(POOL_NAMES[pool], 0) + 1
                else:                                  # flat/rough: origin spawn
                    x = float(eorig[i, 0]) + float(torch.empty(1).uniform_(-0.10, 0.10).item())
                    gz = 0.0
                    env._curr_goal_x[i] = float("inf")   # unreachable -> past False
                    env._curr_stairs[i] = False
                    env._role_tally[POOL_NAMES[pool]] = \
                        env._role_tally.get(POOL_NAMES[pool], 0) + 1
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
                env._curr_x0[i] = x
                if len(z_samples) < 6:
                    z_samples.append((i, POOL_NAMES[pool], round(z, 3), round(gz, 3)))
            if z_samples and not getattr(env, "_spawn_z_printed", False):
                env._spawn_z_printed = True
                print(f"[CURR_SPAWN_Z] off={off:.3f} samples={z_samples}", flush=True)
        return _spawn

    env_cfg.events.reset_robot_base = EventTermCfg(
        func=make_pool_spawn(), mode="reset", params={"manifests": mans})

    print("[CURR_ENV_BEGIN]", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    unw = env.unwrapped
    print("[CURR_ENV_OK]", flush=True)

    # ---- grid verification: row-major sequence <-> terrain_origins grid ----
    terrain = unw.scene.terrain
    dev = unw.device
    gen_types = list(terrain.terrain_generator.sub_terrain_types)
    assert len(gen_types) == num_rows * num_cols, \
        f"cell count {len(gen_types)} != {num_rows * num_cols}"
    mismatch = [(r, c, gen_types[r * num_cols + c], seq[r * num_cols + c])
                for r in range(num_rows) for c in range(num_cols)
                if gen_types[r * num_cols + c] != seq[r * num_cols + c]]
    og = terrain.terrain_origins
    origins_grid = og if torch.is_tensor(og) else torch.tensor(
        og, dtype=torch.float, device=dev)
    drow_x = float(origins_grid[1, 0, 0] - origins_grid[0, 0, 0])   # rows -> +x
    dcol_y = float(origins_grid[0, 1, 1] - origins_grid[0, 0, 1])   # cols -> +y
    print(f"[CURR_GRID_VERIFY] mismatch={mismatch} drow_x={drow_x:.2f} dcol_y={dcol_y:.2f}",
          flush=True)
    assert not mismatch, f"generation order is not row-major: {mismatch}"
    assert abs(drow_x - 8.0) < 1e-3 and abs(dcol_y - 8.0) < 1e-3, (drow_x, dcol_y)

    log_dir = out / "train"
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    runner.alg.actor_critic.load_state_dict(ckpt["model_state_dict"], strict=True)
    runner.alg.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    if "optimizer_state_dict" in ckpt:
        try:
            runner.alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            pass
    print("[CURR_LOAD_OK]", f"parent_iter={ckpt.get('iter')}", flush=True)

    # ---- arm A: heights zeroing (optional control arm) --------------------
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
        orig_step_inner = env.step
        def zeroed_step(actions):
            r = orig_step_inner(actions)
            return (_zero(r[0]), r[1], r[2], r[3])
        env.step = zeroed_step
        print("[CURR_ARM_A] heights zeroed on get_observations AND step returns", flush=True)

    # ---- pool assignment + curriculum tracker hooks -------------------------
    robot = unw.scene["robot"]
    contact = unw.scene.sensors["contact_forces"]
    feet_ids = contact.find_bodies("ankle_pitch_.*")[0]
    feet_body = robot.find_bodies("ankle_pitch_.*")[0]
    tm = unw.termination_manager
    dt = unw.step_dt
    tracker = cc.EpisodeTracker(N, dev, confirm_frames=int(round(0.5 / dt)))

    F, R = args.pin_flat, args.pin_rough
    U, D = args.pool_up, args.pool_down
    pool = torch.zeros(N, dtype=torch.long, device=dev)
    pool[F:F + R] = 1
    pool[F + R:F + R + U] = 2
    pool[F + R + U:] = 3
    unw._curr_pool = pool
    unw._streak = torch.zeros(N, dtype=torch.long, device=dev)
    unw._last_disp = torch.zeros(N, device=dev)

    pool_ids = {"flat": torch.arange(0, F, device=dev),
                "rough": torch.arange(F, F + R, device=dev),
                "up": torch.arange(F + R, F + R + U, device=dev),
                "down": torch.arange(F + R + U, N, device=dev)}
    curr_ids = torch.cat([pool_ids["up"], pool_ids["down"]])

    stats = {"episodes": 0, "up_pool_up": 0, "up_pool_down": 0, "up_pool_top": 0,
             "down_pool_up": 0, "down_pool_down": 0, "down_pool_top": 0}
    # full-episode per-pool statistics (whole-episode sums, finalized at end)
    pool_ep = {p: {"n": 0, "rew": 0.0, "length": 0, "disp": 0.0}
               for p in POOL_NAMES}
    ep_rew = torch.zeros(N, device=dev)
    ep_len = torch.zeros(N, dtype=torch.long, device=dev)

    def refresh_kinds():
        keys = [gen_types[int(l) * num_cols + int(c)]
                for l, c in zip(terrain.terrain_levels.tolist(),
                                terrain.terrain_types.tolist())]
        unw._curr_kind = torch.tensor([kind_of_key[k] for k in keys],
                                      dtype=torch.long, device=dev)

    def frame_signals(idx):
        root_x = robot.data.root_pos_w[idx, 0]
        feet_x = robot.data.body_pos_w[:, feet_body, 0][idx]
        fz = contact.data.net_forces_w_history[:, -1][:, feet_ids].norm(dim=-1)[idx]
        goal = unw._curr_goal_x[idx]
        past = (root_x > goal) & (feet_x.min(dim=-1).values > goal) & unw._curr_stairs[idx]
        vx = robot.data.root_lin_vel_b[idx, 0]
        return past, fz >= cc.CONTACT_FORCE_N, vx

    def apply_curriculum(idx):
        with torch.no_grad():
            past, fc, vx = frame_signals(idx)          # terminal frame accumulation
            tracker.update(idx, past, fc, vx)
            fell = ~tm.get_term("time_out")[idx]
            is_st = unw._curr_stairs[idx]
            disp = robot.data.root_pos_w[idx, 0] - unw._curr_x0[idx]
            unw._last_disp[idx] = disp                 # for full-episode stats
            level = terrain.terrain_levels[idx]
            up, down, top = cc.decide(tracker.passage[idx], fell, is_st, disp,
                                      level, num_rows - 1, min_level=1)
            is_up_pool = unw._curr_pool[idx] == 2
            is_down_pool = unw._curr_pool[idx] == 3
            in_pool = is_up_pool | is_down_pool
            up, down, top = up & in_pool, down & in_pool, top & in_pool
            # streak telemetry (round 1: recorded, NOT gating)
            success_now = in_pool & tracker.passage[idx]
            unw._streak[idx] = torch.where(
                in_pool,
                torch.where(success_now, unw._streak[idx] + 1,
                            torch.zeros_like(unw._streak[idx])),
                unw._streak[idx])
            _, new_org = cc.apply_levels(
                terrain.terrain_levels, terrain.terrain_types,
                origins_grid, idx, up, down, num_rows - 1, min_level=1)
            terrain.env_origins[idx] = new_org         # in-place: scene ref preserved
            stats["episodes"] += int(in_pool.sum())
            stats["up_pool_up"] += int((up & is_up_pool).sum())
            stats["up_pool_down"] += int((down & is_up_pool).sum())
            stats["up_pool_top"] += int((top & is_up_pool).sum())
            stats["down_pool_up"] += int((up & is_down_pool).sum())
            stats["down_pool_down"] += int((down & is_down_pool).sum())
            stats["down_pool_top"] += int((top & is_down_pool).sum())
            unw._streak[idx] = torch.where(
                up | down, torch.zeros_like(unw._streak[idx]), unw._streak[idx])
            refresh_kinds()

    orig_reset = unw._reset_idx
    def reset_hook(env_ids):
        idx = env_ids.to(dev) if torch.is_tensor(env_ids) else torch.tensor(env_ids, device=dev)
        if int(unw.episode_length_buf[idx].max().item()) > 0:
            apply_curriculum(idx)
        else:
            refresh_kinds()                            # startup/forced reset: spawn only
        orig_reset(env_ids)
        tracker.reset(idx)
    unw._reset_idx = reset_hook

    orig_step = env.step
    curr_fine = []                                     # (common_step, level_hist) samples

    def tracked_step(actions):
        nonlocal ep_rew, ep_len                    # += rebinds names -> locals otherwise
        r = orig_step(actions)
        with torch.no_grad():
            rewards, dones = r[1], r[2].bool()
            ep_rew += rewards                          # final step included
            ep_len += 1
            if dones.any():                            # finalize FULL episodes
                didx = dones.nonzero(as_tuple=False).squeeze(-1).to(dev)
                for p in POOL_NAMES:
                    m = unw._curr_pool[didx] == POOL_NAMES.index(p)
                    if m.any():
                        k = didx[m]
                        pool_ep[p]["n"] += int(k.numel())
                        pool_ep[p]["rew"] += float(ep_rew[k].sum())
                        pool_ep[p]["length"] += int(ep_len[k].sum())
                        pool_ep[p]["disp"] += float(unw._last_disp[k].sum())
                ep_rew[didx] = 0.0
                ep_len[didx] = 0
            alive = ~dones
            if alive.any():
                idx = alive.nonzero(as_tuple=False).squeeze(-1).to(dev)
                past, fc, vx = frame_signals(idx)
                tracker.update(idx, past, fc, vx)
            unw._cc_tick = getattr(unw, "_cc_tick", 0) + 1
            if unw._cc_tick % 250 == 0:
                hh = torch.bincount(terrain.terrain_levels[curr_ids],
                                    minlength=num_rows)
                curr_fine.append((int(unw.common_step_counter),
                                  [int(v) for v in hh.tolist()]))
        return r
    env.step = tracked_step

    # ---- initial assignment: pinned replay pools + direction pools --------
    lv, ty = terrain.terrain_levels, terrain.terrain_types
    lv[pool_ids["flat"]] = 0; ty[pool_ids["flat"]] = 0
    lv[pool_ids["rough"]] = 0; ty[pool_ids["rough"]] = 1
    cid = curr_ids.tolist()
    ty[cid] = torch.arange(len(cid), device=dev) % num_cols
    lv[cid] = torch.randint(1, num_rows, (len(cid),), device=dev)
    terrain.env_origins[:] = origins_grid[lv, ty]
    refresh_kinds()
    hist_u = torch.bincount(terrain.terrain_levels[pool_ids["up"]], minlength=num_rows)
    hist_d = torch.bincount(terrain.terrain_levels[pool_ids["down"]], minlength=num_rows)
    print(f"[CURR_INIT] pools flat={F} rough={R} up={U} down={D} "
          f"up_hist={hist_u.tolist()} down_hist={hist_d.tolist()}", flush=True)
    print("[CURR_KNOBS] direction pools v1.3.2: verdict locks at pool goal "
          "(up->TOP platform, down->BOTTOM platform); floor row 1; "
          "single-pass promote / demote-on-fail (streak telemetry only); "
          f"full-episode per-pool stats; track_w={args.track_w}", flush=True)
    with torch.no_grad():
        unw._reset_idx(torch.arange(N, device=dev))    # spawn everyone at assigned cells

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
            hu = torch.bincount(terrain.terrain_levels[pool_ids["up"]],
                                minlength=num_rows).tolist()
            hd = torch.bincount(terrain.terrain_levels[pool_ids["down"]],
                                minlength=num_rows).tolist()
            ep_view = {}
            for p in POOL_NAMES:
                n = pool_ep[p]["n"]
                ep_view[p] = {"episodes": n,
                              "mean_reward": round(pool_ep[p]["rew"] / n, 3) if n else None,
                              "mean_length": round(pool_ep[p]["length"] / n, 1) if n else None,
                              "mean_disp_m": round(pool_ep[p]["disp"] / n, 3) if n else None}
                for k in pool_ep[p]:
                    pool_ep[p][k] = 0 if isinstance(pool_ep[p][k], int) else 0.0
            stk_u = unw._streak[pool_ids["up"]].float()
            stk_d = unw._streak[pool_ids["down"]].float()
            m = {"tag": tag, "iteration": int(runner.current_learning_iteration),
                 "heights_norm_actor": float(W[:, 45:166].norm()),
                 "heights_norm_critic": float(C[:, 45:166].norm()),
                 "perturb_zero_max_P50": float(d_zero.median()),
                 "perturb_zero_max_P95": float(d_zero.quantile(0.95)),
                 "perturb_rand_max_P50": float(d_rand.median()),
                 "perturb_rand_max_P95": float(d_rand.quantile(0.95)),
                 "up_pool_hist": hu, "down_pool_hist": hd,
                 "curriculum_stats": dict(stats),
                 "pool_episodes": ep_view,
                 "streak_mean_up": round(float(stk_u.mean()), 2),
                 "streak_mean_down": round(float(stk_d.mean()), 2),
                 "role_tally": dict(getattr(unw, "_role_tally", {}))}
            try:                                       # native TB curriculum curves
                wr = getattr(runner, "writer", None)
                if wr is not None:
                    itn = int(runner.current_learning_iteration)
                    for name, h in (("up", hu), ("down", hd)):
                        tot = sum(h) or 1
                        dom = max(range(len(h)), key=lambda i: h[i])
                        wr.add_scalar(f"curriculum/{name}_dominant_level", dom, itn)
                        for i, v in enumerate(h):
                            wr.add_scalar(f"curriculum/{name}_share_row{i}", v / tot, itn)
                            wr.add_scalar(f"curriculum/{name}_envs_row{i}", v, itn)
                    for k_ in ("up_pool_up", "up_pool_down", "down_pool_up",
                               "down_pool_down", "episodes"):
                        wr.add_scalar(f"curriculum/{k_}", m["curriculum_stats"][k_], itn)
                    for p in POOL_NAMES:
                        v = m["pool_episodes"][p]
                        if v["mean_reward"] is not None:
                            wr.add_scalar(f"pool_ep/{p}_mean_reward", v["mean_reward"], itn)
                            wr.add_scalar(f"pool_ep/{p}_mean_length", v["mean_length"], itn)
                    for s_step, s_hist in curr_fine:
                        s_tot = sum(s_hist) or 1
                        s_dom = max(range(len(s_hist)), key=lambda i: s_hist[i])
                        wr.add_scalar("curriculum_fine/dominant_level", s_dom, s_step)
                        for i, v in enumerate(s_hist):
                            wr.add_scalar(f"curriculum_fine/share_row{i}",
                                          v / s_tot, s_step)
                    curr_fine.clear()
            except Exception:
                pass
        return m

    mons = [monitor("pre")]
    chunks = max(args.iters // 50, 1)
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
            "stage": "S1-pools-v1.3.2",
            "curriculum": {"up_hist": mon["up_pool_hist"],
                           "down_hist": mon["down_pool_hist"],
                           "stats": mon["curriculum_stats"],
                           "pool_episodes": mon["pool_episodes"],
                           "pools": {"flat": F, "rough": R, "up": U, "down": D}},
            "params_finite": params_finite,
        }, out / f"model_{args.arm}_{(ci+1)*50:03d}.pt")
        print(f"[CURR_CHUNK_OK] {ci+1}/{chunks} it={it} finite={params_finite} "
              f"up_hist={mon['up_pool_hist']} down_hist={mon['down_pool_hist']} "
              f"stats={json.dumps(mon['curriculum_stats'])}", flush=True)
        print(f"[CURR_MON] {json.dumps(mon)}", flush=True)
        if not params_finite:
            print("[CURR_NONFINITE_ABORT]", flush=True)
            break

    (out / f"monitors_{args.arm}.json").write_text(json.dumps(mons, indent=1))
    (out / "run_meta.json").write_text(json.dumps({
        "arm": args.arm, "chunks": chunks, "iters": args.iters,
        "num_envs": N, "track_w": args.track_w,
        "pools": {"flat": F, "rough": R, "up": U, "down": D},
        "design": "DESIGN.md v1.3.2 direction pools",
        "checkpoint": str(Path(args.checkpoint).resolve())}, indent=1))
    env.close()
    print("[CURR_DONE]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Limx-SFYG-TRON2A-Percept-Stair-v0")
    parser.add_argument("--arm", choices=("A", "B"), default="B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--track_w", type=float, default=2.5)
    parser.add_argument("--pin_flat", type=int, default=819)
    parser.add_argument("--pin_rough", type=int, default=819)
    parser.add_argument("--pool_up", type=int, default=1843)
    parser.add_argument("--pool_down", type=int, default=615)
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
        print("[CURR_MAIN_FAIL]", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        import os
        os._exit(1)
    finally:
        print("[CURR_APP_CLOSE_BEGIN] wait_for_replicator=False", flush=True)
        app.app.close(wait_for_replicator=False)
        print("[CURR_APP_CLOSE_OK]", flush=True)
