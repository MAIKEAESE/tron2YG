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
    if args.still_w:  # down-pool sustained-stall penalty (user spec v1.3.3)
        from isaaclab.managers import RewardTermCfg as _RW

        def _down_stall_penalty(env):
            flag = getattr(env, "_down_stall", None)
            if flag is None:
                import torch as _t
                flag = _t.zeros(env.num_envs, dtype=_t.bool, device=env.device)
            return flag.float()

        env_cfg.rewards.down_stall_penalty = _RW(
            func=_down_stall_penalty, weight=-abs(args.still_w))
        print(f"[CURR_STALL_PENALTY] down-pool ONLY; sustained "
              f"{cc.STALL_CONFIRM_WINDOWS} consecutive {cc.STALL_WINDOW_STEPS}-step "
              f"windows < {cc.STALL_PROG_M} m; grace {cc.STALL_GRACE_WINDOWS} "
              f"windows; w={-abs(args.still_w)}", flush=True)
    if args.fh_w:  # down-pool foothold guidance with CoM-stride binding (v1.3.4)
        from isaaclab.managers import RewardTermCfg as _RW2

        def _foothold_reward(env):
            import torch as _t
            fh = getattr(env, "_fh", None)
            if fh is None:
                return _t.zeros(env.num_envs, device=env.device)
            diag = getattr(env, "_fh_diag", None)
            if diag is None:
                diag = env._fh_diag = {"land": 0, "paid": 0, "sum": 0.0}
            if not hasattr(env, "_bonus_diag"):
                env._bonus_diag = {"paid": 0, "sum": 0.0}
            if not hasattr(env, "_fh_r_step"):
                import torch as _rt
                env._fh_r_step = _rt.zeros(env.num_envs, device=env.device)
                env._fh_tgt_step = _rt.full((env.num_envs,), -1,
                                            dtype=_rt.long, device=env.device)
            env._fh_r_step.fill_(0.0)
            env._fh_tgt_step.fill_(-1)
            robot = env.scene["robot"]
            contact = env.scene.sensors["contact_forces"]
            feet_ids = contact.find_bodies("ankle_pitch_.*")[0]
            feet_body = robot.find_bodies("ankle_pitch_.*")[0]
            fz = contact.data.net_forces_w_history[:, -1][:, feet_ids].norm(dim=-1)
            now = fz >= cc.CONTACT_FORCE_N
            landing = now & ~env._fh_contact
            env._fh_contact = now.clone()
            reward = _t.zeros(env.num_envs, device=env.device)
            for foot in (0, 1):
                m = landing[:, foot]
                if m.any():
                    idx = m.nonzero(as_tuple=False).squeeze(-1)
                    r, t_idx = fh.on_landing(
                        idx, foot,
                        robot.data.body_pos_w[idx, feet_body[foot], 0],
                        robot.data.root_pos_w[idx, 0])
                    reward[idx] += r
                    env._fh_r_step[idx] += r
                    env._fh_tgt_step[idx] = t_idx
                    diag["land"] += int(idx.numel())
                    diag["paid"] += int((r > 0.05).sum())
                    diag["sum"] += float(r.sum())
            # RewardManager multiplies by dt (0.02): without this divide, a
            # one-shot event reward is diluted 50x vs per-step terms and the
            # guide is ~1000x weaker than track_lin_vel at episode level
            return reward / env.step_dt

        env_cfg.rewards.down_foothold = _RW2(func=_foothold_reward,
                                             weight=float(args.fh_w))
        print(f"[CURR_FOOTHOLD] down-pool tread-center targets; reward = "
              f"precision({cc.FOOTHOLD_TOL_M}m tol) x CoM-stride("
              f"{cc.FOOTHOLD_STRIDE_REF_M}m ref) — farmed steps pay ~0; "
              f"w={args.fh_w}", flush=True)
        if args.bonus_w:  # one-shot first-step bounty, PARALLEL term (v1.3.6)
            def _first_step_bonus(env):
                import torch as _t
                fh = getattr(env, "_fh", None)
                rstep = getattr(env, "_fh_r_step", None)
                tstep = getattr(env, "_fh_tgt_step", None)
                if fh is None or rstep is None or tstep is None:
                    return _t.zeros(env.num_envs, device=env.device)
                elig = (rstep >= cc.BONUS_QUALITY) & (env._curr_pool == 3)
                bonus = _t.zeros(env.num_envs, device=env.device)
                if elig.any():
                    idx = elig.nonzero(as_tuple=False).squeeze(-1)
                    bonus[idx] = fh.first_step_bonus(idx, rstep[idx], tstep[idx])
                    diag = getattr(env, "_bonus_diag", None)
                    if diag is not None:
                        diag["paid"] += int((bonus > 0).sum())
                        diag["sum"] += float((bonus > 0).sum()) * args.bonus_w
                return bonus * (1.0 / env.step_dt)   # event reward: cancel dt
            # NOTE: relies on term order — down_foothold runs first in the
            # reward manager (insertion order) and fills env._fh_r_step
            env_cfg.rewards.down_first_step_bonus = _RW2(
                func=_first_step_bonus, weight=float(args.bonus_w))
            import torch as _bt
            _bt  # noqa: F401 (keep torch import local)
            print(f"[CURR_FIRST_BONUS] per-TREAD one-shot (variant A): first "
                  f"qualifying landing on EACH tread pays once per episode "
                  f"(up to {cc.FOOTHOLD_NUM_TARGETS}x); quality>="
                  f"{cc.BONUS_QUALITY}; w={args.bonus_w} "
                  f"(parallel to foothold, separate TB curve)", flush=True)

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
            if not hasattr(env, "_fh"):               # foothold guide (v1.3.4)
                import curriculum_core as _cc
                env._fh = _cc.FootholdGuide(eorig.shape[0], dev)
                env._fh_contact = torch.zeros(eorig.shape[0], 2,
                                              dtype=torch.bool, device=dev)
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
                    if pool == 3:                     # down pool: tread targets
                        man_t = man
                        d_t = man_t["tread_m"]
                        dn = sorted([e for e in man_t["edges"] if e["dir"] == "down"],
                                    key=lambda e: e["x_local"])
                        tg = [(float(eorig[i, 0]) + (e["x_local"] + d_t / 2 - 1.0),
                               e["z_low"]) for e in dn]
                        tg.append((float(eorig[i, 0])
                                   + (dn[-1]["x_local"] + d_t + 0.3 - 1.0), 0.0))
                        k_t = min(len(tg), env._fh.targets.shape[1])
                        env._fh.targets[i, :k_t] = torch.tensor(
                            tg[:k_t], dtype=torch.float, device=dev)
                        env._fh.valid[i] = True
                    else:
                        env._fh.valid[i] = False
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
    stall_det = cc.StallDetector(N, dev)
    unw._down_stall = stall_det.stall
    # lead-time telemetry: first tick the next tread is visible in the FWD
    # scan -> first swing-start (foot liftoff) after that; once per episode
    unw._lt_seen = torch.full((N,), -1, dtype=torch.long, device=dev)
    unw._lt_done = torch.zeros(N, dtype=torch.bool, device=dev)
    unw._lt_contact = torch.zeros(N, 2, dtype=torch.bool, device=dev)

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
             "down_pool_up": 0, "down_pool_down": 0, "down_pool_top": 0,
             "lt_sum_s": 0.0, "lt_n": 0}
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
        stall_det.on_reset(idx, unw._curr_x0)          # fresh grace per episode
        unw._lt_seen[idx] = -1                         # fresh lead-time window
        unw._lt_done[idx] = False
        if hasattr(unw, "_fh"):
            unw._fh.reset_env(idx)                     # neutral first-stride per episode
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
            # v1.3.5 lead-time telemetry (down pool): next-tread visibility
            # in the FWD scan -> first foot liftoff after first sight
            if args.fwd_shift > 0 and (unw._curr_pool == 3).any():
                idx_d = (unw._curr_pool == 3).nonzero(as_tuple=False).squeeze(-1)
                fwd_s = unw.scene.sensors.get("height_scanner_fwd")
                if fwd_s is not None:
                    tg = unw._fh.targets[idx_d]                 # [k, T, 2]
                    rx = robot.data.root_pos_w[idx_d, 0]
                    ahead = tg[..., 0] > (rx + 0.15).unsqueeze(-1)
                    err_x = torch.where(ahead, (tg[..., 0] - rx.unsqueeze(-1)).abs(),
                                        torch.full_like(tg[..., 0], 1e9))
                    nxt_i = err_x.argmin(-1)
                    nxt = tg[torch.arange(idx_d.numel(), device=dev), nxt_i]
                    hits = fwd_s.data.ray_hits_w[idx_d]         # [k, 121, 3]
                    hz, hx = hits[..., 2], hits[..., 0]
                    vis = (torch.isfinite(hz)
                           & (hz > nxt[:, 1] - 0.02) & (hz < nxt[:, 1] + 0.02)
                           & (hx > nxt[:, 0] - 0.25) & (hx < nxt[:, 0] + 0.25)).any(-1)
                    seen = unw._lt_seen[idx_d]
                    newly = vis & (seen < 0)
                    if newly.any():
                        unw._lt_seen[idx_d[newly]] = unw._cc_tick
                    fz_now = contact.data.net_forces_w_history[:, -1][:, feet_ids].norm(dim=-1)
                    now_c = fz_now >= cc.CONTACT_FORCE_N
                    lift = (unw._lt_contact[idx_d] & ~now_c[idx_d]).any(-1)
                    acted = lift & (unw._lt_seen[idx_d] >= 0) & ~unw._lt_done[idx_d]
                    if acted.any():
                        k2 = idx_d[acted]
                        lead = (unw._cc_tick - unw._lt_seen[k2]).float()
                        stats["lt_sum_s"] += float(lead.sum()) * dt
                        stats["lt_n"] += int(k2.numel())
                        unw._lt_done[k2] = True
                    unw._lt_contact[idx_d] = now_c[idx_d]
            if unw._cc_tick % cc.STALL_WINDOW_STEPS == 0:
                cmd_x = unw.command_manager.get_command("base_velocity")[:, 0]
                stall_det.on_window(robot.data.root_pos_w[:, 0],
                                    unw._curr_pool == 3,
                                    cmd_x.abs() > 0.1)
                unw._down_stall = stall_det.stall
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

    # ---- v1.3.5: forward-shifted POLICY view; rewards keep centered scan ----
    # Decoupled per user spec: base_height/gait datum still reads the original
    # centered 0.5x0.5 scanner (its reference error audited at ~1 cm); only
    # the policy/critic heights term moves forward so the NEXT tread enters
    # observation before stepping. Misses become an explicit unknown sentinel
    # (-2.0, outside the [-1,1] physical clip range), never a fake height.
    if args.fwd_shift > 0:
        from isaaclab.sensors.ray_caster import RayCasterCfg
        from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg
        from isaaclab.managers import SceneEntityCfg as _SE

        def _shifted_grid_pattern(pcfg, device):
            x = torch.arange(-pcfg.size[0] / 2 + pcfg.shift_x,
                             pcfg.size[0] / 2 + pcfg.shift_x + 1e-9,
                             pcfg.resolution, device=device)
            y = torch.arange(-pcfg.size[1] / 2, pcfg.size[1] / 2 + 1e-9,
                             pcfg.resolution, device=device)
            gx, gy = torch.meshgrid(x, y, indexing="xy")
            starts = torch.zeros(gx.numel(), 3, device=device)
            starts[:, 0] = gx.flatten()
            starts[:, 1] = gy.flatten()
            dirs = torch.zeros_like(starts)
            dirs[:, 2] = -1.0
            return starts, dirs

        from isaaclab.utils import configclass as _configclass

        @_configclass
        class _ShiftedGridCfg(GridPatternCfg):
            shift_x: float = float(args.fwd_shift)
            func = _shifted_grid_pattern

        fwd_cfg = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=_ShiftedGridCfg(resolution=0.05, size=(0.5, 0.5)),
            debug_vis=False, mesh_prim_paths=["/World/ground"])
        fwd_cfg.update_period = env_cfg.decimation * env_cfg.sim.dt
        env_cfg.scene.height_scanner_fwd = fwd_cfg

        def _heights_fwd_obs(env, sensor_cfg):
            s = env.scene[sensor_cfg.name]
            rz = env.scene["robot"].data.root_pos_w[:, 2].unsqueeze(1)
            return cc.heights_forward_miss_marked(s.data.ray_hits_w[..., 2], rz)

        for _grp in (env_cfg.observations.policy, env_cfg.observations.critic):
            _t = getattr(_grp, "heights", None)
            if _t is not None:
                _t.func = _heights_fwd_obs
                _t.params = {"sensor_cfg": _SE("height_scanner_fwd")}
        print(f"[CURR_FWD_VIEW] policy/critic heights -> scan shifted +"
              f"{args.fwd_shift} m (121 rays kept, checkpoint dims intact); "
              f"misses = {cc.HEIGHTS_MISS_SENTINEL} (unknown, not fake ground); "
              f"reward datum UNCHANGED (centered scanner)", flush=True)
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
            m_extra = {}
            fwd_s = unw.scene.sensors.get("height_scanner_fwd")
            if fwd_s is not None:
                miss = ~torch.isfinite(fwd_s.data.ray_hits_w[..., 2])
                m_extra["fwd_miss_frac_down"] = round(
                    float(miss[pool_ids["down"]].float().mean()), 4)
            if stats["lt_n"] > 0:
                m_extra["lead_time_mean_s"] = round(stats["lt_sum_s"] / stats["lt_n"], 3)
                m_extra["lead_time_n"] = stats["lt_n"]
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
                 "fh_diag": dict(getattr(unw, "_fh_diag", {})),
                 "bonus_diag": dict(getattr(unw, "_bonus_diag", {})),
                 **m_extra,
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
    parser.add_argument("--still_w", type=float, default=0.0,
                        help="down-pool sustained-stall penalty weight "
                             "(pre-registered range 1/5~1/3 of track_w; default OFF)")
    parser.add_argument("--fh_w", type=float, default=0.0,
                        help="down-pool foothold guidance weight, CoM-stride "
                             "bound (v1.3.4; default OFF)")
    parser.add_argument("--fwd_shift", type=float, default=0.15,
                        help="forward shift of the POLICY height scan in m "
                             "(v1.3.5; rewards keep the centered scan; 0=off)")
    parser.add_argument("--bonus_w", type=float, default=0.0,
                        help="one-shot first-step bounty for the down pool, "
                             "parallel to foothold (v1.3.6; default OFF)")
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
