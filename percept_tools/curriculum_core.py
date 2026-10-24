"""v1.3 curriculum decision core (DESIGN.md v1.3) — pure torch, CPU-testable.

Replaces official terrain_levels_vel semantics (source-verified direction-
blind and fall-blind on isaaclab 2.3.1 + CPU repro): promotion is manifest
route completion (goal line + debounced both-foot contact past goal + 0.5 s
stability at forward speed); demotion is real failure with priority over any
completion evidence. Buffer-row (flat/rough) promotion is clean forward
displacement. Top row clamps (official random top-scatter disabled).

Run `python curriculum_core.py` for the self-tests, including the
direction-blindness regressions (design smoke assertions d/e).
"""
from __future__ import annotations

import torch

CONTACT_FORCE_N = 1.0
CONTACT_DEBOUNCE_FRAMES = 5
CONFIRM_VX_MIN = 0.05
FLAT_PROMOTE_DISP_M = 2.5          # 2026-10-24: reverted to block-1 value
                                   # (block-2 tried 4.0 + streak-2; rows drained
                                   # by the down-task weakness, knobs rolled back)

# ---- down-pool foothold guidance with anti-farming binding (v1.3.4) ----
# A landing earns precision x stride_factor: precision decays with the x-error
# to the next-ahead tread target; stride_factor is the CoM (root) FORWARD
# displacement since that foot's previous landing, normalized by one tread.
# Stepping precisely WITHOUT body progress ("hitting points for score") pays
# ~0 — the user's binding requirement. z is terrain-enforced at contact, so
# the x-error carries the targeting signal.
FOOTHOLD_TOL_M = 0.25          # precision decays to 0 at this x-error
FOOTHOLD_STRIDE_REF_M = 0.34   # full credit needs a full-tread CoM stride
FOOTHOLD_NUM_TARGETS = 4       # 3 down treads + ground exit band
BONUS_QUALITY = 0.5            # prec x stride threshold for a "committed" step
                                 # (v1.3.6 first-step bounty; blocks farming)
HEIGHTS_OFFSET = 0.5           # official scan formula offset
HEIGHTS_MISS_SENTINEL = -2.0   # unknown marker: OUTSIDE the valid clip range
                                 # [-1, 1] so a miss can never masquerade as a
                                 # definite ground height (user spec v1.3.5)


def heights_forward_miss_marked(hits_z, root_z, offset=HEIGHTS_OFFSET):
    """Policy-side forward-view heights with EXPLICIT unknown marking.
    Valid rays: pos_z - hit_z - offset, clipped to [-1, 1] (finite rays keep
    their in-range values bit-exact). Missed rays (Inf): HEIGHTS_MISS_SENTINEL
    (-2.0), which lies outside the physical clip range — 'unknown', not a
    fabricated ground height. Returns [N, num_rays].
    NOTE: finiteness is checked on the RAW value BEFORE clipping — clip would
    otherwise turn ±Inf into finite bounds and erase the miss marker."""
    raw = root_z - hits_z - offset
    valid = torch.isfinite(raw)
    clipped = torch.clip(raw, -1.0, 1.0)
    return torch.where(valid, clipped,
                       torch.full_like(clipped, HEIGHTS_MISS_SENTINEL))


class FootholdGuide:
    """Per-env foothold reward core. targets/valid are set at spawn (down-pool
    envs get tread-center targets); on_landing() per detected landing event."""

    def __init__(self, num_envs, device):
        self.root_at_land = torch.full((num_envs, 2), float("nan"), device=device)
        self.targets = torch.zeros(num_envs, FOOTHOLD_NUM_TARGETS, 2, device=device)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.bonus_used = torch.zeros(num_envs, FOOTHOLD_NUM_TARGETS,
                                      dtype=torch.bool, device=device)

    def reset_env(self, env_ids):
        self.root_at_land[env_ids] = float("nan")
        self.bonus_used[env_ids] = False

    def on_landing(self, env_ids, foot, foot_x, root_x):
        """env_ids [k] long, foot int (0/1), foot_x [k], root_x [k].
        Returns reward [k]: precision * stride_factor (0 for non-registered
        envs). First landing of an episode gets a neutral stride factor."""
        prev = self.root_at_land[env_ids, foot]
        first = torch.isnan(prev)
        stride = torch.where(first, torch.full_like(prev, FOOTHOLD_STRIDE_REF_M),
                             root_x - prev)
        self.root_at_land[env_ids, foot] = root_x
        stride_factor = (stride / FOOTHOLD_STRIDE_REF_M).clamp(0.0, 1.0)
        origin = root_x - stride                       # root at previous landing
        tg = self.targets[env_ids]                     # [k, T, 2]
        ahead = tg[..., 0] > (origin + 0.10).unsqueeze(-1)   # strictly ahead
        err = (tg[..., 0] - foot_x.unsqueeze(-1)).abs()
        err = torch.where(ahead, err, torch.full_like(err, 1e9))
        best, best_idx = err.min(dim=-1)
        has_t = self.valid[env_ids] & (best < 1e8)
        prec = (1.0 - best / FOOTHOLD_TOL_M).clamp(0.0, 1.0)
        reward = torch.where(has_t, prec * stride_factor, torch.zeros_like(prec))
        tgt_idx = torch.where(has_t, best_idx,
                              torch.full_like(best_idx, -1))
        return reward, tgt_idx

    def first_step_bonus(self, env_ids, r, tgt_idx):
        """Per-TREAD one-shot bounty (variant A, 2026-10-25): the first
        qualifying landing aimed at EACH target pays once per episode
        (up to T awards). Sub-threshold landings never consume anything.
        NOTE: fancy indexing returns a COPY — the mutated rows must be
        written back or the used-flag never persists."""
        qual = r >= BONUS_QUALITY
        tgt_ok = tgt_idx >= 0
        used_now = self.bonus_used[env_ids].clone()
        row = torch.arange(env_ids.numel(), device=r.device)
        already = used_now[row, tgt_idx.clamp(min=0)]
        award = qual & tgt_ok & ~already
        used_now[row, tgt_idx.clamp(min=0)] |= (qual & tgt_ok)
        self.bonus_used[env_ids] = used_now
        return award.float()

    def reset_bonus(self, env_ids):
        self.bonus_used[env_ids] = False
# Applies ONLY to down-pool envs under a forward command; triggers on
# SUSTAINED low progress (not instantaneous vx); foot-swap and brief probing
# stay exempt via windowed measurement + episode-start grace.
STALL_WINDOW_STEPS = 150           # 3 s @ 50 Hz measurement window
STALL_PROG_M = 0.15                # < 0.05 m/s window average counts as low
STALL_GRACE_WINDOWS = 2            # first 6 s of an episode: exempt
STALL_CONFIRM_WINDOWS = 2          # two consecutive low windows -> sustained


class StallDetector:
    """Per-env sustained-low-progress detector (down-pool scoped). Call
    on_window() every STALL_WINDOW_STEPS steps and on_reset() at spawn."""

    def __init__(self, num_envs, device):
        z = torch.zeros
        self.win_x0 = z(num_envs, device=device)
        self.low_cnt = z(num_envs, dtype=torch.long, device=device)
        self.grace = z(num_envs, dtype=torch.long, device=device)
        self.stall = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def on_reset(self, env_ids, x):
        self.win_x0[env_ids] = x[env_ids]
        self.low_cnt[env_ids] = 0
        self.grace[env_ids] = STALL_GRACE_WINDOWS
        self.stall[env_ids] = False

    def on_window(self, x, pool_down, cmd_fwd):
        """x: current root x [N]; pool_down: down-pool mask [N];
        cmd_fwd: forward-command mask [N]. Updates self.stall in place."""
        d = x - self.win_x0
        self.win_x0 = x.clone()
        low = pool_down & cmd_fwd & (d < STALL_PROG_M)
        in_grace = self.grace > 0
        self.grace = torch.where(in_grace, self.grace - 1,
                                 torch.zeros_like(self.grace))
        self.low_cnt = torch.where(low & ~in_grace, self.low_cnt + 1,
                                   torch.zeros_like(self.low_cnt))
        self.stall = self.low_cnt >= STALL_CONFIRM_WINDOWS


class EpisodeTracker:
    """Per-env episode accumulation, one update() per physics frame."""

    def __init__(self, num_envs, device, confirm_frames):
        self.confirm_frames = int(confirm_frames)
        self.contact_deb = torch.zeros(num_envs, 2, dtype=torch.long, device=device)
        self.contacted_past = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.confirm = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.passage = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, env_ids, past, foot_contact, vx):
        """past [k] bool: root AND both feet past goal (stairs envs only).
        foot_contact [k,2] bool: fz >= CONTACT_FORCE_N. vx [k]: forward speed."""
        deb = torch.where(foot_contact, self.contact_deb[env_ids] + 1,
                          torch.zeros_like(self.contact_deb[env_ids]))
        self.contact_deb[env_ids] = deb
        self.contacted_past[env_ids] |= past.unsqueeze(-1) & (deb >= CONTACT_DEBOUNCE_FRAMES)
        stable = past & self.contacted_past[env_ids].all(dim=-1)
        conf = torch.where(stable, self.confirm[env_ids] + 1,
                           torch.zeros_like(self.confirm[env_ids]))
        self.confirm[env_ids] = conf
        self.passage[env_ids] |= (stable & (conf >= self.confirm_frames)
                                  & (vx > CONFIRM_VX_MIN))

    def reset(self, env_ids):
        self.contact_deb[env_ids] = 0
        self.confirm[env_ids] = 0
        self.contacted_past[env_ids] = False
        self.passage[env_ids] = False


def decide(passage, fell, is_stairs, disp_x, level, max_level, min_level=1):
    """Direction-pool semantics (v1.3.2, 2026-10-24): the verdict LOCKS at
    passage. A pool env that passes its own goal (up-pool: TOP platform,
    down-pool: BOTTOM platform) promotes even if it falls afterwards — the
    pool's task ends at its own platform; failures beyond it must not veto
    promotion. Any non-passage outcome (fell or timeout) demotes.
    min_level: pool ladder floor (stair pools clamp at the easiest stair row;
    the flat buffer row is pinned-replay-only and not part of any ladder).
    Returns (up, down, top_pass); top_pass = would-promote clamped at top row."""
    down = is_stairs & ~passage
    up = torch.where(is_stairs, passage, ~fell & (disp_x >= FLAT_PROMOTE_DISP_M))
    up = up & ~down
    top_pass = up & (level >= max_level)
    up = up & (level < max_level)
    return up, down, top_pass


def apply_levels(levels, types, origins_grid, env_ids, up, down, max_level,
                 min_level=1):
    """In-place level bookkeeping; returns (new_levels[k], new_origins[k,3]).
    Replicates importer.update_env_origins mechanics minus random top-scatter;
    clamps at min_level (pool floor) instead of zero."""
    lv = (levels[env_ids] + up.long() - down.long()).clamp(
        int(min_level), int(max_level))
    levels[env_ids] = lv
    return lv, origins_grid[lv, types[env_ids]]


# ----------------------------- self-tests ---------------------------------

def _run_episode(tracker, env_id, frames):
    """frames: list of (past, contact_l, contact_r, vx)."""
    idx = torch.tensor([env_id])
    for past, cl, cr, vx in frames:
        tracker.update(idx, torch.tensor([past]),
                       torch.tensor([[cl, cr]]), torch.tensor([vx], dtype=torch.float))
    return tracker.passage[idx]


def selftest():
    dev = "cpu"
    ok = []

    def check(name, cond):
        ok.append((name, bool(cond)))
        print(("PASS " if cond else "FAIL ") + name)

    # 1. legitimate up passage: past goal, debounced contacts, 0.5 s confirm
    tr = EpisodeTracker(1, dev, confirm_frames=25)
    frames = [(True, True, True, 0.3)] * 35          # >= debounce 5 + confirm 25
    p = _run_episode(tr, 0, frames)
    up, down, top = decide(p, torch.tensor([False]), torch.tensor([True]),
                           torch.tensor([5.0]), torch.tensor([1]), max_level=3)
    check("legit_up_passage", p[0] and up[0] and not down[0])

    # (d) direction-blindness regression: pure lateral strafe far from origin
    # never satisfies goal-past -> no promotion, timeout-no-pass demotes
    tr = EpisodeTracker(1, dev, confirm_frames=25)
    p = _run_episode(tr, 0, [(False, True, True, 0.0)] * 200)   # strafe/backward
    up, down, _ = decide(p, torch.tensor([False]), torch.tensor([True]),
                         torch.tensor([5.0]), torch.tensor([1]), max_level=3)
    check("d_direction_blind_no_up", (not up[0]) and down[0])

    # 2. short confirm window (never reaches 0.5 s) does not pass
    tr = EpisodeTracker(1, dev, confirm_frames=25)
    p = _run_episode(tr, 0, [(True, True, True, 0.3)] * 10)
    check("short_confirm_no_pass", not p[0])

    # 3. contact not debounced (single-frame taps) does not pass
    tr = EpisodeTracker(1, dev, confirm_frames=25)
    frames = [(True, True, True, 0.3)] * 4 + [(True, False, False, 0.3)] * 2 \
             + [(True, True, True, 0.3)] * 4 + [(True, False, False, 0.3)] * 2 \
             + [(True, True, True, 0.3)] * 4
    p = _run_episode(tr, 0, frames)
    check("undebounced_contact_no_pass", not p[0])

    # (e) verdict lock (v1.3.2): passed the pool goal, THEN fell beyond it ->
    # still promotes; the pool's task ended at its own platform
    tr = EpisodeTracker(1, dev, confirm_frames=25)
    p = _run_episode(tr, 0, [(True, True, True, 0.3)] * 35)
    up, down, _ = decide(p, torch.tensor([True]), torch.tensor([True]),
                         torch.tensor([5.0]), torch.tensor([1]), max_level=3)
    check("e_pass_then_fall_still_promotes", p[0] and up[0] and not down[0])
    # fell WITHOUT passage still demotes (failure within the pool's own task)
    tr = EpisodeTracker(1, dev, confirm_frames=25)
    p = _run_episode(tr, 0, [(False, True, True, 0.0)] * 100)
    up, down, _ = decide(p, torch.tensor([True]), torch.tensor([True]),
                         torch.tensor([5.0]), torch.tensor([1]), max_level=3)
    check("fail_before_goal_demotes", (not up[0]) and down[0])

    # 4. buffer row: clean forward displacement promotes (2.5 m, block-1
    #    semantics), short does not
    up, down, _ = decide(torch.tensor([False]), torch.tensor([False]),
                         torch.tensor([False]), torch.tensor([3.0]),
                         torch.tensor([0]), max_level=3)
    check("flat_disp_promotes", up[0] and not down[0])
    up, down, _ = decide(torch.tensor([False]), torch.tensor([False]),
                         torch.tensor([False]), torch.tensor([1.0]),
                         torch.tensor([0]), max_level=3)
    check("flat_short_stays", (not up[0]) and (not down[0]))

    # 5. apply_levels: top clamp, pool floor clamp (min_level=1), origin indexing
    levels = torch.tensor([3, 1, 1, 2])
    types = torch.tensor([2, 1, 0, 3])
    grid = torch.arange(4 * 4 * 3, dtype=torch.float).reshape(4, 4, 3)
    ids = torch.tensor([0, 1, 2, 3])
    up = torch.tensor([True, False, False, False])
    down = torch.tensor([False, True, False, True])
    lv, org = apply_levels(levels, types, grid, ids, up, down, max_level=3,
                           min_level=1)
    check("top_clamp", int(lv[0]) == 3 and torch.equal(org[0], grid[3, 2]))
    check("pool_floor_clamp", int(lv[1]) == 1 and torch.equal(org[1], grid[1, 1]))
    check("demoted_index", int(lv[3]) == 1 and torch.equal(org[3], grid[1, 3]))

    # 6. StallDetector: sustained low progress, down-pool scoped, grace-aware
    sd = StallDetector(1, dev)
    sd.on_reset(torch.tensor([0]), torch.tensor([10.0]))
    dn = torch.tensor([True]); fwd = torch.tensor([True])
    # grace windows (2) with zero progress: no trigger
    sd.on_window(torch.tensor([10.0]), dn, fwd)
    sd.on_window(torch.tensor([10.02]), dn, fwd)
    check("stall_grace_exempt", not sd.stall[0])
    # two consecutive low windows after grace -> sustained
    sd.on_window(torch.tensor([10.05]), dn, fwd)
    check("stall_needs_two_windows", not sd.stall[0])
    sd.on_window(torch.tensor([10.08]), dn, fwd)
    check("stall_sustained_triggers", bool(sd.stall[0]))
    # a progress window (>0.15 m) clears immediately
    sd.on_window(torch.tensor([11.0]), dn, fwd)
    check("stall_progress_clears", not sd.stall[0])
    # brief probe: one low window after progress does not re-trigger
    sd.on_window(torch.tensor([11.05]), dn, fwd)
    check("stall_brief_probe_exempt", not sd.stall[0])
    # non-down-pool envs are never flagged
    sd2 = StallDetector(1, dev)
    sd2.on_reset(torch.tensor([0]), torch.tensor([0.0]))
    for _ in range(6):
        sd2.on_window(torch.tensor([0.0]), torch.tensor([False]), fwd)
    check("stall_up_pool_never_flagged", not sd2.stall[0])

    # 7. FootholdGuide: precision x CoM-stride binding (v1.3.4)
    fg = FootholdGuide(1, dev)
    fg.valid[0] = True
    fg.targets[0] = torch.tensor([[4.5, 0.08], [4.9, 0.04], [5.3, 0.0], [5.8, 0.0]])
    ids = torch.tensor([0])
    fg.root_at_land[0, 0] = 4.15                     # full-tread CoM stride
    r, ti = fg.on_landing(ids, 0, torch.tensor([4.52]), torch.tensor([4.5]))
    check("fh_precise_full_stride", float(r[0]) > 0.9)
    # precise landing but CoM barely moved -> farming pays ~0
    fg.root_at_land[0, 1] = 4.49
    r, _ = fg.on_landing(ids, 1, torch.tensor([4.51]), torch.tensor([4.5]))
    check("fh_farming_blocked", float(r[0]) < 0.1)
    # far off-target landing -> zero
    fg.root_at_land[0, 0] = 4.1
    r, ti_off = fg.on_landing(ids, 0, torch.tensor([6.1]), torch.tensor([6.1]))
    check("fh_offtarget_zero", float(r[0]) == 0.0)
    # non-registered env (up pool) -> zero
    fg3 = FootholdGuide(1, dev)
    r, _ = fg3.on_landing(ids, 0, torch.tensor([4.52]), torch.tensor([4.5]))
    check("fh_invalid_env_zero", float(r[0]) == 0.0)
    # first landing of an episode: neutral stride factor, precision applies
    fg.reset_env(ids)
    r, _ = fg.on_landing(ids, 0, torch.tensor([4.52]), torch.tensor([4.5]))
    check("fh_first_landing_neutral_stride", float(r[0]) > 0.9)
    # k=2 batch regression (broadcast bug: origin [k] vs targets [k,T] blew up
    # with k>1 while every k=1 test passed silently)
    fg4 = FootholdGuide(2, dev)
    fg4.valid[:] = True
    fg4.targets[:, :4] = torch.tensor([[4.5, 0.08], [4.9, 0.04],
                                       [5.3, 0.0], [5.8, 0.0]]).unsqueeze(0)
    ids2 = torch.tensor([0, 1])
    fg4.root_at_land[:, 1] = torch.tensor([4.15, 4.15])
    r, _ = fg4.on_landing(ids2, 1, torch.tensor([4.52, 4.51]),
                          torch.tensor([4.5, 4.5]))
    check("fh_batch_k2_no_broadcast_error",
          r.shape[0] == 2 and float(r[0]) > 0.9 and float(r[1]) > 0.85)

    # 8. forward-view heights: miss sentinel, valid values clipped (v1.3.5)
    rz = torch.tensor([[0.85]])
    hits = torch.tensor([[[0.35, float("inf"), 1.35, 0.85]]])   # [1,1,4]
    out = heights_forward_miss_marked(hits, rz)
    expect_valid = torch.tensor([[0.0, -2.0, -1.0, -0.5]])
    check("fh_view_miss_sentinel", bool((out == expect_valid).all()))

    # 9. per-tread first-step bounty, variant A (v1.3.6a, 2026-10-25)
    fb = FootholdGuide(1, dev)
    fb.valid[0] = True
    fb.targets[0] = torch.tensor([[4.5, 0.08], [4.9, 0.04], [5.3, 0.0], [5.8, 0.0]])
    ids0 = torch.tensor([0])
    # step onto tread 1 (target idx 0): full quality -> pays
    fb.root_at_land[0, 0] = 4.15
    r0, t0 = fb.on_landing(ids0, 0, torch.tensor([4.52]), torch.tensor([4.5]))
    b1 = fb.first_step_bonus(ids0, r0, t0)
    check("bonusA_first_tread_pays",
          float(r0[0]) > 0.5 and int(t0[0]) == 0 and float(b1[0]) == 1.0)
    # SAME tread again: no repeat
    fb.root_at_land[0, 0] = 4.15
    r1, t1 = fb.on_landing(ids0, 0, torch.tensor([4.52]), torch.tensor([4.5]))
    b2 = fb.first_step_bonus(ids0, r1, t1)
    check("bonusA_same_tread_no_repeat",
          float(r1[0]) > 0.5 and int(t1[0]) == 0 and float(b2[0]) == 0.0)
    # NEXT tread (target idx 1, x 4.9): pays again
    fb.root_at_land[0, 1] = 4.55
    r2, t2 = fb.on_landing(ids0, 1, torch.tensor([4.92]), torch.tensor([4.9]))
    b3 = fb.first_step_bonus(ids0, r2, t2)
    check("bonusA_next_tread_pays_again",
          float(r2[0]) > 0.5 and int(t2[0]) == 1 and float(b3[0]) == 1.0)
    # sub-threshold landing consumes nothing
    fb.root_at_land[0, 1] = 4.85
    r3, t3 = fb.on_landing(ids0, 1, torch.tensor([4.88]), torch.tensor([4.87]))
    b4 = fb.first_step_bonus(ids0, r3, t3)
    check("bonusA_subthreshold_no_consume",
          float(r3[0]) < BONUS_QUALITY and float(b4[0]) == 0.0)
    # episode reset re-arms every tread
    fb.reset_env(ids0)
    fb.root_at_land[0, 0] = 4.15
    r4, t4 = fb.on_landing(ids0, 0, torch.tensor([4.52]), torch.tensor([4.5]))
    b5 = fb.first_step_bonus(ids0, r4, t4)
    check("bonusA_reset_rearms", float(r4[0]) > 0.5 and float(b5[0]) == 1.0)

    bad = [n for n, c in ok if not c]
    print(f"[CURR_CORE_SELFTEST] {'ALL_PASS' if not bad else 'FAILURES: ' + str(bad)}")
    return not bad


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
