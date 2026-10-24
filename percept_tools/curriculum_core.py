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


def decide(passage, fell, is_stairs, disp_x, level, max_level, streak=None):
    """Episode-end decision. Failure priority: a real failure vetoes promotion
    and forces demotion. Buffer-row envs promote on clean forward displacement.
    streak is accepted but IGNORED (single-pass promotion, block-1 semantics;
    the streak-2 gate was tried in block 2 and rolled back same day).
    Returns (up, down, top_pass); top_pass = would-promote clamped at top row."""
    down = fell | (is_stairs & ~passage)
    up = torch.where(is_stairs, passage, ~fell & (disp_x >= FLAT_PROMOTE_DISP_M))
    up = up & ~down
    top_pass = up & (level >= max_level)
    up = up & (level < max_level)
    return up, down, top_pass


def apply_levels(levels, types, origins_grid, env_ids, up, down, max_level):
    """In-place level bookkeeping; returns (new_levels[k], new_origins[k,3]).
    Replicates importer.update_env_origins mechanics minus random top-scatter."""
    lv = (levels[env_ids] + up.long() - down.long()).clamp(0, int(max_level))
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

    # (e) walked far past goal and confirmed, THEN fell: failure priority
    tr = EpisodeTracker(1, dev, confirm_frames=25)
    p = _run_episode(tr, 0, [(True, True, True, 0.3)] * 35)
    up, down, _ = decide(p, torch.tensor([True]), torch.tensor([True]),
                         torch.tensor([5.0]), torch.tensor([1]), max_level=3)
    check("e_fall_after_far_walk_overrides", p[0] and (not up[0]) and down[0])

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

    # 5. apply_levels: top clamp, bottom clamp, origin indexing
    levels = torch.tensor([3, 0, 1, 2])
    types = torch.tensor([2, 1, 0, 3])
    grid = torch.arange(4 * 4 * 3, dtype=torch.float).reshape(4, 4, 3)
    ids = torch.tensor([0, 1, 2, 3])
    up = torch.tensor([True, False, False, False])
    down = torch.tensor([False, True, False, True])
    lv, org = apply_levels(levels, types, grid, ids, up, down, max_level=3)
    check("top_clamp", int(lv[0]) == 3 and torch.equal(org[0], grid[3, 2]))
    check("bottom_clamp", int(lv[1]) == 0 and torch.equal(org[1], grid[0, 1]))
    check("demoted_index", int(lv[3]) == 1 and torch.equal(org[3], grid[1, 3]))

    bad = [n for n, c in ok if not c]
    print(f"[CURR_CORE_SELFTEST] {'ALL_PASS' if not bad else 'FAILURES: ' + str(bad)}")
    return not bad


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
