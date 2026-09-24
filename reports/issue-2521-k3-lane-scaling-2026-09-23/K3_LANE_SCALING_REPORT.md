# K3 multi-AIV lane scaling — managed-host `all_to_all_v`, EP4 on 4×910B2

*Campaign: 2026-09-23 · `--impl managed-host`, EP4, NPUs 4,5,6,7 · **50/50 cells OK** ·
kernel under test: pypto branch `feat/all-to-all-v-multiaiv-lanes`, as-run `1af2335b`
(post-rebase equivalent `67572c26`; the 2026-09-24 rebase's only kernel delta is a comment, so these
numbers apply to the current branch head as well)*

## TL;DR

With K3's per-block work partition in place, launching `B > 1` AIV blocks is no longer a
fixed-cost-only exercise — it is the **first configuration where a genuine bandwidth win is
measurable**:

- **1 MiB/peer: −40.3 % at B=4, −50.9 % at B=8** vs the B=1 arm; egress 1.54 → 3.15 GB/s.
- **256 KiB/peer: −17.8 % / −19.3 % / −25.1 %** at B=4 / B=8 / B=16.
- **16 KiB/peer: −7.7 % / −9.0 % / −14.1 %** at B=4 / B=8 / B=16.
- Zero-counts control (no TPUTs): the fixed per-block machinery now costs **+1.5 %** at
  B=8 / EP4 — down from **+7.6 %** in the K2 baseline (the same cell, K2 code).

Contrast with K2's own sweep (same harness, same cards): L=1 vs L=8 deltas were
−3.4 %…+7.6 % with sign flips — no partitioning existed then. K3's curve is monotone-ish and
large where the payload dominates.

## Setup

- `collectives/alltoallv_a1.py`, EP4 (`A2AV_DEV4=4,5,6,7`), `--impl managed-host`, persistent
  windows, `--rounds 100 --warmup 5 --swimlane-rounds 8`, `A2AV_CORE_NUMS=1,2,4,8,16`,
  `A2AV_REPS=2` (interleaved pairs), payloads `{0, 16384, 262144, 1048576}` + `zero@24960`
  control.
- Metric: whole-program timing slot (`host_timing_slot_*`); the HOST rail has no independent
  AIV metric, so read deltas **pairwise**, not absolutely.
- `B = CalAllToAllVBlocks(P, L)` → for EP4: `B = L` for `L ≤ 4`, then 8, then **16** (no `2P`
  cap — see "Two harness catches" below). L=16 exercises `K = 4` lanes/peer.
- The kernel itself passed the multicore ST file **13/13 (888 s)** on the same cards before the
  campaign; the campaign doubles as a ~50 × (5 + 100 + 2×8)-round back-to-back reuse soak.

## Whole-program timing slot — per-arm means and Δ vs `L=1`

| payload B/peer | L=1 (µs) | L=2 | L=4 | L=8 | L=16 | Δ L=4 | Δ L=8 | Δ L=16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7 922.1 | 8 013.9 | 7 741.5 | 7 905.3 | 7 591.1 | −2.28 % | −0.21 % | −4.18 % |
| 16 384 | 4 460.4 | 4 323.1 | 4 119.1 | 4 060.3 | 3 833.6 | −7.65 % | −8.97 % | −14.05 % |
| 262 144 | 6 649.1 | 5 865.3 | 5 463.0 | 5 368.2 | 4 983.3 | −17.84 % | −19.26 % | −25.05 % |
| 1 048 576 | 16 330.3 | 12 524.9 | 9 754.7 | 8 020.5 | 8 239.8 | −40.27 % | −50.89 % | −49.54 % |

Egress (GB/s, per-cell mean): 16 KiB 0.089 → 0.103; 256 KiB 0.946 → 1.263; 1 MiB
1.543 (L=1) → 2.010 (L=2) → 2.581 (L=4) → **3.150 (L=8)** → 3.055 (L=16).

Raw per-rep spreads are in `analysis_output.txt`; 2 reps/arm, so treat ≤ ~3 % steps as noise.

## H1 readout (monotone non-increasing over `B ≥ P`, ±2 % band)

- 16 KiB, 256 KiB: **monotone within noise** (worst step ≈ 0).
- payload 0: flagged at L=4→8 (+2.12 %, i.e. marginally over the band) — all arms sit in the
  ~7.6–8.0 ms fixed-cost regime; the overall trend still improves (L=16 fastest at −4.2 %).
- 1 MiB: flagged at L=8→16 (+2.73 %; 8 021 vs 8 240 µs). Within the observed rep spread
  (L=8 runs: 7 513–8 528 µs), but consistent with `K=4` adding lane overhead where `K=2`
  already saturates the fabric — **A3 should re-race L=8 vs L=16 at 1 MiB with more reps**
  before drawing a conclusion.

## Zero-counts control — fixed per-block cost (no TPUTs)

| payload B/peer | L=1 (µs) | L=2 | L=4 | L=8 | L=16 | Δ L=8 | Δ L=16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 960 (zero) | 7 468.3 | 7 500.2 | 7 504.5 | 7 579.4 | 7 759.6 | +1.49 % | +3.90 % |

K2's same cell was **+7.6 %** at L=8 — the K3 counts-pull ownership change (one block per rank
pulls all `P` words instead of all `B` blocks pulling them) removed most of that fixed cost.

## Two harness catches worth keeping (both fixed here)

1. **`CalAllToAllVBlocks` must not cap at `2P`.** The harness's Python model did
   (`B = 2P when L ≥ 2P`), so for EP4/`L=16` it sized the signal window `[4,8]` while the real
   entry admits `B = (L//P)·P = 16` and enforces `signal_stride ≥ B` — every L=16 dispatch was
   rejected with `INVALID_ARGS` (all ten L=16 cells failed deterministically; the entry's
   `rt_report_fatal` path). Fixed in `78…a53dc` (mirrors the entry exactly; full plan-110
   matrix still reproduces; generates `[4,16]`, and the L=16 cells then pass). Prior matrix
   points were all `L ≤ 2P`, which is why the stale model had survived.
2. **Campaign hygiene**: pin `A2AV_DEV4` (the wrapper defaults EP4 to `0,1,2,3`, not the
   standing 4–7) and keep `A2AV_OUT` **absolute** (the harness subprocess runs with
   `cwd=/opt/pypto`, so a relative path writes the JSONs under `/opt/pypto/...` and every
   attempt shows as `rc=0, no json` "failure").

## Reproduce

```bash
cd /opt/pypto-profiling
A2AV_RESUME=1 A2AV_DEV4=4,5,6,7 \
A2AV_OUT=/opt/pypto-profiling/reports/issue-2521-k3-lane-scaling-2026-09-23 \
A2AV_EPS=4 A2AV_IMPLS=managed-host \
A2AV_CORE_NUMS=1,2,4,8,16 A2AV_REPS=2 \
A2AV_PAYLOADS=0,16384,262144,1048576 A2AV_EXTRA_PATTERNS=zero \
python collectives/alltoallv_a1.py
python collectives/analyze_scaling.py reports/issue-2521-k3-lane-scaling-2026-09-23
```

## Next (hand-off to A3)

- Re-race **L=8 vs L=16** at 1 MiB with ≥5 reps; also 512 KiB to find where the curve turns.
- The EP8/EP16 sweep points remain owed (this box is EP4-only); EP8/`B=16` = 2 lanes/peer is
  the RFC's own worked example.
- Correctness-side narrowing (receiver waits only its `K` columns) is deferred — consumers
  currently wait all admitted columns, a safe superset.
