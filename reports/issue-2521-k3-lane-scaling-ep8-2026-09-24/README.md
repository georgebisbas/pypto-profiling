# K3 lane scaling — EP8 (8-rank) campaign (2026-09-24 → 2026-09-25)

*RFC #2521 K3 validation, 8-rank envelope complementing the EP4 campaigns
([2026-09-23](../issue-2521-k3-lane-scaling-2026-09-23/K3_LANE_SCALING_REPORT.md)
and [2026-09-24](../issue-2521-k3-lane-scaling-2026-09-24/README.md)):
EP8 on NPUs `0-7`, `--impl managed-host`, `L ∈ {1,8,16}`, payloads
`{16 KiB, 256 KiB, 1 MiB}` + `zero@24960` counts control, 2 reps — 24 cells.*

**Result: 24/24 cells OK, RC=0 — uniform protocol**: all cells at
`--rounds 30 --warmup 3` + `SIMPLER_COMM_FORCE_IPC=1` (VMM-IPC window export),
2 reps, `retries=3`, `cooldown=30 s`, EP4 heal smoke between retries (1 retry
total: `zero@24960 L=1`, failed attempt 1 → heal OK → attempt 2 passed). The
six 16 KiB cells were re-measured 2026-09-25 for uniformity; the original
100-round Fabric-protocol values are preserved as `summary_pre_uniform_redo.json`
/ `analysis_mixed_protocol.txt`.

Why 30 rounds: the documented EP8 Fabric-V2 device fault (507018/507014 → lane
poisoned → `comm_release_domain_windows failed with code -1` → rc=-11) fires
probabilistically in **long sessions with ≥256 KiB payloads** — repeated
100-round failures across both transports on 2026-09-24 (six consecutive
sessions recorded at the time; device faults seen on chips 6 and 4), and
~50 %/attempt failure at 100 rounds even on a verified-idle box on 09-25, while
30-round sessions pass reliably (24/24 cells here). Transport and contention are
excluded as causes: `FORCE_IPC` does not prevent it, and the box was
exclusively ours for the final data.

## Results (whole-program timing slot mean — `host_timing_slot_mean_us`, mean-of-2 per arm)

| payload B/peer | L=1 (µs) | L=8 (µs) | L=16 (µs) | Δ L=8 | Δ L=16 | egress L=1/8/16 (GB/s) |
|---|---|---|---|---|---|---|
| 16 KiB | 5473.2 | 5580.0 | 5806.0 | +2.0 % | +6.1 % | 0.168 / 0.164 / 0.158 |
| 256 KiB | 11312.7 | 7534.7 | 7260.1 | −33.4 % | −35.8 % | 1.30 / 1.95 / 2.02 |
| 1 MiB | 30025.6 | 14292.5 | 12333.9 | −52.4 % | −58.9 % | 1.96 / 4.12 / 4.76 |
| zero@24960 (control) | 13295.5 | 12464.4 | 12390.5 | −6.3 % | −6.8 % | — |

- **16 KiB**: fixed-cost dominated (≈13 ms floor) — no gain from lane
  parallelism (H1 flags the `L=8→16` step at +4 % over the noise band; read as
  a small fixed-cost penalty, not a trend — needs reps).
- **256 KiB**: gain saturates by `L=8` (−33 %) with a small further step to
  `L=16` (−3.6 % from L=8).
- **1 MiB**: still −13.7 % from `L=8` → `L=16` (EP4 saw an L=8/L=16 tie;
  EP8's ordering is clean here — still treat 8/16 as one operating point
  pending more reps).
- **Egress** peaks at **4.76 GB/s** (L=16, 1 MiB) vs EP4's 3.14 GB/s.
- **Zero-counts control** drops only −6 %: the block-lane structure slightly
  reduces the fixed per-block floor but cannot move payloads.

## Cross-check vs EP4 (relative gains vs L=1; EP4 ranges span its two campaigns)

| payload | EP4 Δ L=8 | EP8 Δ L=8 | EP4 Δ L=16 | EP8 Δ L=16 |
|---|---|---|---|---|
| 16 KiB | −0.7..−9.0 % | +2.0 % | −5.2..−14.1 % | +6.1 % |
| 256 KiB | −19.3..−28.7 % | −33.4 % | −25.1..−30.5 % | −35.8 % |
| 1 MiB | −47.7..−50.9 % | −52.4 % | −49.5..−50.4 % | −58.9 % |

Shape matches: same conclusions at 8 ranks, slightly larger relative gains
(8 peers to overlap). 16 KiB deltas stay at drift/fixed-cost scale (sign may
flip run-to-run — no throughput benefit either way).

## Method notes

- 2 reps/arm; pairwise deltas; ≤ ~3 % steps are noise, cross-run repeat spread
  ≤ ~5 % (see repeat runs in `summary.json`).
- Artifacts: `summary.json` (per-cell `rounds`/`attempt`/metrics — all rows
  `rounds=30`), `json/` (raw per-cell JSON), `campaign_meta.json`,
  `analysis_output.txt` (verbatim analyzer output), `attempt_logs/`, `heal/`;
  pre-redo mixed-protocol history: `summary_pre_uniform_redo.json`,
  `json_pre_uniform_redo/`, `analysis_mixed_protocol.txt`.
- Reproduce (this dir): `A2AV_RESUME=1 A2AV_FORCE_IPC=1 A2AV_ROUNDS=30
  A2AV_WARMUP=3 A2AV_SWIMLANE_ROUNDS=4 A2AV_EPS=8
  A2AV_DEV8=0,1,2,3,4,5,6,7 A2AV_IMPLS=managed-host A2AV_OUT=<dir>
  A2AV_CORE_NUMS=1,8,16 A2AV_REPS=2
  A2AV_PAYLOADS=16384,262144,1048576 A2AV_EXTRA_PATTERNS=zero A2AV_RETRIES=3
  A2AV_COOLDOWN_S=30 A2AV_HEAL_EP=4 python collectives/alltoallv_a1.py`
- Whole-program timing slot on a shared box; all final cells were measured
  2026-09-25 on a verified-idle box (8/8 NPUs free). The original 09-24
  session overlapped a colleague's Qwen-vLLM on NPUs 0–3 only for its tail —
  the ≥256 KiB failures predate that window and reproduce on an idle box.
