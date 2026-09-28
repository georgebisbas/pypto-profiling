# K3 (PR #2900) — external reproduction of the claimed speedups

**Date:** 2026-09-28 · **Box:** hng-atlas01, 8× Ascend NPUs (a2a3) · **Kernel tree:** `cf136e73` (`feat/all-to-all-v-multiaiv-lanes` head, includes the post-campaign 32-byte lane-alignment fix) · **Harness:** `pypto-profiling/collectives/alltoallv_a1.py` (same protocol as the claimed campaigns; metric `host_timing_slot_mean_us`).

## Why

PR #2900 claims (PR body + comment 5831311329): EP4 1 MiB/peer **−47.7 % (L=8) / −50.4 % (L=16)**; EP8 **−52.4 % / −58.9 %**; egress **3.14 / 4.76 Gbit/s per rank**; the owed items were an **EP8/EP16 sweep** (EP8 since measured, EP16 still owed), the **≥5-rep L=8 vs L=16 re-race**, and optionally the **512 KiB turning point**.

## Method

- **EP4** (NPUs auto-assigned, 4): `managed-host`, `L ∈ {1,2,4,8,16}`, payloads `{16 KiB, 256 KiB, 512 KiB, 1 MiB}` + `zero@24960` control, `--rounds 100 --warmup 5 --swimlane-rounds 8`, 2 reps — protocol parity with the 09-23/09-24 campaigns.
- **EP8** (8 NPUs): `L ∈ {1,8,16}`, payloads `{16 KiB, 256 KiB, 1 MiB}` + `zero@24960`, `--rounds 30 --warmup 3 --swimlane-rounds 4`, `SIMPLER_COMM_FORCE_IPC=1`, 2 reps, retries=3/heal — parity with the 09-24/25 EP8 campaign.
- Deltas are pairwise vs the `L=1` arm, aggregated as mean-of-reps. Comparator: `k3_compare.py` (validated: it reproduces the claimed run1-vs-run2 tables exactly).

## Results — EP8 (claimed vs reproduced)

| payload | L=8 Δ (claimed → repro) | L=16 Δ | egress L=16 |
| --- | --- | --- | --- |
| 16 KiB | +2.0 % → **−0.4 %** (noise; report predicted sign flip) | +6.1 % → **−0.8 %** | 0.17 → 0.17 |
| 256 KiB | −33.4 % → **−33.9 %** ✔ | −35.8 % → **−37.7 %** ✔ | 2.02 → 2.00 |
| 1 MiB | −52.4 % → **−53.5 %** ✔ | −58.9 % → **−56.3 %** ✔ (within run spread) | 4.76 → 4.44 |
| zero control | −6.3 % → +2.9 % (same small fixed-cost region) | −6.8 % → −4.2 % | — |

24/24 cells OK. Absolute `L=1` 1 MiB: 30 025.6 → 30 293.6 µs (0.9 %).

## Results — EP4 (claimed run 2 vs reproduced)

| payload | L=2 | L=4 | L=8 | L=16 |
| --- | --- | --- | --- | --- |
| 16 KiB | +1.7 → **−6.8** | −0.7 → **−2.9** | −0.8 → **+1.1** | −5.2 → **−4.6** |
| 256 KiB | −15.7 → **−15.6** ✔ | −23.6 → **−24.3** ✔ | −28.7 → **−29.2** ✔ | −30.5 → **−30.6** ✔ |
| 1 MiB | −23.7 → **−20.9** | −37.4 → **−37.8** ✔ | −47.7 → **−46.2** | −50.4 → **−52.0** |
| zero control | +4.1 → +2.3 | +6.3 → +4.5 | +4.9 → +0.8 | +3.4 → −0.1 |

50/50 cells OK. Egress 1 MiB: L=1 1.61 → L=16 **3.36** Gbit/s per rank (claimed 1.54→3.14–3.15; ours ~2–7 % higher at both ends).

**New — the owed 512 KiB turning point** (never measured in the claimed campaigns):

| payload | L=1 | L=2 | L=4 | L=8 | L=16 |
| --- | --- | --- | --- | --- | --- |
| 512 KiB Δ | — | −17.4 % | −30.9 % | −40.7 % | −43.6 % |

The curve turns between 256 KiB and 512 KiB; `L=8 → L=16` adds only −2.9 pp (plateau) at 512 KiB, vs −2.0 pp at 1 MiB (EP4) and −2.8 pp (EP8).

## Results — ≥5-rep re-race (L=8 vs L=16 @ 1 MiB/peer)

**EP4 (5 reps/arm, 15/15 cells OK):** the "L=8 vs L=16 tie" resolves — **L=16 ahead in 5/5 paired reps**.

| L | mean of 5 (µs) | per-rep (µs) | Δ vs L=1 |
| --- | --- | --- | --- |
| 1 | 16 044.3 | 16001.7 · 15807.0 · 16186.4 · 15806.6 · 16420.0 | — |
| 8 | 8 637.7 | 8966.2 · 8745.7 · 8906.9 · 8411.2 · 8158.8 | −46.2 % |
| 16 | **7 926.4** | 8135.0 · 7788.7 · 8265.1 · 7347.9 · 8095.2 | **−50.6 %** |

L=16 is ~8.3 % further than L=8, consistently across all five pairs (p≈3 % under a sign test) — the 2-rep campaigns' "one operating point, B≥8" reading was sampling noise.

**EP8 (5 reps/arm, 15/15 cells OK):** L=16 ahead in 4/5 paired reps.

| L | mean of 5 (µs) | per-rep (µs) | Δ vs L=1 |
| --- | --- | --- | --- |
| 1 | 30 548.1 | 30871.9 · 29800.1 · 30338.0 · 30395.8 · 31334.6 | — |
| 8 | 13 866.6 | 14893.8 · 13977.1 · 12656.3 · 13796.2 · 14009.5 | −54.6 % |
| 16 | **12 504.6** | 13136.9 · 12479.1 · 13018.1 · 11403.2 · 12485.4 | **−59.1 %** |

L=16 ~9.8 % further on the mean (the one flipped pair sits inside the ~10–15 % per-rep spread); egress L=16 mean **4.71 Gbit/s** (peak rep 5.15). The EP8-re-race `L=16` delta matches the claimed −58.9 % almost exactly.

**Re-race takeaway:** at 1 MiB/peer both boxes resolve toward **B=16 > B=8** (~8–10 % mean further, EP4 5/5 pairs, EP8 4/5) — the earlier "`L=8`/`L=16` tie / treat `B≥8` as one operating point" call was underpowered at 2 reps.

## Observations

- **Every headline claim reproduces** within the campaigns' own documented noise envelope (≤~3 pp on deltas at 1 MiB; small payloads are fixed-cost/noise-dominated and flip sign run-to-run, exactly as the reports warn).
- **The EP8 Fabric-V2 device fault is real and reproduces**: 4 hits across our runs (`comm_release_domain_windows failed with code -1`, rc=-11) on ≥256 KiB / zero cells — every one healed and passed on retry (`retries=3`, 30 s cooldown, EP4 heal smoke), exactly as documented. 0 cells lost.
- **EP16 remains outstanding** — this box has 8 NPUs; 16-rank points cannot be produced here.
- Repro artifacts (host): `slots-verify/k3-repro-ep8/`, `k3-repro-ep4/`, `k3-reprace-ep4-l816/`, `k3-reprace-ep8-l816/` (per-cell JSONs + summary.json + this comparison).

## Repro commands

```bash
# EP4 (4 cards; rounds=100 etc. are harness defaults)
A2AV_RESUME=1 A2AV_DEV4=0,1,2,3 A2AV_EPS=4 A2AV_IMPLS=managed-host \
A2AV_CORE_NUMS=1,2,4,8,16 A2AV_REPS=2 \
A2AV_PAYLOADS=16384,262144,524288,1048576 A2AV_EXTRA_PATTERNS=zero \
A2AV_OUT=<abs> python collectives/alltoallv_a1.py

# EP8 (8 cards)
A2AV_RESUME=1 A2AV_FORCE_IPC=1 A2AV_ROUNDS=30 A2AV_WARMUP=3 A2AV_SWIMLANE_ROUNDS=4 \
A2AV_EPS=8 A2AV_DEV8=0,1,2,3,4,5,6,7 A2AV_DEV4=0,1,2,3 A2AV_IMPLS=managed-host \
A2AV_OUT=<abs> A2AV_CORE_NUMS=1,8,16 A2AV_REPS=2 \
A2AV_PAYLOADS=16384,262144,1048576 A2AV_EXTRA_PATTERNS=zero \
A2AV_RETRIES=3 A2AV_COOLDOWN_S=30 A2AV_HEAL_EP=4 python collectives/alltoallv_a1.py
```

(Queue note: long campaigns need `MAX_TIME_HARD_CAP` raised on the task-submit invocation; default cap is 3600 s.)
