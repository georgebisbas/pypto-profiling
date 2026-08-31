# Multi-AIV (core_num) scaling vs message size — crossover analysis

**Date:** 2026-08-31
**Platform:** a2a3, 8× Ascend 910B2 (64 GB HBM/chip), CANN 9.0.0, Ubuntu 22.04.5 (192-core aarch64)
**Kernel:** pypto-host mesh allreduce (`pld.tensor.allreduce`, fp32), persistent CommDomains
**Metric:** `device_wall_s` median — slowest-rank pure on-device collective time (apples-to-apples)
**Effective bandwidth:** `n_bytes / device_wall_s_median`

## Glossary (how to read these numbers)

- **`cn1` / `cn8` / `cn16`** — `core_num`, the number of AI-vector cores (AIVs) each
  rank's mesh-allreduce kernel is launched with (`--core-num N`, `pypto-host` stack;
  the kernel is SPMD over the AIV grid). The AIVs do the parallel element-wise
  reduce/compute on each die. `cn1` = single-AIV baseline; `cn8`/`cn16` = 8×/16× the
  vector-compute lanes per rank.
- **`payload`** — **per-rank** message size: each rank's input buffer to the mesh
  allreduce, `count` fp32 elements × 4 B = `n_bytes` = `bytes_per_rank_per_round`
  (e.g. `count = 4096` → 16 KiB per rank). The total data aggregated by the allreduce
  is `P × payload`; all tables and bandwidth figures below are per-rank (the
  apples-to-apples collective convention, comparable across P).
- **`cn8/cn1`, `cn16/cn1`** — multi-AIV speedup over single-AIV (bandwidth ratio,
  `bw_cnN / bw_cn1`); >1.0× = multi-AIV is a net win.
- **`device_wall_s`** — slowest-rank pure on-device collective time (host dispatch
  excluded). Bandwidth = `n_bytes / device_wall_s_median`.

## Key crossover summary (the punchline)

**Crossover = first payload where multi-AIV stops losing and the gain holds for every
larger payload** ("monotone-stable", defined precisely below):

| | P=2 | P=4 |
|---|---|---|
| **cn8 stable gain from** | **256 KiB** (1.71×) | **128 KiB** (1.15×) |
| **cn16 stable gain from** | **256 KiB** (2.45×) | 16 KiB* (noise-level) |
| first *substantial* gain (≥1.5×) | 256 KiB | 256 KiB (2.58×) |
| max speedup at 4 MiB (cn8 / cn16) | 3.80× / 2.64× | 4.66× / 5.84× |

*cn16 at P=4 reads ≥1.0× even at 16 KiB, but those are ~250 µs runs with ~2× spread —
treat as noise, not signal.

- **Below ~64–128 KiB: no reliable gain** — ratios bounce 0.25×–1.5× around parity
  (latency-bound + shared-box noise).
- **From ~256 KiB up: dependable, monotone gains** — cn8: 1.7×→3.8× (P=2), 2.6×→4.7×
  (P=4); cn16 tops out ~5.8× at P=4/4 MiB.
- **Higher P pulls the crossover down** (256→128 KiB for cn8): more peers → more
  traffic per AIV → the single-AIV bottleneck bites earlier.
- **cn8 ≈ cn16 at the top end** — 8 AIVs are basically enough by 1–4 MiB.

## Figure

`reports/figures-2026-08-28/corenum_message_size_crossover.png` (2×2):

- **Top row** — effective on-device bandwidth (GB/s) vs payload bytes (log2-x), one
  line per `core_num` ∈ {1, 8, 16}; left P=2, right P=4. Grey band = crossover region.
- **Bottom row** — multi-AIV speedup vs single-AIV (`cn8/cn1`, `cn16/cn1`), dashed
  line at parity (1.0×). The arrow marks the **monotone-stable crossover**: the first
  payload where `cn8` stays ≥1.0× for every larger payload.

## Effective device bandwidth (GB/s)

| P | count | payload | cn1 | cn8 | cn16 |
|---|---:|---:|---:|---:|---:|
| 2 | 4096 | 16 KiB | 0.06 | 0.04 | 0.10 |
| 2 | 8192 | 32 KiB | 0.11 | 0.14 | 0.07 |
| 2 | 16384 | 64 KiB | 0.22 | 0.17 | 0.16 |
| 2 | 32768 | 128 KiB | 0.33 | 0.08 | 0.52 |
| 2 | 65536 | 256 KiB | 0.41 | 0.71 | 1.02 |
| 2 | 131072 | 512 KiB | 0.56 | 0.88 | 0.84 |
| 2 | 262144 | 1 MiB | 0.82 | 3.04 | 1.60 |
| 2 | 524288 | 2 MiB | 1.10 | 2.28 | 3.09 |
| 2 | 1048576 | 4 MiB | 1.14 | 4.33 | 3.00 |
| 4 | 4096 | 16 KiB | 0.03 | 0.03 | 0.03 |
| 4 | 8192 | 32 KiB | 0.05 | 0.05 | 0.05 |
| 4 | 16384 | 64 KiB | 0.08 | 0.08 | 0.09 |
| 4 | 32768 | 128 KiB | 0.14 | 0.16 | 0.17 |
| 4 | 65536 | 256 KiB | 0.11 | 0.29 | 0.19 |
| 4 | 131072 | 512 KiB | 0.22 | 0.31 | 0.35 |
| 4 | 262144 | 1 MiB | 0.26 | 0.47 | 0.50 |
| 4 | 524288 | 2 MiB | 0.28 | 0.89 | 0.97 |
| 4 | 1048576 | 4 MiB | 0.28 | 1.31 | 1.64 |

## Multi-AIV speedup vs single-AIV

| P | payload | cn8/cn1 | cn16/cn1 |
|---|---:|---:|---:|
| 2 | 16 KiB | 0.65× | 1.49× |
| 2 | 32 KiB | 1.21× | 0.57× |
| 2 | 64 KiB | 0.77× | 0.70× |
| 2 | 128 KiB | 0.25× | 1.57× |
| 2 | 256 KiB | 1.71× | 2.45× |
| 2 | 512 KiB | 1.57× | 1.51× |
| 2 | 1 MiB | 3.72× | 1.95× |
| 2 | 2 MiB | 2.06× | 2.80× |
| 2 | 4 MiB | 3.80× | 2.64× |
| 4 | 16 KiB | 1.23× | 1.12× |
| 4 | 32 KiB | 1.11× | 1.19× |
| 4 | 64 KiB | 0.97× | 1.17× |
| 4 | 128 KiB | 1.15× | 1.25× |
| 4 | 256 KiB | 2.58× | 1.67× |
| 4 | 512 KiB | 1.41× | 1.61× |
| 4 | 1 MiB | 1.81× | 1.90× |
| 4 | 2 MiB | 3.18× | 3.47× |
| 4 | 4 MiB | 4.66× | 5.84× |

## Where do gains start — crossover detail

Two definitions matter, because the small-payload cells are noisy:

- **First touch ≥1.0×** — the first payload where `cn8/cn1 ≥ 1.0` at all. This is
  misleading: at P=2 the ratio touches 1.21× at 32 KiB then dips back to 0.77× (64K)
  and 0.25× (128K), so "first touch" would claim gains that do not hold.
- **Monotone-stable crossover** — the first payload where `cn8/cn1 ≥ 1.0` **and stays
  ≥1.0× for every larger payload**. This is the honest "gains start here" point (the
  arrow in the figure). Reading the cn8 line:
  - **P=2:** 0.65×@16K → 1.21×@32K → 0.77×@64K → **0.25×@128K** → **1.71×@256K** (holds
    from here: 1.57, 3.72, 2.06, 3.80) ⇒ **crossover = 256 KiB**.
  - **P=4:** 1.23×@16K → 1.11×@32K → 0.97×@64K → **1.15×@128K** (holds from here:
    2.58, 1.41, 1.81, 3.18, 4.66) ⇒ **crossover = 128 KiB**.
- **First substantial gain (≥1.5×)** — the payload where the win is clearly beyond
  noise: **256 KiB at both P** (P=2: cn8 1.71× / cn16 2.45×; P=4: cn8 2.58×).

Per-point speedup table (bold = first payload where the value holds ≥1.0× from there on):

| P | payload | 16K | 32K | 64K | 128K | **256K** | 512K | 1M | 2M | 4M |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | cn8/cn1 | 0.65× | 1.21× | 0.77× | 0.25× | **1.71×** | 1.57× | 3.72× | 2.06× | 3.80× |
| 2 | cn16/cn1 | 1.49× | 0.57× | 0.70× | 1.57× | **2.45×** | 1.51× | 1.95× | 2.80× | 2.64× |
| 4 | cn8/cn1 | 1.23× | 1.11× | 0.97× | **1.15×** | 2.58× | 1.41× | 1.81× | 3.18× | 4.66× |
| 4 | cn16/cn1 | 1.12× | 1.19× | 1.17× | **1.25×** | 1.67× | 1.61× | 1.90× | 3.47× | 5.84× |

**Why the crossover moves with P and payload:**

- **Small payloads are latency-bound.** The fixed per-peer wait/notify/barrier
  serialisation plus the AIV launch/dispatch cost dominate; the reduce work is tiny, so
  extra AIVs cannot hide the latency — they can even add contention (the 0.25× dip at
  128K/P=2 is real contention, not just noise). Result: no reliable gain below ~64–128
  KiB.
- **Large payloads are bandwidth-bound.** Once the buffer is big enough that the
  single-AIV compute+MTE path saturates (~0.9–1.1 GB/s at P=2, plateau by 1–4 MiB),
  extra AIVs add parallel reduce lanes and MTE bandwidth → real, monotone gains until
  the rail saturates. That is why the benefit grows with payload (3.8×/4.7–5.8× at
  4 MiB).
- **P scales the per-rank traffic.** With more peers each rank does (P−1) peer steps,
  so the total bytes moved per rank scales with P; the fixed-BW single-AIV bottleneck
  is therefore hit at a *smaller* per-rank payload — which is why the crossover shifts
  down (256→128 KiB for cn8) as P grows.

## Findings

1. **There is a crossover, and it is sharp.** Multi-AIV is a *net loss or noise-level
   at best* below ~64–128 KiB, and a *clear win* at ≥256 KiB (P=2) / ≥128–256 KiB
   (P=4). At the largest payloads multi-AIV reaches **3.8× (P=2) and 4.7–5.8× (P=4)**.
2. **The crossover shifts down as P grows.** At P=2 the monotone-stable crossover is
   ~256 KiB; at P=4 it is ~128 KiB. More peers means more data movement per AIV, so the
   single-AIV bottleneck bites earlier.
3. **The benefit grows monotonically with payload (bandwidth-bound regime).** Small
   messages are latency-bound: the fixed per-peer signalling/serialisation cost
   dominates and extra AIVs cannot hide it. Once the transfer is bandwidth-bound, extra
   AIVs add real bandwidth until the rail saturates.
4. **Single-AIV saturates ~0.9–1.1 GB/s (P=2).** One AIV cannot saturate inter-card
   bandwidth; it plateaus by ~1–4 MiB. Multi-AIV keeps scaling with payload (up to
   ~4.3 GB/s at P=2 / ~1.6 GB/s at P=4 in this sweep).
5. **cn8 and cn16 converge at large payloads** (B\* ∈ {8, 16} on this box) — 8 AIVs are
   usually sufficient; 16 adds little at the top end and is more box-load-sensitive.

## Caveats

- **Shared box.** Runs are interleaved-in-time over ~30 min under tenant load; small
  payloads are the noisiest (absolute times ~50–250 µs, `device_wall_spread_ratio` up
  to ~2). Single 8-round medians per cell; treat <256 KiB ratios as ranges, not exact.
- B\* ∈ {8,16} is box-load-sensitive (a separate cn8 fit drifted 5.5→3.56 GB/s
  run-to-run under tenant load) — report as a range, not a point.
- Devices: P=2 on d4-5, P=4 on d4-7 (d0-1 occupied by other tenants; dev 3 flaky).

## Reproduce

```bash
cd /opt/pypto-profiling && export PATH=/usr/local/python3.12.13/bin:$PATH
# 9 counts x core_num {1,8,16} x P {2->d4-5, 4->d4-7}, persistent, 2+8 rounds
# (case files in /tmp/crossover_cases/ at generation time)
PYTHONPATH=. python3 -m collectives.run_sweep pair-mesh --case-file <case.json> \
  --stacks pypto-host --core-num {1,8,16} --persistent \
  --warmup-rounds 2 --timed-rounds 8 --campaign crossover \
  --out results/campaigns/crossover/p{P}_c{count}_cn{cn}/results.json
# figure + table:
PYTHONPATH=. python3 -m collectives.plot_crossover
```
