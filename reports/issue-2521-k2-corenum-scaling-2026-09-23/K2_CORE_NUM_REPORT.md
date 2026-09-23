# K2 `core_num` scaling baseline — `managed-host` all_to_all_v, EP4 on NPUs 4-7

**Date:** 2026-09-23 · **Branch under test:** `feat/all-to-all-v-multiblock-enable` (PR [hw-native-sys/pypto#2889](https://github.com/hw-native-sys/pypto/pull/2889), RFC #2521 K2) · **Against:** post-K1 `main` + K2.

**Question.** What do multiple launched blocks (K2's `B > 1`) deliver end-to-end today on the HOST rail?

**Headline.** No speedup — and none is expected from this PR. K2 launches and admits `B` blocks *correctly*, but every admitted block still performs the full pull/push (per-block work partitioning is the **K3 follow-up PR**). Measured `L=8` vs `L=1` deltas sit at −3.4 %…+7.6 % with **no monotonic trend** across payloads — i.e. environment-scale variation around a small fixed per-block cost, not a scaling curve.

## Method

- Harness: `pypto-profiling` A1 a2av campaign, extended for the K2 sweep (`A2AV_CORE_NUMS=1,8`; rail-aware gate allows `L>1` for `--impl managed-host` only; the signal window is sized `[nr, max(1,B)]`, matching the entry's `stride ≥ B` admission rule).
- Cell: `--impl managed-host`, EP4 on NPUs **4,5,6,7**; `uniform` counts; `--rounds 100 --warmup 5 --swimlane-rounds 8`; persistent windows; one compile per point.
- Interleaving: per payload, each rep runs as an adjacent `(L=1, L=8)` pair (r0, r1) so drift affects both sides of a pair.
- Metric: whole-program timing slot (`host_timing_slot_*`). The HOST rail's AIV swimlane is unavailable (A1 report §9.1 open gap), so the kernel cannot be isolated — the slot includes the single-AIV stage/consume and the L3 dispatch (~4–16 ms).
- DFX cross-check: device logs (`ascend/`) carry `requested_core_num=8 launched_core_num=8 active_lanes=8 nranks=4` for every L=8 call — the K2 path is what actually ran.

## Results

Per-cell aggregates (mean of the per-run slot mean; the two runs' values in parentheses):

| payload B/peer | L=1 (µs) | L=8 (µs) | Δ | egress L=1 → L=8 |
|---|---|---|---|---|
| 0 | 7674.7 (7528.7, 7820.7) | 7826.6 (7677.3, 7975.9) | **+2.0 %** | 0 → 0 GB/s |
| 16 384 | 4002.6 (3952.2, 4053.1) | 3867.6 (3778.7, 3956.4) | **−3.4 %** | 0.098 → 0.102 |
| 24 960 (zero counts) | 7543.9 (7548.9, 7539.0) | 8119.7 (8080.4, 8159.0) | **+7.6 %** | 0 → 0 |
| 262 144 | 6847.1 (6871.3, 6822.9) | 7234.6 (7184.9, 7284.2) | **+5.7 %** | 0.919 → 0.870 |
| 1 048 576 | 16134.3 (16111.2, 16157.5) | 16115.3 (16218.3, 16012.3) | **−0.1 %** | 1.560 → 1.562 |

Pairwise deltas (same-drift-window comparison; the decisive read):

| payload | r0 Δ | r1 Δ |
|---|---|---|
| 0 | +148.6 µs | +155.2 µs |
| 16 384 | −173.5 µs | −96.7 µs |
| 24 960 (zero) | +531.5 µs | +620.0 µs |
| 262 144 | +313.6 µs | +461.3 µs |
| 1 048 576 | +107.1 µs | −145.2 µs |

## Interpretation

- **No systematic `B`-dependent trend**: the sign flips between payloads (faster at 16 KB, slower at 256 KB, tied at 1 MB). With no work partitioning, `B` blocks redundantly transfer the same bytes, so any systematic cost should grow with payload — it does not, which points to environment drift dominating these sub-8 % deltas rather than a `B` effect.
- The no-payload cells isolate the per-block machinery (launch + wider barrier + redundant count pulls): `zero` counts at 24 960 B move nothing yet cost **+576 µs (+7.6 %)** at `B=8` — the clearest systematic number in the set, an upper bound on today's fixed overhead at this width/EP.
- Absolute scale: stage/consume are single-AIV and dominate the slot; even a perfect kernel speedup would move the whole-program number only modestly at these sizes. Kernel-level isolation needs the HOST AIV capture (A1 §9.1).
- **Where the gain will come from:** K3's per-block work partitioning (each block owning a disjoint slice, per-lane completion). This harness, the interleaved protocol, and `analyze_corenum.py` are ready to re-measure it directly.

## Environment notes

- Runs were unisolated (`task-submit` unavailable); shared 192-core host — within-run per-round spread reaches ±20 %, hence the pair-delta framing and run-level means.
- Campaign extras: `zero@24960` cells are included above; `single-hot` extra cells were still running when this report was written and are not analyzed here.

## Artifacts

Committed: `summary.json` (all analyzed cells OK), `campaign_meta.json`, `campaign_console.log`, `json/` (per-cell harness JSON), `analyze_corenum.py`, `K2_CORE_NUM_REPORT.md`.

On-disk only (too large to commit): `attempt_logs/` (238 MB), `builds/` (119 MB), `ascend/` (45 MB — device logs carrying the K2 DFX lines `requested_core_num=8 launched_core_num=8 active_lanes=8 nranks=4`).
