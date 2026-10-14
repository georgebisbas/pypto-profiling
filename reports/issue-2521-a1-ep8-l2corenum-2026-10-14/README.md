# EP8 `all_to_all_v` — HOST vs L2 at `L=1` and `L=8` (first L2 multi-lane measurement)

**Date:** 2026-10-14 · **Box:** `hng-atlas02` (192.168.150.12), 8× Ascend 910B2 · **Queue task:**
`task_20261014_085749_20485986058` (8/8 cards, exit=0, locks released) · **Container:** `cann9-verify`
· **pypto:** `verify-merge-2976` @ `d2551caa` (has **K3 #2900** `be5168bb` and **O2a #2971**
`fe651f29`) · **Harness:** `/opt/pypto/tests/st/distributed/collectives/all_to_all_v_benchmark.py`,
sha256 `dcfc2337f9534306…` (== plan-129 branch `a2av-harness-l2-core-num` @ `22fc5886`)
· **Driver:** `collectives/alltoallv_a1.py` with `A2AV_HARNESS` pointed at the in-tree harness.

## Why this run

Plan 129 relaxed the A1 harness gate so the **CHIP/L2 rail** can take `L>1` (it was pinned at
`core_num=1` while K2 admitted `L>1` on the HOST rail only). The L2 multi-lane path had **never**
been measured on hardware. This is the first measurement of it, at EP8.

> Scope: a **deliberately bounded first cut**, not the official curve. 64 KiB/peer, uniform counts,
> extra patterns disabled, 3 warmup + 20 timed rounds (timing-slot) + 4 swimlane rounds, one case per
> (impl, L). The official methodology is 5 warmup / 100 measured.

## Results — EP8, 64 KiB/peer, `uniform`, persistent

| impl | L | metric | kernel p50 | host slot p50 | egress Gbit/s/rank | duplex | dispatches |
|---|---|---|---|---|---|---|---|
| managed-host | 1 | `timing_slot` | 6361.3 µs | 6361.3 µs | 0.558 | 1.12 | 3 |
| managed-host | 8 | `timing_slot` | 6179.4 µs | 6179.4 µs | 0.580 | 1.16 | 3 |
| managed-l2 | 1 | `aicore_gang_span` | 1209.6 µs | 4687.4 µs | 3.035 | 6.07 | 1 |
| **managed-l2** | **8** | `aicore_gang_span` | **186.0 µs** | 3513.1 µs | **19.75** | **39.5** | 1 |

4/4 cells `ok`, `n_fail=0`, no retries consumed, no Fabric poison.

## Findings

1. **The L2 multi-lane path works and is the fastest configuration this harness has measured here.**
   `managed-l2 L=8` = **186.0 µs** kernel p50, **19.75 Gbit/s/rank** egress (39.5 duplex).
   `launched_B = expected_B = 8` (the compile-time fold — the L2 rail still has no device-side DFX,
   see caveat 1).
2. **Multi-lane is worth 6.5× on the L2 rail but ~nothing on the HOST rail.** L2 goes 1209.6 → 186.0 µs
   (6.5×); HOST goes 6361.3 → 6179.4 µs (**2.9%**, inside run-to-run noise). At this payload the HOST
   rail's cost is not in the lane width.
3. **Cross-rail, at equal `L=1` the L2 rail is 5.3× faster** (1209.6 vs 6361.3 µs). `L2 @ L=8` is
   **34×** faster than the `HOST @ L=1` baseline. Both rails dispatch once vs three times.
4. **On the host-slot view the multi-lane win almost disappears**: the L2 slot only moves
   4687 → 3513 µs (25%) while the kernel moves 6.5×. At `L=8` the AIV gang span is **~5% of the slot**,
   i.e. the host view is now almost entirely dispatch overhead. Any "A2AV is X µs" claim must say which
   of the two columns it means.

## Caveats — read before quoting

1. **The two rails do not report the same metric.** `managed-host` reports `metric: timing_slot` with a
   *single value replicated across all eight ranks* (`rank_spread_us: 0.0` is an artifact); `managed-l2`
   reports real per-rank `aicore_gang_span` (`aiv_rounds_captured: 4`). This is the "HOST rail has no
   device-side metric" gap the A1 programme has carried since E3.2 — plan 129 predicted it would block a
   like-for-like §10 ratio, and it does.
2. **`kernel_p50` is a fastest-rank statistic, not a collective completion time.** At L2/`L=8` the rank
   spread is **~4.5 ms**: rank 7 = 186 µs, rank 0 ≈ 4.4 ms. End-to-end you are bounded by the slowest
   rank. The slowest rank costs ~4.2–4.4 ms at **both** `L=1` and `L=8`, so multi-lane is mostly
   accelerating the fast ranks.
3. **Bounded sample** (20 rounds, one payload, one pattern) — do not treat as the official curve.
4. **Unqueued history note:** unlike the earlier A1 campaigns, this run *was* queue-isolated
   (8-card exclusive lock via `task-submit`), so the numbers are not shared-box-contended. See
   `../npu-queue-runbook.md` §10.

## HCCL comparison

See [`HCCL_VS_PYPTO_A2AV.md`](HCCL_VS_PYPTO_A2AV.md) — an `HcclAlltoAllV` baseline measured under the
same EP8 / 64 KiB / 20-round conditions (new `--op alltoallv` in `collectives/hccl_bench.cc`), with
figures under [`reports/figures-2026-10-14/`](../../figures-2026-10-14/). Headline: HCCL's host-visible
time is **558 µs** vs PyPTO's best **3513 µs (6.3×)**, even though PyPTO's *device* span at `L=8`
(186 µs) is below HCCL's own wall.

## Reproduce

```bash
# inside a queue-joined container, from the pypto-profiling checkout
A2AV_HARNESS=/opt/pypto/tests/st/distributed/collectives/all_to_all_v_benchmark.py \
A2AV_OUT=<abs>/reports/issue-2521-a1-ep8-l2corenum-2026-10-14 \
A2AV_EPS=8 A2AV_IMPLS=managed-l2,managed-host A2AV_CORE_NUMS=1,8 \
A2AV_PAYLOADS=65536 A2AV_ROUNDS=20 A2AV_WARMUP=3 A2AV_SWIMLANE_ROUNDS=4 \
A2AV_RETRIES=2 A2AV_COOLDOWN_S=30 A2AV_RESUME=1 A2AV_EXTRA_PATTERNS= \
PYTHONPATH=. python3 -u collectives/alltoallv_a1.py
```

Submit it with `task-submit --device auto --device-num 8 --max-time 3600 --timeout 0 --run '…'`.

## Artifacts

`summary.json` · `json/*.json` (per-cell) · `campaign_meta.json` · `hccl_a2av_ep8_65536.log` ·
`HCCL_VS_PYPTO_A2AV.md`. Raw `attempt_logs/` and `builds/` are gitignored (see
[`../README.md`](../../README.md) for the policy); they remain on the machine that ran the campaign.

## Side finding — the A2AV sim gate is broken (separate from the numbers)

Full dispatch under `--platform a2a3sim` fails for **all four** (impl, L) combinations:

```
__builtin_all_to_all_v__int8.cpp:289:47: error: 'ld_dev' was not declared in this scope
```

`ld_dev` is a CANN AIV builtin (declared in CANN's `kernel_scalar.h` /
`dav_c220/kernel_operator_reg_others_impl.h`), used by the emitter template
`python/pypto/runtime/builtins/collectives/all_to_all_v/templates/kernel.cpp.in:289`; the generated
translation unit's include set does not declare it. The failure comes from host `cc1plus`, i.e. the
**CPU-model / sim** build path — a2a3 hardware is unaffected (this run proves that).
`--smoke` and `--codegen-only` are green, so use those as the sim gate for this kernel until it is
fixed. This also means plan 129's "codegen cell matrix green" does **not** extend to a2a3sim dispatch.
