# HCCL `HcclAlltoAllV` vs PyPTO `all_to_all_v` — EP8, 64 KiB/peer

**Date:** 2026-10-14 · **Box:** `hng-atlas02` (192.168.150.12), 8× Ascend 910B2 · **Both sides
queue-isolated** · **pypto** `verify-merge-2976` @ `d2551caa` (K3 #2900 + O2a #2971) · **CANN**
9.0.0

| Side | Queue task | Command |
|---|---|---|
| PyPTO (4 configs) | `task_20261014_085749_20485986058` | `alltoallv_a1.py` → in-tree harness, `--peer-bytes 65536`, 20 rounds |
| HCCL | `task_20261014_093806_326442419312` (8/8 cards, 13 s, exit=0) | `hccl_bench --op alltoallv --dtype int8 --peer-bytes 65536 --timed-rounds 20` |

## The ruler (read this first)

The **only** column both stacks can be measured on with the current tooling is the **host-side wall
around one collective**:

- **HCCL** — `aclrtSynchronizeStream`-bounded wall around `HcclAlltoAllV`, per rank, 20 timed rounds.
- **PyPTO** — the harness's `host_timing_slot_p50_us`: wall clock around one `runtime.run(...)`.

PyPTO *additionally* exposes a device span (the AIV gang span). **HCCL exposes none here**, so that
column is shown for information but is never used to compute a ratio. Comparing PyPTO's AIV span to
HCCL's host wall would be a category error — it is the single easiest way to produce a flattering
and wrong "PyPTO beats HCCL" claim from this data.

## Results

| Config | host-slot p50 | vs HCCL | per-rank egress (remote payload) | device span |
|---|---|---|---|---|
| **HCCL `HcclAlltoAllV`** | **558 µs** | 1.00× | **6.55 Gbit/s** | — (not exposed) |
| pypto `managed-l2` L=8 | 3513 µs | 6.30× | 1.04 Gbit/s | 186 µs |
| pypto `managed-l2` L=1 | 4687 µs | 8.40× | 0.78 Gbit/s | 1210 µs |
| pypto `managed-host` L=8 | 6179 µs | 11.08× | 0.59 Gbit/s | — (timing_slot only) |
| pypto `managed-host` L=1 | 6361 µs | 11.40× | 0.56 Gbit/s | — (timing_slot only) |

HCCL detail: per-rank p50 spread 549–568 µs across the 8 ranks; slowest-rank mean 570.6 µs.

## What it means

1. **On the host-visible ruler, HCCL is 6.3–11.4× faster than every PyPTO configuration.** Even the
   best PyPTO setting (L2, `L=8`) leaves a 6.3× gap.
2. **The gap is dispatch, not the collective.** PyPTO's device-attributable time at L2/`L=8` is
   **186 µs — actually below HCCL's own 558 µs host wall** — but PyPTO pays ~3.3 ms of host-side
   remainder (staging + outer dispatch + sync) around it. That is a 5% duty cycle. See
   `figures-2026-10-14/a2av_ep8_decomposition.png`.
3. **This is the same conclusion the A1 programme reached against the AllReduce baseline**, now
   confirmed against an HCCL AllToAllV baseline instead of an inferred one.
4. **Multi-lane closes the *collective* gap, not the *dispatch* gap.** `L=8` takes L2's device span
   from 1210 → 186 µs (6.5×) while its slot only moves 4687 → 3513 µs (25%).

## How the HCCL baseline was built

The repo had **no** AllToAllV baseline — `hccl_bench.cc` implemented `HcclAllReduce` only. An
`--op alltoallv` was added to the same helper (`HcclAlltoAllV`, `HCCL_DATA_TYPE_INT8`, 20 timed
rounds, same `HCCL_TIMED per_rank=[…]` output so `metrics.py` parsing is unchanged).

**Payload matched deliberately.** The PyPTO harness reports `remote_bytes_per_rank` — at EP8 it
counts 7×64 KiB = 458 752 B, i.e. **self-traffic excluded**, even though the `uniform` count matrix
fills the diagonal. The HCCL call therefore sets `sendcounts[rank] = 0` and `sendcounts[other] =
65536`, so both sides move the same remote volume. `--include-self` flips it if you want the
diagonal counted. Egress uses that same 458 752 B on both sides.

## Caveats

- **One payload, one pattern, 20 rounds.** Not the official 100-round §8.3 curve; treat the ratio as
  indicative, not final.
- **HCCL has no device-side metric** in this harness. The attribution for HCCL ("its whole bar is the
  collective") is an assumption, not a measurement — HCCL's launch overhead is inside the 558 µs.
- **PyPTO's device span is a fastest-rank statistic.** At L2/`L=8` the rank spread is ~4.5 ms
  (rank 7 = 186 µs, rank 0 ≈ 4.4 ms), so "186 µs" is not a collective completion time. HCCL's
  numbers here are per-rank walls, so the two are only comparable through the slot column.
- **HCCL ran with the process group already warm** (`HcclCommInitRootInfo` once per rank, 3 warmup
  rounds) — as did PyPTO. Setup is excluded on both sides (`setup_s` = 3.0 s for HCCL).
- ⚠️ **Process note:** validating the new code path used one *unqueued* 2-card run before the queued
  EP8 run. It should have gone through `task-submit` too. No other user was on the box (0/8), but it
  was still a violation of the repo's own rule in `../npu-queue-runbook.md` §1.

## Figures (`reports/figures-2026-10-14/`)

| File | What it shows |
|---|---|
| `a2av_ep8_host_slot.png` | log-scale host-slot p50, HCCL vs the 4 PyPTO configs |
| `a2av_ep8_egress.png` | effective per-rank egress on the matched remote payload |
| `a2av_ep8_decomposition.png` | the money chart: device-attributable vs host remainder |

Regenerate with:

```bash
python3 collectives/plot_hccl_vs_pypto_a2av.py \
  --campaign reports/issue-2521-a1-ep8-l2corenum-2026-10-14 \
  --hccl-log reports/issue-2521-a1-ep8-l2corenum-2026-10-14/hccl_a2av_ep8_65536.log \
  --out reports/figures-2026-10-14
```
