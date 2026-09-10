# RFC #2521 A1 AllToAllV baseline — UB-fix harness

**Date:** 2026-09-09 (UTC `2026-09-09T18:22:26.274832+00:00`)  
**Status:** **72 / 72 ok** (4 flaky points retried once; final `n_fail=0`)  
**Output:** `/opt/pypto-profiling/reports/issue-2521-a1-alltoallv-ubfix-2026-09-09`

## What changed vs prior baseline

Prior run under `issue-2521-a1-alltoallv-baseline-2026-09-09` failed at C′ sizes
≥ 256 KiB with:

```
ValueError: Vec buffer usage (N bytes) exceeds platform limit (188416 bytes)
```

Harness fix in `tests/st/distributed/collectives/all_to_all_v_benchmark.py`:
stage/consume copy each INT8 row through `pl.unroll(0, width, tile_cols)` with
`STAGE_TILE_BYTES = 16 KiB` (under a2a3 VEC UB). This is **dispatch fill/drain
only** — not the collective kernel and not PR #2690 (InCore composite staging).

## Measurement conditions

- Roadmap: https://github.com/hw-native-sys/pypto/issues/2521#issuecomment-5495422542
- `persistent=True`, `reset_persistent_windows=False`
- Warmup `5`, timing-slot rounds `100`,
  swimlane rounds `8`
- `requested_L=1` (O2 not implemented)
- Devices: EP2=`0,1`, EP4=`0,1,2,3`
- Official L2 metric: swimlane AIV gang span (fastest-rank p50)
- HOST metric: timing_slot (no AIV gang name for builtin on HOST rail yet)
- Driver: `collectives/alltoallv_a1.py`

## Uniform payload sweep

### EP=2 `managed-l2` (uniform)

| peer_B | metric | kernel_p50_µs | slot_p50_µs | egress_Gbps |
|---:|---|---:|---:|---:|
| 0 | `aicore_gang_span` | 3.3 | 3867.8 | 0.0 |
| 32 | `aicore_gang_span` | 4.1 | 1672.9 | 0.1 |
| 128 | `aicore_gang_span` | 4.0 | 1757.2 | 0.3 |
| 1024 | `aicore_gang_span` | 6.5 | 1690.8 | 1.2 |
| 4096 | `aicore_gang_span` | 16.2 | 1680.4 | 1.5 |
| 8192 | `aicore_gang_span` | 29.1 | 1734.4 | 2.3 |
| 16384 | `aicore_gang_span` | 55.3 | 1807.7 | 2.2 |
| 24576 | `aicore_gang_span` | 81.2 | 1758.8 | 2.4 |
| 24960 | `aicore_gang_span` | 107.0 | 3065.2 | 0.7 |
| 32768 | `aicore_gang_span` | 107.1 | 1955.0 | 2.4 |
| 49152 | `aicore_gang_span` | 165.2 | 1955.1 | 2.1 |
| 65536 | `aicore_gang_span` | 210.2 | 1975.2 | 2.5 |
| 131072 | `aicore_gang_span` | 421.4 | 2352.9 | 2.3 |
| 262144 | `aicore_gang_span` | 835.3 | 2804.6 | 2.4 |
| 524288 | `aicore_gang_span` | 1665.8 | 3881.7 | 2.5 |
| 1048576 | `aicore_gang_span` | 3402.5 | 6584.7 | 2.4 |

### EP=2 `managed-host` (uniform)

| peer_B | metric | kernel_p50_µs | slot_p50_µs | egress_Gbps |
|---:|---|---:|---:|---:|
| 0 | `timing_slot` | 5511.3 | 5511.3 | 0.0 |
| 32 | `timing_slot` | 3605.3 | 3605.3 | 0.0 |
| 128 | `timing_slot` | 3536.8 | 3536.8 | 0.0 |
| 1024 | `timing_slot` | 3707.3 | 3707.3 | 0.0 |
| 4096 | `timing_slot` | 3637.5 | 3637.5 | 0.0 |
| 8192 | `timing_slot` | 3547.9 | 3547.9 | 0.0 |
| 16384 | `timing_slot` | 3521.2 | 3521.2 | 0.0 |
| 24576 | `timing_slot` | 3637.6 | 3637.6 | 0.0 |
| 24960 | `timing_slot` | 5732.1 | 5732.1 | 0.0 |
| 32768 | `timing_slot` | 3570.7 | 3570.7 | 0.0 |
| 49152 | `timing_slot` | 3815.4 | 3815.4 | 0.1 |
| 65536 | `timing_slot` | 3812.6 | 3812.6 | 0.1 |
| 131072 | `timing_slot` | 4362.5 | 4362.5 | 0.2 |
| 262144 | `timing_slot` | 4774.4 | 4774.4 | 0.3 |
| 524288 | `timing_slot` | 6246.6 | 6246.6 | 0.6 |
| 1048576 | `timing_slot` | 7879.3 | 7879.3 | 0.9 |

### EP=4 `managed-l2` (uniform)

| peer_B | metric | kernel_p50_µs | slot_p50_µs | egress_Gbps |
|---:|---|---:|---:|---:|
| 0 | `aicore_gang_span` | 78.5 | 5944.9 | 0.0 |
| 32 | `aicore_gang_span` | 9.5 | 2394.2 | 0.1 |
| 128 | `aicore_gang_span` | 9.4 | 2170.2 | 0.1 |
| 1024 | `aicore_gang_span` | 15.3 | 2193.9 | 1.6 |
| 4096 | `aicore_gang_span` | 40.7 | 2303.2 | 2.4 |
| 8192 | `aicore_gang_span` | 72.1 | 2217.7 | 2.3 |
| 16384 | `aicore_gang_span` | 137.4 | 2508.2 | 2.9 |
| 24576 | `aicore_gang_span` | 204.9 | 2553.3 | 2.9 |
| 24960 | `aicore_gang_span` | 268.2 | 6648.8 | 1.4 |
| 32768 | `aicore_gang_span` | 268.9 | 2619.0 | 2.9 |
| 49152 | `aicore_gang_span` | 410.2 | 2703.5 | 2.2 |
| 65536 | `aicore_gang_span` | 534.7 | 2907.6 | 2.9 |
| 131072 | `aicore_gang_span` | 1067.3 | 3784.2 | 2.8 |
| 262144 | `aicore_gang_span` | 2402.3 | 5398.7 | 2.6 |
| 524288 | `aicore_gang_span` | 4231.0 | 7967.1 | 2.9 |
| 1048576 | `aicore_gang_span` | 8564.4 | 13740.9 | 2.8 |

### EP=4 `managed-host` (uniform)

| peer_B | metric | kernel_p50_µs | slot_p50_µs | egress_Gbps |
|---:|---|---:|---:|---:|
| 0 | `timing_slot` | 7400.3 | 7400.3 | 0.0 |
| 32 | `timing_slot` | 4002.7 | 4002.7 | 0.0 |
| 128 | `timing_slot` | 4067.6 | 4067.6 | 0.0 |
| 1024 | `timing_slot` | 3931.0 | 3931.0 | 0.0 |
| 4096 | `timing_slot` | 3909.1 | 3909.1 | 0.0 |
| 8192 | `timing_slot` | 3986.9 | 3986.9 | 0.0 |
| 16384 | `timing_slot` | 4184.9 | 4184.9 | 0.1 |
| 24576 | `timing_slot` | 4294.9 | 4294.9 | 0.1 |
| 24960 | `timing_slot` | 7834.6 | 7834.6 | 0.1 |
| 32768 | `timing_slot` | 4240.1 | 4240.1 | 0.1 |
| 49152 | `timing_slot` | 4092.1 | 4092.1 | 0.2 |
| 65536 | `timing_slot` | 4473.5 | 4473.5 | 0.3 |
| 131072 | `timing_slot` | 5287.5 | 5287.5 | 0.5 |
| 262144 | `timing_slot` | 7096.7 | 7096.7 | 0.8 |
| 524288 | `timing_slot` | 9759.8 | 9759.8 | 1.2 |
| 1048576 | `timing_slot` | 15298.0 | 15298.0 | 1.5 |

### Extra patterns @ 24960 B

| tag | metric | kernel_p50_µs | slot_p50_µs | egress_Gbps |
|---|---|---:|---:|---:|
| `p2_managed-l2_zero_24960` | `aicore_gang_span` | 3.1 | 3836.5 | 0.0 |
| `p2_managed-l2_single-hot_24960` | `aicore_gang_span` | 71.5 | 3643.9 | 2.8 |
| `p2_managed-host_zero_24960` | `timing_slot` | 5032.7 | 5032.7 | 0.0 |
| `p2_managed-host_single-hot_24960` | `timing_slot` | 4861.5 | 4861.5 | 0.0 |
| `p4_managed-l2_zero_24960` | `aicore_gang_span` | 7.0 | 6131.0 | 0.0 |
| `p4_managed-l2_single-hot_24960` | `aicore_gang_span` | 79.5 | 6227.3 | 2.3 |
| `p4_managed-host_zero_24960` | `timing_slot` | 7505.6 | 7505.6 | 0.0 |
| `p4_managed-host_single-hot_24960` | `timing_slot` | 7869.6 | 7869.6 | 0.0 |

## Notes

1. **Large C′ now runnable:** EP2/EP4 L2 `peer_bytes` 262144 / 524288 / 1048576 all
   compiled and timed (`aicore_gang_span`). EP2 L2 plateaus ~2.3–2.5 Gbps egress;
   EP4 L2 ~2.6–2.9 Gbps at large sizes.
2. **HOST** rows remain `timing_slot` only — swimlane does not yet attribute the
   builtin collective on the HOST rail; treat HOST numbers as full-program slot,
   not AIV kernel.
3. **24960 / 0-byte** points show elevated slot times vs neighbors (setup /
   swimlane tax); prefer kernel column for L2.
4. First-pass flakiness (3× `release_domain`, 1× HOST correctness) cleared on
   single retry; artifacts retained under `json/` / `builds/`.

## Artifacts

| File | Role |
|------|------|
| `summary.json` | One row per campaign point |
| `campaign_meta.json` | Conditions + timestamps |
| `sweep.log` | Full stdout/stderr |
| `json/*.json` | Per-point harness JSON |
| `builds/*` | Compile/output dirs |
