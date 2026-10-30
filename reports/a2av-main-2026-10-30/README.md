# AllToAllV on pypto `f32ecf3f0` — L2/HOST vs HCCL (2026-10-30)

> **Report status:** current — first A2AV-vs-HCCL run on the rebuilt image. See
> [`../upstream-status.md`](../upstream-status.md).

The 2026-10-14 A2AV comparison was measured on pypto `d2551caa` with the old harness schema.
This campaign reruns it on the rebuilt image and the current in-tree harness.

## Provenance

| | |
|---|---|
| Box | `hng-atlas01` (192.168.150.11), 8× Ascend 910B2, queue-isolated |
| Image | `pypto3-hw-native-sys:cann9` (rebuilt 2026-10-30) |
| pypto / runtime / pto-isa / PTOAS | `f32ecf3f0` / `6b6b88f2` / `217bb989` / v0.70 |
| Queue task | `task_20261030_161545_299625718603` (8 cards, exit 0) |
| Harness | in-tree `tests/st/distributed/collectives/all_to_all_v_benchmark.py` |
| HCCL baseline | `collectives/.hccl_bench_bin --op alltoallv --dtype int8 --include-self` |

## Method

INT8, EP8, `uniform`, 64 KiB and 1 MiB per peer, 3 warmup + 20 timed rounds; 8 swimlane rounds for
the PyPTO device capture. PyPTO rails: `managed-l2` and `managed-host`, `L ∈ {1,8}`. HCCL ran with
`--include-self`, so both sides move `P × peer_bytes` per rank (the current harness counts the
diagonal; the 2026-10-14 report excluded it on both sides — do not compare absolute bandwidths
across the two reports).

## Results

Host-visible p50 (the only ruler both stacks share):

| peer | config | host-slot p50 | vs HCCL | device `fastest_p50` |
|---:|---|---:|---:|---:|
| 64 KiB | **HCCL `HcclAlltoAllV`** | **455 µs** | 1.00× | — |
| 64 KiB | pypto `managed-l2` L=1 | 4543 µs | 9.98× | 48 µs |
| 64 KiB | pypto `managed-l2` L=8 | 4650 µs | 10.22× | 35 µs |
| 64 KiB | pypto `managed-host` L=1 | 7379 µs | 16.21× | — |
| 64 KiB | pypto `managed-host` L=8 | 7419 µs | 16.30× | — |
| 1 MiB | **HCCL `HcclAlltoAllV`** | **566 µs** | 1.00× | — |
| 1 MiB | pypto `managed-l2` L=1 | 10152 µs | 17.94× | 499 µs |
| 1 MiB | pypto `managed-l2` L=8 | 9924 µs | 17.53× | 123 µs |
| 1 MiB | pypto `managed-host` L=1 | 13326 µs | 23.54× | — |
| 1 MiB | pypto `managed-host` L=8 | 13474 µs | 23.81× | — |

`device fastest_p50` is the fastest rank's per-task kernel time, not a collective completion span;
HCCL exposes no device span. It is shown for attribution, not as a device-to-device ratio.

## Findings

1. **The gap widens with payload.** Best PyPTO config: **10.0× at 64 KiB → 17.5× at 1 MiB**.
   HCCL is nearly flat (455 → 566 µs for 16× more data); PyPTO grows with payload.
2. **Multi-lane moves the kernel, not the host slot.** On `managed-l2`, L=8 vs L=1 changes the
   device task time 48 → 35 µs (64 KiB) and 499 → 123 µs (1 MiB), but the host slot moves only
   +2.4% and −2.2%. Staging + outer dispatch + sync dominate the invocation.
3. **HOST is 16–24× behind and L does not help** (7379 vs 7419 µs at 64 KiB; 13326 vs 13474 µs at
   1 MiB).
4. **The kernel is below HCCL's whole wall, but the invocation is not.** At 1 MiB the L2 L=8 task
   time is 123 µs vs HCCL's 566 µs wall; the host-visible slot is 9924 µs. The ~9.8 ms remainder
   is runtime plumbing, not the collective.
5. **Absolute numbers are not comparable to the 2026-10-14 report.** That run used self-excluded
   volume and the old `host_timing_slot`/AIV-gang metrics; this one uses `end_to_end_latency` and
   per-rank kernel time with `--include-self`. The direction (gap widens) is the same.

## Compatibility work done for this run

- `alltoallv_a1.py` now anchors `A2AV_OUT` absolutely (the harness runs with cwd `/opt/pypto`, so
  relative output paths landed outside the repo) and accepts the current harness schema
  (`end_to_end_latency_p50_us`, per-rank bandwidth dicts, `per_rank_receive_bandwidth` metric).
- `plot_hccl_vs_pypto_payload_sweep.py` derives the ratio subtitle from the sweep instead of the
  old hard-coded 6.3×/23.6×.
- HCCL matches the harness volume with `--include-self`.

## Caveats

- Two payloads, one pattern, 20 rounds; not the official §8.3 100-round curve.
- One image/box state; all eight cards were held for the run.
- The old AIV-gang span metric is gone from the current harness; `fastest_p50` is per-rank kernel
  time.

## Reproduce

```bash
# HCCL (per payload)
./collectives/.hccl_bench_bin --op alltoallv --dtype int8 --peer-bytes 65536 --include-self \
  --devices 0,1,2,3,4,5,6,7 --warmup-rounds 3 --timed-rounds 20 \
  > results/campaigns/<campaign>/hccl/hccl_a2av_ep8_65536.log

# PyPTO
A2AV_OUT=results/campaigns/<campaign>/pypto A2AV_EPS=8 \
A2AV_IMPLS=managed-l2,managed-host A2AV_CORE_NUMS=1,8 \
A2AV_PAYLOADS=65536,1048576 A2AV_ROUNDS=20 A2AV_WARMUP=3 \
A2AV_SWIMLANE_ROUNDS=8 A2AV_RETRIES=1 A2AV_EXTRA_PATTERNS="" \
python3 -u collectives/alltoallv_a1.py

# Join + plot
python3 collectives/join_hccl_vs_pypto_a2av.py results/campaigns/<campaign>
python3 collectives/plot_hccl_vs_pypto_payload_sweep.py \
  --sweep results/campaigns/<campaign>/payload_sweep.json \
  --out results/campaigns/<campaign>/figures
```

## Artifacts

`summary.json` (the joined `payload_sweep.json`), `json/*.json` (per-cell harness output),
`logs/hccl_a2av_ep8_*.log`, `campaign_meta.json`, and `figures/*.png`. Raw `attempt_logs/` and
`builds/` remain under `results/campaigns/a2av_main_2026-10-30/` (gitignored).
