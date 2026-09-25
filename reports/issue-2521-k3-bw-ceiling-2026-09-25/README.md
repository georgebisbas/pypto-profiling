# K3 bandwidth ceiling probe (2026-09-25)

*Follow-up to the [EP8 campaign](../issue-2521-k3-lane-scaling-ep8-2026-09-24/README.md):
how close is lane-mapped `all_to_all_v` to the interconnect peak at large
payloads? EP8 (NPUs 0–7), `--impl managed-host`, `L=16` (max lanes),
`SIMPLER_COMM_FORCE_IPC=1`, 30 rounds, 1 rep. The 256 KiB / 1 MiB rows are the
EP8 campaign's `L=16` arms for context; 2/8/16 MiB were measured here.*

**Peak reference:** HCCS **56 GB/s per link × 7 = 392 GB/s per chip**
(bidirectional aggregate; the verified figure used across the notes corpus).

## Results (whole-program timing slot mean; rates are per rank, bytes)

| payload/peer | slot mean (µs) | egress (Gbit/s) | egress (GB/s) | duplex (GB/s) | % of 392 GB/s | marginal egress (GB/s) |
|---|---|---|---|---|---|---|
| 256 KiB | 7 260.1 | 2.02 | 0.25 | 0.51 | 0.13 % | — |
| 1 MiB | 12 333.9 | 4.76 | 0.60 | 1.19 | 0.30 % | 1.09 |
| 2 MiB | 22 140.6 | 5.30 | 0.66 | 1.33 | 0.34 % | 0.75 |
| 8 MiB | 42 216.5 | 11.13 | 1.39 | 2.78 | 0.71 % | 2.19 |
| 16 MiB | 81 582.8 | 11.52 | 1.44 | 2.88 | **0.73 %** | 1.49 |

`marginal` = Δ(bytes sent per rank) / Δ(slot) between adjacent rows (egress,
GB/s).

## Findings

- **Throughput rises with payload but stays ~2 orders of magnitude below the
  link:** per-rank egress grows 0.25 → 1.44 GB/s (256 KiB → 16 MiB), duplex
  utilization tops out at **≈0.73 %** of the per-chip HCCS aggregate. The
  8→16 MiB marginal (~1.5 GB/s per rank) is flat vs 2→8 MiB — the curve is
  flattening, i.e. the op is **issue-rate/latency-bound, not link-bound**.
- System-wide at 16 MiB: ≈11.5 GB/s aggregate egress (≈2.9 % of the corpus's
  "392 GB/s 8-NPU full-mesh aggregate" reading for the whole box).
- For DSV4-scale payloads (24–48 KiB/peer) the same framing is ~0.01 % — the
  transfer itself is microseconds against a 7–12 ms per-call floor; the win
  there comes from overlap/fusion, not bandwidth.
- **32 MiB is not measurable with the current bench:** the pypto frontend
  rejects the generated loop (`Unroll loop trip count 2048 exceeds maximum
  allowed (1024)`). 16 MiB maps to exactly 1024 trips and compiles.
- **Compile-time note:** ≥8 MiB shapes exceed the default 60 s ptoas compile
  timeout (8 MiB ≈ 2–3 min, 16 MiB ≈ 8–10 min at 600 s). The probe ran with a
  local 600 s override in `python/pypto/backend/pto_backend.py`, reverted after.

## Method notes

- Single rep per large-payload cell (30 rounds measured, p50-stable); 256 KiB /
  1 MiB rows are 2-rep campaign means. `egress`/`duplex` are the harness
  `gbps` values = **Gbit/s** per rank (fastest rank); GB/s columns are ÷8.
- Artifacts: `summary.json`, `json/`, `analysis_output.txt`, `attempt_logs/`
  (incl. the 32 MiB unroll-limit failure), `builds/`.
