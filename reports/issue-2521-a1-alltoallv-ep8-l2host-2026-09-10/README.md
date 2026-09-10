# EP8 L2 vs HOST (known-good / mid-curve)

Focused EP=8 run on devices 0–7. Not a full §8.3 archive.

## Results (2026-09-10)

**L2 vs HOST slot_p50 — L2 remains ~1.6–2.0× faster**

| peer_bytes | L2 slot (µs) | HOST slot (µs) | L2/HOST | L2 AIV p50 (µs) |
|-----------:|-------------:|---------------:|--------:|----------------:|
| 32 | 3174 | 6341 | 0.50 | 20.3 |
| 128 | 3176 | 5846 | 0.54 | 20.1 |
| 1024 | 3392 | 5548 | 0.61 | 33.4 |
| 4096 | 3345 | 5986 | 0.56 | 88.4 |
| 8192 | 3504 | 6091 | 0.58 | 167.1 |
| 16384 | 3716† | 5833 | 0.64 | 313.1† |

† L2 16384 from earlier resume archive (this run’s L2 16384 failed 4/4).

**peer_bytes=0** (campaign settings) OK under `SIMPLER_COMM_FORCE_IPC=1`: AIV p50 ≈ 14.7 µs — see `zero_force_ipc/`.

Full writeup: `../issue-2521-a1-alltoallv-ubfix-2026-09-09/ANALYTIC_REPORT_FOR_COLLEAGUES.md` §9.
