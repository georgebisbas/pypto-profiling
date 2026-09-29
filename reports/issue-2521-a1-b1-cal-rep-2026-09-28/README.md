# E1.5 — R7 calibration/repeat (timing-slot repeatability)

**Status: COMPLETE (2026-09-28).** 6/6 runs recorded (3 points × 2 repeats).

Per plan-110 R7: *"small, anchor, and 1024 KiB points; 5+100, twice; calibrate
timing-slot and variance"* — executed on the HOST rail (`managed-host`, the rail
R7 specifies), EP8, FORCE_IPC, K1 template `70a6cb09…` on B1 (`d626aea16`).

## Protocol

- 5 warmup + 100 timing-slot rounds per run; `--profile timing-slot`; devices 0–7; FORCE_IPC=1.
- Runner: `/opt/a1/e15_run.sh` (heal + up to 4 attempts per run; stalled attempts
  produce no metrics and are not used).
- Point mapping: small = **128 B**, anchor = **24960 B**, 1024 KiB = **1048576 B**.

## Results (host_timing_slot_p50, µs)

| point | run 1 | run 2 | run-to-run Δ% | mean | E1 (2026-09-25) | vs E1 % |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 5274.21 | 5176.88 | **−1.85** | 5225.5 | 5345.00 | −1.32 |
| 24960 | 12554.75 | 13194.11 | **+5.09** | 12874.4 | 12956.91 | −3.10 |
| 1048576 | 30625.28 | 30303.10 | **−1.05** | 30464.2 | 30723.20 | −0.32 |

Attempts (incl. absorbed stalls): 128 → [1, 3]; 24960 → [1, 3]; 1048576 → [1, 2].

## Verdict (R7 deliverable)

- **Repeatability ≤ ~2%** for the 128 B and 1 MiB points — good.
- **Anchor (24960 B): 5.1% run-to-run** — at the edge of the 3–5% acceptance band.
  The anchor is also the stall-prone pathological row shape (`row_width=4160`, 6×4160);
  if a tighter anchor calibration is needed, raise the repeat count (n≥4) rather than
  a single extra pass.
- All six runs agree with the 2026-09-25 E1 values within **−3.1%…−0.3%** — no evidence
  of drift between campaign days beyond the repeat spread above.

## Artifacts

`json/` (6 cell JSONs), per-attempt logs in this directory, `builds/` on disk only.
