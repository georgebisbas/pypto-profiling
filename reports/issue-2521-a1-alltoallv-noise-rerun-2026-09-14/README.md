# Noise re-run (2026-09-14) — DONE

Only cells that looked noisy after weird-recheck.

## 1. EP4 L2 uniform `@0` ×3

| Run | AIV p50 (µs) | slot p50 (µs) |
|-----|-------------:|--------------:|
| archive (ubfix) | 78.5 | 5945 |
| weird-recheck | 149.0 | 6225 |
| **rep1** | **8.04** | 6095 |
| **rep2** | **6.96** | 5740 |
| **rep3** | **6.84** | 5959 |

**Verdict:** Prior high AIV values were **noise / bad swimlane samples**. Fresh repeats cluster ~**7–8 µs** (in line with EP2~3 µs / EP8~15 µs scaling). Prefer ~7 µs; do **not** use 78 or 149 for planning.

## 2. EP2 L2 `32768 / 49152 / 65536`

| peer | archive AIV / eg | weird-recheck AIV / eg | this run AIV / eg |
|-----:|------------------:|------------------------:|------------------:|
| 32768 | 107.1 / 2.45 | 109.6 / 2.39 | **107.1 / 2.45** |
| 49152 | 165.2 / **2.14** | 162.7 / **1.52** | **158.0 / 2.49** |
| 65536 | 210.2 / 2.49 | — | **211.6 / 2.15** |

**Verdict:** EP2 `@49152` egress dip was **noise**. Fresh run sits at **~2.49 Gbps** with neighbors; AIV still ~linear.

Artifacts: `json/`, `sweep.log`, `summary.json` (EP2 trio only).
