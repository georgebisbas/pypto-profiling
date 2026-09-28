# RFC #2521 — A1 executive summary (2026-09-28)

**Scope:** A1 = benchmark harness + pre-change baseline (plan 110). Measurement program
**COMPLETE**; one open deliverable (in-tree harness PR, E4.2). All artifacts pushed to
this repo; change record in `pypto-3.0-notes` (`d66a84e`).

## Deliverable definition (tactic change, 2026-09-25)

- **Official baseline B1** = `d626aea16` (post-K1 + PTOAS v0.65), EP8, `SIMPLER_COMM_FORCE_IPC=1`.
- **Pre-change side** = same-tree kernel-template swap (pre-K1 `kernel.cpp.in`, sha256 `bb503e2b…`; k1ab method).
- Pre-K1 rebuild `166bf7ac4` demoted (era toolchain; wrong counts path). **EP16 blocked** (8-chip box).

## Results at a glance

| Experiment | Headline | Report |
|---|---|---|
| E1 (official, EP8, 100r) | 43/44 cells; L2 aicore curve 16 µs→19.1 ms over 0→1 MiB; HOST slot family ≈5.2–13 ms | `issue-2521-a1-b1-2026-09-25/` |
| E2.1 (paired K1 delta, 30r) | **K1 +2.6% L2 / +2.7% HOST slower (means)** vs pre-K1; host-64 KiB K1 **−10.5%** (follow-up); **no 1.5–1.8×**; pre-K1 counts-bug reproduced | `issue-2521-a1-b1-k1ep8-swap-2026-09-28/` |
| E1.5 (R7 calibration) | repeat ×2: **1.0–1.9%** (128 B, 1 MiB), **5.1%** anchor (band edge) | `issue-2521-a1-b1-cal-rep-2026-09-28/` |
| E1.4 (tails) | all six **N/A**: `row_width=N is not 32-byte aligned for INT8 staging` | `issue-2521-a1-b1-tails-2026-09-28/` |
| Blocked cell | `managed-host random@24960` 100r stall characterized (round-count-dependent, runtime-level); 30/60r supplements archived | `issue-2521-a1-b1-2026-09-25/README + supplement/` |

**Cross-rail view** (same ruler — host timing slot): L2 dispatch ≈ **1.4–1.7× faster** than
HOST at 4 KiB–64 KiB; ≈parity at 0 B and 24960 B (both ~11 ms dispatch-side spikes; the L2
device span stays ~0.6 ms at 24960). Details in the E2.1/E1 reports.

## Open gaps (E3.2)

1. **HOST rail has no device-side (AIV) metric** — only the timing slot; attribution gap (plan §9.1).
2. **EP16 blocked** (hardware) — never interpolated.
3. **Tails N/A by construction** (32-B INT8 staging alignment) — re-open if the contract changes.
4. **Stall hazard**: timing-slot sessions ≥ ~65–85 dispatches can stall (`S1:running-stalled`,
   runtime-level; bisect in `RESUME_COMPLETION_2026-09-28.md`). Recommend a runtime/simpler issue;
   campaigns use 30r + retries + heals meanwhile.
5. **In-tree harness PR (E4.2)** pending — the only open A1 deliverable.

## Policy (adopted 2026-09-28)

Only clean sessions (rc=0 + correctness check) enter tables; stalled/failed runs are never data;
known-fragile configs are excluded explicitly (`A2AV_SKIP_TAGS` → `skipped_fragile` in campaign
meta; list: `issue-2521-a1-b1-k1ep8-swap-2026-09-28/FRAGILE_LIST.md`).

## Artifact map / pushes (this repo)

`ea684805`, `5ca4cef2`, `91016266` (E1 + blocked cell) · `4f8458ae`, `6bd0d4be`, `1347cf5b`
(E2.1) · `0ddea067` (E1.5) · `7335588a` (E1.4) · `684fb059` (skip mechanism + fragile list) ·
`cc187ed8`, `bc831ab1` (k1ab + tactic + B0 record). Notes repo: `d66a84e`.
