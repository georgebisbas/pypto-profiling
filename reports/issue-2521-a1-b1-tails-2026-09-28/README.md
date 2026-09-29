# E1.4 — Tail points (1, 15, 31, 33, 4159, 4161 B) — acceptance probe

**Status: COMPLETE (2026-09-28) — all six points N/A (rejected by the compiler front-end).**

Protocol: attempt-once per point (per exclusion policy), `managed-l2`, EP8, 30r,
FORCE_IPC, K1 template `70a6cb09…` on B1 (`d626aea16`), devices 0–7.

| point (peer B) | result | verbatim reason |
|---:|---|---|
| 1 | REJECTED | `row_width=1 is not 32-byte aligned for INT8 staging; sub-32-byte tail points cannot use the row-loop stage kernel` |
| 15 | REJECTED | `row_width=15 …` (same class) |
| 31 | REJECTED | `row_width=31 …` |
| 33 | REJECTED | `row_width=33 …` |
| 4159 | REJECTED | `row_width=4159 …` |
| 4161 | REJECTED | `row_width=4161 …` |

## Conclusion

Tails (`sub-32-byte` rows) and the `valid_elems<K` class are **structurally out of
scope** for the current INT8 staging path (32-byte row-alignment contract), not
measurement gaps. This matches the pre-existing expectation from older reports and
is now verified first-hand on the frozen baseline B1; no retries were spent
(attempt-once policy). Re-open only if the staging contract changes (e.g. a tail-
capable loader path).

## Artifacts

`logs/` — six rejection logs. No JSONs (all rejected before dispatch).
