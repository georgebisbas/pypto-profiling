# Fragile list — configurations excluded from official-protocol benchmarking

**Rule (2026-09-28, user-approved):** benchmark tables include **only** sessions that
complete cleanly (rc=0, correctness check passed, session exit ok). Stalled or failed
attempts produce no metrics and are **never** used as data. Configurations known to be
stall/fail-heavy at the official protocol are listed here and are excluded from grids
via `A2AV_SKIP_TAGS` — skipped tags are recorded as `skipped_fragile` in
`campaign_meta.json`, so exclusions stay auditable (no silent drops).

| # | Configuration | Condition | Evidence | Replacement |
|---|---|---|---|---|
| 1 | `p8_l1_managed-host_random_24960` | 100-round timing-slot (official protocol) | 10/10 stalls (`S1:running-stalled`), reproduces on K2+K3 tree — see `../issue-2521-a1-b1-2026-09-25/RESUME_COMPLETION_2026-09-28.md` | supplementary 30r/60r values archived in `../issue-2521-a1-b1-2026-09-25/supplement/` |
| 2 | timing-slot sessions with ≥ ~65–85 dispatches (config-dependent, box-state sensitive) | long sessions on this box | 80r/100r stalls; 30r/60r complete (same doc, stall characterization matrix) | run campaigns at 30r (validated) or 60r; annotate protocol deviations in report |

**Correctness-invalidity note (not a config exclusion):** pre-K1 template runs at EP8
intermittently fail `AssertionError: recv_counts=0 != clamped send_counts` (rc=1).
Per plan-110 ("correctness failures invalidate the corresponding timing result") those
attempts are never timing data; they are retained as **correctness evidence** (see
`evidence/` in this directory) supporting the K1 delta.

**Never-skipped-by-default:** the mechanism is off unless `A2AV_SKIP_TAGS` is set, so
the harness stays general and reproducible for third parties.
