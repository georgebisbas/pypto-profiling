# Weird-point recheck (2026-09-11) — COMPLETE

Re-run of points that look off the main trend in the A1 archive.

| Focus | Why |
|-------|-----|
| `0` | Inflated slot; EP4 L2 AIV vs EP2/EP8 |
| `24576` / `24960` / `32768` | 24960 slot/egress spike (DSV4 shape) |
| `49152` | Mild egress dip on EP2/EP4 L2 |

Rails: L2 + HOST. EP: 2, 4, 8. Patterns: uniform only. EP8 uses `A2AV_FORCE_IPC=1`.

## Status

| Scope | Result |
|-------|--------|
| EP2 L2+HOST (5 payloads) | **Done** |
| EP4 L2+HOST (5 payloads) | **Done** |
| EP8 L2+HOST (5 payloads) | **Done** (`sweep_resume_ep8_2026-09-11d.log`) |

## EP8 recheck highlights (reproduced)

| peer | L2 AIV µs | L2 slot µs | HOST slot µs |
|-----:|----------:|-----------:|-------------:|
| 24576 | 467 | 3837 | 6009 |
| **24960** | **615** | **12080** | **11427** |
| 32768 | 585 | 4062 | 6210 |
| 49152 | 878 | 4641 | 6542 |

24960 slot spike is **real** on EP8 L2 and HOST (same shape effect as EP2/EP4).
Artifacts: `json/`, `summary.json`, `sweep.log`, `sweep_resume_ep8_2026-09-11d.log`.

## Resolution (2026-09-14)

Two flags from this recheck were **bad swimlane samples**, not real effects — re-validated in `../issue-2521-a1-alltoallv-noise-rerun-2026-09-14/`:

| Cell | weird-recheck | noise re-run verdict | Use for planning |
|------|---------------|----------------------|------------------|
| EP4 L2 `@0` AIV | 149 µs | three repeats **~7 µs** | **~7 µs** (not 78 or 149) |
| EP2 L2 `@49152` egress | 1.52 Gbps | fresh run **2.49 Gbps** | **~2.49 Gbps** (not 2.14 / 1.52) |

Corrected values are in `../issue-2521-a1-alltoallv-ubfix-2026-09-09/ANALYTIC_REPORT_FOR_COLLEAGUES.md` §4/§5. Raw archive JSON here is unchanged.
