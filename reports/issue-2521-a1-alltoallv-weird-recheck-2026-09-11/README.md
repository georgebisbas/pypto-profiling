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
