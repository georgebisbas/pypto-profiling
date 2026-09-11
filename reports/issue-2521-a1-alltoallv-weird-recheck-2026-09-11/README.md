# Weird-point recheck (2026-09-11) — STOPPED (partial)

Re-run of points that look off the main trend in the A1 archive. Stopped mid-EP8
because the box had other users' processes.

| Focus | Why |
|-------|-----|
| `0` | Inflated slot; EP4 L2 AIV vs EP2/EP8 |
| `24576` / `24960` / `32768` | 24960 slot/egress spike (DSV4 shape) |
| `49152` | Mild egress dip on EP2/EP4 L2 |

## Status at stop

| Scope | Result |
|-------|--------|
| EP2 L2+HOST (5 payloads) | **Done** |
| EP4 L2+HOST (5 payloads) | **Done** |
| EP8 L2 `0` | **OK** (attempt 2) |
| EP8 L2 `24576+` / HOST | **Not run** (stopped) |

Resume later with `A2AV_RESUME=1` into this `A2AV_OUT`.
Artifacts: `json/`, `summary.json`, `sweep.log`, `STOPPED.txt`.
