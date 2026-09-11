# EP8 gap-fill / open-items

## 2026-09-11 open-items (`sweep_open_2026-09-11.log`) — DONE

| Area | Result |
|------|--------|
| L2 all 7 patterns @24960 | **OK** |
| HOST `49152`, `131072…1048576` | **OK** |
| HOST mixed / asymmetric / random @24960 | **OK** |
| HOST `65536` / HOST `self-only` | deferred → last-gaps |

## 2026-09-11 last-gaps (`sweep_last_gaps_2026-09-11.log`) — DONE

| Area | Result |
|------|--------|
| HOST `65536` | **OK** attempt=1 — slot 6586.4 µs, egress 0.55 Gbps |
| HOST `self-only` @24960 | **OK** attempt=2 after heal — slot 12127.4 µs |

EP8 measurable cells on this box are complete. See `json/`, attempt logs, and analytic report §4/§5/§9.1.
