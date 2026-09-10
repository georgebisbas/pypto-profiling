# EP8 A1 continuation with `SIMPLER_COMM_FORCE_IPC`

Probe of remaining EP8 §8.3 points under `SIMPLER_COMM_FORCE_IPC=1`
(local simpler patch; see `collectives/a1_alltoallv/patches/`).

## Status (2026-09-10)

| Tag | Result |
|-----|--------|
| Seeded mid-curve (0…16 KiB L2 + HOST) | Copied OK from prior campaigns |
| `p8_managed-l2_uniform_24576` | **OK** — AIV p50 456.1 µs, slot 3692 µs, egress 3.01 Gbps |
| `p8_managed-l2_uniform_24960` try1–2 | FAIL — Fabric cross-server mid-session |
| `p8_managed-l2_uniform_24960` try3 | **OK** — after hardened patch (any Fabric export → IPC fallback) + rebuild + EP4 heal; AIV 589.2 µs, slot 11672 µs, egress 2.26 Gbps |

See analytic report §9.5:
`../issue-2521-a1-alltoallv-ubfix-2026-09-09/ANALYTIC_REPORT_FOR_COLLEAGUES.md`
