# Host-65536 recheck — interleaved K1 vs pre-K1 (2026-09-29)

**Purpose.** The E2.1 paired campaign measured a single host-65536 session per side
(pre-K1 7010.05 µs vs K1 6272.90 µs → −10.5%) — the only cell in the campaign where
K1 looked strongly faster. This recheck tests whether that holds up under
**interleaving** (alternating variants) to remove session/box-state bias.

**Method.** Same environment as E2.1 (B1 worktree `d626aea16`, runtime `6e383fc5` +
FORCE_IPC patch, PTOAS 0.65, CANN 9.0.0), EP8 on all 8 devices,
`managed-host uniform_65536`, timing-slot protocol, 30 rounds, FORCE_IPC. Kernel
template swapped per run (sha-verified each time); heal + retry ladder per run.
Run order: k1_r1 → prek1_r1 → k1_r2 → prek1_r2 → k1_r3 → prek1_r3.

**Results** (host rail, slot p50, µs; higher = worse):

| run | k1 | pre-K1 |
|---|---|---|
| rep 1 | 6217.8 | 6345.1 |
| rep 2 | 6117.6 | 6701.3 |
| rep 3 | 6326.7 | 6396.0 |
| **median** | **6217.8** | **6396.0** |

**Verdict: K1 −2.79% (K1 faster), non-overlapping** — the largest k1 sample
(6326.7) is still below the smallest pre-K1 sample (6345.1). The single-sample −10.5%
is **superseded**; the real K1 edge at this cell is ~2.8%.

**Attempts.** k1: [1, 1, 1]. pre-K1: [2, 1, 1] — pre-K1 r1 attempt 1 failed with the
counts assertion (`recv_counts=0 != clamped send_counts`, rc=1), heal rc=0, attempt 2
passed. This is another independent reproduction of the pre-K1 counts bug.

**Artifacts.** `json/p8_l1_managed-host_uniform_65536_{k1|prek1}_r{1..3}.json`,
`p8_l1_managed-host_uniform_65536_*_try*.log` (raw attempt logs; not committed —
~9 MB each). Template restored to K1 (`70a6cb09…`) after the race (sha-verified at
script exit).
