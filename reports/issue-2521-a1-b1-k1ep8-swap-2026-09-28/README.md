# E2.1 — K1 delta @ EP8 (paired 30-round protocol)

**Status: COMPLETE (2026-09-28).** Both sides 16/16 cells OK, 0 failures.
Paired comparison of the K1 kernel template vs the pre-K1 template on the frozen
B1 tree, EP8, FORCE_IPC, uniform sweep (0 → 64 KiB), both rails.

## Provenance

| Item | Value |
|---|---|
| Tree | B1 `d626aea16` (`/opt/a1/b1`), toolchain-exact (PTOAS 0.65, CANN 9.0.0) |
| K1 template | `70a6cb09…` (305 lines, K1 squash) — side `k1/` |
| Pre-K1 template | `bb503e2b…` (224 lines, `5cb75663a^`) — side `prek1/` |
| Runtime | `6e383fc5` + FORCE_IPC patch (`a88e4442…`) |
| Protocol | **30 rounds** +5 warmup, swimlane 8, `FORCE_IPC=1`, EP8, devices 0–7 |
| Why 30r | 100r sessions stall (`S1:running-stalled`); aborted runs archived: `prek1_100r_aborted/` (4/4 attempts), `prek1_60r_aborted/` (2 attempts) — see `../issue-2521-a1-b1-2026-09-25/RESUME_COMPLETION_2026-09-28.md` |
| Driver | `collectives/alltoallv_a1.py` (with `A2AV_SKIP_TAGS` support; nothing skipped in this campaign) |

## Result 1 — paired K1 delta (Δ = K1 vs pre-K1, `fastest_p50_us`)

| rail | payload | pre-K1 p50 | K1 p50 | Δ% | E1 (K1@100r) p50 | k1@30r vs E1 |
|---|---:|---:|---:|---:|---:|---:|
| managed-l2 | 0 | 15.45 | 16.34 | **+5.8** | 16.22 | +0.7 |
| managed-l2 | 4096 | 87.43† | 89.66 | +2.6† | 89.29 | +0.4 |
| managed-l2 | 8192 | 161.78 | 164.14 | +1.5 | 164.10 | +0.0 |
| managed-l2 | 16384 | 301.93 | 313.91 | +4.0 | 313.89 | +0.0 |
| managed-l2 | 24576 | 447.61 | 462.34 | +3.3 | 462.47 | −0.0 |
| managed-l2 | 24960 | 579.91 | 576.13 | **−0.7** | 571.48 | +0.8 |
| managed-l2 | 32768 | 591.59 | 610.59 | +3.2 | 609.21 | +0.2 |
| managed-l2 | 65536 | 1199.14 | 1211.61 | +1.0 | 1209.51 | +0.2 |
| managed-host | 0 | 11705.00 | 12689.03 | +8.4 | 12504.48 | +1.5 |
| managed-host | 4096 | 5238.83 | 5623.87 | +7.4 | 5280.03 | +6.5 |
| managed-host | 8192 | 5395.65 | 5534.42 | +2.6 | 5364.76 | +3.2 |
| managed-host | 16384 | 5437.40 | 5655.60 | +4.0 | 5545.96 | +2.0 |
| managed-host | 24576 | 5358.90 | 5450.27 | +1.7 | 5506.59 | −1.0 |
| managed-host | 24960 | 11809.93 | 12244.17 | +3.7 | 12956.91 | −5.5 |
| managed-host | 32768 | 5627.07 | 5854.94 | +4.1 | 5727.95 | +2.2 |
| managed-host | 65536 | 7010.05 | 6272.90 | **−10.5**† | 6587.94 | −4.8 |

\* prek1 L2-4096 raw JSON was missing (relaunch artifact) and was re-acquired
2026-09-28 (rerun @30r: 87.43 µs; original summary value 88.53 — within 1.3%).
† HOST 65536 single-sample **−10.5%** superseded by the interleaved 3×3 recheck
(`recheck_host64k/`): **K1 −2.8%** (non-overlapping).

**Summary:** K1 is **+2.6% slower (mean) on L2** (median +2.9%, 7/8 positive) and
**+2.7% slower (mean) on HOST** (median +3.9%, 7/8 positive), with one exception:
HOST 65536, where the single paired session measured **−10.5%**; the interleaved 3×3
recheck (`recheck_host64k/`) gives **−2.8% (K1 faster; non-overlapping, medians
6217.8 vs 6396.0 µs)** — a real but small K1 advantage at this cell; the −10.5% was
sampling noise.

## Result 2 — control: protocol drift (k1@30r vs E1 K1@100r)

- L2: all eight cells within **±0.8%** → 30r vs 100r protocol drift ≈ zero on the L2 rail.
- HOST: ±6% scatter (day/state sensitivity); mean +0.5%.

Implication: (a) E1's L2 numbers are protocol-robust; (b) the +2.6% L2 delta above is
**not** a protocol artifact.

## Reliability observations (not timing data)

- Attempts: pre-K1 side `[1×12, 2×2, 4×1, 5×1]`; K1 side `[1×9, 2×5, 3×2]` — both
  25 attempts for 16 cells.
- **Correctness:** pre-K1 side hit `AssertionError: recv_counts=0 != clamped
  send_counts` (rc=1) on uniform_16384/24960/32768 retries — the K1-targeted counts
  bug, reproduced at EP8 (evidence logs: `evidence/`). **K1 side: zero correctness
  failures** (only environmental stalls).
- Stalls (`rc=-11 S1:running-stalled`) occurred on both sides and were absorbed by
  the retry/heal ladder; stalled attempts produce no metrics and are not used.

## Conclusions

1. **K1 vs pre-K1 at EP8: performance-neutral within ~±3%.** Direction on this
   dataset is *slightly negative* (K1 a few % slower on both rails), the opposite in
   sign to the EP2/EP4 k1ab result (K1 −1.1%/−2.5% on L2) and the EP2 anchor probe
   (−0.87%). Cross-scale net: no material K1 speed effect either way; the widely
   quoted 1.5–1.8× does **not** reproduce at any scale measured.
2. **K1's measured value is correctness**: it eliminates the intermittent pre-K1
   counts failure at EP8 (reproduced here independently) and unblocks K2/K3/K4/O2/A3.
3. **HOST `65536` rechecked (interleaved 3×3, `recheck_host64k/`):** K1 is
   **−2.8% faster** (medians 6217.8 vs 6396.0 µs; all three K1 reps beat all three
   pre-K1 reps — non-overlapping). Real, small, single-cell K1 gain — the only one
   beyond noise in this campaign; the original single-sample −10.5% is superseded.

## Artifacts

`prek1/` and `k1/` (json, summary, campaign_meta), `provenance.json`, `evidence/`,
`prek1_100r_aborted/`, `prek1_60r_aborted/`, `FRAGILE_LIST.md`, `recheck_host64k/`
(interleaved host-65536 recheck: json + README; raw logs stay on disk, not committed).
Raw attempt logs & builds stay on disk (gitignored / not committed).
