# K3 scaling — repeatability re-run (2026-09-24)

*Same matrix and protocol as the [2026-09-23 campaign](../issue-2521-k3-lane-scaling-2026-09-23/K3_LANE_SCALING_REPORT.md):
EP4 (NPUs 4,5,6,7), `--impl managed-host`, `L ∈ {1,2,4,8,16}`, payloads
`{0, 16 KiB, 256 KiB, 1 MiB}` + `zero@24960` control, `--rounds 100 --warmup 5
--swimlane-rounds 8`, 2 interleaved reps. Kernel tree unchanged (`351158a6`;
rebased content).*

**Result: 50/50 cells OK, zero retries, RC=0.** The 2026-09-23 conclusions
reproduce:

- **1 MiB/peer**: Δ vs `L=1` — `L=2` −23.7 %, `L=4` −37.4 %, **`L=8` −47.7 %,
  `L=16` −50.4 %** (run 1: −23.3 / −40.3 / **−50.9 / −49.5 %**). Egress up to
  **3.14 Gbit/s per rank** (run 1: 3.15; ≈0.39 GB/s).
- **256 KiB/peer**: −15.7 / −23.6 / −28.7 / −30.5 % (run 1: −11.8 / −17.8 /
  −19.3 / −25.1 %).
- **16 KiB/peer**: −0 / −0.7 / −0.7 / −5.2 % (run 1: −3.1 / −7.7 / −9.0 /
  −14.1 %); absolute `L=16` time identical (3 821 vs 3 834 µs) — the delta
  difference is baseline drift on the `L=1` arm (−9.6 %).
- **Zero-counts control**: +4.0 / +6.3 / +4.9 / +3.4 % (run 1: +0.4 / +0.5 /
  +1.5 / +3.9 %) — same small fixed-cost region, run-to-run noise dominates.
- **`L=8` vs `L=16` at 1 MiB is a tie**: the ordering *flips* between runs
  (run 1: L=8 better by 2.7 %; run 2: L=16 better by 5.4 %). This confirms the
  flagged step in run 1 was noise — A3 should treat `B ≥ 8` as one operating
  point, not pick between 8/16 without more reps.

Cross-run per-cell table: `comparison_vs_2026-09-23.txt`. Analyzer output:
`analysis_output.txt` (raw: `summary.json`, `json/`, `campaign_console.log`).
Same caveats as run 1: whole-program timing slot on a shared box, 2 reps/arm →
read deltas pairwise; ≤ ~3 % steps are noise, and cross-run baseline drift is
±5–10 % on the small-payload arms.
