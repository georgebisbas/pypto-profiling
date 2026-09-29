# F4 validation — last-completing-block span (offline)

Companion to the review fixes on PR #2945 (commit a46dd0c4). The swimlane span
aggregation changed from the mean of the valid per-block durations to their max
(the collective completes when its last block completes). Validation is offline:

- **Replay over the archived raw records** (E1 builds + salvage; 703
  `chip_swimlane_records.json` files, 176 with `all_to_all_v` tasks):
  old (mean) vs new (max) -> **0 value changes** (all L=1, single block).
- **Synthetic 16-block fixture** (durations 110-120 us + one far dual-die
  artifact at 1000 us): old mean **115.000 us** -> new max **120.000 us**;
  the artifact is still dropped by the near-min cluster (keeps 16/17).
- **Single-block check:** old == new (110.000 us).

Transcript:

```text
real records found: 703
records with all_to_all_v tasks: 176; mismatches(old!=new): 0
module-function vs max cross-check (first 80): OK
synthetic durs us: [110.0, 110.66, 111.34, 112.0, 112.66, 113.34, 114.0, 114.66, 115.34, 116.0, 116.66, 117.34, 118.0, 118.66, 119.34, 120.0, 1000.0]
synthetic 16 blocks: old(mean)=115.000 -> new(max/last)=120.000; cluster keeps 16/17 (artifact dropped)
module func on synthetic: 120.0
single-block check: old=110.000 new=110.000 equal=True
```

Validator script: `f4_validate.py` (run from any cwd; loads the in-tree
harness by path).
