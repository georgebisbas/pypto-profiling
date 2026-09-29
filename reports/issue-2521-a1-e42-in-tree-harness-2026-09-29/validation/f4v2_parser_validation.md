# Completion metric v2 - trace-view faithful span (offline validation)

Companion to the round-2 review fixes (commit 86abfffb). Duplicate 910B2 stamps
come from two cycle origins ~seconds apart; the second view's durations are
inflated (observed x3..x139). The aggregator splits stamps into views by
start-time gaps (>1 s), keeps the view holding the fastest stamp, and reports
its maximum duration - a genuinely slow block is kept; inflated duplicates are
structurally excluded.

- Replay over the archive (963 records with all_to_all_v stamps): official
  campaign dirs `b1-2026-09-25` (176 records) and `k1ep8-swap` (128 records)
  show **0 changes >1%**; diffs exist only in exploratory runs where the two
  time bases were previously averaged.
- Synthetic fixtures:
  - single-view {15x10us, 1x100us} -> 100.00 us (was 10.00)
  - two-view with inflated duplicates -> 100.00 us (duplicates not counted)
  - B=1 far pair (15.08, 603.62) -> 15.08 us (artifact still guarded)
  - benign cross-view pair (508.18, 674.00) -> 508.18 us (faithful view)

Transcript: `f4v2_transcript.txt`. Validator: `f4_validate_v2.py`.
