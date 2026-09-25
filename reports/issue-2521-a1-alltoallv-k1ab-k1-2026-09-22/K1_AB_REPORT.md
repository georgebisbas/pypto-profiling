# K1 kernel A/B — EP2/EP4, same-toolchain template swap (2026-09-22)

**Question.** PR [#2828](https://github.com/hw-native-sys/pypto/pull/2828) (merged as
`5cb75663`) replaced the hand-written `all_to_all_v` push loop — one `TPUT` per
destination with two caller-side `pipe_barrier(PIPE_ALL)` per chunk — and quoted a
**1.5–1.8×** kernel speedup, described there as *"directional, not a formal A1 number"*.
This campaign tests that claim with the canonical A1 protocol.

**Method.** Only the kernel template is swapped, so toolchain version is held constant:

| Side | Template |
|---|---|
| `prek1` | `git show 5cb75663^:python/pypto/runtime/builtins/collectives/all_to_all_v/templates/kernel.cpp.in` (224 lines, 5 `TNOTIFY`, no `ld_dev`) |
| `k1` | `main` after the merge (305 lines, scalar `ld_dev` pull, two-round credit barrier) |

Both sides: `--core-num 1`, `uniform`, `--rounds 100 --warmup 5 --swimlane-rounds 8`,
persistent workers, `--platform a2a3`. EP2 on devices **4,5**; EP4 on **4,5,6,7**.
Payloads 0, 4096, 8192, 16384, 24576, 24960, 32768, 65536 B/peer.
At 24960 B/peer the cell shape is `row_width=4160`, `max_recv=253` (6 × 4160 = 24960),
i.e. ~98 chunks of 256 — the case K1 targets.

## Result 1 — campaign (32 cells per side, 32/32 OK both sides)

Kernel metric `fastest_p50_us`, median delta K1 vs pre-K1 over non-zero cells:

| Rail | Metric | EP2 | EP4 |
|---|---|---|---|
| `managed-l2` | `aicore_gang_span` (independent) | **−1.1%** (−6.5%…+0.8%) | **−2.5%** (−3.5%…+4.8%) |
| `managed-host` | `timing_slot` (**fallback**) | +3.4% (−10.7%…+8.1%) | +1.1% (−8.0%…+9.6%) |

**Read the HOST row with care:** for that rail `kernel_p50` and `slot_p50` are the same
number — the swimlane capture never fired (`aiv_rounds_captured=None`), so those cells
report the timing slot, not an independent AIV measurement. "HOST AIV" is already listed
as an open gap in the A1 report (§9.1). Only the `managed-l2` rows are a kernel delta.

At the PR's own anchor (EP2, L2, 24960 B/peer): **106.8 → 106.2 µs (−0.5%)**.
At the 16 KB HOST cell the PR quoted as "−11% p50": this campaign measures **+3.4% (EP2)
and +8.2% (EP4)**.

## Result 2 — repeatability probe (the decisive measurement)

The campaign is single-sample per cell, so I ran the anchor cell 3× per variant,
**interleaved** (K1, pre-K1, K1, pre-K1, K1, pre-K1, same 4,5 pair):

| Variant | p50 samples (µs) | mean | spread |
|---|---|---|---|
| pre-K1 | 111.8, 111.7, 111.7 | 111.73 | 0.1 |
| K1 | 110.7, 110.8, 110.8 | 110.77 | 0.1 |

**−0.87%**, and the two distributions do not overlap (gap 0.90 µs ≫ 0.1 µs spread).

So at the anchor the K1 kernel is **a real but ~1% improvement** — not 1.5–1.8×. The
swimlane metric itself is highly repeatable (0.1% spread), so the campaign's wider
per-cell spread is cross-cell/rail variation, not measurement noise.

## Interpretation

- **The directional claim does not reproduce under the A1 protocol.** Note the absolute
  scale differs by 10–20×: the PR reports ~1146–2267 µs at these payloads, this campaign
  ~107–112 µs. That is too large to be noise, so the two are not measuring the same
  quantity (different rail/impl, different warm-up discipline, or a wider span). The
  claim should not be restated as an A1 result without reconciling that.
- **A ~1% effect is physically coherent.** K1 removed *caller-side* barriers that
  duplicated barriers already inside `TPUT_IMPL`; the PR's own table notes the internal
  ~98 `pipe_barrier` per peer are **unchanged**. If the transfer is DMA-bound and those
  internal barriers remain on the path, deleting redundant duplicates buys little — which
  is what this data shows. The kernel is still structurally better (one `TPUT` per
  destination, flat extent, no caller chunk loop) and it is what made the scalar counts
  pull possible; that is a correctness/cleanliness win, not a speed win of this size.

## Caveats

- Single sample per cell in the campaign; only the anchor was repeated (Result 2).
- One flake: `p4_managed-l2_uniform_24960` SIGSEGV'd in `comm_destroy` / `final release
  failed` on the baseline side and needed a resume+retry round (simpler#2192 Fabric
  domain-churn family — host-side, not kernel-related). Both sides ended 32/32.
- Absolute numbers are shared-box measurements; treat deltas as paired, not as absolutes.

## Files

- `report: this file` (`…-k1ab-k1-2026-09-22/K1_AB_REPORT.md`)
- Campaigns: `…-k1ab-prek1-2026-09-22/`, `…-k1ab-k1-2026-09-22/` (`summary.json`,
  `campaign_meta.json`, `json/`, `builds/`, `attempt_logs/`)
- Probe: `/tmp/a2av_probe.sh`, samples `/tmp/a2av_probe/*.json`
- Paired table: `/tmp/k1ab_compare.py`
