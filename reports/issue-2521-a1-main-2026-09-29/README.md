# Main snapshot `54521824` — A1 protocol @ EP8 (L=1 sweep + L=16 spots)

**Status: COMPLETE (2026-09-29).** 10/10 L=1 cells OK, 2/2 L=16 cells OK, 0 failures.

**Tree:** pypto `main` @ `54521824f2777fcba180322b7148ce335d254709` (`/opt/a1/main` worktree; post-K2 `e48937b9` / PR #2889, pre-K3 / PR #2900) · **Box:** hng-atlas01, 8× Ascend NPUs (a2a3) · **Harness:** in-tree `tests/st/distributed/collectives/all_to_all_v_benchmark.py` (e42 @ `86abfffb`, PR #2945; sha256 `3c1f83be…`, byte-identical to the `pypto-profiling/collectives/a1_alltoallv/harness/` copy).

## Why

K2 (`e48937b9`) merged to `main` after the frozen A1 baseline **B1** (`d626aea16`) was cut. This campaign re-measures current main with the frozen A1 protocol to check whether B1 is still a valid denominator. Two **L=16 spot cells** additionally record the **pre-K3 multi-block state** — the operating point K3 (PR #2900) claims −52…−59% at 1 MiB/peer.

## Protocol

Frozen A1, protocol-matched to the E2.1 series:

- EP8, devices 0–7, `SIMPLER_COMM_FORCE_IPC=1`, persistent blocks, warmup 5, **30 timing-slot rounds + 8 swimlane rounds**, uniform counts, L=1, reps=1; driver `collectives/alltoallv_a1.py` (retries 4, cooldown 60 s, heal = EP4 smoke on devices 0–3).
- Window: 09:03:46 → 09:45:40 UTC. One absorbed failure: `managed-l2 @ 65536` attempt 1 → `rc=-11`; heal EP4 OK → attempt 2 OK (known ≥65–85-dispatch stall hazard; 30r is the stable protocol on this box).
- L=16 spots: `managed-host`, `--core-num 16`, same 30r protocol structure; `launched_B=16` verified in JSON; window 09:48:07 → 09:50:02 UTC; attempt 1 for both.
- All cell JSONs stamped `PYPTO_COMMIT=54521824f277` (the B1-era refs still carry `unknown` stamps).

## Results — L=1 on main vs B1 (µs)

| rail | payload | main p50 | main mean | B1 ref p50 | Δ% (p50) |
|---|---:|---:|---:|---:|---:|
| managed-l2 (`aicore_gang_span`) | 8 192 | 164.04 | 163.68 | 164.14 ᴱ² | −0.1 |
| managed-l2 | 24 960 | 573.88 | 595.96 | 576.13 ᴱ² | −0.4 |
| managed-l2 | 32 768 | 610.87 | 697.35 | 610.59 ᴱ² | +0.0 |
| managed-l2 | 65 536 | 1 206.75 | 1 211.38 | 1 211.61 ᴱ² | −0.4 |
| managed-l2 | 1 048 576 | 19 118.34 | 19 133.13 | 19 101.49 ᴱ¹ | +0.1 |
| managed-host (`timing_slot`) | 8 192 | 5 338.43 | 5 333.35 | 5 534.42 ᴱ² | −3.5 |
| managed-host | 24 960 | 13 107.80 | 13 151.01 | 12 244.17 ᴱ² | +7.1 |
| managed-host | 32 768 | 5 999.97 | 5 944.79 | 5 854.94 ᴱ² | +2.5 |
| managed-host | 65 536 | 6 677.81 | 6 779.75 | 6 272.90 ᴱ² | +6.5 |
| managed-host | 1 048 576 | 31 177.31 | 31 598.08 | 30 723.20 ᴱ¹ | +1.5 |

ᴱ¹ E1 @100r (`issue-2521-a1-b1-2026-09-25`). ᴱ² E2.1 k1-side @30r (`issue-2521-a1-b1-k1ep8-swap-2026-09-28/k1`: B1 tree + K1 template).

**Verdict:** the L2 rail matches B1 to **±0.4%** (30r↔100r drift ≤0.8%, E2.1 Result 2) — the K2-era merges do not perturb the kernel rail. HOST deltas (−3.5…+7.1%) sit inside the box's documented repeat scatter: the E1.5 anchor 24 960 repeat spread **[12 554.75, 13 194.11]** contains main's 13 107.80; E2.1 measured ±6% cross-day on HOST. **No actionable regression; B1 remains a valid A1 denominator.**

## Results — L=16 spots (pre-K3 multi-block state)

`managed-host`, `--core-num 16` (B=16 blocks/rank, one per core):

| payload | p50 | mean | own L=1 p50 | Δ vs L=1 (p50 / mean) | egress (Gbit/s/rank) |
|---|---:|---:|---:|---:|---:|
| 24 960 | 12 247.9 | 12 566.9 | 13 107.8 | −6.6% / −4.4% | 0.111 |
| 1 048 576 | 30 790.2 | 31 390.5 | 31 177.3 | −1.2% / −0.7% | 1.87 |

- **Main's multi-block path buys almost nothing at 1 MiB** (0.99× vs its own L=1): no lane scaling exists pre-K3 at this operating point.
- **K3-repro reference** (`issue-2521-k3-repro-2026-09-28`, branch `cf136e73`, same rail/protocol family, 5-rep): L=16 @ 1 MiB/peer = **12 504.6 µs mean** (per-rep 11 403–13 137), egress 4.71 Gbit/s/rank. Main's L=16 operating point is **2.51× slower** (31 390.5 vs 12 504.6 mean; **−60.2%**) and egress 2.5× lower. These spot cells therefore pin the exact gap K3 closes: **gate K3's post-merge L=16 / 1 MiB cell against 30 790.2 (p50) / 31 390.5 (mean)** here.
- 24 960 (the A1 dispatch-tick payload): only −6.6%; K3-repro's smallest cell (16 KiB) was itself a noise sign-flip — no lane-scaling expectation at small payloads.

## Artifacts

- `json/` — 12 cell JSONs (10× L=1, 2× L=16); `summary.json`; `campaign_meta.json`
- `attempt_logs/`, `heal/` — per-attempt harness logs; heal smoke record for the L2@65536 absorbed failure
- `driver/` — `main_run.sh` (campaign driver), `main_l16.sh` (L=16 spot driver)
- `builds/` — per-cell build/capture records (dfx host logs, dispatch identities); **local-only** — `reports/**/builds/` is gitignored repo-wide
- on-box logs: `/opt/a1/main_run.log`, `/opt/a1/main_l16.log`

## Caveats

- L=16 was measured on `managed-host` only (K3's comparator rail); no aicore-rail L=16 cells.
- Harness version delta vs the B1 refs: refs used `dd960384…` (pre-e42) vs this run `3c1f83be…` (e42). The e42 changes are additive (completion metric, capture flags) plus trace-view dedup; validated to reproduce archived outputs with 0 diffs; the timing-slot path is untouched.
- Shared box; the HOST rail has ±6% cross-day scatter — small HOST deltas are not attributable; L2 is the tight rail.
