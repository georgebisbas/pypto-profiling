# K3 (PR #2900) head `cf136e73` — A1 protocol @ EP8 (L=1 sweep + L=16 spots)

**Status: COMPLETE (2026-09-29).** 10/10 L=1 cells OK, 2/2 L=16 cells OK, 0 failures.

**Tree:** `feat/all-to-all-v-multiaiv-lanes` @ `cf136e736566200606a6a2c562731f05bc0e102b` (PR #2900 head at measurement time; PR still **open, not merged**) · **Worktree:** `/opt/a1/k3` · **Runtime:** `6e383fc5` + FORCE_IPC patch `a88e4442…` (same as B1/main) · PTOAS 0.65 · CANN 9.0.0 · **Harness:** e42 `3c1f83be…` (same file both sides of the comparison) · **Box:** hng-atlas01.

## Why

Same-A1 measurement of the K3 head so the merge gate is apples-to-apples with the main snapshot (`reports/issue-2521-a1-main-2026-09-29`, measured ~1 h earlier, same box/protocol):

1. **L=1 must not regress** (the shared path must behave identically).
2. **L=16 at 1 MiB/peer must deliver the claimed lane scaling** (claimed −58.9%, EP8).

## Protocol (identical to the main snapshot)

- EP8, devices 0–7, `SIMPLER_COMM_FORCE_IPC=1`, persistent, warmup 5, **30 timing-slot + 8 swimlane rounds**, uniform, L=1; driver `alltoallv_a1.py` (retries 4, cooldown 60 s, heal EP4 on 0–3). Window 10:13:37 → 11:03:09 UTC.
- L=16 spots: `managed-host`, `--core-num 16`, 30r; window 11:03:50 → 11:05:43 UTC; attempt 1 both.
- All cell JSONs stamped `PYPTO_COMMIT=cf136e736566`.

## Results — L=1: K3 vs main vs B1 refs (p50, µs)

| rail | payload | main `54521824` | **K3 `cf136e73`** | Δ% vs main | vs B1 ref |
|---|---:|---:|---:|---:|---:|
| managed-l2 | 8 192 | 164.04 | 163.87 | −0.1 | −0.2 ᴱ² |
| managed-l2 | 24 960 | 573.88 | 576.64 | +0.5 | +0.1 ᴱ² |
| managed-l2 | 32 768 | 610.87 | 611.68 | +0.1 | +0.2 ᴱ² |
| managed-l2 | 65 536 | 1 206.75 | 1 206.65 | −0.0 | −0.4 ᴱ² |
| managed-l2 | 1 048 576 | 19 118.34 | 19 148.51 | +0.2 | +0.2 ᴱ¹ |
| managed-host | 8 192 | 5 338.43 | 5 260.63 | −1.5 | −4.9 ᴱ² |
| managed-host | 24 960 | 13 107.80 | 12 250.35 | −6.5 | +0.1 ᴱ² |
| managed-host | 32 768 | 5 999.97 | 5 902.09 | −1.6 | +0.8 ᴱ² |
| managed-host | 65 536 | 6 677.81 | 6 327.78 | −5.2 | +0.9 ᴱ² |
| managed-host | 1 048 576 | 31 177.31 | 30 588.49 | −1.9 | −0.4 ᴱ¹ |

ᴱ¹ E1 @100r (`issue-2521-a1-b1-2026-09-25`). ᴱ² E2.1 k1-side @30r (`issue-2521-a1-b1-k1ep8-swap-2026-09-28/k1`).

**Verdict:** **L2 parity to ±0.5%** — the shared path is behaviorally unchanged. HOST deltas (−1.5…−6.5%) put K3 at the low end of the established cross-session spread rather than below it: K3's 24 960 = 12 250.35 ≈ E2.1's 12 244.17 (the main snapshot ran high at 13 107.80), and K3's 1 MiB = 30 588.49 sits inside E1.5's repeat range [30 303.10, 30 625.28]. **No L=1 regression attributable to K3.** (p50 is the A1 headline; means skew on mid payloads for both trees alike.)

## Results — L=16 spots: the merge gate (managed-host, B=16/rank)

| payload | main p50 | K3 p50 | Δ (p50) | main mean | K3 mean | Δ (mean) | egress (Gbit/s/rank) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 960 | 12 247.9 | 12 337.8 | +0.7% | 12 566.9 | 12 378.8 | −1.5% | 0.111 → 0.113 |
| **1 048 576** | 30 790.2 | **12 893.6** | **−58.1%** | 31 390.5 | **12 720.5** | **−59.5%** | 1.871 → **4.616** |

- **Claimed lane scaling delivered.** −58.1% (p50) / −59.5% (mean) at 1 MiB/peer vs main — matching PR #2900's −58.9% claim and the independent repro (`issue-2521-k3-repro-2026-09-28`: 5-rep mean 12 504.6 µs; ours 12 720.5, **+1.7%**, inside the repro's per-rep spread 11 403–13 137). Egress 4.62 vs claimed 4.76 / repro 4.71 Gbit/s/rank.
- vs K3's own L=1: **−58.3%** (mean).
- 24 960: unchanged (≈ main within ±1.5%) — no lane-scaling effect at small payloads, as expected.
- **Merge-gate values** (pinned in the main report as 30 790.2 p50 / 31 390.5 mean): K3 head reads **12 893.6 / 12 720.5 → gate met**.

## Reliability

- 10/10 + 2/2 OK; heal ladder exercised once (`managed-host @ 24960`: attempts 1–2 `rc=-11` → heal EP4 → attempt 3 clean at 12 250.35). Known ≥65–85-dispatch stall hazard; 30r protocol otherwise stable.
- Every cell stamped; toolchain identical to the main snapshot (runtime `6e383fc5` + same patch, PTOAS 0.65, CANN 9.0.0).

## Artifacts

- `json/` — 12 cell JSONs (10× L=1, 2× L=16); `summary.json`; `campaign_meta.json`
- `attempt_logs/`, `heal/` — per-attempt logs; heal records for the host@24960 absorbed failures
- `driver/` — `k3_run.sh` (campaign) + `k3_l16.sh` (L=16 spots)
- `builds/` — per-cell build/capture records; **local-only** (gitignored repo-wide)
- on-box logs: `/opt/a1/k3_run.log`, `/opt/a1/k3_l16.log`

## Caveats

- Single session; HOST rail has ±6% cross-session scatter — L2 is the tight rail and shows parity.
- Measured at head `cf136e73` (2026-09-29 ~10:13–11:06 UTC); if the PR head moves, re-gate.
- Harness e42 `3c1f83be…` — same file used for the main snapshot, so the comparison is harness-neutral.
