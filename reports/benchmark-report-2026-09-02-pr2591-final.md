# PR #2591 — Final performance impact report (2026-09-02)

**Repo:** `pypto-profiling` · **Date:** 2026-09-02 · **Platform:** a2a3 (8× Ascend 910B2, CANN 9.0.0)
**Interpreter:** `/usr/local/python3.12.13/bin/python3` (pypto editable, runtime pinned `dbdd041e`)
**Harness:** `collectives.run_sweep pair-mesh` · `pypto-composite`, `--persistent`

## 1. What was compared

| label | commit | generated-kernel notify path | wait-loop `dcci` |
|---|---|---|---|
| BASE | `49aea216` (origin/main at the branch cut — the pre-2591 state) | `pipe_barrier(PIPE_ALL)` before every `TNOTIFY` | whole-cache `dcci` after every `TWAIT` (`(P−1)`/generation) |
| HEAD | `58608212` (PR #2591 head, `perf/composite-slim-peer-dcci-barrier`) | **no barrier** before `TNOTIFY` (PTOAS owns pre-notify sync) | batched: one whole-cache `dcci` after a pure wait-loop (+ tri-state `WaitPurity`) |

Python is identical across both (the branch is C++-only); only the compiled `pypto_core.so`
differs (codegen-verified: BASE emits `pto.barrier <PIPE_ALL>` before `tnotify`; HEAD emits none).
This isolates PR #2591's change. (Current `origin/main` `d9d3dd6` has diverged past this base with
unrelated runtime/compiler work; a head-vs-today-main run would not isolate the PR.)

## 2. Method

- **Ranks:** P = 2 (d4–5), 4 (d4–7), 8 (d0–7). **Payloads:** 65536 / 262144 / 1048576 fp32
  elements per rank (64 KB–4 MB). Every leg: 3 warmup + **20 timed** `rt.run()` rounds,
  `--persistent`, box-health CommDomain probe + `allreduce_sum_v1` golden check (all PASS).
- **Interleaving:** 2 balanced rounds per rank — r1 BASE→HEAD, r2 HEAD→BASE — to cancel
  drift; metric `device_wall_s_median` per leg; reported as median- and min-of-legs deltas.
- **taskqueue note:** `task-submit` was requested but is **not provisioned in this container**
  (symlink → empty `/var/lib/taskqueue`, verified twice). Runs were **direct** (the original
  plan-98 method), gated by the harness box-health probe. Re-run under taskqueue if required
  for device-lock strictness; the numbers themselves are unaffected by the wrapper.
- `npu-smi` AICore % is a known **visual artifact** on this box (not a contention signal) and
  was ignored; no other processes were present on any device during the run.

## 3. Results (`device_wall_s_median`, µs; 2 legs per build per (P,count))

| P | count | BASE legs | HEAD legs | min Δ | median Δ |
|---|---|---:|---:|---:|---:|
| 2 | 64K | 361, 262 | 289, 233 | −11.1 % | **−16.2 %** |
| 2 | 256K | 661, 450 | 608, 458 | +1.8 % | **−4.1 %** |
| 2 | 1M | 997, 1413 | 895, 1316 | −10.2 % | **−8.3 %** |
| 4 | 64K | 891, 845 | 742, 793 | −12.2 % | **−11.6 %** |
| 4 | 256K | 1379, 1610 | 1302, 1464 | −5.6 % | **−7.5 %** |
| 4 | 1M | 2958, 2913 | 2819, 2492 | −14.5 % | **−9.5 %** |
| 8 | 64K | 2185, 2386 | 2313, 2272 | +4.0 % | +0.3 % |
| 8 | 256K | 2703, 2962 | 2898, 2900 | +7.2 % | +2.3 % |
| 8 | 1M | 6232, 5935 | 6021, 6095 | +1.4 % | −0.4 % |

Negative Δ = HEAD faster. Correctness: every timed round passed the golden check on every leg.

## 4. Verdict — performance impact of PR #2591

1. **Real, on-device improvement at P=2 and P=4 across all measured payloads**: median-of-legs
   **−4 % to −16 %**, min-of-legs up to −14.5 %; the ≥256K rows (where the mechanism — batching
   whole-cache `dcci` + dropping the `PIPE_ALL` drain per barrier generation — should show) give
   roughly **−4 to −10 %** median at P2/P4.
2. **No measurable benefit at P=8** (0 ± 2 %). At 8 ranks the mesh allreduce is dominated by
   O(P) traffic and waiting on the slowest peer; the removed local serialization is no longer on
   the critical path. Report P=8 as "no measurable effect", not as a regression (the small
   positive deltas are within leg-to-leg spread).
3. **Magnitude is smaller than the original −16…−32 % headline.** That figure came from the
   earlier best-leg/"vs best BEFORE" summaries; with 20 reps × 2 balanced legs and median-of-legs
   here, the honest clean-box number for P2/P4 is single-digit to low-teens percent, and ~0 at P8.
   (A same-session PIPE_V-vs-removal head-to-head to verify "removal has the same win as PIPE_V"
   was interrupted; the PIPE_V (`1ddeb601`) reproduction earlier the same day showed −11…−16.5 %
   medians at ≥256K against the same BASE — cross-campaign, so treat as indicative, not paired.)
4. `execute_s` (host dispatch) is flat — the change is purely on-device, as claimed.

## 5. Limitations

- Leg-to-leg drift on this shared box is large (e.g. P2/1M BASE 997→1413 µs across legs);
  medians resist it but 2 legs/build/case is the minimum for a stable point — flagged rows with
  spread > 2 should be read as ranges.
- 64K rows are noisy (spread ratios up to ~17); their deltas should be treated as noise-floor
  signals, not precise points.
- Runs were direct (taskqueue not provisioned); device locks were implicit (box otherwise idle).
- Single platform (910B2, PTOAS v0.57-pinned box install, runtime `dbdd041e`).

## 6. Artifacts

- Campaign results: `results/campaigns/pr2591_final_{base,head}_{r1,r2}/` (gitignored raw data)
- Driver: `pypto-3.0-notes/pr_plans/98-run-scripts/pr2591_final_campaign.sh`
- Analyzer: `pypto-3.0-notes/pr_plans/98-run-scripts/pr2591_final_analyze.py`
- Builds: BASE `/tmp/pypto_core_before_49aea216.so`, HEAD `/tmp/pypto_core_after_58608212.so`
  (both codegen-verified via `check_notify_barrier.py`)
- Plan-98 note: `pypto-3.0-notes/pr_plans/98-pypto-composite-slim-per-peer-dcci-barrier.md`
