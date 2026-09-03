# Plan 101 Phase 0 — reset-cost campaign + `--profile l2 --persistent` crash repro (2026-09-03)

**Status:** executed. Task A (CLI exposure of `reset_persistent_windows`) shipped to
pypto-profiling `main` (`878558d`). Task B (NPU reset-cost tax) measured — 5 cells × 3 legs,
15/15 PASS; verdict below. Task C (`--profile l2 --persistent` crash) **reproduced
deterministically with full traceback** + first-pass root cause; needs its own runtime fix.
Plan 101 (`101-pypto-persistent-domain-default.md`) Phase-0 prerequisites #3 (reset cost) and
#4 (l2 crash) are now measured/repro'd; backlog row 101 updated.

---

## 0. Task A — expose `reset_persistent_windows` on the sweep CLI (shipped)

`pair-mesh` gained `--reset-persistent-windows {true,false}` (default `None` = runtime
default), threaded through `_cmd_pair_impl` → `_run_stack_multi` → `_run_pypto_campaign` →
`get_pypto_session`.

**Cache-key landmine fixed:** `_session_key` omitted `reset_persistent_windows`, so two legs
differing only in that flag silently reused the first run's cached session. The flag is now
part of the key (`None` == legacy key → no surprise rebuilds). Verified on hardware: flipping
only the reset flag between two `get_pypto_session` calls produces a fresh session object
(different `id`, different `init_s`), identical config reuses the cache. Unit tests added
(`tests/test_pypto_session_key.py`, 3 tests). Commit `878558d` on `main` (24 pre-existing +
3 new tests pass). No new pypto branch needed — the runtime mechanism is already on pypto
`main` (#2163).

**Note for Task B/C below:** `distributed_runner._resolve_persistent_window_reset` resolves
`None` → `reset = persistent`, i.e. **reset is already the default for `--persistent`**. The
`reset_true` legs below therefore *are* today's status-quo persistent behaviour; `reset_false`
is the reset-free alternative plan 101 can offer.

## 1. Context

Plan 101's Phase-0 prerequisite #3 asks: does `reset_persistent_windows=True` (the
semantics-preserving default plan 101 wants for Phase A) eat a meaningful slice of the measured
19–25× `execute_s` win, or is it small enough to ignore? The docs (pypto
`docs/en/dev/06-persistent-l3.md`) predict one synchronous whole-buffer host zero-copy per
dispatch, so the cost should scale with payload. This campaign sweeps payload size × P with the
newly exposed flag. Prerequisite #4 asks for the first real repro + root-cause read of the
known `--profile l2 --persistent` crash.

## 2. Environment

| Component | Value |
|---|---|
| Box | 8× Ascend 910B2 (a2a3), shared/multi-tenant; box-health probe passed before every leg |
| pypto | `origin/main` `d9d3dd60` (editable) |
| simpler runtime | `/opt/pypto/runtime` `15f5cbd` (pypto main's submodule pin) |
| pypto-profiling | `main` `878558d` (this task's Task A commit) |
| Interpreter | `/usr/local/python3.12.13/bin/python3` (has pypto + torch) |
| Protocol | `collectives.run_sweep pair-mesh`, `pypto-composite`, mesh, fp32, `core_num=1`, 3 warmup + 15 timed; metric `execute_s` median + `device_wall_s` median |
| Devices | P=2 → d0-1; P=4 → d4-7 (per plan-101 evidence cells) |
| Payloads | counts 65536 / 262144 / 1048576 fp32 (256 KiB / 1 MiB / 4 MiB per rank) |
| Scheduling | direct runs on this dev box (task-submit/taskqueue unavailable by owner instruction); all legs on the same device sets in one ~25 min window |

Raw outputs: `results/campaigns/plan101_reset_cost/p{2,4}_c{...}/{nonpersistent,reset_false,reset_true}.json`
(logs alongside). Task C: `results/campaigns/plan101_l2_crash_repro/`.

## 3. Task B — reset-cost table (execute_s, ms)

Three legs per cell: `nonpersistent` (reference O), `persistent --reset-persistent-windows
false` (M, reset-free), `persistent --reset-persistent-windows true` (N, semantics-preserving
default). All 15/15 correctness PASS.

| Cell | Payload/rank | nonpers exe med | reset=false exe med | reset=true exe med | reset cost (N−M) | device_wall med false→true |
|---|---:|---:|---:|---:|---:|---:|
| P=2 d0-1 c=65536 | 256 KiB | 111.96 ms | 2.69 ms | 5.53 ms | **+2.84 ms** | 0.316→0.353 ms |
| P=2 d0-1 c=262144 | 1 MiB | 115.59 ms | 4.78 ms | 9.43 ms | **+4.66 ms** | 0.491→0.622 ms |
| P=2 d0-1 c=1048576 | 4 MiB | 28.27 ms ⚑ | 6.42 ms | 19.34 ms | **+12.92 ms** | 1.393→1.197 ms |
| P=4 d4-7 c=65536 | 256 KiB | 170.65 ms | 3.66 ms | 9.74 ms | **+6.08 ms** | 1.068→0.750 ms |
| P=4 d4-7 c=1048576 | 4 MiB | 229.04 ms | 9.85 ms | 26.21 ms | **+16.36 ms** | 3.174→2.977 ms |

⚑ The P=2 c=1048576 nonpersistent leg was noise-contaminated (spread 6.6, mean 68.8 ms,
non-monotonic vs the 1 MiB cell) — treat that single reference as lower-bound/noisy; the
plan-101 evidence-table reference (114 ms P2 / 220 ms P4 non-persistent at the headline
config) is consistent with the other four cells.

**Findings:**

1. **The reset tax is real, per-dispatch, and scales with payload** — as the docs predict:
   ≈ +2.8 ms @256 KiB, +4.7 ms @1 MiB, +13 ms @4 MiB (P2); ≈ +6.1 ms @256 KiB, +16.4 ms @4 MiB
   (P4). Roughly linear at ~2.5–4 ms per MiB plus a ~2 ms fixed component (the synchronous
   whole-buffer host re-stage). Reset true→false execute_s is tight (spread 1.3–2.7), so the
   delta is not box noise.
2. **`device_wall_s` is flat between reset=false and reset=true** (0.32→0.35, 0.49→0.62,
   1.39→1.20, 1.07→0.75, 3.17→2.98 ms — deltas all within each leg's device-wall spread of
   2–10×, mixed sign). Confirms reset touches host staging only, never device compute.
3. **Win framing** (median, vs each cell's nonpersistent leg):

| Cell | reset=false win | reset=true win | erosion from reset |
|---|---:|---:|---:|
| P=2 256 KiB | 41.6× | 20.2× | ~2.8 ms of ~109 ms saved |
| P=2 1 MiB | 24.2× | 12.3× | ~4.7 ms of ~111 ms saved |
| P=2 4 MiB | 4.4× | 1.5× ⚑ | ~13 ms of ~22 ms saved |
| P=4 256 KiB | 46.7× | 17.5× | ~6.1 ms of ~161 ms saved |
| P=4 4 MiB | 23.2× | 8.7× | ~16 ms of ~203 ms saved |

### Verdict (Phase A `reset_persistent_windows=True` default)

**Safe to ship as the semantics-preserving default for payloads ≤ 1 MiB/rank** (the collective
regime plan 101 targets): the reset adds 3–6 ms on top of a 3–10 ms persistent `execute_s`,
vs 110–230 ms non-persistent lifecycle — the 19–25× win degrades only to ~12–20× (P2) / ~9–17×
(P4). That is a small, bounded erosion, and reset-on is already the shipped default today
(no regression risk in the flip itself).

**Large-payload caveat (≥ 4 MiB/rank):** the reset tax (~13–16 ms) approaches or exceeds the
reset-free persistent `execute_s` (6–10 ms) and at P=2 can consume most of the lifecycle win
(nonpersistent reference there is noisy but the trend is unambiguous). For MB-scale window
workloads the default should either auto-skip reset when windows are not read back between
dispatches (the common device-consumed collective case), or expose the reset-free path
(`--reset-persistent-windows false` / `reset_persistent_windows=False`) as the documented
choice. Recommend plan 101 records a payload threshold and a follow-up optimization ask on the
reset implementation itself (~2.5–4 ms/MiB is a slow host path).

**Ecosystem precedent — pypto-lib already ships `persistent=True` + `reset=False`:**
pypto-lib's serving/benchmark dispatch paths (`golden/runner.py` `_dispatch_resident` ~line
1161 and non-resident L3 benchmark ~line 807; `models/deepseek_v4_pro/synthetic_token_loop.py`
`_create_persistent_worker` ~line 184) all pass `reset_persistent_windows=False`, with the
explicit rationale *"retain CommDomains across rounds and let kernels clear their own signal
windows."* So the biggest existing persistent consumer never host-reads windows and already
made the reset-off choice — independent support for making reset opt-in/auto rather than a
global on-default, and for promoting Phase C (reset-free fast path) to first-class in plan 101.

## 4. Task C — `--profile l2 --persistent` crash: repro + root cause

### 4.1 Repro (deterministic)

Command (P=2, c=65536, mesh, d0-1): `run_sweep pair-mesh --stacks pypto-composite
--persistent --profile l2 --warmup-rounds 3 --timed-rounds 15`. Reproduced 4/4 across the full
run and minimal 1-warmup+1-timed isolates; exit 139 (SIGSEGV). Log:
`/tmp/plan101_l2_repro.log`; DFX artifacts under
`results/campaigns/plan101_l2_crash_repro/cases/mesh_p2_count65536_fp32_a2a3_d0-1/pypto-composite/dfx/dfx_outputs/`.

**Observed sequence (full run):**
1. Warmup round 1's l2 two-pass completes cleanly on both ranks (dep_gen graph pass + clean
   timing pass → `merged_swimlane_*.json` written for rank0 and rank1).
2. Warmup round 2's dep_gen pass ("run 1/2") fails; the unwind reaches the session `close()`
   in `_run_pypto_campaign`'s `finally`, where the **persistent-domain release** raises:

```
RuntimeError: release_domain(allocation_id=0) failed on 2/2 chips; first error chip=0:
control_release_domain failed: mailbox has an unresolved timed-out control command
```

Full traceback (abridged to the interesting spine):

```
run_sweep.py:725  close_pypto_session()                      # finally of _run_pypto_campaign
pypto_own.py:570  self._rt.close()
distributed_runner.py:3003  self._w.close()
distributed_runner.py:2988  self._release_persistent_domains(self._persistent_domains_by_program)
distributed_runner.py:2036  handle.release()
worker.py:8927  _release_domain_handle → 8996 _free_domain_after_fence → 9068 _release_domain_now
worker.py:9101  release → 9154 _dispatch_control_domain
RuntimeError: release_domain(allocation_id=0) failed on 2/2 chips; first error chip=0:
control_release_domain failed: mailbox has an unresolved timed-out control command
```

A minimal direct-session script (2 rounds) captured the **original round-2 error that the
finally-close masks**:

```
round 2: RAISED RuntimeError: LocalMailboxEndpoint child failed (worker_id=0, code=1):
chip_process dev=0: chip run: RuntimeError: chip run lane is poisoned:
finalize_native_run failed with code 507901 (run_id=3 slot=0 generation=3 dispatch_id=3 run_epoch=3)
```

**Isolation matrix (all P=2 c=65536 d0-1, 1 warmup + 1 timed):**

| Config | Result |
|---|---|
| `--profile l2` (no persistent) | ✅ PASS, exit 0 |
| `--persistent` (no profile) | ✅ PASS (Task B: 15/15 legs incl. this cell) |
| `--persistent --profile l2` (reset default on) | 💥 crash on 2nd dispatch, exit 139 |
| `--persistent --profile l2 --reset-persistent-windows false` | 💥 crash on 2nd dispatch, exit 139 (`child process … exited before mailbox completion`) |

→ the trigger is the **retained-domain dispatch path × l2-profile's two-pass DFX**, independent
of the reset policy. Non-persistent l2 (fresh domain per pass) and persistent-without-l2 both
compose fine. Box health re-verified after the crashes (plain persistent run PASS on d0-1) —
the crash does not wedge the chips.

### 4.2 Root-cause read (first pass)

The "mailbox has an unresolved timed-out control command" string lives in the pinned runtime at
`/opt/pypto/runtime/src/common/hierarchical/worker_manager.cpp` (lines 883 and 1209–1211).
`run_control_command` (worker_manager.cpp:1208) issues a mailbox control sub-command and waits
for the child's CONTROL_DONE. Two ways the endpoint is left poisoned:

- an **explicit finite-timeout control** expires (`mailbox_control_timed_out_` set at
  ~line 1226), or
- the **child dies** while a command is in flight (liveness check, ~line 1238–1240, also sets
  `mailbox_control_timed_out_`).

Once set, *every* later control — including `control_release_domain`, which otherwise uses the
infinite timeout — refuses with "mailbox has an unresolved timed-out control command". The
repro's original error is the second case: round 2's dispatch kills the child, the parent's
endpoint is poisoned through the child-death path, and `close()` → `_release_persistent_domains`
(worker.py `_release_all_live_domains`, via the Simpler cleanup journal) then fails on the
first `control_release_domain`.

The underlying device failure is **`finalize_native_run` code 507901 =
`ACL_ERROR_RT_SOCKET_CLOSE` ("hdc disconnect")** on chip 0 (CANN
`rt_error_codes.h`; runtime comments at `a2a3/platform/onboard/host/device_runner.cpp:142` and
`device_runner_base.cpp:1413` document 507901 as the teardown code that surfaces when a stream
is left in an error state, e.g. after an AICore/DFX-timeout cascade). In `--persistent --profile
l2`, round 1's two-pass DFX (dep_gen + swimlane collectors armed on the retained domain) is
followed by round 2 re-dispatching on the **same retained CommDomain**; the second dispatch's
native-run finalize then trips the HDC disconnect while the l2 collectors/teardown are still in
flight — the child run lane is poisoned, the child exits, and the parent-side mailbox poison
surfaces at domain release with the exact message plan 101 recorded. This is consistent with
plan 101's hypothesis that the persistent-domain release path
(`_release_persistent_domains`) collides with l2-profile's DFX teardown — with the refinement
that the collision is planted one dispatch earlier (round 2's `finalize_native_run`), and only
*surfaces* at release/close. Whether 507901 is itself caused by a mailbox control command whose
30 s explicit timeout (`worker.py _py_control_timeout_s = 30.0`, used for L3↔L2 domain
controls) expired during the DFX drain is the remaining open question — the round-2 error is
reported as a child-side run-lane failure, not a parent-side control timeout, so the 
parent-side poison is most likely the child-death path.

### 4.3 Recommendation

**Not a quick harness fix — needs its own runtime investigation.** The failure is a
runtime/device interaction (retained CommDomain × l2 two-pass DFX → 507901 HDC disconnect →
child exit → poisoned mailbox at release), not a pypto-profiling wiring bug. Suggested next
steps for the runtime owner (filed as plan 101 Phase-0 prerequisite #4, now with a real repro
to work from):

1. Capture the child-side CANN/dlog for run_id=3's `finalize_native_run` to pin whether 507901
   is a DFX-collector drain race or a stream-state carryover between round 1's pass 2 and
   round 2's pass 1 on the retained domain.
2. Determine whether the retained domain should be fenced/quiesced (collector drain completed)
   before the next persistent dispatch when swimlane/dep-gen is armed — or whether the l2
   two-pass should force a transient domain per pass (that is what makes non-persistent l2
   safe).
3. Workaround stands (already in pypto-profiling AGENTS.md §7): do not combine `--profile l2`
   with `--persistent`; l2-profile runs use the transient path.

## 5. Data-quality caveats

- Shared box, direct runs: the nonpersistent P=2 legs at c=262144/c=1048576 were noisy (spread
  6.6–12); persistent legs were tight (spread ≤ 2.7). Medians used throughout; the ⚑ cell's win
  is lower-bound.
- The reset=true legs are the status-quo persistent default, so they carry the same box noise
  profile as all prior plan-101 evidence.

## 6. Deliverables / follow-ups

- pypto-profiling `main` `878558d` — `--reset-persistent-windows` flag + session cache-key fix
  + tests (Task A).
- Campaign data: `results/campaigns/plan101_reset_cost/` (Task B), `results/campaigns/
  plan101_l2_crash_repro/` + `/tmp/plan101_l2_repro.log` (Task C).
- This report.
- pypto-3.0-notes: plan 101 Phase-0 prerequisites #3/#4 + backlog row 101 updated (see plan).
