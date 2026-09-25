# RFC #2521 A1 — frozen-baseline tactic + experiment plan (2026-09-25)

**Scope.** Work item **A1** (“benchmark harness + pre-change baseline”) of
[pypto#2521](https://github.com/hw-native-sys/pypto/issues/2521). This file records the **tactic
change agreed 2026-09-25** and the **experiment program** that delivers A1 under it.
Plan-level docs (plan 110, the 2521 tracker) live in `pypto-3.0-notes`; a paste-ready change
record is at the bottom.

---

## 1. Decision

**The official frozen A1 baseline is B1 = `d626aea16`** (post-K1 + PTOAS v0.65) — **not** the
pre-K1 revision `166bf7ac4`.

The **pre-change comparison** is produced on the **same B1 tree** by swapping in the pre-K1
kernel template (the k1ab method), with the existing 2026-09-09→14 archive kept as a labelled
historical reference. EP16 remains **blocked** (8-chip box) and is recorded as such.

---

## 2. Why the tactic changed (evidence, 2026-09-25)

1. **The pre-K1 revision is era-locked.** `166bf7ac4` requires **PTOAS v0.63**; the box has
   v0.65, which rejects its generated `.pto` (`custom op 'pto.load_scalar' is unknown` —
   v0.65 was a breaking toolchain update, pypto#2837 adapted codegen to it). The environment
   *was* reconstructed (era wheel + python3.10 venv + worktree), but continuing to run and
   maintain it is “fighting a version that does not work”.
2. **B0’s counts path is unreliable.** Clean cards, EP8 `managed-l2` 1 KiB smoke (2026-09-25):
   - Fabric path, run 1: **FAIL** —
     `AssertionError: rank=0 src=4: recv_counts=0 != clamped send_counts=1`
   - Fabric path, run 2: pass (33.2 µs). IPC path: pass (32.4 µs).
   Pre-K1 publishes counts with **`TNOTIFY(Set)`** — precisely the a2a3 `Set` bug
   (pto-isa #322/#323, closed docs-only) that **K1 removed on purpose**. A denominator that
   is intermittently wrong (and can silently corrupt timed rounds) has negative value.
3. **No forward user.** K2/K3/K4/O2/A3 all sit on K1+. The only consumer of a pre-K1 number is
   *K1’s own delta* — already measured by the k1ab A/B at EP2/EP4 (**−1 % at the anchor**, not
   the 1.5–1.8× originally quoted). E2 below supplies the missing EP8 half.
4. **B1 is toolchain-exact** — runtime `6e383fc5` matches the box, PTOAS v0.65 matches, and it
   is the tree family the K2/K3 EP4/EP8 datasets were produced against. It is already built.

---

## 3. Pinned provenance

| Item | Value |
|---|---|
| B1 tree | `d626aea16` (post-K1 + v0.65) — build at `/opt/a1/b1` (pypto_core built 2026-09-25) |
| B1 runtime | `6e383fc5` + FORCE_IPC patch (patch verified to apply cleanly) |
| B1 kernel template | 305 lines, sha256 `70a6cb0943236357caf32bff2c43dc8842b03ad8ef3602a7e285ca4d3ea11066` (= K1 squash template) |
| Pre-change template | `5cb75663a^:…/all_to_all_v/templates/kernel.cpp.in` — 224 lines, sha256 `bb503e2baf5fbe7f157ff8defd0ae8f206001f6fbec82e8eae33c8b47fbe4526` (saved: `/opt/a1/prek1_kernel.cpp.in`) |
| PTOAS | v0.65 (box default). v0.63 only for the optional B0 appendix (`/opt/a1/ptoas-0.63`) |
| FORCE_IPC | **measurement condition for EP8** (all cells), documented in every archive |
| EP16 | blocked — hardware (8 chips); never substituted or interpolated |

Env recipe for every run (worktree isolation — see §6):

```bash
CANN=/usr/local/Ascend/cann-9.0.0
export A1_WORKTREE=<tree>            # /opt/a1/b1 (or /opt/a1/b0 for the appendix)
export PYPTO_ROOT=<tree> PYPTO_DIR=<tree>
export PYTHONPATH=/opt/a1/runenv:<tree>/runtime/examples/scripts:<tree>/runtime/python:<tree>/runtime:<tree>/python:$CANN/python/site-packages:$CANN/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe
export LD_PRELOAD=$CANN/aarch64-linux/lib64/libhccl.so
# EP8: export A2AV_FORCE_IPC=1 (driver) / SIMPLER_COMM_FORCE_IPC=1 (direct harness)
```

---

## 4. Experiment program (what A1 needs now)

| ID | Experiment | Config | Cells | Status |
|---|---|---|---:|---|
| **E0.1** | B1 runtime: FORCE_IPC patch + `build_runtimes --platforms a2a3` | no device | — | to run |
| **E0.2** | `_task_interface` check (B1 runtime HEAD == primary’s stamp `6e383fc5`) | no device | — | to check |
| **E0.3** | B1 smokes: EP2 1 KiB (dev 4,5) then EP8 1 KiB (dev 0–7, both rails) | 2 rounds | 3 | to run |
| **E1.1** | **EP8 official curve** — uniform, full payload sweep, L=1, both rails | 5+100 rounds, 8 swimlane, FORCE_IPC | 32 | to run |
| **E1.2** | **EP8 patterns** — zero, self-only, single-hot, mixed, asymmetric, random @ 24960, both rails | same | 12 | to run |
| **E1.3** | *(secondary)* EP4 uniform sweep + patterns, both rails | same, no IPC | 44 | optional |
| **E1.4** | Tails: `1, 15, 31, 33, 4159, 4161` B + `valid_elems<K`; uniform/mixed | EP8 | ≤8 | to run / document N/A |
| **E1.5** | R7 calibration: swimlane vs timing-slot ≤ 3–5 % on anchors | EP8 | 2–3 | to run |
| **E1.6** | *(optional)* R6 equivalence: `row_numel=1` at anchors | EP8, post-K1 | ≤6 | optional |
| **E2.1** | **K1 delta @ EP8** — pre-K1 template (pinned sha256) swapped into B1 tree; payloads `0, 4096, 8192, 16384, 24576, 24960, 32768, 65536`, both rails, both sides | 5+100, FORCE_IPC | 32 | to run |
| **E2.2** | *(appendix)* true-B0 EP8 uniform curve under `ptoas-0.63` + FORCE_IPC | era toolchain | 32 | optional |
| **E3.1** | Archive: JSONs + `campaign_meta.json` + **`provenance.json`** (pypto sha, runtime sha+patch sha, ptoas, CANN, devices, conditions) | — | — | with each run |
| **E3.2** | Open gaps recorded: HOST-rail AIV attribution (still no surface), tail-point INT8 alignment, EP16 blocked | — | — | docs |
| **E4.1** | Change record → plan 110 + tracker §A1 (§7 below) | — | — | docs |
| **E4.2** | In-tree harness PR (`tests/st/distributed/collectives/all_to_all_v_benchmark.py`; fix the `parents[4]`-style commit lookup) | — | — | separate track |

**Run order:** E0 → E1.1+E1.2 (one campaign) → E1.4/E1.5 → E2.1 → E3 → E4.
Campaign dirs: `reports/issue-2521-a1-b1-2026-09-25/` (E1), `…-k1ep8-swap-2026-09-25/` (E2.1).

---

## 5. Gap analysis — do we already have these experiments?

| Required | Exists today? | Where | Verdict |
|---|---|---|---|
| E1.1 uniform curve EP2/4/8, both rails | **partial** — full curves exist | `…-ubfix-2026-09-09/`, `…-ep8-gapfill-2026-09-10/` | on **INT8-branch tip `fcd92b97` + old toolchain** (PTOAS ~0.63, pto-isa `3b4faf67`) → corroboration only, **not** the frozen baseline → rerun on B1 |
| E1.2 all 7 patterns @24960, both rails | **partial** — EP8 has all 7 | `…-ep8-gapfill-2026-09-10/` | same provenance caveat → rerun on B1 |
| E1.3 EP4 curves | **yes** | ubfix, weird-recheck, K2/K3 campaigns | rerun only if EP4 comparability wanted |
| E1.4 tails (`1/15/31/…` B) | **no** | — | never run; old report: rejected by INT8 32-B row alignment → attempt once, else document |
| E1.5 R7 calibration | **no (not formalized)** | — | small check needed |
| E1.6 R6 equivalence (`row_numel=1`) | **no** | — | optional |
| E2.1 K1 delta @ EP8 (template swap) | **no** | k1ab = EP2/EP4 only | **run** |
| E2.3 K1 delta @ EP2/EP4 | **yes** | `…-k1ab-prek1/-k1-2026-09-22/` + `K1_AB_REPORT.md` (32/32 both sides; −1 % anchor) | do not rerun |
| E2.2 true-B0 EP8 | **no** (env built, smokes only) | `/opt/a1/b0` + smokes 13:57–14:26 | optional appendix; fabric-path flake documented |
| Support: K2/K3 EP4/EP8 datasets | yes | `…-k2-corenum-scaling-2026-09-23/`, `…-k3-lane-scaling*-2026-09-2{3,4}/`, `…-k3-bw-ceiling-2026-09-25/` | context only (K-track trees, B>1) |

**Short answer:** the *shape* of the experiment has been run before, but **nothing exists on the
frozen B1 tree under the official protocol**. The genuinely new runs are E1.1, E1.2, E2.1 (plus
the optional E1.3–E1.6).

---

## 6. Known environment gotchas (carried into the runs)

- Worktree isolation requires the `sitecustomize.py` preamble (`/opt/a1/runenv`) — the dev
  container’s editable installs otherwise pin `pypto`/`simpler` to `/opt/pypto`.
- Deliverable harness runs must not be copied to a shallow path: `read_pypto_commit()` uses
  `Path(__file__).resolve().parents[4]` (assumes the in-tree location
  `tests/st/distributed/collectives/`). Use the frozen bundle path as-is, or fix E4.2 first.
- Build: manual cmake needs `-Dnanobind_DIR=…`; resume builds with `MAKEFLAGS= cmake --build`
  (load-env’s exported `MAKEFLAGS` breaks the libbacktrace ExternalProject jobserver).
- EP8 hygiene: kill → verify all 8 cards idle → EP4 heal smoke → cooldown → run. Killed
  campaign cells can leave poisoned device state (`release_domain` SIGSEGV, simpler#2192).

---

## 7. Paste-ready change record (plan 110 / tracker §A1)

> **2026-09-25 — tactic change (A1).** Official A1 denominator is now **`d626aea16`** (post-K1 +
> PTOAS v0.65), measured with the A1 protocol at EP8 under `SIMPLER_COMM_FORCE_IPC=1`; the
> pre-change comparison is a **same-tree kernel-template swap** (pre-K1 `kernel.cpp.in`,
> sha256 `bb503e2b…`, k1ab method) and the 2026-09-09→14 archive (INT8-branch tip, old
> toolchain) stays as labelled historical reference. Pre-K1 revision `166bf7ac4` is **demoted**
> from “official baseline”: its toolchain is era-locked (PTOAS v0.63) and its TNOTIFY(`Set`)
> counts path is intermittently wrong on the Fabric path (observed
> `recv_counts=0 != clamped send_counts=1`, 2026-09-25). EP16 remains blocked (8-chip box).
> M1 is closed by the archive + swap pair, not by a pre-K1 rebuild.
