# AGENTS.md — Running PyPTO collective benchmarks from scratch

This file tells an agent (or human) exactly how to run the collective benchmark harness in
this repo, what is available, and how to avoid the known foot-guns. Read it top to bottom
before running anything on hardware.

Repo: **pypto-profiling** — personal benchmark harness for PyPTO vs HCCL vs simpler L3
collectives. Authoritative methodology: `pypto-3.0-notes/collectives_benchmarking/`.

---

## 1. What this repo is

- `collectives/` — Python package:
  - `stacks.py` — stack capability registry (hccl, simpler, simpler-own, pypto-composite,
    pypto-host, pto-isa) — the single source of truth for what each stack can run.
  - `equivalence.py`, `golden.py`, `metrics.py`, `model.py` — case model, golden outputs,
    timing parsers (`device_wall`, `HCCL_TIMED`), and the O+N/B bandwidth model.
  - `run_sweep.py` — the benchmark entry point (`pair-mesh`, `cross-variant`, `validate-case`).
  - `runners/` — in-process session runners for pypto and simpler-own; subprocess runners
    for hccl / simpler / pto-isa.
  - `summarize.py` — aggregation, paired comparison, `--model` scorecard, `--emit-report`.
  - `apples_to_apples.py` — **the HPC-view analysis**: decomposes `execute_s` into
    device + dispatch + lifecycle (persistent vs non-persistent), honest ratios vs HCCL,
    effective mesh bandwidth, scaling efficiency vs the mesh-inherent floor. Emits 6 figures.
  - `plot_figures.py` — 10 figures (scaling, efficiency, bw-crossover, wall-vs-device, model fit…).
  - `cases/` — pre-generated EquivalenceCase JSON files (P × count × dtype × variant × devices).
  - `kernels/` — the AIV kernel + orchestration shim for the simpler-own stack.
  - `alltoallv_a1.py` — RFC #2521 A1 AllToAllV campaign driver (EP × impl × payloads).
  - `a1_alltoallv/` — the A1 measurement bundle: a **snapshot** of the measured harness
    (`harness/all_to_all_v_benchmark.py`), the FORCE_IPC patch, the Fabric domain-churn MFE,
    and `PROVENANCE.json`. It is no longer the harness's only home: the in-tree copy lives at
    `pypto/tests/st/distributed/collectives/all_to_all_v_benchmark.py` (pypto PR #2945, branch
    `a2av-in-tree-harness`), and plan 129 (`a2av-harness-l2-core-num` @ `22fc5886`) stacks
    managed-l2 `core_num>1` on top of it. The copy here lags that work — read the sync-status
    block in `a1_alltoallv/README.md` before quoting or re-running it.
- `run_campaign.sh` — campaign driver: runs a sweep, **merges** per-(P,count) results into
  `results.json`, then runs summarize + plot.
- `tests/` — pure unit tests (no hardware): `python3 -m pytest tests/ -v` (27 tests, 2026-10-14).
- `results/` — campaign outputs (gitignored). `reports/` — committed reports + figures.

## 2. Prerequisites (the `pypto3-hw-native-sys:cann9` container)

Everything below is **inside the NPU container**. On a bare box (`hng-atlas02`) none of the
`/opt/*` or CANN paths exist, `import pypto` fails, and there is no `pto-isa`/`PTOAS`/`simpler`
sibling checkout. Values re-read 2026-10-14 from
`pypto-docker/Dockerfile.hw-native-sys.cann9.0` (the root-level file of the same name is a stale
older revision — ignore it).

| Thing | Where | Notes |
|-------|-------|-------|
| NPU | `/dev/davinci*` (8× Ascend 910B2) | **shared box, queue mandatory** — see §2.1 |
| CANN | `/usr/local/Ascend/cann-9.0.0` (`CANN_HOME`, also `ASCEND_HOME_PATH`) | baked into the image; never mount `/usr/local/Ascend` from the host |
| HCCL | `${CANN_HOME}/aarch64-linux/lib64/libhccl.so` | needed for the `hccl` stack; the image defines a `hccl-preload` bash function and deliberately leaves `LD_PRELOAD` unset |
| pypto | `/opt/pypto` (`PYPTO_DIR`) | pinned commit install — current image `PYPTO_COMMIT=4a82915408eaf8aaf0e84dca4c1eab5e92eefed6`; not editable (use the `.dev.` image for that) |
| simpler | `/opt/pypto/runtime` (`SIMPLER_ROOT`) | the **`runtime/` git submodule of the pypto checkout** — not a sibling clone, not a PyPI package |
| pto-isa | `/opt/pto-isa` (`PTO_ISA_ROOT`) | commit derived at build time from `runtime/pto_isa.pin` |
| PTOAS | `/opt/ptoas-bin` (`PTOAS_ROOT`) | py3.10 venv holding the v0.65 cp310 wheel; **appended** to `PATH`, never prepended, so it cannot shadow the system py3.12 |
| **Python** | **bare `python3`** = `/usr/local/python3.12.13/bin/python3` | the py3.12 interpreter that has pypto + torch. The py3.10 `/opt/ptoas-bin` venv is *appended* to `PATH`, so it does not shadow it — the old "you MUST use the absolute path" advice is obsolete; bare `python3` is correct (and the absolute path still works). |
| matplotlib+pandas | `python3 -m pip install "matplotlib>=3.8" "pandas>=2.0"` | only needed for figures/summaries |
| build toolchain | `CC=gcc-15`, `CXX=g++-15`, `MPI_HOME=/usr` | set by the image |

`run_campaign.sh` calls bare `python3`; inside the container that is already the right one. From
a bare host, run inside the container and submit it through the queue (§2.1).

> **Workspace caveat.** In this dev workspace `pypto/runtime` is an **uninitialised submodule**
> (an empty directory), and `collectives/config.py::_first_existing` will happily accept it as
> `SIMPLER_ROOT` — silently breaking the simpler stacks. Run `git submodule update --init
> runtime` in the pypto checkout, or point `SIMPLER_ROOT` at a real tree.

### 2.1 The NPU boxes are queued — read this before touching a card

The NPU hosts are **shared**. On `192.168.150.11` (`hng-atlas01`) and `192.168.150.12`
(`hng-atlas02`) every card-touching command must go through `task-submit`, which takes an
exclusive per-card lock for the life of the job. `.13` (`hng-atlas03`) has no queue; sim work never
needs one. **Never** run on a card you picked by reading `npu-smi` — a card you did not lock is a
card the next user's `--device auto` can land on mid-run. If `task-submit --list` fails, your
container is not on the queue: **stop and report**, do not fall back to a direct run.

- `task-submit` lives on a shared volume at `/var/lib/taskqueue/bin/task-submit`. It is **not** in
  the CANN images and **not** in `pypto-tooling` (that path holds a conceptual doc only).
- Join a shell/container: `export PATH="$PATH:/var/lib/taskqueue/bin"` and
  `export TASKQUEUE_CONF=/var/lib/taskqueue/taskqueue.conf`. For a container use
  `pypto-docker/scripts/attach-taskqueue.sh <container>`; it must bind-mount
  `/var/lib/taskqueue` at the identical path, and `--pid=host` is required for HCCL.
- Run: `task-submit --device auto --device-num 2 --max-time 3600 --timeout 0 --run '<cmd>'`.
  Inside the job the granted card is always logical `0`: reference `$TASK_DEVICE`
  (`$TASK_PHYS_DEVICE` for the physical id) and never set `ASCEND_RT_VISIBLE_DEVICES` yourself.
- `--max-time` is the **kill deadline** (default 300 s, hard cap 3600 s); `--timeout` only controls
  how long the client waits. Campaigns longer than an hour must be **split**, not stretched —
  which is why the sweep recipes in §3 are per-(P, count) units.

Full recipes, the pitfall list (client-shell `LD_PRELOAD`, auto-appended `--device`, `pipefail`,
platform deselection = pytest exit 5, `--timeout` vs `--max-time`, no nesting, never start a
broker) and how to record a task id: **`reports/npu-queue-runbook.md`**. Canonical upstream
sources: `pypto-3.0-notes/tools/npu-task-submit-runbook.md` and `pypto-docker/TASK_QUEUE.md`.

## 3. Quick start

```bash
cd pypto-profiling
export PYTHONPATH=.            # so `collectives` is importable
# inside the cann9 container bare `python3` is already the py3.12 interpreter with pypto

# -1) Join the queue (hosts .11/.12 only; see §2.1) — one-off per shell/container
export PATH="$PATH:/var/lib/taskqueue/bin"
export TASKQUEUE_CONF=/var/lib/taskqueue/taskqueue.conf
task-submit --list             # must print your reserved cards; if it fails, STOP

# 0) Sanity: unit tests (no hardware)
python3 -m pytest tests/ -v

# 1) Smoke test on hardware (validates the whole stack; ~1 min)
python3 -m collectives.run_sweep pair-mesh \
  --case-file collectives/cases/mesh_p2_count65536_fp32_a2a3_d0-1.json \
  --stacks pypto-composite,pypto-host \
  --timed-rounds 2 --warmup-rounds 1 \
  --campaign smoke --out results/campaigns/smoke/run_001/results.json

# 2) Full apples-to-apples (HCCL baseline + pypto stacks), P=2
python3 -m collectives.run_sweep pair-mesh \
  --case-file collectives/cases/mesh_p2_count65536_fp32_a2a3_d0-1.json \
  --stacks hccl,pypto-composite,pypto-host \
  --persistent --timed-rounds 8 --warmup-rounds 2 \
  --campaign demo --out results/campaigns/demo/run_001/results.json

# 3) Strong scaling campaign (P=2,4,8) — merges + summarizes + plots automatically
bash run_campaign.sh --variant mesh --p-values 2,4,8 --count 65536 \
  --stacks hccl,pypto-composite,pypto-host --persistent \
  --warmup-rounds 2 --timed-rounds 8 --campaign strong

# 4) Message-size sweep (feeds the bandwidth model + bw_model_fit figure)
bash run_campaign.sh --variant mesh --p-values 2 \
  --counts 4096,16384,65536,262144,1048576 \
  --stacks hccl,pypto-composite,pypto-host --persistent \
  --warmup-rounds 2 --timed-rounds 8 --campaign sizes
```

**Every recipe above is the *payload*.** On a queue host it must be wrapped in `task-submit`
(§2.1). The case-file device set (`d0-1`) is the *logical* pair, so a `--device-num 2` grant maps
onto it unchanged:

```bash
task-submit --device auto --device-num 2 --max-time 3600 --timeout 0 \
  --run 'cd /path/to/pypto-profiling && export PYTHONPATH=. && python3 -m collectives.run_sweep pair-mesh \
    --case-file collectives/cases/mesh_p2_count65536_fp32_a2a3_d0-1.json \
    --stacks hccl,pypto-composite,pypto-host --persistent \
    --timed-rounds 8 --warmup-rounds 2 \
    --campaign demo --out results/campaigns/demo/run_001/results.json'
```

Keep each submitted unit inside the 3600 s kill deadline — split a campaign by (P, count) or by
stack instead of raising `--max-time`.

## 4. What every experiment option does

Full suite verified 2026-08-28: ring & twophase variants (ring runs pypto-composite/host
+ hccl; twophase is hccl-only), `--core-nums` B\* sweep (pypto-host mesh), `simpler`/
`simpler-own` at count 256, `--profile pmu`, cross-variant (`--mode cross-variant`),
`--mode full-sweep`, and `--platform a2a3sim` (simulator). Known gaps: `fp16` (hccl bench
is fp32-only and the `rank_linear_v1` golden overflows fp16 at count 65536), `pto-isa`
(needs `treduce_test` built via `build_st.py`), P=16 (only 8 devices).

**Added 2026-09-03 (after the 08-28 suite — not covered by that verification line):**
`--reset-persistent-windows` (plan 101 Phase 0), the four composite-gap ablation kernels +
plan 107 clean pass (126/126 correctness), and the `TaskTensor → Tensor` kernel port for
simpler runtime `15f5cbd`. Records: `reports/plan101-phase0-reset-cost-and-l2-crash-2026-09-03.md`,
`reports/plan107-ablation-campaign-record-2026-09-03.md`.

| Flag | Effect |
|------|--------|
| `--stacks` | `hccl,simpler,simpler-own,pypto-composite,pypto-host,pto-isa` (default `hccl,simpler,pypto-composite,pypto-host`) |
| `--persistent` | **Use for all real benchmarks.** `prepare(persistent=True)` keeps CommDomains across dispatches → removes ~95% of per-dispatch overhead (~20–25× faster `execute_s`). |
| `--reset-persistent-windows` | (plan 101 Phase 0, added 2026-09-03) reset CommDomain windows between rounds of a persistent pair-mesh run — isolates the domain-reset cost the persistent default must amortise. |
| `--batch N` | N back-to-back `rt.run()` per timed round, amortizing dispatch (second view). |
| `--count` / `--counts` | payload element count (or sweep). `--counts` → one result per (P,count), merged. |
| `--p-values` | rank sweep. `--platform` `a2a3` (hardware) or `a2a3sim` (sim). |
| `--core-num` / `--core-nums` | AIV launch width (pypto-host mesh only); feeds the B* saturation scorecard. |
| `--profile l2,pmu,dep` | L2 swimlane / PMU / dep-gen DFX capture (see §7 caveats). |
| `--timed-rounds` / `--warmup-rounds` | rounds per stack (defaults from the case file). |

## 5. Metrics — read these, not just `execute_s`

| Field | Meaning |
|-------|---------|
| `device_wall_s_mean` | **THE apples-to-apples metric**: pure on-device collective (slowest-rank `[STRACE] device_wall` span). Use this vs HCCL. |
| `execute_s_mean/median` | `rt.run()` wall (host dispatch + collective). With `--persistent` it's dispatch round-trip + device. **Do not compare non-persistent `execute_s` across stacks.** |
| `execute_spread_ratio` | max/min over timed rounds; `>2` (flagged ⚑) = noisy/shared-box row → trust the median. |
| `setup_s` | one-time compile + init (excluded from timed means). |
| `bw_execute_mb_s` | `n_bytes / execute_s` (host-inflated; use device_wall for real BW). |

## 6. Known environment facts

Each item carries its own verification date. Items still dated 2026-08-28 have **not** been
re-verified and must be re-checked on the box (§10) before you quote them.

1. **(2026-10-14, checked in `cann9-verify`) Interpreter.** In the current image bare `python3`
   resolves to `/usr/local/python3.12.13/bin/python3` — the py3.12 interpreter that has
   `pypto`/`torch`. The py3.10 `/opt/ptoas-bin` venv is *appended* to `PATH`, so it cannot shadow
   it. The absolute path still works; it is simply no longer **required**. (In an *earlier* image
   the venv was prepended, which is where the old "must use the absolute path" warning came
   from.)
2. **(2026-08-28, unverified since) Shared box.** The npu-smi AICore % column is a known visual
   artifact on this box (owner-confirmed 2026-09-02), NOT a real utilization/contention signal.
   Device 3 was intermittently broken (errors `507901`, `-100`, segfaults at domain release).
   The harness's **box-health probe** opens a real CommDomain before each campaign and
   aborts with "passing box condition" when the box is unusable — that is the harness
   working as designed, not a bug. Retry later or use a working device set.
3. **(2026-08-28, unverified since) HCCL device range.** HCCL was reported working on d0-3 and
   failing on d4-7 (`exit=1`, empty log), which is why the 2026-08-28 report split device sets.
   Re-check before relying on it; the queue now grants cards automatically (§2.1).
4. **The harness tracks the current simpler runtime.** Kernels declare `__gm__ Tensor *` args
   (orch `orchestration_api.h`/`ChipTaskArgs`, `worker.make_tensor_arg(t, shapes=, dtype=)`,
   `domain.buffers["scratch"].tensor(...)`). History: the runtime used `TaskTensor` for a stretch;
   commit `7a74a2d` (2026-09-03) ported the kernels back to plain `Tensor` for simpler runtime
   `15f5cbd`. That pin is historical — read the current `runtime` submodule gitlink before
   porting. If you see compile errors naming a
   type or symbol that no longer exists (`BufferDescriptor`/`start_offset`/
   `pto_orchestration_api.h` are the pre-`TaskTensor`-era names; `TaskTensor` was the interim
   one), the simpler runtime moved again — port the runtime's own
   `examples/workers/l3/allreduce` kernel/orch patterns into `collectives/kernels/` and
   `runners/simpler_own.py`.
5. **pypto JIT artifacts** land in `build_output/` (gitignored). Campaign outputs go under
   `results/campaigns/<name>/run_<ts>/` (gitignored). Commit only `reports/` + code.

## 7. Profiling / swimlane (clarity tooling)

- `--profile l2` (no persistent): captures per-rank `dispatch_program.json`, kernels
  (`.pto`/`.cpp`/`.so`/`.bin`) into the artifact bundle `dfx/`; timing is discarded (the
  dispatch runs twice). The merged swimlane JSON is **not retained** by this path.
- `--profile l2 --persistent` **crashes** at domain release — don't combine them.
- `--profile pmu` → `pmu.csv` for the `pmu_utilization` figure.
- For a real per-chip L2 timeline use the pytest route:
  `pytest tests/st/distributed/... --platform a2a3 -d 0 --enable-l2-swimlane` (in pypto),
  then `swimlane_converter`. The harness `t_comm_us` aggregation is not implemented yet (E3).
- `summarize.py --model` fits `T(N)=O+N/B` per (stack,P,variant) and emits a scorecard +
  `reports/model_fit.json` — the standard way to compare asymptotic bandwidth vs HCCL.

## 8. Results → report → figures workflow

```bash
# run_campaign.sh already merges + summarizes + plots. For a manual run dir:
RD=results/campaigns/<name>/run_<ts>
python3 -m collectives.summarize --run-dir "$RD" --emit-report --model
python3 -m collectives.plot_figures --run-dir "$RD" \
  --figures strong_scaling_t_total,strong_scaling_efficiency,message_size_bw_eff,\
paired_stack_ratio,phase_breakdown,setup_breakdown,compile_breakdown,pmu_utilization,\
wall_vs_device,bw_model_fit
# report → $RD/reports/summary.md ; figures → $RD/figures/*.png
# commit important figures under reports/figures-<date>/ and write reports/<report>.md

# Apples-to-apples analysis (persistent vs non-persistent decomposition, ratios vs HCCL):
python3 -m collectives.apples_to_apples --strong results/campaigns/<strong>/run_*/results.json
# → prints the decomposition tables and writes 6 figures to reports/figures-<date>/
```

See `reports/benchmark-report-2026-08-28.md` for a complete worked example (methodology,
per-experiment tables, bandwidth model, findings), and `reports/issue-2521-context.md` for
the upstream RFC behind the residual dispatch cost (L2 orchestration, persistent-vs-per-call
numbers, HCCL launch-width policy).

## 9. Golden rules

1. Always use `--persistent` for publishable `execute_s`; always report `device_wall_s` as
   the cross-stack number.
2. Never mix persistent and non-persistent rows in one comparison.
3. Never mix profiled (`--profile`) and unprofiled rounds (profiling perturbs timing).
4. Never run on a box where the health probe fails — the numbers are meaningless.
5. `execute_s` is NOT comparable to HCCL's collective; `device_wall_s` (or `--batch` means) is.
6. **Every card-touching run goes through `task-submit` on `.11`/`.12`** (§2.1). A card you picked
   by reading `npu-smi` is not a card you hold. Record the task id next to the commit in the
   report — it is the auditable proof the run held a lock.
7. **Don't quote a pre-2026-10-14 campaign as isolated.** Every campaign in `reports/` so far ran
   direct/unqueued (the queue was not provisioned in those containers), so `execute_spread_ratio`
   and the ±20 % spreads are shared-box noise. Re-run under the queue before publishing new
   numbers; see `reports/npu-queue-runbook.md` § "Provenance of the existing campaigns".

## 10. On-box TODO queue

`reports/onbox-todo.md` is the hardware queue — checks this repo owes that cannot be settled in
sim, on the dev box, or by reading source, each with its exit criteria. **Read it before booking an
NPU window**, and delete an entry once it is closed. It currently holds the plan-129 A1
`managed-l2` sweep, the TFILLPAD UB-OOB verification (pto-isa#291) and the ablation-kernel
workaround revert.

`reports/npu-queue-runbook.md` is the companion: how to run any of it through `task-submit`
without losing cards (§2.1).

## 11. Repo hygiene — what a campaign directory may contain

By 2026-10-14 the tree had reached **4.8 GB across 22k files, of which markdown was 0.4 MB** — raw
logs and build trees had been committed under `reports/` (`.log` alone was 4.24 GB over 1570 files,
plus 604 `.so`, 1148 `.o`, 793 `.pto`). Don't repeat that.

**Commit** per campaign: `README.md`, `summary.json`, `campaign_meta.json`, `json/*.json` (the
per-cell records), and any figure you actually reference.

**Never commit:** `attempt_logs/`, `builds/`, `heal/`, `dfx_outputs/`, or a bare `.log` / `.so` /
`.o` / `.pto` / `.bin` / `.lock`. All of these are in `.gitignore` now. They are regenerable, and
the queue task's own log (`task-submit --log <id>`) outlives the repo. If a log excerpt *is* the
evidence, paste the tail into the campaign README instead of committing the file.

The pre-2026-10-14 campaigns still carry ~4.4 GB of such artifacts in history and in the checked-out
tree. Dropping them from the *current* tree (`git rm -r --cached …`) is safe and reversible — the
blobs stay reachable from the old commits, which the reports cite as provenance. **Rewriting
history is not** safe for the same reason: the A1 reports reference commit SHAs (`ea684805`,
`5ca4cef2`, `32c0e37c`, …) as their audit trail, and a force-push would invalidate all of them.

One more gotcha: runs executed inside a container write files as **root**, so new campaign output
lands unwritable by your local user. Chown it back before editing or deleting locally:
`docker exec <container> chown -R $(id -u):$(id -g) /mounted_home/hw-native-sys/pypto-profiling/reports`
