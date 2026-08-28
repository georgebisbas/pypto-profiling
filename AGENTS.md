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
- `run_campaign.sh` — campaign driver: runs a sweep, **merges** per-(P,count) results into
  `results.json`, then runs summarize + plot.
- `tests/` — pure unit tests (no hardware): `python3 -m pytest tests/ -v` (24 tests).
- `results/` — campaign outputs (gitignored). `reports/` — committed reports + figures.

## 2. Prerequisites (dev container)

| Thing | Where | Notes |
|-------|-------|-------|
| NPU | `/dev/davinci*` (8× Ascend 910B2) | **shared box** — another tenant runs at ~100% AICore; verify with `npu-smi info` |
| CANN | `/usr/local/Ascend/cann-9.0.0` | env vars are pre-set in this container |
| HCCL | `libhccl.so` on `LD_LIBRARY_PATH` | needed for the `hccl` stack |
| pypto | `/opt/pypto` (installed, v0.1.0) | auto-detected |
| simpler | `/opt/pypto/runtime` | auto-detected (this is the `simpler` runtime) |
| pto-isa | `/opt/pto-isa` | auto-detected |
| **Python** | **`/usr/local/python3.12.13/bin/python3`** | **MUST use this** — has pypto + torch 2.13. The venv `/opt/ptoas-bin` (3.10) is only for unit tests and will fail with `ModuleNotFoundError: pypto`. |
| matplotlib+pandas | install once: `/usr/local/python3.12.13/bin/python3 -m pip install "matplotlib>=3.8" "pandas>=2.0"` | only needed for figures/summaries |

`run_campaign.sh` calls bare `python3` — always prefix with the right PATH:
```bash
PATH=/usr/local/python3.12.13/bin:$PATH bash run_campaign.sh ...
```

## 3. Quick start

```bash
cd pypto-profiling
export PATH=/usr/local/python3.12.13/bin:$PATH
export PYTHONPATH=.            # so `collectives` is importable

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

## 4. What every experiment option does

| Flag | Effect |
|------|--------|
| `--stacks` | `hccl,simpler,simpler-own,pypto-composite,pypto-host,pto-isa` (default `hccl,simpler,pypto-composite,pypto-host`) |
| `--persistent` | **Use for all real benchmarks.** `prepare(persistent=True)` keeps CommDomains across dispatches → removes ~95% of per-dispatch overhead (~20–25× faster `execute_s`). |
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

## 6. Known environment facts (learned 2026-08-28)

1. **Correct interpreter is `/usr/local/python3.12.13/bin/python3`.** The `ptoas-bin` venv
   cannot import `pypto`/`torch`. `run_campaign.sh` needs `PATH` prefixed (see §2).
2. **The NPU is a SHARED, contended box.** Other tenants run ~100% AICore on most chips;
   device 3 is intermittently broken (errors `507901`, `-100`, segfaults at domain release).
   The harness's **box-health probe** opens a real CommDomain before each campaign and
   aborts with "passing box condition" when the box is unusable — that is the harness
   working as designed, not a bug. Retry later or use a working device set.
3. **HCCL works on d0-3 but fails on d4-7** (`exit=1`, empty log). For P=4 use d0-3 for
   hccl and d4-7 for the pypto stacks (as the 2026-08-28 report did) — or wait for a quiet box.
4. **The harness was fixed to track the current simpler runtime** (kernel `TaskTensor`,
   orch `orchestration_api.h`/`ChipTaskArgs`, `worker.make_tensor_arg(t, shapes=, dtype=)`,
   `domain.buffers["scratch"].tensor(...)`, `run_campaign.sh` CORE_NUMS bug). If you see
   compile errors mentioning `BufferDescriptor`/`start_offset`/`pto_orchestration_api.h`,
   the simpler runtime moved again — port the runtime's own `examples/workers/l3/allreduce`
   kernel/orch patterns into `collectives/kernels/` and `runners/simpler_own.py`.
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
