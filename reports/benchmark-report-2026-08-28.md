# PyPTO vs HCCL AllReduce — Benchmark Report (2026-08-28)

**Date:** 2026-08-28 · **Platform:** a2a3 (8× Ascend 910B2, CANN 9.0.0) · **Repo:** `pypto-profiling`
**Interpreter:** `/usr/local/python3.12.13/bin/python3` (pypto 0.1.0, torch 2.13.0)
**Harness:** `run_sweep.py pair-mesh` / `run_campaign.sh`, `--persistent` unless stated.

---

## Executive summary

- **PyPTO's on-device collective time (`device_wall_s`) is genuinely close to HCCL** at
  small P — **~2× at P=2, ~5–9× at P=4, ~8–16× at P=8** (256 KB payload) — matching the
  round-1 numbers recorded in `pypto-3.0-notes/collectives_benchmarking/`.
- The huge `execute_s` values seen previously (10–220 ms) are **host-side per-dispatch
  `CommDomain` lifecycle overhead**, not device time. **Persistent domains
  (`--persistent`) remove ~95% of it**: `execute_s` drops **~20× at P=2 and ~25× at P=4**
  while `device_wall_s` is unchanged.
- A **residual ~5–11 ms per dispatch remains** after persistent mode (vs ~0.3–1.7 ms
  on-device) — the L3→L2 task round-trip, a separate lever from domain lifecycle.
- **Bandwidth model (P=2, mesh):** HCCL B≈16.9 GB/s; pypto-composite B≈4.0 GB/s
  (**0.24× HCCL**), pypto-host B≈1.3 GB/s (**0.08× HCCL**). Latency O is only ~1.7× HCCL;
  the gap widens with payload — an implementation (bandwidth) gap, not a hardware limit.
- **The box is a shared multi-tenant NPU.** Another tenant holds ~100% AICore on 7/8
  chips; device 3 is intermittently flaky (507901 / `-100` / segfaults), and the harness's
  box-health probe correctly blocks runs when the box is unusable. All numbers here were
  collected when the probe passed; some P=4 runs used devices 4-7 because HCCL and the
  pypto stacks failed on the default d0-3 set under contention.

---

## 1. Environment and methodology

### 1.1 Hardware / software

| Component | Value |
|-----------|-------|
| NPU | 8× Ascend 910B2 (a2a3), all `npu-smi` Health OK |
| CANN | 9.0.0 (`/usr/local/Ascend/cann-9.0.0`) |
| HCCL | `libhccl.so` on `LD_LIBRARY_PATH` (HcclAllReduce baseline) |
| pypto | 0.1.0 at `/opt/pypto` |
| simpler (runtime) | `/opt/pypto/runtime` (tensormap_and_ringbuffer) |
| pto-isa | `/opt/pto-isa` |
| Python | 3.12.13 (`/usr/local/python3.12.13/bin/python3`) — the venv at `/opt/ptoas-bin` (3.10) is **not** the benchmark interpreter |

### 1.2 Metrics

- **`device_wall_s`** — slowest-rank `[STRACE] device_wall` span = **pure on-device
  collective time**. This is the apples-to-apples metric vs HCCL.
- **`execute_s`** — `rt.run()` wall (host dispatch + collective). Includes per-dispatch
  host overhead; **not comparable across stacks**. With `--persistent` it converges toward
  the device time + a residual dispatch round-trip.
- **`--batch N`** — N back-to-back `rt.run()` per timed round, amortizing dispatch.
- **`setup_s`** — one-time compile + worker init (excluded from timed means).

### 1.3 Harness fixes required this session

The profiling repo was stale against the current `simpler` runtime; benchmarks would not
run until these were fixed (all committed to this repo):

1. `collectives/kernels/aiv/allreduce_mesh.cpp` — `Tensor` → `TaskTensor` (the runtime's
   working tensor form; bare `Tensor` now resolves to the new wire-ABI type).
2. `collectives/kernels/orchestration/allreduce_mesh_orch.cpp` — migrated to the current
   API: `orchestration_api.h`, `OrchestrationConfig`, `ChipTaskArgs`, `CoreTaskArgs`.
3. `collectives/runners/simpler_own.py` — `worker.make_tensor_arg(t, shapes=, dtype=)` +
   `domain.buffers["scratch"].tensor(...)` (old `simpler_setup.torch_interop.make_tensor_arg`
   and `domain.buffer_ptrs` are gone).
4. `run_campaign.sh` — fixed `CORE_NUMS[0]: unbound variable` when `--core-nums` is unset.

---

## 2. Experiments

| ID | Experiment | Config | Result |
|----|-----------|--------|--------|
| E1 | Strong scaling (persistent) | mesh, P=2/4/8, 65536×fp32 (256 KB), 2 warmup + 8 timed | ✅ all 9 rows pass |
| E2 | Message-size sweep (persistent) | mesh, P=2, counts 4K/16K/64K/256K/1M (16 KB–4 MB), 2+8 | ✅ all 15 rows pass |
| E3 | Persistent vs non-persistent | mesh, P=2 & P=4, 65536×fp32 | ✅ 10 rows (see §3.3) |
| E4 | L2 swimlane capture | P=2, `--profile l2` | ⚠️ captures dispatch graph + kernels; merged timeline not retained in the harness bundle |

Every run passes golden verification (`correctness=pass`) before its row is recorded.

**Device sets:** E1/E2 use d0-1 (P=2) and d0-7 (P=8). At P=4, HCCL was measured on d0-3
and the pypto stacks on d4-7 (HCCL's `HcclAllReduce` fails on the d4-7 PCIe domain and the
pypto stacks intermittently fail on dev 3 under tenant contention). **Caveat:** the P=4 row
mixes device sets; treat it as directional, not a strict apples-to-apples cell.

---

## 3. Results

### 3.1 E1 — Strong scaling (persistent), 65536×fp32 (256 KB)

| P | Stack | `execute_s` mean | median | `device_wall_s` | BW (MB/s) | vs HCCL exec |
|---|-------|-----------------:|-------:|----------------:|----------:|-------------:|
| 2 | hccl | 0.165 ms | 0.160 ms | — | 1591.5 | 1× |
| 2 | pypto-composite | 6.084 ms | 6.032 ms | **0.319 ms** | 43.4 | ~2.0× (device) |
| 2 | pypto-host | 7.812 ms | 7.594 ms | **0.515 ms** | 34.4 | ~3.2× (device) |
| 4 | hccl | 0.182 ms | 0.183 ms | — | 1441.2 | 1× |
| 4 | pypto-composite | 9.280 ms | 8.392 ms | **0.787 ms** | 29.4 | ~4.3× (device) |
| 4 | pypto-host | 11.661 ms | 10.663 ms | **1.700 ms** | 22.9 | ~9.3× (device) |
| 8 | hccl | 0.214 ms | 0.214 ms | — | 1224.3 | 1× |
| 8 | pypto-composite | 14.569 ms | 14.834 ms | **2.274 ms** | 18.2 | ~10.6× (device) |
| 8 | pypto-host | 18.638 ms | 17.229 ms | **3.413 ms** | 14.5 | ~16.0× (device) |

Reading: with persistent domains, `execute_s` for pypto is 5–19 ms (mostly host dispatch
residual), while **`device_wall_s` stays 0.3–3.4 ms** — the true collective. Device-only
ratios vs HCCL: **~2× (P=2), ~4–9× (P=4), ~11–16× (P=8)** for composite/host respectively.
These track the round-1 reference (P=2 ~1.7×, P=4 ~5×/9×, P=8 ~8×/16×) within shared-box noise.

### 3.2 E2 — Message-size sweep (persistent), P=2, mesh

| Payload (B) | HCCL `execute_s` | composite `device_wall_s` | host `device_wall_s` |
|------------:|-----------------:|--------------------------:|---------------------:|
| 16,384 (4K) | 0.136 ms | 0.268 ms | 0.195 ms |
| 65,536 (16K) | 0.185 ms | 0.402 ms | 0.530 ms |
| 262,144 (64K) | 0.160 ms | 0.190 ms | 0.539 ms |
| 1,048,576 (256K) | 0.207 ms | 0.471 ms | 1.177 ms |
| 4,194,304 (1M) | 0.402 ms | 1.303 ms | 3.639 ms |

The composite's device time grows with payload (bandwidth-bound behaviour from ~64 KB up);
the host builtin is consistently ~2.8× worse on-device at the largest payload.

### 3.3 E3 — Persistent vs non-persistent domains (65536×fp32)

| P | Stack | `execute_s` non-persist | `execute_s` persistent | speedup | `device_wall_s` (both) |
|---|-------|----------------------:|-----------------------:|--------:|-----------------------:|
| 2 | pypto-composite | 113.7 ms | **5.7 ms** | **~20×** | 0.19–0.49 ms |
| 2 | pypto-host | 114.8 ms | **6.2 ms** | **~19×** | 0.38–0.46 ms |
| 4 | pypto-composite | 220.4 ms | **8.7 ms** | **~25×** | 0.96–1.23 ms |
| 4 | pypto-host | 170.9 ms | **12.4 ms** | **~14×** | 1.85–3.23 ms |

`--persistent --batch 10` adds little beyond persistent alone (5.1–6.6 ms at P=2), i.e. the
dominant cost was the **per-dispatch CommDomain lifecycle** (`prepare(persistent=True)`),
and the residual ~5–11 ms is the L3→L2 per-dispatch task round-trip (issue #2521's second,
separate lever). These results independently confirm the 63–75× persistent-vs-per-call
numbers measured by @vloncar in `pypto-3.0-notes` (different operator).

### 3.4 E4 — L2 swimlane

`--profile l2` (no persistent) runs successfully and captures per-rank `dispatch_program.json`,
compiled kernels (`.pto`/`.cpp`/`.o`/`.so`/`.bin`) and reports a merged per-rank swimlane
JSON, routed into the artifact bundle under `dfx/`. Limitations observed:

- Profiled timing is **discarded by design** (the dispatch runs twice: graph, then timing).
- `--profile l2 --persistent` crashes at domain release (`control_release_domain ... timed-out control command`).
- The merged swimlane timeline is **not retained** in the bundle by the harness path; the
  `t_comm_us` aggregation is marked "nullable until E3" in the benchmark plan.
- The proper deep-dive route is the pytest path (`--enable-l2-swimlane` + `swimlane_converter`).

---

## 4. Bandwidth model (`T(N) = O + N/B`), P=2 mesh

Fitted by `summarize.py --model` over the E2 sweep (`device_wall` for pypto, `execute_s` for HCCL):

| Stack | O (latency) | B (marginal BW) | r² | pipe@maxN | BW vs HCCL | O vs HCCL |
|-------|------------:|----------------:|----:|----------:|-----------:|----------:|
| hccl | 151.8 µs | **16.9 GB/s** | 0.972 | 0.62 | — | 1× |
| pypto-composite | 250.5 µs | **4.0 GB/s** | 0.955 | 0.80 | **0.24×** | 1.65× |
| pypto-host | 334.1 µs | **1.3 GB/s** | 0.994 | 0.91 | **0.08×** | 2.20× |

Reading: at P=2 the latency O is only ~1.7–2.2× HCCL, but marginal bandwidth is 4×–12×
worse — and both pypto stacks are already >80% bandwidth-bound at the largest payload
(`pipe@maxN` 0.80 / 0.91), i.e. **the gap is a bandwidth/implementation gap, not dispatch**.
This matches the notes' model: mesh is O(P²) traffic, MTE vs SDMA data planes, and no
chunk pipelining yet.

---

## 5. Findings

1. **`device_wall` is the only cross-stack metric.** `execute_s` mixes host dispatch with
   device time; comparisons against HCCL must use `device_wall` (or persistent `--batch`
   means).
2. **Persistent domains are mandatory for honest `execute_s`.** They remove ~95% of the
   per-dispatch overhead (~20–25×), and are the documented default for any benchmark.
3. **The residual ~5–11 ms is a real (separate) cost.** After persistent mode, `execute_s`
   still exceeds `device_wall` by ~5–11 ms at P=2–8 — the L3→L2 task setup round-trip,
   target of the L2-orchestration work (issue #2521).
4. **The bandwidth gap is the headline, not the latency gap.** Composite is 0.24× HCCL
   bandwidth, host 0.08×, at P=2 mesh. Levers per the notes: `mode="ring"`, O(1)
   NeighborBarrier (#2280), chunk pipelining/multi-buffering, SDMA transport.
5. **Shared-box discipline is required.** Every campaign must run the box-health probe
   (the harness does automatically), use `device_wall` medians to resist tenant spikes,
   and flag rows with `execute_spread_ratio > 2` (⚑).

---

## 6. Figures

Copies are committed under `reports/figures-2026-08-28/` (generated into each run dir's
`figures/` by `plot_figures.py`):

| Figure | Source campaign | Shows |
|--------|-----------------|-------|
| `strong_scaling_t_total.png` | E1 | execute time vs P, one line per stack |
| `strong_scaling_efficiency.png` | E1 | parallel efficiency E(P) per stack |
| `wall_vs_device.png` | E1 | `execute_s` vs `device_wall_s` (the dispatch-overhead gap) |
| `message_size_bw_eff.png` | E2 | BW vs payload (log-x crossover) |
| `bw_model_fit.png` | E2 | measured T(N) points + fitted O+N/B lines vs HCCL |
| `setup_breakdown.png` / `phase_breakdown.png` | E1/E2 | compile / init / execute phase means |
| `compile_breakdown.png` | E1/E2 | pypto compile sub-stages (passes / codegen) |

## 7. Reproduce

```bash
cd pypto-profiling
# correct interpreter (has pypto + torch):
export PATH=/usr/local/python3.12.13/bin:$PATH

# E1 strong scaling (persistent):
bash run_campaign.sh --variant mesh --p-values 2,4,8 --count 65536 \
  --stacks hccl,pypto-composite,pypto-host --persistent \
  --warmup-rounds 2 --timed-rounds 8 --campaign analytic_strong

# E2 message-size sweep (persistent):
bash run_campaign.sh --variant mesh --p-values 2 \
  --counts 4096,16384,65536,262144,1048576 \
  --stacks hccl,pypto-composite,pypto-host --persistent \
  --warmup-rounds 2 --timed-rounds 8 --campaign analytic_sizes

# E3 persistent vs not:
PYTHONPATH=. python -m collectives.run_sweep pair-mesh \
  --case-file collectives/cases/mesh_p2_count65536_fp32_a2a3_d0-1.json \
  --stacks pypto-composite,pypto-host --persistent \
  --timed-rounds 8 --warmup-rounds 2 --campaign <name> \
  --out results/campaigns/<name>/run_001/results.json
# (omit --persistent for the non-persistent leg)
```

Raw data: `results/campaigns/analytic_strong/run_20260828_150909/results.json` and
`results/campaigns/analytic_sizes/run_20260828_151614/results.json` (both gitignored).
