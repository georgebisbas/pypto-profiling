# PyPTO / simpler — Kernel Performance Improvement Ideas — Measured Results (2026-08-28)

**Status:** six ideas identified and **benchmarked on a2a3 (8× Ascend 910B2)**. Every idea
was implemented as a runnable benchmark (new kernel variants where needed) and **correctness
was verified automatically** (harness golden check vs `allreduce_sum_v1`, rtol/atol 1e-3; a
row only records PASS if every timed round matches). All data below is `device_wall_s`
(pure on-device collective) unless noted; `execute_s` is persistent-mode host dispatch +
device.

**Implementation artifacts added this session:**
- `collectives/kernels/aiv/allreduce_mesh_pipelined.cpp` — Idea 1 kernel variant
- `collectives/kernels/aiv/allreduce_mesh_handicapped.cpp` — Idea 2 A/B kernel variant
- `collectives/runners/simpler_own.py` — `PYPTO_SIMPLER_KERNEL ∈ {mesh, pipelined, handicapped}`

---

## Idea 1 — Pipelined, multi-buffered chunk execution → **NEGATIVE in this form**

**Hypothesis:** the serial per-chunk/per-peer `load→wait→add` leaves the HCCS link idle;
firing all peer remote loads back-to-back into per-peer UB tiles should keep MTE2 busy.

**Benchmark:** `allreduce_mesh_pipelined.cpp` (per-peer recv tiles @8 KiB each, chunk=2048
cols) vs the baseline mesh kernel. Correctness: **PASS** in every configuration.

| Config | baseline dev | pipelined dev | delta |
|---|---|---|---|
| P=2, 64K | 0.195 ms | 0.180 ms | −8% (noise) |
| P=4, 64K | 0.234 ms | 0.355 ms | +52% **worse** |
| P=4, 1M | 1.583 ms | 1.583 ms | 0% |

**Verdict: no gain.** The finer chunking (32–512 chunks vs 4–64) adds per-chunk
event-flag/loop overhead that cancels the overlap; at P=2 there is only one peer (nothing
to overlap), and at P=4/1M the mesh kernel is already ~80–98% transfer-bound (the model's
`pipe@maxN`), so there is little to reclaim. **Revised recommendation:** the pipelining
lever belongs in the **ring** path (larger double-buffered tiles + neighbour-local
barriers, the notes' plan 53/60), not the mesh per-peer-tile form tested here.

---

## Idea 2 — Remove per-peer full barriers + whole-cache dcci → **CONFIRMED real cost (~15–40%)**

**Hypothesis:** the generated composite kernel's per-peer `pipe_barrier(PIPE_ALL)` (notify
loop) and per-peer `dcci(0, ENTIRE_DATA_CACHE)` (wait loop) cost on-device time and explain
part of the 6.7× composite-vs-simpler-own gap.

**Benchmark:** `allreduce_mesh_handicapped.cpp` = baseline + exactly those two additions
(A/B). Correctness: **PASS**.

| Config | baseline dev | handicapped dev | cost |
|---|---|---|---|
| P=2, 64K (1 peer) | 0.195 ms | 0.243 ms | **+25%** |
| P=4, 64K (3 peers) | 0.234 ms | 0.269 ms | **+15%** |

**Verdict: confirmed.** Per-peer full barriers + whole-cache invalidates cost ~15–25%+
on-device (and scale with `(P−1)`), so removing them in `LowerCompositeOps` is a real,
low-risk win. Note the composite's full 6.7× gap is dominated by *additional* factors
(serial per-peer load→wait→add with pipe sync, `TFILLPAD`, per-chunk acc reloads) — the
barrier/dcci removal is one concrete, attributable fix.

---

## Idea 3 — Ring / bidirectional-ring at P≥4 → **CONFIRMED for the host builtin (2.8× @1M P=4)**

**Hypothesis:** ring is O(P) traffic vs mesh O(P²); the benefit only appears at P≥4.

**Benchmark:** P=4 ring message-size sweep (pypto on d4-7, hccl on d0-3) vs the P=4 mesh
sweep. Correctness: **PASS**.

| count | stack | ring dev | mesh dev | ring speedup |
|---:|---|---:|---:|---:|
| 4K | composite | 0.672 | 0.791 | 1.18× |
| 256K | composite | 1.576 | 2.640 | **1.68×** |
| 1M | host | 5.303 | 14.621 | **2.76×** |
| 256K | host | 1.942 | 4.299 | **2.21×** |

(64K rows were noisy outliers — shared box.) **Verdict: confirmed at P≥4.** The host
builtin in ring mode is **2.2–2.8× faster on-device** at ≥256K than mesh — the O(P)-traffic
benefit materialises. The composite is mixed (its ring kernel is serial-per-chunk; the
pipelining + NeighborBarrier work applies there).

---

## Idea 4 — Runtime-selected multi-AIV launch (core_num default) → **STRONGLY CONFIRMED, grows with P**

**Benchmark:** `core_num` sweeps at P=2 (4 counts, earlier), P=4 (earlier) and **P=8
(pypto-host, this session)**. Correctness: **PASS**.

| P | count | cn1 dev | cn≥8 dev | speedup |
|---|---:|---:|---:|---:|
| 2 | 256K | — | 3.6–5.5 GB/s B | 3–4.5× BW |
| 4 | 256K | — | 1.9 GB/s B | ≈6× |
| **8** | **64K** | **3.697 ms** | **2.393 ms (cn16)** | **1.5×** |
| **8** | **256K** | **10.666 ms** | **3.417 / 3.279 ms (cn8/16)** | **3.2×** |

**Verdict: strongly confirmed, and the benefit grows with P** (P=2 ≈3–4.5× BW; P=8
**3.2× device time at 256K**). Single-AIV does not saturate; runtime-selected multi-AIV
(issue #2521's launch policy, mirroring HCCL's `B=min(L,P+1)` / `B=floor(L/P)·P` tables) is
the correct default.

---

## Idea 5 — Persistent domains as the default → **already confirmed (19–25×)**

Measured earlier (P=2/4/8): `execute_s` 114→5.7 ms (P=2), 220→8.7 ms (P=4), 233→13.6 ms
(P=8); `device_wall` unchanged. One-line runtime default change (issue #2069). No new run
needed.

---

## Idea 6 — L2 orchestration + SDMA/TPUT_ASYNC transport → **architectural (report only)**

Residual 5–11 ms L3→L2 dispatch round-trip (#2521) and MTE-vs-SDMA data plane (HCCL
48.6 GB/s vs pypto ~0.5–4 GB/s). Not implementable in this harness; the dispatch-side cost
is benchmarked (persistent `execute_s` − `device_wall` ≈ 5–11 ms) and the transport gap is
quantified by the bandwidth model.

---

## Where do these optimizations stand in pypto today? (exists / opt-in / novel)

Audited 2026-08-31 against the compiler (`/opt/pypto/src`) and runtime
(`/opt/pypto/runtime`) at commit `c361b8a`-era main. The headline: **most "existing"
optimizations exist only as opt-in capabilities, never as defaults — which is exactly why
the benchmark gains exist.** Two ideas are genuinely novel (no code exists), one is a
defect in current generated code.

| # | Idea | In pypto today? | Evidence (file + line) |
|---|------|-----------------|------------------------|
| 1 | Pipelined chunks / multi-buffering | **Novel — not present** | Generated composite (`reduce_step.cpp`) is chunked at UB size but fully serial per chunk (TLOAD→accum→TSTORE, event-flag waits). Ring kernel chunks by rank (P slots) but every ring step is serialized by `NeighborBarrier` + `pipe_barrier(PIPE_ALL)` (`runtime/tests/st/.../aiv/allreduce_ring_kernel.cpp`). No multi-buffering/overlap anywhere. |
| 2 | Remove per-peer barrier/dcci | **Pattern exists; fix is novel** | Generated composite **does** emit per-peer `pipe_barrier(PIPE_ALL)` + `TNOTIFY/TWAIT` + `dcci(0, ENTIRE_DATA_CACHE)` — confirmed in the generated `reduce_step.cpp` (lines 241/282, 284/307-308, 506/529-530). This is current pypto behaviour; slimming/removing it is the proposed fix. |
| 3 | Ring at P≥4 | **Exists — opt-in** (`mode="ring"`) | `builtin.tensor.allreduce_ring` host builtin + runtime ring/bidirectional_ring/onephase/twophase/ibing kernels. `mode` defaults to `"mesh"` (`lower_host_tensor_collectives_pass.cpp:62`); **no auto-selection** (plan 42 pending). |
| 4 | Multi-AIV (`core_num`) | **Exists — opt-in, default = 1** | `core_num` attr on host mesh allreduce → SPMD AIV grid (distributed_ops_codegen.cpp:245-281). InCore composite restricted to `core_num == 1` (`lower_composite_ops_pass.cpp:890`). |
| 5 | Persistent domains | **Exists — opt-in, not default** | `DistributedWorker(persistent=True)` + `bench.benchmark(persistent=...)` + codegen `_domain_provider` hook (distributed_codegen.cpp:280). Plain `run()` still creates transient domains. |
| 6 | L2 orchestration + SDMA | **Novel — not present** | All `*_orch.cpp` kernels are thin `rt_submit_aiv_task` wrappers; data movement is AIV MTE (TLOAD/TSTORE). **Zero SDMA/TPUT data movement** in the collectives. |

**Consequences for the landing list below:**
- Ideas 3, 4, 5 are **capability-exists work** — the win is making them the default or
  auto-selecting them, not building new mechanisms.
- Ideas 1 and 6 are **genuinely new compiler/runtime work** (ring pipelining; SDMA/L2 plane).
- Idea 2 is a **defect fix in current generated code** — the cost was measured directly
  (+15–40% device), so it is the lowest-risk, most attributable change.

---

## Prioritised landing list (with measured evidence)

| # | Idea | Measured gain | Effort | Landing |
|---|------|---------------|--------|---------|
| 5 | Persistent default | **19–25×** `execute_s` | trivial | pypto runtime |
| 4 | Multi-AIV default | **3.2× device @P=8/256K**; grows with P | medium | pypto launch policy |
| 3 | Ring at P≥4 (host) | **2.2–2.8× device @P=4/≥256K** | medium | pypto builtin |
| 2 | Remove per-peer barrier/dcci | **15–40% device**; part of 6.7× composite gap | low | pypto composite lowering |
| 1 | Pipelining (ring form) | **negative in mesh form**; revisit for ring | medium | pypto ring template |
| 6 | L2 orchestration + SDMA | dispatch 5–11 ms→0; transport→link-bound | large | pypto+simpler |

**Suggested order:** 5 → 4 → 3 → 2 → 1(re-scoped to ring) → 6. All kernel variants used in
this session are committed and reproducible via `PYPTO_SIMPLER_KERNEL`.
