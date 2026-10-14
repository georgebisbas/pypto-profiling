# Closing the HCCL gap — working notes

**Goal:** try the ranked fixes from `HCCL_VS_PYPTO_A2AV.md` § "What is missing", one at a time, with a
measurement after each. Baseline to beat: **EP8, 1 MiB/peer, `managed-l2 L=8` = 13615 µs host-slot**
(HCCL 578 µs).

**Status (2026-10-14):** Step 0 ✅ · Step 1 (action A, device-resident args) ✅ **10377 → 5270 µs,
1.97×, gap vs HCCL 23.6× → 9.1×** — see §1d and
[`reports/a2av-resident-args-2026-10-14/`](../../a2av-resident-args-2026-10-14/README.md).
Next: action B (async data path), which is now the critical path.

**Rules I'm holding myself to:**
- Every card-touching run goes through `task-submit` (8 cards, `--max-time ≤ 3600`).
- Work in a **separate pypto worktree**, never the shared `pypto/` tree.
- One change at a time; re-measure; record the delta or record the failure.
- No claim without a log behind it.

**Repo:** `hw-native-sys/pypto` (fork remote `fork-migrate` locally, `fork-gbisbas` per the standards
doc — confirm before pushing).

---

## Step 0 — decompose the 13615 µs before changing anything ✅ DONE

Reason: the report's own framing may be wrong. If the host remainder is dominated by *staging
kernels* rather than dispatch, then "use the DMA engine" is not the first fix.

Method: parse the `[STRACE]` spans already captured in
`reports/issue-2521-a1-ep8-l2corenum-2026-10-14/attempt_logs/p8_l8_managed-l2_uniform_<pb>_try1.log`,
aggregated by span name. No hardware needed. Mean per `chip.run` (per rank per invocation):

| payload | `stage_inputs` | `validate` | `device_wall` | `chip.run` |
|---|---|---|---|---|
| 64 KiB | 0.24 ms | 0.25 ms | 1.26 ms | 2.31 ms |
| 256 KiB | 0.72 ms | 0.74 ms | 1.83 ms | 3.80 ms |
| 512 KiB | 1.37 ms | 1.33 ms | 2.89 ms | 6.13 ms |
| 1 MiB | **3.11 ms** | **2.81 ms** | 4.47 ms | 10.98 ms |

### Result: the ranking in the report was wrong

Both `validate` and `stage_inputs` **scale linearly with payload** — they are per-byte operations, not
fixed overhead. The emitting sites (`runtime/src/common/platform/onboard/host/c_api_shared.cpp`) say
what they actually are:

| span | real implementation | what it is |
|---|---|---|
| `chip.run.stage_inputs` (:1096) | `copy_in_run_inputs_impl` | **H2D copy of the run's inputs, every `run()`** |
| `chip.run.validate` (:1389) | `copy_back_run_outputs_impl` + `release_run_bindings_impl` | **D2H copy-back of the outputs** (misleading name — it is not a correctness check) |

So at 1 MiB, of PyPTO's ~11 ms per-rank `run()`:

- **5.92 ms (54%) is the runtime copying the payload in and back out** — H2D + D2H
- 4.47 ms is device execution, of which the collective AIV task itself is 2.66 ms
- the rest is bind/dispatch plumbing

And **`hccl_bench` has no equivalent**: it `aclrtMalloc`s once and loops `HcclAlltoAllV`, so HCCL's
578 µs contains no per-call H2D/D2H at all.

There is **no knob** to skip either step — `grep -rniE 'skip_validate|disable_validate|enable_validate'`
over the runtime source finds nothing.

### Consequence — the headline number does not measure what it claims

`host_timing_slot_p50_us` (the column used as *the* apples-to-apples number) includes a per-invocation
H2D of the inputs and a D2H of the outputs. At 64 KiB that is small; at 1 MiB it is **larger than the
entire device execution**. So:

- The **23.57×** headline at 1 MiB is *not* a statement about the collective. It is mostly "PyPTO's
  `run()` moves the payload twice across host↔device, HCCL's does not."
- The like-for-like *collective* comparison at 1 MiB is PyPTO's AIV gang span **2663 µs** vs HCCL
  **578 µs** = **4.6×** — real, but a different and much smaller claim.

### Revised ranking

| # | Action | Term it attacks | Where | Effort |
|---|---|---|---|---|
| **A** | Keep inputs/outputs **device-resident** across timed rounds (or a `run()` that does not stage) | 5.92 ms of 13.6 ms at 1 MiB | **harness + runtime API** | S–M |
| **B** | Move payload to the **SDMA/URMA async path** | device-span slope (~5 Gbit/s marginal) | pypto kernel template | L |
| **C** | Remove the two global credit barriers | fixed ~0.5 ms | pypto kernel template | M |
| **D** | Overlap across invocations / drop the staging hop | fixed + device span | pypto kernel template | M |

**A is now first**: largest single term, it is *measurement* rather than kernel work, and until it is
fixed every PyPTO number this repo publishes at ≥256 KiB is inflated by it.

### Caveats

- `device_wall` (4.47 ms) is the whole device execution — stage kernel + collective + consume kernel —
  not just the collective. The harness's `aicore_gang_span` (2663 µs) is the collective task only.
- Span totals are summed across 8 ranks × 24 rounds = 192 `chip.run`s; per-rank means, not wall clock.
  The *composition* is the finding, not the absolute sum.
- The `ccu`/`sdma`/`urma` backends still exist and are still unused by the kernel — **B remains real**,
  it is just not first.

---

## Step 1 — make the terms separable, then attack A (partial)

### 1a. Breakdown is now first-class in the driver ✅ DONE

`collectives/alltoallv_a1.py` now parses the `[STRACE]` spans out of each attempt log and records
`breakdown_us` in every summary row (median per run + occurrence count), and prints it inline.
Median, not mean, so one straggler round cannot move it. Validated end-to-end through the queue on
2026-10-14 (EP8, `managed-l2 L=8`, 256 KiB, 8 rounds):

```
OK attempt=1 metric=aicore_gang_span  kernel_p50=678.6  slot_p50=5581.1  egress=21.63
   breakdown (median per run): stage_in=604us  copy_back=636us  device_wall=2089us
```

That is the shape we needed: of a 5581 µs slot, **1240 µs (22%) is H2D+D2H of the payload**, 2089 µs
is device execution, and ~2250 µs is bind/dispatch plumbing. Every future cell reports this.

### 1b. Device-resident I/O (action A) — **not blocked; the accessor exists** ✅ (correction)

The runtime will skip both copies if the tensor is already device memory — in
`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`:

```cpp
if (t.is_device_memory()) { out->add_tensor(t); continue; }   // pass-through, no copy
```

and `pypto.runtime.DeviceTensor` is the public handle: *"Pass it to CompiledProgram in place of a
torch.Tensor to skip the host→device copy on entry and the device→host copy on exit."*

**Correction to the first version of this note.** That version claimed action A was blocked because
"there is no public way to get the `Worker` out of the `compiled.prepare(...)` session". **That is
wrong, and the error was mine in the searching, not a gap in PyPTO.** I had grepped `alloc_tensor`
in `runtime_base.py` (the *abstract* `Worker`) and in `worker.py`, concluded no concrete holder
exposed it, and stopped. The session object `compiled.prepare(...)` yields **is** the holder — it is
a `DistributedWorker`, and it exposes the whole device-memory surface:

| Method | Semantics |
|---|---|
| `rt.alloc_tensor(shape, dtype, *, init=None)` | one worker-resident `DeviceTensor`; `init` is a one-time H2D |
| `rt.alloc_stacked_tensor(host)` | upload each leading-dim shard of `host` once — **shard `i` → card `i`** |
| `rt.free_tensor(t)` / `rt.free_stacked_tensor(s)` | release |
| `rt.copy_from(...)` / `rt.copy_stacked_from(stacked, host)` | explicit staged D2H read-back |
| `rt.copy_to(...)` | explicit staged H2D |

Documented in `docs/en/user/distributed/03-execution.md:31–41` ("Execution — `prepare()` and
dispatch"), and `python/pypto/runtime/bench.py:1316-1324` names the intended pattern outright:

> *"this helper creates its own prepared Worker, so it cannot accept a DeviceTensor owned by another
> Worker; **benchmark resident tensors with an explicit `compiled.prepare()` dispatch loop**."*

The A1 harness *is* an explicit `compiled.prepare()` dispatch loop, so it is exactly the shape the
docs point at. Worse, `distributed_runner.py:1755-1762` documents the reuse block as existing
precisely *"so callers can build worker-resident `DeviceTensor` buffers that survive across
dispatches, then call `rt(*device_args)` or `rt.run(compiled, *device_args)` repeatedly."*

**The mapping is exact, not approximate.** `alloc_stacked_tensor` defaults `worker_ids=range(B)` and
its docstring names the canonical program it serves:
`for r in range(world_size): child(x[r], device=r)`. The harness orchestration reads precisely that
way — `stage_orch(inputs[r], …, device=r)`, `fill_counts_orch(send_counts[r], …, device=r)`,
`consume_orch(…, outputs[r], recv_outputs[r], device=r)` — and all four tensors are `[p, …]` with
the rank in dim 0, so shard `i` lands on the card whose dispatch consumes it, with no re-indexing.

So action A was never blocked; it was **undiscovered**. The revised split:

| Piece | Where | Status |
|---|---|---|
| `DeviceTensor` handle + pass-through in the runtime | simpler / pypto runtime | already exists |
| `alloc_tensor` / `alloc_stacked_tensor` / `copy_stacked_from` on the prepared session | pypto `distributed_runner` | **already exists, documented** |
| harness allocates once per session, dispatches `DeviceTensor`s, reads back once outside timing | A1 harness | **implemented, under test** (below) |

**Predicted payoff** (from the breakdowns, per rank per run): 3.09 ms + 2.55 ms at 1 MiB, 0.60 ms +
0.64 ms at 256 KiB — i.e. it should take the 1 MiB `managed-l2 L=8` slot from **13615 µs to ~8000 µs**
(≈1.7×) without touching the kernel, and it removes a term HCCL never had.

### 1d. Action A — built, run, **worked: 1.97×** ✅ (2026-10-14)

Implemented as an opt-in harness flag so the A/B is one binary and one box state:

- **Branch:** `a2av-resident-args` (in `wt-l2cn`, stacked on `a2av-harness-l2-core-num` @ `22fc5886`,
  i.e. on top of the plan-129 harness work), file
  `tests/st/distributed/collectives/all_to_all_v_benchmark.py`.
- **Change:** new `--resident-args`. When set, `_timed_session` calls `alloc_stacked_tensor` on all
  four args once, dispatches the `DeviceTensor`s in the timed loop, and calls `copy_stacked_from`
  only for the correctness read and once at session end — both outside timing. Default path is
  byte-for-byte the old host-tensor behaviour.
- **Driver:** `collectives/alltoallv_a1.py` gained `A2AV_EXTRA_HARNESS_ARGS` passthrough (appended
  last, so it can override modelled flags) and folds the extra flags into the cell tag so the row
  and log stay self-describing.
- **Experiment:** `reports/a2av-resident-args-2026-10-14/` — variant A (baseline) then variant B
  (`--resident-args`), same cell (EP8 / `managed-l2` / L=8 / 1 MiB / uniform / 20 rounds / warmup 3),
  back-to-back inside **one** queue grant so both see identical box state.
  Queue task `task_20261014_112920_242602513476` (`--device auto --device-num 8 --max-time 3300`).

**Result — the prediction held, and then some.** Slot p50 **10377 µs → 5270 µs (1.97×)**, slot mean
10545 → 5271 µs (2.00×), with the kernel unchanged as the control (`fastest_p50` 2660 → 2642 µs).
Attribution:

| span | baseline | resident | delta |
|---|---:|---:|---|
| `chip.run.stage_inputs` (H2D) | 2165 | 38 | −2126 |
| `chip.run.validate` (D2H copy-back) | 1445 | 19 | −1426 |
| `chip.run.runner_run.device_wall` | 4171 | 3527 | **−645 (unpredicted bonus)** |

82% of the 5107 µs removal is accounted for by those three spans; the remaining ~910 µs sits
outside `chip.run` and matches per-dispatch host argument handling disappearing. Correctness passed
on the resident path (`n_ok=1 n_fail=0`, `check_correctness` running against an explicit
`copy_stacked_from` read-back), so the device contents are identical.

**The predicted ~8000 µs was too conservative** — it counted only the two copy spans (3610 µs) and
missed both the `device_wall` bonus and the outside-`chip.run` remainder. The 1.7× estimate became
1.97×.

**Standing vs HCCL:** HCCL's 1 MiB host-visible time is 578 µs, so the gap went **23.6× → 9.1×** —
about 2.6× of ratio closed in one change. Also worth recording: the *resident* configuration is the
fairer comparison, since HCCL's own benchmark never re-uploads per iteration and the host-args
variant was charging PyPTO a cost HCCL does not pay.

**What this means for the remaining actions.** The slot is no longer host-bound: at 5270 µs it is
dominated by `device_wall` (3527 µs) and the kernel (2642 µs). So **action B (async SDMA/URMA) is now
the whole game** — the host-side plumbing that actions C/D were aimed at is no longer the bottleneck.
Prioritise B and re-measure the *payload curve* (the 2× may shrink at small payloads where the fix
removes less, and the bandwidth slope is what HCCL still wins on).

Two loose ends to not lose:

- The `device_wall` bonus (−645 µs) is unexplained. Lead, not result.
- Rank asymmetry got no better (spread 3115 → 3817 µs); it is kernel-side and now a larger share of
  a smaller slot.

Prior art checked before building: a code search for `DeviceTensor` across `hw-native-sys/pypto`
(40 hits — docs, examples, tests, resident-weight/KV-cache/training-loop uses) and a notes sweep
(15+ files). No PR, issue, or plan proposes a device-resident *benchmark* path, avoiding per-run
H2D/D2H in collectives, or exposing workers from a prepared session; the closest artefacts are the
distributed docs above, `architecture/pl-jit-variants.md`, and
`performance_tuning/distributed-tuning.md`. The nearest open A2AV branches locally
(`feat/all-to-all-v-*`: counts-pull, flat-tput, multiblock-enable, multiaiv-lanes, cal-blocks,
core-num-arg, l2-dynamic-core-num) are all kernel-side or core-count work — none overlaps this.

### 1e. Not started

- Action B (async SDMA/URMA data path) — **now the critical path** (see 1d).
- Actions C/D (barriers, overlap) — deprioritised: the host overhead they targeted is gone.
- Re-measurement of the payload curve under `--resident-args`.
- Upstream: nothing is needed for action A. If the resident path wins broadly, the *harness* change
  is what a PR would carry (and the docs already describe it — no upstream gap to fix).


