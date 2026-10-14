# Closing the HCCL gap — working notes

**Goal:** try the ranked fixes from `HCCL_VS_PYPTO_A2AV.md` § "What is missing", one at a time, with a
measurement after each. Baseline to beat: **EP8, 1 MiB/peer, `managed-l2 L=8` = 13615 µs host-slot**
(HCCL 578 µs).

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

### 1b. Device-resident I/O (action A) — mechanism confirmed, **blocked on an upstream accessor** ⛔

The runtime will skip both copies if the tensor is already device memory — in
`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`:

```cpp
if (t.is_device_memory()) { out->add_tensor(t); continue; }   // pass-through, no copy
```

and `pypto.runtime.DeviceTensor` is the public handle: *"Pass it to CompiledProgram in place of a
torch.Tensor to skip the host→device copy on entry and the device→host copy on exit."* Allocation is
`Worker.alloc_tensor(shape, dtype, init=<host tensor>, worker_id=...)` — allocate once with an
initial upload, then pass the handle to every `run()`.

**What blocks it:** `alloc_tensor` is defined on the abstract `Worker` (`runtime_base.py:247`), but
the harness only ever holds the session yielded by `compiled.prepare(...)`. There is no public way to
get the `Worker` (or workers — `worker_id` is per-rank, so 8 of them) back out of that session. And
the harness's tensors are `share_memory_()` host tensors (`:1070`) that the 8 worker processes share,
so the change is not a one-liner even once the handle is reachable.

So action A splits into:

| Piece | Where | Status |
|---|---|---|
| `DeviceTensor` handle + pass-through in the runtime | simpler / pypto runtime | **already exists** |
| public accessor for the prepared `Worker`(s) | pypto `distributed_runner` | **missing — the blocker** |
| harness passes device tensors, reads outputs back once for `check_correctness` | A1 harness | blocked on the above |

**Predicted payoff** (from the breakdowns, per rank per run): 3.09 ms + 2.55 ms at 1 MiB, 0.60 ms +
0.64 ms at 256 KiB — i.e. it should take the 1 MiB `managed-l2 L=8` slot from **13615 µs to ~8000 µs**
(≈1.7×) without touching the kernel, and it removes a term HCCL never had.

### 1c. Not started

- Action B (async SDMA/URMA data path) — still the right fix for the *slope*, unstarted.
- Actions C/D (barriers, overlap).
- A real PR for the worker accessor.

