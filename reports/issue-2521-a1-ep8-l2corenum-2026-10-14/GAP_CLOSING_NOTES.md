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

## Step 1 — (next) make the terms separable, then attack A

To do when the box is next free:
1. Emit a **breakdown** instead of the single `host_timing_slot`: `stage_in / device / copy_back /
   dispatch`, parsed from the `chip.run.*` spans. Then the headline becomes falsifiable and comparable
   to HCCL line by line.
2. Determine whether the harness can pass **device-resident** tensors (or a reused binding) so
   `copy_in_run_inputs_impl` leaves the timed path. If the API forbids it, that is the change to
   propose upstream in `simpler`.
3. Only then re-measure B (async DMA) against a number that means something.

_(nothing below this line is done yet — the box has not been re-booked for Step 1)_

