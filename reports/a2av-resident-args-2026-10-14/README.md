# Action A — worker-resident benchmark arguments (EP8, 1 MiB)

**Date:** 2026-10-14 · **Box:** hng-atlas02 · **Queue task:** `task_20261014_112920_242602513476`
(`--device auto --device-num 8 --max-time 3300`) · **Branch:** `a2av-resident-args`
(stacked on `a2av-harness-l2-core-num` @ `22fc5886`)

**Headline: the host-visible slot for EP8 / `managed-l2` / L=8 / 1 MiB / uniform goes
10377 µs → 5270 µs — a 1.97× speedup with the kernel untouched.** The `all_to_all_v`
kernel span is identical to within noise (`fastest_p50` 2660 → 2642 µs), which is the control:
only the host-side argument path changed.

## What was changed

One harness flag, `--resident-args`, plus a driver passthrough. No kernel, no runtime,
no upstream change.

`pypto` already ships everything needed. `compiled.prepare(...)` returns a
`DistributedWorker`, not a bare handle, and that object *is* the device-memory surface
(`docs/en/user/distributed/03-execution.md:31-41`):

| Method | What it does here |
|---|---|
| `rt.alloc_stacked_tensor(host)` | uploads each leading-dim shard of `host` **once**, shard `i` → card `i` |
| `rt.copy_stacked_from(stacked, host)` | explicit staged D2H read-back, symmetric |
| `rt.free_stacked_tensor(s)` | release (also auto-freed at `close()`) |

`alloc_stacked_tensor` defaults `worker_ids=range(B)` and its docstring names the program it
serves: `for r in range(world_size): child(x[r], device=r)`. The A1 harness orchestration
reads exactly that way — `stage_orch(inputs[r], …, device=r)`,
`fill_counts_orch(send_counts[r], …, device=r)`,
`consume_orch(…, outputs[r], recv_outputs[r], device=r)` — and all four tensors are `[p, …]`
with the rank in dim 0. So shard `i` lands on the card whose dispatch consumes it; there is
no re-indexing and no `worker_ids` override.

Implementation in `all_to_all_v_benchmark.py::_timed_session`: when `--resident-args` is set,
the four args are allocated once per session, the loop dispatches `DeviceTensor`s, and
`copy_stacked_from` runs only (a) immediately before `check_correctness` and (b) once at
session end — both outside the timed region. `pypto.runtime.bench` points at exactly this
pattern: *"benchmark resident tensors with an explicit `compiled.prepare()` dispatch loop."*
The default (no flag) path is byte-for-byte the old host-tensor behaviour, and the driver
folds the extra flags into the cell tag so `summary.json` and the attempt log stay
self-describing.

## Result

Same binary, same cell, back-to-back inside **one queue grant**.

| metric | baseline (host args) | `--resident-args` | delta |
|---|---:|---:|---|
| **host slot p50** | **10377 µs** | **5270 µs** | **−5107 µs (1.97×)** |
| host slot mean | 10545 µs | 5271 µs | −5274 µs (2.00×) |
| kernel p50 (control) | 2660 µs | 2642 µs | −17 µs (1.01×) |
| kernel mean (control) | 2660 µs | 2638 µs | −21 µs (1.01×) |
| egress | 22.08 GB/s | 22.26 GB/s | +0.8% |
| rank spread | 3115 µs | 3817 µs | +702 µs |

### Where the removed time went (median µs per run, from the `[STRACE]` spans)

| span | baseline | resident | delta |
|---|---:|---:|---|
| `chip.run.stage_inputs` — H2D of inputs | 2165 | 38 | **−2126** |
| `chip.run.validate` — D2H copy-back (misleading name) | 1445 | 19 | **−1426** |
| `chip.run.runner_run.device_wall` — on-NPU orchestration | 4171 | 3527 | **−645** |
| `chip.run.prepare_execution` | 59 | 57 | −1 |
| `chip.run.bind` | 24 | 23 | −1 |
| `chip.run` (whole span) | 8488 | 4014 | −4474 |

The two spans the change targets are gone (2165→38 µs, 1445→19 µs — the residue is one
allocation-time upload and one read-back, not per-run copies). **82% of the removed slot
time is accounted for by the three spans above**; the remaining ~910 µs sits outside
`chip.run`, consistent with per-dispatch host argument handling disappearing.

Two findings worth carrying forward:

1. **The bonus is real but unexplained.** `device_wall` — which measures the on-NPU
   orchestrator, not host copies — also fell 4171 → 3527 µs. The orchestrator's per-rank
   argument marshalling is cheaper when the tensors are already resident, but this was not
   predicted and has not been chased. Treat it as a lead, not a result.
2. **The rank asymmetry is unchanged and now a larger share.** `slowest_mean` 5775 → 6455 µs
   and spread 3115 → 3817 µs: the device-level per-rank asymmetry documented in the parent
   report (rank 7 ≈ 186 µs vs rank 0 ≈ 4.4 ms) is a kernel-side property. Removing host
   overhead did not touch it, so it now accounts for a bigger fraction of a smaller slot.

### Correctness

The resident path is validated: `check_correctness` runs in both variants, and under
`--resident-args` it runs against a `copy_stacked_from` read-back performed just before it.
Both queue cells completed `rc=0` with `n_ok=1 n_fail=0`, and the harness raises on any
mismatch. Device contents are therefore identical, not merely faster to produce.

### Raw evidence

The two driver result blocks, verbatim (the surrounding `variant_*.log` files are gitignored
per policy; they are the driver's own output and regenerate from `run_ab.sh`):

```text
=== p8_l8_managed-l2_uniform_1048576 devices=0,1,2,3,4,5,6,7 attempts=2 ===
  OK attempt=1 metric=aicore_gang_span  kernel_p50=2659.76  slot_p50=10377.059981692582  egress=22.079434480165446
     breakdown (median per run): stage_in=2165us  copy_back=1445us  device_wall=4171us
n_ok=1 n_fail=0

=== p8_l8_managed-l2_uniform_1048576_resident-args devices=0,1,2,3,4,5,6,7 attempts=2 ===
  OK attempt=1 metric=aicore_gang_span  kernel_p50=2642.42  slot_p50=5269.8549698106945  egress=22.25592534897912
     breakdown (median per run): stage_in=38us  copy_back=19us  device_wall=3527us
n_ok=1 n_fail=0
```

`stage_in=38us` / `copy_back=19us` is the proof the `DeviceTensor` path engaged: the spans
are still emitted, but they no longer carry a per-run payload copy.

## Standing vs HCCL

HCCL's `HcclAlltoAllV` host-visible time is **578 µs** at 1 MiB (flat across payloads —
it is a device/RDMA path with almost no host involvement). The gap:

| | slot @ 1 MiB | gap vs HCCL |
|---|---:|---:|
| PyPTO `managed-l2` L=8, host args | 13615 µs (sweep) / 10377 µs (this A/B) | 23.6× / 18.0× |
| PyPTO `managed-l2` L=8, **resident args** | **5270 µs** | **9.1×** |

So this closes about **2.6×** of the ratio in one change. Note also that the resident
configuration is the *fairer* comparison: HCCL's own benchmark is resident by construction
(it never re-uploads per iteration), so the host-args variant was penalising PyPTO with a
cost HCCL never pays.

**What remains is no longer host-side.** At 5270 µs the slot is dominated by
`device_wall` (3527 µs) and the kernel (2642 µs) — i.e. the on-device data path, which is
action B (async SDMA/URMA) in `GAP_CLOSING_NOTES.md`, not more host tuning. The next
experiment should target the bandwidth slope, not the slot plumbing.

## Caveats

1. **One cell, 20 rounds, one payload.** A single EP8 / `managed-l2` / L=8 / 1 MiB / uniform
   point, not the payload curve. The 2× is large and mechanistically explained, but it has
   not been re-measured across payloads or at L=1/L=16.
2. **The baseline moved.** Today's baseline is 10377 µs where the sweep recorded 13615 µs for
   the same nominal cell, so absolute cross-campaign comparisons should use ratios. The A/B
   itself is unaffected — both variants ran minutes apart on an identical box state inside one
   grant, which is the only reason the comparison is clean.
3. **`--resident-args` changes what is being measured**, deliberately: it benchmarks the
   resident dispatch path rather than the upload-per-call path. It is a harness *mode*, not a
   regression fix for existing behaviour.
4. Not yet re-run against HCCL in the same session, so the 9.1× is assembled from two
   campaigns.

## Reproduce

```bash
# inside a queue-joined container (the payload runs in the caller's context — see
# ../../reports/npu-queue-runbook.md trap 10)
docker exec <container> bash -lc '
  export PATH="$PATH:/var/lib/taskqueue/bin"; export TASKQUEUE_CONF=/var/lib/taskqueue/taskqueue.conf
  task-submit --device auto --device-num 8 --max-time 3300 --timeout 0 \
    --run "bash /mounted_home/hw-native-sys/pypto-profiling/reports/a2av-resident-args-2026-10-14/run_ab.sh"'
```

`run_ab.sh` runs variant A then variant B with all other conditions equal;
`python3 compare.py` prints the tables above from the two `summary.json` files.

## Artifacts

`run_ab.sh` · `compare.py` · `variant_a.log` / `variant_b.log` (full driver output) ·
`baseline/` and `resident/` (each with `summary.json`, `json/`, `attempt_logs/`, `builds/`).
`builds/` and `attempt_logs/` are gitignored per
[`../README.md`](../README.md); they remain on hng-atlas02.
