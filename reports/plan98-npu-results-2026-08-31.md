# Plan 98 — Composite barrier/dcci slimming: NPU verification + measured perf (2026-08-31)

**Branch:** `perf/composite-slim-peer-dcci-barrier` on `georgebisbas/pypto`
(2 commits on `49aea216`): `79077d93` (dcci batching), `9189014a` (PIPE_V notify).
**Platform:** a2a3 — 8× Ascend 910B2, CANN 9.0.0, real NPUs.
**Interpreter:** `/usr/local/python3.12.13/bin/python3`, pypto editable at `/opt/pypto`.

---

## What changed (2 commits)

1. **`InsertCommFence` — batch the consume-side whole-GM cacheinvalid**
   (`src/ir/transforms/insert_comm_fence_pass.cpp`): a pure wait-all loop
   (`for src: if src != me: wait(...)`, the mesh composite's per-barrier wait
   loop) performs no memory access between the waits, so ONE whole-GM
   `cacheinvalid` after the loop replaces the per-wait invalidates inside it —
   `(P-1)` whole-cache `dcci` flushes per barrier generation become 1. Also
   batches runs of consecutive waits. Ring's `wait+load` per-step loops are NOT
   pure → unchanged (conservative).
2. **`pld.system.notify` — drain only `PIPE_V` before `TNOTIFY`**
   (`src/backend/common/pto_ops_distributed.cpp`): PTOAS's TNOTIFY already
   drains MTE2/MTE3; the extra `PIPE_ALL` drain existed only for the VEC
   read-complete gap. `PIPE_V` is the minimal scope → each notify drops the
   redundant MTE2/3/other-pipe waits (`(P-1)` full-pipe syncs per generation
   in the mesh composite).

## Correctness (real NPUs)

| Suite | Devices | Result |
|---|---|---|
| `test_l3_tensor_allreduce_intrinsic.py` (mesh composite, Sum/Max/Min/Prod, sizes 8–65536, fp16 arbitrary lengths) | P=2 d0-1 | **18 passed, 7 skipped** |
| same | P=4 d4-7 | **25 passed** |
| `test_l3_self_notify_credit_reset.py`, `test_l3_allreduce_ring.py`, `test_l3_allgather.py`, `test_l3_put.py`, `test_l3_get.py` | P=2 d0-1 | **11 passed, 2 skipped** |
| UTs: `test_insert_comm_fence.py` (incl. 2 new batching tests) + `test_distributed_pto_codegen.py` (notify PIPE_V) | — | **65 passed** (12 `remote_load` failures confirmed **pre-existing on origin/main**) |

Skips are device-count-gated (fp16 P=4 arbitrary lengths). The credit-barrier
protocol, put/get notify sites, and the (unchanged) ring path are all verified.

## Measured perf A/B (device_wall_s median, pypto-composite, `--persistent`)

Controlled interleaved campaign on the shared box (`plan98_before` /
`plan98_after` / `plan98_before_r2` / `plan98_after_r2`, 10–15 timed rounds,
correctness ✅ every run; box-health probe gating):

| P | count | BEFORE | AFTER | robust delta |
|---|---|---:|---:|---:|
| 2 | 64K | 322 µs | 311 µs | **−3.4 %** (noise floor) |
| 2 | 256K | 749 µs | 512 µs | **−31.6 %** |
| 2 | 1M | 1705 µs (1293 µs r2) | 1031 µs | **−20.3 %** (vs best BEFORE) |
| 4 | 64K | 996 µs | 824 µs (r2) | **−17.3 %** |
| 4 | 256K | 1486 µs (1412 µs r2) | 1063 µs | **−24.7 %** (vs best BEFORE) |
| 4 | 1M | 3191 µs | 2679 µs (r2) | **−16.0 %** |

**Headline: 16–32 % on-device time removed** at ≥256K, growing with payload —
consistent with the mechanism (each chunk-barrier generation used to pay
`(P-1)` whole-cache `dcci` + `(P-1)` full-pipe barriers; both are now O(1) per
generation). This reclaims the +15–40 % device cost measured in the 2026-08-28
"handicapped" A/B, confirming the attribution. `execute_s` is essentially flat
(the residual L3→L2 dispatch round-trip dominates it — plan 101/102 territory).

## Reproduce

```bash
cd /opt/pypto-profiling
export PATH=/usr/local/python3.12.13/bin:$PATH
# BEFORE: pip install --no-build-isolation -e /tmp/pypto-main  (origin/main)
# AFTER:  pip install --no-build-isolation -e /opt/pypto       (branch)
for spec in "2 65536 collectives/cases/mesh_p2_count65536_fp32_a2a3_d0-1.json" \
            "2 262144 collectives/cases/mesh_p2_count262144_fp32_a2a3_d0-1.json" \
            "2 1048576 collectives/cases/mesh_p2_count1048576_fp32_a2a3_d0-1.json" \
            "4 65536 /tmp/plan98_cases/mesh_p4_count65536_fp32_a2a3_d4-7.json" \
            "4 262144 /tmp/plan98_cases/mesh_p4_count262144_fp32_a2a3_d4-7.json" \
            "4 1048576 /tmp/plan98_cases/mesh_p4_count1048576_fp32_a2a3_d4-7.json"; do
  set -- $spec
  PYTHONPATH=. python3 -m collectives.run_sweep pair-mesh \
    --case-file "$3" --stacks pypto-composite --persistent \
    --warmup-rounds 3 --timed-rounds 15 --campaign plan98 --out "results/campaigns/plan98/p${1}_c${2}/results.json"
done
# device_wall medians: results/campaigns/plan98_{before,after,before_r2,after_r2}/p*_c*/results.json
```

Raw campaigns (gitignored): `results/campaigns/plan98_{before,after,before_r2,after_r2}/`.

## Notes

- The 2026-08-28 "handicapped" A/B measured the COMBINED cost of both additions
  (+15–40 %); this A/B removes both and lands at the same 16–32 % — the split is
  ~equal or barrier-dominated, per-config dependent (PMU attribution is a
  follow-up).
- Ring path intentionally unchanged (wait+load loop is not pure) — verified by
  `test_l3_allreduce_ring.py` passing.
- Follow-ups (plan 103): hoist the per-notify `PIPE_V` to once per notify-loop,
  and peer-region `cacheinvalid` (plan 68) to drop the whole-GM flush entirely.
