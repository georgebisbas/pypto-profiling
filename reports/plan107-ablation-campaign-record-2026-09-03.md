# Plan 107 — re-anchored ablation campaign record (2026-09-03)

**Status:** executed (re-anchored on the current composite gap) — **attribution table
DEFERRED** (timing contaminated by shared-box noise; see §6). Two hard findings landed:
fillpad NPU crash (root-caused) and P=8 box wedge.
**Companion:** the fillpad minimal-reproducer / root-cause bundle lives in `/tmp/fp_repro/`
(kernels `tfillpad_probe_{top,low}.cpp`, driver, `issue_body.md`, logs).

---

## 1. Context

Plan 107 (`pypto-3.0-notes/pr_plans/107-composite-gap-ablation-campaign.md`) decomposes the
pypto composite-vs-hand-written mesh-allreduce gap into individual codegen constructs, one
construct per ablation kernel. The original premise (6.7× / composite 0.888 ms @64K P=2,
from 2026-08-28 §3.9) was **not reproduced** — see the baseline-gate report
`results/campaigns/plan107-baseline/GATE_REPORT.md`. Per the operator decision the campaign
was **re-anchored** on the current gap (~4× at the headline config) and run against the
current `origin/main` environment.

## 2. Environment

| Component | Value |
|---|---|
| Box | 8× Ascend 910B2 (a2a3), shared/multi-tenant; AICore% column is a known visual artifact |
| pypto | `origin/main` `d9d3dd60` (rebuilt editable) |
| simpler runtime | `/opt/pypto/runtime` `15f5cbd` (pypto main's submodule pin; `_task_interface` rebuilt) |
| pypto-profiling | `a3a58ef` + commit `7a74a2d` (5-file `TaskTensor→Tensor` port) + uncommitted fillpad/allfour slot fix |
| Interpreter | `/usr/local/python3.12.13/bin/python3` |
| Protocol | `collectives.run_sweep pair-mesh`, `--persistent`, 3 warmup + 15 timed, metric `device_wall_s_median` |
| Devices | P=2 d0-1, P=4 d4-7, P=8 d0-7; single-AIV (`core_num=1`) |
| Payloads | counts 256/4096/16384/65536/262144/1048576 fp32 (1 KiB → 4 MiB) |

**Kernel port:** the 7 AIV kernels' tensor decls were renamed `TaskTensor → Tensor`
(required by runtime `15f5cbd`, which removed `TaskTensor`). Committed for the 5 non-fillpad
kernels (`7a74a2d`). `fillpad`/`allfour` carry the rename **plus** an uncommitted TASSIGN
slot-order fix from the fp_repro root-cause work (owned separately).

## 3. Results matrix (what completed / correctness)

| Cell group | Status |
|---|---|
| pypto-composite curve (3 P × 6 counts) | ✅ 18/18 pass |
| P=2 variants (6 × 6) | ✅ mesh/handicapped/chunkbarrier/chunkdcci all counts; fillpad/allfour only c256/c4096 (crash ≥16384) |
| P=4 variants (6 × 6) | ✅ same as P=2 |
| P=8 variants (6 × 6) | ⚠️ mesh 6/6 + handicapped c256/c4096; then **box wedge** (see §5) |
| **Correctness audit** | **82 cells PASS, 0 correctness failures** (every run that completed passed golden; crash/wedge cells wrote no results) |

## 4. Findings

### 4.1 The 6.7× premise is obsolete (re-anchored gap ≈ 2–4×)
Baseline gate (P=2, count 65536, d0-1): **pypto-composite 357 µs vs simpler-own 90 µs
(3.97×)** — not 0.888 ms / 6.7×. The composite has been ≈0.29–0.46 ms at this config since
≥ 2026-08-31 (matches the `pr2591_final_base` legs, 361 µs on `49aea216`). #2591 is not
merged; the 0.888 ms figure was an 08-28 outlier even against that report's own E1/E2/E3 rows.

### 4.2 `fillpad` (and `allfour`) crash the a2a3 device at count ≥ 16384 — ROOT-CAUSED
- Symptom: `chip run lane is poisoned: finalize_native_run failed with code -100`
  (device-scheduler timeout) at the box-health probe for count ≥ 16384; PASS at 256/4096.
  Reproduces at P=2 (d0-1) and P=4 (d4-7); control runs (`mesh`, `chunkdcci` @65536) pass
  immediately after → box healthy, kernel-specific.
- Root cause (fp_repro bundle): a **full-width 16384-float32 tile occupying the top 64 KiB UB
  slot** makes TFILLPAD's tail-pad `vector_dup` address UB offset `0x30000` — one past the
  192 KiB UB. Probe with the tile at slot `0x0` passes; the top-slot tile crashes.
- **The pypto composite is NOT affected** (its TFILLPAD is on 1×4096 tiles with headroom);
  pto-isa not implicated. This is an **ablation-geometry bug**, not a composite/compiler defect.
- Working-tree fix (uncommitted): move `recvTile` (the only TFILLPAD'd tile) to slot `0x0`,
  `stageTile` to `0x20000`. **Not yet re-validated on-device** at the time of writing.

### 4.3 Composite device_wall medians (µs) — the re-anchored reference
| P | 1 KiB | 16 KiB | 64 KiB | 256 KiB | 1 MiB | 4 MiB |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 283 | 256 | 331 | 335 | 561 | 1027 |
| 4 | 760 | 753 | 1067 | 819 | 1464 | 2908 |
| 8 | 1679 | 1701 | 1843 | 2560 | 2849 | 6335 |

## 5. P=8 box wedge (environmental)

The P=8 variant sweep degraded at ~07:50 (after `mesh` 6/6 and `handicapped` c256/c4096):
every subsequent P=8 run hung and failed with `child process(es) did not exit within the
close budget` for over an hour. After killing the sweep, even a `mesh` P=8 smoke hangs at the
CommDomain probe while P=2 on d0-1 and d2-3 pass. `npu-smi` reports all Health OK — the wedge
is not reflected there. The other agent's `fp-crash-confirm` NPU runs (d0-1, 08:12+) overlapped
the tail of the P=8 window. **P=8 variant data is incomplete and blocked until the box recovers.**

## 6. Data quality caveat — attribution table DEFERRED

The timing data collected (07:11–09:00) is **contaminated by shared-box noise/contention** and
is NOT suitable for the per-construct cost table:
- spread ratios up to 289 (bimodal rounds; e.g. `handicapped` P4 @64KiB mean 5879 µs vs median 496 µs);
- non-physical baselines (`mesh` P4: 503 µs @1 KiB vs 228 µs @256 KiB; `mesh` P8 4655 µs @1 KiB);
- `handicapped` (plan-98: +25%) appears *faster* than a contaminated `mesh` in several cells.

Directional only (treat with caution), P=2 d0-1 cells where the baseline is self-consistent
(e.g. `chunkbarrier` @1 MiB/4 MiB ≈ +80–96 % over the same-day `mesh` — consistent with the
per-chunk-barrier hypothesis; `chunkdcci` smaller). **No construct is attributed in this
record.** Raw per-cell medians/spreads are in
`results/campaigns/plan107/<variant>/p{P}_c{count}/results.json`.

## 7. Next steps

1. Validate the fillpad/allfour slot-order fix on-device (fp_repro owner), then decide whether
   to land it and re-measure fillpad/allfour at count ≥ 16384 (unblocks the allfour composition check).
2. Recover P=8 (box) and re-run the missing P=8 cells.
3. Re-run the attribution on a **quiet box with the plan-98 interleaved A/B protocol**
   (mesh↔variant per cell, medians, ⚑ flags), prioritising P=2/P=4 × multi-chunk counts
   (65536/262144/1048576) — the cells where chunkbarrier/chunkdcci actually bite.
4. Then write the per-construct % / lowering-fix-sizing report and file each attributed
   construct as its own pypto plan (anything exceeding the barrier/dcci pair first).

## 8. Report fields

box 8×910B2 a2a3 (shared, contended window) · pypto `d9d3dd60` · runtime `15f5cbd` ·
pypto-profiling `a3a58ef`+`7a74a2d` (+uncommitted fillpad fix) · P 2/4/(8 blocked) ·
payload 1 KiB→4 MiB fp32 · 3 warmup/15 timed · gate reproduced: **no** (composite 357 µs vs
expected 888 µs) · correctness 82/82 pass · sweep attribution: **deferred** (box noise).
