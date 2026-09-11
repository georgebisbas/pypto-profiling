# Message draft for Yunji (#2521 A1 baseline)

Copy/paste-ready note. Absolute links assume repo
`https://github.com/georgebisbas/pypto-profiling` on `main`.

---

Yunji — sharing an A1-style pre-change baseline for managed `pld.tensor.all_to_all_v` against your #2521 roadmap (§8 / work item A1).

**What we archived (measurement contract)**
- INT8 builtin, `core_num=1`, `L=1`, `persistent=True`, 5 warmup + 100 measured, swimlane + timing-slot (`--profile both`)
- §8.3 payload list exact (0…1 MiB, incl. DSV4 **24960** and combine **49152**)
- Rails: `managed-l2` + `managed-host`
- **EP2 + EP4:** full §8.3 + zero / single-hot @24960
- **EP8 (devices 0–7):** full §8.3 on **both** rails + **all seven** §8.4 patterns @24960 on both rails
- Not in this drop: §8.5 `L>1` / EP16 (A3 / needs O2); §8.3 sub-32 B tails (INT8 align)

**Takeaways (use L2 AIV gang as the official §8.7 kernel number)**
1. **Kernel is fast and ~linear in payload** once past ~8–16 KiB. Examples (L2 AIV p50): EP2 8 KiB ≈ **29 µs**, 64 KiB ≈ **210 µs**, 1 MiB ≈ **3.4 ms**; EP8 24960 ≈ **589 µs**, 64 KiB ≈ **1.16 ms**, 1 MiB ≈ **18.0 ms**.
2. **Remote egress plateaus** ~**2.3–2.5 Gbps (EP2)**, ~**2.6–2.9 Gbps (EP4)**, ~**2.3–3.3 Gbps (EP8)** through 1 MiB — not climbing with size.
3. **EP scaling of AIV:** EP4/EP2 ≈ **~2.5×** at matched mid/large sizes; EP8/EP4 ≈ **~1.9–2.3×** (≈2× peer count), with similar Gbps band — consistent with more remote traffic, not a cliff.
4. **Full-program slot ≫ AIV at small/mid sizes** (stage + consume + dispatch in our timing slot; §8.1 wants them out — so treat slot as e2e, AIV as kernel). e.g. EP2 @8 KiB slot/AIV ~**60×**; EP8 @1 MiB slot/AIV ~**1.5×**.
5. **Fair HOST vs L2 (slot vs slot only):** L2 is **~1.1–2.0×** faster end-to-end — **not** 40–100×. Gap shrinks with size: EP2/EP4 @1 MiB ~**1.1–1.2×**; EP8 @24960 ~**1.1×**, @64 KiB ~**1.4×**, @1 MiB ~**1.1×**. Do **not** compare L2 AIV to HOST “kernel” (HOST has no swimlane AIV name yet).
6. **24960** can look like a slot/egress spike vs 24 KiB/32 KiB neighbors — different packed shape (`C=4160` / MAX_RECV path); prefer AIV for kernel trends. Rechecked on EP2/EP4: spike is real, not a one-off.

**Caveats for K1 denominator use**
- EP8 needed **local `SIMPLER_COMM_FORCE_IPC`** (Fabric V2 `ExportToShareableHandleV2` “cross-server” → AICPU `507018` / `release_domain`); numbers are on this box with that workaround, not yet default upstream.
- Harness UB tile fix required for ≥256 KiB C′ compile.
- No EP16; HOST AIV attribution still open.

**Artifacts**

| What | Link |
|------|------|
| Colleague analytic report (tables + §8 checklist) | https://github.com/georgebisbas/pypto-profiling/blob/main/reports/issue-2521-a1-alltoallv-ubfix-2026-09-09/ANALYTIC_REPORT_FOR_COLLEAGUES.md |
| Campaign driver | https://github.com/georgebisbas/pypto-profiling/blob/main/collectives/alltoallv_a1.py |
| Harness | https://github.com/georgebisbas/pypto-profiling/blob/main/collectives/a1_alltoallv/harness/all_to_all_v_benchmark.py |
| FORCE_IPC patch | https://github.com/georgebisbas/pypto-profiling/blob/main/collectives/a1_alltoallv/patches/simpler_comm_force_ipc.patch |
| Bundle README | https://github.com/georgebisbas/pypto-profiling/blob/main/collectives/a1_alltoallv/README.md |
| EP8 gap-fill archive | https://github.com/georgebisbas/pypto-profiling/tree/main/reports/issue-2521-a1-alltoallv-ep8-gapfill-2026-09-10 |
| Roadmap Part 1 (§8) | https://github.com/hw-native-sys/pypto/issues/2521#issuecomment-5495422542 |
| Roadmap Part 2 (A1–A3) | https://github.com/hw-native-sys/pypto/issues/2521#issuecomment-5495424411 |
| Issue #2521 | https://github.com/hw-native-sys/pypto/issues/2521 |

Happy to adjust if anything diverges from §8 intent before you freeze this as the K1 baseline.
