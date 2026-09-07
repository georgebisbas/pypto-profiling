# Reference note — pypto issue #2521 (L2 orchestration & dispatch overhead)

**Source:** https://github.com/hw-native-sys/pypto/issues/2521
**Author:** YunjiQin · **Review:** vloncar (ZeCO) — measured, not projected
**Review 2:** georgebisbas (this repo) — measured, merged PR #2591
**Roadmap:** YunjiQin 2026-09-01 (AllToAllV execution roadmap, C0 freeze)
**Last read:** 2026-09-03

This is the RFC behind the "residual ~5–11 ms L3→L2 dispatch round-trip" finding in
`benchmark-report-2026-08-28.md` §5. Keep this note next to the benchmark report: it is
the *upstream fix* for the dispatch cost our harness measures, and it contains independent
persistent-vs-per-call numbers that corroborate our `--persistent` results.

---

## 1. What the issue is

**"[RFC] Optimize collective API performance with L2 orchestration and runtime-selected
multi-AIV launches"** — moves managed collective communication from **HOST/L3 dispatch** into
**CHIP/L2 orchestration**, and extends the managed path to runtime-selected multi-AIV launch
widths. First target: A2/A3 AIV mode, EP8/EP16, MoE `all_to_all_v`.

It is explicitly **not** a duplicate of:
- **#2069** — per-dispatch `CommDomain` lifecycle (~124 ms; `persistent=True` fixes it).
- **#2398** — the `_ord` WAW-ordering token (out of scope; see §4).
- **#2399** — host graph record/replay.

It optimises the *already-present* managed collective path: the repeated L3→L2 task
setup/param/wakeup/completion round-trips that dominate the collective kernel — the same
host-dispatch overhead our `execute_s - device_wall_s` residual measures.

## 2. Independent persistent-vs-per-call numbers (vloncar, ZeCO ring "AllScan")

p50 ms, 20 samples/row, all verified vs sequential reference, single-AIV:

| shape | A: cost shared over 16/dispatch | B: per call | C: per call + persistent | B→C | C vs A |
|---| ---: | ---: | ---: | ---: | ---: |
| P=2 64² K=1 | 6.29 | 12.10 ⚑ | 2.61 | 4.6× | 2.4× |
| P=2 64² K=4 | 12.12 ⚑ | 11.12 | 2.52 ⚑ | 4.4× | 4.8× |
| P=4 64² K=1 | 13.24 ⚑ | 217.40 | 2.92 | **74.6×** | 4.5× |
| P=4 64² K=4 | 16.23 | 219.22 | 3.05 | **71.9×** | 5.3× |
| P=4 128² K=1 | 16.00 | 218.10 | 3.47 ⚑ | **62.8×** | 4.6× |
| P=4 128² K=4 | 13.18 | 219.10 | 17.65 ⚠️ | 12.4× | 0.7× |

⚑ = >2× sample spread (bimodal per-call); ⚠️ = unexplained outlier, not relied on.

**Key conclusions (directly corroborate our harness data):**
- **Persistent mode is 63–75× faster than per-call at P=4**, and a further **4.5–5.3×
  faster than amortising the cost across a 16-exchange batch**. *"Keeping the domain alive
  is not merely equivalent to amortising it — it is better, because 16 exchanges on the same
  chips contend with each other in a way a single exchange with a live domain does not."*
  → This matches our finding that `--persistent --batch 10` adds little beyond `--persistent`
  (5.1 vs 5.7 ms at P=2) and strengthens the argument for **persistent as the benchmark default**.
- The cost is **strongly rank-dependent**: at P=2 per-call and shared-cost are already close;
  at P=4 it is ~218 ms vs ~3 ms. → Same shape as our measured domain lifecycle
  (108 ms @ P=2 → 159–212 ms @ P=4).
- "Domain lifecycle plus drain overhead, and the underlying data movement was competitive
  all along" — same conclusion as #2069's ~124 ms, reached from a different direction.

## 3. HCCL AIV launch-width policy (useful for the `core_num` / B* axis)

HCCL first picks an algorithm, then its template computes the actual launch width; it does
**not** uniformly launch every available AIV. Notation: `P` = rank count, `L` = requested
block limit, `B` = blocks actually launched.

| Collective / HCCL AIV algorithm | HCCL launch-width policy | EP8, L=16 | EP16, L=16 |
| --- | --- | ---: | ---: |
| AllReduce one-shot | `B=min(L, P+1)` | 9 | 16 |
| AllReduce two-shot | `L<P+1`: `B=L`; else `B=floor(L/(P+1))*(P+1)` | 9 | 16 |
| AllGather | `B=L` | 16 | 16 |
| AllToAll, D≤512 KiB | `L≥P`: `B=P`; `L<P`: balanced | 8 | 16 |
| AllToAll, D>512 KiB | usually `B=floor(L/P)*P`; cap `L` to `4P` for P=8,D≥2MiB | 16 | 16 |
| AllToAllV | `L<P`: `B=L`; else `B=floor(L/P)*P` | 16 | 16 |

Implication for benchmarking: comparing "launch width" apples-to-apples requires knowing
`B` (blocks actually launched), not just the requested `L` — HCCL's `B` is a function of
`(P, L, message size, algorithm)`. Our harness `--core-num`/`--core-nums` sweeps `L` on the
pypto-host mesh rail; HCCL's effective `B` is a reference worth tabulating alongside.

## 4. Other technical details worth remembering

- **`_ord` (#2398) is OUT of scope for #2521.** It keys on the **window-buffer argument
  binding** (not managed collectives) and fixes a deterministic hang in user-written p2p.
  Retiring it is gated on the **manual L3 dependency API** — the runtime already
  distinguishes **`DEP_WAIT`** (ordering-only) from **`DEP_WAIT | DEP_RETAIN`** (retention);
  the compiler-side surface at L3 is not wired up yet.
- **Fence discipline is an ISA requirement, not a choice**: a publishing write must be
  ordered (e.g. `dsb(DSB_DDR)`) before `TNOTIFY` for push templates; pull (mesh allreduce)
  uses pipe-barrier only. Push-vs-pull families differ.
- **#2069** — per-dispatch `CommDomain` lifecycle ~124 ms; `persistent=True` fixes it.
- **#2160** — `core_num` (multi-AIV launch width) exists only on the HOST builtin; helps the
  latency/barrier-dominated regime, does not reduce remote traffic (see
  `multicore_allreduce_speedup_model.md`).
- **#2280** — plan 53: TPUT push + O(1) `NeighborBarrier` to replace the O(P²) `RoundBarrier`
  in the HOST builtin (NPU-safe ordering).

## 5. How this connects to our measured results

| Our measurement (this repo) | Issue #2521 corroboration |
|---|---|
| Domain lifecycle ~108 ms @ P=2, ~159–212 ms @ P=4 | #2069 ~124 ms; vloncar 218 ms @ P=4 |
| `--persistent` ~19–25× faster; `--batch` adds little | 63–75× faster, and beats 16-batch amortisation |
| Residual ~5–11 ms `execute_s − device_wall_s` | the L3→L2 task round-trip this RFC removes |
| `device_wall` ~2–16× of HCCL (the real device gap) | "underlying data movement was competitive all along" |

## 6. Status update (2026-09-03)

The issue thread has moved since this note was written. Additions relevant here:

- **Our measured results are now on the thread** (georgebisbas, 2026-08-31): the persistent-mode
  19–25× / `--batch`-adds-little findings of §5, the pull-family notify-barrier + `dcci`-batching
  work (**merged PR [#2591](https://github.com/hw-native-sys/pypto/pull/2591)**, commit `690da78`,
  refs #2521), and the `core_num` sweep (single-AIV ~1.2 GB/s → cn8–16 ~3.6–5.5 GB/s,
  `B*` ∈ {8,16}, box-load-sensitive). Final balanced numbers for #2591 are in
  [`benchmark-report-2026-09-02-pr2591-final.md`](benchmark-report-2026-09-02-pr2591-final.md)
  (−4…−16 % at P=2/4, ~0 at P=8); the crossover / B\* analysis is in
  [`corenum-message-size-crossover-2026-08-31.md`](corenum-message-size-crossover-2026-08-31.md).
  The A/B campaign, emitted-kernel artifacts (`reports/barrier-dcci-codegen/`), and the analytic
  pipeline are the reproducibility trail behind the numbers posted to the issue.
- **The RFC author published an AllToAllV execution roadmap** (YunjiQin, 2026-09-01): C0 contract
  freeze ✅ (14 items); three parallel tracks (benchmark / L2 sinking / kernel rework); work items
  `A1`+`O1` ready to start; milestones M0–M5. Two corrections that affect how our numbers should
  be read:
  - the current hand-written kernel *already* transfers only the clamped valid prefix
    (exact-traffic fix #2524 / PR #2536, merged 2026-08-28), so there is no "full-capacity →
    exact" work item — the RFC body's "transfers padding" motivation describes the pre-#2536 state;
  - the roadmap's benchmark contract (`persistent=True`, `reset_persistent_windows=False`,
    5 warmup / 100 measured, fastest-rank-mean swimlane gang span, ≥50 runs/case) matches this
    harness's `--persistent` defaults, so our persistent numbers are directly comparable.

See also: `pypto-3.0-notes/collectives_benchmarking/issue2521_expert_review.md` and
`issue2521_l2_orchestration_multiaiv_collectives.md` (fuller review corpus).
