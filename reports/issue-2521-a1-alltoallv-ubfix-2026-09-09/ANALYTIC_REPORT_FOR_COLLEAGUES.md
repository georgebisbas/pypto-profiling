# Analytic report: RFC #2521 A1 AllToAllV baseline (INT8)

**For:** colleagues reviewing A1 performance / HOST vs L2  
**Campaigns:**
- **EP2/EP4 full curve:** `issue-2521-a1-alltoallv-ubfix-2026-09-09` (72/72 OK)
- **EP8 mid-curve + L2 vs HOST:** `issue-2521-a1-alltoallv-ep8-2026-09-10`, `issue-2521-a1-alltoallv-ep8-l2host-2026-09-10` (+ `zero_force_ipc/`)
- **EP8 continuation (FORCE_IPC):** `issue-2521-a1-alltoallv-ep8-forceipc-2026-09-10` (L2 `24576` + `24960` OK after hardened IPC patch)
**When (UTC):** EP2/EP4 2026-09-09; EP8 insight + FORCE_IPC probes 2026-09-10  
**Roadmap (YunjiQin, 2026-09-01):**
[Part 1 — design & §8 benchmark](https://github.com/hw-native-sys/pypto/issues/2521#issuecomment-5495422542) ·
[Part 2 — work items A1–A3](https://github.com/hw-native-sys/pypto/issues/2521#issuecomment-5495424411)  
**Artifacts:** `/opt/pypto-profiling/reports/issue-2521-a1-alltoallv-ubfix-2026-09-09/` (EP2/EP4) · EP8 dirs above  
**Scripts (in this repo):** [`collectives/a1_alltoallv/`](../../collectives/a1_alltoallv/) — harness snapshot, FORCE_IPC patch, Fabric MFE; driver [`collectives/alltoallv_a1.py`](../../collectives/alltoallv_a1.py).  
**Reproduce:** see **§10** (code paths, commits, exact commands, Fabric MFE).

---

## How this was run (one screen)

| Layer | Path in `pypto-profiling` | Timed? |
|-------|---------------------------|--------|
| Campaign driver | `collectives/alltoallv_a1.py` | No (loops cells) |
| Measured program | `collectives/a1_alltoallv/harness/all_to_all_v_benchmark.py` | Yes — each cell |
| Timed boundary | one `runtime.run(...)` wall clock (`host_timing_slot_*`) | Includes stage + consume + **outer L3→L2 dispatch(es)** + collective; **`prepare` outside** |
| Official L2 kernel | swimlane AIV gang → `fastest_p50_us` when `metric=aicore_gang_span` | Collective AIV only (no stage/consume/dispatch) |
| EP8 IPC workaround | `collectives/a1_alltoallv/patches/simpler_comm_force_ipc.patch` → `SIMPLER_COMM_FORCE_IPC` / `A2AV_FORCE_IPC=1` | Skip Fabric + treat any Fabric export fail as IPC fallback |

---

## 1. Executive summary

We measured managed **CHIP/L2** and managed **HOST/L3** `pld.tensor.all_to_all_v` (INT8 builtin, `core_num=1`) on a local 8× Ascend 910B2 box for **EP=2** (devices 0,1) and **EP=4** (devices 0–3), across a payload curve from 0 B to 1 MiB per peer, plus zero / single-hot patterns at the DSV4 dispatch size (24960 B). On **2026-09-10** we added an **EP=8** (devices 0–7) mid-curve insight: L2 AIV through **24 KiB** and DSV4 **24960**, paired L2 vs HOST slots (HOST through 16 KiB), and `peer_bytes=0` under `SIMPLER_COMM_FORCE_IPC=1`.

**Roadmap accordance (short):** EP2/EP4 follow Yunji’s **A1 measurement contract** (§8.2–§8.7) as far as this box allows. EP8 is a **partial** official archive (mid-curve + L2/HOST insight; not full §8.3 to 1 MiB). See **§2** and **§9**.

**Main findings:**

1. **Large C′ payloads (≥256 KiB) are now runnable** after a harness fix that chunks stage/consume `pl.load` tiles to 16 KiB (VEC UB ≈ 184 KiB). Earlier campaigns failed compile with `Vec buffer usage exceeds platform limit`.
2. **L2 collective AIV time is small and scales ~linearly with payload** once past ~8–16 KiB, with remote egress plateauing around **~2.3–2.5 Gbps (EP2)**, **~2.6–2.9 Gbps (EP4)**, and **~2.6–3.0 Gbps (EP8 through 24 KiB)**.
3. **L2 full-program wall time is much larger than AIV time at small/mid sizes** (often ~10–60×), because the timed program includes stage + consume + runtime dispatch—not only the collective kernel.
4. **Fair HOST vs L2 comparison (full-program slot vs slot):** L2 is faster by about **~1.1–2.0×** on EP2/EP4, and **~1.6–2.0×** on EP8 mid-curve — not orders of magnitude. The gap shrinks as payload grows (EP2/EP4 at 1 MiB: ~1.1–1.2×).
5. **Do not compare L2 “kernel_p50” (AIV gang) to HOST “kernel_p50” (timing slot).** HOST has no AIV swimlane attribution yet; its reported kernel column *is* the full-program slot.
6. **EP8 Fabric flake:** intermittent `ExportToShareableHandleV2` “cross-server” poisoned long EP8 jobs (often surfaced as `release_domain -1`). Local `SIMPLER_COMM_FORCE_IPC=1` plus a **hardened Fabric→IPC fallback** (any export failure, not only `FEATURE_NOT_SUPPORT`) got EP8 L2 **24576** and **24960** through after earlier 24960 failures.

---

## 2. Accordance with Yunji’s #2521 roadmap (A1)

Yunji’s roadmap splits AllToAllV work into tracks **A** (benchmark), **O** (L2 sinking), **K** (kernel). This campaign targets **work item A1** only: *“Benchmark harness and pre-change baseline”* — deliver harness, count patterns, swimlane/timing-slot parser, §8.8 JSON, and the full raw curve of the **current single-AIV** kernel, so later K1+ changes have a frozen denominator ([Part 2 §11.2](https://github.com/hw-native-sys/pypto/issues/2521#issuecomment-5495424411)).

| Track item | Roadmap intent | This campaign |
|------------|----------------|---------------|
| **C0** contract freeze | Done 2026-09-01 | Assumed; we did not change frozen ABI |
| **A1** harness + baseline | Ready to start → archive pre-change curve | **Partial archive** (EP2/EP4, methodology aligned) |
| **O1** L2 @ `core_num=1` | Prerequisite for L2 rail numbers | Already on `main`; we **measured** both rails |
| **O2 / K1–K4 / A2 / A3** | Blocked on A1 and/or later items | **Not in scope** (no multi-AIV, no DSV4 demo, no EP8 tuning table) |

### 2.1 Checklist vs roadmap §8 / A1 completion criteria

Legend: **Yes** = done as specified · **Partial** = same idea, incomplete coverage · **No** = not done · **N/A** = correctly deferred

| Roadmap clause | Required | What we executed | Status |
|----------------|----------|------------------|--------|
| **§8.1** harness file | `tests/st/distributed/collectives/all_to_all_v_benchmark.py` | Present (local; used by `alltoallv_a1.py`) | **Yes** |
| **§8.1** `--impl` | `managed-host` \| `managed-l2` | Both, full uniform curve each | **Yes** |
| **§8.1** `--profile` | `swimlane` \| `timing-slot` | `--profile both` (100 slot + 8 swimlane rounds) | **Yes** |
| **§8.1** stage/consume vs slot | Stage/consume **out of** measured timing slot | Stage/consume **in** slot; **out of** official AIV metric | **Partial** |
| **§8.2** dtype / shape | INT8, `C=4160`, `MAX_RECV=CeilDiv(1 MiB, C)=253` | Same | **Yes** |
| **§8.2** INT8 builtin | Add INT8 variant if missing | [PR #2714](https://github.com/hw-native-sys/pypto/pull/2714) | **Yes** |
| **§8.3** main payload curve | 0, 32, 128, 1K…24K, **24960**, 32K, **48K**, 64K…**1024K** | Exact same list | **Yes** |
| **§8.3** C′ non-multiples of C | Separate `[P·MAX_RECV, C']` instance | Used for non-multiples of 4160 (e.g. 256K+) | **Yes** |
| **§8.3** tail cases | 1/15/31/33/4159/4161 B, etc. | Not swept (many rejected by INT8 32-byte row align) | **No** |
| **§8.4** count patterns | All seven: uniform, zero, self-only, single-hot, mixed, asymmetric, random | Campaign archived **uniform + zero + single-hot** only (harness implements all seven) | **Partial** |
| **§8.5** core / `L` sweep | EP8 `L∈{1,2,4,7,8,10,15,16}`; EP16 `L∈{1,4,8,15,16}` | **`L=1` only**; `core_num>1` rejected (O2 not implemented) | **N/A** for single-AIV A1; **No** for full §8.5 |
| **§8.6** persistent | `persistent=True`, `reset_persistent_windows=False` | Same | **Yes** |
| **§8.6** rounds | 5 warmup + **100** measured | Same | **Yes** |
| **§8.6** one prepare / retained domain | One prepared worker; no reset in measured region | Same | **Yes** |
| **§8.6** EP8/16 open probe | Probe before blaming code under test | N/A at EP2/4 | **N/A** |
| **§8.7** primary metric | Swimlane AIV gang span, fastest-rank mean (uniform) | L2 → `aicore_gang_span`; HOST → slot only (no AIV name) | **Yes** (L2) / **Partial** (HOST) |
| **§8.8** JSON | Schema with peer_bytes, persistent, ranks, BW, … | Emitted per point under `json/` | **Yes** (aligned) |
| **A1 completion** | “0–1024 KiB curve, DSV4 24.375 KiB, combine 48 KiB, uniform/zero/single-hot archived with env record” | Full on **EP2+EP4**; **EP8** L2 through **24960** (+ HOST mid-curve through 16 KiB); larger sizes still open | **Partial** |
| **A1 completion** | “every count pattern runs without touching any kernel” | Harness can; campaign did not archive all seven | **Partial** |
| **Official EP priority** | EP8 first (roadmap intro); EP16 in A3 | EP2+EP4 full; **EP8 partial** (see §9) | **Partial** |
| Plan 110 swimlane helper | Prefer `harness.swimlane.read_swimlane()` | Custom `name_map` + `chip_swimlane_records` parser | **Partial** |

### 2.2 What “done” means for a colleague

| Question | Answer |
|----------|--------|
| Can these numbers be used as a **methodology-correct A1-style baseline**? | **Yes**, for EP2/EP4 full curve; **Yes for EP8 mid-curve insight** (not full §8.3). |
| Is this the **immutable EP8 pre-change archive** Yunji wants before K1? | **Not yet** — EP8 missing 32 KiB…1 MiB + most §8.4 patterns; Fabric IPC workaround is local. |
| Does skipping §8.5 (`L>1`) invalidate A1? | **No for single-AIV pre-change baseline** — A1 is explicitly the current single-AIV kernel; multi-AIV is K2/K3/A3 and needs O2. |
| Did we run A2/A3? | **No** — DSV4 demo and recommended-`core_num` table are later milestones. |

### 2.3 Environment vs roadmap assumptions

| Item | Roadmap assumption | This run |
|------|--------------------|----------|
| Hardware | Real a2a3; EP8/EP16 device sets | a2a3; EP2 `0,1`, EP4 `0–3`, EP8 `0–7` |
| Scheduling | Often `task-submit` on shared boxes | Direct container; card 4 sometimes busy |
| PyPTO | Post-O1 L2 lowering | Tip included INT8 builtin (#2714) + local harness UB tile fix |
| Harness UB | Not called out in §8 | Required for ≥256 KiB C′; 16 KiB `pl.unroll` stage/consume tiles |

---

## 3. What we measured (methodology)

### 3.1 Rails under test

| Impl | Where `all_to_all_v` runs | Outer L3→L2 dispatches | Program shape |
|------|---------------------------|------------------------|---------------|
| `managed-l2` | CHIP orchestration → builtin AIV | **1** | Single `chip_pipeline`: stage → collective → consume |
| `managed-host` | HOST orchestration → same builtin AIV | **3** | Host: stage-all-ranks → fill counts → collective → consume-all-ranks |

Both rails render the **same** INT8 kernel template (`__builtin_all_to_all_v__int8` / `builtin.tensor.all_to_all_v__int8`).

### 3.2 Two clocks (do not mix)

| Clock | Field in JSON / tables | How measured | Includes |
|-------|------------------------|--------------|----------|
| **AIV gang span** | `fastest_p50_us` when `metric=aicore_gang_span` | Swimlane: die-mix-filtered per-task duration of the collective AIV | Collective kernel only |
| **Full-program timing slot** | `host_timing_slot_p50_us` (always); also `fastest_p50_us` when `metric=timing_slot` | Host `time.perf_counter` around one `runtime.run(...)` with swimlane **off** | Stage + counts + collective + consume + **outer L3→L2 dispatch(es)** + host/runtime overhead (`prepare` not included) |

Campaign used `--profile both`: a clean timing-slot session (5 warmup / 100 rounds) and a short swimlane session (8 rounds) for AIV capture.

### 3.3 Conditions

| Parameter | Value |
|-----------|-------|
| Platform | `a2a3` (real NPU) |
| Persistent workers | `True`, `reset_persistent_windows=False` |
| `requested_L` / `core_num` | 1 (multi-AIV / O2 not implemented) |
| Dtype | INT8 (RFC #2521 A1 canonical) |
| Devices EP2 / EP4 | `0,1` / `0,1,2,3` |
| Timing rounds / warmup / swimlane | 100 / 5 / 8 |
| Count patterns | uniform (full curve); zero & single-hot at 24960 B |
| Driver | `pypto-profiling/collectives/alltoallv_a1.py` → bundled harness `collectives/a1_alltoallv/harness/` |

### 3.4 Shape / C′ note

Multiples of canonical `C=4160` (and 0 B) share a fixed window with `MAX_RECV=CeilDiv(1MiB, C)`. Other sizes compile a **C′ = peer_bytes** instance (`max_recv=1`). Stage/consume copies rows in 16 KiB VEC tiles so C′ ≥ 256 KiB compiles under UB.

---

## 4. Full result tables

For L2 rows, **AIV/kernel p50** is the official AIV gang; **full-program slot** is always the host wall clock.  
For HOST rows, both columns are the **same timing slot** (no AIV gang captured).

### EP=2 · `managed-l2` · uniform

| peer_B | official metric | AIV/kernel p50 (µs) | full-program slot p50 (µs) | egress Gbps | duplex Gbps | row_width |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `aicore_gang_span` | 3.3 | 3867.8 | 0.00 | 0.00 | 4160 |
| 32 | `aicore_gang_span` | 4.1 | 1672.9 | 0.06 | 0.12 | 32 |
| 128 | `aicore_gang_span` | 4.0 | 1757.2 | 0.25 | 0.50 | 128 |
| 1024 | `aicore_gang_span` | 6.5 | 1690.8 | 1.24 | 2.48 | 1024 |
| 4096 | `aicore_gang_span` | 16.2 | 1680.4 | 1.52 | 3.05 | 4096 |
| 8192 | `aicore_gang_span` | 29.1 | 1734.4 | 2.26 | 4.51 | 8192 |
| 16384 | `aicore_gang_span` | 55.3 | 1807.7 | 2.20 | 4.40 | 16384 |
| 24576 | `aicore_gang_span` | 81.2 | 1758.8 | 2.42 | 4.84 | 24576 |
| 24960 | `aicore_gang_span` | 107.0 | 3065.2 | 0.74 | 1.48 | 4160 |
| 32768 | `aicore_gang_span` | 107.1 | 1955.0 | 2.45 | 4.90 | 32768 |
| 49152 | `aicore_gang_span` | 165.2 | 1955.1 | 2.14 | 4.27 | 49152 |
| 65536 | `aicore_gang_span` | 210.2 | 1975.2 | 2.49 | 4.97 | 65536 |
| 131072 | `aicore_gang_span` | 421.4 | 2352.9 | 2.33 | 4.65 | 131072 |
| 262144 | `aicore_gang_span` | 835.3 | 2804.6 | 2.38 | 4.76 | 262144 |
| 524288 | `aicore_gang_span` | 1665.8 | 3881.7 | 2.48 | 4.96 | 524288 |
| 1048576 | `aicore_gang_span` | 3402.5 | 6584.7 | 2.45 | 4.90 | 1048576 |

### EP=2 · `managed-host` · uniform

| peer_B | official metric | AIV/kernel p50 (µs) | full-program slot p50 (µs) | egress Gbps | duplex Gbps | row_width |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `timing_slot` | 5511.3 | 5511.3 | 0.00 | 0.00 | 4160 |
| 32 | `timing_slot` | 3605.3 | 3605.3 | 0.00 | 0.00 | 32 |
| 128 | `timing_slot` | 3536.8 | 3536.8 | 0.00 | 0.00 | 128 |
| 1024 | `timing_slot` | 3707.3 | 3707.3 | 0.00 | 0.00 | 1024 |
| 4096 | `timing_slot` | 3637.5 | 3637.5 | 0.01 | 0.01 | 4096 |
| 8192 | `timing_slot` | 3547.9 | 3547.9 | 0.01 | 0.03 | 8192 |
| 16384 | `timing_slot` | 3521.2 | 3521.2 | 0.03 | 0.05 | 16384 |
| 24576 | `timing_slot` | 3637.6 | 3637.6 | 0.04 | 0.08 | 24576 |
| 24960 | `timing_slot` | 5732.1 | 5732.1 | 0.03 | 0.05 | 4160 |
| 32768 | `timing_slot` | 3570.7 | 3570.7 | 0.05 | 0.09 | 32768 |
| 49152 | `timing_slot` | 3815.4 | 3815.4 | 0.08 | 0.15 | 49152 |
| 65536 | `timing_slot` | 3812.6 | 3812.6 | 0.10 | 0.21 | 65536 |
| 131072 | `timing_slot` | 4362.5 | 4362.5 | 0.18 | 0.36 | 131072 |
| 262144 | `timing_slot` | 4774.4 | 4774.4 | 0.33 | 0.67 | 262144 |
| 524288 | `timing_slot` | 6246.6 | 6246.6 | 0.55 | 1.11 | 524288 |
| 1048576 | `timing_slot` | 7879.3 | 7879.3 | 0.85 | 1.71 | 1048576 |

### EP=4 · `managed-l2` · uniform

| peer_B | official metric | AIV/kernel p50 (µs) | full-program slot p50 (µs) | egress Gbps | duplex Gbps | row_width |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `aicore_gang_span` | 78.5 | 5944.9 | 0.00 | 0.00 | 4160 |
| 32 | `aicore_gang_span` | 9.5 | 2394.2 | 0.08 | 0.16 | 32 |
| 128 | `aicore_gang_span` | 9.4 | 2170.2 | 0.11 | 0.22 | 128 |
| 1024 | `aicore_gang_span` | 15.3 | 2193.9 | 1.59 | 3.19 | 1024 |
| 4096 | `aicore_gang_span` | 40.7 | 2303.2 | 2.40 | 4.81 | 4096 |
| 8192 | `aicore_gang_span` | 72.1 | 2217.7 | 2.28 | 4.56 | 8192 |
| 16384 | `aicore_gang_span` | 137.4 | 2508.2 | 2.86 | 5.72 | 16384 |
| 24576 | `aicore_gang_span` | 204.9 | 2553.3 | 2.89 | 5.78 | 24576 |
| 24960 | `aicore_gang_span` | 268.2 | 6648.8 | 1.38 | 2.77 | 4160 |
| 32768 | `aicore_gang_span` | 268.9 | 2619.0 | 2.93 | 5.85 | 32768 |
| 49152 | `aicore_gang_span` | 410.2 | 2703.5 | 2.18 | 4.36 | 49152 |
| 65536 | `aicore_gang_span` | 534.7 | 2907.6 | 2.93 | 5.87 | 65536 |
| 131072 | `aicore_gang_span` | 1067.3 | 3784.2 | 2.84 | 5.68 | 131072 |
| 262144 | `aicore_gang_span` | 2402.3 | 5398.7 | 2.64 | 5.28 | 262144 |
| 524288 | `aicore_gang_span` | 4231.0 | 7967.1 | 2.91 | 5.81 | 524288 |
| 1048576 | `aicore_gang_span` | 8564.4 | 13740.9 | 2.82 | 5.63 | 1048576 |

### EP=4 · `managed-host` · uniform

| peer_B | official metric | AIV/kernel p50 (µs) | full-program slot p50 (µs) | egress Gbps | duplex Gbps | row_width |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `timing_slot` | 7400.3 | 7400.3 | 0.00 | 0.00 | 4160 |
| 32 | `timing_slot` | 4002.7 | 4002.7 | 0.00 | 0.00 | 32 |
| 128 | `timing_slot` | 4067.6 | 4067.6 | 0.00 | 0.00 | 128 |
| 1024 | `timing_slot` | 3931.0 | 3931.0 | 0.00 | 0.01 | 1024 |
| 4096 | `timing_slot` | 3909.1 | 3909.1 | 0.02 | 0.04 | 4096 |
| 8192 | `timing_slot` | 3986.9 | 3986.9 | 0.04 | 0.08 | 8192 |
| 16384 | `timing_slot` | 4184.9 | 4184.9 | 0.07 | 0.14 | 16384 |
| 24576 | `timing_slot` | 4294.9 | 4294.9 | 0.10 | 0.21 | 24576 |
| 24960 | `timing_slot` | 7834.6 | 7834.6 | 0.07 | 0.13 | 4160 |
| 32768 | `timing_slot` | 4240.1 | 4240.1 | 0.14 | 0.28 | 32768 |
| 49152 | `timing_slot` | 4092.1 | 4092.1 | 0.22 | 0.45 | 49152 |
| 65536 | `timing_slot` | 4473.5 | 4473.5 | 0.27 | 0.55 | 65536 |
| 131072 | `timing_slot` | 5287.5 | 5287.5 | 0.45 | 0.91 | 131072 |
| 262144 | `timing_slot` | 7096.7 | 7096.7 | 0.77 | 1.54 | 262144 |
| 524288 | `timing_slot` | 9759.8 | 9759.8 | 1.15 | 2.30 | 524288 |
| 1048576 | `timing_slot` | 15298.0 | 15298.0 | 1.54 | 3.08 | 1048576 |

### Extra count patterns @ peer_bytes = 24960

| tag | EP | impl | pattern | metric | AIV/kernel p50 (µs) | full slot p50 (µs) | egress Gbps |
|---|---:|---|---|---|---:|---:|---:|
| `p2_managed-l2_zero_24960` | 2 | `managed-l2` | `zero` | `aicore_gang_span` | 3.1 | 3836.5 | 0.000 |
| `p2_managed-l2_single-hot_24960` | 2 | `managed-l2` | `single-hot` | `aicore_gang_span` | 71.5 | 3643.9 | 2.800 |
| `p2_managed-host_zero_24960` | 2 | `managed-host` | `zero` | `timing_slot` | 5032.7 | 5032.7 | 0.000 |
| `p2_managed-host_single-hot_24960` | 2 | `managed-host` | `single-hot` | `timing_slot` | 4861.5 | 4861.5 | 0.032 |
| `p4_managed-l2_zero_24960` | 4 | `managed-l2` | `zero` | `aicore_gang_span` | 7.0 | 6131.0 | 0.000 |
| `p4_managed-l2_single-hot_24960` | 4 | `managed-l2` | `single-hot` | `aicore_gang_span` | 79.5 | 6227.3 | 2.339 |
| `p4_managed-host_zero_24960` | 4 | `managed-host` | `zero` | `timing_slot` | 7505.6 | 7505.6 | 0.000 |
| `p4_managed-host_single-hot_24960` | 4 | `managed-host` | `single-hot` | `timing_slot` | 7869.6 | 7869.6 | 0.021 |

---

## 5. Analytic comparisons

### EP=2 — fair full-program comparison (slot vs slot) + L2 AIV

| peer_B | L2 AIV p50 (µs) | L2 full slot (µs) | HOST full slot (µs) | HOST/L2 slot | L2 slot / L2 AIV |
|---:|---:|---:|---:|---:|---:|
| 32 | 4.1 | 1672.9 | 3605.3 | 2.16× | 403.6× |
| 128 | 4.0 | 1757.2 | 3536.8 | 2.01× | 434.4× |
| 1024 | 6.5 | 1690.8 | 3707.3 | 2.19× | 258.9× |
| 4096 | 16.2 | 1680.4 | 3637.5 | 2.16× | 103.7× |
| 8192 | 29.1 | 1734.4 | 3547.9 | 2.05× | 59.5× |
| 16384 | 55.3 | 1807.7 | 3521.2 | 1.95× | 32.7× |
| 24576 | 81.2 | 1758.8 | 3637.6 | 2.07× | 21.7× |
| 24960 | 107.0 | 3065.2 | 5732.1 | 1.87× | 28.7× |
| 32768 | 107.1 | 1955.0 | 3570.7 | 1.83× | 18.3× |
| 49152 | 165.2 | 1955.1 | 3815.4 | 1.95× | 11.8× |
| 65536 | 210.2 | 1975.2 | 3812.6 | 1.93× | 9.4× |
| 131072 | 421.4 | 2352.9 | 4362.5 | 1.85× | 5.6× |
| 262144 | 835.3 | 2804.6 | 4774.4 | 1.70× | 3.4× |
| 524288 | 1665.8 | 3881.7 | 6246.6 | 1.61× | 2.3× |
| 1048576 | 3402.5 | 6584.7 | 7879.3 | 1.20× | 1.9× |

### EP=4 — fair full-program comparison (slot vs slot) + L2 AIV

| peer_B | L2 AIV p50 (µs) | L2 full slot (µs) | HOST full slot (µs) | HOST/L2 slot | L2 slot / L2 AIV |
|---:|---:|---:|---:|---:|---:|
| 32 | 9.5 | 2394.2 | 4002.7 | 1.67× | 251.2× |
| 128 | 9.4 | 2170.2 | 4067.6 | 1.87× | 231.2× |
| 1024 | 15.3 | 2193.9 | 3931.0 | 1.79× | 143.1× |
| 4096 | 40.7 | 2303.2 | 3909.1 | 1.70× | 56.6× |
| 8192 | 72.1 | 2217.7 | 3986.9 | 1.80× | 30.7× |
| 16384 | 137.4 | 2508.2 | 4184.9 | 1.67× | 18.3× |
| 24576 | 204.9 | 2553.3 | 4294.9 | 1.68× | 12.5× |
| 24960 | 268.2 | 6648.8 | 7834.6 | 1.18× | 24.8× |
| 32768 | 268.9 | 2619.0 | 4240.1 | 1.62× | 9.7× |
| 49152 | 410.2 | 2703.5 | 4092.1 | 1.51× | 6.6× |
| 65536 | 534.7 | 2907.6 | 4473.5 | 1.54× | 5.4× |
| 131072 | 1067.3 | 3784.2 | 5287.5 | 1.40× | 3.5× |
| 262144 | 2402.3 | 5398.7 | 7096.7 | 1.31× | 2.2× |
| 524288 | 4231.0 | 7967.1 | 9759.8 | 1.23× | 1.9× |
| 1048576 | 8564.4 | 13740.9 | 15298.0 | 1.11× | 1.6× |

### 5.1 Reading guide

- **L2 AIV p50** — “how long does the collective kernel take?”
- **L2 full slot** — “how long does one full L2 benchmark invocation take end-to-end?”
- **HOST full slot** — “how long does one full HOST benchmark invocation take end-to-end?”
- **HOST/L2 slot** — fair rail comparison for this harness.
- **L2 slot / L2 AIV** — how much non-collective work dominates the L2 full program.

### 5.2 Selected headlines (EP2)

| Question | Answer from data |
|----------|------------------|
| Is the L2 **kernel** fast? | Yes. e.g. 8 KiB ≈ 29 µs; 64 KiB ≈ 210 µs; 1 MiB ≈ 3.4 ms; egress ~2.3–2.5 Gbps plateau. |
| Is the L2 **full program** as fast as the kernel? | No. At 8 KiB, slot ≈ 1.7 ms vs AIV 29 µs (~60×). At 1 MiB, slot ≈ 6.6 ms vs AIV 3.4 ms (~1.9×). |
| Is L2 **full program** much faster than HOST? | Moderately. ~1.7–2.0× at mid sizes; ~1.2× at 1 MiB. |
| Why does HOST look 40–100× slower in some slides? | Usually L2 **AIV** compared to HOST **slot** — different clocks. |

### 5.3 EP4 vs EP2 (L2 AIV)

| peer_B | EP2 AIV (µs) | EP4 AIV (µs) | EP4/EP2 |
|---:|---:|---:|---:|
| 8192 | 29.1 | 72.1 | 2.5× |
| 24960 | 107.0 | 268.2 | 2.5× |
| 65536 | 210.2 | 534.7 | 2.5× |
| 262144 | 835.3 | 2402.3 | 2.9× |
| 1048576 | 3402.5 | 8564.4 | 2.5× |

AIV time scales roughly with peer count (more remote traffic), while egress Gbps stays in a similar plateau band.

---

## 6. Caveats and limitations

1. **HOST AIV not captured** — swimlane name_map does not yet attribute the HOST-rail builtin collective; HOST official metric remains timing_slot only (§2 checklist: Partial vs §8.7).
2. **Harness overhead vs §8.1** — roadmap wants stage/consume out of the timing slot; our slot includes them. Official **AIV** column still excludes them. Prefer AIV for kernel claims; prefer slot-vs-slot for HOST/L2 end-to-end.
3. **Box limits** — 8× 910B2, no `task-submit`; EP8 L2 through DSV4 24960 archived (§9), **EP16 not run**; full EP8 §8.3 still open. `core_num>1` rejected (O2).
4. **Pattern coverage** — only uniform / zero / single-hot archived on EP2/EP4; EP8 uniform through 24960 (+ force-IPC zero); self-only, mixed, asymmetric, random still open.
5. **Flakes** — EP2/EP4: 3× `release_domain` + 1× HOST mismatch, cleared on retry (72/72). EP8: Fabric “cross-server” → poison / `release_domain`; mitigated with **`SIMPLER_COMM_FORCE_IPC=1`**, widened Fabric→IPC fallback, retries + EP4 heal.
6. **24960 / 0 B slots** can look inflated vs neighbors (setup / pattern effects); prefer L2 AIV column for kernel trends. EP8 zero slot (~10.7 ms) and EP8 **24960** slot (~11.7 ms) are high vs mid-curve (~3.2–3.7 ms).
7. **PR #2690** (InCore composite staging tile cap) does **not** explain these numbers; this campaign is managed HOST/L2 builtins. The UB fix was harness stage/consume tiling only.
8. **EP8 rank spread** — L2 AIV fastest-rank is usable, but per-rank builtin durations often show ~3–5 ms spread (sync wait inside the named AIV task), not a parser mix-up with stage/consume.

---

## 7. Bottom line for planning

- **Roadmap:** EP2/EP4 = methodology-complete A1-style baseline; EP8 = **usable through DSV4 24960 + L2-vs-HOST insight**, not yet the frozen full-curve denominator. Details in **§2** / **§9**.
- Use **L2 AIV gang** as the A1 “collective performance” number (roadmap official column).
- Use **full-program slot** when comparing HOST vs L2 end-to-end on this harness (~1.1–2× on EP2/EP4; **~1.6–2.0× on EP8** mid-curve).
- Treat large-payload success (≥256 KiB) as a **harness correctness/capacity** win, not a transport redesign.
- **Next gaps:** EP8 32 KiB…1 MiB + HOST for new sizes; remaining §8.4 patterns; HOST AIV attribution; upstream Fabric/IPC robustness; optionally align timing slot with §8.1.

---

## 8. Artifact index

| Path | Contents |
|------|----------|
| `ANALYTIC_REPORT_FOR_COLLEAGUES.md` (this file) | Analysis + roadmap accordance + EP8 §9 |
| `../../collectives/a1_alltoallv/` | Harness snapshot, FORCE_IPC patch, Fabric MFE |
| `../../collectives/alltoallv_a1.py` | Campaign driver |
| `REPORT.md` | Shorter numeric summary (EP2/EP4) |
| `summary.json` | Machine-readable row per point (EP2/EP4) |
| `campaign_meta.json` | Conditions, timestamps |
| `sweep.log` | Full driver log |
| `json/*.json` | Per-point harness JSON |
| `builds/*` | Compile outputs |
| `../issue-2521-a1-alltoallv-ep8-2026-09-10/` | EP8 L2 resume + debug |
| `../issue-2521-a1-alltoallv-ep8-l2host-2026-09-10/` | EP8 L2 vs HOST + `zero_force_ipc/` |
| `../issue-2521-a1-alltoallv-ep8-forceipc-2026-09-10/` | FORCE_IPC continuation (L2 `24576`, `24960` OK) |

---

## 9. EP8 addendum (2026-09-10)

Devices `0–7`, `L=1`, INT8, §8.6 persistent, `--profile both` (100 slot + 8 swimlane). Large §8.3 sizes stopped after Fabric flakes; focus shifted to **known-good mid-curve**, **L2 vs HOST**, then a FORCE_IPC probe of the next §8.3 points.

### 9.1 EP8 L2 uniform (official AIV = fastest-rank p50)

| peer_bytes | AIV p50 (µs) | slot p50 (µs) | egress (Gbps) | rank spread (µs) | Source |
|-----------:|-------------:|--------------:|--------------:|-----------------:|--------|
| 0 | 14.7 | 10720 | 0 | 3103 | `zero_force_ipc` (`SIMPLER_COMM_FORCE_IPC=1`) |
| 32 | 20.3 | 3174 | 0.089 | 4065 | `ep8-l2host` |
| 128 | 20.1 | 3175 | 0.051 | 3696 | `ep8-l2host` |
| 1024 | 33.4 | 3392 | 0.819 | 4476 | `ep8-l2host` |
| 4096 | 88.4 | 3345 | 2.59 | 4128 | `ep8-l2host` |
| 8192 | 167.1 | 3504 | 2.77 | 4063 | `ep8-l2host` |
| 16384 | 313.1 | 3716 | 2.93 | 4770 | earlier EP8 L2 resume |
| 24576 | 456.1 | 3692 | 3.01 | 4934 | `ep8-forceipc` (`SIMPLER_COMM_FORCE_IPC=1`) |
| 24960 | 589.2 | 11672 | 2.26 | 3589 | `ep8-forceipc` try3 (hardened IPC patch) |

Egress ~**2.3–3.0 Gbps** by 4–24 KiB (24960 egress lower — canonical `C=4160` / larger window). AIV scales ~linearly 1 KiB→24 KiB. Prefer AIV over slot for kernel claims; zero / 24960 slots are inflated.

### 9.2 EP8 L2 vs HOST (fair slot_p50)

| peer_bytes | L2 slot (µs) | HOST slot (µs) | L2/HOST | L2 faster |
|-----------:|-------------:|---------------:|--------:|----------:|
| 32 | 3174 | 6341 | 0.50 | ~2.0× |
| 128 | 3175 | 5846 | 0.54 | ~1.8× |
| 1024 | 3392 | 5548 | 0.61 | ~1.6× |
| 4096 | 3345 | 5986 | 0.56 | ~1.8× |
| 8192 | 3504 | 6091 | 0.58 | ~1.7× |
| 16384 | 3716† | 5833 | 0.64 | ~1.6× |

† L2 16384 from earlier resume (same-day `ep8-l2host` L2 16384 failed 4/4 Fabric); HOST 16384 from `ep8-l2host`. HOST metric is timing_slot only (no AIV name). No HOST archive yet for 24576 / 24960.

**Verdict:** on EP8 mid-curve, **L2 remains ~1.6–2.0× faster than HOST** end-to-end — consistent with EP2/EP4.

### 9.3 EP8 vs EP4 AIV p50 (same peer_bytes)

| peer_bytes | EP8 AIV p50 | EP4 AIV p50 | EP8/EP4 |
|-----------:|------------:|------------:|--------:|
| 32 | 20.3 | 9.5 | ~2.1× |
| 128 | 20.1 | 9.4 | ~2.1× |
| 1024 | 33.4 | 15.3 | ~2.2× |
| 4096 | 88.4 | 40.7 | ~2.2× |
| 8192 | 167.1 | 72.1 | ~2.3× |
| 16384 | 313.1 | 137.4 | ~2.3× |
| 24576 | 456.1 | 204.9 | ~2.2× |
| 24960 | 589.2 | 268.2 | ~2.2× |

EP8 gang AIV is ~**2.1–2.3×** EP4 at matched sizes (more ranks / sync), while egress stays in a similar Gbps plateau.

### 9.4 Still missing for full EP8 A1

- §8.3: 32K, 48K, 64K…**1 MiB**
- EP8 HOST for 0 / 24576 / 24960+; EP8 zero/single-hot @ 24960
- Remaining §8.4 patterns; EP16
- Upstream Fabric/IPC fix (local FORCE_IPC + widened fallback)

### 9.5 FORCE_IPC probe notes (2026-09-10)

| Tag | Result | Notes |
|-----|--------|-------|
| `p8_managed-l2_uniform_24576` | **OK** | C′ instance; AIV 456 µs, slot 3692 µs, egress 3.01 Gbps |
| `p8_managed-l2_uniform_24960` try1–2 | **FAIL** | Fabric `0x702001d` → `507018` → `release_domain` / SIGSEGV mid timing-slot (~run 41–77) under FORCE_IPC alone |
| `p8_managed-l2_uniform_24960` try3 | **OK** | After patch harden: cache FORCE_IPC + **any** Fabric export fail → IPC fallback; rebuild `libhost_runtime.so`; EP4 heal. AIV **589.2 µs**, slot **11672 µs**, egress **2.26 Gbps**, EP8/EP4 ≈ **2.2×** |

Artifacts: `../issue-2521-a1-alltoallv-ep8-forceipc-2026-09-10/json/p8_managed-l2_uniform_{24576,24960}.json`; fail logs `attempt_logs/…24960_try{1,2}.log`. Patch: `collectives/a1_alltoallv/patches/simpler_comm_force_ipc.patch`.

---

## 10. How to reproduce (code + commands)

This section is the colleague entry point: **what ran**, **where it lives**, and **exact commands**. Numbers above are meaningless without this.

### 10.1 What code executes

Colleague-facing copies live under **`pypto-profiling/collectives/a1_alltoallv/`** (see that README). Original edit trees may still be under `/opt/pypto` on the measurement box.

| Layer | Path in `pypto-profiling` | Role |
|-------|---------------------------|------|
| Campaign driver | `collectives/alltoallv_a1.py` | Loops EP × impl × payloads; retries / heal / `A2AV_FORCE_IPC` |
| Harness (measured program) | `collectives/a1_alltoallv/harness/all_to_all_v_benchmark.py` | Builds managed-host / managed-l2 graphs, `prepare(persistent=True)`, timing-slot + swimlane sessions, JSON |
| INT8 builtin (codegen) | pypto tree — `pld.tensor.all_to_all_v` → `__builtin_all_to_all_v__int8` ([PR #2714](https://github.com/hw-native-sys/pypto/pull/2714)) | Collective kernel under test |
| Swimlane metric | Same harness (`collect_swimlane_spans_us` / name_map + `chip_swimlane_records`) | Official L2 AIV = fastest-rank gang span |
| Comm windows (EP8 flake) | Apply `collectives/a1_alltoallv/patches/simpler_comm_force_ipc.patch` onto simpler `comm_hccl.cpp` | Fabric V2 vs VMM IPC; `SIMPLER_COMM_FORCE_IPC` |
| Fabric MFE (simpler-only) | `collectives/a1_alltoallv/mfe/repro_domain_churn.py` | Empty `allocate_domain` churn — **no AllToAllV**; flake is intermittent (not a guaranteed repro) |
| Simpler bug | https://github.com/hw-native-sys/simpler/issues/2192 | Fabric “cross-server” issue |

**Harness program shape (what is timed):**

- `managed-l2`: one `chip_pipeline` — stage → `pld.tensor.all_to_all_v` → consume (`core_num=1`, **1** outer L3→L2 dispatch).
- `managed-host`: host orch — stage-all → fill counts → collective → consume-all (**3** outer L3→L2 dispatches).
- `--profile both`: (1) timing-slot session, swimlane off, 5 warmup + 100 measured; (2) swimlane session, 8 AIV rounds. Official L2 kernel column = swimlane AIV; fair HOST vs L2 = **slot vs slot**.

**Note:** this bundle is checked into **`pypto-profiling`** for access; it is **not** assumed to be on `hw-native-sys/pypto` / `simpler` `main`. INT8 builtin is PR #2714. Rebuild host runtime after applying the FORCE_IPC patch.

### 10.2 Versions we used

| Component | Value |
|-----------|--------|
| Box | 8× Ascend 910B2, Linux aarch64 |
| CANN | 9.0.0 (`/usr/local/Ascend/cann-9.0.0`) |
| `LD_PRELOAD` | `/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so` |
| pypto checkout | branch `feat/alltoallv-int8-builtin`, tip `fcd92b971f049f8498b4afd9aed17f5ef2372ec7` |
| simpler (vendored `runtime/`) | tip `39ce891dbb3f665e72e99b4a1387b47012fb18bd` + local `SIMPLER_COMM_FORCE_IPC` in `comm_hccl.cpp` |
| Platform | `a2a3`, runtime `tensormap_and_ringbuffer` |

### 10.3 Re-run EP2/EP4 full curve (ubfix archive)

```bash
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
cd /opt/pypto-profiling
A2AV_OUT=/opt/pypto-profiling/reports/issue-2521-a1-alltoallv-ubfix-REPRO \
A2AV_EPS=2,4 A2AV_IMPLS=managed-l2,managed-host \
A2AV_ROUNDS=100 A2AV_WARMUP=5 A2AV_SWIMLANE_ROUNDS=8 \
python3 -u collectives/alltoallv_a1.py
```

Artifacts to compare: `…/issue-2521-a1-alltoallv-ubfix-2026-09-09/{summary.json,json/,sweep.log}`.

### 10.4 Re-run EP8 mid-curve L2 + HOST (this insight)

```bash
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
cd /opt/pypto-profiling
# Optional but recommended on this box:
export A2AV_FORCE_IPC=1          # → SIMPLER_COMM_FORCE_IPC=1 in child
# Known-good payloads only (full §8.3 still flaky without IPC / retries):
A2AV_OUT=/opt/pypto-profiling/reports/issue-2521-a1-alltoallv-ep8-l2host-REPRO \
A2AV_EPS=8 A2AV_IMPLS=managed-l2,managed-host \
A2AV_PAYLOADS=32,128,1024,4096,8192,16384 A2AV_EXTRA_PATTERNS= \
A2AV_RETRIES=3 A2AV_COOLDOWN_S=25 A2AV_HEAL_EP=4 \
A2AV_ROUNDS=100 A2AV_WARMUP=5 A2AV_SWIMLANE_ROUNDS=8 \
python3 -u collectives/alltoallv_a1.py
```

Single-point harness example (same code path as one campaign cell):

```bash
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
# export SIMPLER_COMM_FORCE_IPC=1   # if Fabric flakes
HARNESS=/opt/pypto-profiling/collectives/a1_alltoallv/harness/all_to_all_v_benchmark.py
python3 "$HARNESS" \
  --ep 8 --peer-bytes 4096 --impl managed-l2 --devices 0,1,2,3,4,5,6,7 \
  --rounds 100 --warmup 5 --profile both --swimlane-rounds 8 --platform a2a3 \
  --output-json /tmp/p8_l2_4096.json --output-dir /tmp/p8_l2_4096_build
```

### 10.5 Re-run EP8 `peer_bytes=0` (needs FORCE_IPC on this box)

```bash
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
export SIMPLER_COMM_FORCE_IPC=1   # requires patch applied + host runtime rebuilt
HARNESS=/opt/pypto-profiling/collectives/a1_alltoallv/harness/all_to_all_v_benchmark.py
python3 "$HARNESS" \
  --ep 8 --peer-bytes 0 --impl managed-l2 --devices 0,1,2,3,4,5,6,7 \
  --rounds 100 --warmup 5 --profile both --swimlane-rounds 8 --platform a2a3 \
  --output-json /tmp/p8_l2_0.json --output-dir /tmp/p8_l2_0_build
```

Reference OK artifact: `…/ep8-l2host-2026-09-10/zero_force_ipc/`.

### 10.6 Attempt Fabric flake in simpler alone (no AllToAllV)

Intermittent; this MFE did **not** always hit the failure in short runs. Prefer A1 harness logs / issue #2192 for evidence.

```bash
cd /opt/pypto/runtime    # simpler checkout (needs its Python bindings)
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
unset SIMPLER_COMM_FORCE_IPC
python3 /opt/pypto-profiling/collectives/a1_alltoallv/mfe/repro_domain_churn.py \
  --iters 30 --devices 0-7 --dual-alloc
```

Issue: https://github.com/hw-native-sys/simpler/issues/2192 · `collectives/a1_alltoallv/mfe/README.md`.

### 10.7 How to read a result JSON

Per-point files under `json/*.json` / harness `--output-json`:

- L2 official kernel: `metric=aicore_gang_span`, use `fastest_p50_us` / `fastest_mean_us` (fastest-rank AIV).
- Fair HOST vs L2: compare `host_timing_slot_p50_us` (both impls). Do **not** compare HOST “kernel” to L2 AIV.
- `rank_spread_us`: EP8 often ~3–5 ms — sync skew inside the named builtin, not stage/consume mix-up.
- `egress_gbps`: from fastest-rank AIV and send bytes.

Campaign rollup: each report dir’s `summary.json` + `campaign_meta.json` + `sweep.log`.

*Updated 2026-09-10: EP8 L2 **24576** + **24960** OK (hardened FORCE_IPC); §10 repro; scripts under `collectives/a1_alltoallv/`.*
