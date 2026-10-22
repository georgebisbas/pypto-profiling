# RFC #2521 O2/A3 gate + #3051 payload re-run on `main` `af75165dd` (2026-10-22)

**Status: PARTIAL.** O2 §11.2 (Part A) held at its gate — the HOST rail returns no AIV span.
A3 Part B pass 1 ran on the L2 rail: 30/30 cells + 8 anchors OK (grants g3–g6); 9 matrix points wait
in g1/g2 behind the HOST-rail fix. The #3051 payload sweep was re-run: 9/11 cells OK, 2 compile
failures above 4 MiB/peer.

**Tree:** pypto `main` @ `af75165dd411596ddb9cd0d3090e68403913add2`, runtime submodule
`d8f997fabd1b6988b69a94feef61ab669dda0eea` (reinstalled from the submodule — the image's prebuilt
`simpler_setup` was from an earlier pin) · **Box:** hng-atlas02, 8× Ascend 910B2 (a2a3), CANN 9.0.0,
full HCCS mesh (`driver/repro3051-npu-topo.txt`) · **Container:** `pypto-2521-sweep`
(`pypto3-hw-native-sys:cann9`) · **Harness:** in-tree
`tests/st/distributed/collectives/all_to_all_v_benchmark.py` · `SIMPLER_COMM_FORCE_IPC`: unpatched
(the runtime has no such switch), so **not** the A1 frozen protocol.

Full analysis lives in the notes hand-off
(`pypto-3.0-notes/pr_plans/2521-sweep-npu-handoff.md`, commits `0257725`…`7be7d44`); this directory
is the evidence.

## Contents

| Path | What |
| --- | --- |
| `json/gate_*.json` | gate grant: 2 OK `managed-l2` L=1 @ 24960 cells (30 timing-slot + 8 swimlane rounds) |
| `json/g{3,4,5,6}_*.json` | Part B pass 1: 30 `managed-l2` uniform cells + 8 `anchor-*` cells, same protocol as the gate |
| `json/repro3051_*.json` | 9 OK `managed-l2` `core_num=16` cells, 8 KiB–4 MiB/peer (5 warmup + 10 swimlane rounds) |
| `json/codegen/` | Phase 2 compile-only stubs (`metric=compile-only`), L ∈ {2,4,7,10,15} × {24960, 16384} |
| `summary.json` | per cell: #3051-definition trend µs / GB/s, best fixed-rank mean, min-rank span range, slowest-rank median |
| `campaign_meta.json` | tree, runtime, queue task ids per grant, protocol, failures |
| `driver/` | `run-2521-sweep.sh` (notes `08eb981`), the re-run runner, Phase 2 script, `cells.tsv` (one row per attempt), per-grant `manifest-*.txt`, `partB-taskids.txt`, topology |
| `logs/` | per-cell console logs — **on disk only, gitignored** |

## Queue grants

| Grant | Task id | Cells |
| --- | --- | --- |
| Phase 2 (no card) | `task_20261022_053841_99463031931` | 10/10 compiled |
| gate | `task_20261022_053929_103696911215` | 2 OK, 1 DEGRADED, stopped by hand |
| g3 / g4 / g5 / g6 | `task_20261022_062722_33869877661` / `…062723_33871785507` / `…062723_338733616086` / `…062723_338761518436` | 10 / 10 / 10 / 8 OK (incl. 2 anchors each), all `exit=0` |
| #3051 64 KiB–2 MiB | `task_20261022_055849_198169625427` | 6/6 OK |
| #3051 8, 16 KiB | `task_20261022_060917_24833563352` | 2/2 OK |
| 4–16 MiB | `task_20261022_061401_271193230738` | 1/3 OK (exit 2) |

## Results — Part B pass 1 (trend GB/s; one pass, not a publishable `B*`)

| L → B | 16 KiB | 24960 | 48 KiB | 256 KiB | 1 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 → 1 | 4.83 | 6.42 | 9.10 | 15.14 | — |
| 2 → 2 | **5.50** | — | 12.22 | 23.80 | 28.47 |
| 4 → 4 | 5.15 | — | **13.08** | 34.58 | 47.81 |
| 7 → 7 | 4.48 | — | 12.40 | 35.35 | 47.23 |
| 8 → 8 | 4.34 | — | 11.94 | **43.39** | — |
| 10 → 8 | 4.41 | — | 12.10 | 43.06 | 70.12 |
| 15 → 8 | 4.38 | 6.43 | 12.17 | 43.05 | **70.35** |
| 16 → 16 | 3.19 | — | 9.47 | 37.44 | — |

"—" = g1/g2 cells, not run (they pair with `managed-host` cells). Provisional `B*`: 2 / 4 / 8 / 8 at
16 KiB / 48 KiB / 256 KiB / 1 MiB. B=16 trails B=8 by 14–27 % where both exist. L ∈ {8, 10, 15}
(all B=8) agree to ≤±1 %. Anchors (`managed-l2` L=1 @ 24960, 9 runs over ~2 h): 30.76–31.42 µs,
−1.2 %…+1.0 % around 31.12 µs.

## Results — #3051 re-run (trend = mean over rounds of the per-round min-rank span)

| Peer KiB | #3051 trend µs (32 KiB stg) | Ours trend µs | Ours GB/s | Δ BW vs #3051 | Ours best fixed-rank µs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 49.4 | 43.5 | 1.507 | +13.6 % | 43.5 |
| 16 | 48.5 | 41.0 | 3.195 | +18.2 % | 41.0 |
| 64 | 50.1 | 45.0 | 11.66 | +11.3 % | 45.0 |
| 128 | 53.8 | 49.7 | 21.10 | +8.3 % | 49.7 |
| 256 | 58.1 | 56.8 | 36.91 | +2.3 % | 56.8 |
| 512 | 72.3 | 74.3 | 56.42 | −2.8 % | 74.3 |
| 1024 | 99.8 | 113.0 | 74.24 | −11.7 % | 113.0 |
| 2048 | 153.0 | 192.6 | 87.09 | −20.6 % | 260.6 |
| 4096 | — | 356.2 | 94.19 | — | 512.1 |

Not like-for-like with #3051: span definition (harness = longest single block; #3051 =
`max(block end) − min(block start)`), receive layout (one `peer_bytes` row vs `[rows, 4096]`), tree
(#3051 ran experimental patches on `fe651f29` / `6e383fc5`), devices. At 4 MiB the HCCS inbound rate
is ⅞ × 94.19 = 82.4 GB/s/rank — 21 % of a 7 × 56 = 392 GB/s ceiling (42 % if 56 GB/s is the
two-way sum). Slowest-rank span is ~4.5–5 ms/round at every payload (launch skew), so
whole-collective utilisation is under ~1 %.

## Failures (log excerpts)

**Gate, `managed-host` L=1 @ 24960** — `rc=0`, `metric=end_to_end_latency`, JSON discarded by the
driver. The builtin collective's dispatch is never swimlane-captured: per round, rank dirs show
`endpoint_dispatch_id` 81/82/84 (stage, fill, consume) and no 83. Generated `host_orch.py`
submits it with a fresh `CallConfig()` carrying only `aicpu_thread_num`
(`src/codegen/distributed/distributed_ops_codegen.cpp:238-241`), dropping the DFX fields:

```text
No L2 program recorded for rank0/d6; converting with anonymous task labels. ...
```

**#3051 re-run, 8192 and 16384 KiB/peer** — compile failure before any round:

```text
pypto.backend.pto_backend.PartialCodegenError: 1 function(s) failed to compile:
  consume_step  | ptoas compilation timed out after 60s
```

The 60 s limit is hard-coded (`python/pypto/backend/pto_backend.py:244`).
