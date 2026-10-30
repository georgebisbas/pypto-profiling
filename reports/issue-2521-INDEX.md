# RFC #2521 campaign index

One row per campaign, newest first. **Report status** is the lifecycle state written at the top of
each report:

- **current** — the latest evidence for its question; safe to quote with its caveats.
- **superseded** — a newer campaign replaces its conclusion; kept for provenance.
- **archived** — historical record; the code or upstream state it measured has moved on.
- **reference** — a note, runbook, or methodology page rather than a campaign.

Upstream SHAs and PR states live in [`upstream-status.md`](upstream-status.md).

## Current

| Date | Campaign | Establishes | Provenance |
|---|---|---|---|
| 2026-10-22 | [o2-a3](issue-2521-o2-a3-2026-10-22/README.md) | O2/A3 gate held (HOST rail emits no AIV span); L2 Part B pass 1 30/30 + anchors; #3051 payload re-run | pypto `af75165dd`, runtime `d8f997fa`; queue ids in report |
| 2026-10-14 | [ep8-l2corenum](issue-2521-a1-ep8-l2corenum-2026-10-14/README.md) | first L2 multi-lane: `managed-l2` L=8 = 186 µs / 19.75 Gbit/s/rank; HOST rail gains ~3% from multi-lane | `task_20261014_085749_20485986058` |
| 2026-10-14 | [a2av-resident-args](a2av-resident-args-2026-10-14/README.md) | `--resident-args` cuts the EP8 L=8 1 MiB host slot 10377→5270 µs (1.97×), kernel unchanged | `task_20261014_112920_242602513476`; branch archived (SHA in upstream-status) |
| 2026-09-28 | [k3-repro](issue-2521-k3-repro-2026-09-28/README.md) | external reproduction of the claimed K3 speedups | K3 head `cf136e73` |
| 2026-09-24 | [k3-lane-scaling](issue-2521-k3-lane-scaling-2026-09-24/README.md) | repeatability re-run of the 2026-09-23 EP4 matrix | EP4, NPUs 4–7 |
| 2026-09-24 | [k3-lane-scaling-ep8](issue-2521-k3-lane-scaling-ep8-2026-09-24/README.md) | EP8 envelope complementing the EP4 campaigns | 8-rank |
| 2026-09-25 | [k3-bw-ceiling](issue-2521-k3-bw-ceiling-2026-09-25/README.md) | how close lane-mapped a2av is to the interconnect peak | follow-up to EP8 campaign |

## Superseded

| Date | Campaign | Superseded by | Note |
|---|---|---|---|
| 2026-09-23 | [k3-lane-scaling](issue-2521-k3-lane-scaling-2026-09-23/K3_LANE_SCALING_REPORT.md) | 2026-09-24 repeatability run | first EP4 matrix |
| 2026-09-22 | [k1ab-k1](issue-2521-a1-alltoallv-k1ab-k1-2026-09-22/) / [k1ab-prek1](issue-2521-a1-alltoallv-k1ab-prek1-2026-09-22/) | K1/K3 campaigns | K1 delta |
| 2026-09-23 | [a1-alltoallv-baseline](issue-2521-a1-alltoallv-baseline-2026-09-23/) | K1/K3 campaigns | pre-K1 baseline |
| 2026-09-25 | [alltoallv-b0-prek1](issue-2521-a1-alltoallv-b0-prek1-2026-09-25/) | K1/K3 campaigns | pre-K1 B0 |

## Archived

| Date | Campaign | Why archived |
|---|---|---|
| 2026-09-29 | [k3-cf136e73](issue-2521-a1-k3-cf136e73-2026-09-29/README.md) | K3 merged; superseded by the 2026-10-22 main run |
| 2026-09-29 | [main snapshot](issue-2521-a1-main-2026-09-29/README.md) | pre-K3/O2 tree |
| 2026-09-29 | [e42 in-tree harness](issue-2521-a1-e42-in-tree-harness-2026-09-29/README.md) | PR #2945 merged 2026-10-20 |
| 2026-09-28 | [b1-cal-rep](issue-2521-a1-b1-cal-rep-2026-09-28/README.md) | timing-slot repeatability, pre-K3 |
| 2026-09-28 | [b1-k1ep8-swap](issue-2521-a1-b1-k1ep8-swap-2026-09-28/README.md) | K1 merged |
| 2026-09-28 | [b1-tails](issue-2521-a1-b1-tails-2026-09-28/README.md) | all tail points N/A by construction |
| 2026-09-25 | [b1](issue-2521-a1-b1-2026-09-25/) | pre-K3 baseline |
| 2026-09-23 | [k2-corenum-scaling](issue-2521-k2-corenum-scaling-2026-09-23/) | K2 merged |
| 2026-09-14 | [noise re-run](issue-2521-a1-alltoallv-noise-rerun-2026-09-14/README.md) | re-run of 2026-09-10 cells |
| 2026-09-11 | [weird recheck](issue-2521-a1-alltoallv-weird-recheck-2026-09-11/README.md) | off-trend-point recheck |
| 2026-09-10 | [EP8](issue-2521-a1-alltoallv-ep8-2026-09-10/README.md) / [forceipc](issue-2521-a1-alltoallv-ep8-forceipc-2026-09-10/README.md) / [gapfill](issue-2521-a1-alltoallv-ep8-gapfill-2026-09-10/README.md) / [l2host](issue-2521-a1-alltoallv-ep8-l2host-2026-09-10/README.md) | first EP8 attempts; superseded by later campaigns |
| 2026-09-09 | [ubfix](issue-2521-a1-alltoallv-ubfix-2026-09-09/README.md) | A0/A1 baseline record |
| 2026-09-02 | [benchmark report (PR #2591)](benchmark-report-2026-09-02-pr2591-final.md) | #2591 merged |
| 2026-08-31 | [barrier-dcci NPU results](barrier-dcci-npu-results-2026-08-31.md) / [reproducibility](barrier-dcci-reproducibility-artifact.md) | DCCI work landed as #2591 |

## Reference

[benchmark-report-2026-08-28](benchmark-report-2026-08-28.md) (methodology baseline) ·
[corenum crossover](corenum-message-size-crossover-2026-08-31.md) ·
[issue-2521-context](issue-2521-context.md) ·
[A1 executive summary](issue-2521-a1-EXECUTIVE-SUMMARY-2026-09-28.md) ·
[frozen-baseline tactic](issue-2521-a1-frozen-baseline-tactic-2026-09-25.md) ·
[npu-queue-runbook](npu-queue-runbook.md) · [onbox-todo](onbox-todo.md) ·
[perf ideas](perf-improvement-ideas-2026-08-28.md) ·
[plan 101](plan101-phase0-reset-cost-and-l2-crash-2026-09-03.md) ·
[plan 107](plan107-ablation-campaign-record-2026-09-03.md)

## Where the measured code lives

- **In-tree harness:** `pypto/tests/st/distributed/collectives/all_to_all_v_benchmark.py`
  (merged via #2945; sha256 `958f6490…` at pypto `f32ecf3f0`).
- **Campaign driver:** [`collectives/alltoallv_a1.py`](../collectives/alltoallv_a1.py) — defaults to
  the in-tree harness as of 2026-10-30; the old vendored snapshot was removed.
- **Archived fork branches:** plan-129 `a2av-harness-l2-core-num` @ `22fc5886…` and
  `a2av-resident-args` @ `9bd13666…` (archive tags on the fork).
