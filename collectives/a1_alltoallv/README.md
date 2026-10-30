# RFC #2521 A1 AllToAllV — measurement bundle (in this repo)

**If you opened the analytic report:** every script it refers to for *inspection*
is here (or next to it as `../alltoallv_a1.py`). Clone / pull **`pypto-profiling`**;
you do not need `/opt/pypto` untracked files to read the harness or patch.

Self-contained copies of what produced the A1 report, so colleagues with only
this repo can inspect (and re-run against a pypto/simpler tree).

| Path | Role |
|------|------|
| [`../alltoallv_a1.py`](../alltoallv_a1.py) | Campaign driver (EP × impl × payloads) |
| in-tree harness | `$PYPTO_ROOT/tests/st/distributed/collectives/all_to_all_v_benchmark.py` (merged via pypto #2945; the vendored snapshot was removed 2026-10-30) |
| [`patches/simpler_comm_force_ipc.patch`](patches/simpler_comm_force_ipc.patch) | Local simpler escape hatch `SIMPLER_COMM_FORCE_IPC` |
| [`mfe/`](mfe/) | Fabric domain-churn MFE + issue draft ([simpler#2192](https://github.com/hw-native-sys/simpler/issues/2192)) |
| [`PROVENANCE.json`](PROVENANCE.json) | Recorded SHAs / patch hash |

**Report:** [`../../reports/issue-2521-a1-alltoallv-ubfix-2026-09-09/ANALYTIC_REPORT_FOR_COLLEAGUES.md`](../../reports/issue-2521-a1-alltoallv-ubfix-2026-09-09/ANALYTIC_REPORT_FOR_COLLEAGUES.md)

**Other report dirs:** `reports/issue-2521-a1-alltoallv-{ubfix,ep8,ep8-l2host,ep8-forceipc}-2026-09-*/`

## What is timed

- Driver invokes harness with `--profile both`.
- **Official L2 kernel column:** swimlane AIV gang span (collective only).
- **Fair HOST vs L2:** `host_timing_slot_p50_us` — wall clock around one `runtime.run(...)` (includes stage/consume + outer L3→L2 dispatch(es); `prepare` is outside the timed loop).

## Re-run (needs Ascend box + pypto runtime)

```bash
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
export PYPTO_ROOT=/path/to/pypto
REPO=/path/to/pypto-profiling
# optional EP8:
#   cd "$PYPTO_ROOT/runtime" && git apply "$REPO/collectives/a1_alltoallv/patches/simpler_comm_force_ipc.patch"
#   # rebuild libhost_runtime.so, then:
#   export A2AV_FORCE_IPC=1

cd "$REPO"
A2AV_OUT=reports/issue-2521-a1-REPRO \
A2AV_EPS=2,4 A2AV_IMPLS=managed-l2,managed-host \
python3 -u collectives/alltoallv_a1.py
```

The driver defaults to the in-tree harness under `$PYPTO_ROOT`. Override with `A2AV_HARNESS=...`.

## Apply FORCE_IPC patch

```bash
REPO=/path/to/pypto-profiling
cd "$PYPTO_ROOT/runtime"   # or standalone simpler checkout
git apply "$REPO/collectives/a1_alltoallv/patches/simpler_comm_force_ipc.patch"
# rebuild libhost_runtime.so (tensormap / host_build_graph) as usual for this tree
```

Then set `SIMPLER_COMM_FORCE_IPC=1` or `A2AV_FORCE_IPC=1` (driver sets the former).

## Note on upstream status

The harness is **in `pypto` main** (merged as
[#2945](https://github.com/hw-native-sys/pypto/pull/2945) on 2026-10-20); this bundle no longer
vendors a copy. The FORCE_IPC patch is still local-only and is **not** on `simpler` main. The INT8
builtin is pypto [PR #2714](https://github.com/hw-native-sys/pypto/pull/2714).

## Sync status (re-checked 2026-10-30)

| | Path / ref |
|---|---|
| In-tree harness | `pypto/tests/st/distributed/collectives/all_to_all_v_benchmark.py` — pypto [#2945](https://github.com/hw-native-sys/pypto/pull/2945), **merged 2026-10-20**; sha256 `958f6490…` at pypto `f32ecf3f0` |
| Upstream enablers | pypto [#2971](https://github.com/hw-native-sys/pypto/pull/2971) (O2a, merged 2026-10-13), [#2889](https://github.com/hw-native-sys/pypto/pull/2889) (K2), [#2900](https://github.com/hw-native-sys/pypto/pull/2900) (K3), [#3051](https://github.com/hw-native-sys/pypto/pull/3051) (32 KiB staging) |
| Plan 129 / resident-args | fork branches `a2av-harness-l2-core-num` @ `22fc5886…` and `a2av-resident-args` @ `9bd13666…` — **archived** 2026-10-30 (tags; SHAs in [`reports/upstream-status.md`](../../reports/upstream-status.md)) |
| This bundle | FORCE_IPC patch, MFE, `PROVENANCE.json`; the vendored harness (`3c1f83be…`, 2026-10-13) was removed |

The in-tree harness carries the plan-129 features (`cal_all_to_all_v_blocks`, `core_num`
pass-through, no `core_num=1` gate) and emits `per_rank_receive_bandwidth_gbps` instead of the old
`egress_gbps`; `alltoallv_a1.py` accepts both keys. Historical A0/A1 numbers produced by the
`3c1f83be…` snapshot remain in the 2026-09 reports and are not comparable at `L>1`.
