# RFC #2521 A1 AllToAllV — local measurement bundle

Self-contained copies of the **scripts and patches** that produced the A1
analytic report, so colleagues with only `pypto-profiling` can inspect (and
re-run against a pypto/simpler tree) without hunting untracked files under
`/opt/pypto`.

| Path | Role |
|------|------|
| [`../alltoallv_a1.py`](../alltoallv_a1.py) | Campaign driver (EP × impl × payloads) |
| [`harness/all_to_all_v_benchmark.py`](harness/all_to_all_v_benchmark.py) | Measured program (managed-host / managed-l2) |
| [`patches/simpler_comm_force_ipc.patch`](patches/simpler_comm_force_ipc.patch) | Local simpler escape hatch `SIMPLER_COMM_FORCE_IPC` |
| [`mfe/`](mfe/) | Fabric domain-churn MFE + issue draft ([simpler#2192](https://github.com/hw-native-sys/simpler/issues/2192)) |
| [`PROVENANCE.json`](PROVENANCE.json) | Snapshot tip SHAs / harness hash |

**Reports (already in this repo):**

- `reports/issue-2521-a1-alltoallv-ubfix-2026-09-09/ANALYTIC_REPORT_FOR_COLLEAGUES.md` — main write-up
- `reports/issue-2521-a1-alltoallv-ubfix-2026-09-09/` — EP2/EP4 full curve
- `reports/issue-2521-a1-alltoallv-ep8-2026-09-10/` — EP8 L2 resume + debug
- `reports/issue-2521-a1-alltoallv-ep8-l2host-2026-09-10/` — EP8 L2 vs HOST + `zero_force_ipc/`
- `reports/issue-2521-a1-alltoallv-ep8-forceipc-2026-09-10/` — FORCE_IPC probes (L2 24576 + 24960 OK)

## What is timed

- Driver invokes harness with `--profile both`.
- **Official L2 kernel column:** swimlane AIV gang span (collective only).
- **Fair HOST vs L2:** `host_timing_slot_p50_us` — wall clock around one `runtime.run(...)` (includes stage/consume + outer L3→L2 dispatch(es); `prepare` is outside the timed loop).

## Re-run (needs Ascend box + pypto runtime)

```bash
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
export PYPTO_ROOT=/path/to/pypto          # import / compile cwd
# optional EP8 Fabric workaround (apply patch to simpler, rebuild host runtime):
#   cd "$PYPTO_ROOT/runtime" && git apply \
#     /path/to/pypto-profiling/collectives/a1_alltoallv/patches/simpler_comm_force_ipc.patch
#   export A2AV_FORCE_IPC=1

cd /path/to/pypto-profiling
A2AV_OUT=reports/issue-2521-a1-REPRO \
A2AV_EPS=2,4 A2AV_IMPLS=managed-l2,managed-host \
python3 -u collectives/alltoallv_a1.py
```

The driver defaults to this bundle’s harness. Override with `A2AV_HARNESS=...`.

## Apply FORCE_IPC patch

```bash
cd "$PYPTO_ROOT/runtime"   # or standalone simpler checkout
git apply /path/to/pypto-profiling/collectives/a1_alltoallv/patches/simpler_comm_force_ipc.patch
# rebuild libhost_runtime.so (tensormap / host_build_graph) as usual for this tree
```

Then set `SIMPLER_COMM_FORCE_IPC=1` or `A2AV_FORCE_IPC=1` (driver sets the former).

## Note on upstream status

These files are a **snapshot for profiling/colleagues**. The harness and FORCE_IPC
patch are **not** assumed to be on `hw-native-sys/pypto` / `simpler` `main`.
INT8 builtin itself is pypto [PR #2714](https://github.com/hw-native-sys/pypto/pull/2714).
