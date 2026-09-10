# MFE: EP8 Fabric V2 “cross-server” domain alloc flake (a2a3)

Vendored under `pypto-profiling/collectives/a1_alltoallv/mfe/`.

## Symptom

On an 8× Ascend 910B2 box (devices spanning multiple PCIe roots), repeated
L3 `orch.allocate_domain` / Worker teardown intermittently fails with:

```text
This device does not support cross-server communication
  drv devId=N localServerId=1023 err=0x702001d
rtMemExportToShareableHandleV2 ... feature not support
→ ACL 507018 / scheduler timeout / poisoned lane
→ release_domain / comm_release_domain_windows failed with code -1
→ often SIGSEGV; next job may fail comm_init
```

This is **simpler host-side Fabric V2 window export**, not an AllToAllV kernel
bug. pypto A1 EP8 campaigns surface it as `release_domain` / `rc=-11`.

## Code under test

`src/a2a3/platform/onboard/host/comm_hccl.cpp` (simpler)

- Prefer Fabric V2 (`export_fabric_window` / `ACL_MEM_SHARE_HANDLE_TYPE_FABRIC`)
- Fall back to VMM IPC **only** when Export returns exactly
  `ACL_ERROR_RT_FEATURE_NOT_SUPPORT` (207000)
- Intermittent “cross-server” failures often arrive as AICPU 507018 instead,
  so IPC fallback never runs

## Repro

Needs a simpler checkout with Python bindings / host runtime on the box.

```bash
cd /path/to/simpler
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
unset SIMPLER_COMM_FORCE_IPC

MFE=/path/to/pypto-profiling/collectives/a1_alltoallv/mfe/repro_domain_churn.py

# Pure simpler: no pypto, no AIV kernel — only allocate_domain on 8 chips
python3 "$MFE" --iters 30 --devices 0-7

# Higher handshake rate (closer to dual prepare sessions):
python3 "$MFE" --iters 30 --devices 0-7 --dual-alloc
```

Short runs often stay green; the flake is intermittent. Empty `allocate_domain`
churn is the minimal simpler API, but hit rate is lower than a full pypto A1
EP8 AllToAllV session. Prefer A1 harness attempt logs for reliable evidence.

### Control (workaround)

Apply `../patches/simpler_comm_force_ipc.patch`, rebuild host runtime, then:

```bash
SIMPLER_COMM_FORCE_IPC=1 python3 "$MFE" --iters 30 --devices 0-7 --dual-alloc
```

Should stay green.

## Suggested fix (simpler)

1. Env escape hatch `SIMPLER_COMM_FORCE_IPC=1` → skip Fabric, use `*_via_ipc`
   for base + domain windows (see patch in this bundle).
2. Widen Fabric→IPC fallback: treat export “feature not support” / cross-server
   as `FabricAttempt::kUnsupported`, not hard `kError`.
3. Harden teardown so a Fabric fail does not leave the next job in `comm_init`
   failure.

## Related

- Issue: https://github.com/hw-native-sys/simpler/issues/2192
- Field notes: `reports/issue-2521-a1-alltoallv-ep8-2026-09-10/debug/DEBUG_NOTES.md`
- Analytic report: `reports/issue-2521-a1-alltoallv-ubfix-2026-09-09/ANALYTIC_REPORT_FOR_COLLEAGUES.md`
