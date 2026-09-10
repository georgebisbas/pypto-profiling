# Issue draft — EP8 Fabric V2 cross-server domain alloc flake

**Filed:** https://github.com/hw-native-sys/simpler/issues/2192

Paste into GitHub bug report (or `gh issue create`).

**Title:** `[Bug] a2a3 Fabric V2 domain export intermittently fails on EP8 (“cross-server”) → release_domain -1 / SIGSEGV`

**Platform:** a2a3 (Ascend 910B/C hardware)  
**Runtime variant:** tensormap_and_ringbuffer (also hits host_build_graph path — same `comm_hccl.cpp`)  
**Host:** Linux (aarch64)  
**Commit:** `39ce891dbb3f665e72e99b4a1387b47012fb18bd` (simpler tip when filed; bump as needed)  
**CANN:** 9.0.0 (`Ascend-cann-toolkit`)  
**Driver / npu-smi:** 26.0.rc1  

## Description

On 8× 910B2 (devices 0–7 spanning multiple PCIe roots), `aclrtMemExportToShareableHandleV2`
with `ACL_MEM_SHARE_HANDLE_TYPE_FABRIC` intermittently fails with driver
“does not support cross-server communication” (`err=0x702001d`). Host then sees
AICPU 507018 / scheduler timeout, domain release fails (`comm_release_domain_windows
-1`), and the process often SIGSEGVs. Subsequent jobs may fail `comm_init` until
devices settle.

IPC fallback in `comm_hccl.cpp` only triggers on exact
`ACL_ERROR_RT_FEATURE_NOT_SUPPORT` (207000), so these failures do not fall back.

## Steps to Reproduce

```bash
cd simpler
export LD_PRELOAD=…/libhccl.so
unset SIMPLER_COMM_FORCE_IPC
python3 tools/mfe/ep8_fabric_cross_server/repro_domain_churn.py --iters 30 --devices 0-7
```

See `tools/mfe/ep8_fabric_cross_server/README.md`.

## Expected Behavior

Domain allocate/release across 8 onboard chips completes stably (or cleanly falls
back to VMM IPC when Fabric is unsupported).

## Actual Behavior

Intermittent:

```text
ExportToShareableHandleV2 … cross-server communication … feature not support
→ 507018 / release_domain failed with code -1 / SIGSEGV
```

## Additional Context

- EP2/EP4 much more stable; EP8 is the sharp repro.
- Not an AllToAllV kernel bug — MFE uses empty `allocate_domain` only.
- Proposed: `SIMPLER_COMM_FORCE_IPC` + broader `kUnsupported` mapping + safer teardown.
