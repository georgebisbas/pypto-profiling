# EP8 failure debug notes

**Date:** 2026-09-10  
**Campaign:** `issue-2521-a1-alltoallv-ep8-2026-09-10`  
**Debug artifacts:** this `debug/` directory

## Verdict

`release_domain` / `comm_release_domain_windows failed with code -1` / `rc=-11` are **teardown symptoms**, not the root fault.

**Root fault (when DFX is preserved):** during an EP8 run, a chip hits Fabric V2 export failure:

```text
ACL_ERROR_RT_AICPU_EXCEPTION (507018)
This device does not support cross-server communication
  drv devId=<N> localServerId=1023 err:0x702001d
rtMemExportToShareableHandleV2 ... feature not support
→ SCHEDULER_TIMEOUT S1:running-stalled
→ chip run lane poisoned (-100)
→ prepare()/close → release_domain barriers time out → -1 → often SIGSEGV
```

Evidence:

| Artifact | Root signal |
|----------|-------------|
| Campaign `builds/.../4096/dfx_outputs/host.3713532.log` | cross-server on **devId=7** |
| `debug/campaign_exact/pb0/log.txt` | cross-server on **devId=5**, then release_domain -1, **rc=139** |
| Successful campaign sizes (32/128/1024/8192) | **no** cross-server in DFX |

a2a3 window path (`runtime/src/a2a3/.../comm_hccl.cpp`): Fabric V2 (`ACL_MEM_SHARE_HANDLE_TYPE_FABRIC`) with IPC shareable-handle fallback. The failing call is **ExportToShareableHandleV2** (Fabric), which this box sometimes rejects mid-run as “cross-server”.

## Causal chain

```text
Fabric export fail (intermittent on EP8)
  → AICPU exception / scheduler stall / poisoned lane
  → release_domain barrier timeout → code -1
  → process SIGSEGV (rc=-11/-139)
  → next job: comm_init failed (poisoned HCCL/device state)
  → after cooldown / lighter EP4 smoke, EP8 often works again
```

## Repro matrix (this session)

| Case | Result |
|------|--------|
| 8× short EP8 L2 4096 `profile both` back-to-back | **8/8 OK** |
| Isolated EP8 L2 4096 rounds=100 ×3 (earlier) | **OK** |
| Campaign-exact EP8 L2 **peer_bytes=0** (100+both) | **FAIL** — cross-server → poison → release -1 → SIGSEGV |
| Immediately after: EP8 4096 / 16384 | **FAIL** — `comm_init failed` (cascade) |
| EP4 smoke after cascade | **OK** |
| EP8 L2 4096 after recovery (20+both) | **OK** (rank_spread ≈ 3.9 ms) |

EP2/EP4 **peer_bytes=0** succeeded in the ubfix archive; EP8 zero-byte is the sharpest repro.

## Swimlane rank skew (separate, on OK runs)

Per-rank `__builtin_all_to_all_v__int8` durations form a smooth rank gradient (~4 ms → ~90 µs). Stage/consume stay ~3–6 µs. Not a collector mix-up. Roadmap fastest-rank still picks the short end; spread documents sync wait inside the AIV task.

Harness now logs session phase markers:

```text
[a2av-bench] session=timing-slot enter|exit ok
[a2av-bench] session=swimlane enter|exit ok
```

## Fix for `peer_bytes=0` / EP8 Fabric flake

**Not a zero-route kernel bug** — short EP8 `peer_bytes=0` can succeed; long sessions hit intermittent Fabric V2 export (“cross-server”).

**Runtime escape hatch (local patch):** `SIMPLER_COMM_FORCE_IPC=1` skips Fabric and uses VMM IPC for base + domain windows (`comm_hccl.cpp`). Rebuilt into `libhost_runtime.so`.

**Campaign:** set `A2AV_FORCE_IPC=1` (wired in `alltoallv_a1.py`).

**Verified:** EP8 L2 `peer_bytes=0`, warmup=5 / rounds=100 / profile both → **OK** with FORCE_IPC (`ep8-l2host-…/zero_force_ipc/`, AIV p50 ≈ 14.7 µs).

## Practical mitigations (campaign) — implemented

Driver `collectives/alltoallv_a1.py` now supports:

| Env | Role |
|-----|------|
| `A2AV_RETRIES` | Extra attempts after first fail (resume used **3** → 4 tries) |
| `A2AV_COOLDOWN_S` | Sleep between tries (25s) |
| `A2AV_HEAL_EP` | EP4 timing-slot smoke before retry |
| `A2AV_RESUME=1` | Keep already-OK tags in `summary.json` |

Resume (no-zero) so far: **4096 OK try1**; **16384 OK try4** after 3× cross-server/`release_domain` fails + heal. `peer_bytes=0` quarantined (poisons the box).

## Open questions

1. Why Fabric V2 export intermittently claims “cross-server” on this 8×910B2 topology (PCIe roots `C*`, `81*`, `01*`, `41*`) while many EP8 sizes succeed?
2. Does zero-route AllToAllV take a worse window/handshake path?
3. Can runtime force IPC fallback instead of Fabric when V2 returns feature-not-support earlier / more reliably?
4. Is the ~4 ms EP8 rank gradient expected HCCL wait skew or a gang-launch ordering issue?

## Key paths

- Harness: `/opt/pypto/tests/st/distributed/collectives/all_to_all_v_benchmark.py`
- Comm: `/opt/pypto/runtime/src/a2a3/platform/onboard/host/comm_hccl.cpp` (`export_fabric_window`)
- Docs: `/opt/pypto/runtime/docs/comm-domain.md` §4
- This debug tree: `debug/campaign_exact/`, `debug/back2back/`, `debug/post_poison_recovery/`
