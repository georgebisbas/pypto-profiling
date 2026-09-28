# E1 resume completion — 2026-09-28

**Context.** E1 (2026-09-25) finished 39/44 cells; 5 cells were lost to the EP8
`release_domain` teardown flake. A resume run (same tree/protocol, `A2AV_RESUME=1`,
`RETRIES=5`, `COOLDOWN=60s`, `HEAL_EP=4`, `SIMPLER_COMM_FORCE_IPC=1`) ran
2026-09-28 07:5x–08:42 UTC.

## Outcome: 43/44 OK, 1 cell blocked

**Recovered (4)** — now in `json/` and `summary.json`:

- `p8_l1_managed-host_uniform_0`
- `p8_l1_managed-l2_single-hot_24960`
- `p8_l1_managed-l2_uniform_262144`
- `p8_l1_managed-l2_uniform_1048576`

**Blocked (1): `p8_l1_managed-host_random_24960`**

10 consecutive failures (4 on 2026-09-25 + 6 on 2026-09-28) on the official
100-round protocol, **identical signature every attempt** — see *Stall
characterization* below (round-count dependent):

1. Dies inside `_timed_session` (timing-slot session, after
   `session=timing-slot enter warmup=5 rounds=100`; the swimlane session is
   never reached — timing-slot runs first, so no swimlane salvage is possible
   from the failed logs).
2. `[ERROR] copy_back_run_outputs_impl: [runtime_maker.cpp:1034] scheduler
   timeout sub_class=S1:running-stalled (detail=1) completed=0/1 run`
3. `comm_release_domain_windows ... release_domain: barrier timed out` (ranks
   vary per attempt) → `comm_destroy ... final release failed` → exit `rc=-11`,
   8 leaked shared_memory objects.

## Stall characterization (completed 2026-09-28 ~09:25 UTC)

The stall is **round-count dependent and tree-independent** — a runtime/
host-path issue, not a pypto K1/K2 gate issue:

| session | rounds | FORCE_IPC | state | outcome |
|---|---|---|---|---|
| timing-slot | 100 | yes | cascade-dirty | stall ×11 (2 days) |
| timing-slot | 100 | yes | clean | **stall** (rc=139) |
| timing-slot | 80 | yes | clean | **stall** (rc=139) |
| timing-slot | 60 | yes | clean | **PASS — 12550.38 µs** |
| timing-slot | 30 | yes | clean | **PASS — 11229.08 µs** |
| timing-slot | 30 | no (Fabric) | clean | **PASS — 12379.51 µs** |
| timing-slot | 30 | yes | after failed heal | stall ×1 |

- Clean-state boundary: passes at 60 rounds (+5 warmup), stalls at 80 (+5)
  ⇒ threshold ≈ 65–85 dispatches.
- The same 100-round stall **reproduces on the K2+K3 tree** (`/opt/pypto` @
  `cf136e73`, pypto_core built after K2 #2889 merged) ⇒ not fixed by, and not
  attributable to, the K2 `allow_wider_lanes` gate diff (B1 lacks sites
  574/604; K3 has them; both stall). Both trees use runtime `6e383fc5` +
  FORCE_IPC patch ⇒ runtime-level.
- Signature chain (every stall): `copy_back_run_outputs_impl ... scheduler
  timeout sub_class=S1:running-stalled (detail=1) completed=0/1 run` →
  `comm_release_domain_windows ... barrier timed out` (ranks vary) →
  `comm_destroy ... final release failed` → `rc=-11` (or `-139` with core
  dump), 8 leaked shared_memory objects.
- Failure hygiene matters: a failed heal / leaked-worker cascade can also
  stall a 30-round session. Always: kill orphans → all cards idle → heal
  smoke rc=0 → measure.
- `--profile swimlane` observations: 8-round run completes clean (rc=0) but
  the host rail yields no AIV capture (`aiv_rounds_captured: None` —
  structural: all 22 AIV-capturing E1 cells are managed-l2) and the DFX-taxed
  slots (~19 s) are unusable; the 100-round swimlane-only run reached deep DFX
  post-processing but died in the known teardown-flake family.

**Delivery recommendation:** measure this cell with the 30/60-round
supplementary protocol (below) and label it; file a runtime/simpler issue for
the 65–85-dispatch stall using this bisect.

## Supplementary measurements (clean state, 2026-09-28)

Official protocol remains 100-round/5-warmup. Because this cell cannot
complete it, the values below were acquired with the reduced protocol already
used elsewhere in this project (e.g. the K3 EP8 uniform 30-round redo).
Artifacts copied to `supplement/` in this report directory; raw logs/builds
stay on disk at `/opt/a1/salvage/`.

| rounds | condition | host_timing_slot_p50 | artifact |
|---|---|---|---|
| 30 | FORCE_IPC | 11229.08 µs | `supplement/p8_l1_managed-host_random_24960_r30.json` |
| 60 | FORCE_IPC (closest to protocol) | **12550.38 µs** | `supplement/p8_l1_managed-host_random_24960_r60.json` |
| 30 | Fabric (no FORCE_IPC) | 12379.51 µs | `supplement/p8_l1_managed-host_random_24960_r30_noforce.json` |

Sanity: neighbors on the official protocol are asymmetric 12817.88 µs and
mixed 12273.95 µs — the 60-round value sits in-family; the 30-round value is
~10% lower (fewer rounds ⇒ different p50 sampling; do not mix protocols
without a footnote).
