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

10 consecutive failures (4 on 2026-09-25 + 6 on 2026-09-28), **identical
signature every attempt** — deterministic, not flaky:

1. Dies inside `_timed_session` (timing-slot session, after
   `session=timing-slot enter warmup=5 rounds=100`; the swimlane session is
   never reached — timing-slot runs first, so no swimlane salvage is possible
   from the failed logs).
2. `[ERROR] copy_back_run_outputs_impl: [runtime_maker.cpp:1034] scheduler
   timeout sub_class=S1:running-stalled (detail=1) completed=0/1 run`
3. `comm_release_domain_windows ... release_domain: barrier timed out` (ranks
   vary per attempt) → `comm_destroy ... final release failed` → exit `rc=-11`,
   8 leaked shared_memory objects.

## Root-cause lead (HOST-rail K2 gate)

B1 = `d626aea16` is post-K1 / pre-K2. K2 (#2889,
`e48937b9 feat(collectives): multi-block all_to_all_v launches`) adds
`allow_wider_lanes=true` at `src/ir/transforms/lower_host_tensor_collectives_pass.cpp`
sites **574** and **604** (the "both sites" repair). Verified tree diff:

| Tree | `allow_wider_lanes=true` sites |
|---|---|
| B1 (`d626aea16`) | 263 only |
| K3 (`cf136e73`, includes K2 #2889) | 263, 574, 604 |

The tracker (§K2) already records that K1 (#2828) regressed the HOST rail and
K2 restored it — consistent with this stall being HOST-path-specific (L2 rail
`random@24960` passes; other `managed-host` cells pass, some with retries).

## Follow-ups (results appended here)

1. Standalone B1 `--profile swimlane` acquisition for this cell (salvage the
   aicore/L2-rail metrics blocked by the timing-slot stall).
2. Counter-check: same cell on the K3 tree (`cf136e73`, includes K2) with
   timing-slot — pass ⇒ stall attributable to the missing K2 HOST-rail gate.
