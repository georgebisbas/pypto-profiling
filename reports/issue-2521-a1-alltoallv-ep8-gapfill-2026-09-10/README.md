# EP8 gap-fill (2026-09-10) — retry results

## Done

| Tag | Result |
|-----|--------|
| L2 uniform `32768…1048576` | **OK** (full large curve) |
| HOST uniform `0`, `24576`, `24960`, `32768` | **OK** |
| HOST zero / single-hot @ `24960` | **OK** |

## Not yet done

| Tag | Notes |
|-----|--------|
| HOST uniform `49152…1048576` | `49152` FAIL 5/5; heal poisoned (`rc=-11`); stopped |
| L2 zero / single-hot @ `24960` | SIGSEGV thrash after long L2 curve; aborted |

Logs: `sweep_retry.log`, `sweep_host_large.log`, `json/`, `attempt_logs/`.
