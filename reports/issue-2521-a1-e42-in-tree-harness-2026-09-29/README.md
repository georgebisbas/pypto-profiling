# E4.2 — in-tree harness verification (2026-09-29)

**Deliverable:** A1/E4.2 — port the measurement harness into the pypto tree
(`tests/st/distributed/collectives/all_to_all_v_benchmark.py`) as
`hw-native-sys/pypto` PR **#2945** (head `georgebisbas:a2av-in-tree-harness`,
commit `a46dd0c4`).

**What changed vs the measurement bundle:** measurement logic unchanged
(formatting-normalized by ruff 0.14.8; no logic edits); the bundle copy is synced
to identical content (pypto-profiling `e071447e`). Functional deltas:
`read_pypto_commit()` repo-root walk + `git rev-parse` (worktree-safe; a timeout
falls back to `unknown`), plus the review-round fixes — empty-sample rank
exclusion (`aiv_ranks_missing` marker), compile-only device-id validation, and
the last-completing-block span (see `validation/`).

**Verification runs** (same env as the B1 measurement bundle; a2a3; devices 4,5;
PTOAS 0.65; harness from worktree `/opt/a1/e42` @ `a46dd0c4`):

| run | args | result |
|---|---|---|
| smoke (compile-only gate) | `--smoke --ep 2 --peer-bytes 1024` | rc=0; schema-valid JSON; `pypto_commit=a46dd0c4f7bd` |
| live micro-run | `--ep 2 --peer-bytes 1024 --count-pattern uniform --core-num 1 --impl managed-l2 --rounds 2 --warmup 1 --profile timing-slot --devices 4,5` | rc=0; metric `timing_slot`; fastest-rank p50 **1653.76 µs**; counts correct (`max_recv=1`) |

The smoke's `pypto_commit` output exercises the new lookup path: it correctly
reports the harness worktree HEAD (`a46dd0c4…`) — the old code crashes there.

**Artifacts:** `json/smoke.json`, `json/live2.json`, `logs/smoke.log`,
`logs/live2.log`, `validation/` (F4 parser replay + synthetic fixture). These are
verification-only runs, not campaign data.

Re-verified on PR head `a46dd0c4` (review fixes included: timeout catch, rank
selection, compile-only device ids, last-block span, tail note).
