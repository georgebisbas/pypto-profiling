# Upstream status — pypto / simpler / pto-isa

**Last verified:** 2026-10-30. This page is the single place where upstream SHAs and PR states are
recorded; other docs link here instead of hard-coding them.

## Workspace checkouts

| Tree | Revision | Notes |
|---|---|---|
| `pypto` | `f32ecf3f05d23a83a52283ad7998ca59cd8d291d` (2026-10-30) | `git -C ../pypto rev-parse HEAD` |
| `pypto/runtime` gitlink | `6b6b88f2353cca12cfafe809750b50b59931b0d1` | the pin recorded by pypto `HEAD` |
| `pypto/runtime` checkout | `6b6b88f2353cca12cfafe809750b50b59931b0d1` (2026-10-29) | synced 2026-10-30; matches the gitlink |
| `pto-isa` pin (from runtime `6b6b88f2`) | `217bb98952a8f9004b95fe938da97d8d5f68f702` | read through the gitlink, not the submodule tree |
| Container image (local `pypto3-hw-native-sys:cann9`) | rebuilt 2026-10-30 at pypto `f32ecf3f0`, runtime `6b6b88f2`, pto-isa `217bb989`, PTOAS 0.70 | current; pypto-docker `dee4d45` carries the build pin. Old image preserved as `:cann9-7fc45aed` (installed pypto `7fc45aed`, runtime `6e383fc5`, PTOAS 0.65) |

The image and the workspace are not the same tree. Quote the image pin for container runs and the
workspace SHA for runs made from this checkout; never mix them in one comparison.

Latest HCCL-vs-pypto comparison on the rebuilt image:
[`post-cleanup-main-2026-10-30/`](post-cleanup-main-2026-10-30/README.md).

## RFC #2521 PRs

| PR | Item | State |
|---|---|---|
| pypto #2591 | slim per-peer dcci + remove notify barrier | merged |
| pypto #2635 | lower managed `all_to_all_v` inside CHIP orchestration | merged |
| pypto #2714 | INT8 variant for builtin `all_to_all_v` | merged |
| pypto #2828 | pull counts from each rank's window (K1) | merged |
| pypto #2889 | multi-block `all_to_all_v` launches (K2) | merged (`e48937b9`) |
| pypto #2900 | multi-AIV lane-mapped launches (K3) | merged |
| pypto #2945 | in-tree AllToAllV harness + `managed-l2` L>1 | **merged 2026-10-20** |
| pypto #2971 | CHIP `all_to_all_v` at compile-time `core_num` (O2) | merged (`fe651f29`, 2026-10-13) |
| pypto #3051 | 32 KiB ping-pong staging buffers | merged 2026-10-19 |
| pto-isa #291 | TFILLPAD UB-OOB | **closed 2026-09-16** |

## Open upstream, relevant here

| PR | What it changes | Why it matters to this repo |
|---|---|---|
| pypto #3229 | capture builtin collective dispatches in DFX | supplies the missing HOST-rail AIV span; unblocks O2/A3 Part B (g1/g2) |
| pypto #3104 | reject non-contiguous operands on the managed L2 rail | prevents silent corruption if a future case feeds strided tensors |
| pypto #2645 | persistent-domain retention (draft) | could change the `--persistent` baseline |
| pypto #2939 | chunk InCore `reduce_scatter` | allreduce-adjacent; watch for harness API changes |

## Archived fork work (2026-10-30)

These branches were stacked on pre-#2945 history and had no upstream PR. On 2026-10-30 they were
tagged and the branch refs deleted from the fork:

| Archive tag | Commit | Evidence it produced |
|---|---|---|
| `archive/a2av-harness-l2-core-num` | `22fc588638a7cd8b009ab084e09f0cc717d4c249` | plan-129 L>1 harness; EP8 first run in `issue-2521-a1-ep8-l2corenum-2026-10-14/` |
| `archive/a2av-resident-args` | `9bd13666add172f1c8a36c91757b09600ab49eaa` | 1.97× host-slot result in `a2av-resident-args-2026-10-14/` |

## Known gaps

1. **O2/A3 Part B** — 9 `managed-host` g1/g2 matrix cells unrun: the HOST rail emits no AIV span
   until #3229 lands.
2. **plan-129** — `L ∈ {4,16}` and the official payload curve were never measured; the branch is
   archived, so a future attempt must be rebased from the archive tag or restarted.
3. **resident-args** — measured at one cell (EP8, L=8, 1 MiB); not re-run across payloads or
   `L ∈ {1,16}`.
4. **pto-isa #291** — the fix is in the current pin, but the `fillpad`/`allfour` slot-order
   workaround is still in the kernels pending a hardware A/B.
5. **#3051 re-run** — 8/16 MiB per peer fail to compile (hard-coded 60 s PTOAS timeout).

## Refreshing this page

```bash
git -C ../pypto rev-parse HEAD
git -C ../pypto ls-tree HEAD runtime
git -C ../pypto/runtime rev-parse HEAD
git -C ../pypto/runtime show "$(git -C ../pypto ls-tree HEAD runtime | awk '{print $3}')":pto_isa.pin
gh pr view 3229 --repo hw-native-sys/pypto --json state,mergedAt

# Verify what the container actually installed (build args can lie):
docker exec <container> git -C /opt/pypto rev-parse HEAD
docker exec <container> git -C /opt/pypto/runtime rev-parse HEAD
```

To measure current `main`, rebuild the image with
`docker build --build-arg PYPTO_COMMIT=f32ecf3f05d23a83a52283ad7998ca59cd8d291d -t pypto3-hw-native-sys:cann9-main -f Dockerfile.hw-native-sys.cann9.0 .`.
