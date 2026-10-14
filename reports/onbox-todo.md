# On-box TODO queue

Hardware-only checks this repo owes someone — things that cannot be settled in sim, on the dev box,
or by reading source, and that are therefore taken on when we next hold an NPU window.

Rules for this file: every entry states **what decides the answer** (not just "re-run X"), the exact
steps, and the exit criteria that close it. Sim is not a substitute for any entry here. Add new
entries at the top of §, keep closed ones out (delete them when done).

Reference: [AGENTS.md](../AGENTS.md) §2 (container, box) and §2.1 (the queue), §3 (quick start),
[`npu-queue-runbook.md`](npu-queue-runbook.md) (in-repo: how to run any entry below through
`task-submit`),
[`npu-task-submit-runbook.md`](https://github.com/hw-native-sys/pypto-3.0-notes/blob/main/tools/npu-task-submit-runbook.md)
(box etiquette — `task-submit` exists on `192.168.150.11` **and** `192.168.150.12`; `.13` has none,
and is **not** `.11`-only as an earlier revision of this file claimed).

---

## 1. RFC #2521 plan 129 — A1 `managed-l2` `core_num>1` sweep

**Reporter:** us. **Upstream status:** O2a landed (pypto #2971, `fe651f29`, merged 2026-10-13) and
the harness side is **implemented** on branch `a2av-harness-l2-core-num` @ `22fc5886`, stacked on
`a2av-in-tree-harness` @ `eb1ed877` (pypto PR #2945, open — bot-approved only). **The hardware
sweep has never been run.**
**Decides:** whether the sanctioned RFC #2521 §10 HOST-vs-L2 comparison can be produced at all —
today the A1 harness refuses `L>1` on the `managed-l2` rail, so no L2 row of any `L`/`B` sweep is
obtainable from the sanctioned tool. Unblocks plan 116 (A3) and the `managed-l2` rows of plan 110.
**Detail:** [`pypto-3.0-notes/pr_plans/129-pypto-a1-harness-managed-l2-core-num.md`](https://github.com/hw-native-sys/pypto-3.0-notes/blob/main/pr_plans/129-pypto-a1-harness-managed-l2-core-num.md)

### 1.1 Status (2026-10-14 — partially done at EP8)

First hardware run of the L2 `L>1` path is **green**: EP8 on cards 0–7, 64 KiB/peer `uniform`,
`L ∈ {1,8}` × {managed-l2, managed-host}, 20 rounds — 4/4 cells OK, `n_fail=0`, no Fabric poison.
Queue task `task_20261014_085749_20485986058`. Results + caveats:
[`issue-2521-a1-ep8-l2corenum-2026-10-14/`](issue-2521-a1-ep8-l2corenum-2026-10-14/README.md).

Headline: `managed-l2 L=8` = **186 µs** kernel p50 / **19.75 Gbit/s/rank** (6.5× over L2 `L=1`,
34× over the HOST `L=1` baseline), while the HOST rail gained ~nothing from multi-lane (2.9%).

**Still owed:** `L ∈ {4,16}`, the rest of the `§8.3` payload curve, the official 100-round
conditions, and the PR after #2945 merges.

### 1.2 What is owed

1. Steps 4–6 of plan 129: the hardware sweep on a real box, both rails, for `L ∈ {1,4,8,16}`.
2. The PR, **only after #2945 merges** (the branch is stacked on `eb1ed877`; rebase onto `main`
   first). Do not add this commit to #2945 — its body already promises "a separate change after
   #2945".

### 1.3 Things that will bite

- **Over-capacity `L`.** `L = 64` is rejected at *submit* by the runtime
  (`REQUIRE_SYNC_START_INVALID`, verified 2026-10-09), not by the compiler. Keep the sweep to
  `L ∈ {1,4,8,16}`; a row at such an `L` should fail cleanly, not hang.
- **`launched_b` is not device-observed on the L2 rail.** It is the compile-time fold
  (`CalAllToAllVBlocks`); #2971 ships no DFX on L2. The stub writes `launched_b = expected_b` — say
  so in the PR body, and do not add a JSON field (the stub is schema-validated).
- **HOST metric fallback.** The HOST cells fell back to the host-slot metric (swimlane naming is
  missing on the HOST path), so a cross-rail ratio stays directional. Plan 129 deliberately does
  **not** fix this — it must be resolved or explicitly caveated before the §10 claim is called
  like-for-like.
- **Work in a separate worktree**, not the shared `pypto` tree.
- Queue: the sweep is a set of sub-hour units, not one long job — see the surrounding host rules in
  [`npu-queue-runbook.md`](npu-queue-runbook.md) (`--max-time` is capped at 3600 s).

### 1.4 Exit criteria

- [x] First hardware run of the L2 `L>1` path (EP8, 64 KiB, `L ∈ {1,8}`) — done 2026-10-14, 4/4 OK
- [ ] `L ∈ {4,16}` × both rails over the official payload curve, correctness passing on every row
- [ ] Cross-rail HOST-vs-L2 ratio either resolved or explicitly caveated as directional
      (still blocked: the HOST rail emits only `timing_slot`, no per-rank AIV span — see the
      2026-10-14 report caveat 1)
- [ ] PR opened after #2945 merges (fork remote only), rebased onto `main`

---

## 2. TFILLPAD UB-OOB (pto-isa#291) — verify the upstream fix and un-hack the ablation kernels

**Reporter:** us (filed 2026-09-03 from the plan-107 `fillpad`/`allfour` crash). **Upstream status:**
fixed by pto-isa `a77a7e2` (2026-09-08) + `3b49066` (2026-09-11), both on `origin/main`; the
pto-isa side asked us to verify and close.
**Ours:** not verified. The two ablation kernels still carry the slot-order workaround (`0790b5d`).
**Decides:** whether pto-isa#291 can be closed, whether the workaround can be reverted, and whether
`pypto/runtime/pto_isa.pin` needs the bump.
**Detail:** [issue291_tfillpad_ub_oob.md](https://github.com/hw-native-sys/pypto-3.0-notes/blob/main/issues/issue291_tfillpad_ub_oob.md)

### 2.1 A/B the standalone probe (discriminating case)

Our `/tmp/fp_repro` bundle is gone; the durable copy is the public
[`tfillpad-ub-oob-repro`](https://github.com/georgebisbas/tfillpad-ub-oob-repro) repo
(`kernels/aiv/tfillpad_ub_{top,low}.cpp`, `run_repro.sh`, `driver.py`, README).

```bash
# pre-fix revision — expect case 1 CRASH (-100), cases 2/3 PASS
git -C <pto-isa> checkout 96ba706c
# post-fix revision — expect all three PASS
git -C <pto-isa> checkout 3b49066
```

Hardware only: `a2a3sim` does not fault in either revision, so a sim run answers nothing here.

### 2.2 pto-isa's own regression ST

```bash
python3 tests/script/run_st.py -r npu -v a3 -t tfillpad      # inside the pto-isa checkout
```

Expect: pre-fix `testKey == 23` **FAIL with 507035**, post-fix **PASS** (upstream reports 23/23 PASS,
~41 s). Note their AIV error code (`507035`, CCU address check) differs from ours (`-100`,
`finalize_native_run`) — same OOB write, different reporting layer.

### 2.3 Prove the fix covers *our* kernel (the one that actually matters)

Passing with the workaround still in place proves nothing. Revert it first: in
`collectives/kernels/aiv/allreduce_mesh_fillpad.cpp` and `allreduce_mesh_allfour.cpp`, restore the
pre-`0790b5d` slot order (TFILLPAD'd `recvTile` back at the top slot, `0x20000`), then:

```bash
export PYTHONPATH=.
for V in fillpad allfour; do
  PYPTO_SIMPLER_KERNEL=$V python3 -m collectives.run_sweep pair-mesh \
    --count 16384 --devices 0,1 --stacks simpler-own --persistent \
    --warmup-rounds 1 --timed-rounds 3 \
    --campaign plan107_onbox_fillpad_verify --out results/campaigns/plan107_onbox_fillpad_verify/$V/results.json
done
```

Expect PASS at 16384 and 65536 on a fixed pto-isa — those are exactly the cells that crashed before
(count ≤ 4096 never crashed, so they are not evidence). Wrap the runs in `task-submit --device …`
where it exists.

### 2.4 Exit criteria

- [ ] 2.1 A/B: pre-fix crashes, post-fix passes
- [ ] 2.2 pto-isa ST case 23: pre-fix FAIL, post-fix PASS
- [ ] 2.3 `fillpad` + `allfour` PASS at count ≥ 16384 with the workaround **reverted**
- [ ] Slot-order workaround reverted in both kernels, and both called out in the follow-up commit
- [ ] `pypto/runtime/pto_isa.pin` bumped `96ba706c` → ≥ `3b49066` (simpler's pin already has it).
      Read the pin **through the recorded gitlink**, not the submodule working tree:
      `git -C pypto show "$(git -C pypto ls-tree HEAD runtime | awk '{print $3}')":pto_isa.pin`
- [ ] Comment the results on [pto-isa#291](https://github.com/hw-native-sys/pto-isa/issues/291) and
      close it; then re-measure `fillpad`/`allfour` at count ≥ 16384 (unblocks plan 107's allfour
      composition check)
