# On-box TODO queue

Hardware-only checks this repo owes someone — things that cannot be settled in sim, on the dev box,
or by reading source, and that are therefore taken on when we next hold an NPU window.

Rules for this file: every entry states **what decides the answer** (not just "re-run X"), the exact
steps, and the exit criteria that close it. Sim is not a substitute for any entry here. Add new
entries at the top of §, keep closed ones out (delete them when done).

Reference: [AGENTS.md](../AGENTS.md) §2 (interpreter, box), §3 (quick start),
[`npu-task-submit-runbook.md`](https://github.com/hw-native-sys/pypto-3.0-notes/blob/main/tools/npu-task-submit-runbook.md)
(box etiquette — `task-submit` exists only on `192.168.150.11`).

---

## 1. TFILLPAD UB-OOB (pto-isa#291) — verify the upstream fix and un-hack the ablation kernels

**Reporter:** us (filed 2026-09-03 from the plan-107 `fillpad`/`allfour` crash). **Upstream status:**
fixed by pto-isa `a77a7e2` (2026-09-08) + `3b49066` (2026-09-11), both on `origin/main`; the
pto-isa side asked us to verify and close.
**Ours:** not verified. The two ablation kernels still carry the slot-order workaround (`0790b5d`).
**Decides:** whether pto-isa#291 can be closed, whether the workaround can be reverted, and whether
`pypto/runtime/pto_isa.pin` needs the bump.
**Detail:** [issue291_tfillpad_ub_oob.md](https://github.com/hw-native-sys/pypto-3.0-notes/blob/main/issues/issue291_tfillpad_ub_oob.md)

### 1.1 A/B the standalone probe (discriminating case)

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

### 1.2 pto-isa's own regression ST

```bash
python3 tests/script/run_st.py -r npu -v a3 -t tfillpad      # inside the pto-isa checkout
```

Expect: pre-fix `testKey == 23` **FAIL with 507035**, post-fix **PASS** (upstream reports 23/23 PASS,
~41 s). Note their AIV error code (`507035`, CCU address check) differs from ours (`-100`,
`finalize_native_run`) — same OOB write, different reporting layer.

### 1.3 Prove the fix covers *our* kernel (the one that actually matters)

Passing with the workaround still in place proves nothing. Revert it first: in
`collectives/kernels/aiv/allreduce_mesh_fillpad.cpp` and `allreduce_mesh_allfour.cpp`, restore the
pre-`0790b5d` slot order (TFILLPAD'd `recvTile` back at the top slot, `0x20000`), then:

```bash
export PATH=/usr/local/python3.12.13/bin:$PATH
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

### 1.4 Exit criteria

- [ ] 1.1 A/B: pre-fix crashes, post-fix passes
- [ ] 1.2 pto-isa ST case 23: pre-fix FAIL, post-fix PASS
- [ ] 1.3 `fillpad` + `allfour` PASS at count ≥ 16384 with the workaround **reverted**
- [ ] Slot-order workaround reverted in both kernels, and both called out in the follow-up commit
- [ ] `pypto/runtime/pto_isa.pin` bumped `96ba706c` → ≥ `3b49066` (simpler's pin already has it)
- [ ] Comment the results on [pto-isa#291](https://github.com/hw-native-sys/pto-isa/issues/291) and
      close it; then re-measure `fillpad`/`allfour` at count ≥ 16384 (unblocks plan 107's allfour
      composition check)
