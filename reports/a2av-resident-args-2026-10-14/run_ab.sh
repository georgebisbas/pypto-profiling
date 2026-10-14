#!/bin/bash
# A/B: host-tensor args (baseline) vs worker-resident args (`--resident-args`).
#
# Same binary, same cell (EP8 / managed-l2 / L=8 / 1 MiB / uniform / 20 rounds), same box,
# back-to-back inside ONE queue grant so both variants see identical conditions.
# Variant B differs only by `A2AV_EXTRA_HARNESS_ARGS=--resident-args`, which the harness turns
# into `alloc_stacked_tensor` args dispatched in place of the host tensors.
#
# Run INSIDE a queue-joined container: task-submit --device auto --device-num 8 --run 'bash …'
set -u

REPO=/mounted_home/hw-native-sys/pypto-profiling
OUT=$REPO/reports/a2av-resident-args-2026-10-14

cd "$REPO" || exit 1

# Queue trap 9: the default ~/ascend/log/debug/device-<id>/ is shared by the whole box.
export ASCEND_PROCESS_LOG_PATH=$OUT/device_logs
mkdir -p "$ASCEND_PROCESS_LOG_PATH"

# The in-tree harness on the container's live pypto (has the --resident-args experiment).
export A2AV_HARNESS=/opt/pypto/tests/st/distributed/collectives/all_to_all_v_benchmark.py
export A2AV_EPS=8
export A2AV_IMPLS=managed-l2
export A2AV_CORE_NUMS=8
export A2AV_PAYLOADS=1048576
export A2AV_ROUNDS=20
export A2AV_WARMUP=3
export A2AV_SWIMLANE_ROUNDS=4
export A2AV_EXTRA_PATTERNS=          # uniform only -> one cell per variant
export A2AV_RETRIES=1
export A2AV_COOLDOWN_S=30
export A2AV_RESUME=1

echo "TASK_DEVICE=${TASK_DEVICE:-<unset>}"

echo "########## VARIANT A: baseline (host tensors, one H2D+D2H per round) ##########"
A2AV_OUT=$OUT/baseline A2AV_EXTRA_HARNESS_ARGS= PYTHONPATH=. \
  python3 -u collectives/alltoallv_a1.py 2>&1 | tee "$OUT/variant_a.log"
echo "A_rc=${PIPESTATUS[0]}"

echo "########## VARIANT B: --resident-args (DeviceTensor args, uploaded once) ##########"
A2AV_OUT=$OUT/resident A2AV_EXTRA_HARNESS_ARGS="--resident-args" PYTHONPATH=. \
  python3 -u collectives/alltoallv_a1.py 2>&1 | tee "$OUT/variant_b.log"
echo "B_rc=${PIPESTATUS[0]}"

echo "########## A/B DONE ##########"
