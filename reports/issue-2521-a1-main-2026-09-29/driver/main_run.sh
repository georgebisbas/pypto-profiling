#!/usr/bin/env bash
set -u
CANN=/usr/local/Ascend/cann-9.0.0
WT=/opt/a1/main
export A1_WORKTREE=$WT PYPTO_ROOT=$WT PYPTO_DIR=$WT PTOAS_ROOT=/opt/ptoas-bin
export PYPTO_COMMIT=$(git -C $WT rev-parse --short=12 HEAD)
export PYTHONPATH=/opt/a1/runenv:$WT/runtime/examples/scripts:$WT/runtime/python:$WT/runtime:$WT/python:$CANN/python/site-packages:$CANN/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe
export LD_PRELOAD=$CANN/aarch64-linux/lib64/libhccl.so
H=/opt/pypto-profiling/collectives/a1_alltoallv/harness/all_to_all_v_benchmark.py
echo "PYPTO_COMMIT=$PYPTO_COMMIT"
echo "=== pre-flight smoke (EP2, devices 4,5) ==="
timeout 420 python3 "$H" --smoke --ep 2 --peer-bytes 1024 --platform a2a3 --devices 4,5 --output-json /opt/a1/main_smoke.json --output-dir /opt/a1/main_smoke_build > /opt/a1/main_smoke.log 2>&1
rc=$?; echo "smoke rc=$rc"; tail -2 /opt/a1/main_smoke.log
python3 -c "import json; print('smoke commit stamp:', json.load(open('/opt/a1/main_smoke.json')).get('pypto_commit'))" 2>/dev/null || true
[ $rc -ne 0 ] && { echo "SMOKE FAILED"; exit 1; }
cd /opt/pypto-profiling
export A2AV_OUT=/opt/pypto-profiling/reports/issue-2521-a1-main-2026-09-29
export A2AV_EPS=8 A2AV_IMPLS=managed-l2,managed-host A2AV_PAYLOADS=8192,24960,32768,65536,1048576 A2AV_EXTRA_PATTERNS=
export A2AV_ROUNDS=30 A2AV_WARMUP=5 A2AV_SWIMLANE_ROUNDS=8 A2AV_RETRIES=4 A2AV_COOLDOWN_S=60 A2AV_HEAL_EP=4 A2AV_FORCE_IPC=1 A2AV_RESUME=1
echo "=== campaign start $(date -u) ==="
python3 -u collectives/alltoallv_a1.py 2>&1 | tee /opt/a1/main_run.log
echo "campaign rc=$?"
