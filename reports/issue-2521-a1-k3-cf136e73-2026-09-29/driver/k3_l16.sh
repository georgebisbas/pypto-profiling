#!/usr/bin/env bash
# L=16 (B=16) spot cells on K3 cf136e73 (multiaiv lanes), managed-host rail — same as the main snapshot spots.
set -u
CANN=/usr/local/Ascend/cann-9.0.0
WT=/opt/a1/k3
export A1_WORKTREE=$WT PYPTO_ROOT=$WT PYPTO_DIR=$WT PTOAS_ROOT=/opt/ptoas-bin
export PYPTO_COMMIT=$(git -C $WT rev-parse --short=12 HEAD)
export PYTHONPATH=/opt/a1/runenv:$WT/runtime/examples/scripts:$WT/runtime/python:$WT/runtime:$WT/python:$CANN/python/site-packages:$CANN/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe
export LD_PRELOAD=$CANN/aarch64-linux/lib64/libhccl.so
export SIMPLER_COMM_FORCE_IPC=1
H=/opt/pypto-profiling/collectives/a1_alltoallv/harness/all_to_all_v_benchmark.py
D=/opt/pypto-profiling/reports/issue-2521-a1-k3-cf136e73-2026-09-29
run_cell() {
  pb=$1
  att=1
  while [ $att -le 3 ]; do
    echo "=== L16 managed-host uniform_${pb} attempt ${att} $(date -u +%H:%M:%S) ==="
    timeout 700 python3 $H --ep 8 --peer-bytes $pb --count-pattern uniform --core-num 16 --impl managed-host --rounds 30 --warmup 5 --profile timing-slot --platform a2a3 --devices 0,1,2,3,4,5,6,7 --output-json $D/json/p8_l16_managed-host_uniform_${pb}.json --output-dir $D/builds/p8_l16_managed-host_uniform_${pb} >> /opt/a1/k3_l16.log 2>&1
    rc=$?
    echo "rc=$rc"
    if [ $rc -eq 0 ] && [ -f $D/json/p8_l16_managed-host_uniform_${pb}.json ]; then
      python3 -c "import json;d=json.load(open('$D/json/p8_l16_managed-host_uniform_${pb}.json'));print('OK', d['metric'], round(d['fastest_p50_us'],1))" | tee -a /opt/a1/k3_l16.log
      return 0
    fi
    echo "[heal] EP4 smoke" | tee -a /opt/a1/k3_l16.log
    timeout 300 python3 $H --ep 4 --peer-bytes 1024 --count-pattern uniform --core-num 1 --impl managed-l2 --rounds 2 --warmup 1 --profile timing-slot --platform a2a3 --devices 0,1,2,3 --output-json /opt/a1/k3_heal.json --output-dir /opt/a1/k3_heal_build >> /opt/a1/k3_l16.log 2>&1
    echo "[heal] rc=$?" | tee -a /opt/a1/k3_l16.log
    sleep 30
    att=$((att + 1))
  done
  return 1
}
echo "start $(date -u)" | tee -a /opt/a1/k3_l16.log
run_cell 24960
run_cell 1048576
echo "done $(date -u)" | tee -a /opt/a1/k3_l16.log
