#!/usr/bin/env bash
# Approximate re-run of #3051's double-buffer payload sweep on current main:
# managed-l2, core_num=16, 64..2048 KiB/peer, 5 warmup + 10 swimlane rounds,
# SIMPLER_COMM_FORCE_IPC unset. One grant; a failed cell does not stop the rest.
set -uo pipefail
DEV="$1"; OUT=/mounted_home/a2av-sweep/repro-3051
cd /opt/pypto
export PYPTO_COMMIT="$(git rev-parse HEAD)"
unset SIMPLER_COMM_FORCE_IPC
echo "kibs=${KIBS:-default} tree=$PYPTO_COMMIT runtime=$(git -C runtime rev-parse HEAD) phys=${TASK_PHYS_DEVICE:-?}" >> "$OUT/manifest.txt"
fail=0
for kib in ${KIBS:-64 128 256 512 1024 2048}; do
  pb=$((kib * 1024)); b="$OUT/managed-l2-L16-PB$pb"
  [ -f "$b.json" ] && python3 -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(d.get('metric')!='per_rank_receive_bandwidth' or d.get('aiv_rounds_captured')!=10)" "$b.json" \
    && { echo "SKIP $kib KiB (done)"; continue; }
  rm -f "$b.json"
  echo "=== $kib KiB/peer  $(date +%T)"
  python tests/st/distributed/collectives/all_to_all_v_benchmark.py --ep 8 --peer-bytes $pb \
    --count-pattern uniform --core-num 16 --impl managed-l2 \
    --profile swimlane --rounds 10 --warmup 5 \
    --output-json "$b.json" --platform a2a3 --devices "$DEV" > "$b.log" 2>&1
  rc=$?
  v=$(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
ok=d.get('metric')=='per_rank_receive_bandwidth' and d.get('aiv_rounds_captured')==10 and not d.get('aiv_ranks_missing') and not d.get('aiv_rounds_incomplete')
print(('ok' if ok else 'BAD'), d.get('metric'), 'aiv', d.get('aiv_rounds_captured'))" "$b.json" 2>&1 | tail -1)
  echo "    rc=$rc $v"
  [ $rc -eq 0 ] && [[ $v == ok* ]] || { fail=$((fail+1)); rm -f "$b.json"; pkill -f all_to_all_v_benchmark 2>/dev/null; sleep 2; }
done
echo "done failures=$fail"; exit $fail
