#!/usr/bin/env bash
# Phase 2: compile-only check of the L values #2945 newly unlocks (no card).
set -uo pipefail
cd /opt/pypto
OUT=/mounted_home/a2av-sweep/codegen; fail=0
for pb in 24960 16384; do for L in 2 4 7 10 15; do
  b="$OUT/managed-l2-L$L-PB$pb"
  python tests/st/distributed/collectives/all_to_all_v_benchmark.py --ep 8 --peer-bytes $pb \
    --count-pattern uniform --core-num $L --impl managed-l2 --codegen-only \
    --output-json "$b.json" --platform a2a3 --devices 0,1,2,3,4,5,6,7 > "$b.log" 2>&1
  rc=$?; m=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('metric'))" "$b.json" 2>/dev/null)
  echo "L=$L pb=$pb rc=$rc metric=$m"; { [ $rc -eq 0 ] && [ "$m" = compile-only ]; } || fail=$((fail+1))
done; done
echo "phase2 failures=$fail"; exit $fail
