#!/usr/bin/env bash
# RFC #2521 — combined O2 §11.2 parity + A3 EP8 uniform sweep driver.
#
# Runnable record, acceptance criteria and result tables:
#   pr_plans/2521-sweep-npu-handoff.md      (NPU work index Job 7)
# Authoritative measurement specs:
#   pr_plans/129-...md § Measurement spec   (Part A)
#   pr_plans/116-...md § Measurement spec   (Part B)
#
# Usage (inside the container, through task-submit — never a bare run):
#   task-submit --device auto --device-num 8 --max-time 3600 --timeout 0 \
#     --run 'bash /mounted_home/run-2521-sweep.sh "$TASK_DEVICE" gate'
#
#   $1  device list   e.g. "$TASK_DEVICE" (single-quote it so it expands at RUN time)
#   $2  grant tag     gate | g1..g6 | all | list
#
# Env overrides:
#   PYPTO_ROOT    pypto checkout        (default: /opt/pypto, or $PWD if it holds the harness)
#   OUT_DIR       results root          (default: /mounted_home/a2av-sweep)
#   ROUNDS        timing-slot rounds    (default: 30 — the A1 protocol)
#   SWIM_ROUNDS   swimlane rounds       (default: 8)
#   DEADLINE_S    stop starting cells   (default: 3300; the queue kills at 3600)
#   ANCHORS       1=anchor each grant   (default: 1)
#   CODEGEN_ONLY  1=compile check       (writes to $OUT_DIR/codegen, needs no card)
#   DRY_RUN=1     print cells and exit
#
# Deliberately NOT `set -e`: one bad cell must not forfeit the rest of the grant.
set -uo pipefail

# The harness commit (#2945). It TRANSITIVELY contains eb28c0c9 (#3043) and
# e34c3c14 (#3051/K4) — the specs state this backwards and pin eb28c0c9, which
# would pass on a tree carrying no harness at all.
readonly TREE_PIN="a20bba866"
readonly BENCH_REL="tests/st/distributed/collectives/all_to_all_v_benchmark.py"

DEV="${1:-}"
TAG="${2:-}"
OUT_DIR="${OUT_DIR:-/mounted_home/a2av-sweep}"
ROUNDS="${ROUNDS:-30}"
SWIM_ROUNDS="${SWIM_ROUNDS:-8}"
DEADLINE_S="${DEADLINE_S:-3300}"
ANCHORS="${ANCHORS:-1}"
CODEGEN_ONLY="${CODEGEN_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
readonly CELL_RESERVE_S=320      # don't start a cell with less than this left

die() { echo "ERROR: $*" >&2; exit 2; }

[ -n "$DEV" ] && [ -n "$TAG" ] || die "usage: $0 <device-list> <gate|g1..g6|all|list>"
case "$TAG" in gate|g1|g2|g3|g4|g5|g6|all|list) ;; *) die "unknown grant tag '$TAG'" ;; esac

# --- locate the harness -------------------------------------------------------
PYPTO_ROOT="${PYPTO_ROOT:-}"
if [ -z "$PYPTO_ROOT" ]; then
  if [ -f "$PWD/$BENCH_REL" ]; then PYPTO_ROOT="$PWD"; else PYPTO_ROOT=/opt/pypto; fi
fi
BENCH="$PYPTO_ROOT/$BENCH_REL"
[ -f "$BENCH" ] || die "harness not found at $BENCH (set PYPTO_ROOT). It landed in #2945."

# --- tree pin -----------------------------------------------------------------
HEAD_SHA="$(git -C "$PYPTO_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
if git -C "$PYPTO_ROOT" merge-base --is-ancestor "$TREE_PIN" HEAD 2>/dev/null; then
  echo "tree pin OK: HEAD=$HEAD_SHA contains $TREE_PIN"
else
  die "HEAD=$HEAD_SHA does not contain the harness commit $TREE_PIN.
     Every cell must measure the same post-K4 tree, or the comparison is void."
fi

# Make the recorded commit an assertion, not an inference: read_pypto_commit()
# prefers this env var over walking up to .git — which would otherwise record
# the harness file's repo even when a different tree's pypto is imported.
export PYPTO_COMMIT="$HEAD_SHA"

RESULTS="$OUT_DIR"
[ "$CODEGEN_ONLY" = "1" ] && RESULTS="$OUT_DIR/codegen"   # never mix paper with real
INDEX="$RESULTS/cells.tsv"

# --- cell list ----------------------------------------------------------------
# One line per cell: impl L peer_bytes label
# Part A keeps each (l2, host) pair ADJACENT so slow drift cancels between rails.
pa_pair()   { echo "managed-l2 $1 $2 -"; echo "managed-host $1 $2 -"; }
cells_gate() { pa_pair 1 24960; }
cells_g1()  { pa_pair 8 24960; pa_pair 16 24960; echo "managed-l2 1 24960 repeat"; }
cells_g2()  { pa_pair 8 1048576; pa_pair 16 1048576
              local l; for l in 2 4 7 10; do echo "managed-l2 $l 24960 -"; done; }
cells_g3()  { echo "managed-l2 15 24960 -"
              local l; for l in 2 4 7 10 15; do echo "managed-l2 $l 1048576 -"; done
              for l in 1 2; do echo "managed-l2 $l 16384 -"; done; }
cells_g4()  { local l; for l in 4 7 8 10 15 16; do echo "managed-l2 $l 16384 -"; done
              for l in 1 2; do echo "managed-l2 $l 49152 -"; done; }
cells_g5()  { local l; for l in 4 7 8 10 15 16; do echo "managed-l2 $l 49152 -"; done
              for l in 1 2; do echo "managed-l2 $l 262144 -"; done; }
cells_g6()  { local l; for l in 4 7 8 10 15 16; do echo "managed-l2 $l 262144 -"; done; }
# Part A's L=1 @ 1 MiB pair lives in g1's tail so no pair is split across grants.
cells_g1x() { pa_pair 1 1048576; }

cell_list() {
  case "$1" in
    gate) cells_gate ;;
    g1)   cells_g1; cells_g1x ;;
    g2)   cells_g2 ;;
    g3)   cells_g3 ;;
    g4)   cells_g4 ;;
    g5)   cells_g5 ;;
    g6)   cells_g6 ;;
    all|list) cells_gate; cells_g1; cells_g1x; cells_g2; cells_g3; cells_g4; cells_g5; cells_g6 ;;
  esac
}

if [ "$TAG" = "list" ]; then
  cell_list all | nl -ba
  echo "--- $(cell_list all | wc -l) cells (46 + 1 deliberate stability repeat)"
  cell_list all | sort | uniq -d | grep -q . && echo "!! DUPLICATES ABOVE" || echo "--- no duplicates"
  exit 0
fi

mapfile -t CELLS < <(cell_list "$TAG")
# Anchor the grant head and tail with the same cheap cell: the only cross-grant
# comparability evidence there is, and it catches the thermal ramp that is
# otherwise confounded with the L-scaling curve.
if [ "$ANCHORS" = "1" ] && [ "$TAG" != "all" ] && [ "$CODEGEN_ONLY" != "1" ]; then
  CELLS=("managed-l2 1 24960 anchor-${TAG}-pre" "${CELLS[@]}" "managed-l2 1 24960 anchor-${TAG}-post")
fi

echo "grant '$TAG': ${#CELLS[@]} cells -> $RESULTS"
if [ "$DRY_RUN" = "1" ]; then
  printf '%s\n' "${CELLS[@]}" | nl -ba; echo "--- DRY_RUN=1, nothing run"; exit 0
fi

# --- anti-mixed-tree gate -----------------------------------------------------
# The container this was first written for had an editable install pointing at a
# FEATURE BRANCH worktree while the harness lived in /opt/pypto. Every cell would
# have recorded one tree's SHA while running another tree's code.
PYMOD="$(python -c 'import pypto,os;print(os.path.realpath(pypto.__file__))' 2>/dev/null || echo "")"
PYSO="$(python -c 'import pypto.pypto_core as m,os;print(os.path.realpath(m.__file__))' 2>/dev/null || echo "")"
ROOT_REAL="$(readlink -f "$PYPTO_ROOT")"
case "$PYMOD" in
  "$ROOT_REAL"/*) echo "import gate OK: $PYMOD" ;;
  "")             die "cannot import pypto — is the editable install built?" ;;
  *)              die "MIXED TREE: pypto imports from '$PYMOD', outside $ROOT_REAL.
     The harness would record $HEAD_SHA while executing a different tree.
     Fix: pip uninstall pypto (repeat until gone), unset PYTHONPATH, reinstall -e." ;;
esac
[ -n "$PYSO" ] && echo "compiled ext:  $PYSO"

# --- protocol: FORCE_IPC is NOT a guard --------------------------------------
# SIMPLER_COMM_FORCE_IPC does not exist in the runtime (verified at pin d8f997fa:
# zero getenv in comm_hccl.cpp). It is an UNMERGED patch in pypto-profiling. We
# still export it so a patched tree honours it, but it is recorded, not trusted.
export SIMPLER_COMM_FORCE_IPC=1
RT_SHA="$(git -C "$PYPTO_ROOT" rev-parse --short=12 HEAD:runtime 2>/dev/null || echo unknown)"
if grep -rqs 'SIMPLER_COMM_FORCE_IPC' "$PYPTO_ROOT/runtime/src" 2>/dev/null; then
  IPC_STATE="patched (runtime honours SIMPLER_COMM_FORCE_IPC)"
else
  IPC_STATE="UNPATCHED — env var is inert; A1 frozen protocol NOT reproduced"
fi
echo "force-ipc:     $IPC_STATE"

mkdir -p "$RESULTS" || die "cannot create $RESULTS"
[ -s "$INDEX" ] || printf 'impl\tL\tpeer_bytes\tlabel\trc\tverdict\tjson\n' > "$INDEX"

cat > "$RESULTS/manifest-$TAG.txt" <<EOF
rfc              2521 — O2 §11.2 (Part A) + A3 EP8 uniform (Part B)
grant_tag        $TAG
started          $(date -Is)
host             $(hostname)
pypto_head       $HEAD_SHA
runtime_pin      $RT_SHA
tree_pin         $TREE_PIN (ancestor: yes)
pypto_import     $PYMOD
compiled_ext     $PYSO
devices_logical  $DEV
devices_physical ${TASK_PHYS_DEVICE:-unknown}
force_ipc        $IPC_STATE
protocol         rounds=$ROUNDS warmup=5 profile=both swimlane-rounds=$SWIM_ROUNDS
count_pattern    uniform
codegen_only     $CODEGEN_ONLY
note             task id has no in-job source (no TASK_ID var) — record it by hand
EOF

# --- cell verdict -------------------------------------------------------------
# A cell is only "ok" if it is a COMPLETE result of the RIGHT KIND. The harness
# silently degrades to metric=end_to_end_latency when no AIV rounds are captured
# (exit 0, schema valid), which would mix two different physical quantities into
# one table. It also emits schema-valid compile-only stubs.
check_cell() {
  python3 - "$1" "$CODEGEN_ONLY" "$SWIM_ROUNDS" <<'PY'
import json, sys
path, codegen, swim = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3])
try:
    with open(path) as fh:
        d = json.load(fh)
except Exception as exc:                      # truncated by SIGKILL, or absent
    print(f"unreadable json ({type(exc).__name__})"); sys.exit(1)
metric = d.get("metric")
if codegen:
    print("ok (compile-only)" if metric == "compile-only" else f"expected compile-only, got {metric}")
    sys.exit(0 if metric == "compile-only" else 1)
if metric == "compile-only":
    print("compile-only stub in the real results dir"); sys.exit(1)
if metric != "per_rank_receive_bandwidth":
    print(f"DEGRADED metric={metric} (no AIV rounds captured)"); sys.exit(1)
if not any((d.get("per_rank_kernel_us") or {}).values()):
    print("empty per_rank_kernel_us"); sys.exit(1)
got = d.get("aiv_rounds_captured")
if got != swim:
    print(f"aiv_rounds_captured={got}, want {swim}"); sys.exit(1)
for k in ("aiv_ranks_missing", "aiv_rounds_incomplete"):
    if d.get(k):
        print(f"{k}={d[k]}"); sys.exit(1)
print("ok"); sys.exit(0)
PY
}

# --- run ----------------------------------------------------------------------
ok=0; failed=0; skipped=0; streak=0; deferred=0
declare -a FAILED_CELLS=()

for cell in "${CELLS[@]}"; do
  read -r impl L pb label <<<"$cell"
  sfx=""; [ "$label" != "-" ] && sfx="-$label"
  base="$RESULTS/${impl}-L${L}-PB${pb}${sfx}"
  json="$base.json"
  idx=$((ok + failed + skipped + deferred + 1))

  if [ -f "$json" ] && verdict="$(check_cell "$json")"; then
    echo "SKIP  [$idx/${#CELLS[@]}] $impl L=$L pb=$pb ${label#-} ($verdict)"
    skipped=$((skipped + 1)); continue
  fi
  rm -f "$json"      # stale/partial/degraded — never let it masquerade as done

  # Deadline: a clean stop beats a SIGKILL mid-write.
  if [ $((SECONDS + CELL_RESERVE_S)) -gt "$DEADLINE_S" ]; then
    echo "DEADLINE reached (${SECONDS}s) — not starting $impl L=$L pb=$pb"
    deferred=$((deferred + 1)); continue
  fi

  echo "=== [$idx/${#CELLS[@]}] $impl L=$L peer_bytes=$pb ${label#-}"
  if [ "$CODEGEN_ONLY" = "1" ]; then
    python "$BENCH" --ep 8 --peer-bytes "$pb" --count-pattern uniform \
      --core-num "$L" --impl "$impl" --codegen-only \
      --output-json "$json" --platform a2a3 --devices "$DEV" > "$base.log" 2>&1
  else
    python "$BENCH" --ep 8 --peer-bytes "$pb" --count-pattern uniform \
      --core-num "$L" --impl "$impl" \
      --rounds "$ROUNDS" --warmup 5 --profile both --swimlane-rounds "$SWIM_ROUNDS" \
      --output-json "$json" --platform a2a3 --devices "$DEV" > "$base.log" 2>&1
  fi
  rc=$?

  verdict="$(check_cell "$json")"; cell_ok=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$impl" "$L" "$pb" "${label#-}" "$rc" "$verdict" "$json" >> "$INDEX"

  if [ "$rc" -eq 0 ] && [ "$cell_ok" -eq 0 ]; then
    ok=$((ok + 1)); streak=0; echo "    ok"
  else
    failed=$((failed + 1)); streak=$((streak + 1))
    FAILED_CELLS+=("$impl L=$L pb=$pb rc=$rc — $verdict")
    rm -f "$json"
    echo "    FAILED rc=$rc ($verdict) — see $base.log"
    case "$rc" in
      139) echo "    rc=139 SIGSEGV: HCCL symbols NULL (libhcomm not NEEDED) — check readelf -d" ;;
      143) echo "    rc=143 SIGTERM: queue deadline hit" ;;
    esac
    grep -qs '507018' "$base.log" && echo "    Fabric fault 507018 — environmental; retry once"
    # Orphaned rank processes hold device contexts and fail every later cell.
    pkill -f 'all_to_all_v_benchmark' 2>/dev/null; sleep 2
    if [ "$streak" -ge 3 ]; then
      echo "ABORT: 3 consecutive failures — not burning the rest of the grant."; break
    fi
  fi
done

# --- summary ------------------------------------------------------------------
echo
echo "grant '$TAG': $ok ok, $failed failed, $skipped already done, $deferred deferred to deadline"
echo "results: $RESULTS   index: $INDEX"
[ "$deferred" -gt 0 ] && echo "Re-submit '$TAG' to pick up the deferred cells."
if [ "$failed" -gt 0 ]; then
  printf '  failed: %s\n' "${FAILED_CELLS[@]}"
  echo "Re-submit '$TAG' to retry only these — completed cells are skipped."
  exit 1
fi
echo "Next: fill the tables in pr_plans/2521-sweep-npu-handoff.md, then archive"
echo "$RESULTS -> pypto-profiling reports/issue-2521-o2-a3-$(date +%F)/"
