# pypto-profiling — agent entry point

Read [`AGENTS.md`](AGENTS.md) before running anything. This file exists so an agent that
auto-loads `CLAUDE.md` sees the one rule that must never be missed.

## Hardware runs are queued — `task-submit` is mandatory

On `hng-atlas01` (192.168.150.11) and `hng-atlas02` (192.168.150.12), **every card-touching
command must be submitted through `task-submit`**. A direct `python3 -m collectives...` or
`pytest ... --platform a2a3` does not hold a card lock and will collide with other users. Do not
pick a card from `npu-smi` — a card you did not lock is not a card you hold.

### Preflight (inside the `pypto3-hw-native-sys:cann9` container)

```bash
export PATH="$PATH:/var/lib/taskqueue/bin"
export TASKQUEUE_CONF=/var/lib/taskqueue/taskqueue.conf
task-submit --list                 # must list free cards; if it fails, STOP and report
```

### Submit

```bash
task-submit --device auto --device-num 2 --max-time 3600 --timeout 0 \
  --run 'cd /mounted_home/hw-native-sys/pypto-profiling && export PYTHONPATH=. && \
         python3 -m collectives.run_sweep pair-mesh \
           --case-file collectives/cases/mesh_p2_count65536_fp32_a2a3_d0-1.json \
           --stacks hccl,pypto-composite,pypto-host --persistent \
           --timed-rounds 5 --warmup-rounds 2 \
           --campaign demo --out results/campaigns/demo/run_001/results.json'
```

- `--device-num` is the rank count `P`; inside the job the granted cards are logical `0..N-1`
  (`$TASK_DEVICE`), which is what the case files use.
- `--max-time` is the **kill deadline** (hard cap 3600 s). Split a campaign by `(P, count)` or by
  stack; never raise it. `--timeout 0` only controls how long the client waits.
- Record the returned task id in the report/commit — it is the proof the run held a lock.
- Set `LD_PRELOAD` for the `hccl` stack **inside** the `--run` payload, never in the client shell
  before `task-submit`.
- Never nest `task-submit`; never start the broker.

### Exceptions

- `192.168.150.13` has no queue; unqueued runs there must be recorded as unqueued.
- Simulator work (`--platform a2a3sim`, image `pypto3-hw-native-sys:sim`) touches no card and
  needs no queue.
- Unit tests (`pytest tests/`) touch no card and need no queue.

Full recipes, pitfalls, and provenance rules: [`AGENTS.md`](AGENTS.md) §2.1 and
[`reports/npu-queue-runbook.md`](reports/npu-queue-runbook.md).
