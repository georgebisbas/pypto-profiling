# NPU queue runbook — how to run this harness on a shared box

Companion to [`onbox-todo.md`](onbox-todo.md) (what is owed) and [`../AGENTS.md`](../AGENTS.md) §2.1
(the short version). This file is the *how*: how to get a card, how to run a campaign on it, and how
to record the result so someone else can tell the run was isolated.

Scope: `192.168.150.11` (`hng-atlas01`) and `192.168.150.12` (`hng-atlas02`). `192.168.150.13`
(`hng-atlas03`) has no queue — direct runs there are unqueued and must be recorded as such. **Sim
work needs none of this** (the `pypto3-hw-native-sys:sim` image touches no card).

---

## 1. Why: the lock, not the smi

The NPU hosts are shared. `task-submit` grants a free card, takes an exclusive `flock` on it for the
life of the job, and releases it on exit. **A card you picked by reading `npu-smi` is not a card you
hold** — the next user's `--device auto` can land on it mid-run and both runs become uninterpretable.

That matters here more than usual: this repo's headline metric is `device_wall_s` on a
slowest-rank span, and its second metric `execute_spread_ratio` is the flag for exactly the
contention the lock removes. An unqueued run on this box has no claim to either.

**If `task-submit --list` fails, your container is not on the queue: stop and report. Do not fall
back to a direct run.**

---

## 2. Where `task-submit` actually is

| | |
|---|---|
| The CLI | `/var/lib/taskqueue/bin/task-submit` (plus `npu-lock`, `task-broker`) |
| How it is distributed | a **shared volume**, bind-mounted at the same path on host and containers |
| In the CANN images? | **No** — this is normal; the image ships no queue client |
| In `pypto-tooling/task-submit/`? | **No** — that path holds one conceptual doc (`task-submit.md`). It is not a script and it is not the CLI. |
| Client config | `/var/lib/taskqueue/taskqueue.conf` (the client default is `/etc/taskqueue.conf`) |
| Card whitelist (client) | `/var/lib/taskqueue/available_devices` — observed `0,1,2,3,4,5,6,7` on 2026-10-14 |
| Card whitelist (broker) | `TASKQUEUE_WHITELIST` env in the `tq-broker` container = `0,1,2,3,4,5,6,7`; when set, the broker ignores the shared file |

> The `.12` admin note (`pypto-docker/TASK_QUEUE_hng-atlas02_admin-note.md`, 2026-09-28) records the
> historical state where the **client** file still read `0,1,2,3`, so `--device 4..7` was rejected
> client-side with *"card(s) N are not in your reserved set"* while the broker already permitted all
> eight. That has since been fixed: `--list` now reports all 8 free. If you see `(x/4)` again, the
> file has been reverted and widening it is a **root/admin action, not an agent step**.

---

## 3. Join a shell or container

Host shell (or any already-joined container):

```bash
export PATH="$PATH:/var/lib/taskqueue/bin"
export TASKQUEUE_CONF=/var/lib/taskqueue/taskqueue.conf
task-submit --list
```

A container is joined only if it bind-mounts the queue volume **at the identical path** (a
bind-mount cannot be added to a live container):

```bash
docker run -d --name pypto-dev-queue \
  --privileged --ipc=host --pid=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /dev:/dev \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /var/lib/taskqueue:/var/lib/taskqueue \
  -v "$HOME:/mounted_home" \
  pypto3-hw-native-sys:cann9 sleep infinity

pypto-docker/scripts/attach-taskqueue.sh pypto-dev-queue
```

`--pid=host` is required for HCCL (Ascend IPC validates host PIDs); `pypto-docker/scripts/queue-smoke.sh`
is the one-case end-to-end check.

> **The payload runs in the caller's context.** `task-submit` executes the command *where you call
> it from* — it does not reach into other containers. Call it **inside** the container that has the
> runtime you need. Calling it from the host and putting `docker exec <container> …` in `--run`
> does **not** work: the task is killed immediately as `completed (exit=137)` with no log
> (observed 2026-10-14 on `hng-atlas02`). `--list`/`--devices` are fine from the host.

---

## 4. The three shapes

```bash
# 1. N cards, watch live (Ctrl+C kills the job; Ctrl+Z detaches and it keeps running)
task-submit --device auto --device-num 2 --max-time 3600 --timeout 0 \
  --run 'cd /path/to/pypto-profiling && export PYTHONPATH=. && python3 -m collectives.run_sweep ...'

# 2. no card at all — build, lint, codegen diff, figures, .so swap
task-submit --no-device --max-time 3600 --run 'cd pypto-profiling && python3 -m collectives.summarize --run-dir ...'

# 3. detached, reconnect later
id=$(task-submit --device auto --device-num 2 --max-time 3600 --run '...')
task-submit --watch  "$id"
task-submit --status "$id"      # pending / running / completed(exit=N)
task-submit --log    "$id"
task-submit --kill   "$id"      # running task
task-submit --cancel "$id"      # still-queued task

# hygiene
task-submit --list
task-submit --devices
task-submit --clean [--days N] [--done-days M]   # prunes EVERYONE's logs — coordinate
```

---

## 5. Device ids: logical vs physical

`task-submit` pins the granted card with `ASCEND_RT_VISIBLE_DEVICES`, which the CANN runtime
**re-indexes from 0**. So inside your job the card is always `0` (or `0,1,…`) — never its physical
number.

| Use | Value |
|---|---|
| `$TASK_DEVICE` | the **logical** id — always `0`, or `0,1,…` for a multi-card grant |
| `$TASK_PHYS_DEVICE` | the physical card the grant came from |
| `ASCEND_RT_VISIBLE_DEVICES` | set **by the queue**; never set, unset or override it yourself |

This is why the harness recipes below work unchanged: a `--device-num 2` grant appears to the job as
`d0-1`, so the checked-in case files (which pin `d0-1`) are already correct inside the grant.

---

## 6. Timeouts — split, don't stretch

| Flag | Meaning | Default |
|---|---|---|
| `--max-time N` | **kill deadline.** Hard cap **3600 s**. | 300 s |
| `--timeout N` | how long the *client* waits before detaching (`0` = wait forever). Does **not** stop the job. | 600 s |

`--timeout` alone does not extend run time: a job with `--timeout 1800` and no `--max-time` is killed
at **300 s**. Do not raise the cap — a hung job holding cards for over an hour is the failure mode the
queue exists to bound. Instead split a campaign into sub-hour units: per (P, count), per stack, or
per payload band.

---

## 7. Traps

1. **Never `export LD_PRELOAD=libhccl.so` in the client shell** before `task-submit` — the task dies
   instantly as `completed (exit=137)` with no log. Scope it *inside* the `--run` string.
2. **Auto-appended `--device`.** With `--device auto`, the client appends `--device <logical>` to a
   command that does not already mention the device. Always reference `$TASK_DEVICE` (or `{}`, a
   `--device` token, `-d N`) in the payload.
3. **`set -o pipefail` + an early-exit pipeline** (e.g. `npu-smi info | head`) → `exit=141`. Capture
   to a file first.
4. **Wrong `--platform` is deselection, not failure** — pytest exits **5** with "N deselected". Check
   the `PASSED` count, never the exit code alone.
5. **No nesting.** `task-submit` inside a submitted task is rejected — wrap the script, not its loop.
6. **`--device N` outside your reserved set is rejected.** Use `--device auto`.
7. **Never run `task-broker`.** One broker serves the node; a second double-grants cards. If
   `--list` errors, tell the admin.
8. **`--device` still defaults to `0` in pytest**, so a multi-rank ST can silently run 1 rank and
   exit 0. Always pass `--device=$TASK_DEVICE`.
9. **Isolate device logs.** The default `~/ascend/log/debug/device-<id>/` is shared by every user and
   process on the box; set `ASCEND_PROCESS_LOG_PATH` (and create the directory) before the run.
10. **`docker exec` inside `--run` dies.** The payload runs in the caller's context, so from the
    host a `--run 'docker exec …'` payload is killed instantly as `exit=137` with no log — a
    *different* cause from trap 1's `LD_PRELOAD` 137. Call `task-submit` from inside the target
    container instead.

---

## 8. Recipes for this repo

Every command in [`../AGENTS.md`](../AGENTS.md) §3 and every entry in
[`onbox-todo.md`](onbox-todo.md) is a **payload**: wrap it.

Strong-scaling campaign (the case file pins the logical pair, so a 2-card grant maps onto it):

```bash
task-submit --device auto --device-num 2 --max-time 3600 --timeout 0 \
  --run 'cd /path/to/pypto-profiling && export PYTHONPATH=. && \
    bash run_campaign.sh --variant mesh --p-values 2 --count 65536 \
      --stacks hccl,pypto-composite,pypto-host --persistent \
      --warmup-rounds 2 --timed-rounds 8 --campaign strong'
```

Sweep longer than an hour → submit one unit per P (or per count band) rather than one long job.

A1 AllToAllV campaign driver (payloads are documented in
`reports/issue-2521-k3-repro-2026-09-28/README.md`):

```bash
task-submit --device auto --device-num 2 --max-time 3600 --timeout 0 \
  --run 'cd /path/to/pypto-profiling && export PYTHONPATH=. && \
    A2AV_EPS=2 A2AV_IMPLS=managed-l2,managed-host \
    python3 -u collectives/alltoallv_a1.py'
```

Notes specific to this harness:

- The A1 harness already defaults its device from the environment
  (`TASK_DEVICE` → `PYPTO_DEVICES`; `collectives/a1_alltoallv/harness/all_to_all_v_benchmark.py`), so
  it needs no extra plumbing.
- **Post-processing does not need a card.** `summarize.py`, `plot_figures.py` and
  `apples_to_apples.py` read `results.json`; run them under `--no-device` and give the card back
  sooner.
- The harness **box-health probe** opens a real CommDomain before a campaign and aborts with
  "passing box condition" when the box is unusable. That is the harness working as designed. Under
  the queue it means "your granted card is bad" — retry, and if it persists report the physical id
  from `$TASK_PHYS_DEVICE`.
- Keep figures/reports out of the card window: write `results/campaigns/...` inside the run, plot
  afterwards.

---

## 9. Recording a run

A **task id** is the auditable proof the run held a lock. Put it next to the commit in the report's
results block:

```text
Verified commit: <full-sha>
Host:            192.168.150.11 | 192.168.150.12
Task id(s):      <from task-submit>
Commands:        <paste the wrapped invocation>
Result:          PASS | FAIL (with log excerpt)
```

Unqueued runs (`.13`, or a container that could not join) must be **labelled as unqueued** in the
same block.

---

## 10. Provenance of the existing campaigns

Every campaign under `reports/` up to 2026-10-14 was run **direct, without the queue** — the queue
was not provisioned in those containers or was unavailable by instruction. The record says so in
several places:

- `reports/benchmark-report-2026-09-02-pr2591-final.md` — *"task-submit was requested but is **not
  provisioned in this container** … Runs were **direct**"*
- `reports/issue-2521-k2-corenum-scaling-2026-09-23/K2_CORE_NUM_REPORT.md` — *"Runs were
  **unisolated** … within-run per-round spread reaches ±20 %"*
- `reports/plan101-phase0-reset-cost-and-l2-crash-2026-09-03.md` — *"direct runs on this dev box
  (task-submit/taskqueue unavailable by owner instruction)"*
- `reports/issue-2521-a1-alltoallv-ubfix-2026-09-09/ANALYTIC_REPORT_FOR_COLLEAGUES.md` — *"8× 910B2,
  no `task-submit`"*

**Consequence for reading those numbers:** `execute_spread_ratio` and the ±20 % spreads are shared-box
noise, not stack-intrinsic variance, and `setup_s` / `execute_s` were measured with no isolation
guarantee. Those reports are **records and are not rewritten** (rewriting them would falsify the
record) — this section is the single shared caveat they all inherit. New numbers should be produced
under the queue.

---

## 11. Canonical sources

| What | Where |
|---|---|
| Box etiquette, recipes, traps | `pypto-3.0-notes/tools/npu-task-submit-runbook.md` |
| Conceptual explainer + `507018` triage | `pypto-3.0-notes/tools/task-submit.md` |
| Container/host recipes, pitfalls P1–P7 | `pypto-docker/TASK_QUEUE.md` |
| `.12` whitelist + health-probe admin note | `pypto-docker/TASK_QUEUE_hng-atlas02_admin-note.md` |
| Conceptual doc (partly stale — see its banner) | `pypto-tooling/task-submit/task-submit.md` |
| Hardware job index (upstream track) | `pypto-3.0-notes/pr_plans/00-npu-work-index.md` |

*Last verified: 2026-10-14 on `hng-atlas02` (192.168.150.12) — `task-submit --list` reported
`(0/8) all free [0,1,2,3,4,5,6,7]`.*
