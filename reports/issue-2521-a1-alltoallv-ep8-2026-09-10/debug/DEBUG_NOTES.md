# EP8 failure debug notes

**Date:** 2026-09-10  
**Campaign:** `issue-2521-a1-alltoallv-ep8-2026-09-10`  
**Debug artifacts:** `debug/` under that report dir

## Symptoms (campaign)

| Observation | Detail |
|-------------|--------|
| Fail rate | ~50% of early `managed-l2` uniform points |
| Error | `release_domain` → `comm_release_domain_windows failed with code -1` → process `rc=-11` (SIGSEGV) |
| Compile | Always OK (build dirs + kernels present) |
| Result JSON | Absent on fail (crash before write) |
| Swimlane on fail | No `chip_swimlane_records.json` |
| Payload correlation | None (0/4096/16384 fail; 32/128/1024/8192 OK) |

## Controlled repro matrix

| Case | Result |
|------|--------|
| EP8 L2 4096, `--profile timing-slot`, rounds=2 | **OK** |
| EP8 L2 4096, `--profile swimlane`, rounds=2 | **OK** |
| EP8 L2 4096, `--profile both`, rounds=2 | **OK** |
| EP8 L2 1024, `--profile both`, rounds=2 | **OK** |
| EP4 L2 4096, `--profile both`, rounds=2 | **OK** |
| EP8 L2 4096, `--profile both`, rounds=100, warmup=5, swim=8 × **3 tries** | **OK ×3** |

**Conclusion so far:** the failure is **not** deterministic for peer_bytes=4096 or for `--profile both`. Isolated runs after a clean NPU state succeed. Campaign failures look like **intermittent teardown / HCCL domain-destroy flakes under sequential job churn**, possibly worsened by leftover state from a previous SIGSEGV.

## Swimlane rank skew (separate bug / measurement issue)

Even on successful EP8 runs, per-rank `__builtin_all_to_all_v__int8` **task durations** form a smooth rank gradient (not a collector mix-up with stage/consume):

Example (`debug/ep8_l2_4096_swim_only`, last dispatch `d5`):

| rank | builtin med (µs) | stage | consume |
|-----:|-----------------:|------:|--------:|
| 0 | 4061 | 5.8 | 4.1 |
| 1 | 3582 | 6.3 | 3.0 |
| 2 | 2924 | 6.0 | 3.2 |
| 3 | 2172 | 6.1 | 3.4 |
| 4 | 1256 | 6.5 | 4.1 |
| 5 | 1113 | 6.2 | 3.1 |
| 6 | 834 | 6.5 | 3.4 |
| 7 | **88** | 6.2 | 3.1 |

Stage/consume are fine (~3–6 µs). The **builtin task itself** is multi-ms on low ranks and ~90 µs on rank 7. Roadmap “fastest-rank” therefore picks rank 7 (~true gang completion for the last arriver’s wait-minimal view). Spread ~4 ms is **cross-rank sync wait inside the AIV task**, not wrong task naming.

## Hypotheses (ordered)

1. **H1 — Sequential campaign teardown race at EP8**  
   Back-to-back `prepare()`/release across processes leaves HCCL windows wedged; next job or current release then SIGSEGVs.  
   *Test:* rapid back-to-back jobs; add inter-job sleep / device reset.

2. **H2 — Dual `prepare()` in `--profile both` under long sessions**  
   Slot session release then swimlane session alloc/release; rare failure on first release.  
   *Evidence against:* short and 100-round isolated `both` all OK after clean NPUs.

3. **H3 — Rank-skew / long waits stress destroy**  
   Multi-ms blocked AIV tasks + 100 rounds leave ranks in uneven state at destroy.  
   *Test:* correlate fail with high host slot / hung rank (no JSON makes this hard).

4. **H4 — Unrelated box load / prior zombie workers**  
   Campaign kill left workers; subsequent jobs flaky until hard kill.  
   *Mitigation:* ensure clean `npu-smi` before each EP8 job.

## Next debug steps

1. Finish rapid back-to-back churn test (campaign simulation).
2. If H1 confirmed: insert cooldown / `npu-smi` drain between campaign jobs; optionally catch `release_domain` and retry once.
3. For metrics: keep fastest-rank AIV; document EP8 rank-skew as sync-wait, not parser bug.
4. Optional: log whether crash is on **first** or **second** `prepare()` exit when using `--profile both` (instrument harness).

## How to re-run debug matrix

```bash
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
# see commands in agent history / debug/ directories
```
