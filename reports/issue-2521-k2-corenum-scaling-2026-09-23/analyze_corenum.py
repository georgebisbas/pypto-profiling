#!/usr/bin/env python3
"""K2 core_num scaling analysis — managed-host all_to_all_v, EP4 on NPUs 4-7.

Reads ``summary.json`` (and each cell's harness JSON) produced by
``collectives/alltoallv_a1.py`` with ``A2AV_CORE_NUMS=1,8`` and prints the
tables used by K2_CORE_NUM_REPORT.md.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parent
rows = json.loads((R / "summary.json").read_text())
ok = [r for r in rows if r.get("ok")]

by: dict[tuple[int, int], list[dict]] = {}
for r in ok:
    by.setdefault((r["peer_bytes"], r["core_num"]), []).append(r)


def _mean(runs: list[dict], key: str) -> float | None:
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


print("## Per-cell aggregates")
print()
print("| payload B/peer | L | runs | slot mean us (runs) | egress GB/s |")
print("|---|---|---|---|---|")
for (b, l), runs in sorted(by.items()):
    m = _mean(runs, "host_timing_slot_mean_us")
    e = _mean(runs, "egress_gbps")
    spread = ", ".join(f"{r['host_timing_slot_mean_us']:.1f}" for r in runs)
    print(f"| {b} | {l} | {len(runs)} | {m:.1f} ({spread}) | {e:.3f} |")

print()
print("## L=1 vs L=8 deltas (whole-program timing slot)")
print()
print("| payload B/peer | L=1 mean us | L=8 mean us | delta us | delta % |")
print("|---|---|---|---|---|")
for b in sorted({k[0] for k in by}):
    r1, r8 = by.get((b, 1), []), by.get((b, 8), [])
    if not r1 or not r8:
        continue
    m1 = statistics.mean([r["host_timing_slot_mean_us"] for r in r1])
    m8 = statistics.mean([r["host_timing_slot_mean_us"] for r in r8])
    print(f"| {b} | {m1:.1f} | {m8:.1f} | {m8 - m1:+.1f} | {100 * (m8 - m1) / m1:+.2f}% |")

print()
print("## Pooled per-round stats (slot samples, rank 0)")
print()
for b in sorted({k[0] for k in by}):
    for l in (1, 8):
        samples: list[float] = []
        for r in by.get((b, l), []):
            doc = json.loads(Path(r["json_path"]).read_text())
            samples.extend(float(v) for v in (doc.get("per_rank_kernel_us") or {}).get("0", []))
        if samples:
            sem = statistics.stdev(samples) / len(samples) ** 0.5
            print(
                f"- b={b:>8} L={l}: n={len(samples)} mean={statistics.mean(samples):.1f} "
                f"sd={statistics.stdev(samples):.1f} sem={sem:.1f}"
            )
