#!/usr/bin/env python3
"""K3 core_num (B) scaling analysis — managed-host all_to_all_v on EP4.

Reads ``summary.json`` (and each cell's harness JSON when present) produced by
``collectives/alltoallv_a1.py`` with ``A2AV_CORE_NUMS=1,2,4,8,16`` and prints
the tables needed for the K3 per-``B`` write-up:

  1. per-cell aggregates (slot mean/spread, egress GB/s, launched ``B``)
  2. Δ vs the ``L=1`` arm, per payload (uniform pattern)
  3. H1 readout: for ``B >= P`` the per-block work should fall ~1/K, so the
     whole-program slot time must be monotone non-increasing in ``B`` within
     the noise band (the fixed per-block cost is the competing term);
     reported per payload with the first violating step, if any
  4. the zero-counts control pattern (no TPUTs) as its own fixed-cost table

Usage:
    python analyze_scaling.py <campaign_dir> [--pattern uniform] [--noise 2.0]

``campaign_dir`` defaults to ``$A2AV_OUT`` or the newest
``reports/issue-2521-k3-*`` directory next to this repo.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def cal_all_to_all_v_blocks(p: int, l: int) -> int:  # noqa: E741
    """Entry-side admission: B = CalAllToAllVBlocks(P, L) = L < P ? L : (L // P) * P."""
    return l if l < p else (l // p) * p


def _mean(runs: list[dict], key: str) -> float | None:
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def _spread(runs: list[dict], key: str) -> str:
    vals = [r[key] for r in runs if r.get(key) is not None]
    return ", ".join(f"{v:.1f}" for v in vals)


def load_rows(campaign: Path, pattern: str | None) -> list[dict]:
    rows = json.loads((campaign / "summary.json").read_text())
    ok = [r for r in rows if r.get("ok")]
    if pattern is not None:
        ok = [r for r in ok if r.get("count_pattern") == pattern]
    return ok


def group(rows: list[dict]) -> dict[tuple[int, int], list[dict]]:
    by: dict[tuple[int, int], list[dict]] = {}
    for r in rows:
        by.setdefault((r["peer_bytes"], r["core_num"]), []).append(r)
    return by


def table_cells(by: dict[tuple[int, int], list[dict]], ep: int) -> None:
    print("## Per-cell aggregates")
    print()
    print("| payload B/peer | L | B (admitted) | runs | slot mean us (runs) | egress GB/s |")
    print("|---|---|---|---|---|---|")
    for (b, l), runs in sorted(by.items()):
        m = _mean(runs, "host_timing_slot_mean_us")
        e = _mean(runs, "egress_gbps")
        launched = runs[0].get("launched_B")
        if launched is None:
            launched = cal_all_to_all_v_blocks(ep, l)
        print(f"| {b} | {l} | {launched} | {len(runs)} | {m:.1f} ({_spread(runs, 'host_timing_slot_mean_us')}) | {e:.3f} |")
    print()


def table_deltas(by: dict[tuple[int, int], list[dict]], title: str = "") -> None:
    payloads = sorted({k[0] for k in by})
    arms = sorted({k[1] for k in by})
    ref = arms[0]
    print(f"## {title or 'Whole-program timing slot'} — Δ vs the L={ref} arm")
    print()
    header = "| payload B/peer | " + " | ".join(f"L={a} (us)" for a in arms) + " | " + " | ".join(f"Δ L={a}" for a in arms[1:]) + " |"
    print(header)
    print("|" + "---|" * (1 + len(arms) + len(arms) - 1))
    for b in payloads:
        r0 = by.get((b, ref))
        if not r0:
            continue
        m0 = statistics.mean([r["host_timing_slot_mean_us"] for r in r0])
        cells = [f"{m0:.1f}"]
        deltas = []
        for a in arms[1:]:
            ra = by.get((b, a))
            if not ra:
                cells.append("—")
                deltas.append("—")
                continue
            ma = statistics.mean([r["host_timing_slot_mean_us"] for r in ra])
            cells.append(f"{ma:.1f}")
            deltas.append(f"{100 * (ma - m0) / m0:+.2f}%")
        print(f"| {b} | " + " | ".join(cells) + " | " + " | ".join(deltas) + " |")
    print()


def h1_readout(by: dict[tuple[int, int], list[dict]], ep: int, noise_pct: float) -> None:
    """Monotone non-increasing slot time over the B >= P arms, within a noise band."""
    print(f"## H1 readout — monotone non-increasing slot time over B >= P arms (noise band ±{noise_pct:.1f}%)")
    print()
    for b in sorted({k[0] for k in by}):
        arms = sorted(k[1] for k in by if k[0] == b)
        seq: list[tuple[int, int, float]] = []
        for a in arms:
            launched = by[(b, a)][0].get("launched_B")
            if launched is None:
                launched = cal_all_to_all_v_blocks(ep, a)
            if launched < ep:  # stride regime: work is duplicated per block, not split
                continue
            seq.append((a, launched, statistics.mean([r["host_timing_slot_mean_us"] for r in by[(b, a)]])))
        if len(seq) < 2:
            continue
        verdict = "monotone non-increasing (within noise)"
        worst = 0.0
        for (a0, _b0, m0), (a1, _b1, m1) in zip(seq, seq[1:]):
            step = 100 * (m1 - m0) / m0
            if step > noise_pct:
                verdict = f"NON-MONOTONE at L={a0}→{a1}: {step:+.2f}% (exceeds +{noise_pct:.1f}%)"
                break
            worst = max(worst, step)
        print(f"- payload {b:>8}: " + ", ".join(f"B={bb}:{mm:.0f}us" for _a, bb, mm in seq) + f" → {verdict}" + ("" if "NON" in verdict else f" (worst step {worst:+.2f}%)"))
    print()


def pooled_round_stats(campaign: Path, rows: list[dict]) -> None:
    print("## Pooled per-round stats (slot samples, rank 0)")
    print()
    by = group(rows)
    for (b, l), runs in sorted(by.items()):
        samples: list[float] = []
        for r in runs:
            jp = r.get("json_path")
            if not jp or not Path(jp).is_file():
                continue
            doc = json.loads(Path(jp).read_text())
            samples.extend(float(v) for v in (doc.get("per_rank_kernel_us") or {}).get("0", []))
        if samples:
            sem = statistics.stdev(samples) / len(samples) ** 0.5
            print(f"- b={b:>8} L={l}: n={len(samples)} mean={statistics.mean(samples):.1f} sd={statistics.stdev(samples):.1f} sem={sem:.1f}")
    print()


def default_campaign() -> Path | None:
    cands = sorted(REPO.glob("reports/issue-2521-k3-*"))
    return cands[-1] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", nargs="?", type=Path, help="campaign directory (default: $A2AV_OUT or newest issue-2521-k3-*)")
    ap.add_argument("--pattern", default="uniform", help="count_pattern to analyze (default: uniform; 'all' for no filter)")
    ap.add_argument("--noise", type=float, default=2.0, help="monotonicity noise band in percent (default 2.0)")
    args = ap.parse_args()

    import os

    campaign = args.campaign or (Path(os.environ["A2AV_OUT"]) if os.environ.get("A2AV_OUT") else None) or default_campaign()
    if campaign is None or not (campaign / "summary.json").is_file():
        print(f"no summary.json found (campaign={campaign})")
        return 1

    pattern = None if args.pattern == "all" else args.pattern
    rows = load_rows(campaign, pattern)
    if not rows:
        print(f"no OK rows for pattern={args.pattern} in {campaign}")
        return 1
    ep = rows[0]["ep"]
    print(f"campaign: {campaign}  (ep={ep}, impl={rows[0]['impl']}, pattern={args.pattern})")
    print()
    by = group(rows)
    table_cells(by, ep)
    table_deltas(by)
    h1_readout(by, ep, args.noise)

    if pattern is None or pattern != "zero":
        zero = [r for r in json.loads((campaign / "summary.json").read_text()) if r.get("ok") and r.get("count_pattern") == "zero"]
        if zero:
            table_deltas(group(zero), title="Zero-counts control (no TPUTs — isolates the fixed per-block cost)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
