#!/usr/bin/env python3
"""Compare the A/B variants of the resident-args experiment.

Reads `baseline/summary.json` and `resident/summary.json` and prints the slot-kernel
side by side plus the per-span breakdown, so the win (or lack of one) is attributable
rather than just a number that moved.

Usage: python3 compare.py [campaign_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SPANS = (
    "chip.run.stage_inputs",
    "chip.run.validate",
    "chip.run.runner_run.device_wall",
    "chip.run.prepare_execution",
    "chip.run.publish_image",
    "chip.run.bind",
    "chip.run",
)

# What each span is (from GAP_CLOSING_NOTES.md §Step 0) — the point of the table is to
# show which of these the resident path removes.
SPAN_MEANING = {
    "chip.run.stage_inputs": "H2D of the run's inputs",
    "chip.run.validate": "D2H copy-back of outputs (misleading name)",
    "chip.run.runner_run.device_wall": "actual device execution",
    "chip.run.prepare_execution": "dispatch plumbing",
    "chip.run.publish_image": "dispatch plumbing",
    "chip.run.bind": "dispatch plumbing",
    "chip.run": "whole chip.run span",
}


def load_rows(campaign: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for variant in ("baseline", "resident"):
        path = campaign / variant / "summary.json"
        if not path.is_file():
            out[variant] = {}
            continue
        data = json.loads(path.read_text())
        # summary.json is the campaign's row list (one dict per cell, `ok` False for failures).
        raw = data.get("rows", []) if isinstance(data, dict) else data
        rows = [r for r in raw if r.get("ok")]
        out[variant] = rows[0] if rows else {}
    return out


def fmt(value: object, spec: str = ".0f", suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:{spec}}{suffix}"
    return str(value)


def main() -> int:
    campaign = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    rows = load_rows(campaign)

    if not rows["baseline"] or not rows["resident"]:
        missing = [v for v in rows if not rows[v]]
        print(f"missing a completed variant: {', '.join(missing) or 'both'}")
        for variant, row in rows.items():
            print(f"  {variant}: {'ok' if row else 'no ok row'}")
        return 1

    base, res = rows["baseline"], rows["resident"]

    print("=" * 78)
    print("A/B: host-tensor args  vs  --resident-args")
    print(f"cell: {base.get('tag')}  (EP{base.get('ep')}, {base.get('impl')}, "
          f"L={base.get('core_num')}, {base.get('peer_bytes')} B/peer, {base.get('count_pattern')})")
    print("=" * 78)

    print("\n-- headline --")
    header = f"{'metric':<34}{'baseline':>14}{'resident':>14}{'delta':>14}"
    print(header)
    print("-" * len(header))
    for label, key, spec, unit in (
        ("host slot p50 (us)", "host_timing_slot_p50_us", ".0f", ""),
        ("host slot mean (us)", "host_timing_slot_mean_us", ".0f", ""),
        ("kernel p50 (us)", "fastest_p50_us", ".0f", ""),
        ("kernel mean (us)", "fastest_mean_us", ".0f", ""),
        ("egress (GB/s)", "egress_gbps", ".2f", ""),
        ("rank spread (us)", "rank_spread_us", ".0f", ""),
    ):
        b, r = base.get(key), res.get(key)
        delta = f"{r - b:+.0f}" if isinstance(b, (int, float)) and isinstance(r, (int, float)) else "—"
        ratio = ""
        if isinstance(b, (int, float)) and isinstance(r, (int, float)) and r:
            ratio = f"  ({b / r:.2f}x)"
        print(f"{label:<34}{fmt(b, spec):>14}{fmt(r, spec):>14}{delta + ratio:>14}")

    print("\n-- per-span breakdown (median us per run, from the attempt logs) --")
    header = f"{'span':<36}{'baseline':>12}{'resident':>12}{'delta':>12}"
    print(header)
    print("-" * len(header))
    base_bd = base.get("breakdown_us") or {}
    res_bd = res.get("breakdown_us") or {}
    for span in SPANS:
        b = (base_bd.get(span) or {}).get("median_us")
        r = (res_bd.get(span) or {}).get("median_us")
        delta = f"{r - b:+.0f}" if isinstance(b, (int, float)) and isinstance(r, (int, float)) else "—"
        print(f"{span:<36}{fmt(b):>12}{fmt(r):>12}{delta:>12}")

    print("\n-- what each span is --")
    for span in SPANS:
        print(f"  {span:<36} {SPAN_MEANING.get(span, '')}")

    # The two spans the experiment exists to delete.
    print("\n-- where the removed slot time went --")
    for span, label in (
        ("chip.run.stage_inputs", "H2D of inputs"),
        ("chip.run.validate", "D2H copy-back"),
        ("chip.run.runner_run.device_wall", "device wall (bonus)"),
    ):
        b = (base_bd.get(span) or {}).get("median_us")
        r = (res_bd.get(span) or {}).get("median_us")
        if isinstance(b, (int, float)) and isinstance(r, (int, float)):
            print(f"  {label:<24}{b:>9.0f} -> {r:>7.0f}   {r - b:>+8.0f}")

    bslot, rslot = base.get("host_timing_slot_p50_us"), res.get("host_timing_slot_p50_us")
    if isinstance(bslot, (int, float)) and isinstance(rslot, (int, float)) and bslot:
        removed = bslot - rslot
        accounted = 0.0
        for span in ("chip.run.stage_inputs", "chip.run.validate",
                     "chip.run.runner_run.device_wall"):
            b = (base_bd.get(span) or {}).get("median_us")
            r = (res_bd.get(span) or {}).get("median_us")
            if isinstance(b, (int, float)) and isinstance(r, (int, float)):
                accounted += b - r
        print(f"\n  slot: {bslot:.0f}us -> {rslot:.0f}us  ({bslot / rslot:.2f}x, "
              f"removed {removed:.0f}us)")
        print(f"  accounted by the three spans: {accounted:.0f}us "
              f"({accounted / removed * 100:.0f}% of the removal)")
        print(f"  remainder (outside chip.run, e.g. host-side arg handling): "
              f"{removed - accounted:+.0f}us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
