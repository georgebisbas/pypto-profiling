#!/usr/bin/env python3
"""HCCL `HcclAlltoAllV` vs PyPTO `all_to_all_v` — apples-to-apples figures.

The only column both stacks can be measured on with the current tooling is the
**host-side wall around one collective** (PyPTO's `host_timing_slot_p50_us`, the
slowest-rank `aclrtSynchronizeStream`-bounded wall for HCCL). PyPTO additionally
exposes a device span (the AIV gang span) that HCCL does not, so that column is
shown but never used for the ratio.

Usage::

    python3 collectives/plot_hccl_vs_pypto_a2av.py \
        --campaign reports/issue-2521-a1-ep8-l2corenum-2026-10-14 \
        --hccl-log reports/issue-2521-a1-ep8-l2corenum-2026-10-14/hccl_a2av_ep8_65536.log \
        --out reports/figures-2026-10-14
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_hccl_log(path: Path) -> dict:
    text = path.read_text()
    rows = [
        [float(x) for x in m.split(",")]
        for m in re.findall(r"HCCL_TIMED round=\d+ per_rank=\[([^\]]+)\]", text)
    ]
    if not rows:
        raise SystemExit(f"no HCCL_TIMED rows in {path}")
    world, rounds = len(rows[0]), len(rows)
    per_rank = [[rows[r][i] for r in range(rounds)] for i in range(world)]
    p50_us = [st.median(v) * 1e6 for v in per_rank]
    means_us = [st.mean(v) * 1e6 for v in per_rank]
    return {
        "label": "HCCL HcclAlltoAllV",
        "world": world,
        "rounds": rounds,
        "slot_p50_us": st.median(p50_us),          # median rank of per-rank p50
        "slot_slowest_rank_us": max(means_us),
        "device_us": None,                          # no device span exposed
        "remote_bytes": (world - 1) * 65536,
    }


def load_pypto_cells(campaign: Path) -> list[dict]:
    out = []
    for f in sorted((campaign / "json").glob("*.json")):
        d = json.loads(f.read_text())
        out.append(
            {
                "label": f"{d['impl']} L={d['requested_L']}",
                "world": d["P"],
                "rounds": d["rounds"],
                "slot_p50_us": d["host_timing_slot_p50_us"],
                "slot_slowest_rank_us": d["slowest_mean_us"],
                "device_us": d["fastest_p50_us"] if d["metric"] == "aicore_gang_span" else None,
                "device_metric": d["metric"],
                "remote_bytes": d["per_rank_send_bytes"][str(d["fastest_rank"])],
            }
        )
    return out


def gbps(nbytes: int, us: float) -> float:
    return nbytes * 8 / (us * 1e-6) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--hccl-log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    hccl = parse_hccl_log(args.hccl_log)
    cells = load_pypto_cells(args.campaign)
    rows = cells + [hccl]
    labels = [r["label"] for r in rows]

    # ── figure 1: host-slot p50 (the apples-to-apples column) ──────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [r["slot_p50_us"] for r in rows]
    colors = ["#4C72B0" if r["device_us"] is None else "#DD8452" for r in rows]
    ax.bar(labels, vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("host-slot p50 per collective  [µs, log]")
    ax.set_title("EP8 all_to_all_v, 64 KiB/peer, uniform — host-visible time\n"
                 "(HCCL = per-rank sync wall; PyPTO = host timing slot)")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.05, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(vals[-1], color="k", ls="--", lw=0.8)
    ax.text(len(rows) - 0.4, vals[-1] * 1.02, "HCCL", fontsize=8, va="bottom", ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "a2av_ep8_host_slot.png", dpi=150)
    plt.close(fig)

    # ── figure 2: egress per rank ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bw = [gbps(r["remote_bytes"], r["slot_p50_us"]) for r in rows]
    ax.bar(labels, bw, color=colors)
    ax.set_ylabel("egress  [Gbit/s per rank, remote payload]")
    ax.set_title("EP8 all_to_all_v, 64 KiB/peer — effective per-rank egress\n"
                 f"(remote payload = (P-1) x 64 KiB = {hccl['remote_bytes']:,} B)")
    for i, v in enumerate(bw):
        ax.text(i, v + max(bw) * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "a2av_ep8_egress.png", dpi=150)
    plt.close(fig)

    # ── figure 3: where the PyPTO host-slot time goes ──────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    sel = [r for r in rows if r["device_us"] is not None] + [hccl]
    x = [r["label"] for r in sel]
    dev = [r["device_us"] or 0 for r in sel]
    rem = [r["slot_p50_us"] - (r["device_us"] or 0) for r in sel]
    ax.bar(x, dev, color="#55A868", label="device-attributable (AIV gang span)")
    ax.bar(x, rem, bottom=dev, color="#C44E52", label="host remainder (dispatch / staging / sync)")
    for i, r in enumerate(sel):
        ax.text(i, r["slot_p50_us"] * 1.02, f"slot {r['slot_p50_us']:,.0f} µs",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("host-slot p50  [µs]")
    ax.set_title("EP8 all_to_all_v, 64 KiB/peer — decomposition of the host-visible time\n"
                 "(HCCL exposes no device span; its whole bar is the collective)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "a2av_ep8_decomposition.png", dpi=150)
    plt.close(fig)

    summary = {"hccl": hccl, "pypto": cells,
               "ratios_slot_vs_hccl": {r["label"]: r["slot_p50_us"] / hccl["slot_p50_us"] for r in cells}}
    (args.out / "a2av_ep8_comparison.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"figures": sorted(p.name for p in args.out.glob('*.png')),
                      "ratios_slot_vs_hccl": summary["ratios_slot_vs_hccl"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
