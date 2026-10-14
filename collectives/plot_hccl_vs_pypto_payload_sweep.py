#!/usr/bin/env python3
"""EP8 payload sweep — HCCL `HcclAlltoAllV` vs PyPTO `all_to_all_v` (figures).

Reads the `payload_sweep.json` written when assembling the matrix (HCCL numbers
parsed from `hccl_a2av_ep8_<peer_bytes>.log`, PyPTO numbers from the driver's
`summary.json`).

Two rulers are drawn, and they answer different questions:

* **host-slot** (the only column both stacks share) — what a caller sees.
* **device span** (PyPTO only; HCCL exposes none) — how fast the collective
  itself is. Comparing it to HCCL's *host wall* is a category error, but it
  bounds how much of the host-slot gap is dispatch rather than kernel.

Usage::

    python3 collectives/plot_hccl_vs_pypto_payload_sweep.py \
        --sweep reports/issue-2521-a1-ep8-l2corenum-2026-10-14/payload_sweep.json \
        --out   reports/figures-2026-10-14
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "HCCL HcclAlltoAllV": "#111111",
    "pypto managed-l2 L=8": "#C44E52",
    "pypto managed-l2 L=1": "#DD8452",
    "pypto managed-host L=8": "#4C72B0",
    "pypto managed-host L=1": "#8C8C8C",
}
ORDER = ["HCCL HcclAlltoAllV", "pypto managed-l2 L=8", "pypto managed-host L=8",
         "pypto managed-l2 L=1", "pypto managed-host L=1"]


def series(rows: list[dict], cfg: str) -> tuple[list[float], list[float]]:
    sel = sorted((r for r in rows if r["config"] == cfg), key=lambda r: r["peer_bytes"])
    return [r["peer_bytes"] / 1024 for r in sel], [r["slot_p50_us"] for r in sel]


def gbps(rows: list[dict]) -> dict[str, tuple[list[float], list[float]]]:
    out = {}
    for cfg in ORDER:
        sel = sorted((r for r in rows if r["config"] == cfg), key=lambda r: r["peer_bytes"])
        x = [r["peer_bytes"] / 1024 for r in sel]
        y = [r["remote_bytes"] * 8 / (r["slot_p50_us"] * 1e-6) / 1e9 for r in sel]
        out[cfg] = (x, y)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.sweep.read_text())

    # ── 1: time vs payload ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cfg in ORDER:
        x, y = series(rows, cfg)
        ax.plot(x, y, "o-", label=cfg, color=COLORS[cfg], lw=2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("payload per remote peer  [KiB]")
    ax.set_ylabel("host-slot p50  [µs]")
    ax.set_title("EP8 all_to_all_v — host-visible time vs payload\n"
                 "HCCL is flat; PyPTO grows roughly linearly")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(args.out / "a2av_ep8_payload_time.png", dpi=150); plt.close(fig)

    # ── 2: egress vs payload ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cfg, (x, y) in gbps(rows).items():
        ax.plot(x, y, "o-", label=cfg, color=COLORS[cfg], lw=2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("payload per remote peer  [KiB]")
    ax.set_ylabel("effective egress  [Gbit/s per rank]")
    ax.set_title("EP8 all_to_all_v — effective per-rank bandwidth vs payload\n"
                 "(remote payload = (P-1) x peer_bytes, both sides)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(args.out / "a2av_ep8_payload_bw.png", dpi=150); plt.close(fig)

    # ── 3: ratio vs payload — the gap WIDENS ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    hx, hy = series(rows, "HCCL HcclAlltoAllV")
    for cfg in ORDER[1:]:
        x, y = series(rows, cfg)
        ax.plot(x, [b / a for a, b in zip(hy, y)], "o-", label=cfg, color=COLORS[cfg], lw=2)
    ax.set_xscale("log", base=2)
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    ax.set_xlabel("payload per remote peer  [KiB]")
    ax.set_ylabel("PyPTO / HCCL  host-slot p50  [x]")
    ax.set_title("EP8 all_to_all_v — the PyPTO/HCCL gap WIDENS with payload\n"
                 "6.3x at 64 KiB -> 23.6x at 1 MiB (best PyPTO config)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(args.out / "a2av_ep8_payload_ratio.png", dpi=150); plt.close(fig)

    # ── 4: device span vs HCCL's whole wall — the crossover ───────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    px = [r["peer_bytes"] / 1024 for r in sorted(rows, key=lambda r: r["peer_bytes"])
          if r["config"] == "pypto managed-l2 L=8"]
    pads = [r["device_us"] for r in sorted(rows, key=lambda r: r["peer_bytes"])
            if r["config"] == "pypto managed-l2 L=8"]
    ax.plot(px, pads, "o-", color=COLORS["pypto managed-l2 L=8"], lw=2,
            label="pypto managed-l2 L=8 — AIV gang span (device)")
    ax.plot(hx, hy, "o-", color=COLORS["HCCL HcclAlltoAllV"], lw=2,
            label="HCCL HcclAlltoAllV — whole host wall")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("payload per remote peer  [KiB]")
    ax.set_ylabel("time  [µs]")
    ax.set_title("EP8 all_to_all_v — where the gap comes from\n"
                 "PyPTO's device span is BELOW HCCL's wall at 64 KiB and ABOVE it from 256 KiB")
    for x, y in zip(px, pads):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color=COLORS["pypto managed-l2 L=8"])
    ax.annotate("crossover\nbetween 64 and 256 KiB", xy=(128, 400), fontsize=8,
                color="#333333", ha="center")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(args.out / "a2av_ep8_payload_device_vs_hccl.png", dpi=150); plt.close(fig)

    print("wrote:", ", ".join(sorted(p.name for p in args.out.glob("a2av_ep8_payload_*.png"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
