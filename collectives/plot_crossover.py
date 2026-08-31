"""Crossover figure: multi-AIV (core_num) scaling vs message size.

Loads ``results/campaigns/crossover/**/results.json`` (P=2/P=4 x 9 counts x
core_num {1,8,16}, pypto-host mesh allreduce, persistent) and emits a 2x2 figure:

  top row    — effective on-device bandwidth (GB/s) vs payload bytes (log-x),
               one line per core_num {1,8,16}; left P=2, right P=4
  bottom row — multi-AIV speedup vs single-AIV (cn8/cn1, cn16/cn1),
               dashed line at 1.0; the crossover = where multi-AIV flips from
               a net loss to a net win

The apples-to-apples metric is ``device_wall_s`` (slowest-rank pure on-device
collective time); effective bandwidth = n_bytes / device_wall_s_median.

Usage:
    PYTHONPATH=. python3 -m collectives.plot_crossover \
        [--campaign-dir results/campaigns/crossover] \
        [--out reports/figures-2026-08-28/corenum_message_size_crossover.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CORE_NUMS = (1, 8, 16)
CN_STYLE = {
    1: dict(color="#2c3e50", marker="o", linestyle="-", label="1 AIV (baseline)"),
    8: dict(color="#e67e22", marker="s", linestyle="--", label="8 AIVs"),
    16: dict(color="#8e44ad", marker="^", linestyle=":", label="16 AIVs"),
}


def load_results(campaign_dir: Path) -> list[dict]:
    """Flatten campaign results.json files into rows."""
    rows: list[dict] = []
    for f in sorted(campaign_dir.glob("*/results.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        case = data.get("case", {})
        runs = data.get("runs", [])
        if not runs:
            continue
        run = runs[0]
        if run.get("correctness") != "pass":
            print(f"  skip {f.name}: correctness={run.get('correctness')}")
            continue
        dw = run.get("device_wall_s_median")
        if not dw or dw <= 0:
            print(f"  skip {f.name}: no device_wall_s_median")
            continue
        n_bytes = case.get("n_bytes") or case.get("bytes_per_rank_per_round")
        rows.append(
            {
                "p": int(case.get("p", run.get("p"))),
                "count": int(case.get("count", run.get("count"))),
                "core_num": int(case.get("core_num", run.get("core_num"))),
                "n_bytes": int(n_bytes),
                "device_wall_s": float(dw),
                "bw_gb_s": float(n_bytes) / float(dw) / 1e9,
            }
        )
    return rows


def _fmt_bytes(v: float, _pos=None) -> str:
    b = int(v)
    for unit, div in (("M", 1 << 20), ("K", 1 << 10)):
        if b >= div:
            return f"{b / div:.0f}{unit}"
    return f"{b}B"


def plot(rows: list[dict], out: Path) -> None:
    # index: (p, core_num) -> {count: bw_gb_s}
    bw: dict[tuple[int, int], dict[int, float]] = {}
    for r in rows:
        bw.setdefault((r["p"], r["core_num"]), {})[r["count"]] = r["bw_gb_s"]

    ps = sorted({r["p"] for r in rows})
    counts = sorted({r["count"] for r in rows})
    payload = {c: c * 4 for c in counts}  # fp32 -> bytes

    fig, axes = plt.subplots(2, len(ps), figsize=(13, 8), sharex="col")
    if len(ps) == 1:
        axes = axes.reshape(2, 1)

    for j, p in enumerate(ps):
        # --- top row: effective bandwidth ---
        ax = axes[0, j]
        for cn in CORE_NUMS:
            g = bw.get((p, cn), {})
            if not g:
                continue
            xs = [payload[c] for c in sorted(g)]
            ys = [g[c] for c in sorted(g)]
            ax.plot(xs, ys, **CN_STYLE[cn], linewidth=1.8, markersize=7)
        ax.set_title(f"P={p} — effective on-device bandwidth", fontsize=11)
        ax.set_ylabel("GB/s (device_wall)")
        ax.grid(True, alpha=0.3, which="both")
        ax.axvspan(16 * 1024, 256 * 1024, color="gray", alpha=0.08,
                   label="crossover region")
        # single-AIV saturation plateau annotation (P=2)
        if p == 2:
            g1 = bw.get((2, 1), {})
            if g1:
                sat = g1[max(g1)]
                ax.axhline(sat, color="#2c3e50", linestyle=":", linewidth=0.8,
                           alpha=0.5)
                ax.annotate(f"single-AIV saturates ≈{sat:.2f} GB/s",
                            xy=(payload[max(g1)], sat),
                            xytext=(payload[max(g1)] * 0.32, sat * 0.62),
                            fontsize=8, color="#2c3e50",
                            arrowprops=dict(arrowstyle="->", color="#2c3e50",
                                            lw=0.8))

        # --- bottom row: multi-AIV speedup ---
        ax2 = axes[1, j]
        g1 = bw.get((p, 1), {})
        for cn in (8, 16):
            g = bw.get((p, cn), {})
            if not g or not g1:
                continue
            xs = sorted(c for c in g if c in g1 and g1[c] > 0)
            ys = [g[c] / g1[c] for c in xs]
            ax2.plot([payload[c] for c in xs], ys,
                     color=CN_STYLE[cn]["color"],
                     marker=CN_STYLE[cn]["marker"],
                     linestyle=CN_STYLE[cn]["linestyle"],
                     linewidth=1.8, markersize=7,
                     label=f"cn{cn}/cn1")
        ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax2.text(0.99, 1.04, "parity (1.0×)", transform=ax2.get_yaxis_transform(),
                 ha="right", fontsize=8, color="gray")
        ax2.set_title(f"P={p} — multi-AIV speedup vs single-AIV", fontsize=11)
        ax2.set_ylabel("speedup (BW ratio)")
        ax2.grid(True, alpha=0.3, which="both")
        ax2.axvspan(16 * 1024, 256 * 1024, color="gray", alpha=0.08)
        # annotate the crossover point on the cn8 line: first payload where
        # cn8/cn1 >= 1.0 and stays >= 1.0 for every larger payload (monotone-stable)
        if g1:
            g8 = bw.get((p, 8), {})
            if g8:
                ratios = {c: g8[c] / g1[c] for c in sorted(g8)
                          if c in g1 and g1[c] > 0}
                cross = None
                for i, c in enumerate(sorted(ratios)):
                    if all(ratios[cc] >= 1.0 for cc in sorted(ratios)[i:]):
                        cross = c
                        break
                if cross:
                    ax2.annotate(f"crossover @ {_fmt_bytes(payload[cross])}",
                                 xy=(payload[cross], 1.0),
                                 xytext=(payload[cross] * 0.45, 1.35),
                                 fontsize=9, fontweight="bold",
                                 arrowprops=dict(arrowstyle="->", lw=1.0))

    for ax in axes[0]:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_bytes))
        ax.legend(fontsize=8, loc="lower right")
    for ax in axes[1]:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_bytes))
        ax.set_xlabel("payload bytes (fp32, log2)")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Multi-AIV (core_num) scaling vs message size — pypto-host mesh "
                 "allreduce, a2a3 (Ascend 910B2), device_wall_s",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"crossover figure → {out}")


def summarize(rows: list[dict]) -> None:
    """Print a compact table for the companion report."""
    print("\nEffective device bandwidth (GB/s) = n_bytes / device_wall_s_median")
    print(f"{'P':>2} {'count':>8} {'payload':>8} " + "".join(
        f"{f'cn{cn}':>9}" for cn in CORE_NUMS))
    for p in sorted({r['p'] for r in rows}):
        for c in sorted({r['count'] for r in rows}):
            vals = {r['core_num']: r['bw_gb_s']
                    for r in rows if r['p'] == p and r['count'] == c}
            print(f"{p:>2} {c:>8} {_fmt_bytes(c * 4):>8} " + "".join(
                f"{vals.get(cn, float('nan')):>9.2f}" for cn in CORE_NUMS))
    print("\nSpeedup vs cn1:")
    for p in sorted({r['p'] for r in rows}):
        for c in sorted({r['count'] for r in rows}):
            g1 = next((r['bw_gb_s'] for r in rows
                       if r['p'] == p and r['count'] == c and r['core_num'] == 1), None)
            if not g1:
                continue
            sp = {r['core_num']: r['bw_gb_s'] / g1 for r in rows
                  if r['p'] == p and r['count'] == c and r['core_num'] != 1}
            print(f"P={p} {_fmt_bytes(c * 4):>6}: " +
                  "  ".join(f"cn{cn}/cn1={v:.2f}x" for cn, v in sorted(sp.items())))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", type=Path,
                    default=Path("results/campaigns/crossover"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/figures-2026-08-28/"
                                 "corenum_message_size_crossover.png"))
    ap.add_argument("--no-plot", action="store_true", help="only print summary")
    args = ap.parse_args(argv)

    rows = load_results(args.campaign_dir)
    if not rows:
        print("no results found under", args.campaign_dir)
        return 1
    print(f"loaded {len(rows)} rows")
    summarize(rows)
    if not args.no_plot:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plot(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
