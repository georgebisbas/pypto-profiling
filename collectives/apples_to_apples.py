"""Apples-to-apples collective analysis — persistent vs non-persistent, dispatch
decomposition, and honest ratios vs HCCL.

Reads the strong-scaling campaign (analytic_strong) plus the persistent /
non-persistent clarity runs and renders the figures an HPC/compiler engineer
needs to separate *device* time from *dispatch* time from *domain-lifecycle*
time, and to compare each against HCCL fairly.

Figures (written to --out, default reports/figures-2026-08-28/):
  a2a_time_breakdown.png        stacked execute_s = device + dispatch + lifecycle,
                                persistent vs non-persistent per (P, stack)
  a2a_ratios_vs_hccl.png        log bars: device / persist-exec / nonpersist-exec
                                ratio vs HCCL per (P, stack)
  a2a_persistent_gain.png       nonpersist/persist execute_s speedup vs P
  a2a_effective_bw_vs_p.png     effective mesh bandwidth ((P-1)*N*4/time) vs P;
                                HCCL execute-based, pypto device_wall-based
  a2a_scaling_efficiency.png    strong-scaling efficiency: device vs execute vs
                                the mesh algorithm's inherent 2/(P(P-1)) floor

Usage:
    PYTHONPATH=. python3 -m collectives.apples_to_apples \
        [--strong results/campaigns/analytic_strong/run_*/results.json] \
        [--out reports/figures-2026-08-28]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

STACK_COLORS = {
    "hccl": "#e74c3c",
    "pypto-composite": "#3498db",
    "pypto-host": "#2ecc71",
}
STACK_MARKERS = {"hccl": "s", "pypto-composite": "o", "pypto-host": "^"}
COMPONENT_COLORS = {"device": "#2ecc71", "dispatch": "#f39c12", "lifecycle": "#e74c3c"}

_CAMPAIGNS = {
    "strong": _REPO / "results/campaigns/analytic_strong/run_20260828_150909/results.json",
    "np2": _REPO / "results/campaigns/clarity_p2/run_nonpersist/results.json",
    "pp2": _REPO / "results/campaigns/clarity_p2/run_persist/results.json",
    "bp2": _REPO / "results/campaigns/clarity_p2/run_persist_batch/results.json",
    "np4": _REPO / "results/campaigns/clarity_p4/run_nonpersist/results.json",
    "pp4": _REPO / "results/campaigns/clarity_p4/run_persist/results.json",
}


def _load_runs(path: Path) -> dict[tuple[int, str], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(r["p"], r["stack"]): r for r in data.get("runs", [])}


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # noqa: BLE001
        print(f"  matplotlib unavailable ({exc}); emitting text summaries only")
        return None


def _ms(seconds: float | None) -> float:
    return (seconds or 0.0) * 1e3


def _decomposition(strong, np2, pp2, bp2, np4, pp4):
    """Per (P, stack): nonpersist / persist / batch / device / dispatch / lifecycle."""
    out = {}
    for p, s in ((2, "pypto-composite"), (2, "pypto-host"),
                 (4, "pypto-composite"), (4, "pypto-host")):
        np_, pp_, bp_ = (np2, pp2, bp2) if p == 2 else (np4, pp4, {})
        nonpersist = _ms(np_.get((p, s), {}).get("execute_s_mean"))
        persist = _ms(pp_.get((p, s), {}).get("execute_s_mean"))
        batch = _ms(bp_.get((p, s), {}).get("execute_s_mean")) if bp_ else None
        device = _ms(pp_.get((p, s), {}).get("device_wall_s_mean"))
        dispatch = max(0.0, persist - device)
        lifecycle = max(0.0, nonpersist - persist)
        gain = nonpersist / persist if persist > 0 else None
        out[(p, s)] = dict(nonpersist=nonpersist, persist=persist, batch=batch,
                           device=device, dispatch=dispatch, lifecycle=lifecycle,
                           gain=gain)
    return out


def _plot_time_breakdown(plt, deco, fig_dir: Path) -> Path:
    """Stacked execute_s = device + dispatch + lifecycle (persistent vs not)."""
    order = [(2, "pypto-composite"), (2, "pypto-host"),
             (4, "pypto-composite"), (4, "pypto-host")]
    labels = [f"P{p}\n{s.split('-')[-1]}" for p, s in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, mode in zip(axes, ("nonpersist", "persist")):
        bottoms = [0.0] * len(order)
        for comp, color in COMPONENT_COLORS.items():
            if comp == "lifecycle" and mode == "persist":
                continue  # lifecycle is gone in persistent mode
            vals = [deco[(p, s)][comp] for p, s in order]
            ax.bar(range(len(order)), vals, bottom=bottoms, color=color,
                   edgecolor="white", label=comp)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"execute_s decomposition — {mode}")
        ax.grid(True, axis="y", alpha=0.25)
        if mode == "persist":
            ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("time per dispatch (ms, log)")
    for ax in axes:
        ax.set_yscale("log")
    fig.suptitle("Where the per-call time goes: device vs host dispatch vs CommDomain lifecycle",
                 fontsize=10)
    fig.tight_layout()
    path = fig_dir / "a2a_time_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  a2a_time_breakdown -> {path}")
    return path


def _plot_ratios_vs_hccl(plt, strong, deco, fig_dir: Path) -> Path:
    """Log bars: device / persist-exec / nonpersist-exec ratio vs HCCL."""
    order = [(2, "pypto-composite"), (2, "pypto-host"),
             (4, "pypto-composite"), (4, "pypto-host"),
             (8, "pypto-composite"), (8, "pypto-host")]
    labels = [f"P{p}\n{s.split('-')[-1]}" for p, s in order]
    series = [("device_wall", "#2ecc71", "device (device_wall)"),
              ("persist-exec", "#f39c12", "persistent execute_s"),
              ("nonpersist-exec", "#e74c3c", "non-persistent execute_s")]
    import numpy as np

    x = np.arange(len(order))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, color, label) in enumerate(series):
        vals = []
        for p, s in order:
            hccl = strong[(p, "hccl")]["execute_s_mean"] * 1e3
            if key == "device_wall":
                v = _ms(strong[(p, s)].get("device_wall_s_mean"))
            elif key == "persist-exec":
                v = _ms(strong[(p, s)]["execute_s_mean"])
            else:
                v = deco.get((p, s), {}).get("nonpersist")
                if v is None:
                    v = float("nan")
            vals.append(v / hccl if hccl else float("nan"))
        ax.bar(x + (i - 1) * width, vals, width=width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("ratio vs HCCL execute_s (log)")
    ax.set_title("Apples-to-apples: how much of the gap is device vs host dispatch vs domain lifecycle")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    path = fig_dir / "a2a_ratios_vs_hccl.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  a2a_ratios_vs_hccl -> {path}")
    return path


def _plot_persistent_gain(plt, deco, fig_dir: Path) -> Path:
    """Nonpersist/persist speedup vs P."""
    import numpy as np

    p_groups = {2: [], 4: []}
    for (p, s), d in deco.items():
        if d["gain"] is not None:
            p_groups[p].append((s, d["gain"]))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = []
    labels = []
    for p in (2, 4):
        for s, g in p_groups[p]:
            xs.append((p, s))
            labels.append(f"P{p}\n{s.split('-')[-1]}")
    gains = [deco[x]["gain"] for x in xs]
    colors = [STACK_COLORS[s] for _, s in xs]
    ax.bar(range(len(xs)), gains, color=colors, edgecolor="white")
    for i, g in enumerate(gains):
        ax.text(i, g + 0.5, f"{g:.0f}x", ha="center", fontsize=9)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("execute_s speedup (non-persistent / persistent)")
    ax.set_title("Persistent-domain gain per (P, stack) — 65536 fp32")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "a2a_persistent_gain.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  a2a_persistent_gain -> {path}")
    return path


def _plot_effective_bw(plt, strong, fig_dir: Path) -> Path:
    """Effective mesh bandwidth ((P-1)*N*4 / time) vs P.

    HCCL uses execute_s (host-observed); pypto uses device_wall (on-device) —
    the fairest cross-stack bandwidth number for each stack's own measurement.
    """
    import numpy as np

    ps = [2, 4, 8]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for stack in ("hccl", "pypto-composite", "pypto-host"):
        bws = []
        for p in ps:
            nbytes = (p - 1) * 65536 * 4
            r = strong[(p, stack)]
            if stack == "hccl":
                t = r["execute_s_mean"]
            else:
                t = r["device_wall_s_mean"]
            bws.append(nbytes / t / 1e6 if t else 0.0)
        ax.plot(ps, bws, marker=STACK_MARKERS[stack], color=STACK_COLORS[stack],
                label=f"{stack} ({'execute' if stack=='hccl' else 'device_wall'})",
                linewidth=1.6, markersize=8)
    ax.set_xlabel("Number of ranks (P)")
    ax.set_ylabel("effective mesh bandwidth (MB/s)")
    ax.set_title("Effective allreduce bandwidth vs P — mesh traffic (P-1)*N per rank")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "a2a_effective_bw_vs_p.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  a2a_effective_bw_vs_p -> {path}")
    return path


def _plot_scaling_efficiency(plt, strong, fig_dir: Path) -> Path:
    """Strong-scaling efficiency: device vs execute vs mesh-inherent 2/(P(P-1))."""
    import numpy as np

    ps = [4, 8]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for stack in ("pypto-composite", "pypto-host"):
        t2d = strong[(2, stack)]["device_wall_s_mean"]
        t2e = strong[(2, stack)]["execute_s_mean"]
        dev_eff = [t2d * 2 / (strong[(p, stack)]["device_wall_s_mean"] * p) for p in ps]
        exe_eff = [t2e * 2 / (strong[(p, stack)]["execute_s_mean"] * p) for p in ps]
        ax.plot(ps, dev_eff, marker="o", color=STACK_COLORS[stack], linestyle="-",
                label=f"{stack} — device_wall", linewidth=1.6, markersize=7)
        ax.plot(ps, exe_eff, marker="o", color=STACK_COLORS[stack], linestyle="--",
                label=f"{stack} — execute_s", linewidth=1.4, markersize=7)
    mesh = [2 / (p * (p - 1)) for p in ps]
    ax.plot(ps, mesh, marker="x", color="#7f8c8d", linestyle=":",
            label="mesh-inherent floor 2/(P(P-1))", linewidth=1.6, markersize=8)
    ax.set_xlabel("Number of ranks (P)")
    ax.set_ylabel("strong-scaling efficiency (base P=2)")
    ax.set_title("Strong-scaling efficiency at fixed N=256 KB — device vs host view")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "a2a_scaling_efficiency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  a2a_scaling_efficiency -> {path}")
    return path


def _plot_device_bw_vs_payload(plt, sizes_path: Path, fig_dir: Path) -> Path:
    """Bandwidth vs payload: HCCL execute-based vs pypto device_wall-based (solid)
    vs pypto persistent-execute-based (dashed) — the true apples-to-apples curve.

    Directly exposes the end-to-end vs on-device bandwidth gap that the harness
    ``message_size_bw_eff`` figure (execute_s for every stack) hides for pypto.
    """
    sizes_raw = json.loads(Path(sizes_path).read_text(encoding="utf-8")).get("runs", [])
    by_stack: dict[str, dict[int, dict]] = {}
    for r in sizes_raw:
        by_stack.setdefault(r["stack"], {})[r["count"]] = r
    counts = sorted(by_stack.get("hccl", {}))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for stack in ("hccl", "pypto-composite", "pypto-host"):
        color = STACK_COLORS[stack]
        marker = STACK_MARKERS[stack]
        rows = by_stack.get(stack, {})
        bws_exec = [rows[c]["count"] * 4 / rows[c]["execute_s_mean"] / 1e6 for c in counts if c in rows]
        bws_dev = []
        for c in counts:
            if c not in rows:
                continue
            dev = rows[c].get("device_wall_s_mean")
            bws_dev.append(rows[c]["count"] * 4 / dev / 1e6 if dev else None)
        if stack == "hccl":
            ax.plot(counts[:len(bws_exec)], bws_exec, marker=marker, color=color,
                    label=f"{stack} (execute_s)", linewidth=1.6, markersize=7)
        else:
            ax.plot(counts[:len(bws_dev)], bws_dev, marker=marker, color=color,
                    linestyle="-", label=f"{stack} (device_wall)",
                    linewidth=1.8, markersize=7)
            ax.plot(counts[:len(bws_exec)], bws_exec, marker=marker, color=color,
                    linestyle="--", label=f"{stack} (persistent execute_s)",
                    linewidth=1.3, markersize=6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Payload elements per rank (log2)")
    ax.set_ylabel("Bandwidth (MB/s)")
    ax.set_title("Apples-to-apples bandwidth: on-device vs end-to-end (P=2 mesh)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    path = fig_dir / "a2a_device_bw_vs_payload.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  a2a_device_bw_vs_payload -> {path}")
    return path


def _emit_summary(deco, strong) -> None:
    """Print the analytical summary tables (also used by the report)."""
    print("\n== Per-call time decomposition (65536 fp32, ms) ==")
    print(f"{'P':<3}{'stack':>16}{'nonpersist':>12}{'persist':>10}{'batch':>9}"
          f"{'device':>9}{'dispatch':>10}{'lifecycle':>11}{'gain':>7}")
    for (p, s) in ((2, "pypto-composite"), (2, "pypto-host"),
                   (4, "pypto-composite"), (4, "pypto-host")):
        d = deco[(p, s)]
        batch = f"{d['batch']:.2f}" if d["batch"] is not None else "—"
        gain = f"{d['gain']:.1f}x" if d["gain"] else "—"
        print(f"{p:<3}{s:>16}{d['nonpersist']:>11.2f}{d['persist']:>10.2f}"
              f"{batch:>9}{d['device']:>9.3f}{d['dispatch']:>10.2f}"
              f"{d['lifecycle']:>11.1f}{gain:>7}")

    print("\n== Ratio vs HCCL execute_s ==")
    print(f"{'P':<3}{'stack':>16}{'device':>9}{'persist-exec':>14}{'nonpersist-exec':>16}")
    for p in (2, 4, 8):
        hccl = strong[(p, "hccl")]["execute_s_mean"] * 1e3
        for s in ("pypto-composite", "pypto-host"):
            dev = _ms(strong[(p, s)].get("device_wall_s_mean"))
            exe = _ms(strong[(p, s)]["execute_s_mean"])
            npv = deco.get((p, s), {}).get("nonpersist")
            npv_str = f"{npv / hccl:.0f}x" if npv else "—"
            print(f"{p:<3}{s:>16}{dev / hccl:>8.2f}x{exe / hccl:>13.1f}x{npv_str:>16}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong", type=Path, default=_CAMPAIGNS["strong"])
    parser.add_argument("--sizes", type=Path,
                        default=_REPO / "results/campaigns/analytic_sizes/run_20260828_151614/results.json")
    parser.add_argument("--out", type=Path,
                        default=_REPO / "reports" / "figures-2026-08-28")
    for key in ("np2", "pp2", "bp2", "np4", "pp4"):
        parser.add_argument(f"--{key}", type=Path, default=_CAMPAIGNS[key])
    args = parser.parse_args(argv)

    strong = _load_runs(args.strong)
    np2, pp2, bp2 = _load_runs(args.np2), _load_runs(args.pp2), _load_runs(args.bp2)
    np4, pp4 = _load_runs(args.np4), _load_runs(args.pp4)

    deco = _decomposition(strong, np2, pp2, bp2, np4, pp4)
    _emit_summary(deco, strong)

    args.out.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    if plt is None:
        print("figures skipped (no matplotlib)")
        return 0
    _plot_time_breakdown(plt, deco, args.out)
    _plot_ratios_vs_hccl(plt, strong, deco, args.out)
    _plot_persistent_gain(plt, deco, args.out)
    _plot_effective_bw(plt, strong, args.out)
    _plot_scaling_efficiency(plt, strong, args.out)
    _plot_device_bw_vs_payload(plt, args.sizes, args.out)
    print(f"\nfigures written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
