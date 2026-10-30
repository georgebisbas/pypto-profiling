#!/usr/bin/env python3
"""Join HCCL HcclAlltoAllV logs with the PyPTO A1 summary into payload_sweep.json.

Usage:
    python3 collectives/join_hccl_vs_pypto_a2av.py <campaign-dir>

The campaign dir must contain:
    hccl/hccl_a2av_ep8_<peer_bytes>.log   (HCCL_TIMED per_rank rows)
    pypto/summary.json                    (alltoallv_a1.py rows)

Writes <campaign-dir>/payload_sweep.json for
collectives/plot_hccl_vs_pypto_payload_sweep.py and prints the ratio table.
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path


def parse_hccl(path: Path) -> tuple[float, int]:
    """Return (median rank of per-rank p50, world size) in microseconds."""
    text = path.read_text()
    rows = [
        [float(x) for x in m.split(",")]
        for m in re.findall(r"HCCL_TIMED round=\d+ per_rank=\[([^\]]+)\]", text)
    ]
    if not rows:
        raise SystemExit(f"no HCCL_TIMED rows in {path}")
    world = len(rows[0])
    per_rank = [[r[i] for r in rows] for i in range(world)]
    p50 = [st.median(v) * 1e6 for v in per_rank]
    return st.median(p50), world


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    campaign = Path(sys.argv[1]).resolve()
    hccl_dir = campaign / "hccl"
    summary_path = campaign / "pypto" / "summary.json"

    rows: list[dict] = []
    for log in sorted(hccl_dir.glob("hccl_a2av_ep8_*.log")):
        peer_bytes = int(log.stem.rsplit("_", 1)[1])
        slot, world = parse_hccl(log)
        rows.append({
            "config": "HCCL HcclAlltoAllV",
            "peer_bytes": peer_bytes,
            "slot_p50_us": slot,
            "remote_bytes": peer_bytes * world,  # --include-self matches the harness
            "device_us": None,
        })

    for r in json.loads(summary_path.read_text()):
        if not r.get("ok"):
            continue
        ep = r["ep"]
        rows.append({
            "config": f"pypto {r['impl']} L={r['core_num']}",
            "peer_bytes": r["peer_bytes"],
            "slot_p50_us": r.get("host_slot_p50_us") or r.get("end_to_end_latency_p50_us"),
            "remote_bytes": r.get("send_bytes") or r["peer_bytes"] * ep,
            "device_us": r.get("fastest_p50_us")
            if r.get("metric") in ("aicore_gang_span", "per_rank_receive_bandwidth")
            else None,
        })

    out = campaign / "payload_sweep.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out} ({len(rows)} rows)")

    hccl = {r["peer_bytes"]: r["slot_p50_us"] for r in rows if r["config"] == "HCCL HcclAlltoAllV"}
    print(f"{'peer':>8} {'config':<26}{'slot_p50':>10}{'vs HCCL':>9}{'device':>9}")
    for r in sorted(rows, key=lambda x: (x["peer_bytes"], x["config"])):
        base = hccl.get(r["peer_bytes"])
        ratio = f"{r['slot_p50_us'] / base:.2f}x" if base else "n/a"
        dev = f"{r['device_us']:.0f}us" if r["device_us"] else "n/a"
        print(f"{r['peer_bytes']:>8} {r['config']:<26}{r['slot_p50_us']:9.0f}us{ratio:>9}{dev:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
