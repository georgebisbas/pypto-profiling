#!/usr/bin/env python3
"""RFC #2521 A1 AllToAllV campaign wrapper (YunjiQin 2026-09-01 roadmap).

Calls the in-tree harness
``tests/st/distributed/collectives/all_to_all_v_benchmark.py``.

Measurement conditions (roadmap §8.6):
  persistent=True, reset_persistent_windows=False, one compile per point,
  correctness dispatch, then a clean timing-slot session (5 warmup / 100
  measured) and a short swimlane session for the AIV gang (name_map parse).

Official kernel metric is swimlane AIV exec (fastest-rank mean of the
die-mix-filtered per-task duration). HOST vs L2 outer dispatch is a separate
column from the clean timing-slot session.

This box has 8× 910B2 and no EP16. Default is EP2 (devices 0,1) and EP4
(devices 0,1,2,3). core_num=1 only (O2 is not implemented).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HARNESS = Path("/opt/pypto/tests/st/distributed/collectives/all_to_all_v_benchmark.py")
LD_PRELOAD = "/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so"

# Roadmap §8.3 main curve. Sub-32 B tails are rejected by INT8 pl.load align.
PAYLOADS = [
    0,
    32,
    128,
    1024,
    4 * 1024,
    8 * 1024,
    16 * 1024,
    24 * 1024,
    24960,
    32 * 1024,
    48 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
]

DEVICES = {
    2: os.environ.get("A2AV_DEV2", "0,1"),
    4: os.environ.get("A2AV_DEV4", "0,1,2,3"),
    8: os.environ.get("A2AV_DEV8", "0,1,2,3,4,5,6,7"),
}


def run_one(
    *,
    out: Path,
    ep: int,
    impl: str,
    peer_bytes: int,
    pattern: str,
    rounds: int,
    warmup: int,
    swimlane_rounds: int,
) -> dict:
    tag = f"p{ep}_{impl}_{pattern}_{peer_bytes}"
    out_dir = out / "builds" / tag
    json_path = out / "json" / f"{tag}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(HARNESS),
        "--ep",
        str(ep),
        "--peer-bytes",
        str(peer_bytes),
        "--count-pattern",
        pattern,
        "--core-num",
        "1",
        "--impl",
        impl,
        "--rounds",
        str(rounds),
        "--warmup",
        str(warmup),
        "--profile",
        "both",
        "--swimlane-rounds",
        str(swimlane_rounds),
        "--platform",
        "a2a3",
        "--devices",
        DEVICES[ep],
        "--output-json",
        str(json_path),
        "--output-dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    env["LD_PRELOAD"] = LD_PRELOAD
    print(f"\n=== {tag} devices={DEVICES[ep]} ===", flush=True)
    proc = subprocess.run(cmd, env=env, cwd="/opt/pypto", capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2500:]
        print(tail, flush=True)
        return {
            "tag": tag,
            "ok": False,
            "ep": ep,
            "impl": impl,
            "peer_bytes": peer_bytes,
            "count_pattern": pattern,
            "rc": proc.returncode,
            "stderr_tail": tail[-800:],
        }
    harness = json.loads(json_path.read_text()) if json_path.is_file() else {}
    row = {
        "tag": tag,
        "ok": True,
        "ep": ep,
        "impl": impl,
        "peer_bytes": peer_bytes,
        "count_pattern": pattern,
        "devices": DEVICES[ep],
        "metric": harness.get("metric"),
        "persistent": harness.get("persistent"),
        "warmup": harness.get("warmup"),
        "rounds": harness.get("rounds"),
        "outer_l3_l2_dispatches": harness.get("outer_l3_l2_dispatches"),
        "row_width": harness.get("row_width"),
        "max_recv": harness.get("max_recv"),
        "requested_L": harness.get("requested_L"),
        "launched_B": harness.get("launched_B"),
        "fastest_mean_us": harness.get("fastest_mean_us"),
        "fastest_p50_us": harness.get("fastest_p50_us"),
        "fastest_p95_us": harness.get("fastest_p95_us"),
        "slowest_mean_us": harness.get("slowest_mean_us"),
        "rank_spread_us": harness.get("rank_spread_us"),
        "egress_gbps": harness.get("egress_gbps"),
        "duplex_gbps": harness.get("duplex_gbps"),
        "host_timing_slot_mean_us": harness.get("host_timing_slot_mean_us"),
        "host_timing_slot_p50_us": harness.get("host_timing_slot_p50_us"),
        "host_timing_slot_includes_swimlane_tax": harness.get(
            "host_timing_slot_includes_swimlane_tax"
        ),
        "aiv_rounds_captured": harness.get("aiv_rounds_captured"),
        "send_bytes": (harness.get("per_rank_send_bytes") or {}).get("0"),
        "json_path": str(json_path),
    }
    print(
        f"  metric={row['metric']}  kernel_p50={row['fastest_p50_us']}  "
        f"slot_p50={row['host_timing_slot_p50_us']}  "
        f"egress={row['egress_gbps']}",
        flush=True,
    )
    return row


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path(
        os.environ.get(
            "A2AV_OUT",
            f"/opt/pypto-profiling/reports/issue-2521-a1-alltoallv-baseline-{stamp}",
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    rounds = int(os.environ.get("A2AV_ROUNDS", "100"))
    warmup = int(os.environ.get("A2AV_WARMUP", "5"))
    swim_rounds = int(os.environ.get("A2AV_SWIMLANE_ROUNDS", "8"))
    eps = [int(x) for x in os.environ.get("A2AV_EPS", "2,4").split(",") if x.strip()]
    impls = os.environ.get("A2AV_IMPLS", "managed-l2,managed-host").split(",")
    payloads = [
        int(x)
        for x in os.environ.get("A2AV_PAYLOADS", ",".join(str(p) for p in PAYLOADS)).split(",")
        if x.strip()
    ]
    extra_patterns = [
        p.strip()
        for p in os.environ.get("A2AV_EXTRA_PATTERNS", "zero,single-hot").split(",")
        if p.strip()
    ]

    rows: list[dict] = []
    summary_path = out / "summary.json"
    meta = {
        "roadmap": "https://github.com/hw-native-sys/pypto/issues/2521#issuecomment-5495422542",
        "persistent": True,
        "reset_persistent_windows": False,
        "warmup": warmup,
        "timing_slot_rounds": rounds,
        "swimlane_rounds": swim_rounds,
        "requested_L": 1,
        "devices": DEVICES,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "campaign_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    jobs: list[tuple[int, str, int, str]] = []
    for ep in eps:
        for impl in impls:
            impl = impl.strip()
            for peer_bytes in payloads:
                jobs.append((ep, impl, peer_bytes, "uniform"))
            for pattern in extra_patterns:
                jobs.append((ep, impl, 24960, pattern))

    for ep, impl, peer_bytes, pattern in jobs:
        row = run_one(
            out=out,
            ep=ep,
            impl=impl,
            peer_bytes=peer_bytes,
            pattern=pattern,
            rounds=rounds,
            warmup=warmup,
            swimlane_rounds=swim_rounds,
        )
        rows.append(row)
        summary_path.write_text(json.dumps(rows, indent=2) + "\n")

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["n_ok"] = sum(1 for r in rows if r.get("ok"))
    meta["n_fail"] = sum(1 for r in rows if not r.get("ok"))
    (out / "campaign_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("\nWrote", summary_path, flush=True)
    return 0 if all(r.get("ok") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
