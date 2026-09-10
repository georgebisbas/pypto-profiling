#!/usr/bin/env python3
"""RFC #2521 A1 AllToAllV campaign wrapper (YunjiQin 2026-09-01 roadmap).

Calls the A1 harness. Preferred location (checked into this repo)::

  collectives/a1_alltoallv/harness/all_to_all_v_benchmark.py

Override with ``A2AV_HARNESS=/path/to/all_to_all_v_benchmark.py`` or fall back
to ``$PYPTO_ROOT/tests/st/distributed/collectives/all_to_all_v_benchmark.py``.

Related bundle (patch, Fabric MFE, provenance): ``collectives/a1_alltoallv/``.

Measurement conditions (roadmap §8.6):
  persistent=True, reset_persistent_windows=False, one compile per point,
  correctness dispatch, then a clean timing-slot session (5 warmup / 100
  measured) and a short swimlane session for the AIV gang (name_map parse).

Official kernel metric is swimlane AIV exec (fastest-rank mean of the
die-mix-filtered per-task duration). HOST vs L2 outer dispatch is a separate
column from the clean timing-slot session.

This box has 8× 910B2 and no EP16. Default is EP2 (devices 0,1) and EP4
(devices 0,1,2,3). core_num=1 only (O2 is not implemented).

EP8 on this box intermittently hits Fabric V2 export / device poison
(``cross-server`` → ``release_domain -1`` / SIGSEGV). Mitigations:
``A2AV_RETRIES``, ``A2AV_COOLDOWN_S``, optional ``A2AV_HEAL_EP`` smoke, and
``A2AV_RESUME=1`` to skip already-OK tags in ``summary.json``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUNDLED_HARNESS = (
    _REPO_ROOT / "collectives" / "a1_alltoallv" / "harness" / "all_to_all_v_benchmark.py"
)


def _resolve_harness() -> Path:
    override = os.environ.get("A2AV_HARNESS", "").strip()
    if override:
        return Path(override)
    if _BUNDLED_HARNESS.is_file():
        return _BUNDLED_HARNESS
    pypto = Path(os.environ.get("PYPTO_ROOT", "/opt/pypto"))
    return pypto / "tests" / "st" / "distributed" / "collectives" / "all_to_all_v_benchmark.py"


def _resolve_pypto_cwd() -> Path:
    return Path(os.environ.get("PYPTO_ROOT", "/opt/pypto"))


HARNESS = _resolve_harness()
LD_PRELOAD = os.environ.get(
    "A2AV_LD_PRELOAD",
    "/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so",
)

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


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _tail_text(text: str, n: int = 2500) -> str:
    return (text or "")[-n:]


def _run_harness(cmd: list[str], *, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LD_PRELOAD"] = LD_PRELOAD
    # Optional: skip a2a3 Fabric V2 windows (EP8 multi-root flake workaround).
    if os.environ.get("A2AV_FORCE_IPC", "").strip() in ("1", "true", "yes"):
        env["SIMPLER_COMM_FORCE_IPC"] = "1"
    proc = subprocess.run(cmd, env=env, cwd=str(_resolve_pypto_cwd()), capture_output=True, text=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"cmd={' '.join(cmd)}\nrc={proc.returncode}\n"
            f"----- stdout -----\n{proc.stdout or ''}\n"
            f"----- stderr -----\n{proc.stderr or ''}\n"
        )
    return proc


def heal_smoke(out: Path, ep: int) -> bool:
    """Cheap EP2/EP4 timing-slot run to settle devices after EP8 poison."""
    if ep not in DEVICES:
        return False
    heal_dir = out / "heal" / f"ep{ep}_{int(time.time())}"
    heal_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(HARNESS),
        "--ep",
        str(ep),
        "--peer-bytes",
        "1024",
        "--count-pattern",
        "uniform",
        "--core-num",
        "1",
        "--impl",
        "managed-l2",
        "--rounds",
        "2",
        "--warmup",
        "1",
        "--profile",
        "timing-slot",
        "--platform",
        "a2a3",
        "--devices",
        DEVICES[ep],
        "--output-json",
        str(heal_dir / "out.json"),
        "--output-dir",
        str(heal_dir / "build"),
    ]
    print(f"  [heal] EP{ep} smoke on {DEVICES[ep]} ...", flush=True)
    proc = _run_harness(cmd, log_path=heal_dir / "log.txt")
    ok = proc.returncode == 0
    print(f"  [heal] rc={proc.returncode} {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        print(_tail_text(proc.stderr or proc.stdout or "", 800), flush=True)
    return ok


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
    retries: int,
    cooldown_s: float,
    heal_ep: int | None,
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

    attempts = max(1, retries + 1)
    last_fail: dict | None = None
    print(f"\n=== {tag} devices={DEVICES[ep]} attempts={attempts} ===", flush=True)

    for attempt in range(1, attempts + 1):
        if json_path.is_file():
            json_path.unlink()
        log_path = out / "attempt_logs" / f"{tag}_try{attempt}.log"
        proc = _run_harness(cmd, log_path=log_path)
        if proc.returncode == 0 and json_path.is_file():
            harness = json.loads(json_path.read_text())
            row = {
                "tag": tag,
                "ok": True,
                "ep": ep,
                "impl": impl,
                "peer_bytes": peer_bytes,
                "count_pattern": pattern,
                "devices": DEVICES[ep],
                "attempt": attempt,
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
                f"  OK attempt={attempt} metric={row['metric']}  "
                f"kernel_p50={row['fastest_p50_us']}  "
                f"slot_p50={row['host_timing_slot_p50_us']}  "
                f"egress={row['egress_gbps']}",
                flush=True,
            )
            return row

        tail = _tail_text(proc.stderr or proc.stdout or "")
        print(f"  FAIL attempt={attempt}/{attempts} rc={proc.returncode}", flush=True)
        print(tail, flush=True)
        last_fail = {
            "tag": tag,
            "ok": False,
            "ep": ep,
            "impl": impl,
            "peer_bytes": peer_bytes,
            "count_pattern": pattern,
            "rc": proc.returncode,
            "attempt": attempt,
            "stderr_tail": tail[-800:],
            "log_path": str(log_path),
        }
        if attempt >= attempts:
            break
        if heal_ep is not None:
            heal_smoke(out, heal_ep)
        if cooldown_s > 0:
            print(f"  [cooldown] sleeping {cooldown_s:.0f}s before retry", flush=True)
            time.sleep(cooldown_s)

    assert last_fail is not None
    return last_fail


def _load_ok_tags(summary_path: Path) -> dict[str, dict]:
    if not summary_path.is_file():
        return {}
    try:
        rows = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return {}
    return {r["tag"]: r for r in rows if r.get("ok") and "tag" in r}


def _upsert_row(rows: list[dict], row: dict) -> list[dict]:
    tag = row["tag"]
    out = [r for r in rows if r.get("tag") != tag]
    out.append(row)
    out.sort(key=lambda r: (r.get("ep", 0), r.get("impl", ""), r.get("count_pattern", ""), r.get("peer_bytes", 0)))
    return out


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path(
        os.environ.get(
            "A2AV_OUT",
            f"/opt/pypto-profiling/reports/issue-2521-a1-alltoallv-baseline-{stamp}",
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    rounds = _env_int("A2AV_ROUNDS", 100)
    warmup = _env_int("A2AV_WARMUP", 5)
    swim_rounds = _env_int("A2AV_SWIMLANE_ROUNDS", 8)
    retries = _env_int("A2AV_RETRIES", 2)
    cooldown_s = float(os.environ.get("A2AV_COOLDOWN_S", "20"))
    heal_raw = os.environ.get("A2AV_HEAL_EP", "").strip()
    heal_ep = int(heal_raw) if heal_raw else None
    resume = os.environ.get("A2AV_RESUME", "0").strip() in ("1", "true", "yes")
    eps = [int(x) for x in os.environ.get("A2AV_EPS", "2,4").split(",") if x.strip()]
    impls = [x.strip() for x in os.environ.get("A2AV_IMPLS", "managed-l2,managed-host").split(",") if x.strip()]
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

    summary_path = out / "summary.json"
    ok_by_tag = _load_ok_tags(summary_path) if resume else {}
    rows: list[dict] = list(ok_by_tag.values()) if resume else []
    if resume and rows:
        print(f"[resume] keeping {len(rows)} OK tags from {summary_path}", flush=True)

    meta = {
        "roadmap": "https://github.com/hw-native-sys/pypto/issues/2521#issuecomment-5495422542",
        "persistent": True,
        "reset_persistent_windows": False,
        "warmup": warmup,
        "timing_slot_rounds": rounds,
        "swimlane_rounds": swim_rounds,
        "retries": retries,
        "cooldown_s": cooldown_s,
        "heal_ep": heal_ep,
        "resume": resume,
        "requested_L": 1,
        "devices": DEVICES,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "campaign_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    jobs: list[tuple[int, str, int, str]] = []
    for ep in eps:
        for impl in impls:
            for peer_bytes in payloads:
                jobs.append((ep, impl, peer_bytes, "uniform"))
            for pattern in extra_patterns:
                jobs.append((ep, impl, 24960, pattern))

    for ep, impl, peer_bytes, pattern in jobs:
        tag = f"p{ep}_{impl}_{pattern}_{peer_bytes}"
        if resume and tag in ok_by_tag:
            print(f"\n=== {tag} SKIP (resume OK) ===", flush=True)
            continue
        row = run_one(
            out=out,
            ep=ep,
            impl=impl,
            peer_bytes=peer_bytes,
            pattern=pattern,
            rounds=rounds,
            warmup=warmup,
            swimlane_rounds=swim_rounds,
            retries=retries,
            cooldown_s=cooldown_s,
            heal_ep=heal_ep,
        )
        rows = _upsert_row(rows, row)
        summary_path.write_text(json.dumps(rows, indent=2) + "\n")

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["n_ok"] = sum(1 for r in rows if r.get("ok"))
    meta["n_fail"] = sum(1 for r in rows if not r.get("ok"))
    (out / "campaign_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("\nWrote", summary_path, flush=True)
    print(f"n_ok={meta['n_ok']} n_fail={meta['n_fail']}", flush=True)
    return 0 if meta["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
