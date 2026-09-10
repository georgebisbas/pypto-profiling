#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MFE: EP8 Fabric V2 domain-window churn on a2a3 onboard.

Stresses ``orch.allocate_domain`` across 8 chips with **no** AICore kernel and
**no** pypto AllToAllV. Each iteration builds a fresh L3 Worker, allocates one
8-rank domain (Fabric V2 export by default), releases it, and tears the Worker
down — the same allocate/release cadence that A1 ``prepare()`` sessions hit.

Expected failure mode (intermittent on multi-root 8×910B2):

  This device does not support cross-server communication ... ExportToShareableHandleV2
  → ACL_ERROR_RT_AICPU_EXCEPTION (507018) / scheduler timeout
  → release_domain / comm_release_domain_windows failed with code -1
  → often SIGSEGV on process exit; next job may fail ``comm_init``

Workaround (once landed): ``SIMPLER_COMM_FORCE_IPC=1`` skips Fabric and uses
VMM IPC windows.

Run (from simpler repo root, a2a3 box with 8 NPUs)::

    export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
    # unset SIMPLER_COMM_FORCE_IPC   # must be unset to hit Fabric
    python3 tools/mfe/ep8_fabric_cross_server/repro_domain_churn.py --iters 20

    # control (should stay green once FORCE_IPC exists):
    SIMPLER_COMM_FORCE_IPC=1 python3 tools/mfe/ep8_fabric_cross_server/repro_domain_churn.py --iters 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from simpler.task_interface import CallConfig  # noqa: E402
from simpler.worker import Worker  # noqa: E402


def parse_devices(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = (int(x) for x in spec.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def one_churn(
    *,
    device_ids: list[int],
    window_size: int,
    platform: str,
    runtime: str,
    dual_alloc: bool,
) -> None:
    """One Worker lifetime: init → allocate_domain(all ranks) → release → close."""

    def orch_fn(orch, _args, _cfg):
        # Empty buffers: pure domain-window Fabric/IPC handshake — no kernel.
        with orch.allocate_domain(
            name="mfe_ep8_fabric",
            workers=list(range(len(device_ids))),
            window_size=window_size,
            buffers=[],
        ):
            pass
        if dual_alloc:
            # Second domain in the same run mimics dual prepare()/session churn.
            with orch.allocate_domain(
                name="mfe_ep8_fabric_b",
                workers=list(range(len(device_ids))),
                window_size=window_size,
                buffers=[],
            ):
                pass

    worker = Worker(
        level=3,
        platform=platform,
        runtime=runtime,
        device_ids=device_ids,
        num_sub_workers=0,
    )
    try:
        worker.init()
        worker.run(orch_fn, args=None, config=CallConfig())
        if dual_alloc:
            # Second run on the same Worker: closer to slot then swimlane prepare.
            worker.run(orch_fn, args=None, config=CallConfig())
    finally:
        worker.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devices", default="0-7", help="device id list/range (default: 0-7)")
    p.add_argument("--iters", type=int, default=20, help="Worker create/alloc/teardown iterations")
    p.add_argument("--window-size", type=int, default=4 * 1024 * 1024, help="domain window bytes")
    p.add_argument("--platform", default="a2a3", choices=("a2a3",))
    p.add_argument("--runtime", default="tensormap_and_ringbuffer")
    p.add_argument("--sleep", type=float, default=0.0, help="optional delay between iters (seconds)")
    p.add_argument(
        "--dual-alloc",
        action="store_true",
        help="two domains per run + a second run (raises Fabric handshake rate)",
    )
    args = p.parse_args(argv)

    device_ids = parse_devices(args.devices)
    if len(device_ids) < 2:
        print("need at least 2 devices", file=sys.stderr)
        return 2

    force = os.environ.get("SIMPLER_COMM_FORCE_IPC", "")
    print(
        f"[mfe] platform={args.platform} runtime={args.runtime} "
        f"devices={device_ids} iters={args.iters} window_size={args.window_size} "
        f"dual_alloc={args.dual_alloc} SIMPLER_COMM_FORCE_IPC={force!r}",
        flush=True,
    )

    for i in range(1, args.iters + 1):
        t0 = time.perf_counter()
        try:
            one_churn(
                device_ids=device_ids,
                window_size=args.window_size,
                platform=args.platform,
                runtime=args.runtime,
                dual_alloc=args.dual_alloc,
            )
        except BaseException as exc:  # noqa: BLE001 — MFE must surface any crash/poison
            dt = time.perf_counter() - t0
            print(f"[mfe] FAIL iter={i}/{args.iters} after {dt:.1f}s: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            print(
                "[mfe] Grep host/DFX logs for: cross-server | ExportToShareableHandleV2 | "
                "release_domain | 507018",
                flush=True,
            )
            return 1
        dt = time.perf_counter() - t0
        print(f"[mfe] OK   iter={i}/{args.iters} ({dt:.1f}s)", flush=True)
        if args.sleep > 0:
            time.sleep(args.sleep)

    print("[mfe] all iterations OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
