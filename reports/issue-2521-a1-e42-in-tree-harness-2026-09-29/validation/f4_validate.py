"""F4 validation: old (mean) vs new (max/last-block) aggregation over real + synthetic records."""
import glob
import importlib.util
import json
import pathlib
import shutil
import sys

MOD_PATH = "/opt/a1/e42/tests/st/distributed/collectives/all_to_all_v_benchmark.py"
spec = importlib.util.spec_from_file_location("a2av_bench", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["a2av_bench"] = mod
spec.loader.exec_module(mod)


def durations(path):
    names = mod._load_name_map(path.parent / "name_map.json")
    rec = json.loads(path.read_text())
    freq = float((rec.get("metadata") or {}).get("clock_freq_hz") or 50_000_000)
    us_per_cycle = 1e6 / freq
    durs = []
    for row in rec.get("aicore_tasks") or []:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        func_id = int(row[1]) & 0xFFFFFFFF
        name = names.get(str(func_id), "")
        if "all_to_all_v" not in name:
            continue
        start, end = int(row[3]), int(row[4])
        dur = (end - start) * us_per_cycle
        if dur > 0:
            durs.append(dur)
    return durs


def old_agg(durs):
    c = mod._cluster_near_min(durs)
    return (sum(c) / len(c)) if c else None


def new_agg(durs):
    c = mod._cluster_near_min(durs)
    return max(c) if c else None


roots = [
    "/opt/pypto-profiling/reports/issue-2521-a1-b1-2026-09-25/builds",
    "/opt/a1/salvage/b1_swim_build",
]
files = []
for r in roots:
    files += sorted(glob.glob(r + "/**/chip_swimlane_records.json", recursive=True))
print(f"real records found: {len(files)}")

nz = 0
mism = 0
for f in files:
    d = durations(pathlib.Path(f))
    if not d:
        continue
    nz += 1
    o, n = old_agg(d), new_agg(d)
    if abs((o or 0.0) - (n or 0.0)) > 1e-9:
        mism += 1
        if mism <= 10:
            c = mod._cluster_near_min(d)
            print(f"MISMATCH {f}: rows={len(d)} cluster={len(c)} old={o:.3f} new={n:.3f}")
print(f"records with all_to_all_v tasks: {nz}; mismatches(old!=new): {mism}")

bad = 0
for f in files[:80]:
    d = durations(pathlib.Path(f))
    if not d:
        continue
    v = mod._aiv_exec_us_from_raw_records(pathlib.Path(f))
    if v is not None and abs(v - max(mod._cluster_near_min(d))) > 1e-9:
        bad += 1
print("module-function vs max cross-check (first 80):", "OK" if bad == 0 else f"{bad} BAD")

S = pathlib.Path("/tmp/f4_synth/dfx_outputs/rank0/d0")
shutil.rmtree("/tmp/f4_synth", ignore_errors=True)
S.mkdir(parents=True)
freq = 50_000_000
us = lambda x: int(round(x * freq / 1e6))  # noqa: E731
rows = [[0, 7, 0, 0, us(110 + i * (10 / 15))] for i in range(16)]
rows.append([0, 7, 0, 0, us(1000)])  # far dual-die artifact
rec = {"metadata": {"clock_freq_hz": freq}, "aicore_tasks": rows}
(S / "chip_swimlane_records.json").write_text(json.dumps(rec))
(S / "name_map.json").write_text(json.dumps({"callable_id_to_name": {"7": "builtin.tensor.all_to_all_v__int8"}}))
p = S / "chip_swimlane_records.json"
d = durations(p)
print("synthetic durs us:", [round(x, 2) for x in d])
print(f"synthetic 16 blocks: old(mean)={old_agg(d):.3f} -> new(max/last)={new_agg(d):.3f}; cluster keeps {len(mod._cluster_near_min(d))}/{len(d)} (artifact dropped)")
print("module func on synthetic:", mod._aiv_exec_us_from_raw_records(p))

rec["aicore_tasks"] = [[0, 7, 0, 0, us(110)]]
(S / "chip_swimlane_records.json").write_text(json.dumps(rec))
o, n = old_agg(durations(p)), new_agg(durations(p))
print(f"single-block check: old={o:.3f} new={n:.3f} equal={abs(o - n) < 1e-9}")
