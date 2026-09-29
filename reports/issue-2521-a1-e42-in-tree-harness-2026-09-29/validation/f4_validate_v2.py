"""Completion-metric validation v2 (trace-view faithful completion).

Compares aggregations over archived raw records:
  old_mean_cluster - original archive semantics (mean of near-min cluster)
  old_max_cluster  - a46dd0c4 semantics (max of near-min cluster)
  new (module)     - faithful trace view (start-gap split), max within view
Plus synthetic fixtures: single-view multicore, two-view inflated duplicates,
single-block far pairs.
"""
import glob
import importlib.util
import json
import pathlib
import sys

MOD_PATH = "/opt/a1/e42/tests/st/distributed/collectives/all_to_all_v_benchmark.py"
spec = importlib.util.spec_from_file_location("a2av_bench", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["a2av_bench"] = mod
spec.loader.exec_module(mod)


def stamps(path):
    names = mod._load_name_map(path.parent / "name_map.json")
    rec = json.loads(path.read_text())
    freq = float((rec.get("metadata") or {}).get("clock_freq_hz") or 50_000_000)
    upc = 1e6 / freq
    tasks = []
    for row in rec.get("aicore_tasks") or []:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        fid = int(row[1]) & 0xFFFFFFFF
        if "all_to_all_v" not in names.get(str(fid), ""):
            continue
        dur = (int(row[4]) - int(row[3])) * upc
        if dur > 0:
            tasks.append((int(row[3]), dur))
    return tasks, upc


def _durs(tasks):
    return [d for _, d in tasks]


def old_mean_cluster(tasks):
    durs = _durs(tasks)
    if not durs:
        return None
    mn = min(durs)
    c = [d for d in durs if d <= max(mn * 2.5, mn + 5.0)]
    return sum(c) / len(c)


def old_max_cluster(tasks):
    durs = _durs(tasks)
    if not durs:
        return None
    mn = min(durs)
    c = [d for d in durs if d <= max(mn * 2.5, mn + 5.0)]
    return max(c)


files = []
for root in ["/opt/pypto-profiling/reports", "/opt/a1/salvage"]:
    files += glob.glob(root + "/**/chip_swimlane_records.json", recursive=True)
print("files scanned:", len(files))

n = 0
diff_mean = []
diff_max = []
for f in files:
    try:
        tasks, upc = stamps(pathlib.Path(f))
    except Exception:
        continue
    if not tasks:
        continue
    n += 1
    new = mod._completion_us(tasks, upc)
    om = old_mean_cluster(tasks)
    ox = old_max_cluster(tasks)
    if om is not None and abs(new - om) / max(om, 1e-9) > 0.01:
        diff_mean.append((f, round(om, 2), round(new, 2)))
    if ox is not None and abs(new - ox) / max(ox, 1e-9) > 0.01:
        diff_max.append((f, round(ox, 2), round(new, 2)))
print(f"records with all_to_all_v stamps: {n}")
print(f"new vs old_mean_cluster: {len(diff_mean)} diffs >1%")
for f, a, b in diff_mean[:6]:
    print(f"   mean={a} new={b}  {f.split('reports/')[-1][:90] if 'reports/' in f else f}")
print(f"new vs old_max_cluster: {len(diff_max)} diffs >1%")
for f, a, b in diff_max[:6]:
    print(f"   maxc={a} new={b}  {f.split('reports/')[-1][:90] if 'reports/' in f else f}")

UPC = 0.02  # 50 MHz
viewA = [(0, 10.0)] * 15 + [(0, 100.0)]
viewB = [(int(3.6e8), 20.0)] * 15 + [(int(3.6e8), 1000.0)]
cases = [
    ("single-view multicore {15x10us, 1x100us}", viewA),
    ("two-view w/ inflated duplicates", viewA + viewB),
    ("B=1 far pair (15.08, 603.62)", [(0, 15.08), (int(3.6e8), 603.62)]),
    ("B=1 benign diff-view pair (508.18, 674.0)", [(0, 508.18), (int(3.6e8), 674.0)]),
]
for label, tasks in cases:
    print(f"synthetic {label}: old_max_cluster={old_max_cluster(tasks):.2f} -> new={mod._completion_us(tasks, UPC):.2f}")

bad = 0
checked = 0
for f in files[:120]:
    try:
        tasks, upc = stamps(pathlib.Path(f))
    except Exception:
        continue
    if not tasks:
        continue
    checked += 1
    v = mod._aiv_exec_us_from_raw_records(pathlib.Path(f))
    if v is None or abs(v - mod._completion_us(tasks, upc)) > 1e-9:
        bad += 1
print(f"module-function cross-check ({checked} files):", "OK" if bad == 0 else f"{bad} BAD")
