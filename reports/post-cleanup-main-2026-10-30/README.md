# Post-cleanup rerun on pypto `f32ecf3f0` (2026-10-30)

> **Report status:** current — rerun of the HCCL-vs-pypto comparison on the rebuilt image. See
> [`../upstream-status.md`](../upstream-status.md).

The 2026-10-30 cleanup benchmark ran the container's preinstalled pypto, which turned out to be
`7fc45aed` (2026-09-23), not current main. This campaign re-ran the same matrix on a freshly built
image at pypto `f32ecf3f0` (2026-10-30).

## Provenance

| | |
|---|---|
| Box | `hng-atlas01` (192.168.150.11), 8× Ascend 910B2 |
| Image | `pypto3-hw-native-sys:cann9` (retagged 2026-10-30 from `cann9-f32ecf3f0`) |
| pypto | `f32ecf3f05d23a83a52283ad7998ca59cd8d291d` |
| runtime (simpler) | `6b6b88f2353cca12cfafe809750b50b59931b0d1` |
| pto-isa | `217bb98952a8f9004b95fe938da97d8d5f68f702` |
| PTOAS | v0.70 |
| Old image (preserved) | `pypto3-hw-native-sys:cann9-7fc45aed` — installed pypto `7fc45aed`, runtime `6e383fc5`, PTOAS 0.65 |
| Queue tasks | P=2/4: `task_20261030_152948_13429518657`; P=8: `task_20261030_153314_14698862220` |
| Baseline task | old-image run: `task_20261030_150559_5091542642` |

## Method

Mesh allreduce, 65,536×fp32 (256 KiB/rank), P=2/4/8, stacks `hccl,pypto-composite,pypto-host`,
`--persistent`, 2 warmup + 5 timed rounds. `execute_s` is host-visible dispatch + collective;
`device_wall` is the on-device collective. The "device vs HCCL" ratio uses HCCL's `execute_s`
median as the denominator because HCCL emits no `device_wall`.

## Results — pypto `f32ecf3f0`

| P | stack | `execute_s` p50 | spread | `device_wall` | device vs HCCL | correct |
|---:|---|---:|---:|---:|---:|:--:|
| 2 | hccl | 161 µs | 1.10 | — | — | pass |
| 2 | pypto-composite | 7256 µs | 1.83 | 332 µs | 2.06× | pass |
| 2 | pypto-host | 7623 µs | **3.29** | 406 µs | 2.52× | pass |
| 4 | hccl | 184 µs | 1.10 | — | — | pass |
| 4 | pypto-composite | 10456 µs | **2.37** | 912 µs | 4.96× | pass |
| 4 | pypto-host | 12191 µs | 1.20 | 1448 µs | 7.87× | pass |
| 8 | hccl | 221 µs | 1.11 | — | — | pass |
| 8 | pypto-composite | 15731 µs | 1.29 | 1719 µs | 7.78× | pass |
| 8 | pypto-host | 20010 µs | 1.16 | 3326 µs | 15.05× | pass |

## Old (`7fc45aed`) vs new (`f32ecf3f0`)

| P | stack | old exec | new exec | Δ | old device | new device | Δ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | pypto-composite | 6717 µs | 7256 µs | +8.0% | 354 µs | 332 µs | −6.2% |
| 2 | pypto-host | 8226 µs | 7623 µs | −7.3% | 376 µs | 406 µs | +8.0% |
| 4 | pypto-composite | 8140 µs | 10456 µs | +28.5% | 817 µs | 912 µs | +11.6% |
| 4 | pypto-host | 9864 µs | 12191 µs | +23.6% | 1436 µs | 1448 µs | +0.8% |
| 8 | pypto-composite | 14551 µs | 15731 µs | +8.1% | 1728 µs | 1719 µs | −0.5% |
| 8 | pypto-host | 16687 µs | 20010 µs | +19.9% | 2855 µs | 3326 µs | +16.5% |

## Reading

- The **P=8 row is the clean comparison**: it held all eight cards and every spread is 1.1–1.3.
  There, `pypto-composite` is unchanged on-device (−0.5%); `pypto-host` is ~16% slower.
- **P=2/P=4 are noisier**: those runs held only four cards while another user held 4–5, which is why
  `pypto-host` P=2 (3.29) and `pypto-composite` P=4 (2.37) are flagged. Use the medians, but do not
  read the P=4 deltas as a clean regression.
- HCCL itself is stable across the two images (161/184/221 µs vs 162/188/231 µs).

## Compatibility fix found by the rerun

The first attempt failed at the box-health probe:

```text
domain window_size must be a multiple of 64 bytes
```

The current runtime (`6b6b88f2`) requires `window_size` and every comm-buffer `nbytes` to be
64-byte multiples. `simpler-own`'s scratch was `count*4 + nranks*4` (262,152 B at P=2/64 KiB),
which is unaligned. `collectives/runners/simpler_own.py::_scratch_layout()` now pads the physical
scratch to 64 bytes; the signal tail stays at `count*4`, which is aligned for the supported counts.
`tests/test_simpler_layout.py` locks the invariant.

## Caveats

- One payload size (64 KiB); no message-size sweep.
- P=2/P=4 shared the box with another tenant (see above); only P=8 is contention-free.
- The container lacks matplotlib, so figures are text fallbacks.
- The old image's build arg (`5f71449f`) did not match its installed files (`7fc45aed`); the new
  image was verified by reading `/opt/pypto` and `/opt/pypto/runtime` inside a queued job.

## Artifacts

`summary.json` (9 runs) and `json/results_p{2,4,8}.json` are the durable per-cell records. Raw
driver logs and run directories remain under `results/campaigns/post_cleanup_main_*` (gitignored).
