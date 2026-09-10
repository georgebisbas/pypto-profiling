# RFC #2521 A1 — EP8 attempt (roadmap-aligned)

EP=8 on devices 0–7, same §8.6 conditions as the EP2/EP4 ubfix campaign.
Follows Yunji roadmap priority for official A1 archive (EP8 first).

Driver: `collectives/alltoallv_a1.py` with `A2AV_EPS=8`.

## Status

- First pass (2026-09-10 morning): **4/8 OK** then aborted; failures were Fabric/`release_domain` teardown (see `debug/DEBUG_NOTES.md`).
- Resume (no-zero): recovered **4096, 16384**; **24576/24960** exhausted 4 tries; large-size grind **stopped** by request.
- Usable L2 OK so far: `32,128,1024,4096,8192,16384`.
- Follow-up L2 vs HOST on those sizes only: `../issue-2521-a1-alltoallv-ep8-l2host-2026-09-10/`.
