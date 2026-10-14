# reports/ — campaign records

Each `issue-2521-*/` or `<date>-<slug>/` directory is one measurement campaign.

| Path | Committed? | What it is |
|------|-----------|------------|
| `README.md` | ✅ | the report: method, results, caveats |
| `summary.json` | ✅ | machine-readable per-cell results |
| `json/*.json` | ✅ | the per-cell harness output the summary is built from |
| `campaign_meta.json`, `provenance.json` | ✅ | provenance: commits, harness hash, environment |
| campaign-level logs (`sweep*`, `campaign*`, `console*`) | ✅ where small | driver console output |
| `attempt_logs/`, `build/`, `builds/`, `heal/`, `cache/`, `dfx_outputs/`, `*.so`, `*.o`, `*.pto`, `*.bin`, `*.lock` | ❌ gitignored | raw harness logs and build trees |

## Why the artifacts are not committed

Until 2026-10-14 the tree tracked **4.8 GB across 22k files, of which markdown was 0.4 MB** — raw
`attempt_logs/` alone was 4.24 GB (`[STRACE]` dumps, up to 21 MB per cell). The hygiene commit of
2026-10-14 untracked 21,005 artifact files, taking tracked content to **35 MB / 1049 files**. The
blobs remain in git history, and the files remain on disk wherever the campaign was run.

Policy, rationale and the provenance constraints: [`../AGENTS.md`](../AGENTS.md) §11.

## Recovering an untracked artifact

Nothing was lost — artifacts are reachable from history:

```bash
git log --oneline --all --diff-filter=D -- 'reports/<campaign>/attempt_logs/*'
git checkout <sha>^ -- 'reports/<campaign>/attempt_logs/'
```

Do **not** re-commit them. If a log excerpt is the evidence, paste the tail into the campaign
README instead — a raw file carries minutes of `[STRACE]` spam per useful line.
