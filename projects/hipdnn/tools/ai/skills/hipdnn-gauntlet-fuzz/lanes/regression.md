# Lane: base vs PR transitions (`REGR`)

**Trigger:** a baseline exists (the engine serves the arch on base). **Agent:** `task`. **Output:** `R/triage/REGR.md`, findings.

## Method

1. Transitions from `analyze_runs.py` on **primary rows only** (pairwise ids below the space's `rows`; repair rows are different
   graphs per run) and on same-seed random runs. Confirm each pair is the same graph (same axes in both outcome records).
2. Classify every transition:

| Base | PR | Kind | Action |
|---|---|---|---|
| PASS | WRONG, CRASH, DECLINED | `regression` | replay `row.json` solo on both builds with `--row`; `--shrink 20` on PR; read `log_tail.txt` |
| DECLINED | PASS | new acceptance | good if the PR claims it (PR description, coverage doc); else an `applicability` finding |
| DECLINED | WRONG, CRASH | new acceptance, wrong | hand the rows to the group that holds them; note the transition |
| WRONG | WRONG | `pre-existing` | note; not the PR's |
| probe violated on both | | `pre-existing` gap | list the families |
| probe violated on PR only | | the PR exposed or caused it | hand to `PROBE` |

3. A WRONG on a repair row has no base twin: replay its `failures/<case>/row.json` on both builds.
4. File one finding per cause; unclaimed new acceptances are one finding listing the families.
