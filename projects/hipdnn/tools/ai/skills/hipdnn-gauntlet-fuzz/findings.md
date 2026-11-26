# Formats

Formats for lane reports, decisions, verifier files and the ledger. `scripts/ledger.py` parses them: write every block as plain
lines, never inside a code fence, each label starting its line (no bullet, bold or quote). A blank line or a `##` heading ends a
block; fence any command or log inside a field. Mark anything not observed directly `[INFERENCE]`. Cite repository paths as
`path:line` read at the pinned commit, and run outputs relative to the round directory (`r<k>/...`); never write absolute local paths.

## Lane report

Every lane writes one file, `r<k>/lanes/<LANE>.md` (search lanes) or `r<k>/triage/<LANE>.md` (triage lanes):

```markdown
# <LANE>: <one-line scope>
Read:     the files and line ranges read, the runs and folders opened
## Map            (only lanes whose brief asks for a table first: KMAP, NUM, DIFF, GRIDYZ)
## Ideas            (search lanes)   or   ## Findings   (triage lanes)
<blocks, one blank line apart>   or   None: <why nothing was found>
## Checked, no idea / Checked, no finding
<what was looked at and ruled out, one line each, so the next round does not redo it>
## Handoffs
<LANE or ORCH>: <what another lane, or the orchestrator, should look at, and why>
```

Provisional IDs are `<LANE>-p<n>`, `n` from 1 in the file. The orchestrator assigns final IDs (`ledger.py ids`).

## Idea (search lanes)

ID:        <LANE>-p<n>
Graph:     the concrete graph: op, axis values, tensor dims and strides, attributes; one graph, not a family
Why:       why it could break, with path:line evidence
Expect:    PASS | DECLINED | WRONG | CRASH | ACCEPT-UNSERVABLE | DECLINE-SERVABLE | FRONTEND-GAP | BACKEND-MANGLE | KERNEL-UNREACHED
Judgeable: yes | bounded: <the input range that keeps it judgeable> | no: <why no fixed tolerance can judge it>
Covered:   yes: <space.axis> | partly: <space.axis, what is missing> | no
Place:     axis <space>.<axis>=<values> | probe <space>.<family> expect <declined|declined_or_pass> | kernel-rows <kernel> | hand-probe | unreachable: <reason>

`Expect` values that decide much (`ACCEPT-UNSERVABLE`, `DECLINE-SERVABLE`, `FRONTEND-GAP`, `BACKEND-MANGLE`, `WRONG`, `CRASH`) are
**decisive**: an idea with one of them is verified before it changes a space ([RUNBOOK.md](RUNBOOK.md) *Decide*).
`Place` is the lane's proposal; the orchestrator decides.

## Decision (orchestrator, `r<k>/DECISIONS.md`)

One table row per idea, by final ID:

| ID | Decision | Where | Evidence checked | Result |
|---|---|---|---|---|
| `KMAP-3` | `kernel-rows` | `FlydslBwdSplitKernels` rows 1-9 | read `<kernel generator>:212`; verify `CONFIRMED` | filled after the run: outcomes of those rows |

`Decision` is one of `axis`, `probe`, `kernel-rows`, `hand-probe`, `unreachable`, `dropped`, `known` (already in the ledger: cite
its ID in `Where`). `dropped` and `unreachable` say why in `Evidence checked`.

## Finding (triage lanes)

ID:        <LANE>-p<n>
Kind:      defect | regression | applicability | conditioning | tolerance | divergence | pre-existing | harness-gap | unknown
Severity:  HIGH | MEDIUM | LOW | INFO
Groups:    the group names from r<k>/groups.tsv this finding explains, comma-separated
Graphs:    how many rows, in which runs and spaces; the axis signature they share; one example case and its failure folder
What:      per failing tensor: mismatch count and percent, max abs diff against the tolerance in force, finite or not
Proof:     what was run or read, each with its output path under r<k>/ (replays, A/B on base, domain A/B, class check, both references, rule ids)
Verdict:   the reasoning that ties the proof to the kind; "probably" when [policy.md](policy.md) *Proof per kind* is not fully met
Fix:       the smallest change that resolves it, or the decision a human must make

`Kind` and `Severity` follow [policy.md](policy.md). One finding per cause: two groups with one cause share a finding.

## Verifier output (`r<k>/verify/<ID>.md`)

ID:        <final ID>
Verdict:   CONFIRMED | REFUTED | PARTIAL | UNVERIFIED
Checked:   what was re-read or re-run, with output paths
Claim:     the narrowed claim (PARTIAL only), else -
Note:      what the reporter missed, or what would settle an UNVERIFIED verdict

A second verifier writes `r<k>/verify/<ID>-2.md`; where it differs it replaces the first.

## Groups (`r<k>/groups.tsv`)

Tab-separated, one line per problem row (WRONG, CRASH, UNEXERCISED, probe violation), header `run space case group`. Groups come
from `collect_report.py`'s categories and AOTriton-rule groups, then are split or merged by the orchestrator so one group has one
likely cause. Every group must be named in some finding's `Groups:` field (`ledger.py gate triage`).

## Ledger (`<run-dir>/engines/<engine-slug>/LEDGER.md`)

One table, one row per idea or finding across every round and PR for the engine. `ledger.py ids` adds rows and `ledger.py gate
verify` sets verdicts; the orchestrator edits `Status` by hand when a round closes a row.

| ID | Type | Kind | Severity | Verdict | Status | Round | Summary |
|---|---|---|---|---|---|---|---|
| `GROUP-4` | finding | divergence | LOW | CONFIRMED | DOCUMENTED | pr<N>/r3 | fp16 subnormal flush: dV exact zero where P < 2^-14 |

- `Type`: `idea` or `finding`. Ideas carry their `Expect` in `Kind` and a `-` severity.
- `Verdict`: the verifier's, `-` until verified.
- `Status` for ideas: `PROPOSED`, then the `Decision` (`AXIS`, `PROBE`, `KERNEL-ROWS`, `HAND-PROBE`, `UNREACHABLE`, `DROPPED`, `KNOWN`).
- `Status` for findings: `OPEN`, `FIXED` (a later round shows it gone, cite the round), `DOCUMENTED` (a known divergence, cite the
  doc), `ACCEPTED` (the user decided to live with it), `REFUTED`, `PRE-EXISTING` (also on base or the shipped engine).
- `Round`: `pr<N>/r<k>` (or `<engine-slug>/r<k>`) where the row was first filed. Later rounds update the row; they never add a twin.
