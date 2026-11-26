# Report formats

## FINDINGS.md (`R/FINDINGS.md`, internal)

```markdown
# <PR N | engine> fuzz report: <engine, plain name>, round <k>

**Engine** `<id>` · **Arches** <arch (SKU), ...> · **Build** <base vs PR | PR only | engine only> · **Status** <closed | not closed: n HIGH open>

## Summary
**Verdict:** two or three sentences: does the change work, and what kinds of problems were found (bug in the change, older bug
the change exposes, older bug, fuzzer noise).
**Ask the PR author to:** numbered list, one line each, each linking its finding. **Not for the PR author (yet):** what needs
more evidence, and fuzzer noise. **How much was tested:** cases, executed vs declined, spaces, GPUs, wall time, crashes.

## Findings at a glance
| # | ID | Plain description (one line, no jargon) | Sev | Arch | How sure | PR-related |
One row per finding, worst first, ID linking to its section. Start with a one-paragraph key (kinds in one line each) and the row
counts by type. Never put the proof in the table.

## <ID>   (one section per finding, worst first; `###` heading = the ID so links and the ledger gate work)
**<Plain title: what goes wrong in about 10 words>** `SEV` `kind` `confirmed|probably|unknown` `arch` [triage](...) · [verifier](...)
**In one line:** one sentence a reader outside the project understands.
**Example graph:** one concrete graph on one line (dtype, mode, dims, strides, attributes).
**What goes wrong:** 3-5 bullets: the failing tensor, counts and percent, max diff against the tolerance in force, finite or not;
what passes (controls), what fails (signature), how many rows and where.
**Why:** the cause with `path:line`, or "not identified".
**Is it the PR's fault:** yes / partly / probably not / unknown, with the evidence (diff, base, twin graph); say when no baseline ran.
**Suggested fix:** the narrowest change that makes it go away; the decision a human owns, if any.
**Not shown:** what the proof still lacks. **Replay:** one command or folder path.
Harness-gap, conditioning and INFO findings get a short paragraph instead of the full template. An untriaged candidate is listed
last with its example folder and what would settle it. Keep each section under about 20 lines.

## What was tested
| Space | What it tests | <arch>: pass / wrong (declined) | ... | Findings |      (one row per space, all runs summed)
One line on the invalid-graph space. Kernel coverage as a table: kernel (short name) x arch, pass / wrong.

## Not tested or not reachable
bullets: features no space can build or the harness cannot grade, and what was not run, each with the reason

## Next
**Decisions for you** (policy.md *Decisions for the user*), then **Proposed round <k+1>** as a numbered list.

## Appendix   (each block in `<details><summary>...</summary>`, so the report stays short)
Build and runs (bullets: build, grading TOML and any rows the orchestrator added, runs per arch, invalid runs, baseline or not);
Oracle notes; Transitions base vs PR (when a baseline exists); full outcomes per run; ideas and where they went
(`| ID | Lane | Decision | Where | Result after the run |`); changes since the last round (ledger rows FIXED, newly OPEN, REFUTED).
```

Every statement traces to a ledger row, a lane report or a run file. Numbers are counts of rows with the run named.

## Findings folder (`R/report/`)

`collect_report.py` output (internal): `REPORT.md` (what ran, counts, links, Missing), `ISSUES.md` (categories,
then one row per problem worst first, with empty verdict and evidence columns), `analysis_<run>.txt`, `index.json`, and
`graphs/<run>/<Space>/<case>/` for every WRONG, CRASH, UNEXERCISED and probe violation (graph, fill metadata, `row.json`,
`repro.json`, `replay.sh`, `shrunk/`, log slice, `case.md`). Fill each row's verdict and evidence with the finding ID that explains
it. A WRONG with no failure folder is listed under Missing with a regenerated `row.json`. `replay.sh` holds local paths.

## Chat summary

Short, for the user:
1. One line: what ran (engine, arch, PR, rounds) and the headline (defects found or not).
2. The findings table (ID, kind, severity, verdict, one-line what), worst first.
3. Transitions base vs PR, when there is a baseline.
4. What could not be reached, and what was not run (arches, spaces).
5. Decisions the user owns, and the proposed next round.

## Anything that leaves the workspace

A zip for another reviewer, a PR comment draft, an issue: strip host names, user names, home paths, machine-pool or site names and
internal ticket ids, and grep the text for each before it goes out. Include a short reading guide. Never include dumps or failure
folders.
