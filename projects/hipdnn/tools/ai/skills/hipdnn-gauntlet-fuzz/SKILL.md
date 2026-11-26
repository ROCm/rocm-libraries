---
name: hipdnn-gauntlet-fuzz
description: "Fuzz a hipDNN engine, or a PR that changes one, with the integration gauntlet (RFC 0023): search lanes find what to fuzz, the oracle is calibrated, base and PR builds are fuzzed on a GPU, triage lanes explain every failure group, verifiers try to refute each idea and finding, and a per-engine ledger carries results across rounds. Use for 'fuzz PR N', 'fuzz engine X on gfx950', 'widen the fuzz net', or 'is this a regression or already on base'."
argument-hint: "[PR number | engine id] [arch: gfx942|gfx950|gfx90a|gfx1151|...] [ops/spaces] [budget: <fuzz minutes per run>] [round:<N>]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# hipDNN Gauntlet Fuzz

Execute [RUNBOOK.md](RUNBOOK.md), the only ordered workflow. You are the orchestrator: you pin the builds, run the search lanes,
interrogate and place their ideas, run the fuzz, group the failures, run the triage and verifier lanes, keep the ledger and write
the report. Lanes do the searching and the explaining; you decide. The other pages supply contracts:

| Page | For |
|---|---|
| [findings.md](findings.md) | Formats of lane reports, decisions, verifier files, groups and the ledger |
| [report.md](report.md) | The round report and the chat summary |
| [policy.md](policy.md) | What counts as a finding, and the proof each kind needs |
| [lessons.md](lessons.md) | Why the rules exist |
| `lanes/` | One brief per lane; [lanes/common.md](lanes/common.md) is the contract every lane follows |

## What the fuzzer is

The gauntlet is a Python driver over the integration-test bundle format. A **space** (`<Op>[.<Name>].space.json`) describes a
family of graphs as axes, derived values and constraints. The driver draws cases (random seeds, pairwise coverage or a kernel
catalog), writes them as sweep bundles, runs the unchanged `hipdnn_integration_tests` binary against one engine and records one
outcome per case: `PASS`, `DECLINED` (a clean typed refusal), `WRONG`, `CRASH` or `UNEXERCISED`. The contract it enforces: an
engine must either decline a graph cleanly or compute it correctly, never accept and be wrong.

Sources this skill drives (read the parts a stage names, not all of them):

| Source | For |
|---|---|
| `dnn-providers/integration-tests/docs/running-tests.md`, "Gauntlet fuzz runs", "Writing a space the oracle can judge", "Reports and offline grading" | driver flags, outcomes, failure folders, space rules, report tools |
| `dnn-providers/integration-tests/docs/file-formats.md` | the bundle and sweep format the spaces render into |
| `dnn-providers/integration-tests/gauntlet/` | `run_gauntlet_fuzz.py`, `analyze_runs.py`, `collect_report.py`, `tools/sdpa_bwd_aotriton_check.py`, `generators/`, `spaces/` |
| `projects/hipdnn/docs/ai-rules.md` | project rules |
| `projects/hipdnn/tools/ai/skills/hipdnn-superbuild/SKILL.md` | building hipDNN and its providers |
| `scripts/` (this skill) | `ledger.py` (IDs, ledger, gates), `calibration_rows.py`, `calibration_report.py`, `class_check.py`, `triage_wrong.py`, `make_variant_space.py` |

## Entry contract

Ask for what is missing; record it in `PIN.md` ([RUNBOOK.md](RUNBOOK.md) *Pin*):

| Input | Notes |
|---|---|
| Target | A PR number (base vs PR), or an engine id alone (one build, no baseline: every WRONG or CRASH is a finding to explain) |
| Arch | Every arch the change touches. Name the SKU for gfx942 (MI300X, MI325X and MI308X have different kernel objects). Know which machine runs each arch before you start |
| Ops | Forward, backward, norm, ... when the engine serves several |
| Budget | Fuzz minutes per run (`--time-budget`), and the GPU time the user approves |
| GPU access | How to run a command on a GPU of each arch. This skill has no machine-reservation tooling: the user supplies the way in, and approves the plan once as a whole |
| Subagents | The most subagents to run at once (default 8), if the agent host has any. Without subagents, run each lane yourself in turn |
| Round | Default: one more than the highest `r<k>/` in the run directory; 1 when none |

Paths: pick a **run directory** outside the source tree, one `r<k>/` per round. `<run-dir>/pr<N>/` holds a PR's rounds
(`<run-dir>/<engine-slug>/` without a PR). The engine record kept across PRs is `<run-dir>/engines/<engine-slug>/`
(`BRIEF.md`, `LEDGER.md`). `<engine-slug>` is the engine id with characters outside `A-Za-z0-9._-` replaced by `_`.

## Rules for the whole task

- Do not post on the PR, push to the author's branch, open issues or run CI. Do not edit the PR's code; describe fixes in the report.
- Run directories, ledgers and reports are working notes, not published text. Anything that leaves the workspace (a branch name,
  a commit message, a space description, a PR comment) names no host, user, machine pool or site: describe hardware as "an
  MI325X (gfx942) machine" at most.
- Cap build parallelism at 16 jobs. Release or finish every GPU job; prefer one-shot jobs over held machines.
- **WRONG is a measurement, not a verdict.** It means one output element was off by more than `atol + rtol*|reference|`. Never write
  WRONG without WHAT (tensor, mismatch count and percent, max abs diff against the tolerance in force, finite or not) and WHY (a
  finding with its proof, [findings.md](findings.md)).
- Do not wait for the user between stages. Stop only for what this contract reserves: GPU access not yet approved, anything that
  leaves the workspace, a change to the PR's code, or a [policy.md](policy.md) *Decisions for the user* item.

## Completion

A round is complete when every gate in [RUNBOOK.md](RUNBOOK.md) holds and `r<k>/FINDINGS.md` and the engine `LEDGER.md` are
written. The engine is **closed** for this PR when no `HIGH` or `MEDIUM` ledger row is `OPEN` or `UNVERIFIED` and every open
question has a decision. Otherwise the report lists what keeps it open and *Iterate* starts the next round; the user decides when
to stop.
