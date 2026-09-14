<!-- Copyright © Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier:  MIT -->

# Orchestrator

Runs agent pipelines with a bounded repair loop. Point it at a hipDNN graph; it drives
agent steps and ordinary command-line tools, feeds each step's structured output into
the next, and repeats a loop until a **measured** exit condition holds.

`DESIGN.md` is the design rationale. This file is how to operate it.

## Setup

```bash
cd projects/hipdnn/tools/orchestrator
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt         # Linux
```

Only PyYAML is required at runtime; pytest is for the test suite.

## The two config files

| File | Owns | Changes when |
|---|---|---|
| `configs/tools.yaml` | *Where* each executable lives on this machine (`exe`, plus `env`/`path_prepend` when a tool cannot launch without them) | You move to a different machine |
| `configs/flows/*.yaml` | *What* gets run: steps, arguments, prompts, declared outputs, the loop and its exit condition | The workflow changes |

Arguments never appear in the registry. The loader rejects `args`/`cwd`/`timeout`/`stdin`
there and tells you which step they belong on.

Prompts live in `configs/prompts/*.md`, not inside YAML, and are rendered with the same
`${...}` references as the flow.

## First flow: `rtc-kernel-review`

Generate a HIP RTC kernel for a graph, then have a **second, independent** agent review
it. Critical issues are appended to a shared feedback file and the generator runs again;
a review with zero critical issues ends the run.

```
loop review_cycle (max 3, until ${steps.review.outputs.critical_count} == 0)
  generate  claude -> ${run.dir}/kernel/kernel.hip  + generate.json
  review    claude -> review.json (verdict, critical_count, critical_issues, feedback)
```

The reviewer is a fresh session on purpose -- `--resume`ing the generator's session
would give the reviewer the generator's blind spots and get its own work rubber-stamped.
Continuity across iterations comes from two files instead: the kernel on disk, and
`feedback.md`.

## Running it

```bash
P=.venv/Scripts/python.exe          # or .venv/bin/python

$P orchestrate.py inputs   configs/flows/rtc-kernel-review.yaml      # what it needs
$P orchestrate.py validate configs/flows/rtc-kernel-review.yaml      # schema + every ${ref}
$P orchestrate.py doctor   --flow configs/flows/rtc-kernel-review.yaml  # can I find the tools?

# Resolved argv, env and fully rendered prompts. Launches nothing.
$P orchestrate.py run configs/flows/rtc-kernel-review.yaml --dry-run \
    --input graph=<path-to-graph.json>

# For real.
$P orchestrate.py run configs/flows/rtc-kernel-review.yaml \
    --input graph=<path-to-graph.json> --input arch=gfx1151 \
    --input notes=@my-constraints.md
```

Useful switches: `--max-iterations 1` (single pass while debugging), `--tee` (mirror step
output to your terminal), `--run-dir DIR`, `--only ID`, `--from ID`, `--profile NAME`.

**Read the rendered prompts with `--dry-run` before the first real run.** Also confirm
the `claude` arguments in the flow (`--permission-mode acceptEdits`, `--add-dir`) match
how you want the agent to be allowed to touch this checkout.

## What a run leaves behind

```
runs/<flow>/<utc-stamp>/
  run.json          resolved inputs/vars, every step: status, exit code, duration, outputs
  inputs.json       exactly what this run was asked to do
  feedback.md       accumulated review feedback, one section per failed iteration
  kernel/           the generated kernel (this flow's ${vars.kernel_dir})
  review_cycle/iter-00/<step>/{cmd.txt,argv.json,stdin.txt,stdout.log,stderr.log,result.json}
  review_cycle/iter-01/...
```

`stdin.txt` is the prompt the agent actually received, after interpolation. When a run
goes wrong, read that first: a reference that resolved to something unexpected is
invisible in the flow file and obvious here.

`runs/` is gitignored.

## Writing a flow

Step keys: `id`, `tool`, `args`, `env`, `cwd`, `timeout`, `stdin` | `prompt_file`,
`result_file`, `result_schema`, `outputs`, `assert`, `when`, `expect_exit`, `retries`,
`continue_on_error`, `after`.

References: `${inputs.x}`, `${vars.x}`, `${env.X}`, `${platform.x}`, `${run.dir}`,
`${step.result_file}`, `${steps.<id>.outputs.<name>}`, and inside a loop
`${loop.iteration|attempt|attempt_dir|feedback_path|max_iterations}` and
`${loop.previous.<step>.outputs.<name>}` (empty string on the first iteration).

Every reference is resolved strictly -- including references inside prompt files, which
are checked by `validate` before anything launches. Nothing renders as a silent empty
string.

### Agent steps: use the result-file contract

Do not parse an agent's prose. Declare `result_file` (exposed to the prompt as
`${step.result_file}`, so instruction and extractor cannot disagree), list the keys you
require in `result_schema`, and read them with `json_file` extractors. A missing file,
invalid JSON or a missing key fails the step; `retries: N` re-asks.

### Conditions

`when`, `assert.that` and `loop.until` share one closed grammar -- no `eval`:

```
<operand> <op> <operand> [and|or ...]        ops: == != < <= > >= contains matches
```

Operands are references, numbers, quoted strings or barewords, optionally summed with
`+`. Numeric-looking strings compare numerically (`"10" > 9` is true).

### Make the loop's exit condition measurable

`until` should read a number a *program* produced, not an agent's self-assessment. For
the review loop that is `critical_count` from a schema-checked result file, with asserts
that reject a self-inconsistent review (`verdict: pass` alongside listed issues).

This matters more once the loop reaches the integration suite: `ctest` with a label
matching nothing exits 0, a gtest filter matching nothing prints `PASSED 0 tests`, and
an engine that supports nothing reports `Passed: 0, Skipped: 6772, Failed: 0` and exits
0. An agent asked "did it pass?" says yes to all three. Assert that work actually
happened.

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q
```

Hermetic: the only executable launched is this Python, standing in for an agent CLI.
