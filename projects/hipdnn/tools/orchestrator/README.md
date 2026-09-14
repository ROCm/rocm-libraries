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
runs/<flow>/<utc-stamp>-<suffix>/
  run.json          resolved inputs/vars, provenance, every step: status, exit code, duration, argv, outputs
  inputs.json       exactly what this run was asked to do
  feedback.md       accumulated review feedback, one section per failed iteration
  kernel/           the generated kernel (this flow's ${vars.kernel_dir})
  review_cycle/iter-00/<step>/{cmd.txt,argv.json,stdin.txt,stdout.log,stdout.pretty.json,stderr.log,result.json}
```

`stdout.log` is byte-exact. `stdout.pretty.json` is written alongside it when stdout
parses as JSON -- an agent CLI in `--output-format json` mode emits its whole session on
one line with the answer buried in an escaped string, and embedded multi-line strings
are split into real lines so the final message is readable.

`stdin.txt` is the prompt the agent actually received, after interpolation. When a run
goes wrong, read that first: a reference that resolved to something unexpected is
invisible in the flow file and obvious here.

`runs/` is gitignored.

`run.json` is rewritten atomically after every step transition, so a run that dies in
its third hour still leaves a usable report, and a run in progress can be read while it
is still going. It also records `provenance`: the sha256 of the flow, the tool registry
and each prompt file, plus the checkout revision when one is available.

A run directory is created, never joined. The timestamp carries a random suffix because
two runs starting in the same second is ordinary, and `--run-dir` pointing at a
directory that already holds a run is refused rather than merged into it.

## Writing a flow

Step keys: `id`, `tool`, `args`, `env`, `cwd`, `timeout`, `stdin` | `prompt_file`,
`result_file`, `result_schema`, `outputs`, `assert`, `when`, `expect_exit`,
`continue_on_error`, `after`.

Output extractors: `regex`, `json`, `json_file`, `file`, `glob`, `tail`, `lines`,
`sha256`. Output types: `string`, `int`, `float`, `bool`, `path`, `json`, `count`
(the length of an extracted list or mapping). A scalar type handed a list or mapping is
an error, not a silent pass-through -- `type: int` holding `[1, 2]` makes every later
comparison against it meaningless.

`stdout` and `stderr` are implicit outputs alongside `exit_code`, `duration_s`,
`stdout_path`, `stderr_path` and `workdir`. The text ones are read from disk on demand:
a multi-megabyte build log stays evidence on disk rather than being held in the run's
state because some later condition might read it.

Loop keys: `max_iterations`, `until`, `on_exhausted` (`fail` | `continue`),
`on_step_failure` (`fail` | `retry`), `feedback_from`.

`until` is the loop's **goal**, evaluated after each iteration over that iteration's
outputs. True → the loop is done and exits successfully. False → run the steps again,
up to `max_iterations`. So `until: "${steps.review.outputs.critical_count} == 0"` means
*keep generating and reviewing until a review finds zero critical issues*.

The log reports both the goal and what was measured against it, because the goal alone
never says why the loop kept going:

```
[review_cycle] iteration 1/3 is not done yet: ${steps.review.outputs.critical_count} == 0 (2 == 0) is false - looping to repair it
[review_cycle] iteration 2/3 is not done yet: ${steps.review.outputs.critical_count} == 0 (1 == 0) is false - looping to repair it
[review_cycle] done after 3 iteration(s): ${steps.review.outputs.critical_count} == 0 (0 == 0) is true
```

The parenthesised form is the condition with its references resolved: the reviewer found
2 critical issues, then 1, then none. Skipped steps and failed assertions report the same
way.

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
invalid JSON or a missing key fails the step.

**Denying an agent the tool it needs to honour the contract is a silent, expensive
failure.** The review step is denied `Edit`/`MultiEdit`/`NotebookEdit` so it cannot
rewrite the kernel it is judging, but keeps `Write` -- that is how it produces
`review.json`. Denying `Write` too made every review session end with "I completed the
review, but the Write tool is disabled", no result file, and a failed step. If you
tighten an agent's permissions, check the result contract still has a way to be met.

Belt and braces: both flows hash the kernel in `generate` and again in `review`
(`sha256` extractor) and assert the two match, so a reviewer that edits the kernel by
some other route is caught by evidence rather than trusted not to. The hash watches the
one designated path, and `generate` asserts that the path the agent reports is that same
file -- otherwise the reviewer reads one kernel while the integrity check hashes another.

### Retry is a property of the loop, not of a step

There is no per-step retry. A step that fails does not re-run in place: its inputs have
not changed, so the second attempt asks the same question of the same state and usually
fails the same way, at the same cost. Instead, `loop.on_step_failure: retry` abandons
the rest of that iteration and starts the next one, recording why in `feedback.md` so
the agents can see what went wrong. `on_step_failure: fail` (the default) stops the run.

A step whose failure is expected and informative -- a test binary that legitimately
exits non-zero -- should say so with `expect_exit: [0, 1]` or `continue_on_error`,
rather than being retried.

### Conditions

`when`, `assert.that` and `loop.until` share one closed grammar -- no `eval`:

```
condition   := disjunction
disjunction := conjunction ("or" conjunction)*
conjunction := clause ("and" clause)*
clause      := operand [OP operand]      ops: == != < <= > >= contains matches
operand     := term ("+" term)*
term        := ${ref} | number | 'string' | "string" | bareword
```

`and` binds tighter than `or`, as everywhere else, and both short-circuit. Numeric-looking
strings compare numerically (`"10" > 9` is true).

Conditions are **parsed when the flow loads**, so a malformed one is rejected before any
agent launches -- `validate` and `run` both catch it. Every position is checked: two
terms with nothing between them is a syntax error, not a silent concatenation. That is
the point. `${steps.x.outputs.failed} = 0` (one `=`) used to fold into the non-empty
string `"1=0"` and report an exit condition as satisfied while the step it measured was
failing. Concatenation must be spelled `+`.

### Make the loop's exit condition measurable

`until` should read a number a *program* produced, not an agent's self-assessment.

Where a number can only come from an agent, derive it from the thing it can be checked
against rather than from a second field the same agent wrote. The review loop counts
`critical_issues` with a `count` output and asserts that the reviewer's own
`critical_count` agrees; the loop exits on the derived value. A review that lists an
out-of-bounds write and reports `critical_count: 0` fails the step instead of ending
the run.

Note what this does and does not establish. A clean review means **review accepted**:
the response was well-formed, self-consistent, and the reviewer found nothing critical.
It is not a correctness claim. Nothing here has been compiled or executed. Correctness
arrives with a build and a run against a reference, not with agent judgement.

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
