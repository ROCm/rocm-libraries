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

## The flows

### `rtc-kernel-review`

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

### `test-engine-kernel`

Hand it a graph; it makes `TEST_ENGINE` run that graph. `TEST_ENGINE` lives in
`dnn-providers/hip-kernel-provider/src/engines/test_engine`: the engine, its plan
builders and its kernel compilation path all work, and its kernel source is a set of
empty stubs, so it runs and computes nothing. An agent fills that in, the provider is
built, the integration cases covering the graph's operations have to pass, and the
graph is then timed against `HIP_MLOPS_ENGINE` -- the engine `TEST_ENGINE` duplicates
-- which the new kernels have to **beat** by a stated margin.

```
profile           python            -> which suites cover this graph
reference_probe   graph_bench       -> HIP_MLOPS_ENGINE runs it, so a baseline exists
baseline_coverage integration_tests -> how many cases TEST_ENGINE accepts today
scope_baseline    python            -> hash the files that define the measurement
loop kernel_cycle (max 6, until ${steps.compare.outputs.meets_target} == 1)
  implement   claude            -> writes the engine in the checkout + implement.json
  scope       python            -> the engine changed; nothing that measures it did
  build       cmake --build     -> the provider compiles
  integration integration_tests -> zero failures, no fewer passes than the floor
  bench_new   graph_bench       -> the graph, 50 iterations, pinned to TEST_ENGINE
  bench_ref   graph_bench       -> the same graph, same 50, HIP_MLOPS_ENGINE
  compare     python            -> new/ref median ratio vs the speed target
```

Each stage gates the next. A stage that fails abandons the iteration and records the
failing step and its log directory in `feedback.md`; the prompt tells the agent to
open that directory, because the one-line note is not the evidence -- the compiler
diagnostic and the failing test names are.

**Nothing in it is specific to one operation.** `scripts/graph_profile.py` maps the
graph's node types to the bundle suites that exercise them (`BatchnormInferenceAttributes`
→ `quick_BatchnormInference_*`, `MatmulAttributes` → `quick_Matmul_*`, a fused graph
to both), the graph is what gets benchmarked, and the coverage floor is measured from
this checkout at the start of the run. Point it at a batchnorm graph and the agent
writes batchnorm kernels; point it at a graph whose operation `TEST_ENGINE` cannot
plan for yet and the job includes porting that plan family from `HIP_MLOPS_ENGINE`.

Four things this flow does that are worth copying:

**Fail fast on the things that make the run impossible.** Before any agent launches:
the derived filter has to select tests, the reference engine has to actually run the
graph, and `TEST_ENGINE` has to accept at least one selected case. Each of those would
otherwise surface hours in, as a gate that cannot be satisfied for reasons that have
nothing to do with the kernel.

**A measured coverage floor, not "zero failures."** `Failed: 0` is also what a run
that skipped everything prints, and an engine makes a case skip by declining the
graph. So `baseline_coverage` runs the suite once up front and the loop requires
`passed >= baseline passed + baseline failed` -- pass at least as many cases as the
engine accepted before the run started. Measured on the batchnorm graph, filter
`quick_BatchnormInference_*` selects 72 cases and stub `TEST_ENGINE` reports
`Passed 0, Skipped 24, Failed 48` -- so the floor is 48, measured rather than written
down anywhere. The token is anchored between `_` separators on purpose: an unanchored
`quick_*BatchnormInference*` also drags in `BatchnormInferencePointwise`,
`BatchnormInferenceAttributesVarianceExt` and `BatchnormInferencePointwiseBatchnormBackward`
-- 434 cases, most of them fusions of a different operation.

**The scope guard, both directions.** Every gate above can be made green by editing
the wrong file: weaken a tolerance in the harness, edit a bundle's shapes, add a
`test_skips` entry to the engine TOML, or "fix" the kernel by changing the reference
engine it is compared against. None of that shows up as a failing test. Every gate
above can *also* be passed by an agent that wrote nothing at all -- the build
succeeds, the suite reports what it reported last round, the benchmark times the
previous engine. `scripts/guard_scope.py` hashes the measurement files and the engine
files before the loop and re-hashes both after every agent step: the first set must be
unchanged, the second must not be. It runs before the build, so a violation costs
seconds rather than an hour.

**Arithmetic lives in a step.** The condition grammar compares numbers and cannot
divide them, so the ratio is computed by `scripts/compare_bench.py`, which writes one
integer the loop's `until` reads. That script also refuses to compare two runs that
disagree on graph, iteration count or timing method -- comparing a 50-iteration run
against a 5-iteration one produces a number, and that number is the bug.

#### The speed target is what stops it copying the reference

`perf_target_ratio` is the largest acceptable `new/ref` median ratio, and it reads in
either direction: `1.15` allows a 15% regression, `1.00` demands parity, and the
default `0.80` demands **at least 20% faster**.

That default is not decoration. The first real run of this flow passed every gate on
iteration 1 at ratio **1.0046** -- and the agent's own transcript shows why: it `cp`'d
the three `HIP_MLOPS_ENGINE` batchnorm kernel sources, renamed them, and repointed the
plans. A correct answer, and a ratio of ~1.0 by construction. Under a parity-or-better
bar that is a pass; under `0.80` it cannot be, so the loop has to find something the
generic reference kernel leaves on the table. The prompt says this outright rather
than leaving the agent to discover it after four wasted iterations.

Room to move the bar: the reference engine's run-to-run spread against itself on the
benchmark graph is ~1.5% (medians 4.355 / 4.346 / 4.410 ms), so anything from `0.98`
down is comfortably outside the noise.

#### Size the graph you hand it

`hipdnn_graph_bench` times submit-plus-drain, which has a floor of roughly 0.03 ms on
gfx1151. On `test-graphs/batchnorm_inference_fp32_nchw.json` (2352 elements),
`HIP_MLOPS_ENGINE` and a stub `TEST_ENGINE` both measure ~0.03 ms: the benchmark
cannot tell a good kernel from no kernel. Use
`test-graphs/batchnorm_inference_fp32_nchw_xl.json`, where the same pair reads
4.35 ms against 0.11 ms. `test-graphs/README.md` has the full measured sizing ladder,
including why `bench_warmup` defaults to 10 at that size.

```bash
$P orchestrate.py run configs/flows/test-engine-kernel.yaml \
    --input graph=test-graphs/batchnorm_inference_fp32_nchw_xl.json --tee
```

This flow writes into the checkout, not into the run directory -- a kernel that is not
in the tree cannot be built. Run it in a worktree you are willing to have edited, and
expect `git status` to show changes under `test_engine/` afterwards.

### `ingestor-engine-kernel`

Hand it a graph and a name for a new engine. It produces a hipRTC kernel that computes
the graph, lands that kernel as a **new generic-kernel-ingestor pack** -- native symbols,
descriptors, registration with the shared test suite, bundle graphs -- and proves the
result from the installed tree with a complete corpus accounting. Three agents, each
running one skill from `projects/hipdnn/tools/ai/skills`, in order:

```
preflight   device_arch, profile, identity, configure, cache_check, build, comgr, venv, install, hash
loop author_cycle      (max 4, until mismatched_outputs == 0)
  author         hipdnn-kernel-authoring    -> authoring.json
  author_scope   nothing in the product tree changed
  author_contract  the contract describes the files it points at
  rtc_compile    hipRTC compiles it here, resolves every entry point, proves every guard
  numerics       the orchestrator runs the agent's harness and reads its report
loop integration_cycle (max 5, until gate.meets_target == 1)
  integrate      hipdnn-kernel-integration  -> integration.json
  scope, placeholders, integration_contract, build, install
  validate_descriptors, census, suite, absent_control, ctest_listing, gate
loop ingestor_cycle    (max 3, until corpus.meets_target == 1)
  ingest         hipdnn-ingestor-engine     -> ingestor.json
  ingest_contract, device_probe, final_validate, final_suite, corpus
```

```bash
$P orchestrate.py run configs/flows/ingestor-engine-kernel.yaml \
    --input graph=test-graphs/conv_fwd_pointwise_fp32_nchw.json \
    --input engine_name=hipkernel:ConvPointwiseRtc \
    --input corpus_dir=test-graphs \
    --input arch=gfx1151 --tee
```

`arch` is required and has no default. It reaches `-DGPU_TARGETS`, the pack and every
device gate, and nothing between the start of the run and the first kernel launch can
tell a wrong value from a right one -- hipRTC returns `HIPRTC_SUCCESS` compiling for an
architecture the box does not have, so `rtc_compile` stays green and the mistake only
lands in `numerics`, after a superbuild and up to four agent rounds. The `device_arch`
preflight now compares it against `hipInfo`'s `gcnArchName` and fails in under a second
if they disagree.

`build_parallel` defaults to 16 (AGENTS.md's cap) for both superbuilds; raise it on a
bigger machine.

It configures its own build tree (`build-ingestor`) and install prefix
(`install-ingestor`) with `HIPDNN_ENABLE_KERNEL_INGESTOR=ON` and
`HIPKERNELPROVIDER_ENABLE_ROCKE=ON`, both of which are OFF by default and neither of
which any preset sets. It writes into the checkout from stage two onwards.

#### The three contracts are the handover

`authoring.json`, `integration.json` and `ingestor.json` are the only things that pass
between the agents. No agent reads another's prose, and each is handed the previous
contract file by path. `scripts/contract_check.py` then checks that the contract
describes the checkout it claims to -- every path opened, every symbol looked for in the
source that should define it.

#### Five gates worth copying

**The orchestrator compiles the kernel itself.** `scripts/rtc_compile.py` loads hipRTC
through `ctypes`, compiles every source the contract lists for the named architecture,
and asks the code object whether each declared entry point is in it -- by the *literal*
name, because the descriptor stores a plain string and `getKernel()` looks that string
up. A kernel that forgot `extern "C"` compiles, resolves under its mangled name, and is
unreachable; measured on a deliberately-mangled kernel, the gate reports it as an
unresolved symbol rather than as a pass.

**Every `-D` macro is proved to be guarded by a negative compile.** For each entry of
`required_defines` the source is compiled again with that one macro dropped, and the
compile is required to *fail*. A macro whose absence still compiles is not a
specialization axis: it is a token that silently means whatever it happens to mean, and
the failure shows up only in the numbers.

**The launch ABI is compared across the seam.** The authoring agent chose the kernel's
parameter list and the integration agent wrote `launch()`; nothing in the toolchain
compares them. hipRTC compiles the kernel, `getKernel()` resolves it, and
`hipModuleLaunchKernel` reads one pointer per parameter the kernel declared, so a short
argument list reads whatever is next in memory and two same-typed pointers swapped is a
wrong answer with no diagnostic. `contract_check.py` compares `launch_arg_order` against
the kernel's parameter names element by element, and the integration contract must echo
the sha256 of the authoring contract it consumed -- an ABI taken from a previous round's
kernel fails on the digest.

**The absent-engine control.** Because an unservable case *skips*, a `--test-engine` run
in which the engine was never loaded at all is indistinguishable from one in which it
served everything: both exit 0. So the same suite is run a second time naming an engine
that does not exist, and is required to fail with `Error: Engine '<name>' is not loaded.`
and exit 1. If that control does not fail, the positive run proved nothing, and
`integration_gate.py` says so in those words.

**The feature flags are read back out of the cache.** `HIPDNN_ENABLE_KERNEL_INGESTOR=OFF`
compiles `discoverDescriptorSets()` out entirely, so the engine does not exist, every
case skips and the suite exits 0 -- identical, at the exit code, to a matcher that
declined everything. `scripts/cache_check.py` asserts both flags in the configured cache
before any agent launches, and `validator_present` then confirms the build actually
produced `hipdnn_validate_descriptors`, which exists only under that flag. Two
independent witnesses, because a configure that exits 0 is not one.

#### Append-only is a third scope category

`guard_scope.py` classified `added` as a violation alongside `modified` and `removed`.
That is right for the measurement harness and wrong for the bundle tree, which this flow
*requires* its agent to add cases to -- `import_graph.py` either creates a new
template+sweep directory or appends a case to an existing topology's `sweep.json`. Not
watching the bundle tree instead leaves the obvious cheat open: edit an existing case's
shapes, or widen a tolerance, until the pack that cannot serve them passes.

So `--allow-added GLOB` and `--allow-grow GLOB` add the category between "must not
change" and "unwatched". A file under `--allow-grow` is forgiven only when its baseline
JSON is a structural *prefix* of its current content: lists may gain elements at the end
and dicts may gain keys, and every element that was already there must be deep-equal.
Baseline mode records those files' text, not just their hash, because proving growth
needs the old content; a baseline written before the flag simply has no record, and the
file stays a violation. Measured against this checkout, appending a case to
`quick/Pointwise/Nchw/sweep.json` reports `grown 1, outside 0`, and editing the case that
was already in it reports `outside 1` with its own distinct feedback paragraph.

#### What it does not prove

The stage-one numerics gate runs a harness the *agent* built. `run_report.py` deletes the
report before launching, records the harness's digest and refuses an interpreter or a
shell as `harness_command[0]`, which makes a stale or trivially fabricated report visible
-- but the harness is still the agent's code. The correctness claim this flow actually
stands on is stage two's: the shared integration suite, against the same reference
executor and the same bundle corpus every other provider is held to, from the install,
pinned to the engine by name, with the absent-engine control beside it.

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

`stdout.log` is byte-exact. For **agent steps it is JSONL**: they run with
`--output-format stream-json --verbose`, so the CLI emits one event per turn *as it
happens* rather than buffering the whole session and dumping it at exit. That is what
makes a three-hour step observable instead of a process that either finishes or does
not. `stdout.pretty.json` is still written alongside any step whose stdout parses as a
single JSON document; a JSONL stream does not, so agent steps no longer get one --
`scripts/agent_log.py` is what reads them.

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

## Watching a run

Agent steps are the long ones, and their log is machine-shaped. `scripts/agent_log.py`
renders it:

```bash
$P scripts/agent_log.py                 # newest run, newest iteration, implement step
$P scripts/agent_log.py --follow        # ...and keep reading while it runs
$P scripts/agent_log.py --no-thinking   # tool calls and messages only
$P scripts/agent_log.py --run runs/test-engine-kernel/<stamp> --step review --iter 2
```

```
== kernel_cycle\iter-00\implement  (20260915T033903Z-a232)
-- session afb706e5-aa4d-4ee8-aa25-0e407343fbb6  cwd D:\...\agent_kernel_integration_poc
   . The plans compile TestEngineBatchnormNoop.cpp.
   . I need to check the launch argument order first.
   Reading the plan sources.
  -> Read  file_path=.../plans/batchnorm/BatchnormFwdInferencePlan.cpp
  -> Grep  pattern=getKernel\(
  !! error: File does not exist: .../nope.cpp
-- done  7 turns  94s  $1.23  stop=end_turn
```

Lines starting `.` are reasoning, `->` is a tool call, `!!` is a tool error, and the
last line is turns, wall time and cost. `--tee` on the run itself shows the same events
live but raw, one long JSON line each; this is the readable form, and it works on a
finished run too.

### Reopening the conversation

Every agent step records the CLI's own `session_id` as a step output, so `run.json`
carries it and the session stays reachable after the run:

```bash
$P scripts/agent_log.py --session-id            # -> afb706e5-aa4d-4ee8-aa25-...
claude --resume afb706e5-aa4d-4ee8-aa25-...     # from the step's cwd
```

That is the difference between a run that leaves a verdict and one that leaves the
reasoning behind it. Ask the session why it chose a tile size; the result file cannot
tell you.

## Driving flows from Graph Studio

`flowmcp/` is an MCP server over the same engine: it launches a flow, supervises
the worker, and serves the run as it happens. Graph Studio's **Implement** tab is
one client of it; a Claude Code session with the server registered is another,
and drives the same flows with no app running.

The SDK it needs is deliberately not in `requirements.txt` — `runner/` stays
PyYAML-only and the test suite runs without it. Install the extra into the same
virtualenv:

```bash
$P -m pip install -r requirements-mcp.txt
```

Then point Graph Studio at this checkout and start it:

```bash
cd ../graph-studio
bun install
bun run electron:start
```

The tab resolves this orchestrator by walking up from its own directory, so a
normal checkout needs no configuration. `HIPDNN_ORCHESTRATOR_DIR` overrides it.
The registry it passes the server is `configs/tools.local.yaml` when that exists,
otherwise `configs/tools.yaml` — so a machine-local registry is picked up without
editing the shared one.

**If the tab's controls are disabled**, it is telling you which of those is
missing: it reports the reason rather than failing silently. The usual answer is
that `requirements-mcp.txt` was never installed, or that `tools.yaml` points at
an executable this machine does not have.

`fast-converge` is the flow to try first. It runs no agent, finishes in about two
seconds, and exercises the whole path — launch, live timeline, artifacts,
cancellation — without spending a session.

Run the server by hand to see what a client sees:

```bash
$P -m flowmcp.server --tools configs/tools.local.yaml \
   --flows-dir configs/flows --run-root runs
```

It speaks MCP on stdio, so it is not meant to be read directly; `--help` lists
the options, of which `--max-concurrent` and `--max-iterations-ceiling` are the
two worth knowing. Both bound what a caller can spend.

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
