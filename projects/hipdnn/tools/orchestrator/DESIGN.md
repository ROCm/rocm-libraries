<!-- Copyright © Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier:  MIT -->

# Orchestrator tool — design plan (for review)

Location: `projects/hipdnn/tools/orchestrator/`
Language: Python 3 (stdlib + PyYAML), matching `IngestorGenerator`'s packaging shape.

## 1. What it is

A config-driven runner for **agent pipelines with a repair loop**. The target workload:
hand it a hipDNN graph JSON, and it drives agent → agent → agent, building and testing
between them, looping back with failure context until the integration actually validates
or an iteration budget is spent.

Three inputs, each with one job:

1. **Tool registry** (`tools.yaml`) — machine-local: *where each executable lives*, and
   the minimum environment wiring it needs to launch at all. No arguments.
2. **Flow** (`flow.yaml`) — machine-independent: steps, their arguments, their declared
   outputs, the loop and its exit condition.
3. **Run inputs** (`--input graph=…`) — what *this* run is about. The initial context.

Everything else (run directories, rendered prompts, logs, result manifest) is generated.

## 2. Tool registry schema

The registry answers exactly one question: **given the name `claude`, what do I exec on
this machine?** Arguments are never here — the same tool is called with different
arguments at different points in a flow, so arguments belong to the step.

```yaml
version: 1

vars:                                   # machine-local values, referencable everywhere
  repo_root: D:/develop/omp_workspace/worktrees/rocm-libraries/agent_kernel_integration_poc
  build_dir: "${vars.repo_root}/build"
  rocm_path: C:/Users/addickin/.../rocm_sdk

tools:
  claude:
    exe:                                # per-platform executable
      windows: C:/Users/addickin/AppData/Roaming/npm/claude.cmd
      linux: /usr/local/bin/claude

  cmake:
    exe: { windows: "C:/Program Files/CMake/bin/cmake.exe", linux: cmake }

  ctest:
    exe: { windows: "C:/Program Files/CMake/bin/ctest.exe", linux: ctest }

  integration_tests:
    exe: "${vars.build_dir}/bin/hipdnn_integration_tests${platform.exe_suffix}"
    path_prepend: ["${vars.rocm_path}/bin"]    # else STATUS_DLL_NOT_FOUND on Windows

  import_graph:
    exe: python
    # argv (script path, flags) is the step's business, not the registry's

  git:
    exe: git
```

Per-tool keys, complete: `exe` (string or per-platform map — the only required one),
`env`, `path_prepend`. `env`/`path_prepend` are admitted only because a tool can be
*unlaunchable* without them (the ROCm DLL search path above); they are location wiring,
not invocation policy. Anything an invocation chooses — args, cwd, timeout, stdin —
lives in the step.

`exe` resolution: absolute path used as-is; bare name resolved via `PATH` at run start,
not at exec time, so a missing tool fails before any step launches. `platform.*` exposes
`os`, `exe_suffix`, `path_sep`.

Optional `profiles:` block (e.g. `gfx1151-windows`, `ci-linux`) selected with
`--profile`; it overlays `vars` and per-tool `exe`/`env`/`path_prepend`.

## 3. Flow schema

```yaml
version: 1
name: kernel-integration

inputs:                                  # supplied per run; this is the initial context
  graph:
    description: hipDNN graph JSON to satisfy
    type: path
    required: true
  engine:
    description: Engine name the new kernel registers as
    default: RTC_AGENT_ENGINE
  arch:
    default: gfx1151

vars:                                    # flow constants, not run-time knobs
  tier: quick

steps:
  - id: build
    tool: cmake
    args: ["--build", "${vars.build_dir}", "--parallel", "16"]
    cwd: "${vars.repo_root}"
    timeout: 7200
    outputs:
      config_line: { regex: 'Build files .* written to: (.*)' }
```

**Step keys**: `id`, `tool`, `args`, `env`, `cwd`, `timeout`, `stdin`, `prompt_file`,
`after`, `outputs`, `result_file`, `result_schema`, `assert`, `when`, `expect_exit`,
`retries`, `continue_on_error`. The step is the *only* place an invocation is described; the
registry contributes nothing but the executable and its launch wiring. Step `env` merges
over the tool's `env`, step wins. `stdin` and `prompt_file` are mutually exclusive.

**Implicit outputs on every step**: `exit_code`, `stdout`, `stderr`, `stdout_path`,
`stderr_path`, `duration_s`, `workdir`.
**Declared extractors**: `regex` (+`group`, `type`), `json` (JSONPath-lite over stdout),
`json_file` (over a file the step produced — the one that matters for agents, §4),
`file`, `glob`, `lines`/`tail`.

**Reference syntax**: `${inputs.x}`, `${vars.x}`, `${env.X}`, `${platform.x}`,
`${steps.<id>.outputs.<name>}`, and inside a loop `${loop.*}` (§5). Resolved strictly at
load time — an unknown reference is a config error before any process starts, not a
runtime `None`.

**Asserts** use a closed comparison grammar (`<ref> <op> <literal|ref>`, ops
`== != < <= > >= contains matches`, joined by `and`/`or`). No `eval`. A flow file is
data.

## 4. Agent steps — context in, structured results out

Agent calls are most of this pipeline, so their plumbing is the feature, not a detail.

**Context in.** `inputs:` declares what a run needs; `--input graph=…/conv.json`,
`--input context=@brief.md` (`@` = file contents), or `--inputs-file f.yaml` supplies
it. A missing `required` input fails before anything launches, printing the input's
`description`. `type: path` is absolutized and existence-checked at load time. Prompts
live in files (`prompt_file: prompts/generate.md`), rendered with the same `${...}`
resolution and piped to stdin; the rendered text is written to
`<step-dir>/stdin.txt`, so what the agent actually received is always inspectable — a
prompt that silently interpolated an empty string is otherwise invisible.

**Structured results out — by file, not by stdout.** An agent's stdout is prose plus
whatever the CLI wraps it in. Parsing it for `kernel_path` is a trap. Instead the step
*dictates* a result path and validates it:

```yaml
  - id: generate
    tool: claude
    args: ["--print", "--output-format", "json"]
    prompt_file: prompts/generate_kernel.md     # tells the agent to write ${step.result_file}
    result_file: "${loop.attempt_dir}/generate.json"
    result_schema: { required: [kernel_path, entry_point, launch_notes] }
    retries: 2                                  # malformed/missing result => re-ask
    timeout: 3600
    outputs:
      kernel_path:  { json_file: result, path: "$.kernel_path", type: path }
      entry_point:  { json_file: result, path: "$.entry_point" }
      session:      { json: "$.session_id" }    # from the CLI's own JSON envelope
```

`result_file` is exported to the prompt as `${step.result_file}` so the instruction and
the extractor can never disagree about the path. A missing file, invalid JSON, or a
missing `required` key fails the step *before* any downstream step consumes garbage —
and `retries` makes the common case (agent forgot the file) self-healing.

**Threading agent to agent.** Two shapes, both plain refs: stateless chaining
(interpolate `${steps.generate.outputs.kernel_path}` into the next prompt) or session
continuation (capture `session_id`, pass `--resume ${steps.generate.outputs.session}`).
The runner models no conversation, which is why a second agent CLI is a registry entry
rather than a code change.

**Large payloads go by path.** Interpolating a test log into argv hits Windows' 32 KiB
command-line limit; pass `${steps.validate.outputs.stdout_path}` and let the agent read
the file. Validation warns above 8 KiB per argv element.

**Agent realities the runner tolerates**: hour-long wall clock (per-step `timeout`,
streamed logs, never buffered), non-zero exit on refusal (`expect_exit`), and
non-determinism (a re-run is a new run dir, never an overwrite).

## 5. The loop — the actual target flow

The pipeline is cyclic by nature: generate → ingest → validate → *fix what's broken* →
generate again. That is expressed as one `loop` step group, not `goto`:

```yaml
  - id: integrate
    loop:
      max_iterations: 3
      until: "${steps.validate.outputs.failed} == 0 and ${steps.validate.outputs.passed} > 0"
      on_exhausted: fail
    steps:

      # 1. Graph in, HIP RTC kernel out.
      - id: generate
        tool: claude
        args: ["--print", "--output-format", "json"]
        prompt_file: prompts/generate_kernel.md
        result_file: "${loop.attempt_dir}/generate.json"
        result_schema: { required: [kernel_path, entry_point, launch_notes] }
        retries: 2
        timeout: 3600
        outputs:
          kernel_path: { json_file: result, path: "$.kernel_path", type: path }

      # 2. Ingest that kernel into hipDNN via the ingestor skill; it owns its own
      #    host-side checks (descriptor/native census) and reports what it proved.
      - id: ingest
        tool: claude
        args: ["--print", "--output-format", "json"]
        prompt_file: prompts/ingest_kernel.md
        result_file: "${loop.attempt_dir}/ingest.json"
        result_schema: { required: [descriptor_dir, engine_name, checks_passed] }
        retries: 1
        timeout: 5400
        outputs:
          engine_name: { json_file: result, path: "$.engine_name" }

      # 3–5. Deterministic gates. No agent judgement below this line.
      - id: build
        tool: cmake
        args: ["--build", "${vars.build_dir}", "--parallel", "16"]
        cwd: "${vars.repo_root}"
        timeout: 7200

      - id: bundle
        tool: import_graph
        args: ["${vars.repo_root}/dnn-providers/integration-tests/migration-scripts/import_graph.py",
               "--graph", "${inputs.graph}",
               "--bundle-dir", "${vars.repo_root}/dnn-providers/integration-tests/integration-test-bundles/",
               "--tier", "${vars.tier}"]
        outputs:
          bundle_filter: { regex: 'registered suite:\s+(\S+)' }

      - id: validate
        tool: integration_tests
        args: ["--test-engine", "${steps.ingest.outputs.engine_name}",
               "--verification-mode", "cpu",
               "--gtest_filter", "${steps.bundle.outputs.bundle_filter}"]
        cwd: "${vars.build_dir}"
        expect_exit: [0, 1]                      # a real failure is data, not a crash
        outputs:
          passed:  { regex: 'Passed:\s+(\d+)',  type: int }
          skipped: { regex: 'Skipped:\s+(\d+)', type: int }
          failed:  { regex: 'Failed:\s+(\d+)',  type: int }
        assert:
          - { that: "${steps.validate.outputs.passed} + ${steps.validate.outputs.failed} > 0",
              message: "zero cases ran — engine declined the graph or the filter matched nothing" }

      # 6. Only now an agent, and only to write the next iteration's context.
      - id: feedback
        tool: claude
        when: "${steps.validate.outputs.failed} > 0"
        args: ["--print", "--output-format", "json"]
        prompt_file: prompts/diagnose.md          # reads ${steps.validate.outputs.stdout_path}
        result_file: "${loop.attempt_dir}/feedback.json"
        result_schema: { required: [summary, suspected_cause, next_action] }
        timeout: 1800
```

### 5.1 Loop semantics

- `until:` is evaluated **after** each iteration over that iteration's outputs. True →
  loop exits successfully. False → next iteration, up to `max_iterations`
  (`on_exhausted: fail|continue`, default `fail`).
- Loop-scoped refs: `${loop.iteration}` (0-based), `${loop.attempt_dir}` (this
  iteration's evidence dir), `${loop.feedback_path}` (accumulated feedback file), and
  `${loop.previous.<step>.outputs.<name>}` (empty on iteration 0 — documented, not an
  error).
- **Iteration context is a file, not a template conditional.** Every iteration appends
  its `feedback` result to `<run-dir>/feedback.md`. `prompts/generate_kernel.md` always
  interpolates `${loop.feedback_path}`; on iteration 0 that file exists and says
  `first attempt — no prior failures`. That is how "loop back with the additional
  context of what's wrong, and that an existing kernel needs fixing" is expressed
  without conditionals in the prompt or branching in the runner. `${loop.previous.
  generate.outputs.kernel_path}` tells the agent *which* file to fix.
- **Attempts are cumulative, not isolated.** Iteration 2 fixes iteration 1's kernel in
  the checkout; `attempt_dir` holds evidence (prompts, logs, result JSON), not sources.
  An optional `snapshot` step (`git -C ${vars.repo_root} diff` → `attempt_dir/work.diff`)
  gives per-iteration forensics without the runner knowing about git.
- `when:` skips a step on a false condition (same closed grammar), recording it as
  `skipped` rather than failing the iteration.

### 5.2 Why the exit condition is measured, not asked

The obvious `until:` is "ask the agent whether it worked." That is exactly the failure
this suite is prone to. Per `hipdnn-integration-testing`: `ctest` with a label matching
nothing prints `No tests were found!!!` and **exits 0**; a gtest filter matching nothing
prints `[ PASSED ] 0 tests`; and a full engine run that supports nothing reports
`Passed: 0 / 6772, Skipped: 6772, Failed: 0`, exit 0, target succeeded. A generated
kernel that fails to register produces precisely that shape — and an agent reading
"0 failed" will report success.

So: the loop's success predicate reads `passed > 0 and failed == 0` from the binary's
own coverage summary, and the `validate` step additionally asserts that *some* case
actually executed. `--verification-mode cpu` is pinned because a freshly imported graph
has no golden data and `auto` would silently degrade to SKIP. Agents supply diagnosis;
they never supply the verdict.

## 6. Execution model

- Steps execute in **declaration order**, sequentially. `after:` is a validated
  assertion that a named step really does precede this one, not a scheduling hint;
  reference-implied edges are enforced the same way -- a step may only read outputs of
  steps declared before it, so declaration order is always a valid topological order.
  A `loop` group is that same sequence, re-executed per iteration.
- Each step: resolved argv + env + cwd → `subprocess.Popen`, output streamed to files
  (never buffered in memory), optional live tee (`--tee`).
- Timeout → process **tree** kill (`taskkill /T /F` on Windows, process-group kill on
  POSIX), recorded as `timed_out`.
- Failure policy per step: `expect_exit: [0]`, `retries: N`, `continue_on_error: true`.
- Run artifacts:
  ```
  runs/<flow-name>/<utc-stamp>/
    run.json            # resolved inputs/vars, per-iteration step status/outputs/timings
    inputs.json         # exactly what this run was asked to do
    feedback.md         # accumulated cross-iteration context
    integrate/iter-00/<step-id>/{cmd.txt,argv.json,stdin.txt,stdout.log,stderr.log,result.json}
    integrate/iter-01/...
  ```
  `run.json` is the machine-readable result and the basis for later `--resume`.

## 7. Multiple runs / sweeps (later phase)

```yaml
  - id: validate
    matrix:
      engine: [MIOPEN_ENGINE, HIPBLASLT_ENGINE]
      repeat: 3
```
Instances get their own step dirs; downstream steps either `for_each:` (fan out 1:1) or
read `${steps.validate.results_json}` (aggregate). Useful for flakiness checks and for
comparing the generated kernel against incumbent engines — not needed for the first
working loop.

## 8. CLI

```
orchestrate.py run <flow.yaml> [--tools tools.yaml] [--profile P]
                               [--input k=v | --input k=@file] [--inputs-file f.yaml]
                               [--dry-run] [--only ID] [--from ID]
                               [--max-iterations N] [--run-dir DIR] [--tee]
orchestrate.py validate <flow.yaml> [--tools tools.yaml]   # schema + refs, no execution
orchestrate.py inputs   <flow.yaml>                        # list declared inputs + defaults
orchestrate.py doctor   [--tools tools.yaml] [--profile P] # resolve every exe; report missing
orchestrate.py tools list
```

`--dry-run` prints resolved argv/env/cwd **and** the rendered prompt per step — the
answer to both "is my environment right on this machine" and "is this actually the
prompt I'm sending" without launching anything. `--max-iterations` overrides the flow's
loop budget (1 = single pass, for debugging).

## 9. Layout

```
projects/hipdnn/tools/orchestrator/
  DESIGN.md                 # this document (rationale)
  README.md                 # how to operate it
  orchestrate.py            # CLI entry (mirrors IngestorGenerator/generate.py)
  runner/
    __init__.py  errors.py
    refs.py                 # ${...} resolution + the closed condition grammar
    toolreg.py  flow.py     # schema load/validate
    process.py              # launch, stream, timeout, tree-kill
    outputs.py              # extractors + result_file/result_schema validation
    engine.py               # step execution, loop driver, run dir, manifest
  configs/
    tools.yaml                        # this machine
    flows/rtc-kernel-review.yaml      # shipped: generate + independent review loop
    prompts/generate_kernel.md  review_kernel.md
  tests/                    # pytest; a fake agent CLI stands in, hermetic
  pyproject.toml  requirements.txt  .gitignore
```

The ingest/build/validate flow of §5 is the next flow to add; it needs no runner
changes, only `configs/flows/kernel-integration.yaml` plus its prompts.

Conventions carried over: MIT + AMD copyright header on every file, `from __future__
import annotations`, argparse, typed dataclasses, pytest config in `pyproject.toml`.

## 10. Phasing

| Phase | Content | Status |
|---|---|---|
| P1 | registry + flow load/validate, `inputs:`/`prompt_file` rendering, strict refs (including inside prompts), sequential execution, `result_file`+`result_schema`, extractors, run dir + `run.json`, CLI `run/validate/inputs/doctor/tools/--dry-run` | **done** |
| P2 | `loop` (`until`/`max_iterations`/`on_exhausted`/`feedback_from`), `${loop.*}` refs, `feedback.md` accumulation, `when:`, `assert`, `expect_exit`, `retries`, `continue_on_error` | **done** |
| P3 | `--resume`, `matrix`/`for_each`, bounded parallelism, no-progress detection (identical failure signature twice → stop early) | not started |

The §5 kernel-integration loop is not blocked on P3: it is a flow file, and every
construct it uses exists today.

## 11. Decisions taken (challenge any)

- **YAML, not TOML/JSON** — matches `IngestorGenerator` configs and the existing PyYAML
  dependency; comments matter in a hand-edited flow.
- **Registry locates, flow invokes** — no args/cwd/timeout/stdin in `tools.yaml`.
- **No expression `eval`** — closed grammar for refs, `when`, `assert`, `until`.
- **Agent results arrive as a validated file**, not as parsed prose.
- **Agents diagnose; binaries decide** — the loop's exit condition is measured test
  counts, never an agent's self-assessment (§5.2).
- **Bounded loop, explicit budget** — `max_iterations` with `on_exhausted: fail`. An
  unbounded repair loop burns hours and hardware unattended.
- **Context is data, not runner state** — `inputs`, refs, an accumulated feedback file,
  and optionally the agent's own session id.
- **Generic runner, hipDNN knowledge in YAML/prompts** — no built-in ctest/gtest logic.

## 12. Open questions

1. **How does the generated kernel reach a testable engine?** The flow assumes the
   ingest agent produces a registered engine name that `--test-engine` accepts, which
   implies a full provider rebuild each iteration (the `build` step, ~minutes to tens of
   minutes). Is that the intended loop cost, or is there a faster direct-load path
   (`kind: direct_load` per the ingestor skill) that skips the packaged rebuild?
2. **Graph → bundle.** `import_graph.py` is the documented path for turning a graph JSON
   into a runnable bundle, and it dedups by structure hash. Do we import the input graph
   into the real `integration-test-bundles/` tree (mutates the checkout) or into a
   throwaway dir pointed at by `--golden-data-dir`? I lean throwaway — a validation run
   should not leave bundles behind.
3. **Correctness reference.** `--verification-mode cpu` compares against the CPU
   reference. Is a CPU reference guaranteed to exist for every graph we will feed this?
   If not, iteration can end in "skipped, no reference" — which must be a loop failure,
   not a pass.
4. **Which agent CLI** — `claude` only, or Codex too? No runner code depends on it, but
   the shipped prompts and the `session_id` extractor match one CLI's JSON shape.
5. **Iteration isolation** — cumulative edits in one checkout (assumed above), or a
   scratch worktree per attempt so a bad iteration can be discarded wholesale?
6. **Ingest agent's own tests.** "Basic tests to ensure the integration is valid" — is
   that the ingestor skill's internal host-side census (agent-reported, inside
   `ingest.json`), or should the flow run an explicit deterministic gate (e.g. `ctest -L
   quick` on the provider) before reaching `validate`? Same argument as §5.2 says the
   latter.
