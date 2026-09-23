# Coverage, correctness and performance sweeps

[RUNBOOK.md](../../ai/skills/hipdnn-ingestor-engine/RUNBOOK.md) owns the ordered
create/extend workflow; this page specifies the sweep interface, input data, measurement
protocol and evidence. A sweep needs installed arms and the actual target device, and
replaces neither early feasibility, native host proof, nor the engine-pinned integration
tests.

## Invocation and prerequisites

Use the generator venv and absolute paths, independent of the current directory:

```text
<PY> <GEN>/tools/sweep.py --config <absolute-YAML>
```

`<PY>` is `<GEN>/.venv/bin/python`. Start from
`configs/sweep-isolation.sweep.yaml.example` and set paths, identities and counts from the
actual experiment. The runner reads safe YAML as data: configuration is never executed,
commands are argument lists, and there is no environment interpolation.

Prerequisites: the requested GPU on the execution host, a visible writable sweep root,
readable graph corpora, each current installed provider arm, and a working benchmark
executable with the selected numerical reference and hipDNN bindings. `dnn-benchmark` comes
from the dnn-benchmarking project; a provider build does not install it. Run in an
allocated device job -- a login machine's local disk need not be compute-node-visible --
and retain source, artifact, device and job identities.

The device probe has two explicit interfaces:

```text
<PY> <GEN>/tools/device_probe.py --mode early --arch gfx942 --sweep-root <existing-root>
<PY> <GEN>/tools/device_probe.py --mode installed --arch gfx942 --sweep-root <existing-root> --install <existing-install>
```

Early mode has no installation prerequisite and ignores inherited `INSTALL`; installed mode
follows installation. Exit 0 covers the checks applicable to that mode, exit 1 is a
device/path/write failure, exit 2 an invalid invocation, exit 3 means neither `rocminfo` nor
`hipInfo` (what the Windows ROCm wheels ship) could run, so the device was never observed --
a packaging fact about the host, remedied by an inspection utility. Neither mode proves
plugin loading, engine dispatch or numerical correctness.

## YAML schema

All filesystem paths are plain strings, absolute or relative to the YAML file's directory,
never the process working directory. A bare executable name in `argv[0]` resolves through
`PATH` and is recorded; one with path components resolves from the config directory, which
is also each child's cwd.

| Key | Contract |
|---|---|
| `sweep_root` | Required existing, execution-host-visible directory |
| `output_dir` | Required dedicated directory inside `sweep_root`; cannot equal or contain an install or corpus input root |
| `corpus_dir` | Required existing corpus root; corpus entries still declare their paths explicitly |
| `arch` | Required exact gfx architecture token |
| `engine_ued_name` | Required exact installed UED engine name |
| `engine_name` | Required exact identity in benchmark results, never a prefix or regex; installed baseline discovery must connect it to `engine_ued_name` |
| `corpora` | Nonempty ordered list of `{name, path, expected_graphs}`; explicit directory path and positive count covering every staged graph |
| `arms` | Nonempty ordered list of `{name, install_tree, expected_descriptors}`; descriptor count is the positive total of all KDP `kernelDescriptors` entries in the installed tree |
| `warmup_arm` | Required arm name or explicit `null`; comparative runs use the first/baseline arm |
| `rounds` | Positive integer; drift-reporting comparisons require at least three |
| `min_served` | Positive integer no greater than any corpus count, set near the approved served population, not a success-by-one-row default |
| `exclude_tensors` | Required literal `none` or nonempty list of exact tensor names, compared case-insensitively; a fail-if-present hazard gate, not silent filtering |
| `benchmark` | `{argv, warmup, iters}`; nonempty string argument list, nonnegative warmup, positive iterations |
| `correctness` | `{enabled, reference, warmup, iters}`; explicit boolean; when enabled, a nonempty supported reference name, nonnegative warmup and positive iterations |
| `probe_env` | Optional `null` or string argument list for a directly invoked provenance executable; a declared missing/failing probe is an error |

Arm/corpus names must be unique safe single path components. Unknown or duplicate keys,
wrong types (including booleans as counts), duplicate names, missing required
paths/counts/identities/exclusions, invalid warmup selection, unreadable or malformed graphs
and input/output overlap are invalid configuration, not partial measurement. Metacharacters
in a YAML string remain data.

The measurement controls look like this; the complete example also supplies
experiment-specific roots, ordered arms/corpora and exact counts:

```yaml
warmup_arm: parity
rounds: 3
benchmark:
  argv: [dnn-benchmark]
  warmup: 10
  iters: 50
correctness:
  enabled: true
  reference: pytorch
  warmup: 1
  iters: 3
probe_env: null
```

Do not assume the benchmark uses the UED spelling: if discovery reports `engine_<id>`,
record that exact identity in `engine_name` and retain its observed mapping to
`engine_ued_name`. Another engine's timings never credit this engine. There is no benchmark
engine-selection flag; targeted device proof comes from the installed engine-pinned
integration registration.

## Corpus and coverage accounting

Keep external caller workloads separate from kernel-owner benchmark/published graphs, and
preserve original corpus/source/graph identities and manifests: a `microbench` path says
nothing about provenance. Stage actual graph JSON, not the semantic request list
mining/parity tools consume.

Declare every corpus count before measurement; hazard exclusions do not shrink that
denominator. For attention use the exact names from the miner's
`BACKWARD_GRADIENT_TENSOR_NAMES`, and declare `none` explicitly where the approved
operation has no such hazard class. Missing or malformed graph input is an error, never an
empty population.

Coverage, numerical correctness and performance answer different questions. Retain served,
explicitly declined, execution-error, missing and ambiguous outcomes; a missing timing row
is not a decline. Complete the final runtime outcome join before
`reconcile_applicability.py --declines` -- see
[workloads.md](../../ai/skills/hipdnn-ingestor-engine/workloads.md) for the join and
semantic identity contract. Offline reconciliation proves no dispatch.

## Measurement protocol

Use one target device, node, session and job for a comparative cohort. Warm each ordered
corpus with `warmup_arm`, discarding those timings but requiring its gates. The timed grid
is round order, then YAML corpus order, then YAML arm order: baseline first, never sorted,
rotated or reordered. Several rounds expose drift.

Each phase invokes `benchmark.argv` directly with `--graph <staged-corpus/*.json>`,
`--plugin-path <arm-install>/lib/hipdnn_plugins/engines`, `--warmup <count>`,
`--iters <count>` and `-o <attempt-output.json>`. Correctness runs once per ordered
corpus/arm after the timed grid, with `--validate <correctness.reference>`, never mixed
into timed sampling.

Construct each child's environment afresh from the caller environment, setting the current
arm's `ROCM_PATH` and `LD_LIBRARY_PATH` prefix without accumulating prior arms. Use
phase-specific `HIPDNN_CACHE_DIR` and `HIPDNN_LOG_FILE`, `HIPDNN_FORCE_BENCHMARKING=1` for
timing and `HIPDNN_LOG_LEVEL=info`; correctness has its own cache and logs, and another
sweep's paths are not shared state.

Report geometric mean of per-graph ratios and time-weighted
`sum(baseline time) / sum(arm time)` together, split by corpus provenance and round; a
geometric mean alone is not a wall-clock saving. Take byte-identical controls from
descriptor/payload hashes, never from graphs that timed alike: their ratio is the noise
floor, not an exact 1.000x.

## Phase gates and completion

A phase requires zero command exit, readable/parseable results and relevant hipDNN logs,
the expected installed descriptor count, and plugin-path provenance for the intended arm.
Result inventory must account for the staged graph identities without merging duplicate
names or accepting unknown/missing rows. Timing credits only `status: success` rows with
finite positive `mean_ms` for the **exact** configured engine, counting unique graphs
against `min_served`; `role: reference` and other-engine rows cannot satisfy it, and an
omitted `role` means engine. An engine row reporting `plugin_path` is attributed to the
arm only when, with both paths resolved, it names the arm's `lib/hipdnn_plugins/engines`
directory itself or a plugin directly inside it; a sibling tree or a deeper descendant
makes that graph ambiguous. Failures stay in the outcome ledger, and benchmark
`graph_name` is the graph JSON name or file stem.

With correctness enabled, every claimed served graph needs a real comparison against the
declared independent reference with `passed: true`, `execution_success: true` and
`tolerance_match: true`. A failed/missing comparison, malformed result, nonzero command
exit, unavailable reference, NaN or unwritten output fails the gate, and
reference-provider rows without comparison evidence are not validated graphs. Only a
row with an explicit `role: reference` is reference evidence: a requested-provider row
that omits `role` is an engine row, so it cannot satisfy the reference gate.

Select a reference capable of the actual graph semantics: neither current CPU nor GPU SDPA
reference supports a sink UID, and an unsupported reference means **BLOCKED**, not a CPU
fallback or unverified expected output.

| Marker / exit | Meaning |
|---|---|
| `SWEEP_DONE` / 0 | Validated completion: every required timed and correctness phase plus applicable current-invocation warmup passes |
| `SWEEP_TIMING_ONLY` / 0 | Explicit `correctness.enabled: false`; timing gates only, never validated completion or final RUNBOOK success |
| `SWEEP_INCOMPLETE` / 1 | Required gates unmet; raw diagnostics are not completion evidence |
| Exit 2 | Invalid configuration/invocation; no valid completed sweep |

Completion checks the exact required phase-key set, not a success count or a nonempty
output file. A failed correctness command with otherwise plausible JSON is still failure.

## Isolation and content-bound resume

`output_dir/.running.lock` gives exclusive ownership. Each run stages validated source
graphs into fresh output-owned attempt directories with sorted relative names, byte hashes
and original provenance, never merging a surviving stage or using global corpus/cache
paths. Cleanup touches only tool-created temporary stage/cache/probe paths, and an existing
lock is never deleted automatically.

Each phase's fingerprint binds normalized effective YAML, ordered arms/corpora, phase key,
executable arguments/counts, reference/correctness mode, engine identity, architecture,
expected descriptors, served floor and exclusions, plus current corpus names/content,
installed descriptors and referenced payload or embedded-source inputs, loaded
plugin/runtime artifacts, resolved executable and applicable reference environment, and
target host/device with relevant child environment/provenance. Root names, timestamps or
counts alone are insufficient.

A phase-specific `<tag>.complete.json` sidecar records `status=success`, phase key, input
fingerprint, output/log hashes, command exit and every applicable gate. Before rerunning a
stale/failed phase the old sidecar is invalidated under the output lock; results go to a
fresh attempt path, and the completion record is installed atomically only after gates pass
and evidence is flushed. Failure or interruption may retain diagnostic output, never
successful evidence.

Resume recomputes **current input content** and checks sidecar status/key, evidence hashes
and gates before skipping a phase. Missing, corrupt, partial, failed, stale or edited
evidence is rerun or rejected before measurement; correctness has its own record; any
measuring invocation runs a fresh warmup. Editing a corpus, installed artifact or relevant
config invalidates reuse, and a timed JSON surviving a failed served/correctness gate never
permits a skip.

Diagnostic resume may establish phase gates across sessions, but not a single-session
comparative timing cohort. Final publishable comparisons use a fresh output directory and a
complete single-job grid after final selection, regeneration, rebuild and installation.
Preserve raw results, logs, winners, fingerprints and per-corpus coverage/correctness
summaries with the run evidence, and claim no unexecuted architecture or unrelated
installation.
