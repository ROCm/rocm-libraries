# Coverage, correctness and performance sweeps

[RUNBOOK.md](../../ai/skills/hipdnn-ingestor-engine/RUNBOOK.md) owns the ordered
create/extend workflow. This page specifies the Python sweep interface, input data,
measurement protocol and evidence. A sweep requires installed arms and the actual
target device; it does not replace early feasibility, native host proof or the
engine-pinned integration tests.

## Invocation and prerequisites

Use the generator venv and absolute paths, independent of the current directory:

```text
<PY> <GEN>/tools/sweep.py --config <absolute-YAML>
```

`<PY>` is `<GEN>/.venv/bin/python`. Start from
`configs/sweep-isolation.sweep.yaml.example`, and set paths, identities and counts
from the actual experiment. The runner reads safe YAML as data; configuration is
not executable. It accepts argument lists, not command strings or launchers that
interpret configuration as code. There is no environment interpolation or implicit
environment configuration.

Required prerequisites are the requested GPU on the execution host, a visible
writable sweep root, readable graph corpora, each current installed provider arm,
and a working benchmark executable with the selected numerical reference and
hipDNN bindings. `dnn-benchmark` comes from the dnn-benchmarking project; a provider
build alone does not establish its installation. Follow that project's setup
without substituting unrelated plugins or bindings for the integration under test.
Paths on a login machine's local disk are not necessarily compute-node-visible.
Use an allocated device job and retain source, artifact, device and job identities.

The device probe has two explicit interfaces:

```text
<PY> <GEN>/tools/device_probe.py --mode early --arch gfx942 --sweep-root <existing-root>
<PY> <GEN>/tools/device_probe.py --mode installed --arch gfx942 --sweep-root <existing-root> --install <existing-install>
```

Early mode has no installation prerequisite and ignores inherited `INSTALL`.
Installed mode follows installation. Exit 0 covers checks applicable to that mode;
exit 1 reports device/path/write failure; exit 2 reports invalid invocation. Neither
mode proves plugin loading, engine dispatch or numerical correctness.

## YAML schema

All filesystem paths are plain strings, absolute or relative to the YAML file's
directory, never the process working directory. A bare executable name in `argv[0]`
is resolved through `PATH` and recorded; an executable with path components resolves
from the config directory. Child processes use the config directory as their cwd.

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

Arm/corpus names must be unique safe single path components. Unknown or duplicate
keys, wrong types (including booleans as counts), duplicate names, missing required
paths/counts/identities/exclusions, invalid warmup selection, unreadable or malformed
graphs and input/output overlap are invalid configuration, not partial measurement.
Metacharacters in a YAML string remain data and never execute.

A configuration's measurement controls can look like this; the complete example
also supplies the experiment-specific roots, ordered arms/corpora and exact counts:

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

Do not assume the benchmark uses the UED spelling. If actual discovery reports
`engine_<id>`, record that exact identity in `engine_name` and retain its observed
mapping to `engine_ued_name`. Another engine's timings never credit this engine.
There is no invented benchmark engine-selection flag; the separate installed
engine-pinned integration registration establishes targeted device proof.

## Corpus and coverage accounting

Keep external caller workloads and kernel-owner benchmark/published graphs separate.
Preserve original corpus/source/graph identities and manifests; a `microbench` path
alone says nothing about provenance. Stage actual graph JSON for measurement, not
just the semantic request list consumed by mining/parity tools.

Declare every corpus count before measurement. Hazard exclusions do not shrink that
denominator. For attention, use the exact backward-gradient names from the miner's
`BACKWARD_GRADIENT_TENSOR_NAMES`, not a copied subset. Declare `none` explicitly
when the approved operation has no such hazard class. Missing or malformed graph
input is an error, never evidence of an empty population.

Coverage, numerical correctness and performance answer different questions. Retain
served, explicitly declined, execution-error, missing and ambiguous outcomes;
absence of a successful timing row is not a decline reason. A complete final
runtime outcome join is required before `reconcile_applicability.py --declines`.
See [workloads.md](../../ai/skills/hipdnn-ingestor-engine/workloads.md) for the join
and semantic identity contract. Offline reconciliation alone proves no dispatch.

## Measurement protocol

Use one target device, node, session and job for a comparative cohort. Warm each
ordered corpus using `warmup_arm`; discard those timings but require its gates.
The timed grid is round order, then YAML corpus order, then YAML arm order. Keep
baseline first and never sort, rotate or silently reorder arms. Several rounds
expose drift; position-sensitive results need that context.

Each phase directly invokes `benchmark.argv` with these distinct arguments:
`--graph <staged-corpus/*.json>`,
`--plugin-path <arm-install>/lib/hipdnn_plugins/engines`, `--warmup <count>`,
`--iters <count>` and `-o <attempt-output.json>`. Correctness runs once for each
ordered corpus/arm after the timed grid, with `--validate <correctness.reference>`;
it is not mixed into timed sampling.

Construct each child's environment afresh from the original caller environment.
Set the current arm's `ROCM_PATH` and `LD_LIBRARY_PATH` prefix without accumulating
prior arms. Use phase-specific `HIPDNN_CACHE_DIR` and `HIPDNN_LOG_FILE`,
`HIPDNN_FORCE_BENCHMARKING=1` for timing and `HIPDNN_LOG_LEVEL=info`. Correctness has
its own cache and logs. Cache or staged-input paths from another sweep are not
shared state.

Report geometric mean of per-graph ratios and time-weighted
`sum(baseline time) / sum(arm time)` together, split by corpus provenance and round.
A geometric mean alone is not a wall-clock saving. Establish byte-identical controls
from descriptor/payload hashes, never by selecting graphs that timed alike; their
observed ratio measures the noise floor, not an assumed exact 1.000x result.

## Phase gates and completion

A phase requires zero command exit, readable/parseable results and relevant hipDNN
logs, the expected installed descriptor count, and plugin-path provenance for the
intended arm. Result inventory must account for the staged graph identities without
silently merging duplicate names or accepting unknown/missing rows. Timing credits
only `status: success` rows with finite positive `mean_ms` for the **exact**
configured engine, counting unique graphs against `min_served`. `role: reference`
and other-engine rows cannot satisfy it; omitted `role` means engine. Failures
remain in the outcome ledger. Benchmark `graph_name` uses the graph JSON name or
file stem, so ambiguous names cannot be silently merged.

When correctness is enabled, every claimed served graph needs an actual comparison
against the declared independent reference with `passed: true`,
`execution_success: true` and `tolerance_match: true`. A failed/missing comparison,
malformed result, nonzero command exit, unavailable reference, NaN or unwritten
output fails the gate. Reference-provider rows without comparison evidence are not
validated graphs. A plausible timing result cannot discharge correctness.

Select a reference capable of the actual graph semantics. Neither current CPU nor
GPU SDPA reference supports a sink UID. An unsupported reference means **BLOCKED**,
not an automatic CPU fallback or unverified expected output. Zero finite mismatch
counts do not excuse NaNs or unwritten output.

| Marker / exit | Meaning |
|---|---|
| `SWEEP_DONE` / 0 | Validated completion: every required timed and correctness phase plus applicable current-invocation warmup passes |
| `SWEEP_TIMING_ONLY` / 0 | Explicit `correctness.enabled: false`; timing gates only, never validated completion or final RUNBOOK success |
| `SWEEP_INCOMPLETE` / 1 | Required gates unmet; raw diagnostics are not completion evidence |
| Exit 2 | Invalid configuration/invocation; no valid completed sweep |

Completion checks the exact required phase-key set, not merely a success count or a
nonempty output file. A failed correctness command with otherwise plausible JSON is
still failure.

## Isolation and content-bound resume

`output_dir/.running.lock` gives exclusive ownership. Each run stages validated
source graphs into fresh output-owned attempt directories with sorted relative
names, byte hashes and original provenance. It never merges a surviving stage or
uses global corpus/cache paths. Cleanup is confined to tool-created temporary
stage/cache/probe paths; an existing lock is not automatically deleted.

Each phase's fingerprint binds normalized effective YAML, ordered arms/corpora,
phase key, executable arguments/counts, reference/correctness mode, engine identity,
architecture, expected descriptors, served floor and exclusions. It also binds
current corpus names/content, installed descriptors and referenced payload or
embedded-source inputs, loaded plugin/runtime artifacts, resolved executable and
applicable reference environment, plus target host/device and relevant child
environment/provenance. Root names, timestamps or counts alone are insufficient.

A phase-specific `<tag>.complete.json` sidecar records `status=success`, phase key,
input fingerprint, output/log hashes, command exit and every applicable gate.
Before rerunning a stale/failed phase the old success sidecar is invalidated under
the output lock. Results go to a fresh attempt path; only after gates pass and
evidence is flushed/closed is a temporary completion record atomically installed.
Failure/interruption can retain diagnostic output but cannot leave successful
evidence for that attempt.

Resume recomputes **current input content** and checks sidecar status/key, evidence
hashes and gates before skipping a phase. Missing, corrupt, partial, failed, stale
or edited evidence is rerun, or rejected as invalid input before measurement.
Correctness has its own record. Any invocation doing measurement runs a fresh
warmup. Editing a corpus, installed artifact or relevant config invalidates reuse;
a timed JSON surviving a failed served/correctness gate never permits a skip.

Retain resumed-session boundaries honestly. Diagnostic resume may establish phase
gates across sessions, but not a single-session comparative timing cohort. Final
publishable comparisons use a fresh output directory and complete single-job grid
after final selection, regeneration, rebuild and installation. Preserve raw results,
logs, winners, fingerprints and per-corpus coverage/correctness summaries with the
run evidence; do not claim unexecuted architectures or unrelated installations.
