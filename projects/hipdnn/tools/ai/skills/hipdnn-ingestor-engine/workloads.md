# Workload identity, coverage and runtime evidence

[RUNBOOK.md](RUNBOOK.md) alone owns the ordered create/extend workflow. This page
specifies corpus selection, semantic accounting and final runtime joins; it does
not define a competing sequence. The [sweep reference](../../../IngestorGenerator/tools/README-sweeps.md)
owns declarative YAML, measurement, correctness gates and resume.

## Corpus provenance and scope

Kernel-side dispatchers, spec predicates and tuning documentation say what a kernel
can build or serve, not what callers ask for. An AOT set that lacks a candidate for
a supported real shape still declines that caller. Inventory both external workloads
and the kernel owners' benchmark/published shapes before approving the baseline.
Keep their results separate rather than hiding one population in a mixed aggregate.

The separate `ROCm/dnn-benchmarking` project provides caller graphs and the benchmark
CLI. Use its current README/setup guidance and each workload's `MANIFEST.md`; do not
assume the executable ships in the provider build. A `microbench/` path is a
provenance label, not proof of synthetic data. Read manifests before excluding
library-derived shape collections. Published result CSVs can preserve resolved
shapes and priorities that source-only benchmark mining cannot reconstruct.

Record every declared source's total, parsed, servable, proposed-covered and
excluded populations, with reasons and original identities. A missing/unreadable
source does not establish a zero population. Approved exclusions remain visible;
never silently reduce the denominator to successful timing rows.

## Graph semantics and request identity

Inspect real graphs as well as in-tree bundles. Compare per-operand dims/strides,
optional/deprecated attribute spelling, topology/UID edges and shape magnitude.
Use the complete graph/kernel restrictions, not a convenient subset that overstates
support. Distinguish valid unsupported requests from missing compiled variants and
from malformed or unrepresentable input.

All semantic request fields participate in identity. In particular, preserve
unmasked versus causal versus bounded-window attention, sink semantics and
independent Q/K/V dimensions; a Q/K contraction dimension does not fix V/output
width. Provenance alone does not split semantic identity, but deduplication must
retain **every original corpus/source/graph occurrence** for runtime accounting.
Never infer semantics from a filename or drop a field to fit a request schema.

The supported miner interface, using absolute script/input paths, is:

```text
<PY> <GEN>/tools/mine_shapes.py --published <owner-results.csv> --graphs <graph-directory> --arch gfx942 --include-windowed --out <request-shapes.json>
```

`<PY>` is the generator's `.venv/bin/python`. An actual benchmark source tree can
also be supplied through `--rocke-bench`. Omit a source only when explicitly absent
from the approved scope. Request JSON is for authoring/parity analysis; the sweep
consumes actual graph JSON directories. Reconcile source counts and exclusions
across both forms.

## Offline applicability is not runtime coverage

For a rocKE integration, use its actual scoped profile and request list:

```text
<PY> <GEN>/tools/reconcile_applicability.py --profile <profile.yaml> --shapes <request-shapes.json>
```

A validated false predicate with a valid reason is an ordinary decline. Missing or
noncallable APIs, signature/binding failure, invocation exception and invalid return
values are `ParityError`/exit 2, including under narrowing/escape flags. Generic
constructor/factory exceptions are operational errors, not evidence of unsupported
input. `_supports` is a degraded fallback only when `admits` is absent, not when it
is present but broken.

Reference-only supported rows identify missing variants, matcher mistakes or a
reference claim requiring investigation. An explicit scope decision is required
for exclusions; an escape flag cannot erase a broken reference API or justify a
coverage gap. Applicability agreement is not numerical truth. Direct-load engines
use their own explicit semantic corpus and reference; no fictitious rocKE profile
is required.

## Installed measurement inputs

Keep each arm's source/config, descriptors/payload, plugin/runtime and installation
identities. Use bindings and plugins from the accepted coherent stack, not a setup
helper's unrelated default checkout or an arbitrary wheel plugin directory.
Check the actual capability/production build flags, installed descriptor shard and
engine discovery. A plugin file or an empty registry alone does not diagnose why an
engine is unavailable; missing descriptors, disabled capability, wrong architecture,
wrong install paths and failed native registration need distinct evidence.

Baseline installed discovery must connect the exact UED name to the exact benchmark
`engine_name` and engine ID. Prefixes, another engine's rows and a reference provider
cannot satisfy attribution. The RUNBOOK's separate engine-pinned integration
registration proves targeted device behavior; do not invent a benchmark selection
flag to replace it.

Use the Python sweep with explicit declarative input:

```text
<PY> <GEN>/tools/sweep.py --config <absolute-YAML>
```

Start with `configs/sweep-isolation.sweep.yaml.example`. Its ordered corpus/arm
lists, counts, installed paths and served floor must describe the actual run;
example counts are not measurements. Paths resolve from the YAML directory, not
cwd, and configuration has no environment interpolation. Hazard exclusions are a
fail-if-present gate, never implicit filtering; declare `exclude_tensors: none`
when appropriate.

Comparative measurement uses one device/node/session/job, baseline-first fixed arm
order, discarded gated warmup, at least three rounds and separate correctness.
Cache/log isolation and content-bound completion sidecars prevent other arms or
stale results from satisfying a phase. Diagnostic resume across sessions is not a
single-session comparative cohort. Final comparisons use a fresh output directory
after final selection, regeneration, rebuild and installation.

`SWEEP_DONE` means validated completion only. Explicit
`correctness.enabled: false` permits `SWEEP_TIMING_ONLY`/exit 0, never final RUNBOOK
success. Unmet gates produce `SWEEP_INCOMPLETE`/exit 1; invalid configuration exits 2.
A correctness command failure, missing comparison, mismatch, NaN or unwritten output
cannot be excused by successful timing.

Reference capability must cover the actual feature and shapes. Neither current CPU
nor GPU SDPA reference supports a sink UID. Without an actually capable independent
reference, the claimed feature is **BLOCKED**; CPU fallback and unverified golden
output are not remedies. Keep numerical work affordable without silently excluding
production shapes from the approved correctness obligation. Allocation/container
failure is infrastructure evidence, not a kernel test result.

## Complete final runtime join

After the final installed sweep, harvest its result files and available engine logs
into an outcome ledger for every input in every corpus/phase. Retain:

| Identity/evidence | Required content |
|---|---|
| Semantic key | Every semantic request field, excluding provenance only |
| Original occurrence | Corpus, source, original graph identity and staged file identity |
| Input binding | Current phase key/fingerprint and final artifact identities |
| Attribution | Exact expected engine and observed engine ID/name |
| Outcome | Served, explicitly declined, execution error, missing or ambiguous |
| Evidence | Result/log location and an actually observed decline reason when applicable |

A missing timing row is not a decline. An unavailable runtime reason remains
unavailable; do not reconstruct it from offline source or policy. Join within each
corpus/phase, reject duplicate/ambiguous graph names, mismatched fingerprints and
missing outcomes, and preserve all source occurrences even when mining merged
semantic requests. Missing, ambiguous or execution-error outcomes block acceptance.

Only a complete join permits construction of the existing graph-name-to-reason
JSON for that corpus:

```text
<PY> <GEN>/tools/reconcile_applicability.py --profile <profile.yaml> --shapes <this-corpus-requests.json> --declines <this-corpus-runtime-declines.json>
```

Do not mix same-named graphs across corpora or feed a sparse mapping as complete
runtime evidence. Without the join, label reconciliation **offline only**, even
if the CLI accepts the file. The integration walkthrough includes a disposable
negative join probe for omitted and same-named mismatched outcomes.

## Reporting boundary

Report covered/servable/total populations by original source, exact-engine served
and independently validated counts, all declines and exclusions, and every blocked
outcome. A minimum served floor is not permission to omit the rest. Report
geomean-of-ratios and time-weighted sum-baseline/sum-arm together, drift by round,
and byte-identical controls selected from artifact hashes rather than timings.

Claims refer to the final installed artifact and exercised device/graphs only.
Regeneration or changed installed content invalidates results bound to old inputs
and returns to the RUNBOOK's artifact/native/device gates. Extension acceptance
must exercise the addition: the concrete pointwise walkthrough selects HALF/256
on a one-element graph and retains old ADD/MUL/SUB coverage, not just the unchanged
default-FLOAT case. Logs, proposed commands or queued jobs are not completed proof.
