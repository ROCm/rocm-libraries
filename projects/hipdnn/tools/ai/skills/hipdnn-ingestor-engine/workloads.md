# Workload identity and runtime evidence

[RUNBOOK.md](RUNBOOK.md) owns execution and command order. The
[sweep reference](../../../IngestorGenerator/tools/README-sweeps.md) owns its Python
CLI and YAML schema; this page defines evidence and accounting contracts.

## Provenance, scope and semantic identity

Inventory external workloads and the owners' benchmarks/published results. Kernel
predicates say what can build or serve, not what callers ask for. Keep populations
separate by source. The external `ROCm/dnn-benchmarking` project supplies the benchmark
CLI and graph corpora; use its current setup and workload manifests, not an assumed
provider-installed executable. `microbench/` is a provenance label, not proof of
synthetic data.

Each declared source needs total, parsed, servable, covered and excluded counts with
reasons and original identities. Missing/unreadable input is not an empty population.
Request JSON feeds mining/parity; actual graph JSON directories feed the sweep.
`mine_shapes.py` accepts published CSV, graph directories and optional `--rocke-bench`
input. Omit sources only when outside the approved scope.

Every semantic request field participates in identity, including unmasked/causal/window
and sink semantics and independent Q/K/V dimensions. Provenance does not split the
semantic key, but deduplication must retain **all original corpus/source/graph
occurrences**. Inspect real dims/strides, attributes and UID topology; filenames do
not define semantics. Distinguish unsupported, malformed/unrepresentable and missing-
variant outcomes. Never reduce the denominator to successful timing rows.

## Applicability and reference contract

rocKE profiles scope the candidate registry to the actual kernel family/algorithm
and required opt-in selector. Reference candidates must implement
`admits(request) -> (bool, str)`; **there is no `_supports` fallback**. False requires
a nonempty reason. Missing/noncallable APIs, bad signatures, exceptions, invalid
returns and generic constructor/factory failures are operational errors: reconciliation
exits 2, including under escape flags. They are not unsupported-shape evidence.

Reference-only support requires investigation of variants, matcher semantics or the
reference claim and an explicit scope decision for exclusions. Applicability does
not prove numerical truth; use [graph-contract.md](graph-contract.md)'s reference
capability rules. Direct-load engines use their own explicit corpus/reference,
without a fictitious rocKE profile.

## Installed measurement contract

Each arm retains source/config, descriptor/payload, plugin/runtime and installation
identities from a coherent stack. The exact installed UED name must map to the expected
benchmark engine name/ID; another engine, a name prefix or a reference-provider row
cannot satisfy attribution. The sweep supplies the benchmark's `--engine` argument
from that discovered ID; phase-owned arguments must not be overridden.

The YAML example is not an inventory. Replace corpus counts, total installed KDP-entry
counts, paths and served floors with actual inputs. Paths resolve from the YAML
directory with no shell/environment interpolation. Hazard exclusions fail if present,
not silently filter; use `exclude_tensors: none` when appropriate.

Comparisons require one device/node/session/job, baseline-first fixed arm order,
a discarded **gated** warmup, at least three rounds, isolated caches/logs and separate
correctness once per corpus/arm. Reference capability must cover the approved features
and shapes; keep runtime affordable without silently dropping correctness obligations.
Diagnostic cross-session resume is not a single-session comparative cohort. Final
measurements use fresh output after final generation/build/install.

`SWEEP_DONE` means all required phases/gates completed with correctness enabled.
`correctness.enabled: false` can yield `SWEEP_TIMING_ONLY`/exit 0, not final success.
Unmet gates produce `SWEEP_INCOMPLETE`/exit 1; invalid config, operational errors and
interruption exit 2. Timing cannot excuse a failed/missing comparison, mismatch,
NaN or unwritten output. Infrastructure failure is not a kernel test result.

## Complete final runtime join

Every input in every final corpus/phase needs an outcome ledger containing:

| Field | Required evidence |
|---|---|
| Semantic key | All request fields except provenance |
| Original occurrence | Corpus/source/graph identity and staged file identity |
| Input binding | Phase key/fingerprint and final artifact identities |
| Attribution | Exact expected and observed engine ID/name |
| Outcome | Served, explicitly declined, execution error, missing or ambiguous |
| Evidence | Result/log path and actually observed decline reason where applicable |

Absence of a timing row is not a decline. Reasons unavailable in runtime evidence
cannot be reconstructed from offline policy. The join is corpus/phase-local and
rejects missing outcomes, duplicate/ambiguous names and mismatched fingerprints;
semantic deduplication must not discard original occurrences. Missing/ambiguous/error
outcomes block runtime acceptance.

Only the complete join supports the per-corpus graph-name-to-reason JSON consumed by
`reconcile_applicability.py --declines`. Do not mix same-named graphs across corpora or
present a sparse accepted mapping as complete runtime evidence. Without the join,
reconciliation is **offline only**.

Report covered/servable/total, exact-engine served and independently validated counts,
all declines/exclusions and blocked outcomes by original source. A minimum served floor
does not waive the rest. Report geomean-of-ratios and time-weighted sum-baseline/sum-arm,
round drift and byte-identical controls determined from artifact hashes. Changed final
installed content invalidates evidence bound to old inputs.
