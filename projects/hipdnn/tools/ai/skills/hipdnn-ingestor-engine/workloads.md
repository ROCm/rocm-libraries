# Workload identity and runtime evidence

**Authoritative for the measurement and accounting rules** — cohort conditions, sweep
terminal statuses, the runtime outcome ledger and the required reporting statistics.
[RUNBOOK.md](RUNBOOK.md) owns execution and command sequencing; where the two appear to
differ on a measurement rule, this page governs.

The sweep reference — from the repository root in
`projects/hipdnn/tools/IngestorGenerator/tools/README-sweeps.md`, RUNBOOK's
`$GEN/tools/README-sweeps.md` — owns the Python CLI and YAML schema.

## Provenance, scope and semantic identity

Inventory external workloads and the owners' benchmarks and published results, keeping
populations separate by source. Kernel predicates say what can build or serve, not what
callers ask for. The external `ROCm/dnn-benchmarking` project supplies the benchmark CLI
and graph corpora; use its current setup and workload manifests, not an assumed
provider-installed executable. `microbench/` is a provenance label, not proof of
synthetic data.

Each declared source needs total, parsed, servable, covered and excluded counts with
reasons and original identities. Missing or unreadable input is not an empty population.
Request JSON feeds mining and parity; actual graph JSON directories feed the sweep.
`mine_shapes.py` accepts published CSV, graph directories and optional `--rocke-bench`
input. Omit sources only when outside the approved scope.

Every semantic request field participates in identity, including unmasked/causal/window
and sink semantics and independent Q/K/V dimensions. Provenance does not split the
semantic key, but deduplication must retain **all original corpus/source/graph
occurrences**. Inspect real dims/strides, attributes and UID topology, not filenames.
Distinguish unsupported, malformed/unrepresentable and missing-variant outcomes, and
never reduce the denominator to successful timing rows.

## Applicability and reference contract

rocKE profiles scope the candidate registry to the actual kernel family/algorithm and
required opt-in selector. Reference candidates must implement
`admits(request) -> (bool, str)`; **there is no `_supports` fallback**, and False
requires a nonempty reason. Missing or noncallable APIs, bad signatures, exceptions,
invalid returns and generic constructor/factory failures are operational errors:
reconciliation exits 2, including under escape flags. They are not unsupported-shape
evidence.

Reference-only support requires investigating variants, matcher semantics or the
reference claim, plus an explicit scope decision for exclusions. Applicability does not
prove numerical truth — use [graph-contract.md](graph-contract.md)'s reference
capability rules. Direct-load engines use their own explicit corpus and reference.

## Installed measurement contract

Each arm retains source/config, descriptor/payload, plugin/runtime and installation
identities from a coherent stack. The exact installed UED name must map to the expected
benchmark engine name/ID; another engine, a name prefix or a reference-provider row
cannot satisfy attribution. The sweep supplies the benchmark's `--engine` argument from
that discovered ID; phase-owned arguments must not be overridden.

The YAML example is not an inventory. Replace corpus counts, total installed KDP-entry
counts, paths and served floors with actual inputs. Paths resolve from the YAML
directory with no shell or environment interpolation. Hazard exclusions fail if present
rather than silently filtering; use `exclude_tensors: none` when appropriate.

### Conditions a comparative cohort must satisfy

All hold simultaneously, or the numbers are not comparable:

- **One session.** A single device, node, session and job for every arm. A diagnostic
  cross-session resume is not a comparative cohort, whatever its status token says.
- **Baseline first, fixed order**, so ordering effects land identically on each arm.
- **Gated warmup, discarded.** The warmup is gated on the arm actually being served; an
  ungated warmup can time a decline.
- **At least three rounds**, with the round drift reported, not averaged away.
- **Isolated caches and logs** per arm, so no arm inherits another's compiled or tuned
  state.
- **Correctness separately, once per corpus per arm** — never inferred from a timing
  row.

Reference capability must cover the approved features and shapes. Narrow the corpus
under an explicit scope decision to keep runtime affordable, never by dropping
correctness obligations. Final measurements use fresh output produced after the final
generation, build and install.

### Terminal sweep statuses

| Status | Exit | Means | Accepts? |
|---|---|---|---|
| `SWEEP_DONE` | 0 | Every required phase and gate completed **with correctness enabled** | Yes — the only final acceptance |
| `SWEEP_TIMING_ONLY` | 0 | Reached under explicit `correctness.enabled: false`; timing ran, correctness was never asked | **No** — exit 0 here is not success |
| `SWEEP_INCOMPLETE` | 1 | One or more required gates unmet | No |
| — | 2 | Invalid config, operational error, or interruption | No |

Exit 0 alone proves nothing: read the status token. Timing results cannot excuse a
failed or missing comparison, a numerical mismatch, a NaN or unwritten output, and an
infrastructure failure is never reportable as a kernel test result.

## Complete final runtime join

Every input in every final corpus/phase needs an outcome ledger row. These fields are
the complete required set:

| Field | Required evidence |
|---|---|
| Semantic key | All request fields except provenance |
| Original occurrence | Corpus/source/graph identity and staged file identity |
| Input binding | Phase key/fingerprint and final artifact identities |
| Attribution | Exact expected and observed engine ID/name |
| Outcome | Served, explicitly declined, execution error, missing or ambiguous |
| Evidence | Result/log path and actually observed decline reason where applicable |

Absence of a timing row is not a decline. A decline is only what runtime evidence
recorded; reasons unavailable there cannot be reconstructed from offline policy.

The join is **corpus/phase-local** — never across corpora or phases — and rejects an
input with no outcome row, a graph name duplicated or otherwise ambiguous within its
corpus, and an input-binding fingerprint that does not match the final artifact
identities.

Semantic deduplication must not discard original occurrences: the ledger carries one row
per original occurrence even when several share a semantic key. Missing, ambiguous and
error outcomes block runtime acceptance; they are not roundable to a decline.

Only the complete join supports the per-corpus graph-name-to-reason JSON consumed by
`reconcile_applicability.py --declines`. Do not mix same-named graphs across corpora or
present a sparse accepted mapping as complete runtime evidence. Without the join,
reconciliation is **offline only**.

## Required reporting statistics

A measurement report states all of the following; none substitutes for another.

| Statistic | Definition |
|---|---|
| Coverage counts | Covered, servable and total, each by original source |
| Exact-engine served | Count whose observed engine ID equals the expected installed UED's ID |
| Independently validated | Count whose numerics were checked against a capable independent reference |
| Declines, exclusions, blocked | Every one, by original source, with its actually observed reason |
| Geomean of ratios | Geometric mean of per-input arm/baseline time ratios |
| Time-weighted total | Sum-baseline over sum-arm across the corpus, reported alongside the geomean |
| Round drift | Spread across the at-least-three rounds, so a single round cannot stand alone |
| Byte-identical controls | Determined from artifact hashes, not from equal displayed metadata |

A minimum served floor does not waive any other line. Changed final installed content
invalidates every result bound to the old inputs; re-measure rather than reuse.
