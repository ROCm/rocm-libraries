# gfx950 forward convolution verification and coverage

This snapshot records the specialized POC verified on gfx950 on 2026-09-09.
The shipped catalog contains 71 variants for 61 requests: ten verification
requests with two `tile_k` values each, plus 51 distinct headline requests with
the dispatcher's default `tile_k=64`. The [graph contract](graph_contract.md)
defines the accepted operation and storage semantics.

## Verification

| Check | Result |
|---|---|
| Adapter, original-builder IR equivalence, library layering, and miner tests | 98 passed (56 adapter/layering and 42 miner). |
| Provider unit tests | 21 passed, zero skipped. |
| Provider GPU integration tests | 24 passed, zero skipped; includes all 20 forced verification variants with two input seeds. |
| Installed shared integration harness | 11 passed, zero skipped, through the exact engine name; both registered CTest and direct `--fail-on-unsupported` invocation passed. |
| Installed descriptor validator | Passed, including all four native symbols and exact engine registration. |
| Catalog checks | 71 unique packaged symbols/TOC entries, complete metadata equality, and 71 reachable forced variants. |
| Native geometry comparison | Agreed with rocKE on 1,093 cases; eight additional default-epilogue cases also checked. |
| Independent corpus correctness | 106 source graphs passed PyTorch comparison; zero failed or skipped. |
| Direct rocKE comparison | Both smoke tuning choices for both dtypes and all 51 headline requests passed independent correctness before timing. |
| Selection and cache | Both dtypes measured two candidates, selected the fastest recorded candidate, and reused the winner on repeated execution, plan recreation, and separate-process restart. |

Engine attribution is explicit: `hipkernel:Gfx950ConvFwd`
(`0xAD075A5EA86DD563`). Selecting another engine cannot satisfy these checks.
The installed descriptor loader emits its existing warnings for the optional
`provenance` extension; validation reported no errors or unresolved symbols.

The 106 corpus graphs represent 51 distinct requests. Together with the ten
small verification requests, they exercise every shipped request. Duplicate
source graphs remain separate coverage records so their provenance is retained.

## External corpus

The inventory combines rocKE's canonical `bench_cases_conv.json` at
`faf4621de4be118568e30eb8f69f8f743d4eb773` and dnn-benchmarking's headline
convolution and convolution-sweep archives at
`36f95cc7a8b8b3ee5ba82b195a8e51a7cd47f6cc`. The miner preserves source hashes,
URIs, original attributes, exclusions, and deduplicated requests. It follows
tensor attributes when filenames and dtypes disagree.

| Source | Records | Graphs audited | Dispatched and correct | Declined | Could not be rendered faithfully |
|---|---:|---:|---:|---:|---:|
| rocKE canonical cases | 111 | 96 | 51 | 45 | 15 |
| Headline convolution | 179 | 179 | 55 | 124 | 0 |
| Convolution sweep | 429 | 429 | 0 | 429 | 0 |
| Total | 719 | 704 | 106 | 598 | 15 |

These are actual installed-runtime counts. All 704 representable graphs passed
frontend construction. The exact engine declined 598 at engine selection;
there were no discrepancies between predicted and observed dispatch and no
audit errors. Every accepted graph then ran against an independent PyTorch
reference, with explicit execution and tolerance-comparison success checked.
The miner also rejects shape-override-enabled graphs as outside this fixed-shape
contract. None of the 704 audited source graphs enables that flag.

## Remaining coverage

The 598 runtime declines split into three populations:

- **349 backward graphs:** 168 input-gradient and 181 weight-gradient graphs.
  They request different operations from this forward engine.
- **124 catalog gaps:** these are 124 distinct 2D forward sweep requests within
  the integration contract. The real rocKE factory and validity predicate accept
  every request, but no matching variant ships. They remain open integration
  coverage work; they are not kernel-family limitations.
- **125 forward graphs outside the integration contract:** the graph matcher
  rejects their recorded attributes, including grouped, 1D/3D, other layouts or
  dtypes, and asymmetric padding. The family comparison below distinguishes
  proven support from requests that cannot be projected faithfully.

For those 125 forward declines, the scoped rocKE comparison reports:

| Classification | Records | Evidence and interpretation |
|---|---:|---|
| Family accepts an equivalent request | 71 | Additional integration-contract gaps, including grouped and 3D convolution. Acceptance is a capability/predicate result; these declined graphs have not been validated numerically through this engine. |
| Family declines an equivalent request | 13 | Two channels-first requests, seven actual int8 requests, and four FP32 requests fail the forward family's layout or dtype capability gate. |
| No faithful request projection | 41 | Twenty-three asymmetrically padded 3D graphs cannot be represented by the oracle's symmetric-padding request fields; eighteen 1D graphs are outside its 2D/3D projection. Some also mix operand layouts or have asymmetric padding. No rocKE support verdict is claimed. |

The remaining 15 canonical source records were never submitted as simplified
graphs: two composed sums, one transpose convolution, one deformable convolution,
three streaming/cache-equivalence workloads, and eight causal 1D workloads with
implicit cropping. Preserving those semantics requires additional graph or
state modeling. They are recorded as unrenderable source records, not runtime
declines or successful tests.

Of 175 distinct eligible requests, the catalog covers 51 and omits 124. The
broader runbook's goal of zero reference-only declines therefore remains open.
The next catalog expansion should start with those 124 requests, followed by
separate contract and reference work for grouped, 3D, and fused operations.
The successful POC checks do not establish those broader capabilities.

## Performance evidence

The integration probe checks the installed descriptor UUID and all compiled
spec fields, rebuilds direct rocKE from the equivalent original-builder spec,
and uses the same gfx950 device, LLVM flavor, and COMGR library. Compilation,
plan creation, first execution/search, correctness, capture, and warmup are
excluded from steady-state kernel comparison. HIP graph replay and ordinary
submission timing are reported separately.
Selection-log callbacks and backend/plugin logging are suspended during timing
and restored for subsequent cache assertions. The report checks the logging
state and zero callback invocations within that interval.

Candidate ranking uses the existing runtime's one warmup and seven timed
executions with `robustMean`; the probe's steady-state report uses the median
of sample batches. These are separate measurements. A winning variant is the
fastest valid candidate measured during that search, with no claim that a close
ordering is invariant across runs. Persistent cache checks compare UUIDs and
ranking hashes across different process IDs.

Measured performance reports and historical baseline comparisons are retained
as private evidence outside the source tree. The 51 headline requests, the
48-request canonical subset, and historical-baseline matches are reported as
separate populations. Historical timings do not establish integration overhead;
the current same-spec, same-device comparison supplies that evidence.
