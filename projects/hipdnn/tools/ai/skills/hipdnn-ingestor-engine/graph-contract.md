# Graph contract

Record these facts for RUNBOOK's contract stage. The evidence is source semantics
and actual graphs, not merely a completed checklist.

## Operations, topology and UID edges

A kernel may implement a node, legal fusion or subgraph. Absence of a dedicated
FlatBuffers table for a fusion is not a schema gap. Read the current catalog in
`projects/hipdnn/frontend/include/hipdnn_frontend/node/NodeType.hpp`, matched
`projects/hipdnn/flatbuffers_sdk/schemas/*_attributes.fbs`, frontend attributes/nodes
and shipped multi-node bundles.

For every edge, record producer/consumer tensor UIDs, shape, dtype, layout, virtual
status and whether it is internal or exposed. State whether the kernel serves the
whole graph or a supported subgraph. Node names alone do not establish connectivity;
do not invent a generic `inputs`/`outputs` JSON mapping. A shipped `nodeCount() != 1`
gate is that engine's single-node contract, not a framework restriction.

## Field dispositions and kernel mapping

Every matched schema field must be consumed, explicitly rejected when non-inert,
or proven inert under an enforced condition. Include alternate/deprecated spellings,
optional UIDs, scalars and output fields. Read frontend defaults and actual
runtime/reference interpretation; framework defaults are not automatically hipDNN
omission semantics. Decline a required launch scalar with no graph-defined default.

For SDPA, keep independent Q/K/V dimensions, GQA, dtype and layout constraints.
Q/K contraction width does not fix V/output width. Distinguish unmasked, causal and
bounded-window requests, alignment and deprecated-field precedence. Derive the key-set
equations on both sides: window conventions may differ by one or alignment offset.
Fused intermediates add edge constraints that no single attribute table describes.

Map each kernel field to graph spelling and source evidence: direct correspondence,
vocabulary translation, derivation, capacity bound, tuning-only, or genuinely
unrepresentable semantics. Candidate-dependent facts belong in `kernel_match`;
engine-wide rejection belongs in `graph_match`. A capacity may require inequality;
a performance knob needs no fictitious graph field. Use [rocke-mining.md](rocke-mining.md)
for specialization bindings, not an independently copied compiler policy.

**Declare the admitted envelope as data.** The envelope the engine's matcher enforces
is a contract, not an implementation detail of its control flow. Declare, at minimum:
node type and count; admitted ranks; admitted dtypes per operand role; required operand
shape relations; stride admissibility; virtual and pass-by-value disposition; aliasing
rules; and every node-level field the matcher gates on. The declaration is the input to
coverage computation, to RUNBOOK §2's scope rules and to its matcher-versus-reference
parity check. Drift runs in both directions: a matcher enforcing a condition the
declaration omits is drift, and so is a declaration claiming a condition the matcher
does not enforce.

*No config schema carries this today.* The generator config's engine-level graph-match
block is layout and discrimination documentation only
(`IngestorGenerator/codegen/models.py:324-336`). Until a schema carries the envelope,
reconstruct it by reading the `graph_match` and kernel-matcher bodies, record it in the
run's evidence as data with the source lines each item came from, and treat it as valid
only as of the revision you read. The drift rule applies to the reconstruction
unchanged, and the reconstruction is a recorded escalation under
[RUNBOOK](RUNBOOK.md#tools-this-skill-assumes-and-what-to-do-without-them)'s tools
section — never a discharged fact, because nothing recomputes it when the matcher
changes.

## Corpus and numerical reference

Compare real in-tree and external graphs: per-operand dims/strides, optional-field
spellings, topology and shape magnitude. Read workload manifests before exclusions;
`microbench/` is provenance, not evidence of synthetic data. Preserve semantic
identity and all source occurrences as specified by [workloads.md](workloads.md).
Malformed or unrepresentable inputs remain explicit outcomes.

Read the current [reference limits](../../../../../../dnn-providers/integration-tests/README.md#what-the-reference-executors-cannot-verify)
and actual plan predicates. Both current CPU and GPU SDPA plans reject
`sink_token_tensor_uid`; **CPU is not a sink fallback**. Representation does not
establish reference support. `auto` may exhaust golden/GPU/CPU choices and skip;
that validates nothing. Missing capable independent numerics blocks the feature.
Do not fabricate golden data or copy the implementation into a private reference.

**Derive the mode from capability, not from a default.** Resolve which reference
executors implement the operation under test before selecting one. The GPU reference
implements a plan builder per op family; which families those are today, the headers
that are their source of truth, and the gpu/cpu/never-`auto` decision table are owned
by [hipdnn-kernel-integration](../hipdnn-kernel-integration/RUNBOOK.md) and are
maintained there as one copy — link to it rather than restating the list, so an added
family does not have to be added in two places. A reference that does not implement
the operation declines every case and can still exit zero, reporting a pass in which
nothing was verified. **Zero selected, or every case declined, is a failure regardless
of exit status**, and selected, served, declined and failed are four counts to be
recorded separately, never one.

Paged KV (block-table indirection), lengths inside padded buffers and offset-based
ragged storage are distinct. A name such as `seq_lens_ptr`, dense approximation or
unavailable reference cannot discharge their semantics. Before declaring a schema
gap, inspect legal composition, optional modes, semantic equivalents and planned
schema work. Partial expressibility requires a scope decision, not silent narrowing;
`CUSTOM_OP` is not a shortcut to general reference coverage.

Retain topology/edges, field dispositions, default/deprecation observations, graph
examples and mappings. Unresolved correctness-relevant questions block the affected
path until explicitly decided.
