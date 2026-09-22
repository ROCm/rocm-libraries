# Graph contract

Record these facts for RUNBOOK's contract stage, from source semantics and actual
graphs rather than a completed checklist.

## Operations, topology and UID edges

A kernel may implement a node, legal fusion or subgraph; absence of a dedicated
FlatBuffers table for a fusion is not a schema gap. Read the current catalog in
`projects/hipdnn/frontend/include/hipdnn_frontend/node/NodeType.hpp`, matched
`projects/hipdnn/flatbuffers_sdk/schemas/*_attributes.fbs`, frontend attributes/nodes
and shipped multi-node bundles.

Record for every edge: producer/consumer tensor UIDs, shape, dtype, layout, virtual
status, and internal versus exposed. State whether the kernel serves the whole graph or
a supported subgraph. Node names do not establish connectivity, and there is no generic
`inputs`/`outputs` JSON mapping to invent. A shipped `nodeCount() != 1` gate is that
engine's single-node contract, not a framework restriction.

## Field dispositions and kernel mapping

Every matched schema field must be consumed, explicitly rejected when non-inert, or
proven inert under an enforced condition — including alternate/deprecated spellings,
optional UIDs, scalars and output fields. Read frontend defaults and actual
runtime/reference interpretation; framework defaults are not hipDNN omission semantics.
Decline a required launch scalar with no graph-defined default.

For SDPA, keep independent Q/K/V dimensions, GQA, dtype and layout constraints; Q/K
contraction width does not fix V/output width. Distinguish unmasked, causal and
bounded-window requests, alignment and deprecated-field precedence, and derive the
key-set equations on both sides, since window conventions may differ by one or by an
alignment offset. Fused intermediates add edge constraints no single attribute table
describes.

Map each kernel field to graph spelling and source evidence: direct correspondence,
vocabulary translation, derivation, capacity bound, tuning-only, or genuinely
unrepresentable. Candidate-dependent facts belong in `kernel_match`, engine-wide
rejection in `graph_match`. A capacity may require inequality; a performance knob needs
no fictitious graph field. Take specialization bindings from
[rocke-mining.md](rocke-mining.md), not a copied compiler policy.

## Corpus and numerical reference

Compare real in-tree and external graphs on per-operand dims/strides, optional-field
spellings, topology and shape magnitude. Read workload manifests before exclusions;
`microbench/` is provenance, not evidence of synthetic data. Preserve semantic identity
and all source occurrences per [workloads.md](workloads.md). Malformed or
unrepresentable inputs remain explicit outcomes.

Read the current reference limits — the **What the reference executors cannot verify**
section, documented from the repository root in
`dnn-providers/integration-tests/README.md` — and the actual plan predicates. Both
current CPU and GPU SDPA plans reject `sink_token_tensor_uid`; **CPU is not a sink
fallback**. Representation does not establish reference support, and `auto` may exhaust
golden/GPU/CPU choices and skip, validating nothing. Missing capable independent
numerics blocks the feature. Do not fabricate golden data or copy the implementation
into a private reference.

Paged KV (block-table indirection), lengths inside padded buffers and offset-based
ragged storage are distinct semantics; a name such as `seq_lens_ptr`, a dense
approximation or an unavailable reference cannot discharge them. Before declaring a
schema gap, inspect legal composition, optional modes, semantic equivalents and planned
schema work. Partial expressibility requires a scope decision, not silent narrowing, and
`CUSTOM_OP` is not a shortcut to general reference coverage.

Retain topology/edges, field dispositions, default/deprecation observations, graph
examples and mappings. Unresolved correctness-relevant questions block the affected path
until explicitly decided.
