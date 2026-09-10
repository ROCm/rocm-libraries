# Graph contract: what hipDNN can ask for

This is reference material for [RUNBOOK.md](RUNBOOK.md)'s contract stage, not an
independent execution sequence. Record the graph contract before translating
kernel restrictions into native hooks. The required evidence is semantic, not
merely a file with a nonempty table.

## Operations and real topology

hipDNN is a graph API. A kernel may implement a single node, a legal fusion or a
subgraph; absence of a dedicated FlatBuffers table for a fused kernel is not a
schema gap. The current catalog is in
`projects/hipdnn/frontend/include/hipdnn_frontend/node/NodeType.hpp` and
`projects/hipdnn/flatbuffers_sdk/schemas/*_attributes.fbs`. Read actual frontend
attributes/nodes and shipped multi-node bundles as well as that catalog.

Record the kernel's mathematics, matching node types and **tensor-UID edges**.
For each fused edge identify producer/consumer, shape, dtype, layout and virtual
status, and whether it is internal or exposed. State whether the kernel serves the
whole graph or the supported subgraph. Walk the graph's actual attribute/UID
representation; a list of node names is not topology, and a guessed generic
`inputs`/`outputs` JSON mapping is not evidence of how this schema encodes edges.

`graph_match` sees the graph. A shipped pack's `nodeCount() != 1` gate is that
pack's single-node contract, not a framework limit. Never copy it into a fusion.
Conversely, recognizing the right node types without checking their connectivity
can admit a different computation.

## Every field and edge has a disposition

For each matched attribute table, account for every field: consumed with its
semantics, explicitly rejected when present/non-inert, or proven inert under a
stated enforced condition. Unchecked is not an acceptable category. One feature
may have a scalar, a mode, multiple optional tensor UIDs and output fields;
rejecting only one spelling silently accepts the others.

Read each schema beside the frontend header for defaults, deprecations and intent.
Cross-check actual runtime/reference interpretation and the corresponding
framework operator. A framework default is not automatically hipDNN's omission
semantics. Where a required ABI scalar is absent and the graph contract supplies
no default, decline rather than inventing one.

For SDPA, retain independent Q/K/V dimensions and dtype/layout constraints. Equal
Q/K contraction dimensions do not imply an equal V/output dimension. Distinguish
unmasked, causal and bounded-window requests, including alignment and deprecated
field precedence. Do not infer a mask from a filename. Derive the key-set equations
on both sides; a window convention may differ by one or by alignment offset.

A fusion's intermediate constraints belong here too: no single node attribute
table accounts for the edge dtype/layout assumptions the kernel bakes in.

## Mapping to kernel semantics

For each relevant kernel field, record the graph spelling, source evidence and
whether it is a direct correspondence, vocabulary translation, derivation,
capacity bound, tuning-only field or genuinely unrepresentable semantic feature.
The mapping is not an independent compiler-policy implementation. Effective
specialization bindings are described in [rocke-mining.md](rocke-mining.md) and the
[generator reference](../../../IngestorGenerator/README.md).

Graph-derived quantities compared with candidate specialization belong in
`kernel_match`; facts rejecting the graph regardless of candidate belong in
`graph_match`. A capacity can require an inequality rather than equality. A
performance knob is not automatically a graph feature and does not require a
fictitious graph field. If a kernel cannot implement a graph feature, reject that
feature explicitly. If a semantic feature really cannot be expressed, do not
silently enable it in a UKD or label it matcher-only to evade a proof obligation.

## Real graphs and corpus identity

Read actual graphs from both in-tree integration bundles and the declared external
corpus. Record differences in layouts, optional-field spelling, topology and shape
magnitude. `microbench/` is provenance, not proof of synthetic data; read the corpus
manifest before proposing any exclusion.

Carry every semantic request field into shape identity, including window and sink
semantics and independent operand dimensions. Exclude provenance from semantic
deduplication, but retain every original corpus/source/graph occurrence. See
[workloads.md](workloads.md) for full denominators and the final runtime join.
Malformed/unrepresentable inputs are explicit findings, not silently dropped rows.

## Numerical reference capability

Use the current [integration-test reference limits](../../../../../../dnn-providers/integration-tests/README.md#what-the-reference-executors-cannot-verify)
and actual plan predicates to establish that the selected reference can evaluate
the intended graph. Representation and reference support are different questions.
Both current CPU and GPU SDPA plans reject `sink_token_tensor_uid`. CPU is **not**
a sink fallback, regardless of whether a sink seems mathematically expressible by
a dense reference. The declared backend must actually support the graph.

Paged KV, valid-length/varlen inputs and ragged tensor storage are distinct:
block-table indirection, lengths inside padded buffers, and offset-based unpadded
storage respectively. A parameter name such as `seq_lens_ptr` does not tell you
which is meant. Neither a dense shape approximation nor an unavailable reference
can discharge these semantics.

`auto` verification can exhaust golden/GPU/CPU choices and skip. Such a skip does
not validate the engine. Missing capable numerics blocks the feature; bring a
scoped proposal to the user instead of fabricating golden data or adding a private
reference copied from the implementation under test.

## Evidence required before implementation

Retain the matched topology/UID-edge inventory, per-table field dispositions,
frontend/default/deprecation observations, actual corpus examples and differences,
and graph-to-kernel mappings. Unresolved semantic questions block the affected
implementation until decided; an arbitrary reading cap or elapsed time cannot
turn them into assumptions.

Before concluding that nothing maps, check semantic equivalents, legal node
composition, existing-node optional modes, frontend/compatibility vocabulary and
planned schema work. A partly expressible fusion requires an explicit scope
decision, not automatic narrowing. `CUSTOM_OP` has no general reference coverage
and is not a shortcut to numerical acceptance. If the request still cannot be
expressed or verified, report the missing mechanism and alternatives, then stop
that path. Return to RUNBOOK for the next stage only when its gate is met.
