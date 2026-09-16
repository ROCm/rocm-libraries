# Reading the graph, specifying the operation

[RUNBOOK.md](RUNBOOK.md) owns execution order; this page owns what a graph *is* and
what must be extracted from it before a line of kernel source is written. Paths are
relative to `projects/hipdnn` unless stated otherwise.

## Five forms, one object

| Form | Produced by | Read with |
|---|---|---|
| C++ source that builds a graph (a sample, a test, a reproducer) | a human | read it, then build and lower it to obtain the forms below; dims and dtypes may be command-line parameters, so record the invocation too |
| Live `hipdnn_frontend::graph::Graph` | the fluent builder (`Graph::tensor()`, `graph.<op>(...)`) | `graph.getTensorsByName()`, the `TensorAttributes` each builder call returns |
| JSON string | `Graph::serialize(std::string&)` | any JSON reader for the key layout (`flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp`); to get a *live* graph back, `Graph::from_json(json)` — a wrapper over `deserialize(jsonData)`, structure only, so `build_operation_graph()` finalizes it (`frontend/include/hipdnn_frontend/Graph.hpp:3010-3013`, `:3055-3058`) |
| Binary FlatBuffers blob | `Graph::serialize(std::vector<uint8_t>&)` / `to_binary()` | `GraphWrapper::fromSerializedBlob(buffer, size)`; `Graph::from_binary(data)` to reload it into a live graph |
| Template-sweep bundle: a shared `graph.template.json` plus a `sweep.json` of cases | the integration tests' bundle layout | expand it the way the loader does — **never textually**; see below |

`GraphWrapper` (`flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp`)
is the accessor to prefer for a handed graph: `nodeCount()`, `getNode(i)` — whose
`attributes_type()` identifies the op through the `NodeAttributes` union — and
`getTensorMap()`, a `uid -> const TensorAttributes*` map. The binary blob is also
exactly what the reference executor consumes, so the object you inspect and the
object you validate against are the same bytes.

There is no Python binding and no environment variable that dumps a graph. A dump is
always an explicit `serialize` call in caller code.

### UIDs exist only after lowering

`TensorAttributes::_uid` defaults to `0` and `has_uid()` to `false`. Callers may
assign UIDs with `set_uid()`, and most samples do not. Lowering assigns the rest:
`lowerGraphToDescriptors` calls `assignUnsetTensorUids()` before `validate()`
(`frontend/include/hipdnn_frontend/Graph.hpp:840-846, 864-870`), which fills every
unset tensor with an unused id (`detail/GraphTensorIds.hpp`).

So `get_uid()` on a freshly built graph is not yet meaningful, and a UID read before
lowering is a bug in your inspection, not a property of the graph. Enumerate UIDs
from a lowered graph or from a serialized blob — the blob is post-lowering by
construction. Before lowering, a tensor's identity is its object and its name.
Assigned ids are stable within one lowering, not across program runs, so key your
notes on names and record the id alongside.

### A template-sweep bundle is expanded in C++, not by string replacement

A sweep root is any directory holding both a `graph.template.json` and a `sweep.json`,
and each `cases[]` entry in the sweep expands the template into one logical graph
(paths in this section are relative to the repository root:
`dnn-providers/integration-tests/src/harness/bundle/BundleDiscovery.hpp:261-263`,
`IntegrationTestBundle.hpp:799-830`). A naive textual `${...}` substitution produces a
graph that parses, classifies and specifies cleanly — and is the wrong graph. Step 3's
gate cannot catch that, so the discipline has to sit here.

What the loader actually does:

- A placeholder is a **whole JSON value** `"${field}"`, replaced by the case's *value*
  — a number or an array — not spliced into surrounding text
  (`dnn-providers/integration-tests/src/harness/bundle/IntegrationTestBundle.hpp:380-396`,
  `:502-513`).
- Resolution is **uid-scoped**. The walk carries the enclosing object's `uid` as the
  current tensor (`:526-540`), and a placeholder resolves first against *that uid's*
  entry in the case's per-uid tensor map, falling back to the case's global `values`
  only afterwards (`buildCaseTensorMap`, `:431-466`; `resolvePlaceholder`, `:468-500`).
  So one `${dims}` in the template means a different value per tensor.
- `dims`, `strides` and `data_type` are per-tensor by rule: a missing per-uid value is
  an error, never a fallback to the global one (`requiresPerTensorValue`, `:398-402`).
- Structural `tensor_patches` are applied by uid *after* expansion, setting and erasing
  whole fields (`applyTensorPatches`, `:609-667`). A substitution pass skips them
  entirely.

Reproducing that by hand is the only path available to a standalone program: the
integration-tests harness is source-tree only and not linkable from an install prefix
([harness.md](harness.md)). So read the loader and reproduce it; do not improve on it.
Cheapest alternative, when the bundle permits it: expand one case, serialize the result,
and work from the blob — one form, post-expansion, for the rest of the workflow.

## What to extract

`tensor_attributes.fbs` carries `uid`, `name`, `data_type`, `dims`, `strides`,
`virtual`, `value`, `is_runtime_pass_by_value`, `ragged_offset_tensor_uid`,
`alignment`. `graph.fbs` carries the tensor list, the node list, and the graph-level
`compute/intermediate/io_data_type`.

- **Edges are UID references, not an edge list.** Cross-reference each node's
  attribute table's `*_tensor_uid` fields against the tensor map. The node order is
  topologically sorted.
- **Layout is not a field.** It exists only as the relationship between `strides` and
  `dims`. Derive it; never assume packed row-major because the dims look familiar.
- **`virtual: true` means no caller buffer.** A virtual tensor is an internal edge.
  It is what makes a graph fused, and it is what a multi-launch decomposition must
  find scratch for.
- **Absent and explicitly null are both absence** — read them identically and
  explicitly at the read site.

### Role is not a tensor field — derive it from the node edges

`tensor_attributes.fbs` has no role. `getTensorMap()` answers dims, strides, dtype and
`virtual` per uid and nothing about input versus output, so a tensor's role is a
property of the *nodes* that reference it. There are exactly two derivations, matching
the two forms you may hold:

- **From a live `Graph`:** `graph.visit(...)` over the nodes, taking
  `INode::getNodeOutputTensorAttributes()` and `getNodeInputTensorAttributes()`. That is
  what the integration harness does to build its two bundles — outputs first, recorded
  as output uids, then inputs
  (`dnn-providers/integration-tests/src/harness/IntegrationGraphVerificationHarness.hpp:323-337`).
- **From JSON or a blob:** the uid's *position* in the node's attribute table. The field
  name carries the role — `x_tensor_uid` and `w_tensor_uid` are inputs, `y_tensor_uid`
  is the output (`flatbuffers_sdk/schemas/convolution_fwd_attributes.fbs:13-15`). The
  bundle loader reads exactly those positions: each node's `outputs` sub-object, or a
  flat `output_tensor_uids` array, or the flat `out_tensor_uid` / `y_tensor_uid`
  spellings that Reduction, ResampleFwd and CustomOp use instead
  (`dnn-providers/integration-tests/src/harness/bundle/IntegrationTestBundle.hpp:153-201`),
  and then drops the ones the tensor map marks `virtual_()` (`:750-762`).

The rule the two share: **output** is a uid named in a node's output position;
**virtual** is the tensor map's own flag; a **graph output** is output and not virtual;
**input** is a uid referenced in an input position and never in an output position. Both
derivations are in tree and both are load-bearing — reproduce the one that matches your
form; do not improve on it.

### Disposition is per enumerator, not per field

"Every matched schema field consumed, rejected or proven inert" is a field-level
rubric, and an enumerated field passes it while only one of its values works. A kernel
that implements `CROSS_CORRELATION` and quietly treats everything else the same way has
consumed `conv_mode` and is wrong for `CONVOLUTION` — the two are separate enumerators
of `ConvMode` (`flatbuffers_sdk/schemas/convolution_common.fbs:6-10`), and nothing in
the graph, the compile or the harness distinguishes them for you.

So for an enumerated field the disposition is **per enumerator**: each value either
implemented, explicitly rejected at a stated runtime check, or proven unreachable for
this graph under a condition you name. "The graph I was handed uses this one" is a
scope statement, not a disposition — record it as the third case, with the check.

**The zero enumerator is one of them.** FlatBuffers gives an absent scalar its default,
and these enums spell that default `UNSET` (`:7`). For an enum, therefore, absence is
not a hole: it is delivered as a valid enumerator carrying no defined kernel meaning.
That is how it interacts with *absent and explicitly null are both absence* above — the
reading rule is unchanged, absent and null are still read identically, but for an enum
what they both read as is `UNSET`, which is one more enumerator needing a disposition.
Reject it at a stated check, or resolve it to the default the frontend attribute class
defines and cite where that default lives. Never let it fall through to the enumerator
you happened to implement.

## Where the semantics actually live

Ground truth for an operation is the pair: the frontend attribute class in
`frontend/include/hipdnn_frontend/attributes/<Op>Attributes.hpp` and its node in
`frontend/include/hipdnn_frontend/node/<Op>Node.hpp` (defaults, validation), together
with the matched wire table in `flatbuffers_sdk/schemas/<op>_attributes.fbs`. The op
catalog — 23 node types — is `frontend/include/hipdnn_frontend/node/NodeType.hpp`.

Ground truth is **not** the reference executor's plan/signature-key files: they encode
what the reference implements, not what the operation means. It is also not the
framework the graph came from; another library's omission default is not hipDNN's.

### Conventions that carry real traps

- **Mask semantics (SDPA).** Deprecated `causal_mask` / `causal_mask_bottom_right`
  booleans take precedence over `diagonal_alignment` in the shared `getMaskType`, so
  a bottom-right-causal graph can be silently read as top-left — wrong triangle, no
  error. Read the precedence,
  do not infer it.
- **Independent operand widths.** Q/K contraction width does not fix V/output width.
- **Scale, epsilon and other scalars.** A scalar may arrive as an attribute, as a
  pass-by-value tensor, or as a device buffer. Which one this graph uses changes the
  kernel's argument list.
- **Window and offset conventions** differ by one between libraries. Derive the
  key-set equation on both sides rather than copying an implementation's bounds.

## The operation specification

Before authoring, write down — in the report, not only in reasoning:

1. The mathematics, with the accumulation order and the accumulate dtype named.
2. Per tensor: UID, role (input/output/virtual) derived as *Role is not a tensor
   field* above requires, dims, strides, dtype, alignment.
3. Per matched schema field: consumed, explicitly rejected, or inert under a stated
   enforced condition — including alternate and deprecated spellings and optional UIDs,
   and, where the field is an enum, one such disposition **per enumerator**.
4. The numeric envelope: dtypes, layouts, and the shape range claimed.
5. Every question left unresolved. An unresolved correctness-relevant question blocks
   the path it touches.

## Fusion or multiple launches

A kernel may serve a node, a legal fusion, or a subgraph. Decide per edge:

| Producer → consumer edge | Decision |
|---|---|
| Elementwise, same iteration space | Fuse. The consumer reads a register, not memory. |
| Consumer's tile reads only the producer's tile | Fuse, tiled together. |
| Consumer needs a value the whole grid must finish producing (a full reduction, a normalization statistic, a transpose across tiles) | Separate launch. A grid-wide barrier inside one kernel is not available. |
| Producer output is also a graph output | It needs a real buffer either way; fusing is then a write, not an elision. |

Multiple launches are a correct answer. Their price is scratch for each virtual
intermediate and a documented launch order. State both. If the intended integration
ABI cannot supply that scratch, that is an integration constraint to raise
immediately, not a reason to silently re-derive the mathematics.

## Generalization, concretely

Prefer a kernel general over the schema's own dimensions; specialize only where
generality would change the code's structure.

| Quantity | Treatment | Why |
|---|---|---|
| batch, heads, sequence lengths, channels, spatial dims, counts | kernel argument | costs registers, not correctness |
| strides, offsets, alignments | kernel argument | the graph already varies them |
| scalars (scale, epsilon, alpha) | kernel argument, unless the graph fixes them | a value bound as a macro cannot be rendered if it is a float — see [device-envelope.md](device-envelope.md) |
| element type | `-D` specialization | changes the declared types |
| tile shape, unroll factor, shared-memory budget, vector width | `-D` specialization | changes the code's structure |
| "supported shapes" as an enumerated table | do not | this is the coverage model of existing engines, recorded as a limitation, not a design to copy |

A generality claim is bounded by what ran. Parameterizing over a dimension and
validating it at one value is "parameterized, validated at one point" — and a kernel
whose correctness depends on a size being a multiple of the tile must either handle
the remainder or reject the shape explicitly, never compute it wrong quietly.

## Worked examples in tree

| Path | Shape of graph |
|---|---|
| `samples/sdpa/SdpaFprop.cpp` | single-node `SDPA_FWD`, attribute-heavy |
| `samples/convolution/FusedConvFpropBiasActiv.cpp` | conv + bias + activation through virtual intermediates |
| `samples/batchnorm/FusedBnInfDReluBnBwd.cpp` | three nodes chained through two virtual tensors, and the in-tree CPU-reference validation pattern |

Larger and more varied real graphs come from the external `ROCm/dnn-benchmarking`
graph-JSON corpora referenced by `hipdnn-ingestor-engine`'s
[workloads.md](../hipdnn-ingestor-engine/workloads.md), not from `samples/`. Corpus
identity and coverage accounting are that skill's, not the integration skill's:
[hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md) takes graph coverage
as bundle cases instead.
