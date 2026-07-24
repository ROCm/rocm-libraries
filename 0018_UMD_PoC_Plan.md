# RFC 0018 UMD — Proof-of-Concept Implementation Plan

Scope: parse a Universal Match Descriptor JSON, build a `UniversalGraphMatcher` that
matches arbitrary graphs and returns a queryable bindings object, with automatic codegen
of the op-schema registry from the `umd_*` annotations. **SDPA forward only.** Driven by a
gtest suite (graphs built in code or from JSON→flatbuffer). Not plumbed as a provider engine.

Proving out **the bindings** — the schema-generated, strongly-typed path→value layer that turns a
UMD variable reference (`$q.head_size`, `$sdpa_fwd.dropout_probability`, name→tensor) into a concrete
value read off the flatbuffer graph — is a first-class deliverable of this PoC, not a side effect of
matching. The phases below generate that layer at build time and prove it end-to-end.

Source RFC: `projects/hipdnn/docs/rfcs/0018_UniversalMatchDescriptor.md`.

---

## What already exists (reuse, don't rebuild)

- **JsonLogic** — full compile-once/evaluate-many implementation at
  `dnn-providers/hip-kernel-provider/include/hip_kernel_provider_common/JsonLogic.hpp`.
  Already implements the RFC `$`-sigil variable convention, `Value` variant
  (bool/int/double/string/array), operators `==`/`!=`/`<`/`<=`/`>`/`>=`/`in`/`and`/`or`/`!`/
  arithmetic/`min`/`max`/`ceil_div`/`abs`/`pow`/`log2`/`rsqrt`/`value_or_default`/`if`, and
  `compile<DataT>(rule)` → `Expression<DataT>` over any `getData(path)->Value` data source.
  Tests already include a `Umd0018ConstraintShapes` case (`src/tests/core/TestJsonLogic.cpp`).
  **Gap: no `shape` short-hand (lowered by the UMD compiler, Phase 1). The custom-operation
  (native-predicate) hook is out of scope for this PoC — see Scope decisions.**
- **UMD annotations already in the schema** — the table-level `umd_opcode` shorthand (e.g.
  `SdpaAttributes (umd_opcode: "sdpa_fwd")`) and the field-level `umd_input_tensor` / `umd_output_tensor` /
  `umd_name` are declared once (`flatbuffers_sdk/schemas/data_types.fbs`) and applied on the SDPA
  table + its UID fields (`flatbuffers_sdk/schemas/sdpa_attributes.fbs`). But flatc runs `--cpp` only
  (`cmake/flatc_flags.txt`); no `.bfbs` is emitted, so the annotations are invisible at
  build/runtime. **Gap: no `.bfbs` emit, no reflection-driven registry generator.**
- **Graph model** — `IGraph` / `GraphWrapper`
  (`flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp`),
  `INodeWrapper::attributesAs<T>()`, `getTensorMap()` (uid→`TensorAttributes*`).
  JSON→flatbuffer graph via `hipdnn_flatbuffers_sdk::json::to<Graph>(builder, json)`
  (`utilities/json/Graph.hpp`). In-code SDPA graph builder `createValidSdpaFwdGraph(...)` in
  `test_sdk/include/hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp:2218`.
- **Test infra** — GoogleTest only. Provider suite `hip_kernel_provider_tests` already links
  `nlohmann_json` and the SDPA `plan_utils` (`getMaskType`, `byteStrideFitsU32`).
  `src/tests/engines/asm_sdpa_engine/TestSdpaFwdPlanBuilder.cpp` is the closest template.
- **Parity oracle** — `SdpaFwdPlanBuilder::isApplicable`
  (`dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.cpp:167-296`):
  the declarative gates the UMD must reproduce. Arch (gfx942/gfx950) and kernel-table lookup are
  **out** of the UMD per RFC §5/§8 (pack property + KDP Launch).

---

## Home / layout

Provider-local, under `dnn-providers/hip-kernel-provider` (reuses JsonLogic + SDPA `plan_utils`).
Registry generation lives in `flatbuffers_sdk` (schema owner), emitting a jlogic-agnostic header
the provider consumes. Matcher is exercised directly by tests — not registered as an engine.

---

## Phase 0 — Op-schema registry codegen (RFC Appendix B)

Goal: generate, from the `umd_*` annotations, a C++ registry keyed by opcode — the table's
`umd_opcode` shorthand (e.g. `sdpa_fwd`), falling back to the `NodeAttributes` union member name when
absent. Each entry also carries the table name (diagnostics) and the integer `NodeAttributes` value
(O(1) lookup against `Node::attributes_type()`), and lists input/output tensors (name, optionality,
UID reader) and scalar attributes (name, optionality, typed reader).

1. **Emit `graph.bfbs`.** Add a dedicated flatc invocation (NOT via `flatc_flags.txt`, which is
   `--cpp`-only) in `projects/hipdnn/cmake/FlatBuffersGenerate.cmake` — a sibling function
   `hipdnn_generate_bfbs(graph.fbs)` running
   `flatc -b --schema -I <schemas> -o <out> graph.fbs`; the custom `umd_*` attributes are retained in
   the `.bfbs` with no extra flag. `graph.fbs` transitively covers every attribute table + the `umd_*`
   declarations. Wire into `flatbuffers_sdk/CMakeLists.txt`.
2. **Generator tool.** Small standalone C++ executable `umd_registry_gen` linking flatbuffers
   reflection (`flatbuffers/reflection.h`). Loads `graph.bfbs`, enumerates `NodeAttributes` union
   members (opcode→table); reads the table's `umd_opcode` shorthand and applies B.3 classification
   per field:
   - `umd_input_tensor` + `umd_name` → input tensor name; `umd_output_tensor` + `umd_name` → output tensor name
     (type MUST be `long`, `umd_name` non-empty).
   - neither flag → scalar attribute, bind-named by field name.
   - optionality derived from `= null` default (not re-annotated).
   - **Fail the build** on B.3 violations (both flags on one field, `umd_name` without a flag,
     flag on a non-integer field, duplicate `umd_name` within an op, a name colliding with a
     reserved token `graph`/`kernel`/`device`, or a duplicate `umd_opcode` across ops).
3. **Emit `op_schema_registry_generated.hpp`** (header-only: inline per-op tables + inline lookup
   functions, so `flatbuffers_sdk` stays INTERFACE-only) into the committed `flatbuffers_sdk` include
   dir (`include/hipdnn_flatbuffers_sdk/umd/`). The neutral types live in a hand-written
   `umd/OpSchemaRegistry.hpp`.
   Neutral (jlogic-agnostic) shape so `flatbuffers_sdk` needn't depend on the provider: per opcode,
   arrays of `{name, optional, UID-reader int64_t(const void*)}` and
   `{attr-name, optional, AttrType, reader→neutral ScalarValue}`. Readers use the generated typed
   accessors (`&SdpaAttributes::q_tensor_uid`, …) — **no runtime reflection** (RFC B.4). The
   generator is general (it emits an entry for every `NodeAttributes` member); SDPA is the only
   annotated op today, so the `SdpaAttributes` entry is the one carrying edges — that entry proves the
   codegen. Each entry carries its `umd_opcode` shorthand (fallback: table name), the table name, and
   the integer `NodeAttributes` value, so the matcher resolves it by `Node::attributes_type()` (O(1)),
   and the UMD compiler resolves a node `op: "sdpa_fwd"` by the shorthand.
   - Enum-typed scalar attributes surface as `AttrType::Dtype` carrying the enum-value name string
     (via the generated `EnumName…`). Unannotated non-scalar fields (vectors/sub-tables) are not UMD
     scalars and are skipped; SDPA's attribute table is all UID + scalar, so it is fully covered.
   - Each attr entry's `AttrType` maps to a jlogic `ValueKind` (Int/Float/Bool/Dtype), consumed by the
     Phase 1 compiler for compile-time criteria type-checking (A.10 §9). This carries strong typing
     from the schema through to expression validation and is the payoff that makes the codegen worth
     more than a hand-written table.
4. Wire `umd_registry_gen` as an `add_custom_command`/target; make it a dependency of
   `hipdnn_flatbuffers_sdk` (the provider consumes the header transitively), gated on
   `HIPDNN_GENERATE_SDK_HEADERS` (default ON) like the existing flatc header generation.

**Verify:** registry unit test — `SdpaAttributes` entry lists Q/K/V (required operands), O (required
result), `attn_mask`/`page_table_k`/`page_table_v` (optional operands), and scalar attrs
(`dropout_probability` optional, `alibi_mask`/`padding_mask`/`causal_mask` with correct optionality).
A B.3-violation fixture fails generation.
- **Accessor value round-trip (proves the bindings, not just the shape):** build an `SdpaAttributes`
  with known UIDs and scalars; assert every generated UID reader and every scalar reader returns the
  exact value. Registry *shape* (names listed) is necessary but not sufficient — this proves the
  generated accessors actually read the right field.
- **Optionality parity with the header:** the generator's `= null`-derived optionality matches the
  generated header's `Optional<T>` fields (an absent optional scalar reports not-present, a required
  scalar always present).

---

## Phase 1 — Bindings system + UMD compiler + matcher

1. **`BindingContext`** (the queryable "bindings object" and the JsonLogic data source). Built per
   graph after structural match; resolves all five namespaces via `getData`:
   - **Tensor**: `$q` (→ uid int64), `$q.uid`, `$q.rank` (`dims->size()`), `$q.dtype` (enum name
     string), `$q.dims[i]`, `$q.strides[i]`, `$q.<named-dim>` (via compiler-supplied name→index map),
     `$q.stride_order` (IntArray, computed from strides as in `ApplicabilityChecks.cpp:17`),
     `$q.packed` (bool, from dims+strides), `$q.virtual`, `$q.present` (optional operands).
     Absent-optional field access → decline (fail closed, A.4).
   - **Graph**: `$graph.node_count`.
   - **Attributes**: `$<node_id>.<attr>`, `$<node_id>.<attr>.present`, `$<node_id>` (→ node index
     for attribute reads).
   - **Device**: `$device.<field>` from `Handle` (`lds_size`, `warp_size`, …).
     (Kernel namespace omitted — SDPA fwd criteria references none.)
   - **Binding architecture (the two resolvers).** Edge and scalar-attr resolution use the Phase 0
     *generated* typed accessors — name→UID→tensor via `getTensorMap()`, and `$<node>.<attr>` via the
     typed reader. The Tensor-namespace path resolver (`dims[i]`/`strides[i]`/`rank`/`stride_order`/
     `packed`/`virtual`/`uid`) is hand-written **once** over the single `TensorAttributes` shape, since
     there is exactly one tensor table. jlogic `Value` is the sole type-erasure boundary: strong typing
     lives inside each accessor and is erased exactly once, at its return.
2. **UMD compiler.** Parse `{schema,id,name,allow_override_shape,nodes,criteria}`; validate the A.10
   subset (schema exact, single key per op-object, names resolve in registry, `?` ↔ registry
   optionality, node-id/tvar disjoint from reserved roots, single-producer per variable).
   JSON→JSON lowering pass:
   - Expand layout aliases (`"nhwc"`→`[0,2,3,1]`, …, A.8).
   - Expand `shape` (A.5): record per-tensor `{name→index, rank, capture}`; rewrite
     `{"shape":["$q",[...]]}` → `{"==":["$q.rank",N]}` (no capture) / `{">=":["$q.rank",N-1]}`
     (capture). Named-dim reads then resolve through `BindingContext`.
   - `jlogic::compile<BindingContext>(loweredCriteria)`; collect referenced symbols
     (`Expression::variables()`) → published bound-symbol set; validate each resolves against
     registry + shape maps (RFC §4 build check).
   - **Static criteria type-check (A.10 §9).** Using the generated `AttrType`/tensor-field kinds from
     Phase 0, verify each operator's argument types at compile (`$q.dtype` is `Dtype`, `$q.head_size`
     is `Int`, `$q.stride_order` is `IntArray`); a type mismatch is a compile error, not a runtime
     decline. This is where the generated strong typing pays off in the compiler.
3. **`UniversalGraphMatcher`.** Root-opcode index (opcode→compiled UMDs). `match(handle, graph)`:
   - structural: locate node(s) by opcode (SDPA: the single node), build per-graph
     uid→producer/consumer index, bind names (decline on missing required tensor), honor
     `allow_override_shape` gate.
   - construct `BindingContext`, evaluate criteria `Expression` → bool.
   - return `MatchResult{ matched, bindings (BindingContext), umd_id }`. Bindings queryable
     post-match (`bindings.get("$q.head_size")`).

---

## Phase 2 — Test suite (the deliverable driver)

New gtest source added via `target_sources` in
`dnn-providers/hip-kernel-provider/src/tests/engines/asm_sdpa_engine/CMakeLists.txt` (already links
nlohmann_json + plan_utils; add dep on the generated registry).

- **Graph fixtures:** both paths — manual (`createValidSdpaFwdGraph(...)`, head_size=128, bf16) and
  JSON (`json::to<Graph>` from an inline JSON fixture) → `GraphWrapper`.
- **UMD fixture:** the §18 SDPA-forward descriptor JSON (head_size 128, bf16/fp8), **with the two
  custom-operation gates (`strides_fit_u32`, `sdpa_mask_consistent`) removed** — the custom-op hook
  is out of scope, so those gates are held constant in the graph battery, not expressed as criteria.
- **Match-equivalence (RFC §16 primary):** battery of accepting/rejecting graphs through both
  `SdpaFwdPlanBuilder::isApplicable` and the matcher; assert identical accept/reject on the
  declarative gates. Hold device=gfx942, a valid kernel key, and the two non-declarative gates
  (uint32 stride fit, mask self-consistency) constant across the battery — arch/kernel-table are
  outside the UMD per §5/§8, and the custom-op gates are out of scope; all four are held explicitly,
  not silently skipped.
- **Binding proof (RFC §4/§15, primary deliverable):** the auto-binding layer, proven end-to-end:
  - *Completeness (§4 "complete symbol table for free"):* after a successful match the published
    symbol table contains every operand/result tensor and, for each, `uid`/`rank`/`dtype`/`dims[i]`/
    `strides[i]`/`stride_order`/`packed`/`virtual`, plus every scalar attr of the matched node. Assert
    against the enumerated expected set, not a sampled few.
  - *Binding view (§15):* dump the full bound table as an inspectable artifact and assert its contents.
  - *Path resolution per form:* `$q`, `$q.uid`, `$q.rank`, `$q.dtype`, `$q.dims[i]` (positional),
    `$q.head_size` (named), `$q.strides[i]`, `$q.stride_order`, `$q.packed`, `$q.virtual`,
    `$attn_mask.present`==false, `$sdpa_fwd.dropout_probability` + `.present`, `$graph.node_count`,
    `$device.lds_size` — each resolves to the expected typed value.
  - *Generated-accessor correctness through a live graph:* name→UID→`TensorAttributes`→field returns
    the graph's actual values, tying the Phase 0 accessors to the Phase 2 resolver on real data.
  - *Bad-path fail-closed:* out-of-range `dims[i]`, unknown dim-name, absent-optional field read →
    decline, never a wrong value.
- **Fail-closed:** unknown symbol, absent-optional field access, `node_count`≠1 → decline (never
  match by default).
- **Compiler validation:** bad schema version, `?` on a required name, unknown name → compile error /
  quarantine.

---

## Scope decisions & deferrals

- **Only SDPA forward.** Registry generator is general but only SDPA is annotated, so it is the only
  exercised opcode.
- **`shape` lowered in the UMD compiler**, not JsonLogic — keeps the shared language pure; matches
  RFC §10 "compile expands short-hands."
- **No JsonLogic core change.** The custom-operation (native-predicate) hook is skipped for this
  PoC; the two SDPA gates that need it (uint32 stride fit, mask self-consistency) are held constant
  in the test battery rather than expressed as criteria. `all` / `rank`-op / `divisible` are also
  unused by SDPA fwd — deferred (add if broader §A.7 coverage is wanted).
- **Not plumbed as a provider engine** — matcher is exercised directly by tests.
- **Deferred (RFC-acknowledged):** custom-operation (native-predicate) hook and predicate registry
  (§8), static/bytecode matcher (§11), fuzzing (§14), arbitration across multiple UKDs (§12), UDD
  dispatch formulas, per-plan match cache, multi-node fusion patterns.
- **Two non-declarative gates out of the UMD.** `strides_fit_u32` and `sdpa_mask_consistent`
  (RFC §8 custom operations) are excluded with the hook; match-equivalence covers the declarative
  subset with those two gates held constant, exactly as arch/kernel-table are.

## Primary risks

- **Reflection availability/version** — flatbuffers 25.9.23 ships `reflection.h`; `graph.bfbs` must
  compile cleanly with the pinned flatc. Verify early (Phase 0 gate).
- **Parity honesty** — the UMD deliberately omits arch, kernel-table, and (with the hook skipped) the
  two custom-op gates; equivalence is over the declarative subset. Tests must fix those variables,
  not fudge them.
- **`shape` name-binding ordering** — resolved by static compile-time lowering (name maps are
  UMD-static, graph-independent), avoiding mutation during eval.

## Open questions for iteration

- Registry generator home: `flatbuffers_sdk` (schema owner, neutral output) vs. provider-local
  (more contained for a PoC). Plan assumes `flatbuffers_sdk`.
- Include `all` / `divisible` / `rank`-op now for fuller A.7 coverage, or defer?
- Neutral `ScalarValue` type location for the generated registry (shared header vs. `flatbuffers_sdk`).
- Static criteria type-checking (A.10 §9) via the generated `AttrType` in the PoC, or defer to the
  full matcher? Plan includes it as part of the binding proof, since it is the direct payoff of the
  strongly-typed codegen.
