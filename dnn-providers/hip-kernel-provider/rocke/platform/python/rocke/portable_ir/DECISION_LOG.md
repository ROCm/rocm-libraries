# JIT recipe robustness decision log

This log separates verified recipe support from design intent. The working
baseline is PR #10456 commit `c05f6bbde46bffc90259d30c712f1e48e7357c86` on
`users/mpodkory/rocke/jit-recipe-robustness`.

## Coverage model

- Ordinary operations are recorded through `IRBuilder._emit` and replayed via
  the C opcode registry. Broad coverage therefore comes from testing actual
  emitters, not adding one recipe instruction per builder method.
- Explicit coverage is still required for regions, compile-time expressions,
  types and attributes, variable-length lists, and schema validation.
- Baseline production coverage was 56 recorded emitters and 12 skips. Current
  coverage is 66 of 68, with 91 opcodes in representative recipes.
- An exploratory all-configuration pass found 545 distinct kernel-name/program
  combinations and 109 opcodes. This is audit evidence, not an always-on gate.

## 2026-08-06: reject unsafe compile-time evaluation

Decision: division/modulo by zero, signed overflow at `LONG_MIN / -1` and
`LONG_MIN % -1`, non-positive `static_for` or rolled-list steps, and loop
increment overflow are recipe errors.

Reason: these cases previously changed specialization silently or could fail to
terminate. The roller only emits increasing ranges.

Regression: the hermetic C++ replay test rejects zero division, signed
division/modulo overflow, and a negative `static_for` step.

## 2026-08-06: remove replay's hidden result ceiling

Decision: generic `emit` result arrays are dynamically sized; conflicting
`out`/`outs` forms and malformed multi-result declarations are rejected.

Reason: replay silently truncated results after sixteen although the recorder
has no such limit.

Regression: a synthetic 17-result inline-assembly operation replays intact.

## 2026-08-06: make online construction linkable

Decision: exclude legacy `cpp/core/build_id.cpp` from the globbed core sources
and retain canonical `rocke_build_id.cpp`.

Reason: `online.build_lib()` uses `--whole-archive` and exposed their duplicate
symbols, which ordinary static links hid.

Regression: build a fresh shared library, load it, and resolve
`rocke_online_recipe_cbor_to_llvm`.

## 2026-08-06: expand production recorder coverage

Decision: distinguish `_build(spec, arch=None)` adapters from index factories,
register dynamic emitter modules in `sys.modules`, and select the non-string
spec from tuple configurations.

Reason: the old harness incorrectly skipped valid elementwise, normalization,
pooling, reduction, quantization, transpose, and target-intrinsic emitters.

Regression: require at least 66 successes and exactly the two known
multi-kernel skips, `gfx950_attention_tiled_3d` and `fused_moe_e2e`.

## 2026-08-06: preserve output-affecting data

Decision: grow kernel-name formatting dynamically and reject unresolved or
malformed placeholders. Decode scalar kernel attributes and wrapped integer
lists through the typed attribute path.

Reason: the 545-configuration audit found eight parity mismatches from truncated
names and four from dropped `agpr_alloc=(0, 0)` attributes.

Regression: replay preserves a name longer than 300 bytes and lowers the
integer list to `amdgpu-agpr-alloc="0,0"`. Every exposed configuration passed
when rerun after the fixes.

## 2026-08-06: validate schema-shaped runtime input

Decision: runtime specs must match each declared name and kind exactly, with no
missing, duplicate, wrong-kind, or extra values. Register lists, rolled loop
initializers, kernel attribute objects, scalar attribute values, and formatted
register names are checked before use; malformed forms return `ROCKE_ERR_VALUE`.

Reason: these forms previously became empty lists/default scalar values, left
null names for later lookup, or allowed declarations and supplied values to
diverge.

Regression: the C++ replay test covers missing/wrong/duplicate/extra specs, a
non-array operand list, missing loop init, non-object kernel attributes,
nonnumeric float attributes, and an unterminated register placeholder.

## 2026-08-06: validate scalar predicates and instruction fields

Decision: `spec_str_eq` requires exactly two strings and a declared runtime
string spec. `const_f32.fval` must be numeric. When present, `scf_for.unroll`
and `scf_for.elide_trailing_barrier` must be booleans; their existing defaults
remain unchanged when absent. Known parameter attributes require their expected
DOM kinds, and alignment must be a positive power-of-two integer fitting `int`.

Reason: these fields previously read inactive DOM-union members or converted a
missing value to a default, allowing malformed recipes to select the wrong
compile-time arm or silently change emitted IR.

Regression: the hermetic C++ replay test covers unknown and malformed string
predicates, missing and nonnumeric float constants, invalid loop flags, and
invalid parameter-attribute kinds and alignment values. Positive true/false
string predicates, valid flags, and absent optional flags still replay.

## Current verified result

- 30 Python portable-IR unit tests pass.
- Both portable-IR C++ tests pass.
- Fresh online shared-library build/load passes.
- Recorder coverage: 66 pass, 2 explicit skips, 0 failures.
- On both `gfx942` and `gfx950`, engine and recipe replay are byte-identical for
  all 55 target-applicable kernels; 11 kernels per target are inapplicable.

## Known boundaries

- Regions: only `scf.for` and then-only `scf.if`; no general else/results form.
- Attributes: wrapped integer lists work; nested list-of-map values do not.
- Integer expressions do not range-check numeric casts or add/sub/mul overflow.
- The Python recipe expander does not yet mirror the C VM's new rejection and
  signed division/modulo semantics. The
  [active PR handoff](PR_10492_REVIEW_FIX_HANDOFF.md) tracks this correctness
  fix.
- Single-result `emit.out` still defaults a missing bind and does not reject a
  repeated concrete SSA definition. The active PR handoff tracks this fix.
- Rolling: one structural axis; no joint multi-axis interaction inference.
- Coverage: two multi-kernel adapters and a chunked 545-configuration replay
  gate remain. A single-process full sweep retained lowering output and grew
  memory substantially.
- Runtime parity is always-on for `gfx942` and `gfx950`, not every target family.
- Provider `ArtifactStore`/dispatch integration remains outside this prototype.

## Documentation lifecycle

Completed fixes and durable boundaries live in this decision log. Only active,
unresolved work belongs in the
[PR correctness handoff](PR_10492_REVIEW_FIX_HANDOFF.md). When that work is
verified, move its final decisions and regression evidence here, update the
verified result, and remove the handoff rather than retaining a historical plan.
