# rocKE four-role MMA atom spike

## Status and scope

- Worktree: `/home/AMD/mpodkory/agents-scratch/rocm-libraries-mma-four-operands-spike`
- Branch: `users/mpodkory/rocke/mma-four-operands-spike`
- Base: `origin/develop` at `419dac888d3a19b22ec0f521630ab94da486493a`
- Implemented as a clean source/schema break in this worktree. The architecture
  SSOT, Python and C models, atom helpers, direct consumers, tests, and
  documentation now use explicit C-input and D-result metadata.
- Current catalog rows intentionally give C and D identical values, while tests
  prove the model can represent different C/D dtypes, fragment lengths, and
  layouts.

The proposed model is the conventional matrix contract:

```text
D = A * B + C
```

Here, A, B, and C are source operands and D is the produced result. This gives
an atom four logical roles, but it does **not** give `tile.mma` four SSA input
operands. The IR operation remains:

```text
d = mma(op, a, b, c)
```

Adding D as a fourth SSA input would duplicate the result, disagree with the
LLVM intrinsic contract, and unnecessarily change serialization and every
lowerer.

## Naming evidence

AMD's machine-readable ISA schema represents each instruction operand with an
encoding `FieldName` and independent `Input` and `Output` flags. It therefore
describes encoded sources/destinations, not the higher-level matrix role model.
MFMA assembly/encoding terminology is `vdst, srcA, srcB, srcC, ...`.

LLVM is explicit about the semantic names:

- `AMDGPUWmmaIntrinsic` documents a return `%D` and arguments `%A`, `%B`, `%C`.
- `AMDGPUMfmaIntrinsic` documents `vdst, srcA, srcB, srcC, cbsz, abid, blgp`.
- Newer WMMA definitions include `AMDGPUWmmaIntrinsicModsCDiff`, specifically
  for cases where D and C have different types.

Evidence snapshots checked on 2026-09-01:

- LLVM `llvm/include/llvm/IR/IntrinsicsAMDGPU.td`, commit
  `e2a87e12d2561865ed36b264bf48b7423d450041`.
- GPUOpen `isa_spec_manager` schema documentation, commit
  `452645535ac05f466b06a13e5eafeb5a86d3ad11`. The actual architecture XML
  files are distributed separately from GPUOpen's machine-readable ISA page.

## Implemented rocKE contract

### Architecture/catalog model

`MmaOp` models all four roles independently:

```text
a_dtype, b_dtype, c_dtype, d_dtype
a_frag_len, b_frag_len, c_frag_len, d_frag_len
a_layout(), b_layout(), c_layout(), d_layout()
```

- C means the accumulator source fragment.
- D means the result fragment.
- `acc_layout()` remains as a convenience spelling for the produced D layout;
  code that needs a specific role uses `c_layout()` or `d_layout()`.
- C and D may initially contain identical data for all registered atoms, but
  they should be separate fields and layout objects from day one. LLVM already
  has WMMA forms where their types differ, so aliasing them in the data model
  would preserve the defect this change is intended to remove.

The existing names are interpreted as output metadata today. Therefore the
migration is:

```text
old c_dtype     -> new d_dtype
old c_frag_len  -> new d_frag_len
old _c_layout   -> new _d_layout
old c_per_lane  -> new d_per_lane
```

New C fields are then populated from the actual accumulator input contract. For
the current catalog, they can initially be seeded from D and verified against
the LLVM/builtin signatures.

### IR contract

Keep the operation shape stable:

```python
def mma(op, a, b, c, *extra) -> Value:  # returned Value is d
```

Result construction must use `d_dtype` and `d_frag_len`; operand validation
must use C metadata. Lowerers continue to pass `(A, B, C)` to LLVM/HIP and bind
the call result as D.

Scaled atom operands remain `*extra`; they are modifiers/scales, not additional
matrix roles.

### Compatibility policy

Use a clean schema/API rename for the spike: no deprecated getters, duplicate
struct members, JSON fallback aliases, or forwarding shims. This is a
source/schema break and requires Python/C consumers to move together. It does
not change the serialized `tile.mma(%a, %b, %c)` operation shape.

## Change inventory

### 1. Architecture SSOT and Python model

Required files:

- `platform/python/rocke/core/arch/data/arch_specs.json`
  - bump `version` from 1 to 2;
  - change each MMA row from `{a,b,c,...}` to `{a,b,c,d,...}`;
  - migrate all 47 rows across 7 targets;
  - today the old output values are only `fp32` and `i32`.
- `platform/python/rocke/core/arch/target.py`
  - extend `MmaOp` and `_FragInfo` to four roles;
  - split C and D layout maps and fragment lengths;
  - replace `_op_id_c_dtype()` with `_op_id_d_dtype()` for result typing;
  - make catalog queries accept both `c_dtype` and `d_dtype`;
  - update support predicates and error messages.

Design note: if fragment C and D are identical for an instruction, they may
reuse the same mapping function implementation, but must still be constructed
as role-labelled `LayoutMap` objects.

### 2. C mirror and public architecture API

Required files:

- `platform/cpp/include/rocke/arch_target.h`
- `platform/cpp/core/arch/data.cpp`
- `platform/cpp/core/arch/helpers.cpp`
- `platform/cpp/core/arch/query.cpp`

Changes:

- extend `rocke_mma_op_t` with distinct C and D dtype, fragment, and layout
  fields;
- rename result helpers such as `rocke_arch_mma_c_frag_len` and
  `rocke_arch_mma_op_id_c_dtype` to D;
- add genuine C helpers where consumers need accumulator-input metadata;
- update the manually embedded catalog and layout tables in exact parity with
  the JSON SSOT.

This is a public struct/API break. Under the clean-spike policy, change all
in-tree callers in the same patch rather than retaining ABI padding or aliases.

### 3. Neutral IR, verifier, and serialization

Required files:

- `platform/python/rocke/core/ir.py`
- `platform/python/rocke/core/verify.py`
- `platform/cpp/include/rocke/ir.h`
- `platform/cpp/core/ir/ir_tile.cpp`

Changes:

- keep `mma(op, a, b, c) -> d` and `rocke_b_mma(..., a, b, c, ...)`;
- derive the result type and length from D metadata;
- validate the third source against C metadata;
- rename result-local variables/comments from C or accumulator-result wording
  to D where they describe the produced value;
- retain the three-source serialized form and its `op_id` attribute.

No serialization version bump is needed if the IR operand/result structure is
unchanged. Golden text may still change where variable names or documentation
use `%c` for a result.

### 4. LLVM and HIP lowering

Primary files:

- `platform/python/rocke/core/isa/backend.py`
- `platform/python/rocke/core/lower_llvm.py`
- `platform/python/rocke/core/lower_hip.py`
- `platform/cpp/core/lower_llvm/mma.cpp`
- `platform/cpp/core/lower_hip/lower_hip_mma.cpp`
- `platform/cpp/core/lower_cktile.cpp`
- `platform/python/rocke/core/lower_cktile.py`

Changes:

- continue unpacking three matrix sources as `a, b, c`;
- name the operation result/local destination `d`;
- split table fields currently called `acc_ty` or `acc_vec` when they silently
  assume C and D are identical;
- use C types for the third intrinsic/builtin argument and D types for the
  return declaration/result;
- preserve all LLVM call signatures and generated HIP bytes for the current
  atom catalog.

The many fixed `op.operands[0..2]` assumptions remain correct. They should not
be mechanically changed to expect operand index 3.

### 5. Legacy atom helpers

Primary files:

- `platform/python/rocke/helpers/atoms.py`
- `platform/cpp/include/rocke/helper_rocke.helpers.atoms.h`
- `platform/cpp/helpers/atoms.cpp`

Changes:

- rename output metadata `c_per_lane` to `d_per_lane`;
- add real `c_per_lane` for the accumulator source;
- split `dtype_out` into explicit C/D role metadata where necessary;
- keep `emit(builder, a, b, c) -> d`;
- rename output-only helpers and prose, for example:
  - `lane_to_output` may remain because it is role-neutral;
  - `make_c_warp_dstr_encoding` -> `make_d_warp_dstr_encoding`;
  - output-oriented `c_warp_params` -> `d_warp_params`;
  - `zero_acc` should be reviewed semantically, not blindly renamed: it creates
    the initial C value, while the value returned by an MMA is D.

`c_per_lane` currently appears in at least 23 Python helper/instance files and
33 C helper/instance/header files. These are mechanical consumers after the
semantic definitions are fixed, but should be changed in bounded batches.

### 6. Direct kernel consumers and generated C ports

Consumer families requiring review include:

- common GEMM and epilogues;
- implicit/direct convolution and deep fused convolution;
- attention and split-K/decode paths;
- MoE and block-scaled/MX GEMM;
- gfx1151, gfx1201, and gfx1250 WMMA kernels;
- Python builders and their C ports using `b.mma(...)` / `rocke_b_mma(...)`.

Only atom-level C/D names should change. Do not globally rename GEMM API
objects: GEMM C/output and auxiliary D-tensor naming have separate established
semantics.

### 7. Tests

Primary focused tests:

- `platform/tests/core/test_arch_mma_ssot.py`
- `platform/tests/core/test_mma_frag_tables.py`
- `platform/tests/core/mma_frag_ssot.cpp`
- `platform/tests/core/test_ir_serialize.py`
- `platform/tests/core/test_wmma_gfx12_acc_layout.py`
- relevant MFMA/WMMA sections in `platform/tests/test_rocke.py`

Add assertions that:

- every catalog row has A/B/C/D metadata;
- C and D maps are role-labelled independently even when coordinates match;
- result types come from D, not C;
- a synthetic C/D-different atom is representable and validated correctly;
- Python and C catalogs/layout maps remain identical;
- serialized `tile.mma` retains three matrix source operands and one result.

### 8. Documentation

Primary pages:

- `platform/dsl_docs/architecture/multi_arch_data_layout.md`
- `platform/dsl_docs/architecture/authoring_model.md`
- `platform/dsl_docs/architecture/ir_serialization_format.md`
- `platform/dsl_docs/reference/mfma_atom_catalog.md`
- `platform/dsl_docs/reference/glossary.md`
- `platform/dsl_docs/primitives/intrinsics_and_primitives.md`
- module/class documentation in `platform/python/rocke/helpers/atoms.py`

Document both views side by side:

```text
matrix semantics: A, B, C -> D
ISA encoding:     srcA, srcB, srcC -> vdst
LLVM intrinsic:   (A, B, C, modifiers...) -> D
rocKE IR:         tile.mma(a, b, c, extras...) -> d
```

## Implementation sequence

1. Add failing SSOT/model tests for four independent roles, including one
   synthetic C/D-different case.
2. Migrate JSON schema v2 and the Python `MmaOp`/fragment/layout model.
3. Migrate the C catalog mirror and public architecture API; pass catalog and
   layout parity tests.
4. Switch IR result construction to D while preserving `(a,b,c) -> d`.
5. Update LLVM, HIP, and CK-Tile lowering tables so C argument and D result
   types are distinct in the model; prove current emitted bytes are unchanged.
6. Migrate `MfmaAtom`/`WmmaAtom` and output-layout helper names.
7. Update direct consumers in family-sized batches, regenerating or updating C
   ports alongside their Python source.
8. Update documentation and run the full parity matrix.

## Validation plan

The change is intended to be semantic-model-only for the current catalog. The
main regression invariant is therefore:

```text
same registered atom + same kernel inputs
    => same serialized operation shape
    => byte-identical LLVM/HIP output
    => same compiled code and numeric result
```

Validation is staged so a metadata defect fails before expensive compilation or
GPU execution.

### Validation status (2026-09-01)

- Schema inventory: version 2, 7 architectures, 47/47 MMA rows with explicit
  A/B/C/D fields.
- Focused model and atom tests: 72 passed, 1 skipped, 71 subtests passed.
- rocKE core suite: 278 passed, 58 skipped, 90 subtests passed.
- Full platform pytest: 778 passed, 112 skipped; the only four failures are the
  pre-existing debug-info object tests invoking host LLVM 18 `llc`, which does
  not recognize `gfx950` and aborts during instruction selection.
- C/C++ build with `ROCKE_BUILD_PYBIND=ON`: passed. The existing unrelated
  duplicate `rocke_conv_problem` ODR warning remains.
- CTest: 5/5 passed, including the C four-role metadata assertions and IR
  serialization round trip.
- Representative LLVM golden inventory: passed for LLVM 20, 22, and 23 without
  re-blessing hashes.
- Python/C LLVM differential gate: all 69 registered families green.
- `git diff --check`: passed.

### Gate 0: freeze the baseline

Before source edits, record:

- the base commit and dirty-state check;
- the normalized 47-row architecture catalog;
- representative serialized `tile.mma` operations;
- LLVM and HIP output hashes from the registered parity families for each
  supported LLVM flavor.

Keep baseline artifacts outside the source tree. A comparison must use the same
Python, compiler, rocKE configuration, target, and test input on both sides.

Pass condition: the baseline is reproducible in two consecutive runs. This
prevents an existing nondeterministic emitter from being mistaken for a C/D
rename regression.

### Gate 1: schema and model invariants

Focused Python tests:

```bash
cd dnn-providers/hip-kernel-provider/rocke/platform
PYTHONPATH=python python -m pytest \
  tests/core/test_arch_mma_ssot.py \
  tests/core/test_mma_frag_tables.py \
  tests/core/test_wmma_gfx12_acc_layout.py -v
```

Required assertions:

- schema version 2 rejects MMA rows missing any of `a`, `b`, `c`, or `d`;
- every one of the 47 rows loads with normalized A/B/C/D dtypes;
- atom lookup filters C and D independently;
- C and D fragment lengths and layouts carry the correct role label;
- current atoms have the expected C/D equality where the hardware contract is
  tied;
- a synthetic C/D-different atom proves the model does not infer D from C;
- `_op_id_d_dtype()` drives result typing and never consults C implicitly;
- duplicate/conflicting `op_id` rows fail deterministically rather than using
  first-match behavior.

Pass condition: Python model tests are green and no compatibility fallback for
old schema-v1 MMA rows remains.

### Gate 2: Python/C SSOT parity

Build the C engine and run the focused host test:

```bash
cmake -S dnn-providers/hip-kernel-provider/rocke/platform \
      -B /tmp/rocke-mma-four-role-build \
      -DROCKE_BUILD_PYBIND=ON
cmake --build /tmp/rocke-mma-four-role-build --parallel
ctest --test-dir /tmp/rocke-mma-four-role-build \
      --output-on-failure -R 'rocke_mma_frag_ssot|ir_serialize_roundtrip'
```

Extend `mma_frag_ssot.cpp` so Python and C are compared for:

- row count, target, family, shape, and `op_id`;
- A/B/C/D dtype and fragment length;
- A/B/C/D lane/slot coordinate maps;
- missing-layout errors and role names;
- the synthetic C/D-different representation.

Pass condition: the C mirror matches the JSON/Python SSOT field-for-field and
coordinate-for-coordinate. A matching row count alone is insufficient.

### Gate 3: IR contract and serialization

Focused tests:

```bash
cd dnn-providers/hip-kernel-provider/rocke/platform
PYTHONPATH=python python -m pytest \
  tests/core/test_ir_serialize.py \
  tests/core/test_arch_mma_ssot.py -v
```

Required assertions:

- `tile.mma` has three matrix sources `(a, b, c)` and exactly one result D;
- scaled forms retain their existing extra modifier/scale operands;
- verifier diagnostics distinguish C source type/length failures from D result
  type/length failures;
- result type and vector width are selected from D metadata;
- serialize -> parse -> serialize is byte-identical;
- parse -> lower is byte-identical to direct lowering;
- old schema-v1 catalog data fails with a clear version/schema diagnostic.

Pass condition: no serialized IR format/version change is required and all
existing serialized kernels remain readable. If a fourth SSA input appears,
the design has drifted and this gate must fail.

### Gate 4: lowering and cross-engine byte identity

Run the repository's registered-family gate for each supported flavor:

```bash
cd dnn-providers/hip-kernel-provider/rocke/platform
ROCKE_LLVM_FLAVOR=llvm20 python tools/check_byte_identity.py
ROCKE_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py
ROCKE_LLVM_FLAVOR=llvm23 python tools/check_byte_identity.py
```

The gate must cover both LLVM IR and HIP-source emitters where registered.
Add direct unit coverage for the type split:

- the third intrinsic/builtin matrix argument is typed from C;
- the call return and rocKE result are typed from D;
- MFMA still lowers as `srcA, srcB, srcC -> vdst`;
- WMMA still lowers as `(A, B, C, modifiers...) -> D`;
- current-catalog emitted bytes match the frozen baseline exactly.

Pass condition: all registered Python/C emitters are byte-identical to each
other, and before/after output hashes are identical. Updating goldens merely to
accept renamed metadata is not permitted unless the textual difference is an
intentional user-visible diagnostic rather than generated kernel code.

### Gate 5: full host suite

Run the canonical entrypoint against the built C engine:

```bash
cd dnn-providers/hip-kernel-provider/rocke/platform
PYTHONPATH="/tmp/rocke-mma-four-role-build:$PWD/python" \
  python tests/run_all.py --build-root /tmp/rocke-mma-four-role-build
```

This covers the relative-path guard, byte-identity gate, pytest, the
`ROCKE_BACKEND=both` differential pass, and registered CTest tests.

Pass condition: the full runner is green with no new skips or reduced parity
family count.

### Gate 6: target admission and complete-kernel compilation

Compile at least one real kernel using each relevant matrix-engine family:

| Target | Required coverage |
| --- | --- |
| gfx942 | wave64 MFMA, fp16/bf16 and one fp8/bf8 atom |
| gfx950 | wave64 MFMA, K-packed atom and one scaled/low-bit path |
| gfx1151 | wave32 WMMA, fp16/bf16 and integer atom |
| gfx1201 | wave32 WMMA with the gfx12 fragment layout |
| gfx1250 | current WMMA form, including a modifier-bearing form if registered |

For each target, preserve:

- selected `op_id` and A/B/C/D catalog row;
- emitted intrinsic/builtin signature;
- LLVM assembly/verification result;
- COMGR or compiler result and code-object hash;
- whether the result is compilation-only or ran on matching hardware.

Pass condition: target selection is unchanged, all representative kernels
compile, and code-object differences are investigated rather than assumed to
be harmless. Compiler-version or nondeterministic metadata differences must be
separated from instruction/code differences.

### Gate 7: GPU numeric smoke tests

Run only on matching available hardware. Use a small deterministic GEMM or
existing correctness kernel per available target, with nonzero C so the test
actually exercises accumulation rather than only `A * B`.

For each run, compare:

- Python versus C engine output;
- pre-change versus post-change result;
- device result versus a host reference;
- selected atom and generated code hash.

Include at least one multi-step accumulation where the D from iteration N is
fed back as C for iteration N+1. That catches accidental swapping of input C
metadata and output D metadata.

Pass condition: existing test tolerances are met with no tolerance widening.
Unavailable devices are reported as an unrun runtime lane, never as a pass.

### Final acceptance record

The implementation is ready only when the handoff records:

- exact source commit and toolchain versions;
- all commands and exit codes;
- catalog/parity counts;
- before/after IR, HIP, and code-object hash verdicts;
- GPU model and runtime results, where available;
- an explicit list of skipped/unavailable lanes;
- `git diff --check` and a final changed-file inventory confirming that
  unrelated GEMM-level C/D naming was not modified.

## Non-goals

- Adding D as a fourth SSA input operand.
- Renaming unrelated GEMM-level C/D tensors.
- Changing intrinsic selection, instruction encodings, or generated code.
- Adding new MFMA/WMMA instructions or datatypes.
- Preserving the old public source/ABI/schema names during this spike.
