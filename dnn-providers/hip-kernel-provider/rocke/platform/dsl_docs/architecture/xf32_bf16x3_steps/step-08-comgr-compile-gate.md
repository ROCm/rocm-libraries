# Step 8 — the expansion survives codegen for gfx950

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4,
step 8).

## Contents

- [Why a text check is not enough](#why-a-text-check-is-not-enough)
- [What the gate compiles](#what-the-gate-compiles)
- [What the disassembly is read for](#what-the-disassembly-is-read-for)
- [Observed ISA](#observed-isa)
- [Where it skips](#where-it-skips)
- [Verification](#verification)

## Why a text check is not enough

[Steps 5+6](step-05-bf16x3-lowering.md) pin the emitted LLVM IR, which proves
the recipe is *written* correctly. It does not prove the recipe is
*codegen-able*. Two of its ingredients are things a text check cannot vouch for:

- a **vector** `fptrunc <N x float> ... to <N x bfloat>`, which needs a legal
  packed narrowing on the target;
- three MFMA intrinsics that the backend must actually carry for gfx950.

Either could pass every assertion in the lowering tests and then fail — or
SIGABRT — inside comgr. So this step compiles the IR the whole way to a
loadable code object.

## What the gate compiles

[`tests/core/test_xf32_comgr_compile.py`](../../../tests/core/test_xf32_comgr_compile.py)
builds, for each of the two emulated atoms, a kernel that loads both operand
fragments from global memory, issues the emulated `tile.mma`, and stores every
accumulator lane back. The loads and stores are not decoration: a kernel whose
MMA result is dead compiles to nothing at `-O3`, and would gate on nothing.

Storing is element-wise rather than one wide store because the 32x32
accumulator is 16 lanes, past what `global_store_vN` accepts for f32.

The LLVM flavor is deliberately **not** pinned. `_lower_kernel_to_llvm_python`
resolves it from the comgr that will actually compile the IR, and that pairing
is the only safe one — a mismatched flavor is exactly what
`_assert_ir_flavor_matches_lib` exists to refuse.

## What the disassembly is read for

Two facts the design rests on that are invisible in the IR:

1. **Exactly three `v_mfma_*_bf16`** instructions reach the ISA, and no other
   MFMA does. The backend neither folded two terms together nor duplicated one.
2. **The narrowing selects the packed `v_cvt_pk_bf16_f32`**, which rounds to
   nearest-even. This is the claim
   [steps 5+6](step-05-bf16x3-lowering.md#rounding-rne-not-truncation) makes
   about matching CK's *shipping packed* path; had the backend fallen back to a
   scalar truncating sequence instead, the emulation would silently disagree
   with CK on the low bits of every input, and nothing upstream of the ISA would
   have shown it.

The disassembly is read with the repo's own `rocke.analysis.isa` line reader
rather than a second regex for the same format.

## Observed ISA

Both atoms, gfx950, `-O3`, `llvm23` IR:

| atom | MFMA | packed convert |
|---|---|---|
| `mfma_f32_16x16x32_xf32` | 3 × `v_mfma_f32_16x16x32_bf16` | `v_cvt_pk_bf16_f32` |
| `mfma_f32_32x32x16_xf32` | 3 × `v_mfma_f32_32x32x16_bf16` | `v_cvt_pk_bf16_f32` |

No scalar `v_cvt_f32_bf16` truncating sequence appears in either. The
subtraction lowers through `v_pk_add_f32`, so the split costs packed VALU work
rather than a per-lane chain.

The cap that holds the leading bf16 at its largest finite value
([step 5](step-05-bf16x3-lowering.md#saturation-the-top-of-the-finite-range)) adds
integer VALU work on the narrowed value — on gfx950 the backend selects a packed
unsigned minimum (`v_pk_min_u16`) for its compare-and-select — and changes neither the
MFMA count nor the packed convert, which are what this gate asserts.

The instruction *count* of the convert is not asserted — only its presence. How
many packed converts the scheduler emits for a given fragment width is a
backend choice with no bearing on the numerics; which opcode it picks is not.

## Where it skips

No GPU is needed — comgr compiles for a named ISA, it does not run anything —
so this gate is a plain host test. Whether to skip is decided *before*
compiling, and only on the environment:

- comgr cannot be imported or loaded, or
- the loaded comgr does not list `amdgcn-amd-amdhsa--gfx950` among its targets
  (`rocke.runtime.comgr.supported_isas`, over `amd_comgr_get_isa_name`).

Past that check a `ComgrError` is the IR failing to build, and it **fails** the
test. An earlier version caught `ComgrError` from the compile itself and
skipped, which hid exactly the codegen regression this gate exists for: comgr
reports an unknown target and rejected IR with the same exception type, so the
target has to be asked about separately.

`llvm-objdump` is needed only by the two disassembly checks, which skip without
it; the compile check still runs, since it needs nothing but the code object.
ROCm installs `llvm-objdump` outside the default `PATH`, so
`$ROCM_PATH/llvm/bin` is checked as a fallback before giving up.

## Verification

- `tests/core/test_xf32_comgr_compile.py` is green against the dev image's comgr. Its
  kernel wraps the loaded fp32 fragments as TF32 operands, as for a native atom
  ([step 11](step-11-tf32-operand-abi.md)).
- After review, each skip/fail path was driven by injecting the fault: IR that
  comgr rejects → 6 errors, no skips; comgr not listing gfx950, or not
  loading → all skip; no `llvm-objdump` → only the 4 disassembly subtests skip
  and the compile check passes.
- No emission changed in this step, so the byte-identity result from
  [steps 5+6](step-05-bf16x3-lowering.md) still stands.
