# Step 5+6 — the BF16x3 expansion, in both engines

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4,
steps 5 and 6). Ticket AICK-2208.

## Contents

- [Why the two steps are one change](#why-the-two-steps-are-one-change)
- [What the expansion emits](#what-the-expansion-emits)
- [Where it hooks in](#where-it-hooks-in)
- [Why the split is written as raw LLVM lines](#why-the-split-is-written-as-raw-llvm-lines)
- [Rounding: RNE, not truncation](#rounding-rne-not-truncation)
- [Synthetic ops in the C engine](#synthetic-ops-in-the-c-engine)
- [One asymmetry left standing](#one-asymmetry-left-standing)
- [Verification](#verification)
- [Follow-ups left open](#follow-ups-left-open)

## Why the two steps are one change

Plan §4 lists the Python expansion (step 5) and the C++ mirror (step 6)
separately, but the repo's #1 invariant is that an emission change lands in both
engines *in the same change*. Splitting them would put a commit on the branch
whose two engines disagree, which is precisely the state the invariant exists to
forbid. They are one commit.

## What the expansion emits

For each fp32 operand fragment `x`, a Dekker-style split into two bf16
fragments:

```llvm
%big     = fptrunc <N x float>  %x       to <N x bfloat>   ; x, rounded
%big_f32 = fpext   <N x bfloat> %big     to <N x float>    ; ... widened back
%diff    = fsub    <N x float>  %x, %big_f32               ; ... what it lost
%small   = fptrunc <N x float>  %diff    to <N x bfloat>   ; ... rounded too
```

then three MMAs on the catalog's `physical_op_id`, threading one fp32
accumulator:

| # | term | accumulator in | result |
|---|---|---|---|
| 1 | `a_small * b_big` | the caller's `C` | temp |
| 2 | `a_big * b_small` | temp | temp |
| 3 | `a_big * b_big` | temp | the caller's `D` |

`a_small * b_small` is not computed: its magnitude is the same order as the
split's own representation error, so forming it would cost a fourth MMA and buy
nothing.

**The term order is load-bearing.** fp32 accumulation is not associative, so
issuing the same three products in any other order gives different result bits.
It is pinned by test, not left to the reader.

Only the third term writes the SSA value the caller named, so everything
downstream of the `tile.mma` is untouched by the expansion.

## Where it hooks in

At the top of the `tile.mma` lowering, *before* ISA dispatch — Python
[`_op_tile_mma`](../../../python/rocke/core/lower_llvm.py), C
[`_op_tile_mma`](../../../cpp/core/lower_llvm/mma.cpp). Both first ask
`_emulated_mma_atom` whether the `op_id` names an emulated atom **on the arch
being lowered**; a native atom falls straight through and its emission is
bit-for-bit what it was before this change.

The question is asked per-arch rather than against a bare `op_id` table on
purpose: the same logical atom may be native on one target and emulated on
another, which is exactly the situation a native gfx942 XF32 atom (AICK-1714)
would create.

## Why the split is written as raw LLVM lines

No existing arith op can express a *vector* fp32→bf16 narrowing —
`arith.trunc_f32_to_bf16` and `arith.cast_f32_to` both hardcode the scalar
`float` type. Rather than widen an op that every other kernel family already
depends on, the split emits its four lines directly at the lowering site. That
keeps the blast radius at zero for existing emission and makes the C mirror a
transcription rather than a port.

## Rounding: RNE, not truncation

Both narrowing steps are `fptrunc ... to bfloat`, which LLVM defines as
round-to-nearest-even — it has no other fp32→bf16 rounding — and which the
AMDGPU backend selects the packed `v_cvt_pk_bf16_f32` for in the vector form on
gfx950. That is CK's *shipping packed* path.

CK's scalar fallback truncates instead. It is deliberately not the parity
target: an RNE split is strictly the more accurate of the two, and the emulation
as a whole is already more accurate than real TF32.

## Synthetic ops in the C engine

The three sub-MMAs are stack-allocated `rocke_op_t` / `rocke_value_t` rather
than arena-built IR. That is safe because the MFMA emission path reads only
`rocke_ll_operand(L, v)` and the result name, and `rocke_ll_operand` returns
`v->name` whenever `v->op == NULL` — it never dereferences the producing op. The
synthetic bf16 values still carry a properly-formed stack `rocke_type_t`
(`ROCKE_TYPE_VECTOR` over `rocke_scalar_by_name("bf16")`) so that anything which
*does* look at the type in future sees the truth rather than a null.

Fresh names come from `rocke_ll_fresh` in emission order on both sides, which is
what makes the two engines agree on the temp numbering and therefore on bytes.

## One asymmetry left standing

Python dispatches each sub-MMA through `backend.emit_mma`, which serves MFMA and
WMMA alike. C dispatches through the dense MFMA table. Every emulated row the
catalog ships is MFMA, so the engines cannot disagree on emitted bytes today; a
future WMMA-arch emulated row would hit the C engine's explicit failure rather
than emit something wrong. The comment at the dispatch site says so, rather than
claiming a parity that is not yet there.

## Verification

- [`tests/core/test_xf32_bf16x3_lowering.py`](../../../tests/core/test_xf32_bf16x3_lowering.py)
  — 7 tests. The four-line split per operand, read back and re-derived; RNE and
  not truncation; three terms in the pinned order threading one accumulator;
  `small*small` absent; no emulated `op_id` surviving into the emitted IR; only
  the last term writing the caller's value; and a native bf16 atom still
  lowering to exactly one call with no split.
- [`tests/core/test_xf32_bf16x3_lowering.cpp`](../../../tests/core/test_xf32_bf16x3_lowering.cpp)
  — the same properties pinned directly on the C engine, by parsing serialized
  IR and inspecting the emitted text. This exists because the Python file's
  engine-parity test needs the `rocke_engine` pybind module and *skips* where it
  is absent — which is every environment that builds only the archive. Wired
  into ctest as `rocke_xf32_bf16x3_lowering`.
- Python↔C byte identity for both atoms at llvm20, llvm22 **and** llvm23:
  `IDENTICAL` on all six, checked by lowering the serialized IR through
  `ir_lower_cli` and diffing against `_lower_kernel_to_llvm_python`. This is the
  check that actually proves the mirror; the ctest above proves the recipe.
- `tests/core` — 398 passed, 23 skipped, 329 subtests.
- C build clean; `ctest` 13/13.
- Byte-identity gate at llvm20 **and** llvm22: 69 GREEN / 1 DRIFT, the DRIFT
  being the `gfx942_attention_tiled_3d` cfg[3] divergence proven pre-existing in
  [step 1](step-01-mma-result-dtype.md). No new drift — expected, since no
  golden kernel selects xf32 yet.

## Follow-ups left open

- **Closed after review.** The bare-`op_id` fragment lookups did not know the
  emulated atoms, so neither the C builder's `rocke_b_mma` nor the bare-string
  form of Python's `IRBuilder.mma` could construct an xf32 `tile.mma`. Both now
  resolve the logical op_id through its physical atom — see
  [step 4](step-04-gfx950-xf32-rows.md#follow-ups-left-open). The C test still
  embeds serialized IR, now so that it pins the recipe independent of the
  builder.
- No golden kernel exercises the expansion yet, so the byte-identity gate covers
  it only through the two focused tests above. That changes when a kernel family
  selects xf32.
