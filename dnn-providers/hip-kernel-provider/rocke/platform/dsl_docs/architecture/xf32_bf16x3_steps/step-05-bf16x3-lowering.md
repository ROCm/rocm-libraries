# Step 5+6 — the BF16x3 expansion, in both engines

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4,
steps 5 and 6).

## Contents

- [Why the two steps are one change](#why-the-two-steps-are-one-change)
- [What the expansion emits](#what-the-expansion-emits)
- [Where it hooks in](#where-it-hooks-in)
- [Why the split is written as raw LLVM lines](#why-the-split-is-written-as-raw-llvm-lines)
- [Rounding: RNE, not truncation](#rounding-rne-not-truncation)
- [Saturation: the top of the finite range](#saturation-the-top-of-the-finite-range)
- [Synthetic ops in the C engine](#synthetic-ops-in-the-c-engine)
- [One asymmetry left standing](#one-asymmetry-left-standing)
- [Verification](#verification)
- [Follow-ups, closed after review](#follow-ups-closed-after-review)

## Why the two steps are one change

Plan §4 lists the Python expansion (step 5) and the C++ mirror (step 6)
separately, but the repo's #1 invariant is that an emission change lands in both
engines *in the same change*. Splitting them would put a commit on the branch
whose two engines disagree, which is precisely the state the invariant exists to
forbid. They are one commit.

## What the expansion emits

Each operand is a logical TF32 fragment, an I32 carrier. The expansion first views
its payload as fp32 — a `bitcast`, never a rounding, so every one of the 32 bits
reaches the split — and then takes a Dekker-style split into two bf16 fragments:

```llvm
%x       = bitcast <N x i32>    %tf      to <N x float>   ; payload, reinterpreted
%rne     = fptrunc <N x float>  %x       to <N x bfloat>   ; x, rounded to nearest even
%bits    = bitcast <N x bfloat> %rne     to <N x i16>      ; ... as bit patterns
%sign    = and     <N x i16>    %bits, 0x8000              ; ... the sign bit
%cap     = or      <N x i16>    %sign, 0x7f7f              ; ... largest finite bf16 of that sign
%over    = icmp ugt <N x i16>   %bits, %cap                ; ... is the rounding past it?
%sat     = select  <N x i1> %over, <N x i16> %cap, <N x i16> %bits ; ... held at it
%big     = bitcast <N x i16>    %sat     to <N x bfloat>
%big_f32 = fpext   <N x bfloat> %big     to <N x float>    ; ... widened back
%diff    = fsub    <N x float>  %x, %big_f32               ; ... what the PAYLOAD lost
%small   = fptrunc <N x float>  %diff    to <N x bfloat>   ; ... rounded too
```

(The constants are spelled as explicit `<N x i16>` lane lists, `-32768` and `32639`.
Everything between `%rne` and `%big` is the cap, explained
[below](#saturation-the-top-of-the-finite-range); on every payload it does not fire for,
`%big` is `%rne`.)

then three MMAs on the catalog's `physical_op_id`, threading one fp32
accumulator:

| # | term | accumulator in | result |
|---|---|---|---|
| 1 | `a_small * b_big` | the caller's `C` | temp |
| 2 | `a_big * b_small` | temp | temp |
| 3 | `a_big * b_big` | temp | the caller's `D` |

`a_small * b_small` is not computed. It is below `2^-16` of the product, which is the
same order as the representation error the split already carries (each `small` is
itself rounded to 8 significant bits), so forming it would cost a fourth MMA to
remove an error of that size from only one of its sources. The bound is reached, not
just approached, by an operand exactly halfway between two bf16 values
([step 11](step-11-tf32-operand-abi.md#no-payload-bit-is-dropped)).

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
another, which is exactly the situation the base's native gfx942 XF32 atoms
create: the same logical TF32 atom is native on gfx942 and emulated on gfx950.

## Why the split is written as raw LLVM lines

No existing arith op can express a *vector* fp32→bf16 narrowing —
`arith.trunc_f32_to_bf16` and `arith.cast_f32_to` both hardcode the scalar
`float` type. Rather than widen an op that every other kernel family already
depends on, the split emits its lines directly at the lowering site. That
keeps the blast radius at zero for existing emission and makes the C mirror a
transcription rather than a port.

## Rounding: RNE, not truncation

Both narrowing steps are `fptrunc ... to bfloat`, which LLVM defines as
round-to-nearest-even — it has no other fp32→bf16 rounding — and which the
AMDGPU backend selects the packed `v_cvt_pk_bf16_f32` for in the vector form on
gfx950. That is CK's *shipping packed* path.

CK's scalar fallback truncates instead. It is deliberately not the parity
target: an RNE split is the more accurate of the two, and on raw fp32 operands
the emulation as a whole is more accurate than real TF32 (on operands already
rounded to TF32 precision the comparison reverses; see
[step 11](step-11-tf32-operand-abi.md#no-payload-bit-is-dropped)).

## Saturation: the top of the finite range

Round-to-nearest-even overflows to infinity for every fp32 payload of magnitude
`0x7f7f8000` or more: the top `2^-9` of the finite range, which holds TF32's own
largest finite value, `0x7f7fe000`. Left alone, `big` rounds to `+inf`, `small`
becomes the opposite infinity, and the `inf * 0` that results in the cross terms
turns a *finite* product into NaN. The boundary example is `0x7f7fe000 *
0x00800000`, whose exact value is `3.998046875`. Uncapped, the split gives its first
operand `big = +inf` and `small = -inf` (LLVM's own constant folder confirms it on the
emitted IR) and the result is NaN on the host model and on a gfx950.

So `big` is held at the largest finite bf16, `0x7f7f`, keeping its sign. The test is
on the rounded value's *bit pattern*, because bf16 patterns are ordered like
their magnitudes: an **unsigned** compare against `0x7f7f` with the value's own sign bit
attached (unsigned, since a negative pattern has the sign bit set and would
otherwise sort below every positive one; with the sign attached, so that both
infinities are caught). That catches `±inf` and every NaN — all `0x7f80` and up in
magnitude — and no finite value, so the cap is the identity on every payload that
never needed it.

`small` is still taken from the *payload*, not from the capped copy. For a payload the
cap clipped, `small` then holds what was clipped — finite, and within bf16's own
precision of it — and the three terms sum to the same product as anywhere else. The
boundary example now comes out exactly `3.998046875`.

The same cap is what defines the non-finite behaviour, with no special case: it makes
`big` finite for an infinity or NaN as well, while `small`, computed from the
payload, stays non-finite. The first term issued, `small * big`, is then non-finite
whatever the other operand holds. The contract this gives, and its one weakness against
IEEE, are in [step 11](step-11-tf32-operand-abi.md#the-edges-of-the-payload-domain).

The cap costs a few integer operations on the narrowed value per operand fragment —
an `and`, an `or`, an unsigned compare and a select, which the AMDGPU backend turns
into a packed unsigned minimum on gfx950. The MMA count, their order and the packed
convert are unchanged
([step 8](step-08-comgr-compile-gate.md#observed-isa)).

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
  — the bitcast and ten-line split per operand, read back and re-derived; the cap's
  constants, its unsigned compare, and that the MMAs consume the capped `big`; RNE and
  not truncation; three terms in the pinned order threading one accumulator;
  `small*small` absent; no emulated `op_id` surviving into the emitted IR; only
  the last term writing the caller's value; a native bf16 atom still lowering to
  exactly one call with no split; and the operand contract
  ([step 11](step-11-tf32-operand-abi.md)). The emitted split is also *evaluated*, not
  just read: `TestLlvmFoldsTheSplit` hands the lowered kernel to LLVM's `opt` with
  constant payloads, which folds the narrowing, the cap, the widening and the
  subtraction down to the constants the three MMAs receive, and compares them with
  values derived by hand — the top of the finite range, both signs, the first
  overflowing payload, infinities, a NaN, and ordinary payloads that must not move.
  Both engines' output is evaluated, not only the Python one: the C++ engine's
  through the `rocke_engine` module (that lane skips where only the archive is
  built). Against the uncapped split it fails with `big = +inf` and `small = -inf`.
  The folded parts are read as integers, never as printed `bfloat` constants: the
  test bitcasts the four operands the first two MMAs consume to `<N x i16>` and hands
  them to an opaque sink call, so what it compares is the bit pattern, which every
  LLVM generation prints the same way. The spelling of a folded `bfloat` constant
  differs between generations (a hex literal, a decimal, `+inf`), and a pattern
  written for one of them reads nothing on another.
  It needs no GPU and skips without an `opt`.
- [`tests/core/test_xf32_bf16x3_lowering.cpp`](../../../tests/core/test_xf32_bf16x3_lowering.cpp)
  — the same properties pinned directly on the C engine, by parsing serialized
  IR and inspecting the emitted text. Wired into ctest as
  `rocke_xf32_bf16x3_lowering`. It pins the recipe, not byte identity: the
  Python file's engine-parity test needs the `rocke_engine` pybind module and
  *skips* where only the archive is built, which is why the
  `xf32_bf16x3_mma` parity family (below) was added after review.
- Python↔C byte identity for both atoms at llvm20, llvm22 **and** llvm23:
  `IDENTICAL` on all six, checked by lowering the serialized IR through
  `ir_lower_cli` and diffing against `_lower_kernel_to_llvm_python`. This is the
  check that actually proves the mirror; the ctest above proves the recipe.
- Byte-identity gate at llvm20 **and** llvm22 was 69 GREEN / 1 DRIFT when this step
  landed, the DRIFT being the `gfx942_attention_tiled_3d` cfg[3] divergence, proven
  pre-existing (see [step 10](step-10-byte-identity.md)). No new drift — expected,
  since no golden kernel selected an emulated atom yet.

## Follow-ups, closed after review

- The bare-`op_id` fragment lookups did not know the emulated atoms, so neither
  the C builder's `rocke_b_mma` nor the bare-string form of Python's
  `IRBuilder.mma` could construct an emulated TF32 `tile.mma`. Both now resolve the
  logical op_id through its physical atom — see
  [step 4](step-04-gfx950-xf32-rows.md#follow-ups-left-open). The C test still
  embeds serialized IR, now so that it pins the recipe independent of the
  builder.
- No shipped kernel selects an emulated atom, so the byte-identity gate never
  compared the engines on the expansion: the pybind parity test above skips in
  archive-only builds, and the C test checks only C output. The
  [`xf32_bf16x3_mma`](../../../tests/instances/parity/xf32_bf16x3_mma_emit.py)
  parity pair closes that. It builds both atoms through each engine's builder
  from the bare op_id and is byte-compared by `tools/check_byte_identity.py`,
  which needs only the archive. Swapping two terms in the C++ expansion alone
  turns both of its configs into `MISMATCH`.
