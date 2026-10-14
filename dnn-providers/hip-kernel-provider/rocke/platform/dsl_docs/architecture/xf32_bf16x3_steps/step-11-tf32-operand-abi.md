# Step 11 — the emulated atoms take the base's TF32 operand type

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4,
step 11).

## Contents

- [Why](#why)
- [What changed](#what-changed)
- [Design calls](#design-calls)
  - [One operand ABI per logical dtype](#one-operand-abi-per-logical-dtype)
  - [No payload bit is dropped](#no-payload-bit-is-dropped)
  - [The edges of the payload domain](#the-edges-of-the-payload-domain)
  - [The verifier reads the catalog](#the-verifier-reads-the-catalog)
  - [A catalog that cannot over-claim](#a-catalog-that-cannot-over-claim)
  - [The native-preferred rule keys on the full shape](#the-native-preferred-rule-keys-on-the-full-shape)
  - [HIP rejects by name](#hip-rejects-by-name)
  - [What was removed](#what-was-removed)
- [Why gfx942 is untouched](#why-gfx942-is-untouched)
- [Verification](#verification)
- [Follow-ups left open](#follow-ups-left-open)

## Why

Steps 1–10 were written against a base with no TF32 atom anywhere. The base then
gained a logical `tf32` dtype — a real 32-bit IR type carried as I32, so that raw
payload bits and explicit round-to-nearest-even preparation stay distinguishable —
and two native gfx942 atoms that take `vec<tf32x2>` operands. The emulated atoms
here took fp32 vectors, under an `xf32` selection tag modelled on CK's `tf32_t`.

Two operand ABIs under one logical dtype is the defect. A kernel selected through a
`(tf32, tf32, fp32)` catalog query would hold `vec<tf32x2>` on gfx942 and need
`vec<f32x8>` on gfx950, so the one capability the emulation exists to provide —
write the kernel once, let the catalog pick native or emulated — would not exist. And
the base's verifier rejects a TF32-typed operand on any atom but its two native ones,
so the emulated atoms could not even have been handed the type the query promises.

## What changed

- **Operand type.** The emulated atoms take `vec<tf32xN>` A and B and a `vec<f32xM>`
  accumulator, as the native ones do. N is the fragment length they borrow from the
  physical atom (8 for both).
- **Names.** `xf32_bf16x3_f32_16x16x32` → `mfma_f32_16x16x32_xf32`, and likewise
  `…32x32x16`. The catalog rows are `tf32` / `tf32` / `fp32`; `xf32` survives as the
  alias spelling the base accepts.
- **Expansion**, in both engines: view each operand's I32 carrier as fp32 with a
  `bitcast`, then split exactly as before. Operands that are not TF32 vectors of the
  atom's own length are rejected, with a message that says how to fix it.
- **Verifier**, in both engines
  ([`core/tf32.py`](../../../python/rocke/core/tf32.py),
  [`tf32_internal.h`](../../../cpp/include/rocke/tf32_internal.h)): a new branch for
  emulated atoms, reading operand and accumulator lengths from the catalog. The native
  branch and its `gfx942` guards are untouched.
- **Catalog checks**, at load: the recipe fixes the operand dtype, and the physical atom
  must be a native row of the *same* arch.
- **HIP lowerers** reject an emulated atom by name.
- **Parity pair, device harness and tests** moved to the new type and names; the
  fp32 loads are wrapped by a `bitcast`, the same way the base's own probe wraps them.
- **Removed:** the MMA result-dtype field and the `xf32` selection-tag registration, with
  the tests that pinned it.

## Design calls

### One operand ABI per logical dtype

The alternative with the least churn was to leave the emulated atoms on fp32 vectors:
the base's verifier would never see them, since it only objects to TF32-typed operands.
That is exactly why it is wrong — it would work, and quietly split the dtype in two.
The cost of the chosen alternative is one extra `bitcast` at the start of the expansion
and a verifier branch in code the base owns; the second is additive and small.

### No payload bit is dropped

A native XF32 instruction ignores the low 13 bits of each operand. The expansion
reinterprets the entire 32-bit payload as fp32 and splits that, so every bit the caller
supplied reaches the result. Three outcomes follow from one input — the base's own
boundary example is an operand `0x3f801001` multiplied by 1.0:

| path | result |
|---|---|
| native atom, raw payload | `0x3f800000` — low bits dropped |
| native atom, after explicit RNE preparation | `0x3f802000` — rounded to 10 mantissa bits |
| BF16x3 emulation | `0x3f801000` — the payload to within one fp32 ulp |

All three are correct for what each is. The contract the emulated atom promises is
therefore *to track the exact product of the payloads it is given, to within the split's
own error; not bit-reproducible against the native atom across targets*. Which of the
two is closer to that exact product depends on the operands:

- on raw fp32 payloads the emulation is the more accurate, because the native atom
  truncates each operand to 10 mantissa bits first;
- on payloads already prepared to TF32 precision the native atom is exact, while BF16x3
  omits the `small*small` term. That term is below `2^-16` of the product and reaches
  99% of it. For `A = B = 1 + 2^-8` each operand sits exactly halfway between two bf16
  values, so round-to-nearest-even keeps `big = 1` and leaves `small = 2^-8`; the three
  terms give `1.0078125` against the exact `1.0078277587890625`, a relative error of
  about `1.514e-5`. That is far below TF32's own `2^-11`, and not zero. An earlier
  version of this note bounded the term at `2^-18`; that operand pair exceeds it eight
  times over.

So "more accurate than native" is a statement about raw payloads, and no more. A caller
who hands the atom operands it has already rounded to TF32 precision gets up to that
`1.5e-5` per product less accuracy than the native atom gives it: the price of having
the operation on a target whose hardware lacks it.

The device harness pins the raw-payload half with operands that carry bits below the
native mantissa, and the host model pins the midpoint pair
([`test_xf32_bf16x3_numerics.py`](../../../tests/core/test_xf32_bf16x3_numerics.py),
`TestTheOmittedTerm`).

### The edges of the payload domain

The operand ABI promises that every 32-bit payload reaches the split, so the split has to
be defined for all of them rather than for the ones a model would draw.

**The top of the finite range.** bf16 round-to-nearest-even overflows to infinity for
every payload of magnitude `0x7f7f8000` or more: the top `2^-9` of the finite fp32
range, which includes TF32's own largest finite value, `0x7f7fe000`. An uncapped split
then gives `big = +inf` and `small = -inf`, and the `inf * 0` in the cross terms turns a
*finite* product into NaN. `0x7f7fe000 * 0x00800000` is exactly `3.998046875`, and came
out NaN. The split now holds `big` at the largest finite bf16, `0x7f7f`, keeping its
sign, and takes `small` from the payload as before, so it holds what the cap threw away
([step 5](step-05-bf16x3-lowering.md#saturation-the-top-of-the-finite-range)). The
product comes out exactly `3.998046875`. Nothing below the threshold changes: the cap is
the identity on every finite bf16, so every other payload splits bit for bit as it did.

**Non-finite payloads.** The same cap makes `big` finite for an infinity or NaN, while
`small`, computed from the payload, stays non-finite. The first term issued,
`small * big`, is then non-finite whatever the other operand holds. The contract is:

> an infinity or NaN in an operand makes every output element it feeds non-finite, and
> no other output.

`inf * 1` is `inf`, `inf * inf` is `inf`, `inf * 0` is NaN as IEEE says, and a NaN
anywhere gives NaN. The one place this is weaker than IEEE: against a partner of
magnitude 512 or more, the partner's own residual can overflow against the cap with the
opposite sign, and the two infinities sum to NaN, so an infinity against a large partner
may come out as NaN rather than `inf`. It is still not finite, and a finite-looking
result is the failure this contract excludes. A native XF32 instruction handles these
payloads in its own way, so this is not a claim of agreement with it either.

**Subnormal payloads.** bf16 has fp32's exponent range but only seven mantissa bits, so
its smallest positive value is `2^-133`. A payload below the smallest normal fp32 is
therefore split to an *absolute* error of `2^-134` per operand rather than a relative
one, and a payload no larger than `2^-134` (about `4.6e-41`) contributes nothing: both of
its parts come out zero, though the exact product with a large partner is not. On a
gfx950, subnormals the bf16 grid holds exactly (`0x00040000`, `0x00200000`) and the
largest subnormal came out exact to `1.2e-7`, and `0x00000001` and `0x00001000` against
`2^100` came out zero. Signed zero accumulates into the `+0` the accumulator starts at,
so `-0 * 1` is `+0`. The contract above says "tracks the exact product" for payloads of
ordinary magnitude; this is its floor.

**What the cap costs** is a few integer operations on the narrowed value per operand
fragment. The MMA count, their order and the packed convert are unchanged.

### The verifier reads the catalog

The base's TF32 contract is a two-entry table keyed by op_id in Python and a pair of
`strcmp`s in C++. Adding the emulated ids to it would have conflated two properties it
currently holds in one place: *which atoms take TF32 operands* (a property of the
logical dtype) and *which atoms are native gfx942 instructions* (a property of the
hardware and its emitter). Only native atoms had existed, so the two coincided. The
emulated atoms are the first to separate them, and the catalog already stores them
apart: the logical dtype is the row's operand dtype, whether it is native is the
`emulation` descriptor, and availability on a target is whether the row is in that
target's list.

So the new branch asks the catalog by op_id
(`target.emulated_tf32_operand_lens` / `rocke_arch_mma_op_id_emulated_operands`) and
leaves the table, its `gfx942` guards and the HIP emitter that builds a
`__builtin_amdgcn_<op_id>` name from it exactly as they were. The lookup keys on op_id
for the same reason `_op_id_c_dtype` does: an op_id names one atom, so what the verifier
reads from it is invariant across the arches that list it, and the lookup raises if a
catalog says otherwise.

### A catalog that cannot over-claim

The old load-time check on an emulated row asked whether its physical atom had fragment
metadata *anywhere*. A row could therefore name an atom its own arch cannot issue: the
catalog would load, claim support, and the compiler would reject the kernel at codegen
(`Cannot select`). The check is now per arch — the physical atom must be a native row of
the same arch — and a test builds exactly the bad catalog and expects the rejection.
The C catalog is static data, so its twin is the existing
`test_emulated_rows_mirror_physical`, which already walks each emulated row's own arch.

### The native-preferred rule keys on the full shape

`_prefer_native` drops an emulated atom only when a native atom of the same
`(m, n, k)` is listed. A native atom of a *different* K does not shadow an emulated
one, and `select_largest_k` then takes the larger K — which would be the emulated atom,
three MFMAs instead of one. This was reproduced against the real `MmaCatalog` with a
synthetic native K=8 atom and an emulated K=16 or K=32 one: all three queries returned
the emulated atom.

Nothing triggers it, because no arch lists both: gfx942 has only native TF32 atoms and
gfx950 only emulated ones. `test_no_arch_mixes_native_and_emulated_tf32` pins that, and
the rule's docstring says where it has to be re-keyed (on `(m, n)`) before an arch
breaks it.

### HIP rejects by name

The expansion exists only in the LLVM lowerer. Before this step an emulated atom reached
the HIP lowerers' generic dispatcher as `tile.<op_id>` and failed with "no HIP lowering
for op", which, once the name looks like a hardware instruction, reads as a missing
hardware atom. Both HIP lowerers now say that the atom is emulated and has no HIP
lowering.

### What was removed

- **The MMA result-dtype field.** Step 1 added `d_dtype` so an atom's D could differ from
  its C. No catalog row ever set it, no emission path read it, and the base's contract is
  that a TF32 atom's result is its FP32 accumulator, enforced by the verifier rather than
  by a field. It is gone from both engines.
- **The `xf32` selection tag** and its tests. The base's `tf32` supersedes it: `tf32` is
  canonical, `xf32` an alias, and it is a real IR type with loads, stores and bitcasts —
  the opposite of what those tests asserted.

## Why gfx942 is untouched

Offering the emulated atoms on gfx942 as well was considered and dropped:

- The two atoms here expand into the K-packed bf16 MFMAs, which exist only on gfx950; the
  compiler-probed intrinsic table records `mfma.f32.16x16x32.bf16` and
  `mfma.f32.32x32x16.bf16` as unavailable on gfx942. A gfx942 emulation would expand into
  that arch's own `16x16x16` and `32x32x8` instead, so it is two different atoms, not the
  same two with another arch added.
- Listing a native and an emulated atom for one `(m, n)` is the case the selection rule
  above gets wrong.
- It would put two numerics — truncating native, full-payload emulated — behind one
  dtype on one arch, which needs a stated selection policy rather than K ordering.

## Verification

- `tests/core` is green apart from `test_arch_domain_artifact.py`'s two failures (the
  `llvm22` intrinsic-domain artifact lacks the cluster and prefetch intrinsics the base
  added, and the "regenerating this host's column is a no-op" check follows from it),
  which fail identically on the unmodified base.
- The C++ tests for this work run under `ctest`: `rocke_tf32`, `rocke_dtypes`,
  `rocke_mma_catalog_lookup`, `rocke_xf32_bf16x3_lowering`, `rocke_mma_frag_ssot`.
- The `xf32_bf16x3_mma` parity family is byte-identical between the engines at llvm20,
  llvm22 and llvm23, in both `ll` and canonical `ir` mode.
- On gfx950 the emulated `tile.mma` matches the hand-written BF16x3 kernel bit for bit —
  compared on the `uint32` patterns, with the top of the finite range mixed into the
  operands — and lands at the figures in [step 9](step-09-numeric-regression.md), under
  the Python lowerer, the C++ engine with its fallback disabled, and both engines
  cross-checked; the two injected faults are still rejected. The edges of the payload
  domain run there too: the top-binade products come out finite and exact, an infinity
  or NaN reaches exactly the outputs that consume it (16 or 32 of them, the rest bit for
  bit unchanged), and the same harness run against the uncapped split fails on all of
  them.
- The emitted split is also evaluated by LLVM's own constant folder, with the payloads
  of the table above and the boundary example, and compared with hand-derived values
  (`TestLlvmFoldsTheSplit`); against the uncapped split it fails with `big = +inf` and
  `small = -inf`, the mechanism described above.
- Byte identity across every kernel family: [step 10](step-10-byte-identity.md).

## Follow-ups left open

- **Emulation on gfx942**, as an accuracy option rather than a capability. It needs the
  re-keyed selection rule, the `16x16x16` / `32x32x8` atoms, and its own device
  validation.
- **The native guard stays hand-written.** `TF32_MMA` and the five `gfx942` checks could
  themselves be derived from the catalog; they belong to the code that added the native
  atoms, and nothing here needs them to change.
- **The tiling atom table** (`mma_traits.json`) lists hardware atoms with their
  `llvm_builtin` and has no notion of an emulated one, so the tiling planner cannot plan
  with these atoms.
- **No selector maps a request onto these atoms yet.** They are reachable by a catalog
  query and by op_id; production dispatch is a separate change.
