# f8f6f4 format parameterization + intrinsic decl-table validation

Working notes and plan. Extends
[`dtype_coverage_audit.md`](dtype_coverage_audit.md) P0 ("complete fp8/bf8
(OCP) end-to-end"), starting from the ticket
*"`_op_tile_mfma_scale_f32_16x16x128_f8f6f4` behavior"*.

> **Lifecycle — this file is temporary.** It is tracked so the measurements in
> §1/§2 and the re-verification checklist in §6 survive a machine change, not
> because it is reference documentation. **Delete it once Plan A (§3) and Plan B
> (§4) have landed**; anything in it still worth keeping at that point belongs in
> `dtype_coverage_audit.md`, `TESTING.md`, or the validator docstrings, not here.

> **Toolchain — read first.** This revision was re-measured end to end on
> **ROCm 10.0.0 / AMD LLVM 23.0.0git** (`/opt/rocm`, comgr at
> `/opt/rocm/lib/libamd_comgr.so`, rocKE flavor detection reports **`llvm23`**).
> The previous revision was measured on ROCm 7.1 / LLVM 20 and is superseded.
> Intrinsic signatures, mangling and auto-upgrade behavior are
> LLVM-version-dependent: on a *different* ROCm, re-run §6 before trusting §1
> or acting on §3.
>
> **The validatable flavor has flipped.** `llvm23` is now the only flavor this
> host can validate; `llvm20` and `llvm22` are now the unverifiable ones. Any
> "llvm20" claim below is inherited from the previous revision, not re-measured.

---

## 1. What the hardware op actually is (re-measured, ROCm 10.0 / LLVM 23)

Unchanged from the LLVM 20 measurement. The real intrinsic is **9 arguments**,
with **two** mangling suffixes (the return type is *not* overloaded — only the
two `anyvector` operands are):

```llvm
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(
    <8 x i32>,        ; A
    <8 x i32>,        ; B
    <4 x float>,      ; C
    i32 immarg,       ; cbsz    - A format selector
    i32 immarg,       ; blgp    - B format selector
    i32 immarg,       ; opsel_a
    i32,              ; scale_a (E8M0, runtime value)
    i32 immarg,       ; opsel_b
    i32)              ; scale_b (E8M0, runtime value)
```

Verified on LLVM 23: `opt -passes=verify` passes and the round-tripped declare
name is **byte-identical** (no auto-rename).

CBSZ / BLGP → format and required operand width. Re-verified on LLVM 23 by
compiling each selector to gfx950 assembly (`clang -x ir -S -mcpu=gfx950`);
every row below emits a real `v_mfma_scale_f32_16x16x128_f8f6f4`:

| selector | format | operand vector type | LLVM 23 |
|---|---|---|---|
| 0 | fp8 (E4M3) | `<8 x i32>` | codegen OK |
| 1 | bf8 (E5M2) | `<8 x i32>` | codegen OK |
| 2 | fp6 (E2M3) | `<6 x i32>` | codegen OK |
| 3 | fp6 (E3M2) | `<6 x i32>` | codegen OK |
| 4 | fp4 (E2M1) | `<4 x i32>` | codegen OK |

Notes:
- **Mixed A/B formats still work.** fp8 × fp4 (`cbsz=0, blgp=4`, `v8i32` ×
  `v4i32`) compiles to the instruction.
- The `2` vs `3` (E2M3 vs E3M2) split is still read from ISA documentation,
  **not** tested. Both *assemble*; which one is which is unverified. Verify
  numerically before relying on it.
- The `32x32x64` sibling **exists** in LLVM 23
  (`llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4`, confirmed in the intrinsic
  enumeration — see §4.1). Plan A step 5 is unblocked.
- **Clang's HIP builtin still pins A/B to `v8i32` on LLVM 23** — it accepts the
  `v8i32` call and rejects a `v4i32` one with a hard type error. So **fp6/fp4
  must still be emitted as raw IR**, not via `__builtin_amdgcn_mfma_scale_*`.

## 2. What is broken in the repo today (re-measured on LLVM 23)

| # | Location | Problem | LLVM 23 status |
|---|---|---|---|
| B1 | `lower_llvm.py:855` | Decl has **11 args**, no mangling suffix. | **Still broken.** `opt`: *"incorrect number of args. Expected 9, but got 11"* |
| B2 | `lower_llvm.py:881` (`mfma.f32.16x16x128.fp8.hero`) | **9 args but unmangled**, and declares the *same symbol name* as B1 with different arity → the two collide. | **Still broken.** Silently renamed to `.v8i32.v8i32` |
| B3 | `lower_llvm.py:863` (`mfma.f32.16x16x128.fp4`) | **Intrinsic does not exist.** | **Still fictional** — absent from the LLVM 23 enumeration. Deletion, not a signature fix. |
| B4 | `lower_llvm.py:867` (`mfma.f32.16x16x96.fp6`) | Same. | **Still fictional.** |
| B5 | `lower_llvm.py:3219` + `cpp/core/lower_llvm/mma.cpp:766` | Handler hardcodes `cbsz=0, blgp=0` → both operands pinned to fp8e4m3; operand types fixed at `<8 x i32>`. **This is the ticket.** | unchanged |
| B6 | base decl table, fp8/bf8 MFMA entries | Base table uses `<2 x i32>`; LLVM 23 rejects it (*"argument 0 type expected i64, but got `<2 x i32>`"*). The override dict's `i64` is what works. | **Confirmed stale**, same verdict as LLVM 20 |
| B7 | `make.buffer.rsrc.p1` | Base decl uses `i32` offset; LLVM 23 wants `i64` (the override is correct). | **Confirmed stale** |
| B8 | `ds.read.tr16.b64` | Silently renamed to `...v4i16`. | **Still renames** |

### New on LLVM 23 (not in the previous revision)

| # | Location | Problem |
|---|---|---|
| **B9** | `ds.read.tr16.b128` (`lower_llvm.py`, `ir.py:3060`, `lower_hip.py:143`) | **Intrinsic does not exist.** LLVM 23 has `ds.read.tr16.b64` and `ds.load.tr16.b128`, but **no `ds.read.tr16.b128`**. This is a *live* path — public op `ir.ds_read_tr16_b128`, an LLVM handler at `:3997`, and a HIP `__asm` alias at `lower_hip.py:145` naming the same fictional symbol. Fails at link on this toolchain. Likely a rename to `ds.load.tr16.b128` (which rocke already declares under two other keys). |
| **B10** | `global.atomic.fadd.v2f16` / `.v2bf16` | **Removed from LLVM 23.** The declare is auto-upgraded *away entirely* into `atomicrmw fadd ... syncscope("agent") seq_cst`. Semantics look preserved, but the emitted IR no longer contains what rocke thinks it emits. ~~Both entries have zero `_need()` call sites~~ — **corrected, see §4.9**: only `.v2bf16` is dead. |
| **B11** | `amdgcn.cvt.scalef32.pk.fp8.f32` / `.bf8.f32` | The previous revision left these *unresolved* (LLVM 20 reported a misleading `Attribute after last parameter!`). LLVM 23 gives a precise diagnosis and the drift is **two-dimensional**: repo declares `i32 f(i32, <2 x float>, float, i1)`; LLVM 23 wants a **`<2 x i16>` return** *and* **5 arguments**. Also zero `_need()` call sites. |

**Why none of this has blown up in production:** the shipping fp8 path
(`instances/common/moe_fused_mega_fp8.py:72`) uses **inline asm**
(`helpers/asm.mfma_f8f6f4_agpr*`), so these decls are never emitted.
`instances/common/mx_gemm.py` uses the explicit v1 chain
(`decode_mx_scale_e8m0` + `apply_mx_scale`), not the scaled intrinsic.
`helpers/atoms.py:576` `emit_scaled()` still has **zero callers** (re-confirmed).
B9 is the exception — it is reachable, and is the one new item worth triaging
independently of the f8f6f4 work.

Also wrong: **audit §4A credits gfx950 with `mfma_f32_16x16x128_fp4` and
`mfma_f32_16x16x96_fp6` atoms. Those intrinsics do not exist** (B3/B4, still
true on LLVM 23). `core/arch/data/arch_specs.json:92-93` lists them, and does
*not* list `mfma_scale_f32_16x16x128_f8f6f4`. Both need correcting.

## 3. Plan A — f8f6f4 format parameterization (the ticket)

Unchanged in shape; LLVM 23 confirms every premise it rests on.

1. **Fix the ABI.** Collapse B1 + B2 into a single correct mangled 9-arg decl
   (§1). Delete the `.hero` duplicate. Mirror into
   `cpp/core/lower_llvm/data.cpp:88`.
2. **Add the compile gate for fp8 only** (= Plan B Tier 0/1/2, §4). This locks
   the ABI so step 3 cannot silently regress it. *Do this before step 3.*
3. **Parameterize `cbsz` / `blgp` and the operand vector widths.** Thread an
   A-format and B-format knob through `ir.py:2126`
   (`mfma_scale_f32_16x16x128_f8f6f4`) → handler `lower_llvm.py:3219` →
   `cpp/core/lower_llvm/mma.cpp:766`. Emit the type-overloaded intrinsic form
   per format pair. Widths per the §1 table.
4. **Re-point the fp4/fp6 atoms** (`helpers/atoms.py:415` `fp4_16x16x128`,
   `fp6_16x16x96`) at the scaled op with the right selector, and **delete**
   B3/B4 plus their handlers (`lower_llvm.py:3260`, `:3289`;
   `mma.cpp:823`, `:872`). Update `arch_specs.json` and
   `core/arch/target.py:652` frag info to match. Note the ticket's own
   remark: this op *cannot* serve as the fp4/fp6 fallback the other handlers
   reference until this step lands.
5. **Add `32x32x64`** (confirmed to exist, §1), and wire `mx_gemm` to the
   scaled intrinsic path.

Touched tiers, end to end: `ir.py` (op surface) → `lower_llvm.py` decl table
+ handler → `cpp/.../data.cpp` + `mma.cpp` (mirror) → `helpers/atoms.py` →
`core/arch/target.py` + `arch_specs.json` → `lower_hip.py:1665` (currently a
debug shim that just copies `c`).

## 4. Plan B — make the decl table validatable

> **Status update — two tiers have landed and the tier design changed.** §4.1's
> header-enumeration tier was **dropped**; §4.2's tier table is superseded by
> §4.8. Read §4.8 first; §4.1–§4.7 are kept for the measurements they carry.

**Today:** 151 hand-written strings in `lower_llvm.py:391`, hand-mirrored into
flat C arrays at `cpp/core/lower_llvm/data.cpp:88`. Nothing checks either side.
`_need()` (`lower_llvm.py:1555`) only sets a flag; it validates nothing, so a
bad decl surfaces only when some kernel happens to emit it.

The goal stated for this work is exactly right and unchanged: **turn the decl
table from "hand-written" into "checkable"**, without a `.td`-driven generator
(the `.td` files are not in the install package, and a generator would create a
source-tree dependency).

### 4.1 The main change this revision: a cheap *existence* oracle

The previous plan used **comgr compile-to-hsaco** to catch
*intrinsic-does-not-exist*. That is no longer the cheapest option. The installed
ROCm ships

```
/opt/rocm/llvm/include/llvm/IR/IntrinsicsAMDGPU.h
```

which enumerates **every** AMDGPU intrinsic by name — 1470 of them on this host,
one per line as `llvm.amdgcn.<name>` in a trailing comment. This is a plain
header in the install tree: no `.td`, no source checkout, no generator, no GPU,
no comgr, and it answers the existence question in **milliseconds**.

Matching rule: a declared name is valid if it **or any dot-prefix of it** is in
the enumerated set (the header lists base names; declared names may carry
overload-mangling suffixes the header omits).

Measured against the current table — 5 decls name an intrinsic that does not
exist:

```
global.atomic.fadd.v2bf16    llvm.amdgcn.global.atomic.fadd.v2bf16.p1   (B10)
global.atomic.fadd.v2f16     llvm.amdgcn.global.atomic.fadd.v2f16.p1    (B10)
ds.read.tr16.b128            llvm.amdgcn.ds.read.tr16.b128              (B9, NEW)
mfma.f32.16x16x128.fp4       llvm.amdgcn.mfma.f32.16x16x128.fp4         (B3)
mfma.f32.16x16x96.fp6        llvm.amdgcn.mfma.f32.16x16x96.fp6          (B4)
```

**B3 and B4 pass `opt -passes=verify` silently** — the verifier does not object
to a declare for a nonexistent `llvm.amdgcn.*` symbol. So this tier is not
redundant with the `opt` tier; it is the only cheap thing that catches them,
and it found B9, which the LLVM 20 revision missed entirely.

29 decls are generic `llvm.*` (not `llvm.amdgcn.*`) and are skipped by this tier.

### 4.2 Tiers

Still true that **no single check catches every bug class** — re-measured:

| check | B1 (11-arg) | B2 (9-arg unmangled) | B3 (fictional fp4) | B9 (fictional b128) |
|---|---|---|---|---|
| header enumeration | — | — | **FAIL** | **FAIL** |
| `opt -passes=verify` | **FAIL** | pass | pass | pass |
| name round-trip compare | — | **RENAMED** | — | — |

- **Tier 0 — dual-engine table equality.** Pure Python, zero LLVM dependency,
  runs anywhere. Compare the Python tables vs the `data.cpp` arrays on key set,
  **insertion order** (it drives emit order in `finalize`), and decl text.
  Cheapest tier and the one that directly guards byte-identity upstream.
  **Prototyped and already 100% green** — all three arrays match exactly:

  | array | cpp | py | keys + order | decl text |
  |---|---|---|---|---|
  | `ROCKE_LL_INTRINSIC_DECLS` | 151 | 151 | MATCH | 0 mismatches |
  | `..._LLVM22_OVERRIDES` | 5 | 5 | MATCH | 0 mismatches |
  | `..._LLVM23_OVERRIDES` | 5 | 5 | MATCH | 0 mismatches |

  So Tier 0 lands green with an **empty** `KNOWN_BAD`. It is a pure regression
  lock and should land first, on its own, ahead of everything else.

- **Tier 1 — existence (header enumeration).** §4.1. No LLVM invocation at all.
  Skip with a clear "unvalidated" report if the header is absent (mirror the
  `shutil.which("opt")` guard style). **This replaces the comgr tier as the
  default**; comgr compile-to-hsaco stays available as an optional deep tier
  but is no longer needed to catch the bugs we actually have.

- **Tier 2 — `opt` round-trip.** `opt -passes=verify` catches arity/type errors;
  then compare the round-tripped `declare` name against the original to catch
  silent renames, and distinguish a third outcome — **`UPGRADED-AWAY`**, where
  the declare vanishes because LLVM auto-upgraded the intrinsic into plain IR
  (B10). The previous revision folded that into "renamed" and would have
  reported an empty rename target. Skip via `shutil.which("opt")` when absent.

- **Tier 3 — coverage.** Every key passed to `_need()` must exist in the table
  (measured: **0 violations** — no `_need()` key is missing), and table keys
  with no `_need()` call site are dead. **See the f-string warning below.**

### 4.3 Probe synthesis — corrections to the previous revision

The parser design holds (~20-line `declare` parser: depth-aware comma split over
`<> {} () []`, strip attribute words). It parses 151/151. Four corrections:

1. **Drop `-O0`.** The previous revision said to compile probes at `-O0`.
   LLVM 23's `opt` **rejects** `-O0` combined with `-passes=`
   (*"Cannot specify -O# and --passes="*) — it fails all 151 probes with a
   flag error that looks exactly like a verify failure. It is also unnecessary:
   `-passes=verify` runs no optimization, so there is no DCE to defend against
   in this tier. The `-O0` + `store volatile` sink advice remains correct and
   necessary **only** for the comgr/codegen tier, where the original false-green
   was observed.
2. **`metadata` params are not SSA values.** `av.load.b128.*` / `av.store.b128.*`
   take a `metadata` operand. Passing `metadata %a1` is a parse error and
   passing a bare `metadata !"av"` is rejected by the verifier
   (*"must be a metadata string"*). The accepted form is a **node-wrapped**
   MDString: `metadata !0` with `!0 = !{!"av"}`. With that, all four entries
   pass — they were probe bugs, exactly as the previous revision suspected, and
   are now fixed rather than allowlisted.
3. **"Pin every integer to 0" is too blunt.** Some `immarg`s are
   *range-validated*: the load-to-LDS family (`raw.ptr.buffer.load.lds`,
   `global.load.lds`, `raw.ptr.buffer.load.async.lds`) rejects size 0 with
   *"invalid data size ...; must be 1, 2, 4, 12, or 16"*. Pinning all integers
   to 0 produces three false VERIFY-FAILs. Keep the pin-all-integers default
   (it removes the under-annotation noise the previous revision documented) but
   add a small explicit **`IMMARG_HINTS`** table, `{intrinsic: {param_index:
   value}}`, for the handful with validated ranges. Three entries cover it today.
4. **Tier 3 must expand f-strings.** 31 of the `_need()` call sites are
   **f-strings** (`f"fmuladd.{ty_name}"`, `f"av.load.b128.p{space}"`,
   `f"amdgcn.s.wqm.{ty_name}"`, …). A literal-only scan finds 71 literal keys
   and reports **80 of 151 table entries as dead** — including obviously-live
   ones like `workitem.x` and `fmuladd.f32`. Convert each f-string call site to
   a glob/regex before differencing, and keep the tier **warn-only** until that
   is proven quiet.

### 4.4 Regenerated `KNOWN_BAD` (LLVM 23, from scratch)

Full sweep of all 151 decls at the installed flavor: **under 1 s** wall clock
(~6 ms per decl; faster than the LLVM 20 revision's 6.2 s, largely because the
bogus `-O0` is gone). Cheap enough to run by default in the normal test suite.

**`llvm23` effective table (base + llvm23 overrides) — 7 entries:**

| key | tier | finding |
|---|---|---|
| `mfma.scale.f32.16x16x128.f8f6f4` | 2 | VERIFY-FAIL: expected 9 args, got 11 — **B1**, Plan A step 1 |
| `mfma.f32.16x16x128.fp8.hero` | 2 | RENAMED → `.v8i32.v8i32` — **B2**, Plan A step 1 |
| `ds.read.tr16.b64` | 2 | RENAMED → `.v4i16` — **B8** |
| `amdgcn.cvt.scalef32.pk.fp8.f32` | 2 | VERIFY-FAIL: return must be `<2 x i16>`, and 5 args — **B11** |
| `amdgcn.cvt.scalef32.pk.bf8.f32` | 2 | VERIFY-FAIL: same — **B11** |
| `global.atomic.fadd.v2f16` | 1+2 | UPGRADED-AWAY / not in enumeration — **B10** |
| `global.atomic.fadd.v2bf16` | 1+2 | UPGRADED-AWAY / not in enumeration — **B10** |

Tier 1 adds three more that Tier 2 cannot see: `mfma.f32.16x16x128.fp4` (**B3**),
`mfma.f32.16x16x96.fp6` (**B4**), `ds.read.tr16.b128` (**B9**).

**Union: 10 distinct broken entries.** Of these, four (`global.atomic.fadd.*`,
`cvt.scalef32.pk.{fp8,bf8}.f32`) have **zero `_need()` call sites** and should be
**deleted**, not allowlisted — dead entries do not deserve an allowlist slot.
That leaves a real initial `KNOWN_BAD` of **6**: B1, B2, B3, B4, B8, B9.

**Override-completeness — a new derived check.** Running the *base* table
through the installed verifier yields 12 failures; the difference from the
`llvm23` run is exactly 5 keys — the 4 fp8/bf8 MFMA entries (**B6**) plus
`make.buffer.rsrc.p1` (**B7**) — and that set is **exactly** the key set of
`_INTRINSIC_DECLS_LLVM22_OVERRIDES` / `_LLVM23_OVERRIDES`. So we can assert:

> *the set of base-table entries that fail under the installed verifier must
> equal the override key set for the installed flavor.*

This is worth landing as its own assertion. It is the one check that says
something meaningful about the **override mechanism** rather than about a single
flavor, and today it passes exactly. (It also settles §6 item 4: the base table
is what is stale, not the flavor mapping.)

### 4.5 Landing strategy

Mirror the `KNOWN_VIOLATIONS` convention already used by
[`library/tests/test_library_layering.py`](../../../library/tests/test_library_layering.py):
land the validator green behind an explicit **`KNOWN_BAD` allowlist**, each entry
annotated with reason + owning step, then burn entries down as Plan A lands.
Same rule: **only shrinks, never grows.**

Suggested landing order (each independently green):

1. **Tier 0** — empty `KNOWN_BAD`, no LLVM needed. Pure regression lock.
2. **Tier 1** — `KNOWN_BAD` = {B3, B4, B9}; delete B10's two dead entries in the
   same change.
3. **Tier 2 + override-completeness** — `KNOWN_BAD` = {B1, B2, B8}; delete B11's
   two dead entries in the same change.
4. **Tier 3** — warn-only until the f-string expansion is proven quiet.

Then Plan A step 1 burns B1 + B2, and step 4 burns B3 + B4.

### 4.6 Limitation to document in the validator docstring

Only the **installed** flavor can be validated, and on this host that is
**`llvm23`**. `_INTRINSIC_DECLS` in its base (LLVM 20) shape and
`_INTRINSIC_DECLS_LLVM22_OVERRIDES` (`lower_llvm.py:994`) are unverifiable here
— mark them explicitly *unvalidated* and skip. Do not let them report green.
Note this is the **inverse** of the previous revision's limitation, which could
validate llvm20 and not llvm22/llvm23: the limitation travels with the host, so
the docstring must derive it from the detected flavor rather than hardcode it.

The override-completeness assertion in §4.4 is the partial exception — it is the
only thing here that says something checkable about a flavor the host does not
run.

### 4.7 Optional next step (unchanged, still tier-2 work)

Build the declare text from `(intrinsic_base_name, [operand_llvm_types],
ret_type)` and compute the mangling suffix from "which parameters are
`anyvector`". That removes the need to hand-write 10 strings for f8f6f4's
5 formats × 2 shapes. Still an implementation detail to do **when format
parameterization actually lands** (Plan A step 3), not before. The §4.1
enumeration header makes a light version of this more attractive than it was:
the base name can be validated against the enumeration at construction time.

### 4.8 What actually landed (supersedes §4.1, §4.2, §4.5)

Two new empirical facts changed the design:

1. **`clang -S` does not catch a fictional intrinsic either.** §4.1 assumed
   `opt -passes=verify` was the thing that missed B3/B4 and that only the
   enumeration header could see them cheaply. Measured on LLVM 23: codegen also
   accepts them — the backend treats the unknown `llvm.*` name as an ordinary
   external function and emits a GOT-relative call
   (`...fp4@gotpcrel32@lo`, `s_swappc_b64`). The undefined symbol appears
   **only at link**. So a link-based module gate subsumes the enumeration tier
   entirely, and does it without depending on an install-layout header. §4.1 is
   dropped.
2. **An in-process compile cannot be the oracle.** A backend failure is a
   `report_fatal_error`: the first full-corpus sweep exited with
   `LLVM ERROR: Do not know how to expand this operator's operand!` and produced
   no report at all. Every compile must run in a subprocess.

Revised tier map:

| tier | what | status |
|---|---|---|
| 0 | dual-engine decl-table equality (keys, order, text) — pure Python | **landed**, `tests/core/test_intrinsic_decl_table_parity.py` |
| 1 | header enumeration | **dropped** — subsumed by tier 2b |
| 2a | `opt -passes=verify` over whole modules | landed as the cheap pre-pass inside the gate |
| 2b | **compile + link every corpus module** | **landed**, `tools/check_ir_validity.py` — this is the gate |
| 3 | `_need()` coverage | not started; still needs f-string expansion (§4.3 item 4) |

The per-decl probe harness (§4.3) is **not** a gate. Module-level validation
judges what is actually emitted, which is stronger; the probe harness stays
useful only as a diagnostic for *which* decl made a module fail.

Consequence worth stating plainly: B3, B4 and B9 are **not** caught by the module
gate, because no corpus case emits them. They are dead paths (B3/B4) or a live
path with no corpus coverage (B9) — deletion and a new case respectively, not
allowlist entries.

The gate's own initial `KNOWN_BAD` has nothing to do with this document's bug
list: it is `attention/gfx942/3d_{bf16,fp16}_d128_b64`, an LLVM 23 backend fatal
error that the first sweep discovered.

### 4.9 Dead-entry deletion — what the call-site audit actually found

§4.5 step 2 said "delete B10's two dead entries" and §4.5 step 3 "B11's two dead
entries". Auditing the call sites before deleting showed **three** of those four
are dead and one is not:

| key | verdict |
|---|---|
| `global.atomic.fadd.v2bf16` | **dead, deleted.** Zero `_need()` sites. `_op_memref_global_atomic_add_pk_bf16` lowers to a generic `atomicrmw fadd <2 x bfloat>` with the fine/remote-memory metadata, and its own docstring says the intrinsic does not exist in shipping ROCm. The table entry was left behind by that fix. |
| `amdgcn.cvt.scalef32.pk.fp8.f32` | **dead, deleted.** Zero `_need()` sites, and no lowering handler in either engine — the op name is registered in `ir.py` / `core_types.cpp` but unreachable. Deleting the (wrong, per B11) declare means whoever wires the op up has to write a correct one. |
| `amdgcn.cvt.scalef32.pk.bf8.f32` | **dead, deleted.** Same. |
| `global.atomic.fadd.v2f16` | **LIVE — kept.** B10's "zero `_need()` call sites" is wrong for this one: `lower_llvm.py` `_op_memref_global_atomic_add_pk_f16` sets `_needs_intrin["global.atomic.fadd.v2f16"]` and `cpp/core/lower_llvm/mem.cpp` calls `rocke_ll_need` for it. The B10 *observation* still holds — LLVM auto-upgrades the declare away — but auto-upgrade is why it keeps compiling, so this is a latent-divergence item, not a deletion. |

The registered-but-unreachable op names (`arith.cvt_scalef32_pk_{fp8,bf8}_f32`)
were deliberately **not** removed: they sit in index-parallel arrays that the IR
serializer keys off, so removing them is a separate, riskier change.

Follow-up left open: decide whether `_op_memref_global_atomic_add_pk_f16` should
follow the bf16 path onto a generic `atomicrmw fadd <2 x half>`, so rocke emits
what the backend actually consumes instead of relying on auto-upgrade. That is an
emission change and needs a golden re-bless, so it is not folded in here.

## 5. Verification / test surface

Byte-identity gate (mandatory for any emission change, both engines):

```bash
cd platform && export ROCKE=$(pwd) && export PYTHONPATH=$ROCKE/python
python tools/check_byte_identity.py                            # llvm20
ROCKE_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py   # llvm22
ROCKE_LLVM_FLAVOR=llvm23 python tools/check_byte_identity.py   # llvm23 (this host's native flavor)
python tools/check_byte_identity.py --only mx_gemm
```

> Note: `export A=$(pwd) B=$A/x` does **not** work — `$A` is not yet set when
> the word list is expanded. Use two `export` statements, as above.

Instances usable as the functional test bed:

- `tests/instances/parity/mx_gemm_emit.py` / `.c` — 6 sampled configs
  (indices 0–5), fp8e4m3 + bf8e5m2, gfx950. Closest existing harness to the
  scaled path.
- `instances/common/mx_gemm.py` — spec knob
  `mantissa_dtype: MxMantissaDType = "fp8e4m3"`, validated at `:147` against
  `("fp8e4m3", "bf8e5m2")`. Extending that tuple is the natural place to
  expose fp6/fp4 once step 4 lands.
- `tests/instances/parity/moe_fused_mega_fp8_emit.py` / `.c`,
  `tests/instances/test_moe_fused_mega_fp8.py` — the inline-asm fp8 path;
  guards against regressing what currently ships.
- `tests/test_rocke.py:6725` `test_mx_gemm_fp8_uses_mfma`.

Also relevant: `instances/common/gemm_universal.py:659` raises
`NotImplementedError`, and `mfma_gemm.py:62-65` — audit §5 lists these as the
blocked families.

## 6. Re-verification checklist on a different ROCm

Run these first; §1 and §2 are only valid for the toolchain they were measured
on. Answers below are for **ROCm 10.0.0 / LLVM 23.0.0git**.

| # | Check | Result on this host |
|---|---|---|
| 1 | Record the toolchain and the flavor rocKE detects. | `/opt/rocm` = 10.0.0, AMD LLVM 23.0.0git, flavor **`llvm23`** |
| 2 | Re-confirm the §1 signature: 9-arg mangled declare + call + `store volatile`; `opt -S -passes=verify` passes and the round-tripped name is unchanged. | **Confirmed**, name byte-identical |
| 3 | Re-confirm the CBSZ/BLGP width table by generating one kernel per selector, including one **mixed** pair. | **Confirmed**, all 5 + mixed. Note `llc` is **not shipped** in this ROCm — use `clang -x ir -S -target amdgcn-amd-amdhsa -mcpu=gfx950` |
| 4 | Re-check B6: is the base-table `<2 x i32>` fp8 form still rejected? | **Still rejected**; `i64` is correct. The base table is what is stale — see the override-completeness check, §4.4 |
| 5 | Re-check whether B3/B4 (dense fp4/fp6) exist. If a newer LLVM added them, Plan A step 4 becomes *fix the signature* instead of *delete*. | **Still absent.** Step 4 stays a deletion |
| 6 | Re-check the clang-builtin `v8i32` pinning. If lifted, fp6/fp4 no longer need raw-IR emission. | **Not lifted.** Raw IR still required |
| 7 | Re-run the full sweep and regenerate `KNOWN_BAD` from scratch — do not carry the old list forward. | **Done**, §4.4. Regenerating found B9 and B10, and resolved B11 |

Add for next time:

8. **Re-run the enumeration tier** (§4.1) and re-confirm the header path exists
   at `<rocm>/llvm/include/llvm/IR/IntrinsicsAMDGPU.h`. It is an install-layout
   assumption, not an API guarantee.
9. **Re-check the auto-upgrade set.** B10 shows LLVM removes intrinsics between
   releases and rewrites them silently. Diff the `UPGRADED-AWAY` list against
   the previous toolchain's.
10. **Re-check which tools ship.** `llc` was present on ROCm 7.1 and is absent
    on ROCm 10.0. Prefer `clang`/`opt`, which are both present, and gate on
    `shutil.which`.

## 7. Compliance

`platform/AGENTS.md` §Compliance binds this work and **overrides any other
instruction**: no AMD Restricted/Confidential data, NPI, product/marketing
code names, internal links (Jira/Confluence/Perforce), or **software-achieved
performance numbers** in the repo, git history, PRs, or logs. Methodology and
levers may be documented; measured numbers go only to the protected AMD
Confluence page. Reference tickets by bare ID.

(The validator wall-clock figures in §4.4 are test-harness tool timings, not
kernel performance results, and are in scope for this document.)

Other standing rules that bite here: never `ruff check --fix` emitter code
(`core`, `helpers`, `instances`, `library/kernels`) — the IR builder is
side-effecting and F841 autofix silently changes kernels. Relative paths only
under `platform/`; the validator must locate the LLVM tools and the enumeration
header via `shutil.which` / the detected ROCm root, never a hardcoded
`/opt/rocm`. Never report speed without correctness.
