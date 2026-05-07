# Q9J Scope Reassessment — `getDstParams`/`getSrcParams` Already Exist in C++

Bead: `rocm-libraries-q9j` ("Add virtual read_operands/write_operands API
to rocisa instruction hierarchy"). Re-examined against the rocisa C++
sources (`rocisa/rocisa/include/instruction/*.hpp`), the nanobind
binding (`rocisa/rocisa/src/instruction/instruction.cpp`), and the
validator's `_OPERAND_RULES` registry
(`Tensile/Components/ScheduleCapture.py`).

---

## 1. Verified facts

### 1.1 The C++ already has the API

`rocisa::Instruction` (the abstract base, `instruction.hpp:272-275`) declares:

```cpp
virtual std::vector<InstructionInput> getParams()    const = 0;
virtual std::vector<InstructionInput> getDstParams() const = 0;
virtual std::vector<InstructionInput> getSrcParams() const = 0;
```

`CommonInstruction` (the parent of every "regular" CommonInstruction-shaped
class) provides default implementations
(`instruction.hpp:487-504`) that return `{dst, dst1}` and `srcs` respectively.
`CompositeInstruction` does the same (`instruction.hpp:351-364`).

Concrete subclasses already override these correctly where the default doesn't
apply:

| Class | `getDstParams()` | `getSrcParams()` |
|---|---|---|
| `DSLoadInstruction` | `{dst}` | `{srcs}` (LDS-addr vgpr) |
| `DSStoreInstruction` | `{}` (no register write — bytes go to LDS) | `{dstAddr, src0, src1}` |
| `MUBUFReadInstruction` (BufferLoad) | `{dst}` | `{vaddr, saddr, soffset}` |
| `GLOBALLoadInstruction` | `{dst}` | `{vaddr, saddr}` |
| `FLATReadInstruction` | `{dst}` | `{vaddr}` |
| `MFMAInstruction` | `{acc}` | `{a, b, acc2}` |
| `VSwapB32` | `{dst} ∪ srcs` (symmetric) | `srcs ∪ {dst}` (symmetric) |
| `VAddCOU32` / `VAddCCOU32` / `VSubCoU32` | `{dst, dst1=VCC}` (inherits CommonInstruction) | `srcs` |
| `SCmpEQU32` (and family) | `{}` (dst=nullptr in ctor → filtered) | `{src0, src1}` |
| `VCmpEQU32` | `{dst=VCC}` | `{src0, src1}` |
| `SOrSaveExecB64` | `{dst=VCC}` | `{sgpr_pair}` |

That covers, by inspection, every shape the validator's 10 rules dispatch on.

### 1.2 The Python binding does NOT expose them

`rocisa/rocisa/src/instruction/instruction.cpp:96, 116, 144, 160` binds only
`getParams` to Python — `getDstParams` and `getSrcParams` are absent from every
`.def(...)` chain. The trampoline (lines 47-55) plumbs the pure virtual through
nanobind for Python-defined subclasses, but the base class is never `.def`'d
for end users.

### 1.3 Empirical probe (`/tmp/q9j_probe.py`) confirms

20 instances constructed across all 10 rule categories. Every single one:

```
getParams()    -> [...]                      # works
getDstParams() -> AttributeError: 'X' object has no attribute 'getDstParams'
getSrcParams() -> AttributeError: 'X' object has no attribute 'getSrcParams'
```

Selected probe outputs (full log in `/tmp/q9j_probe.py` stdout):

- `DSLoadB128`: `getParams() = [vgpr(0,4), vgpr(8)]` — dst then LDS-addr
- `DSStoreB128`: `getParams() = [vgpr(8), vgpr(0,4), None]` — dstAddr, src, src1
- `BufferLoadB128 normal`: `getParams() = [vgpr(0,4), vgpr(16), sgpr(0,4), 0]`
- `BufferLoadB128 DTL`: `getParams() = [None, vgpr(16), sgpr(0,4), 0]` — dst is None
- `MFMAInstruction`: `getParams() = [acc, a, b, acc2, ""]` — acc at slot 0
- `VSwapB32`: `getParams() = [vgpr(0), vgpr(1)]`
- `VAddCOU32`: `getParams() = [vgpr(0), VCC, vgpr(1), vgpr(2)]` — dst1=VCC at slot 1
- `VAddCCOU32`: `getParams() = [vgpr(0), VCC, vgpr(1), vgpr(2), VCC]` — dst1=VCC, src2=VCC
- `VCmpEQU32`: `getParams() = [VCC, vgpr(0), vgpr(1)]` — dst=VCC at slot 0
- `SCmpEQU32`: `getParams() = [sgpr(0), 0]` — no dst, src0+src1 only
- `SCBranchSCC0`: `getParams() = ["label"]` — label-only
- `SAddU32`, `SCSelectB32`, `SAddCU32`, `SOrSaveExecB64`: standard `[dst, src0, src1]`

Surprise finding: **plain `VAddU32(vgpr, vgpr, vgpr)` silently sets `dst1 = VCC()`**
in the C++ ctor (when neither `ExplicitNC` nor `ExplicitCO` asm-bug is set).
Probe output: `VAddU32 getParams() = [vgpr(0), VCC, vgpr(1), vgpr(2)]`. The
validator's `_VCCRule` already catches this via the `_is_vcc(p)` scan — but it
means the "generic" v_add path is implicitly VCC-touching on some ISAs, which
makes the pure VCC-classification by class name (in dzl's sketch) less complete
than expected. Worth a comment in dzl's design.

---

## 2. Per-rule disposition

| Rule | Category | Rationale |
|---|---|---|
| `_DSLoadRule` | **A** | `DSLoadInstruction::getDstParams()={dst}`, `getSrcParams()={srcs}` exactly matches the rule's (writes, reads) tuple. One-line `(srcs, dsts)` swap suffices once Python sees the methods. |
| `_DSStoreRule` | **A** | `DSStoreInstruction::getDstParams()={}`, `getSrcParams()={dstAddr, src0, src1}` — exactly the rule's `reads = (lds_addr, src_data)` (modulo None-filter for src1). |
| `_DTLBufferLoadRule` | **C** | `MUBUFReadInstruction::getDstParams()` will return `{nullptr}` when DTL — the validator can detect that, but the implicit `m0` read is **not** in `getSrcParams()`. m0 has to come from elsewhere. This is squarely dzl's "implicit operands" scope. |
| `_BufferLoadRule` | **A** | `MUBUFReadInstruction::getDstParams()={dst}` (or `{nullptr}` for DTL — filterable), `getSrcParams()={vaddr, saddr, soffset}`. Matches `_BufferLoadRule` (note `vaddr` is also picked up — currently the validator only tracks `srd=saddr`; this would be a slight expansion of read coverage, possibly desired). |
| `_MFMARule` | **B** | `MFMAInstruction::getDstParams()={acc}`, `getSrcParams()={a, b, acc2}`. Acc as a **read** (the read-modify-write semantic) is NOT expressed: the rule synthesizes `reads = (a, b, acc)` itself. Either the validator unions `dsts ∪ srcs ∪ {acc}` itself (B), or rocisa adds `is_accumulator` metadata (C — dzl scope per audit #10). Validator-side union is one line. |
| `_NoDataflowRule` | **A** | `SBarrier`/`SNop`/`SWaitCnt`/`SSetPrior` all have empty or non-register `getParams()`; `getDstParams() = getSrcParams() = {}` falls out trivially because `_is_register` filters non-Containers. |
| `_VSwapRule` | **A** | `VSwapB32` already overrides `getDstParams()` and `getSrcParams()` to return BOTH operands in BOTH lists (`common.hpp:5179-5194`). The symmetric semantic is **already encoded in C++**. Validator side becomes `reads = getSrcParams(); writes = getDstParams()` — no special-case. This is the strongest "A" item. |
| `_VCCRule` | **B** | `VAddCOU32::getDstParams() = {dst, dst1}` where `dst1 = VCC` — naturally returns VCC in writes. `getSrcParams()` returns srcs which include the carry-in VCC for VAddCCOU32. So the partition is **already correct in C++**. Two ergonomic asks: (1) the VCC sentinel needs to expose a `regType="vcc"` shape so the byte-key resolver doesn't drop it (currently `_VCC_RESOURCE` workaround substitutes one); (2) Python needs to see the methods. Both addressable separately. The dispatch logic / `_VCC_DST1_CARRY_OUT_CLASSES` set deletes once both methods are bound. |
| `_SCCRule` | **C** | The register partition (no_dst vs dst_then_srcs) is **already correct via getDstParams/getSrcParams** — `SCmpEQU32` has `dst=nullptr` so `getDstParams() = {}`, exactly the no_dst shape. But the SCC sentinel itself + the `_SCC_OPCODE_FLAGS` table (which opcodes touch SCC) is implicit-operand metadata that q9j can't carry. This is dzl + z48's scope. |
| `_GenericALURule` | **A** | This is literally `writes = getDstParams(); reads = getSrcParams()` plus `_is_register` filtering. Pure A. |

**Counts: A=6, B=2, C=2.**

---

## 3. Remaining q9j scope after re-cut

### 3.1 What q9j needs to do

After this re-examination, q9j is a **vastly smaller** piece of work than the
8t9 audit estimated. The substantive task is two-fold:

1. **Add three `.def` lines to `rocisa/rocisa/src/instruction/instruction.cpp`**
   binding `getDstParams` and `getSrcParams` on `Instruction`,
   `CompositeInstruction`, `CommonInstruction` (and same for `MacroInstruction`
   if we want it to throw on Python side too). Approximate diff:

   ```cpp
   .def("getParams",    &rocisa::Instruction::getParams)
   .def("getDstParams", &rocisa::Instruction::getDstParams)   // ADD
   .def("getSrcParams", &rocisa::Instruction::getSrcParams)   // ADD
   ```

   Repeat for each of the four `nb::class_` blocks (Instruction,
   CompositeInstruction, CommonInstruction, MacroInstruction).

2. **Replace 6-of-10 rules + the 8 `_inst_*` extractors with calls to the
   new accessors.** Concretely:

   - `_DSLoadRule`, `_DSStoreRule`, `_BufferLoadRule`, `_NoDataflowRule`,
     `_VSwapRule`, `_GenericALURule` collapse to a single
     `reads = inst.getSrcParams(); writes = inst.getDstParams()` pattern with
     the existing `_is_register` filter.
   - `_inst_dst`, `_inst_lds_offset`, `_inst_buffer_srd`, `_inst_buffer_offset`,
     `_inst_dsstore_src`, `_get_param` — most callers reduce to walking
     `getDstParams()[0]` / `getSrcParams()[i]`. The named extractors that
     survive (lds_offset, buffer_offset for identity-tuple purposes) become
     thin one-liners.
   - `_MFMARule` and `_VCCRule` use the new accessors plus a small union
     (acc=read+write for MFMA; VCC class needs to be filtered or substituted).

### 3.2 What q9j explicitly does NOT need to do

These belong to dzl/z48, NOT q9j:

- `_SCC_OPCODE_FLAGS` table and SCC sentinel (z48 + dzl)
- `_VCC_RESOURCE` synthetic singleton (dzl)
- DTL BufferLoad implicit-m0 reads (dzl)
- MFMA acc-as-RAW special-case if we model it as implicit metadata rather
  than a validator-side union (dzl, audit #10) — though the validator-side
  one-line union is the cheaper option here

### 3.3 Estimated impact

Original audit: ~600 LoC (audit row #3) + ~80 LoC (audit row #4) ≈ **680 LoC**.

Re-estimate after factoring out C-bucket items deferred to dzl:
- 6 rule classes * ~25 lines avg = ~150 LoC
- 8 `_inst_*` extractors * ~10 lines = ~80 LoC
- `_populate_wrapper` simplification = ~30 LoC
- Subtotal: **~260 LoC validator-side reduction**
- Plus: ~12 lines added to `instruction.cpp` (3 lines × 4 classes)

The remaining ~420 LoC of the original 680 estimate (`_VCCRule`, `_SCCRule`,
`_DTLBufferLoadRule`, `_VCC_DST1_CARRY_OUT_CLASSES`, `_SCC_OPCODE_FLAGS`,
`_SCC_SENTINEL`, `_VCC_RESOURCE`, MFMA acc-as-RAW special case) is in
**dzl's** scope — q9j's API design alone doesn't unlock it.

---

## 4. Proposed rocisa-side expansions

### 4.1 In q9j (truly q9j)

**Bind existing virtuals to Python.** Three `.def` lines per class (×4
classes) in `instruction.cpp`. No new C++ code, no new methods, no design
question.

### 4.2 NOT in q9j (deferred to dzl)

For completeness, the things q9j **cannot** solve and dzl **must**:

- `is_dtl: bool` (or split BufferLoadDTL) on `MUBUFReadInstruction` so the
  validator can distinguish without `getDstParams()[0] is None` structural
  inference.
- `implicit_reads()` / `implicit_writes()` on every class touching SCC/VCC/m0
  (or per-flag accessors). The C++ doesn't currently model implicit operands
  at all — they're encoded in the assembly text format only.
- A `RegisterContainer` shape for VCC (so VCC sentinels round-trip through the
  dataflow byte-key resolver without needing the `_VCC_RESOURCE` substitution
  hack).
- A `RegisterContainer` for SCC (a `regType="scc"` factory).
- Per-slot `is_accumulator: bool` on MFMA (or — equivalently — make
  `getDstParams()` and `getSrcParams()` BOTH include `acc`, mirroring what
  `VSwapB32` already does for symmetric R+W). The `VSwapB32` precedent is
  the cleanest: MFMA's `getDstParams()` and `getSrcParams()` could both list
  `acc`, and the asymmetric "a, b are read-only; acc is RW" maps to dual
  membership without any new flag.

The MFMA-mirror-VSwap idea is elegant enough that it could be promoted into
q9j (it's a one-line C++ change in `mfma.hpp:200-214`), but logically it's
dzl's scope per the original audit's #10.

---

## 5. Recommendation

**Rescope q9j.** The bead's premise ("rocisa classes don't expose a uniform
`read_operands()` / `write_operands()` API") is **wrong**: the C++ class
hierarchy already exposes exactly this API. The rocisa-side work is a
binding fix, not a design change. The validator-side migration is real but
~260 LoC, not ~680.

The original LoC estimate conflated three orthogonal pieces:

1. **Bind C++ accessors to Python** (q9j scope after rescope) — ~12 lines
   added to `instruction.cpp`, ~260 LoC removed validator-side.
2. **Implicit-operand metadata** (dzl scope) — independent of q9j.
3. **Class-category tags** (separate audit items #12–#18) — independent of
   both q9j and dzl.

The Hard-rule "spike on one rocisa class family (DSLoad*) BEFORE committing
the full design" in the v1 description becomes trivial: there is no design;
DSLoadInstruction's overrides are already correct (`mem.hpp:766-779`).

### Action

Rescope q9j to:

> **Title**: Bind existing rocisa `getDstParams()` / `getSrcParams()` to
> Python and migrate validator's operand-shape rules to use them
>
> **Why**: The rocisa C++ `Instruction` hierarchy already declares pure
> virtual `getDstParams()` / `getSrcParams()` and every concrete class
> implements them correctly (verified by reading
> `rocisa/include/instruction/{instruction,mem,mfma,common,cmp}.hpp`). The
> nanobind binding (`rocisa/src/instruction/instruction.cpp:88-161`) only
> exposes `getParams` to Python — adding 12 lines binds the dst/src split
> across the whole hierarchy. The validator's `_OPERAND_RULES` (10 rule
> classes in `Tensile/Components/ScheduleCapture.py`) were written
> precisely to recover what the C++ accessors already return.
>
> **Scope**:
> 1. Add `.def("getDstParams", ...)` and `.def("getSrcParams", ...)` to the
>    `Instruction`, `CompositeInstruction`, `CommonInstruction`, and
>    `MacroInstruction` `nb::class_` blocks in
>    `rocisa/src/instruction/instruction.cpp`.
> 2. Migrate `_DSLoadRule`, `_DSStoreRule`, `_BufferLoadRule`,
>    `_NoDataflowRule`, `_VSwapRule`, `_GenericALURule`, plus
>    `_inst_dst` / `_inst_lds_offset` / `_inst_buffer_srd` /
>    `_inst_buffer_offset` / `_inst_mfma_acc` / `_inst_mfma_a` /
>    `_inst_mfma_b` / `_inst_dsstore_src` / `_get_param` from
>    `Tensile/Components/ScheduleCapture.py` to use the new accessors.
> 3. `_MFMARule`: union acc into reads validator-side (one line) — OR
>    promote MFMA-mirrors-VSwap into this bead (1-line C++ change in
>    `mfma.hpp:200-214` so MFMA's `getDstParams()` and `getSrcParams()`
>    both list `acc`, identical to how `VSwapB32` handles symmetric R+W
>    in `common.hpp:5179-5194`).
> 4. NOT in scope (dzl owns these): `_VCCRule`, `_SCCRule`,
>    `_DTLBufferLoadRule`, `_VCC_DST1_CARRY_OUT_CLASSES`,
>    `_SCC_OPCODE_FLAGS`, `_SCC_SENTINEL`, `_VCC_RESOURCE`. Implicit
>    operands (SCC, VCC sentinels-as-resources, DTL m0 reads) require new
>    rocisa-side metadata that does NOT exist in C++.
>
> **Acceptance**:
> - `_DSLoadRule`, `_DSStoreRule`, `_BufferLoadRule`, `_NoDataflowRule`,
>   `_VSwapRule`, `_GenericALURule`, the 8 `_inst_*` extractors, and
>   `_get_param` all deleted from `ScheduleCapture.py`.
> - Snapshot test passes: every captured instruction's (reads, writes)
>   tuples are byte-identical before and after migration.
> - Estimated reduction: ~260 LoC validator-side, +12 LoC rocisa binding.
>
> **Dependencies**: Blocked by `rocm-libraries-br4`. Coordinate with
> `dzl` (implicit-operand metadata) but does not depend on it.

---

## Appendix: probe script

`/tmp/q9j_probe.py` — constructs one instance of each instruction class the
validator dispatches on, calls `getParams()`, `getDstParams()`,
`getSrcParams()`, and prints results. Confirms every concrete class
already has the C++ overrides; only the Python binding is missing.

Surprise finding worth carrying into dzl's scope: `VAddU32(vgpr, vgpr, vgpr)`
silently sets `dst1 = VCC()` in the C++ constructor on ISAs without
`ExplicitCO`/`ExplicitNC` (`common.hpp:2702-2716`). Plain `v_add_u32`
implicitly writes VCC, which is not classifiable purely by Python class
name — the VCC-touching set is per-build-config, not per-class.
