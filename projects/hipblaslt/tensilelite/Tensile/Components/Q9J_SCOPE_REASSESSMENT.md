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
| `_MFMARule` | **A** | `MFMAInstruction::getDstParams()={acc}`, `getSrcParams()={a, b, acc2}`. **Correction (2026-05-07):** the initial "B" categorization assumed `acc` was the read register and the rule needed a validator-side union to add it. That's wrong. Per `mfma.hpp:76,96`: `acc` is the destination accumulator (write); `acc2` is the **source** accumulator (read), defaulting to `acc` for the in-place RMW case (the dominant case). The assembly form is `v_mfma_* dst_acc, src_a, src_b, src_acc[, neg]` where `src_acc = acc2`. The current validator's `reads = (a, b, acc)` synthesis is correct only when `acc2 == acc` (in-place); for the out-of-place case (`acc2 != acc`) it's silently wrong because the actual hardware read is from `acc2`. Once the binding lands, the validator just uses `getSrcParams()` and gets the correct register whether in-place or out-of-place — strictly more correct than today, no special-case. Pure A. |
| `_NoDataflowRule` | **A** | `SBarrier`/`SNop`/`SWaitCnt`/`SSetPrior` all have empty or non-register `getParams()`; `getDstParams() = getSrcParams() = {}` falls out trivially because `_is_register` filters non-Containers. |
| `_VSwapRule` | **A** | `VSwapB32` already overrides `getDstParams()` and `getSrcParams()` to return BOTH operands in BOTH lists (`common.hpp:5179-5194`). The symmetric semantic is **already encoded in C++**. Validator side becomes `reads = getSrcParams(); writes = getDstParams()` — no special-case. This is the strongest "A" item. |
| `_VCCRule` | **REMOVED** | **(2026-05-07)** VCC dataflow tracking is permanently removed from the validator. Bead `rocm-libraries-uraq` deletes `_VCCRule` and supporting helpers (`_is_vcc`, `_vcc_resource`, `_VCC_RESOURCE`, `_VCC_DST1_CARRY_OUT_CLASSES`) entirely. q9j MUST land after `uraq` so the VCC machinery is already gone when q9j executes. No Category B / no migration / no replacement. See `CMSValidator_LIMITATIONS.md` §"VCC dataflow tracking is intentionally not provided". |
| `_SCCRule` | **C** | The register partition (no_dst vs dst_then_srcs) is **already correct via getDstParams/getSrcParams** — `SCmpEQU32` has `dst=nullptr` so `getDstParams() = {}`, exactly the no_dst shape. But the SCC sentinel itself + the `_SCC_OPCODE_FLAGS` table (which opcodes touch SCC) is implicit-operand metadata that q9j can't carry. This is dzl + z48's scope. |
| `_GenericALURule` | **A** | This is literally `writes = getDstParams(); reads = getSrcParams()` plus `_is_register` filtering. Pure A. |

**Counts: A=7, REMOVED=1 (`_VCCRule`), C=2.** (Updated 2026-05-07: `_MFMARule` reclassified A→A; `_VCCRule` removed entirely from q9j's scope per bead `uraq` — VCC tracking is permanently dropped from the validator. q9j is now A-only plus the C-deferral to dzl.)

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
   - `_MFMARule` uses the new accessors directly. **Correction
     (2026-05-07):** the original draft said "plus a small union
     (acc=read+write for MFMA)". That was wrong — `acc` is the destination
     accumulator (write); `acc2` (which defaults to `acc` for the in-place
     case but can differ) is the source accumulator (read). The validator
     just reads `getSrcParams()` and gets `{a, b, acc2}` — the correct read
     set, no union needed. See the §2 table for details.
   - `_VCCRule` is **deleted** by bead `uraq` before q9j runs. q9j does
     nothing with VCC. See `CMSValidator_LIMITATIONS.md` §"VCC dataflow
     tracking is intentionally not provided".

### 3.2 What q9j explicitly does NOT need to do

These belong to dzl/z48, NOT q9j:

- `_SCC_OPCODE_FLAGS` table and SCC sentinel (z48 + dzl)
- ~~`_VCC_RESOURCE` synthetic singleton~~ — REMOVED by `uraq` (not deferred to dzl; deleted permanently as part of dropping VCC dataflow tracking)
- DTL BufferLoad implicit-m0 reads (dzl)

**Removed (2026-05-07):** the original draft listed "MFMA acc-as-RAW
special-case" as deferred-to-dzl. That was based on a misreading. MFMA's
read accumulator is `acc2`, already returned by `getSrcParams()` — so the
read-modify-write semantic is already expressible at the rocisa level
without any new metadata or special-case. q9j subsumes it via Category A.

### 3.3 Estimated impact

Original audit: ~600 LoC (audit row #3) + ~80 LoC (audit row #4) ≈ **680 LoC**.

Re-estimate after factoring out C-bucket items deferred to dzl:
- 6 rule classes * ~25 lines avg = ~150 LoC
- 8 `_inst_*` extractors * ~10 lines = ~80 LoC
- `_populate_wrapper` simplification = ~30 LoC
- Subtotal: **~260 LoC validator-side reduction**
- Plus: ~12 lines added to `instruction.cpp` (3 lines × 4 classes)

The remaining ~420 LoC of the original 680 estimate splits two ways:
- **Removed permanently by `uraq`:** `_VCCRule`, `_VCC_DST1_CARRY_OUT_CLASSES`, `_VCC_RESOURCE`, `_is_vcc`, `_vcc_resource` (plus the two contract-pinning tests). VCC dataflow tracking is not part of the validator going forward; this is not deferred work.
- **Deferred to `dzl`:** `_SCCRule`, `_DTLBufferLoadRule`, `_SCC_OPCODE_FLAGS`, `_SCC_SENTINEL`. SCC + m0 implicit-operand work; q9j's API design alone doesn't unlock these.

(The original draft also listed "MFMA acc-as-RAW special case" here; per the 2026-05-07 correction it's q9j Category A, not deferred — see §2 table.)

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
- `implicit_reads()` / `implicit_writes()` on every class touching SCC/m0
  (or per-flag accessors). The C++ doesn't currently model implicit operands
  at all — they're encoded in the assembly text format only. **VCC was
  originally on this list; removed 2026-05-07 because VCC tracking is no
  longer a validator goal (bead `uraq`).**
- A `RegisterContainer` for SCC (a `regType="scc"` factory).
**Removed (2026-05-07):** the original draft proposed a "Per-slot
`is_accumulator: bool` on MFMA" or a "MFMA-mirrors-VSwap" 1-line C++
change to make `getDstParams()` and `getSrcParams()` both include `acc`.
Both proposals are unnecessary — `acc2` already serves as the source
accumulator and is already returned by `getSrcParams()`. The
read-modify-write semantic is already expressible. No new MFMA-side
metadata or C++ change needed.

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
>    `_NoDataflowRule`, `_VSwapRule`, `_MFMARule`, `_GenericALURule`,
>    plus `_inst_dst` / `_inst_lds_offset` / `_inst_buffer_srd` /
>    `_inst_buffer_offset` / `_inst_mfma_acc` / `_inst_mfma_a` /
>    `_inst_mfma_b` / `_inst_dsstore_src` / `_get_param` from
>    `Tensile/Components/ScheduleCapture.py` to use the new accessors.
>    `_MFMARule` reads `getSrcParams() = {a, b, acc2}` directly — no
>    union or special-case needed; `acc2` is the source accumulator
>    (defaults to `acc` for in-place RMW).
> 3. NOT in scope (dzl owns these): `_VCCRule`, `_SCCRule`,
>    `_DTLBufferLoadRule`, `_VCC_DST1_CARRY_OUT_CLASSES`,
>    `_SCC_OPCODE_FLAGS`, `_SCC_SENTINEL`, `_VCC_RESOURCE`. Implicit
>    operands (SCC, VCC sentinels-as-resources, DTL m0 reads) require new
>    rocisa-side metadata that does NOT exist in C++.
>
> **Acceptance**:
> - `_DSLoadRule`, `_DSStoreRule`, `_BufferLoadRule`, `_NoDataflowRule`,
>   `_VSwapRule`, `_MFMARule`, `_GenericALURule`, the 8 `_inst_*`
>   extractors, and `_get_param` all deleted from `ScheduleCapture.py`.
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
