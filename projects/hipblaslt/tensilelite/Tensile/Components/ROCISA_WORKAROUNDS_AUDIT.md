# Rocisa-Deficiency Workarounds in the Validator — Audit

Bead: `rocm-libraries-8t9`. Read-only audit. Cites by name (function /
class / constant), not line number — the validator surface is mid-
migration via the `br4` epic and line numbers will drift.

The validator (currently spread across `Tensile/Components/CMSValidator.py`
and `Tensile/Components/ScheduleCapture.py`, consolidating into
`CMSValidator.py` per `br4`) has accreted a Python-side layer of
wrappers, shim classes, and lookup tables that exist primarily because
the underlying rocisa instructions don't carry the metadata the
validator needs. Each entry below is a workaround for a specific
rocisa-side deficiency.

---

## Status update (2026-05-07)

Subsequent investigations re-categorized several rows in this audit.
The table and prose below are preserved as the original artifact, but
several entries no longer reflect current scope:

- **Row #3 (`_OPERAND_RULES`):** estimated ~600 LoC. Reassessed in
  `Q9J_SCOPE_REASSESSMENT.md` (commit `c12dd242e7`). The rocisa
  `Instruction` C++ hierarchy already exposes `getDstParams()` /
  `getSrcParams()` as pure virtuals (`instruction.hpp:272-275`); only
  the nanobind binding is missing. q9j is now ~12 lines C++ binding
  + ~260 LoC validator-side migration, not new rocisa API design.
- **Row #7 (`_VCC_RESOURCE` / `_vcc_resource`):** REMOVED. Bead `uraq`
  deleted these entirely (commit `7cd1e994e1`/merged as `~fd7de8ccfe`).
  VCC dataflow tracking is permanently dropped from the validator's
  scope — no rocisa-side `vcc_resource()` factory will be added.
- **Row #8 (`_VCCRule` + `_VCC_DST1_CARRY_OUT_CLASSES`):** REMOVED
  by `uraq`. No `reads_vcc` / `writes_vcc` will be added to rocisa.
- **Row #19 (`_VCC_DST1_CARRY_OUT_CLASSES`):** REMOVED by `uraq`.
  No per-slot `OperandRole` work for VCC.
- **MFMA accumulator special-casing** (referenced under Category C):
  RECLASSIFIED. Per Q9J_SCOPE_REASSESSMENT.md §2 (commit `14efe6225f`),
  MFMA's `acc` is the destination accumulator and `acc2` is the
  source accumulator (defaults to `acc` for in-place RMW). Both are
  already in the C++ `getDstParams()` / `getSrcParams()` partition.
  No new MFMA-side metadata needed; q9j Category A handles it.
- **Category C (SCC + m0 only, no longer SCC/VCC/m0/acc):** SCC and
  m0 confirmed to genuinely need new C++ work
  (`DZL_SCOPE_REASSESSMENT.md`, commit `2e76bffa5a`; source-level
  confirmation that `CommonInstruction` has only `dst`/`dst1`/`srcs`
  fields and no field for SCC). dzl scope is unchanged for the SCC +
  m0 piece; VCC and acc are out.

In short: the audit's broad strokes hold (validator workarounds exist
because of rocisa-side gaps), but four of the listed workarounds are
either resolved by binding existing C++ surface (Row #3 → q9j) or
removed entirely without rocisa-side replacement (Rows #7/#8/#19 →
uraq), and one Category-C item has been reclassified out (MFMA acc).
The remaining live rows are #1, #2, #4, #5 (SCC subset), #6 (covered
by dzl jointly), #9 (m0/DTL), #10–18, and #20 — see the docs cited
above for the current canonical reframing of each.

---

## 1. Inventory

| # | Workaround | Lives in | Root cause | Rocisa-side fix sketch | Est. LoC saved |
|---|---|---|---|---|---|
| 1 | `WrappedInstruction` proxy class | `ScheduleCapture.py` | Rocisa nanobind classes refuse `setattr` (no `__dict__`) and can't be weakref'd; per-instance `(reads, writes)` metadata has nowhere to live | Either expose mutable attribute slots on rocisa instructions, OR (preferred) compute reads/writes natively from the rocisa class so the per-instance cache becomes optional | ~60 (the class + deepcopy/copy plumbing) |
| 2 | `TaggedInstruction(wrapped, category, slot)` | `ScheduleCapture.py` | Rocisa instructions carry no `category` attribute distinguishing scheduler role (LRA0 vs PackB3 vs MFMA-as-Pack) | Allow rocisa instructions to be tagged at construction with an opaque `category` enum / string field readable from Python | ~20 (the dataclass; downstream consumers can read `inst.category` directly) |
| 3 | `_OPERAND_RULES` dispatch registry (`_DSLoadRule`, `_DSStoreRule`, `_DTLBufferLoadRule`, `_BufferLoadRule`, `_MFMARule`, `_NoDataflowRule`, `_VSwapRule`, `_VCCRule`, `_SCCRule`, `_GenericALURule`) and `_populate_wrapper` | `ScheduleCapture.py` | Rocisa classes don't expose a uniform `read_operands()` / `write_operands()` API; each shape is recovered by reading positional `getParams()` slots in a class-specific way | Add virtual `reads()` / `writes()` methods to the rocisa instruction class hierarchy returning the right register containers; the entire rule registry collapses into one call | ~600 (10 rule classes + dispatch loop + per-shape `_inst_*` extractors) |
| 4 | `_inst_dst`, `_inst_lds_offset`, `_inst_buffer_srd`, `_inst_buffer_offset`, `_inst_mfma_acc`, `_inst_mfma_a`, `_inst_mfma_b`, `_inst_dsstore_src`, `_get_param` | `ScheduleCapture.py` | Rocisa instructions don't expose constructor args as named Python attributes (`hasattr(inst, "dst")` returns False even for non-DTL BufferLoads); validator must read them positionally via `getParams()` | Expose named accessors (`inst.dst`, `inst.src`, `inst.acc`, `inst.lds_addr`, `inst.srd`, `inst.lds_offset`, `inst.imm_offset`) on the relevant rocisa classes | ~80 (eight extractors plus the comments explaining the attribute-vs-getParams duality) |
| 5 | `_SCC_OPCODE_FLAGS` table (~50 entries) and `_SCCRule` | `ScheduleCapture.py` | Rocisa SALU classes don't carry `reads_scc` / `writes_scc` / `operand_shape` metadata; validator hand-maintains a parallel restatement | Add `static constexpr OperandShape operand_shape; bool reads_scc; bool writes_scc` to every SALU rocisa class; `_SCC_OPCODE_FLAGS` deletes (covered by sibling bead `z48`) | ~120 (the table + the rule + the SCmp* "no_dst" footgun documentation in `_GenericALURule`) |
| 6 | `_SCC_SENTINEL` + `_get_scc_sentinel` (synthetic SCC RegisterContainer singleton) | `ScheduleCapture.py` | Rocisa has no `RegisterContainer`-shaped representation of SCC; SCC is invisible to the dataflow graph unless the validator manufactures a stand-in | Add an `scc_resource()` factory to rocisa returning a singleton container of a new `regType="scc"` (single-bit, regIdx=0, regNum=1); validator imports it instead of synthesizing | ~20 (the singleton, lazy-init, and the `vgpr(0)` mutating constructor explained in the docstring) |
| 7 | `_VCC_RESOURCE` + `_vcc_resource` (synthetic VCC RegisterContainer singleton) | `ScheduleCapture.py` | Rocisa exposes VCC as an opaque sentinel class with no `regType`/`regIdx`; the byte-key resolver filters it out of every rule's reads/writes | Add a `vcc_resource()` factory to rocisa returning a singleton `RegisterContainer("vcc", ..., 0, 2)`; the VCC sentinel class itself should expose its container form | ~15 |
| 8 | `_VCCRule` + `_VCC_DST1_CARRY_OUT_CLASSES` set | `ScheduleCapture.py` | Rocisa classes don't expose implicit-VCC operand metadata (neither which slot is the carry-out dst1, nor that the class touches VCC at all) | Add `reads_vcc` / `writes_vcc` and a slot-position descriptor (or virtual `read_operands()` / `write_operands()` from #3) to VAddCO/VAddCCO/VSubCo/VCmp* and SOrSaveExec*/SAndSaveExec* classes | ~80 (the rule class + the carry-out class set + the slot-position logic) |
| 9 | `_VSwapRule` (symmetric R+W on both operands) | `ScheduleCapture.py` | Rocisa doesn't model "swap" as a first-class operation kind; VSwapB32 is a generic CommonInstruction whose semantic (both operands read AND written) is invisible | Add `OperationKind::Swap` (or equivalent `is_swap`/symmetric-operand flag) on the rocisa VSwap* class hierarchy; the rule reduces to noticing the flag | ~50 (the rule + the long docstring justifying why asymmetric modeling drops one of the four edge classes) |
| 10 | `_MFMARule` accumulator special-casing (`_inst_mfma_acc` plus the read+write treatment of acc) | `ScheduleCapture.py` | Rocisa MFMA classes don't expose acc semantics in a queryable way; the read-modify-write nature of the accumulator is implicit | Add an `is_accumulator: bool` per-operand-slot flag (or a typed return from the proposed virtual `read_operands()` / `write_operands()`) on MFMA classes so the acc slot is automatically published as both read and write | ~20 |
| 11 | `_DTLBufferLoadRule` + `_is_dtl_buffer_load` discriminator | `ScheduleCapture.py` | Rocisa BufferLoad classes don't distinguish DTL-mode (dst=None, writes LDS, implicitly reads m0) from normal loads — the validator infers it structurally from `_inst_dst(inst) is None` | Add `is_dtl: bool` (or split `BufferLoadDTL` as a distinct rocisa class) plus `implicit_reads = [m0]` metadata; the discriminator + dedicated rule both collapse into the dispatch from #3 | ~40 |
| 12 | `_LR_CLASS_NAMES` / `_LW_CLASS_NAMES` / `_GR_CLASS_NAMES` / `_MFMA_CLASS_NAMES` / `_SWAIT_CLASS_NAMES` / `_SBARRIER_CLASS_NAMES` / `_SNOP_CLASS_NAMES` / `_SSETPRIO_CLASS_NAMES` / `_CVT_PACK_CLASS_NAMES` / `_MIDDLE_PACK_CLASS_NAMES` (10 hand-maintained class-name sets) | `ScheduleCapture.py` | Rocisa classes don't carry a semantic-category tag (`category=LR/LW/GR/MFMA/SWAIT/...`); validator discriminates by `type(inst).__name__ in {...}` | Add a `category` (or `kind`) class-level attribute on each rocisa instruction class returning a member of an `InstructionCategory` enum; the sets all collapse to `inst.category == InstructionCategory.LR` etc. | ~150 (10 sets + the documentation for each + the maintenance-burden comments) |
| 13 | `_is_lr` / `_is_lw` / `_is_gr` / `_is_mfma` / `_is_swait` / `_is_sbarrier` / `_is_snop` / `_is_ssetprio` / `_is_middle_pack` / `_is_cvt_pack` discriminators | `ScheduleCapture.py` | Same root cause as #12 — class-name-set membership predicates compensating for missing `category` on rocisa classes | Same as #12 — once `inst.category` exists, the discriminators reduce to `inst.category == X` (or delete entirely; callers can compare directly) | ~30 (the function bodies; consumers shrink too) |
| 14 | `_SMEM_CLASS_NAMES` / `_FLAT_CLASS_NAMES` / `_VECTOR_STORE_CLASS_NAMES` (capture-builder guard sets) | `ScheduleCapture.py` | Same root cause as #12 — the finalize() guards are also class-name-set membership predicates because rocisa has no `category` | Same fix as #12 unblocks these too — guards become `inst.category == InstructionCategory.SMEM` etc. | ~30 |
| 15 | `_class_tag(inst)` — produces the LR/LW/GR/MFMA/SWAIT/SBARRIER/SNOP/SSETPRIO string tag for graph identity | `CMSValidator.py` | Same root cause as #12 — chained `_is_*` calls because the category isn't on the rocisa class | Once `inst.category` exists, `_class_tag` becomes `return inst.category.name` (or deletes; callers use the enum directly) | ~30 |
| 16 | `_class_tag_from_category(category, inst)` — maps validator-side TaggedInstruction.category to the same tag scheme, with fallback to `_class_tag` | `CMSValidator.py` | Compounds #2 and #12: TaggedInstruction.category exists because rocisa instructions can't be tagged; the cross-mapping exists because the same rocisa class can carry different scheduler roles (a `MFMAInstruction` can be category="MFMA" OR "PackA3") | If #2 is fixed (category lives on the rocisa instance), `_class_tag_from_category` becomes a single `inst.category.tag()` call. If only #12 is fixed (category is a class-level constant, not per-instance), this function survives because the per-instance scheduler role is orthogonal | ~50 |
| 17 | `PACK_TYPE_MAP` (10 entries: rocisa class -> (ValidatorInstruction subclass, asm label)) | `CMSValidator.py` | Rocisa classes have no self-describing tag attribute mapping them to their validator-dataclass kind / human-readable assembly mnemonic | Add `validator_kind: str` and `asm_label: str` (or similar) class-level metadata on the rocisa classes; `PACK_TYPE_MAP` deletes, `resolve_pack_type` becomes `(getattr(cls, "validator_kind"), getattr(cls, "asm_label"))` | ~30 (the table + the exact-then-isinstance two-pass lookup) |
| 18 | `GR_TYPE_MAP` (5 entries distinguishing actual GR loads from m0 pointer ops) | `CMSValidator.py` | Same as #17 — needs a `is_gr_load: bool` (or finer category) on rocisa | Same as #17; `GR_TYPE_MAP` deletes, `is_gr_load` queries the rocisa class directly | ~20 |
| 19 | `_VCC_DST1_CARRY_OUT_CLASSES` set (3 entries: VAddCOU32/VAddCCOU32/VSubCoU32) | `ScheduleCapture.py` | Rocisa carry-chain classes don't expose "the second slot is a write (carry-out dst1), not a read" structurally | Add a per-slot `OperandRole` (read/write/dst/dst1) descriptor on the rocisa class; this set deletes (covered jointly with #8) | ~10 |
| 20 | `_NUMERIC_REG_FACTORIES` lazy-init lookup table (5 factory entries: v/s/acc/m/vcc/scc) | `ScheduleCapture.py` | Rocisa doesn't expose a uniform "factory by regType character" entry point; validator hand-curates one for the byte-key resolver | Add a `RegisterContainer.factory_for(reg_type)` (or `make_register(reg_type, idx, count)`) helper to rocisa; the table deletes | ~25 (the dict + lazy-init dance + comments) |

**Subtotal:** ~20 distinct workarounds. Estimated LoC reclaimable: **~1,400–1,500**, plus the indirect simplifications in the consumers that no longer have to thread "category" alongside "instruction" through every call.

---

## 2. Classification by root cause

### Category A — Rocisa instructions have no `category` / `kind` tag attribute

Affects #2, #12, #13, #14, #15, #16, #17, #18.

By far the largest category, by both LoC and by surface area. Most of the
class-name-set + discriminator + tag-mapping machinery exists solely to
recover information that could trivially be a `static constexpr
InstructionCategory category` (or per-class `validator_kind` /
`asm_label`) on the rocisa class. The user's framing — "every one of
these is a parallel Python-side restatement" — applies most strongly
here.

### Category B — Rocisa instructions don't expose a uniform `read_operands()` / `write_operands()` API

Affects #3, #4, #10, #11.

The entire `_OPERAND_RULES` registry plus the `_inst_*` positional
extractors plus the per-shape rule classes exist because each rocisa
class structures its constructor params differently and the validator
has to reverse-engineer each shape. A virtual `reads()` / `writes()`
returning typed register containers from the rocisa side would collapse
the registry to a single dispatch.

### Category C — Rocisa classes don't expose implicit-operand metadata (SCC / VCC / m0 / acc)

Affects #5, #6, #7, #8, #10, #11, #19.

Implicit operands (SCC bit on most SOPx ops, VCC carry on V*CO ops, m0
on DTL BufferLoads, acc as both read AND write on MFMA) are
ISA-intrinsic but invisible to the rocisa Python binding. Validator
either hand-tabulates which class touches which (#5, #19) or invents
synthetic resources to stand in (#6, #7) or specializes the rule (#8,
#10, #11). Pushing implicit-operand metadata into the rocisa class
collapses all of these.

### Category D — Rocisa doesn't model operation-shape semantics (swap, no-dst-compare, etc.)

Affects #9, the "no_dst" shape inside #5, the SCmp* footgun
documentation in `_GenericALURule`.

`v_swap_b32` is symmetric R+W; `s_cmp_eq_u32` and `s_cbranch_scc1` have
no destination at all. Rocisa offers only one CommonInstruction shape
(asymmetric, `params[0]` is dst); each exception requires a hand-coded
override. An `OperationShape` enum (or per-class flag) on the rocisa
side eliminates them.

### Category E — Rocisa nanobind classes refuse Python attribute attachment

Affects #1.

Distinct from A–D: this is an attribute-attachment limitation of the
nanobind binding, not missing data. `WrappedInstruction` exists to give
the validator anywhere to put per-instance computed metadata. The fix is
either binding-level (allow `__dict__` / `__slots__` on the Python view)
or design-level (make the metadata derivable from the class so per-
instance caching is unnecessary; if every rocisa class implements
`reads()` / `writes()` natively, the cache becomes a memoization
optimization, not a correctness requirement).

### Category F — Rocisa has no unified register / resource factory entry point

Affects #20.

Tangential to the others; this is a small surface but worth tagging
because the lookup table exists for the same "no native API" reason.

---

## 3. Remediation roadmap

The categories interact: fixing A unblocks the largest LoC reduction
quickly; fixing B is the highest-leverage architectural change because
it cuts the entire rule registry; C and D are smaller but eliminate
silent-miss bug classes; E is a foundation for full-fidelity validator
decoupling; F is opportunistic.

Recommended ordering — pick the dependency root first, work down:

### Step 1 — Land Category A (`category` / `kind` attribute on rocisa)

Highest immediate ROI. Touches roughly 30+ rocisa classes (LR/LW/GR/MFMA/
SWait/SBarrier/SNop/SSetPrior families + Pack/CVT-pack/middle-pack /
swap-pack / MFMA-pack subdivisions + SMEM / FLAT / vector-store guard
classes). Eliminates ~10 class-name sets, 10 `_is_*` discriminators, two
tag mappers, and two PACK/GR maps. Required preceding work: settle the
canonical category enum (probably `InstructionCategory` with
LR/LW/GR/MFMA/SWAIT/SBARRIER/SNOP/SSETPRIO/PACK/SMEM/FLAT/STORE).

Unblocks: #2, #12–#18. Does not by itself simplify the operand-rule
machinery — that's #3.

### Step 2 — Land Category B (virtual `read_operands()` / `write_operands()` on rocisa)

Highest architectural ROI. Touches every CommonInstruction-shaped rocisa
class (~100+). The reward: the entire `_OPERAND_RULES` registry (10 rule
classes) collapses into `wrapper.reads, wrapper.writes = inst.reads(),
inst.writes()`. Side benefit: the named-accessor extractors (#4) become
unnecessary because the typed return values name themselves.

Sequence: depends on Category C decisions. If implicit operands also
flow through `read_operands()` / `write_operands()` (the cleanest
design), then C is folded into B and the SCC/VCC sentinels become
ordinary register containers returned from the rocisa side.

Unblocks: #3, #4, #10, #11, plus most of C.

### Step 3 — Land Category C (implicit-operand metadata)

Either folded into Step 2 (if `read_operands()` returns implicit
operands too — preferred) or as a parallel `implicit_reads` /
`implicit_writes` accessor. Either way, push the SCC sentinel and VCC
sentinel containers down to rocisa as `scc_resource()` /
`vcc_resource()` factories so the validator imports them rather than
synthesizing.

Unblocks: #5, #6, #7, #8, #19. Also retires the `_SCC_OPCODE_FLAGS`
table (this is bead `z48`'s scope).

### Step 4 — Land Category D (operation-shape metadata)

Smallest scope, very high silent-miss-prevention value. Add an
`OperationShape` enum (or `is_swap` / `has_no_dst` flags) on
VSwap*/SCmp*/SCBranchSCC*/SBitcmp* rocisa classes. Most of this is
already implied by C (the SCC opcode flags include "no_dst"); the swap
case is the standalone item.

Unblocks: #9.

### Step 5 — Land Category E (mutable Python attributes on rocisa)

Optional once B is done. If `read_operands()` / `write_operands()` are
fast enough to call on demand, the per-instance cache (`WrappedInstruction.reads`
/ `.writes`) becomes a pure memoization optimization. If we still want
to cache, fix the nanobind binding to allow `__slots__` or `__dict__`
attachment.

Unblocks: #1 — `WrappedInstruction` reduces to a thin alias or deletes
entirely.

### Step 6 — Land Category F (uniform register factory)

Cosmetic. Add `RegisterContainer.factory_for(reg_type)` to rocisa once
the SCC / VCC containers from Step 3 land. `_NUMERIC_REG_FACTORIES`
deletes.

Unblocks: #20.

### Interactions with adjacent beads

- `br4` epic (validator consolidation into one file): should land first
  so the migration target is stable. All Step-1–6 work happens AFTER
  `br4` closes.
- `c70` (Register abstraction class): if the validator adopts a
  homegrown `Register` abstraction for symbolic / numeric / mixed
  comparison, the rocisa-side `factory_for` from Step 6 may not be
  needed — the abstraction owns the construction surface. Resolve `c70`
  before Step 6.
- `nn0` (methods-on-classes refactor): aligned with Steps 1–4 in spirit.
  If the validator adopts the same "methods on classes" idiom on the
  Python side, it makes the rocisa-side push easier to argue: the
  validator already wants to call `inst.reads()`; the question is just
  where the implementation lives. Coordinate scope.
- `z48` (SCC opcode flags specifically): already planned, is a strict
  subset of Step 3.

---

## 4. Risk-prioritized recommendations

### Tackle first (high leverage / low controversy)

- **Step 1, Category A.** Largest LoC reduction. Requires only a class-
  level constant per rocisa class — no method-resolution-order surprises,
  no virtual-dispatch ambiguity, low risk of regression. Pure additive
  change on the rocisa side; the validator can adopt incrementally
  (read the new attribute when present, fall back to the existing class-
  name set when absent), so the migration is reversible at every step.
- **Step 4, Category D.** Tiny scope, prevents an entire class of silent-
  miss bugs (the `_GenericALURule` "DANGER ZONE" docstring spells out
  the failure mode). Worth doing as soon as the operand-shape enum is
  agreed on.

### Tackle next (high leverage / requires design discussion)

- **Step 2, Category B.** Architectural payoff is enormous (~600 LoC
  saved, single-source-of-truth for operand shape). But: needs design
  agreement on the return type — does `read_operands()` return
  `vector<RegisterContainer*>`? Does it include implicit operands?
  Does the validator's symbolic / numeric register-name handling
  survive the round-trip? Recommend a small spike on one rocisa class
  family (DSLoad*) before committing.
- **Step 3, Category C.** Best done jointly with Step 2 — the design
  question "do implicit operands flow through `read_operands()`?" forces
  the answer.

### Tackle last (lower leverage, preserves current behavior)

- **Step 5, Category E** — only worth doing if profiling shows the
  per-instance caching matters; otherwise the wrapper deletes naturally
  once Step 2 lands.
- **Step 6, Category F** — cosmetic. One small lookup table.

### Leave alone (do NOT chase)

- **`_FakeLR` / `_FakeLW` / `_FakeGR` / `_FakeMFMA` / `_FakeSWait` /
  `_FakeSBarrier` / `_FakeSNop` / `_FakeCVTPack`** test-only stand-ins
  in `_LR_CLASS_NAMES` etc. These exist for test isolation, NOT for
  rocisa-deficiency reasons. Bead `904` already addresses the test-
  fixture restructure for `_FakeLW`. The other `_Fake*` entries can
  follow the same pattern in their own beads. Out of scope here.
- **`MemoryRegion`** — not a workaround. It models LDS / scratch / GDS
  byte regions as first-class resources because they don't fit the
  RegisterContainer mold. The right home for this IS the validator
  (it's about validator-internal modelling of the dataflow graph
  resource set, not about restating something rocisa knows). Leave.
- **`_canonical_render`** + `_COMMENT_STRIP_RE` — a normalization layer
  for cross-graph identity comparison. Not a rocisa workaround; a
  validator-policy choice about identity equivalence. Leave.

### Surprises uncovered during the audit

The bead description listed the categories thoroughly, but a few
cross-cutting items emerged:

- `_NUMERIC_REG_FACTORIES` (#20) wasn't in the starting list but is
  cleanly the same shape — a Python-side restatement of construction
  knowledge that lives intrinsically on the rocisa side.
- `_VCC_DST1_CARRY_OUT_CLASSES` (#19) is its own micro-table separate
  from the main `_VCCRule`; it's the slot-position metadata that
  Category C / D fix would absorb.
- The `_FakeLW`-style test stand-ins are a scaffolding inside the LR /
  LW / GR / MFMA / SWAIT / SBARRIER / SNOP class-name sets themselves —
  the production sets contain `_Fake*` entries. Good rocisa-side
  metadata makes the production paths cleaner, but the test fixtures
  need their own home (out of scope; bead `904` and follow-ups).
- `CaptureSMEMError` / `CaptureFlatError` / `CaptureStoreError` exist
  as named exceptions specifically because the discriminator sets
  (`_SMEM_CLASS_NAMES` / `_FLAT_CLASS_NAMES` /
  `_VECTOR_STORE_CLASS_NAMES`) are class-name-string lookups — an
  `inst.category == InstructionCategory.SMEM` query would let the
  guards stay in the validator without needing the parallel class-name
  table. Bundled into Category A.

The total addressable workaround surface is larger than the bead's
opening list, but every additional finding belongs in one of the four
root-cause categories already named — no NEW root cause emerged. The
classification is stable.
