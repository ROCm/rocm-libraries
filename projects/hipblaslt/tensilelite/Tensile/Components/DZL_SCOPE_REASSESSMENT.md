# DZL Scope Reassessment — SCC and m0 Implicit-Operand Tracking

Bead: `rocm-libraries-dzl` ("Push implicit-operand metadata (SCC/m0) onto
rocisa classes"). Re-examined against the rocisa C++ headers
(`rocisa/include/instruction/{instruction,common,cmp,branch,mem}.hpp`),
the nanobind binding sources (`rocisa/src/{container.cpp,
instruction/{instruction,common,mem,cmp}.cpp}`), and the validator's
`_SCC_OPCODE_FLAGS` / `_SCCRule` / `_SCC_SENTINEL` / `_DTLBufferLoadRule`
machinery (`Tensile/Components/ScheduleCapture.py`, validator branch).

The investigation parallels the q9j reassessment. The result here is the
opposite: dzl is **not** a binding gap. It is genuinely C-category work.
SCC and m0 implicit-operand metadata are not modeled in rocisa at any
level — not in C++, not in the bindings, not in any base class, not in
any flag, and not in any factory.

---

## 1. Verified facts

### 1.1 SCC: zero rocisa-side modeling

Probe (`/tmp/scc_m0_probe.py`) instantiated the canonical SCC-touching
classes and dumped every public attribute. Findings:

**(Source-level confirmation, 2026-05-07.)** The Python probe could
only inspect bound surface; to rule out the q9j-style "the C++ side
has the API, only the binding is missing" scenario, we read the
rocisa C++ source for `CommonInstruction` (the parent of every
SCC-touching SOPx class) at `rocisa/include/instruction/instruction.hpp:382`.

`CommonInstruction` has exactly three operand-bearing fields:

```cpp
struct CommonInstruction : public Instruction {
    std::shared_ptr<Container>    dst;
    std::shared_ptr<Container>    dst1;   // VCC carry-out lives here for VAddCOU32
    std::vector<InstructionInput> srcs;
    // ... no scc field anywhere ...

    std::vector<InstructionInput> getDstParams() const override {
        std::vector<InstructionInput> dsts;
        if (dst)  dsts.push_back(dst);
        if (dst1) dsts.push_back(dst1);
        return dsts;
    }
    std::vector<InstructionInput> getSrcParams() const override {
        return srcs;
    }
};
```

For `SCmpEQU32`: ctor passes `dst=nullptr, srcs=[src0, src1]`. So
`getDstParams() → {}` and `getSrcParams() → [src0, src1]`. **SCC
appears in NEITHER.** For `SAddU32` (silently writes SCC):
`getDstParams() → {sgpr}` and `getSrcParams() → [src0, src1]`. Still
no SCC.

This confirms the Python-side findings below at the source level: no
binding gap; SCC is not modeled anywhere in the rocisa data model.
Contrast with VCC, which IS modeled — `VAddCOU32`'s ctor explicitly
sets `dst1 = VCC()`, so VCC shows up in `getDstParams()` natively.
There's no equivalent path for SCC because there's no field to store
it in.

**No SCC base class.** Every SCC-touching SOP1/SOP2/SOPC/SOPK class
inherits the same MRO as every other CommonInstruction:

```
SCmpEQU32.__mro__   = [SCmpEQU32,   CommonInstruction, Instruction, Item, object]
SAddU32.__mro__     = [SAddU32,     CommonInstruction, Instruction, Item, object]
SAddCU32.__mro__    = [SAddCU32,    CommonInstruction, Instruction, Item, object]
SCSelectB32.__mro__ = [SCSelectB32, CommonInstruction, Instruction, Item, object]
SCBranchSCC0.__mro__= [SCBranchSCC0, BranchInstruction, Instruction, Item, object]
```

There is no `SCCWritingInstruction`, no `SCCReadingInstruction`, no
mixin. SCmpEQU32 (writes SCC, no register dst) and SAddU32 (writes SCC
+ has sgpr dst) are differentiated only by their `dst=nullptr`-vs-`dst=sgpr`
ctor argument and their `setInst()` mnemonic string. The class hierarchy
itself carries zero SCC information.

**No SCC attribute on instances.** `dir()` on every SCC-touching instance
is identical and contains no SCC hint:

```
['archCaps', 'asmBugs', 'asmCaps', 'comment', 'dpp', 'dst', 'getMemToken',
 'getParams', 'instType', 'kernel', 'memToken', 'name', 'parent',
 'prettyPrint', 'regCaps', 'sdwa', 'setInlineAsm', 'setMemToken',
 'setSrc', 'srcs', 'vgprIdx', 'vgprMsb', 'vop3']
```

None of `reads_scc`, `writes_scc`, `scc_read`, `implicit_reads`,
`implicit_writes`, `is_scc_writer`, `operand_shape`, `category`, `kind`
exist on any instance. This was confirmed by the probe's
`hasattr` sweep.

**No SCC RegisterContainer.** `dir(rocisa.container)` lists every public
symbol:

```
['Container', 'ContinuousRegister', 'DPPModifiers', 'DSModifiers',
 'EXEC', 'EXECHI', 'EXECLO', 'FLATModifiers', 'GLOBALModifiers',
 'HWRegContainer', 'Holder', 'HolderContainer', 'MUBUFModifiers',
 'MemTokenData', 'RegName', 'RegisterContainer', 'SDWAModifiers',
 'SMEMModifiers', 'True16Modifiers', 'VCC', 'VOP3PModifiers',
 'accvgpr', 'mgpr', 'replaceHolder', 'sgpr', 'vgpr']
```

There is no `scc`, no `SCC`, no `scc_resource`, no `regType="scc"` factory.
Grep across `rocisa/src/**` for the string `"scc"` returns zero hits.
For comparison, the rocisa C++ source does mention SCC but only in
codegen comments (`common.hpp:1068, 1106` — `// if (SCC == 1) { ... }`),
not as a modeled register or property.

**Not even VCC is a `RegisterContainer`.** `dir(VCC())` is `[]` — VCC is
an opaque sentinel class with no `regType`, no `regIdx`, no anything.
This is why the validator's pre-uraq `_VCC_RESOURCE` had to be
synthesized; the same shape would apply to SCC if rocisa-as-is were
used. There is no precedent in rocisa for "implicit-operand register
modeled as a real container."

### 1.2 m0: register exists, DTL discriminator does not

**`mgpr(0)` works as a real RegisterContainer.** The probe confirmed:

```
rocisa.container.mgpr(0) -> RegisterContainer regType='m' regIdx=0
dir: ['getCompleteRegName', 'getRegNameWithType', 'regIdx', 'regName',
      'regNum', 'regType', 'replaceRegName', 'splitRegContainer', ...]
```

So unlike SCC, m0 is already a first-class RegisterContainer. The
validator's `_DTLBufferLoadRule` already constructs `mgpr(0)` and uses
it directly — that part needs no rocisa change.

**No DTL discriminator.** `BufferLoadB128.__mro__` =
`[BufferLoadB128, MUBUFReadInstruction, GlobalReadInstruction,
ReadWriteInstruction, Instruction, Item, object]`. There is no
`BufferLoadDTL` subclass. The DTL-vs-normal distinction is entirely
encoded in two places at construction:

1. `dst` is `nullptr` (Python `None`) — the validator can see this
   because `getParams()[0]` is None.
2. `mubuf.lds == True` — the validator **cannot** see this. Probe
   confirmed: `hasattr(inst, "mubuf") == False` on both normal and DTL
   BufferLoadB128 instances.

The C++ field `std::optional<MUBUFModifiers> mubuf` exists on
`MUBUFReadInstruction` (`mem.hpp:293`) but is not bound to Python
(`mem.cpp:233-250` only `.def`s `getParams` + `__str__` for the base;
subclasses add only the constructor binding). Even if `.mubuf` were
exposed, `MUBUFModifiers` itself only `def_rw`s `isStore`
(`container.cpp:348`); `lds`, `offen`, etc. are write-once via the ctor
but unreadable post-construction (the only path is `__getstate__`).

So today the validator's `_is_dtl_buffer_load` heuristic
(`getParams()[0] is None`) is the **only** observable signal — and it
relies on the convention that no other BufferLoad shape produces a None
dst. That convention is correct (verified against
`KernelWriterAssembly.py:15312-15343`: kernel writer constructs
`dst=None if lds else vgpr(...)`) but it's structural inference, not a
named flag.

**No implicit-m0-read accessor.** No `implicit_reads`,
`reads_m0`, `implicit_operands`, etc. on `MUBUFReadInstruction` or any
subclass. The "DTL implicitly reads m0 for the LDS destination" semantic
exists nowhere in rocisa.

### 1.3 Comparison to the q9j surprise

q9j was a binding gap: `getDstParams()` / `getSrcParams()` were declared
pure-virtual on `rocisa::Instruction` (`instruction.hpp:274-275`),
implemented correctly on every concrete class, but not `.def`'d in
nanobind. The C++ already had the answer; the validator just couldn't
see it.

dzl is **not** that. The C++ does not declare any SCC accessor, any
implicit-operand accessor, any DTL discriminator method, or any
`scc_resource()` factory. Grepping the rocisa headers and src for
`scc`, `implicit`, `is_dtl`, `reads_`, `writes_` (other than the
existing `getDstParams`/`getSrcParams`) returns nothing. There is no
hidden API to bind. The only marginal exception is `mgpr(0)` (which
already works) and `MUBUFModifiers.lds` (whose binding is missing but
which only partially solves DTL detection — see §3).

---

## 2. Per-workaround disposition

| Workaround | Cat | Rationale |
|---|---|---|
| `_SCC_OPCODE_FLAGS` (~50 entries) | **C** | The C++ class hierarchy carries zero SCC information. The hand-curated table maps `class_name → (shape, reads_scc, writes_scc)` — every fact in it would have to be added to rocisa as either a flag/method on each of the ~50 classes (~50 ctor bodies), a base-class change (`SCCWritingInstruction`), or a static metadata table on the C++ side. No path through rocisa-as-is. |
| `_SCCRule` | **C** | Consumes `_SCC_OPCODE_FLAGS`. Cannot exist without SCC metadata in some form — either the table or a rocisa-side equivalent. The "no_dst" vs "dst_then_srcs" shape distinction does fall out of `getDstParams()` (q9j-A: `SCmpEQU32::getDstParams()` returns `{}` because `dst=nullptr`), but the `(reads_scc, writes_scc)` tuple is purely Python-side. |
| `_SCC_SENTINEL` / `_get_scc_sentinel` | **C** | No `scc` regType anywhere in rocisa. The sentinel hack (mutate `vgpr(0).regType = "scc"`) survives only because `RegisterContainer.regType` is `def_rw`. No `scc_resource()` factory exists, no `RegName("scc", ...)` precedent. Need new C++ work to add: a public `scc_resource()` factory exporting a singleton with `regType="scc"`, plus a constructor path for `RegisterContainer` that doesn't require mutating a vgpr. |
| `_DTLBufferLoadRule` | **C** (with B-flavored partial workaround) | The rule itself encodes (a) DTL-detection heuristic and (b) the `mgpr(0)` implicit-read insertion. (a) is inferable today via `getParams()[0] is None` — no rocisa change strictly needed for detection, though it's structural inference rather than a named flag. (b) requires implicit-operand metadata that does not exist. The cleanest fix is rocisa-side: split `BufferLoadDTL` as a distinct subclass OR add `is_dtl: bool` and `implicit_reads()` accessor. A thin B-grade adapter (bind `MUBUFModifiers.lds` as `def_rw` + bind `MUBUFReadInstruction.mubuf` as `def_rw`) would let the validator distinguish DTL by reading `inst.mubuf.lds` — strictly more reliable than `dst is None` — but the m0 read still has to come from a Python-side rule. |
| `_is_dtl_buffer_load` | **C** (or B partial) | Same as `_DTLBufferLoadRule`. Today: pure validator heuristic. With B-grade adapter: `inst.mubuf is not None and inst.mubuf.lds`. With full C-grade: `inst.is_dtl` or `isinstance(inst, BufferLoadDTL)`. |

**Counts: A=0, B=0 (C with partial-B path on DTL only), C=6.**

There is no q9j-style surprise here. All six items genuinely need new
rocisa C++ work. The smallest possible scope-reduction is to bind
`MUBUFModifiers.lds` and `MUBUFReadInstruction.mubuf` to Python so DTL
detection stops being a structural-inference heuristic — that's a
B-flavored partial that touches only the binding layer, not new C++
methods. Even that doesn't shrink dzl's main task (SCC).

---

## 3. Remaining dzl scope

After this reassessment dzl's scope is essentially unchanged from its
current description, with one minor refinement:

### 3.1 SCC piece (dominant)

Three C++ additions, in order of cleanest-design-to-smallest-change:

1. **Add `reads_scc()` and `writes_scc()` virtual methods on
   `rocisa::Instruction`** (defaulting to false) and override in each of
   the ~50 SCC-touching classes. Validator deletes `_SCC_OPCODE_FLAGS`,
   `_SCCRule` becomes `if inst.reads_scc(): reads += (scc,)`.
2. **Add `scc_resource()` factory in `rocisa.container`** returning a
   singleton `RegisterContainer` with `regType="scc"`, `regIdx=0`,
   `regNum=1`, and a usable `regName`. Validator deletes
   `_SCC_SENTINEL` / `_get_scc_sentinel`, imports
   `from rocisa.container import scc_resource` instead.
3. (Optional, only if q9j's `getDstParams()` integration is in flight)
   **Fold SCC into the typed operand surface** — have
   `getDstParams()` / `getSrcParams()` return `scc_resource()` for
   classes that touch SCC. The "no_dst" vs "dst_then_srcs" distinction
   already falls out of q9j's binding fix (probe shows
   `SCmpEQU32::getDstParams() = {}`), so the SCC piece reduces to
   "include SCC in the dst/src lists for SCC-touching classes." This
   eliminates `_SCC_OPCODE_FLAGS` AND `_SCCRule` AND `_SCC_SENTINEL` in
   one go.

The bead description's preference for option 3 ("preferably, fold into
the typed `read_operands()` / `write_operands()` accessors from bead q9j")
remains correct.

### 3.2 m0 / DTL piece (smaller)

Two paths, partially independent:

1. **Minimal binding patch (B-grade partial).** Bind `def_rw("mubuf",
   ...)` on `MUBUFReadInstruction` and `def_rw("lds", ...)` on
   `MUBUFModifiers`. Validator's `_is_dtl_buffer_load` becomes
   `inst.mubuf is not None and inst.mubuf.lds` — strictly more reliable
   than `getParams()[0] is None`. This does NOT eliminate
   `_DTLBufferLoadRule`; the m0 implicit-read still has to be expressed
   somewhere on the validator side.
2. **Full DTL modeling (C-grade).** Either split `BufferLoadDTL` as a
   distinct rocisa class (cleanest; matches the `dst=None`-vs-`dst=vgpr`
   semantic split) or add `is_dtl()` + `implicit_reads()` virtual
   methods on `MUBUFReadInstruction`. The validator's
   `_DTLBufferLoadRule` and `_is_dtl_buffer_load` then delete entirely;
   m0 reads flow through the same rule as everything else.

If the validator does (1) without (2), dzl shrinks but does not close.
Closing dzl requires (2).

---

## 4. Proposed rocisa-side expansions (genuine C work)

### 4.1 SCC virtual accessors (option 1)

```cpp
// instruction.hpp, on the abstract base
struct Instruction {
    // ... existing virtuals ...
    virtual bool reads_scc()  const { return false; }
    virtual bool writes_scc() const { return false; }
};

// common.hpp, per SCC-touching class:
struct SAddU32 : public CommonInstruction {
    // ... existing ctor ...
    bool writes_scc() const override { return true; }
};
struct SAddCU32 : public CommonInstruction {
    bool reads_scc()  const override { return true; }
    bool writes_scc() const override { return true; }
};
struct SCSelectB32 : public CommonInstruction {
    bool reads_scc()  const override { return true; }
};
// ... ~50 such overrides total, mostly one-liners ...

// branch.hpp:
struct SCBranchSCC0 : public BranchInstruction {
    bool reads_scc() const override { return true; }
};
```

Plus three `.def` lines per class block in the binding (or one per
hierarchy level if the virtual is bound at base).

### 4.2 SCC RegisterContainer factory

```cpp
// container.hpp (or container.cpp namespace)
std::shared_ptr<RegisterContainer> scc_resource() {
    static auto singleton = std::make_shared<RegisterContainer>(
        "scc", RegName("scc", {}), 0, 1);
    return singleton;
}

// container.cpp binding
m_con.def("scc_resource", &rocisa::scc_resource);
```

The current Python-side hack (`vgpr(0)` + `regType = "scc"` mutation) is
a workaround for a missing factory and a constructor that requires
non-None `regName`. Either would suffice; the factory is cleaner.

### 4.3 DTL split / accessor

Preferred — split as distinct subclass:

```cpp
// mem.hpp
struct MUBUFReadDTLInstruction : public MUBUFReadInstruction {
    MUBUFReadDTLInstruction(InstType instType, /* ... no dst ... */) { ... }
    std::vector<InstructionInput> getDstParams() const override { return {}; }
    std::vector<InstructionInput> getSrcParams() const override {
        return {vaddr, saddr, soffset, mgpr_resource()};  // implicit m0
    }
    bool is_dtl() const override { return true; }
};
```

Less-disruptive — flag + accessor on existing class:

```cpp
struct MUBUFReadInstruction : public GlobalReadInstruction {
    bool is_dtl() const { return mubuf && mubuf->lds; }
    std::vector<InstructionInput> implicit_reads() const {
        return is_dtl() ? std::vector{InstructionInput{mgpr_resource()}}
                        : std::vector<InstructionInput>{};
    }
};
```

Plus the small B-grade binding patch (`.def_rw("mubuf", ...)` and
`.def_rw("lds", ...)` on `MUBUFModifiers`) that's worth doing
independently regardless of path.

### 4.4 What does NOT need new C++

The DTL-detection heuristic itself (`getParams()[0] is None`) survives
without any change — the q9j binding lands `getDstParams()`, validator
checks `not inst.getDstParams()`. That gives "is this a no-dst load?"
for free, but it doesn't tell you "and therefore m0 is read." That last
step is the implicit-operand metadata gap that needs new work.

---

## 5. Recommendation

**Proceed with dzl as-is. Do not close, do not rescope.**

The dzl bead description (current) accurately describes a six-item C++
expansion in rocisa. The empirical probe confirms each of the six
workarounds genuinely needs new metadata that doesn't exist anywhere in
rocisa today — not in headers, not in bindings, not in factories, not
in base classes. There is no q9j-style hidden API to surface.

Two minor refinements worth noting in the bead description (added
empirical-probe evidence; not material to scope):

- **Independent micro-win**: bind `MUBUFModifiers.lds` (and ideally
  `.offen`/`.glc`/etc.) as `def_rw` and bind
  `MUBUFReadInstruction.mubuf` as `def_rw`. This isn't part of dzl's
  acceptance criteria but it (a) makes DTL detection more principled
  even before the full split lands, and (b) is the kind of thing that
  could be done cheaply alongside q9j's binding-fix work. Worth a
  one-line note in dzl's description.
- **Confirmation of design direction**: option 3 in §3.1 (fold SCC into
  q9j's typed operand surface) is the cleanest and matches the bead's
  current "preferably" preference. The probe confirms this is feasible
  — `SCmpEQU32`'s `dst=nullptr` pattern means q9j's `getDstParams()`
  already does the no-dst shape work; SCC just needs to be appended to
  the appropriate list per-class.

Bead remains open. Description gets a small note about the binding
micro-win. No fundamental rescope.

---

## 6. The probe

`/tmp/scc_m0_probe.py` — instantiates SCmpEQU32, SCmpKEQU32, SAddU32,
SAddCU32, SCSelectB32, SAndSaveExecB64, SOrSaveExecB64, SCBranchSCC0,
plus normal and DTL BufferLoadB128. For each, dumps `__mro__`, every
public attribute via `dir()`, calls `getParams()`, and probes for ~25
candidate implicit-operand attribute names. Also enumerates every public
symbol in `rocisa.container` to confirm the absence of `scc`-anything.

Key probe outputs preserved verbatim in §1.1 / §1.2 above.

---

## Appendix: comparison summary

| Property | q9j (binding gap) | dzl (genuine new work) |
|---|---|---|
| C++ pure virtual exists | YES (`getDstParams`, `getSrcParams`) | NO |
| C++ concrete overrides exist | YES (every concrete class) | NO |
| Python binding present | NO (~12 `.def` lines missing) | NO (no source method to bind) |
| Validator workaround removable by binding fix | YES (~260 LoC) | NO |
| Required new C++ work | None | ~50 class-level overrides + 1 factory + DTL split (or 2 virtuals) |

dzl is a real epic. q9j was a bug.
