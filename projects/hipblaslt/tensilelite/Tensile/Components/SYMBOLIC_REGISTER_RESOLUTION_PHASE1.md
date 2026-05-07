# Symbolic → Numeric Register Resolution: Phase 1 Investigation

**Bead:** `rocm-libraries-d0xd` (Phase 1 only — research, no implementation)
**Branch:** `users/alvasile/validator_long_term_plans` (vlt)
**Predecessor:** `rocm-libraries-c70` (commit `449fadd493` — `Register` abstraction landed)
**Status:** Phase 1 complete. Phase 2 (design questions for the user) is pre-staged at the end.

---

## Why this epic exists (recap of c70 rationale)

c70 introduced `Register` (`Tensile/Components/register.py`), a Python wrapper around
rocisa's nanobind-locked `RegisterContainer` that unifies the validator's handling of
two register forms:

- **Numeric** (`regIdx >= 0`, `regName is None`): post-allocation form.
- **Symbolic** (`regIdx == -1`, `regName` set): pre-allocation form keyed on
  `regName.name` + `getTotalOffsets()`.

c70's first draft shipped with an `allocation_table: Mapping[str, int]` parameter
on `Register.from_rocisa(...)` to fold symbolic to numeric. **It was removed before
landing** because:

1. No production caller passed one.
2. No tests exercised it.
3. The codebase had no public source of a `Mapping[str, int]` of the right shape.
4. It violated the project's clean-implementation directive ("no speculative
   scaffolding without a real consumer").

The need it was anticipating is real — the validator will eventually compare two
graphs where one carries symbolic registers and the other carries the numeric
registers they were resolved to. This memo investigates *what shape* the
resolution mechanism should take, *where* the data comes from, and *who* would
actually consume it. It does **not** prescribe an implementation.

---

## Q1: Where does symbolic-form data come from in real captures?

### Finding: symbolic registers absolutely appear in real captures today.

The kernel writer constructs many container instances in symbolic form via the
`vgpr(name: str, regNum, ...)` factory overload (rocisa
`container.cpp:128`). These are explicitly the `regIdx == -1` shape — `regNum`
is set, `regName` carries the symbolic root, and `regIdx` defaults to `-1`.

Concrete grep evidence (representative, not exhaustive):

- `Tensile/Components/LocalRead.py:76` — `destVgpr = vgpr("Valu%s_X%u_I%u+%u" %
  (tc, bufferIdx, iui, valuIdx), blockWidth)` — every LR destination is symbolic.
- `Tensile/Components/LocalRead.py:79–162` — `vgpr("LocalReadAddr%s+%u" %
  (tc, num))` — LR address operand symbolic.
- `Tensile/KernelWriterAssembly.py` — across roughly lines 5000–7600 (and
  further occurrences out past line 8800), the MFMA emission paths repeatedly
  use `vgpr("ValuA_X%u_I%u+%u" % (bufferIdx, iui, ...))`,
  `vgpr("ValuC+%u" % (...))`, `vgpr("ValuMXSDummy")`,
  `vgpr("ValuMetadata+%u" % i)`, etc. The whole MFMA feed-in path uses
  symbolic vgprs. Concrete spot-checked sites:
  `KernelWriterAssembly.py:5253` (`ValuC` accumulator),
  `KernelWriterAssembly.py:7529` (`ValuA_X` MFMA operand),
  `KernelWriterAssembly.py:7559` (`ValuB_X` MFMA operand).
- `Tensile/Tests/unit/test_dataflow_graph_comparison.py:482–484` — production
  fixture `MFMAInstruction(acc=vgpr("ValuA_T0_I0", 4), a=vgpr(74, 2),
  b=vgpr("ValuA_X0_I0", 2))` — *mixes* symbolic and numeric in the same
  instruction. The XF32-emulation pattern. Confirms that even within one
  instruction, the symbolic/numeric mix is real, not a synthetic-test
  artefact.

The capture machinery (`ScheduleCapture.LoopBodyCaptureBuilder.add(inst, ...)`,
`ScheduleCapture.py:465–469`) wraps `inst` directly without resolving operands,
so the captured `TaggedInstruction.wrapped.rocisa_inst` retains whatever symbolic
RegisterContainers the emit code constructed.

Validator code already grapples with this:

- `ScheduleCapture.py:1053–1060` (`_byte_keys_for_resource`) keys numeric
  registers by `(regType, regIdx + i)` and symbolic registers by
  `(regType, name, base + i)` — disjoint key spaces. **A symbolic and a numeric
  register that refer to the same physical vgpr will not collide in this map.**
  This is precisely the wx9.3 7a limitation, and it's structural.
- `test_dataflow_graph_comparison.py:500–517`
  (`test_symbolic_and_numeric_for_same_logical_reg_unchanged`) pins this
  limitation as a known gap. Documented, not failing.

### Lifecycle observation

The symbolic form is deliberate, *not* an unresolved intermediate that gets
rewritten. The kernel writer emits symbolic operand strings into
RegisterContainer objects so the assembler later renders them as macro
references like `vgprValuA_X0_I0+0` against `RegSet` directives that the same
kernel writer also emits. Symbolic-ness is preserved through
`_getKernelSource()` — the assembler resolves names, not the Python writer.

This means **the captured RegisterContainer instances are still symbolic at the
time the validator builds its graphs** — there is no point in the existing
pipeline where they get rewritten to numeric form.

### Empirical confirmation — caveat

The Q1 chain above is **deductive**: writers emit symbolic vgprs via the
`vgpr(name, regNum)` factory; the capture machinery wraps instructions
without resolving operands; therefore captures retain symbolic
`RegisterContainer` instances with `regIdx == -1`. I did not actually run a
representative capture test (e.g. `test_tf32_4x4_tn_capture_shape`) and count
the `regIdx == -1` instances, because doing so would require a rocisa
rebuild in this worktree that is out of scope for a doc-only Phase 1 fix
pass. The deductive chain is short and the per-step evidence (factory
overload at `container.cpp:128`, capture-builder code at
`ScheduleCapture.py:465–469`, and the symbolic-form assertions in the
existing pinned-limitation test
`test_symbolic_and_numeric_for_same_logical_reg_unchanged`) is direct, but
the count itself is unverified empirically. Phase 2 should confirm with a
quick `sum(1 for inst in capture.instructions for op in operands(inst) if isinstance(op, RegisterContainer) and op.regIdx == -1)` measurement
when it picks up the design work.

### Bottom line for Q1

Symbolic registers are **not future-only**; they appear in every real capture
today. The wx9.3 7a "documented limitation" is currently latent only because
both the reference and subject captures consume the **same** kernel writer
state, so they happen to use the same symbolic names. The instant a future
scheduler (or codegen-vs-codegen comparison per wic4 Part C) constructs the
"same" register with a different identifier on either side, identity comparison
diverges silently.

---

## Q2: Where would the allocation table come from?

### `RegisterPool` does not carry it (today)

`Tensile/Common/RegisterPool.py` (Python wrapper) and
`rocisa/rocisa/include/register.hpp` (C++ implementation) define `RegisterPool`
as a *slot-indexed* allocator. State is `std::vector<Register>` where each
slot has `{Status, std::string tag}`. The `tag` is a freeform allocation hint
("init", "ValuA", whatever the caller passed to `checkOutAligned`) — multiple
slots can share a tag, the tag is overwritten on re-checkout, and there is **no
unique inverse "name → base index" view**.

Public API today (paraphrased): `add`, `addRange`, `checkOut`, `checkOutAligned`,
`checkOutMulti`, `checkIn`, `getPool` (returns the raw vector), `state` (debug
string). Nothing returning a `Mapping[str, int]`.

To extract a name → base map from `getPool()` you'd have to walk the vector and
collect the first-occurrence index of each tag — and you'd be wrong, because
tags are not stable identifiers (a slot's tag is overwritten by every checkout
that lands on it).

### Where the data actually lives — sgprs vs vgprs are asymmetric

This is the most important Q2 finding.

**SGPRs.** The `name → base` mapping for sgprs is stored explicitly on the
kernel writer:

```python
# Tensile/KernelWriter.py:6105
self.sgprs = collections.OrderedDict()

# Tensile/KernelWriterAssembly.py:503–531
def defineSgprIdx(self, name, numSgprs, align=1):
    sgprIdx = self.sgprPool.checkOutAligned(numSgprs, align, tag=name, ...)
    self.sgprs[name] = sgprIdx          # canonical name → base mapping
    return sgprIdx
```

So for sgprs there is exactly the `Mapping[str, int]` the c70 speculation
wanted, and it's a public attribute on `KernelWriterAssembly`. Lifetime: it
grows monotonically through the build via `defineSgpr` calls, with entries
removed by `undefineSgpr` (which also calls `sgprPool.checkIn` — but the dict
entry is **not** deleted, so old name lookups still resolve to the now-recycled
base index). This is a subtle correctness footgun for any post-hoc resolver.

**VGPRs.** No comparable Python-side dict. `self.vgprs` (`KernelWriter.py:605,
6104`) is a `StateVgprs` *dataclass* with hand-named integer attributes
(`coord0`, `coord1`, `addrD`, …) — fixed schema, not a `Mapping[str, int]`.
Per-name vgpr allocations live elsewhere, distributed across `self.states.a`,
`self.states.b`, `self.states.c`, etc., and are resolved at *assembly* time
(not Python time) via a global rocisa-side mechanism:

```cpp
// rocisa/rocisa/include/base.hpp:162–198
std::map<std::string, int> getVgprIdx() { return m_vgpridx[std::this_thread::get_id()]; }
void setVgprIdx(const std::string& s, const int idx) { m_vgpridx[id][s] = idx; }

// rocisa/rocisa/include/code.hpp:892–904 (RegSet directive)
inline void setIdx(int val, int offset) const {
    rocIsa::getInstance().setVgprIdx(name.substr(4), val + offset);  // strip "vgpr" prefix
}
```

So for **vgprs** the resolution mechanism is a **per-thread global map** in
rocisa, populated by `RegSet` directives that the kernel writer emits as part
of `defineSgpr`/`defineVgpr` (`KernelWriterAssembly.py:517` —
`RegSet("s", "sgpr"+name, ...)`; equivalent vgpr emission elsewhere). It is
already bound to Python:

```cpp
// rocisa/rocisa/src/base.cpp:124
.def("getVgprIdx", &rocisa::rocIsa::getVgprIdx, "Get vgpr idx.")
```

And `RegName.getTotalIdx()` (`container.hpp:691–695`) already calls
`setNameIdx()` which queries this map to compute `getTotalOffsets() + nameIdx`
— i.e., the **C++ side already does symbolic-to-numeric resolution for vgprs**,
just not in a way that's exposed on `RegisterContainer.regIdx` (which stays
`-1` even after the global map is populated).

### Synthesis

Today's reality, by namespace:

| reg_type | name → base mapping source                    | Public from Python?               | Stable?           |
|----------|------------------------------------------------|-----------------------------------|-------------------|
| `s` (sgpr) | `KernelWriterAssembly.sgprs: OrderedDict[str, int]` | Yes (instance attribute)         | Monotonic add; entry survives `undefineSgpr` (footgun) |
| `v` (vgpr) | `rocIsa.getInstance().getVgprIdx()` (per-thread) | Yes (`rocisa.base.rocIsa` binding) | Per-thread; cleared/populated by `RegSet` emission |
| `acc` (accvgpr) | Same global vgpr map (no separate namespace)? | Same as vgpr                     | Unverified — needs inspection of `accvgpr()` factory |
| `m` (mgpr)  | Unknown — no factory uses `setVgprIdx`         | Probably none                    | N/A               |

The clean answer to "where does the table come from?" is **two different
places, with different lifetimes and different scoping**, depending on
`reg_type`. The c70 speculation about "a Mapping[str, int]" was naïve in
treating vgpr and sgpr as the same shape.

### What it would cost to "expose" `RegisterPool.snapshot()`

It would not give you the right data. The pool tracks slot occupancy by
tag-as-freeform-string, not by canonical-name-as-key. The right API additions
(if we go down this road) are:

- For **sgprs**: nothing — `KernelWriterAssembly.sgprs` already is the table.
  Just expose it as a read-only view (`MappingProxyType`?) and document the
  `undefineSgpr` footgun.
- For **vgprs**: nothing on `RegisterPool` — extend the binding around
  `rocIsa.getVgprIdx()` (already exists) into a more ergonomic accessor on
  `KernelWriter`, e.g. `KernelWriter.vgpr_alloc_table() -> Mapping[str, int]`.
- For **accvgpr / mgpr**: confirmed-unknown; needs Phase 2 question.

### Bottom line for Q2

The `Mapping[str, int]` data exists, but it lives in two completely different
places (`KernelWriterAssembly.sgprs` for sgpr, `rocIsa.getVgprIdx()` for vgpr)
and has materially different lifetimes and scoping. Any unified resolver API
must either accept both or be `reg_type`-dispatched. There is no single source
of truth.

---

## Q3: What are the real consumers?

### wx9.3 (deferred to 2027-01-01 unless triggered)

`rocm-libraries-wx9.3` covers two related concerns:

- **7a** — same logical register, different identifier (symbolic vs numeric)
  across two captures. Pinned as a documented limitation in
  `test_symbolic_and_numeric_for_same_logical_reg_unchanged`. Doesn't fire
  today because both captures consume the same writer state, but a future
  scheduler that constructs registers differently triggers it.
- **7b** — topology equivalence (same producer-consumer structure, different
  register strings labeling edges).

The bead's own appended research report (comment id 36) recommends solving 7a
**at graph build frontend, not in the comparator**: maintain a single
`SymbolMap` that resolves every register reference (symbolic or numeric) to a
unique `PhysicalRegister` object before identity is constructed.

That recommendation aligns with our Q2 finding: the resolution data already
exists at writer-state time, so resolving at graph-build-frontend has the data
it needs without any new plumbing through the comparison pass.

The wx9.3 surface in code is `Tensile/Components/CMSValidator.py` (the
`compare_graphs` / `_identity_for` path). Today's identity:

```python
# CMSValidator.py — _identity_for builds (cls_tag, loop_index, render_string)
# render_string is _canonical_render(rocisa_inst), which calls str(inst) and
# strips comments/whitespace. The render string contains the SYMBOLIC name
# (e.g. "vgprValuA_X0_I0") for symbolic registers, NEVER the numeric form.
```

So 7a's exact failure mode is: `_canonical_render(inst_with_symbolic_v)` and
`_canonical_render(inst_with_numeric_v_at_same_index)` produce different
strings, identity tuples diverge, comparison says "different" when they're the
same. Resolution would canonicalize both sides to (say) `"v[34:37]"` before
rendering.

### wic4 (open, P2; blocked by 5gd, wx9.3, 9lcs)

`rocm-libraries-wic4` Part C explicitly calls out the same surface:

> "Codegen-vs-codegen will routinely produce graphs that have **identical
> topology but different register names** (different register allocator runs,
> different live-range pressure)."

It then explicitly defers to wx9.3 for the underlying resolution mechanism.

wic4 also adds Part B (per-source label rendering — the assembly mnemonic +
operands, not just CMS-shape categories) which means the consumer side will
want a register *string* canonicalization step, not just a *tuple*
canonicalization. A resolver that returns "the canonical numeric form" is more
useful to wic4 than one that returns "a hash that proves equality."

### Other potential consumers (today, before wx9.3/wic4)

Searched `Tensile/Components/` and `Tensile/Tests/unit/` for any current code
path that would benefit. Nothing actionable. The validator works around the
symbolic/numeric divergence by relying on the "both sides see same writer
state" invariant, and no other consumer of `Register` exists.

The closest near-term application: `Register.overlaps()` *currently rejects*
mixed numeric/symbolic comparisons (`register.py:131–135`). If a resolver
existed, callers could opt into "resolve first, then compare" to handle the
mixed case. But no current call site needs this; rejecting is correct behaviour.

### Bottom line for Q3

- Two named consumers exist (wx9.3, wic4). Neither is active today; wx9.3 is
  deferred to 2027 and wic4 is blocked on three other beads.
- wx9.3's own research recommends solving the problem at the graph-build
  frontend (one `SymbolMap` resolution before identity construction), NOT in
  the comparator and NOT inside `Register.overlaps()`.
- wic4's Part B rendering wants canonical *strings* (assembly text), so a
  resolver returning "canonical numeric form" is the right direction.
- No current code path needs symbolic→numeric folding. The c70 decision to
  drop `allocation_table` was correct.

---

## Q4: What are the constraints?

### Stability (per-build vs per-codepath vs other)

- **sgpr table**: per-`KernelWriterAssembly` instance. One writer = one
  `self.sgprs` dict. Persistent for the life of the build. Within a build it
  grows monotonically by `defineSgpr` and (importantly) `undefineSgpr` does
  NOT delete entries — it only frees the pool slot. So lookups during the
  build remain valid even after a name is "undefined." After
  `_getKernelSource()` returns, the writer object's `sgprs` dict is still
  populated; nothing tears it down.
- **vgpr table**: per-thread global, in `rocIsa::m_vgpridx` (keyed on
  `std::this_thread::get_id()`). Populated as `RegSet` directives are
  emitted (via `RegSet::setIdx`). The per-thread map is **reset** at the start
  of every kernel build: `rocIsa::setKernel()` (`rocisa/rocisa/include/base.hpp:108–118`)
  explicitly does `m_vgpridx[id] = std::move(std::map<std::string, int>())`,
  and `setKernel` is called from the kernel-init path
  (`Tensile/KernelWriter.py:6100`, immediately before the per-build state
  containers `self.vgprs` and `self.sgprs` are constructed at lines 6104–6105).
  So a snapshot taken at end-of-build A is valid until the *next* build's
  `setKernel` call wipes the per-thread map; it does not progressively decay
  via overwrite. The hazard is staleness via reset, not progressive
  overwrite — a snapshot bound to a specific build is safe to query as long
  as no later build on the same thread has started. This is still a hidden
  global-state hazard the c70 speculation didn't account for, but the
  failure mode is "whole-table replacement at next `setKernel`," not
  "progressive collision as more names are defined."

### Lifetime / "when is it final?"

- **sgpr table** is final-enough at end of `_getKernelSource()`, but it's also
  *valid earlier* — every `defineSgpr` immediately commits the binding. So
  callers in the same build can resolve any sgpr name as soon as it's been
  defined.
- **vgpr table** is final at end of `_getKernelSource()` because RegSet
  directives are emitted at write time. But because it's a per-thread global
  that `setKernel` resets at the start of every build, "final" only holds
  until the next kernel build on the same thread starts (at which point the
  whole map is replaced with an empty one).
- Per-codepath: I found no evidence either table is ever rewound or branched
  for an alt codepath. CMS captures multiple codepaths on the same writer,
  using the same sgpr/vgpr tables.

### Mutability — can a name's numeric assignment change during a build?

- **sgprs**: Yes, via `undefineSgpr(name)` followed by a fresh
  `defineSgpr(name, ...)`. The dict entry gets overwritten. So a "name → base"
  lookup mid-build that gets cached and used later may be stale. **This is
  why a resolver that takes a snapshot at construction time and queries it
  later must either be coupled to the build's lifecycle or accept the staleness.**
- **vgprs**: Per-thread global mutates for every `RegSet` directive emit. Same
  staleness risk.

So the table is **not monotonic**. The c70 speculation implicitly assumed
monotonicity (any `Mapping[str, int]` snapshot would be valid forever). It's
not.

### Multiple-namespace

Yes, definitely one table per `reg_type`:

| reg_type | namespace                                           |
|----------|----------------------------------------------------|
| `s`      | `KernelWriterAssembly.sgprs`                       |
| `v`      | `rocIsa.getVgprIdx()`                              |
| `acc`    | Probably `rocIsa.getVgprIdx()` (same global; needs verification) |
| `m`      | Likely none (mgpr factory doesn't populate vgpr global)         |
| `scc`/`vcc`/`exec` | Sentinel registers; no name → base allocation     |

A unified resolver MUST be `reg_type`-dispatched. The c70 single-`Mapping[str,
int]` shape would have been wrong for everything except sgprs.

### Bottom line for Q4

The constraints are messier than the c70 speculation acknowledged:

1. The data lives in **two places**, asymmetrically split by `reg_type`.
2. Both tables are **mutable mid-build** (sgpr via undefine+redefine; vgpr via
   global per-thread mutation).
3. The vgpr table is **per-thread global**, not per-kernel — but each
   kernel build's `setKernel` call **resets** the map to empty before
   populating it. Successive builds on the same thread don't progressively
   collide; they replace. A snapshot is valid until the next build starts,
   then the underlying map is empty.
4. Lookups must be `reg_type`-dispatched.
5. There is no single `Mapping[str, int]`. Any API that pretends there is one
   is wrong.

---

## Phase 2 questions to surface to the user

When this comes back to design, the user needs to weigh in on these. Listed
roughly in dependency order — earlier answers constrain later ones.

### Decision 1: Is symbolic→numeric resolution actually the right direction?

The wx9.3 research report (comment 36 on rocm-libraries-wx9.3) recommends
**canonicalizing all references to a single `PhysicalRegister` object** at
graph-build frontend, *before* identity construction. That's NOT the same as
"fold symbolic to numeric." It could equally mean:

- (a) Fold both forms to numeric via the allocation table.
- (b) Fold both forms to *symbolic* via reverse lookup (numeric→symbolic).
  Cleaner because numeric vgprs have no name, but you'd have to invent one
  ("v_at_34") which doesn't gain you anything.
- (c) Use a separate `PhysicalRegister` token type that isn't either form.

(a) requires the allocation table; (c) doesn't. **Should we be writing a
resolver at all, or should `Register` carry a `physical_id` token whose
construction knows how to query the right table?**

### Decision 2: Eager vs lazy resolution

If we go with (a):

- **Eager**: `Register.from_rocisa(rc, table=...)` resolves at construction
  time. All `Register` instances downstream are numeric. Comparison is cheap.
  Risk: if the table is mutated after construction (sgpr undefine+redefine),
  cached `Register` instances are stale.
- **Lazy**: `Register` keeps both forms and resolves at compare time. Symbolic
  comparisons that don't need resolution stay cheap. Risk: comparison API has
  to thread the table through every overlap/contains call site.

The Q4 finding that the table is mutable mid-build leans toward lazy — eager
resolution captures a snapshot that may be invalidated.

### Decision 3: One table or per-`reg_type`?

Given Q2/Q4 — the data lives in two places — does the API:

- (a) Accept a single `AllocationTable` object that internally dispatches by
  `reg_type` and queries the right backend?
- (b) Accept a `Mapping[reg_type, Mapping[str, int]]`?
- (c) Bind directly to `KernelWriterAssembly` and call its accessors on demand?
- (d) Something else (e.g., a `ResolverProtocol` interface)?

(c) couples the validator to the writer, which is currently a clean separation
boundary (validator consumes capture, doesn't reach back into writer state).
But (c) is also the only way to get a always-current view through mutability.

### Decision 4: Where does resolution actually live — `Register`, the comparator, or graph build?

wx9.3's research recommends **graph-build frontend**. That means:

- `Register.from_rocisa(rc)` stays single-arg.
- `build_dataflow_graph(...)` gains a resolver argument that canonicalizes
  every `RegisterContainer` into a `Register` with consistent form before
  identity construction.
- `compare_graphs` doesn't change.

This is the cleanest separation. Open question: do we instead want a
`Register.resolve(table)` method for callers that DO want to resolve
post-construction (e.g., after-the-fact analysis tools)?

### Decision 5: Numeric form for the vgpr global hazard

The vgpr `m_vgpridx` is per-thread global. It is reset to empty at the start
of every kernel build by `rocIsa::setKernel()` and then populated by RegSet
emissions during that build. A snapshot taken at the end of build A is valid
for build A's captures *until* build B's `setKernel` call wipes the map.
Querying the live `getVgprIdx()` after build B has started returns build B's
table (or empty, if no RegSets have fired yet), which is the WRONG table for
build A's captures. **Should the vgpr table be snapshotted per-build at the
graph-build callsite (so the captured `Mapping[str, int]` is bound to the
kernel's captures and survives later builds), or queried live (which only
works if no further builds have happened on the same thread)?**

A snapshot-at-end-of-build approach is safer than the eager/lazy tradeoff
in Decision 2 might suggest: because the vgpr map is reset (not progressively
overwritten), a single snapshot taken once at end-of-build is *fully valid*
for the lifetime of that build's captures — there's no in-build staleness on
the vgpr side. The sgpr-side staleness (mutation via undefine+redefine
mid-build) remains the harder problem.

This still pushes toward "snapshot at the callsite that has the writer in
scope" — which is `_getKernelSource` or its callers, not `compare_graphs` —
but the snapshot itself can be a plain `dict(rocIsa.getInstance().getVgprIdx())`
copy taken once, with no need for live re-query.

### Decision 6: Behavior on missing names

If a symbolic name appears in a capture but the table doesn't carry a binding:

- (a) `KeyError` — strict, surfaces capture-vs-table mismatches loudly.
- (b) Silent fallback to symbolic form — the comparison falls back to today's
  behavior (mixed-form compares fail).
- (c) Configurable — caller chooses.

### Decision 7: Mismatched-table comparisons

Two `Register` instances resolved with different tables (e.g., kernel A's
table vs kernel B's table — possible in cross-kernel comparison scenarios
that wic4 hints at):

- (a) Runtime error.
- (b) Silently allowed — the resolved numeric forms are what's compared, even
  if they came from different allocators.

(b) is what's needed for codegen-vs-codegen comparison (the WHOLE POINT is
that two different allocators produced different bindings of the same logical
register and we want to verify the dataflow is the same regardless). But it's
also a footgun if applied accidentally to comparisons that should fail.

### Decision 8: accvgpr / mgpr

I did not verify whether `accvgpr(name, regNum)` populates `m_vgpridx` (so
acc shares the vgpr global), or has its own namespace, or has neither. mgpr
likely has neither. **Does this epic try to handle all four `reg_type`s, or
scope down to vgpr+sgpr (the only ones with established resolution
mechanisms) and explicitly defer acc/m?**

### Decision 9: Where in the API surface does the table accessor live?

Pending Decisions 1–4, the user needs to decide what concrete shape goes
where. Some options:

- New module `Tensile/Components/register_resolver.py` exposing
  `AllocationTable.from_writer(writer) -> AllocationTable`.
- Method on `KernelWriter`: `writer.snapshot_allocation_table() -> AllocationTable`.
- Method on `RegisterPool`: rejected (Q2 finding — wrong data shape).
- Free function: `make_allocation_table(writer) -> AllocationTable`.

### Decision 10: Test corpus shape (precondition for Phase 3)

Bead description's Phase 3 test list assumes:

> - Symbolic→numeric resolution: a `Register` constructed from a symbolic
>   container gets resolved to numeric form using a representative allocation
>   table.

This presupposes Decision 1 chose path (a) (fold to numeric). If Decision 1
chooses (c) (`PhysicalRegister` token), the test corpus needs to be
re-specified before Phase 3 starts.

---

## Hard rules respected by this Phase 1 work

- Phase 1 is research only. **No code changes.** This memo is the only artefact.
- `Register.from_rocisa(rc)` stays single-arg. Not modified.
- c70 commit (`449fadd493`) and the rationale for dropping `allocation_table`
  are referenced in the "Why this epic exists" section above.
- Phase 2 questions are pre-staged here so the user can review and weigh in
  efficiently when this comes back to them.
