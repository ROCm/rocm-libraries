# Epic: make intrinsic availability decidable at build time

Planning document for the declaration-validation work. Five stories, each
ending with something that was not true before. No measured performance numbers
appear here by design.

Engineering detail: [`arch_axis_proposal.md`](arch_axis_proposal.md). Wider
toolchain context: [`multi_toolchain_strategy.md`](multi_toolchain_strategy.md).

---

## Epic

**Title:** rocKE LLVM-IR declaration validation — make intrinsic availability
decidable at build time

### The problem

rocKE's emitter writes `llvm.amdgcn.*` declarations into every kernel it
produces. Nothing verifies that those declarations exist for the target the
kernel is being built for.

Availability is the conjunction of two orthogonal axes:

```
available(key, arch, flavor) ≈ arch_domain(key, arch) ∧ exists(key, flavor)
```

* the **flavor axis** — does this LLVM vintage know this intrinsic name at all?
* the **arch axis** — can this GPU target actually lower it?

rocKE resolves declarations on the flavor axis only. **The arch axis is asked
nowhere at build time.** A kernel that names an intrinsic its target does not
have is emitted, accepted, and shipped.

### Why this is not caught today

The obvious checks do not work, and understanding why is the whole reason this
epic exists.

`opt -passes=verify` accepts a declaration for an intrinsic that does not exist.
So does `clang -S`: the backend treats an unrecognised `llvm.amdgcn.*` name as
an ordinary external function and emits a GOT-relative call to it. The module
compiles. The assembly looks plausible. **Only linking surfaces the problem**, as
`ld.lld: error: undefined symbol`.

Worse, some cases pass even the link. If ISel selects an instruction the
subtarget does not have — which happens today, in shipping LLVM, in both
directions — the result is either a compiler crash or an object that links and
cannot run.

So the defect class is silent by construction: it survives every check a
developer would naturally run, and surfaces as a link failure on someone else's
machine or a wrong answer on someone else's GPU.

### What this epic builds

1. **Measure** both axes against a real toolchain and commit the result as data
   that is gated against drift.
2. **Consume** that data at the emitter's demand-registration chokepoint, so a
   kernel asking for an unavailable intrinsic is diagnosed where it is written.
3. **Share** one table between the Python and C++ engines, so the byte-identity
   invariant survives, and then make the diagnostic binding.

### Value

Converts "links fine here, fails to link — or silently misbehaves — on that
other target" into a build-time diagnostic at the point of authorship.

### Constraints that shape every story

* **A host can only measure its own LLVM.** One artifact per flavor. A single
  shared file would mean whichever flavor ran last silently overwrote the
  others.
* **Not every non-`ok` answer is a negative answer.** `arch_absent` means the
  target genuinely cannot lower the intrinsic and is actionable.
  `target_unsupported` (the compiler does not know the target) and
  `toolchain_crash` (the compiler fell over) mean *no data*. Treating them as
  negatives would condemn 116 intrinsics on one target purely because an older
  compiler had never heard of it.
* **Error direction is asymmetric.** A false `arch_absent` breaks a working
  build — loud, annoying, safe. A false `ok` silently admits a kernel that
  cannot run. Every judgement call biases toward the loud failure, and the
  rollout is warn-first for the same reason.

---

## Story 1 — A trustworthy, drift-proof availability table

**Status: one item left.** Generator, both flavor columns, the gate and the
document corrections have landed. Outstanding: re-measure the nine
`-O0`-sensitive keys on `llvm20`, which is blocked on a host with ROCm 7.1
installed.

### Motivation

Everything downstream needs ground truth, and ground truth cannot be
hand-written. The decl table carries 148 keys across 7 wired targets: roughly a
thousand cells, each answerable only by asking the compiler. An earlier
hand-made probe table of 15 cells did not survive contact with a second flavor —
which is precisely the argument for generating it.

But generating it once is not enough. **A generated artifact that is not gated
drifts from its generator**, and a stale truth table is worse than none, because
it is trusted. The gate is also what keeps the generator honest: the
run-to-run instability described below was found only because a check re-ran the
generator and compared.

### Scope

* Generate, for each `(key, arch)`, a probe module from the emitter's own
  declaration; compile **and link** it; classify into `ok` / `name_absent` /
  `arch_absent` / `target_unsupported` / `toolchain_crash` / `probe_error`. One
  committed JSON artifact per LLVM flavor. *(done — `tools/gen_arch_domain.py`,
  columns for `llvm20` and `llvm22`)*
* Gate it. A structural half that runs with no toolchain and catches the
  realistic drift — a decl key added, renamed or removed while the artifact
  quietly stops covering it — plus a regeneration half that re-probes this
  host's flavor and requires the committed column back byte-identical. *(done —
  `tests/core/test_arch_domain_artifact.py`; the full sweep is ~15 s, so the
  oracle is a test rather than a nightly)*
* Correct the two existing arch-axis documents, which present the hand-made
  probe table as data and describe a schema and a taxonomy that are not what
  shipped. *(done)*
* Re-measure the nine `-O0`-sensitive keys on the vintage that has not yet been
  re-measured (see "the `-O0` lesson" below).

### Design decisions worth preserving

* **The IR-validity oracle must be a link**, for the reason given above.
* **LLVM's `report_fatal_error` kills the process**, so every probe runs in a
  subprocess.
* **The flavor axis is answered first, by `opt -S` round-trip.** LLVM resolves a
  recognised `llvm.*` declaration on parse — attaching the attribute group,
  remangling overloads, or letting AutoUpgrade consume the declaration entirely.
  An unrecognised name round-trips verbatim. This stage must come first because
  it is the only one that can answer a key whose codegen crashes: an unknown
  `llvm.*` name with a `metadata` operand is lowered through
  `TargetLowering::LowerCallTo`, and taking a metadata argument's alignment
  faults.
* **Probe operands must be real SSA values, not `poison`.** A poison `i32` sends
  one buffer intrinsic into a legalisation failure on a target where concrete
  integers lower cleanly — a false `arch_absent`.
* **Some operands must be literals even where the declaration does not say
  `immarg`.** One vintage fails to legalise; a newer one rejects it in the
  verifier. A literal retry covers both, and once every integer operand is a
  constant, a surviving legalisation failure is the target's answer rather than
  the probe's fault.
* **The `-O0` lesson.** At `-O3` the IR pipeline can delete the very call being
  asked about, and a module with nothing left to select links happily. This
  produced a false `ok` for a cross-lane intrinsic on a target that does not
  have it — the object contained no such instruction at all. Nine of the 148
  keys are foldable this way. The outstanding re-measurement is scoped to those
  nine keys: the rest had their calls survive `-O3` and were already measured
  honestly.
* **Transient failures must not be recorded as facts.** Under `-j` the probes
  fork enough linkers to occasionally hit the process limit; one abort was
  otherwise recorded as a permanent compiler crash. A serial retry pass
  separates real results (stable) from resource failures (not).
* **Every non-`ok` cell carries the compiler's own diagnostic.** A status with
  no evidence behind it cannot be acted on by whoever hits it, and cannot be
  told apart from a bug in the probe.

### Acceptance criteria

* One artifact per flavor; zero `probe_error` cells; regeneration byte-identical.
* The gate is red on a fabricated cell and on a decl key the artifact does not
  cover, and skips — never passes silently — where the flavor has no column.
* The compiler's self-reported vintage is cross-checked against the flavor rocKE
  resolved; disagreement is fatal, because otherwise the measurement is filed
  under the wrong LLVM.
* No numeric claim in either document lacks a generated artifact behind it.

---

## Story 2 — Make the emitter ask the table, warn-only

**This is the keystone of the epic.** Until it lands, the table is inert data
with no consumer, and data with no consumer rots.

### Motivation

The table's whole purpose is to be asked. The emitter already has exactly the
right place to ask: `_need()`, the single-line demand-registration chokepoint
that every intrinsic request passes through. All 129 call sites route through
it, and there is exactly one bypass that writes the demand map directly. That is
an unusually clean insertion point — the hard part of this kind of change is
usually finding a chokepoint that is actually a chokepoint, and here it already
exists.

The implicit target default is folded into this story rather than kept separate,
because it is not an independent cleanup: **a validation layer sitting on top of
a silent default is worse than no validation, because it looks like it
checked.** Three layers independently default the target when the caller
supplies none. Today that silently emits for one specific target. After this
story it would validate against a target the caller never chose, and report
confidently about the wrong thing. The two changes only make sense together.

### Scope

* Route the single bypass through `_need()`, so the chokepoint is airtight.
* Give `_need()` the target architecture and the committed table; warn on
  `arch_absent` with the key, the target, and the evidence.
* Remove the implicit target default in both engines; an unspecified
  architecture becomes an error rather than a guess, and callers relying on it
  pass the target explicitly.

### Why warn-only

The error-direction asymmetry. A false `arch_absent` would break builds that
work today; a false `ok` leaves us no worse off than the status quo. Warning
first lets the table be validated against real kernels before it can block
anyone. Story 3 promotes it once that confidence exists.

### Critical detail

`_need()` must distinguish the statuses. Only `arch_absent` is a negative
answer. `target_unsupported`, `toolchain_crash`, and a missing flavor column are
absence of data and must stay silent — otherwise a developer on an older ROCm
gets a wall of warnings for intrinsics that are fine.

### Acceptance criteria

* No path registers an intrinsic demand without passing through `_need()`.
* Requesting an `arch_absent` intrinsic warns, with actionable evidence.
* No-data statuses produce nothing.
* No implicit target default remains in either engine.
* **No emitted bytes change.** This story adds diagnostics and changes how the
  target is obtained, never what is emitted; byte identity green for every
  family at every flavor.

---

## Story 3 — One table for both engines, and make the diagnostic binding

### Motivation

Two motivations that share a delivery boundary: the check is not real until it
blocks, and it cannot safely block until both engines agree on what it says.

**One table.** The byte-identity invariant requires the Python and C++ engines to
emit identical bytes for every kernel family. Two independently maintained
availability tables would diverge, and the first divergence would be a
byte-identity failure whose root cause is a data file rather than the emitter —
an expensive thing to debug. One generated source, two consumers.

**Binding.** A warning that is never promoted is a warning that is eventually
ignored. Story 2 buys confidence in the table; this story spends it. Promotion
is deliberately gated on the shared table, because escalating to a hard error
while the two engines could disagree would turn a data-file skew into a build
outage.

### Scope

* Emit the C++ engine's table from the same artifact the Python engine reads, as
  a build step rather than a checked-in hand-maintained copy.
* Turn the `arch_absent` diagnostic into a build failure, with a documented,
  narrow escape hatch so a wrong row cannot block all work while it is fixed.

### Acceptance criteria

* One source of truth; the C++ side is generated, never hand-edited;
  regenerating updates both consumers with no manual step.
* Byte identity green for every family at every flavor.
* An `arch_absent` request fails the build with an actionable message.
* Promotion happens only after warn-only has run long enough to produce no false
  positives on real kernels — a judgement call that should be recorded, not
  assumed.

---

## Story 4 — Fix the declaration defects the table found

### Motivation

Three real defects, each found by the generator and by nothing else. Grouped
because they share a shape: the emitted declaration disagrees with reality, and
the fix is a decl-table correction mirrored in both engines.

**A tile-load declaration is `name_absent` on both vintages.** The emitter can
declare an intrinsic that no shipping LLVM knows; any kernel taking that path
cannot link.

**A scaled-MFMA declaration marks six trailing integer parameters `immarg` where
LLVM marks four.** This is not cosmetic: comgr verifies top-level declarations
**before** auto-upgrade runs, so the signature must be right as emitted, not
merely right after LLVM fixes it up. It is a real failure mode in the comgr path
even though the standalone compile path tolerates it.

**The intrinsic behind `math.tanh` is `arch_absent` on every wired target on both
vintages.** Not a target gap, not a vintage artifact — the op cannot work
anywhere and has presumably never been exercised. This is the clearest
demonstration of the epic's premise: a defect that no existing check catches,
sitting in the tree, found only by asking the compiler directly.

### Scope

* Remove the dead tile-load declaration in both engines, make the default
  backend's corresponding spec raise rather than emit it, drop the alias, and
  keep the working path for the target that has a real equivalent.
* Align the scaled-MFMA declaration with LLVM's actual intrinsic signature.
* Determine the correct tanh lowering and implement it in both engines, or
  remove the op if it has no users.

### Acceptance criteria

* Both engines updated in the same change for each fix.
* The tile-load fix is a zero-emitted-byte change — nothing that links today
  should change.
* Where output legitimately changes, the golden is re-blessed in the same
  change.
* A parity/verification run covers tanh if it stays.

---

## Story 5 — Report the LLVM subtarget-predicate crashes upstream

### Motivation

Two intrinsics are selected by ISel for targets whose ISA does not contain the
corresponding instruction. The illegal machine instruction then reaches the
first pass that needs an instruction size, and the compiler segfaults computing
it — a kernel with no branches at all crashing in the branch-relaxation pass.

The two cases point in opposite directions (a CDNA-only instruction selected for
RDNA targets, and an RDNA-only instruction selected for CDNA targets) and in
different vintages, which suggests a class of missing subtarget predicate rather
than a one-off. Expected behaviour is a "cannot select" diagnostic, not a crash.

Worth doing on its own merits, and it also retires cells in our own table that
currently read `toolchain_crash` — no data — where the real answer is almost
certainly `arch_absent`.

### Scope

A minimal reproducer per case, root-cause evidence showing the illegal
instruction present after ISel, and the affected-target matrix. Reproducers and
a report draft already exist outside the repo.

### Acceptance criteria

* Reproducers are minimal and verified with the exact commands in the report.
* Reproduced against an upstream build before filing upstream — the crashes were
  found with a vendor toolchain, and a fork-only issue should not go upstream.
* Public-disclosure review of target names before filing.
* One caveat found the hard way: an optimising pipeline masks one of the two
  crashes by folding the call away, so the reproducer must not rely on it.

---

## Ordering

```
1 (finish: the llvm20 re-measure, when a ROCm 7.1 host is available)
2 (keystone — turns the table into a gate)
3 (shared table, then binding)
4 (independent defect fixes, any time)
5 (independent, external, any time)
```

Story 2 is the keystone and is no longer blocked: the table's correctness for
this host's flavor is gated. Stories 4 and 5 are independent of the others and
can run in parallel. The outstanding re-measurement in Story 1 is gated on
toolchain availability, so take it opportunistically whenever ROCm 7.1 is
installed.

## Outstanding gate

The byte-identity check has not been run on the machine used for Story 1 — no
cmake available there. Story 1 changed no emitter code, so no byte change is
expected, but this is the project's primary invariant and must be verified
rather than reasoned about before any of this merges.
