# Engine invariants (the landmines)

These are rules that are **not obvious from reading the code** and that you will
silently violate if nobody tells you they exist. Each one has burned someone.
For every rule: what it is, *why* it exists, and *what breaks* if you ignore it.

This page is for engine contributors. If you are authoring kernels, you mostly
need [`onboarding.md`](./onboarding.md); come back here if you touch
`rocke/core/`, `rocke/helpers/`, `rocke/instances/`, or `Cpp/`.

---

## 1. LLVM-IR emission is byte-identical across the two engines

**The rule.** For every kernel family and config, the LLVM-IR text emitted by the
Python engine (`rocke/`) and by the C++ engine (`Cpp/`) must be the *same
bytes*.

**Why.** The C++ engine is what the hipDNN provider links so kernels can be served
with no Python at runtime. The only way to trust that the served kernel is the
one you authored and tested in Python is to guarantee they emit identically.
Byte-identity is the cheapest contract that gives that guarantee and is
mechanically checkable.

**What breaks if you ignore it.** Nothing visible — the build passes, the Python
tests pass — until the differential gate catches it, or a provider ships a kernel
that no longer matches the Python-validated one. A change to one engine that
isn't mirrored in the other is the single most common way to break the codebase.

**How to stay safe.** Run the gate (`tools/check_byte_identity.py`) as
your definition of done. See [`engine_contributing.md`](./engine_contributing.md).

---

## 2. The IR builder must emit operands left-to-right

**The rule.** When a builder helper constructs an op from sub-expressions, emit
(or bind to temporaries) the operands **in left-to-right source order**. Do not
write them as nested arguments and rely on the call expression to order them:

```cpp
// WRONG: argument evaluation order is unspecified in C++.
emit_mul(b, make_const(b, idx), make_const(b, stride));

// RIGHT: bind left-to-right, then combine.
auto a = make_const(b, idx);
auto c = make_const(b, stride);
emit_mul(b, a, c);
```

**Why.** C++ does **not** specify the evaluation order of function arguments — and
the compiler used for the C++ engine evaluates them **right-to-left**. Python
evaluates left-to-right. So `f(g(), h())` emits `g`'s instructions before `h`'s in
Python but *after* in C++. The two engines then assign SSA ids in different orders
and the IR diverges.

**What breaks if you ignore it.** This is the nastiest landmine in the codebase.
It is **not** undefined behavior — it's *unspecified* behavior, so sanitizers
(ASan/UBSan) will never flag it. The `.ll` may even stay byte-identical (the
lowerer renumbers SSA), so `--mode ll` can look green while `--mode ir` shows
dozens of families drifting. A single careless nested-arg call site can desync a
whole family's IR. This exact bug once affected ~40 families.

**How to catch it.** `run_diff --mode ir --canonical`. `CANON_EQUAL` = harmless
renumbering; `STRUCT_DRIFT` = a real divergence, usually this.

---

## 3. Build artifacts carry freshness, not the `.ll`

**The rule.** Never embed a build id, version stamp, timestamp, or content hash
**into the emitted `.ll`**. Freshness/version stamps belong on the *artifacts* —
the archive, the manifest, the HSACO note section — never in the kernel IR text.

**Why.** Anything embedded in the `.ll` would differ between two builds (or two
engines) and break byte-identity (rule 1). But staleness *does* need to be
detectable — hence stamps live on the artifacts around the IR, where they can
change freely without touching the contract.

**What breaks if you ignore it.** A well-meaning "add a build stamp to the kernel"
turns the byte-identity gate permanently red, or makes every build differ from the
golden snapshot.

---

## 4. The C header surface is a deliberately stable C ABI

**The rule.** The public engine entry points (`rocke_lower_kernel_to_llvm`,
`rocke_ir_serialize`, …) are `extern "C"` and unmangled. Don't give them C++
linkage, don't change their signatures casually, and don't rename the headers
under `Cpp/include/rocke/`.

**Why.** The pybind binding and the hipDNN provider link against these symbols by
their unmangled C names. C++ linkage would mangle them and break the `.so`/plugin
ABI. The headers are intentionally written as a C-compatible surface even though
every implementation file is now C++ — that is a feature (a checkable
public/internal boundary), not leftover C99.

**What breaks if you ignore it.** The provider or the pybind module fails to load
or link with undefined-symbol errors, often only at runtime.

**Note.** The `extern "C"` on the many *internal* helper headers is now vestigial
(no C consumers remain) but harmless; leave it. Stripping it is high-churn,
low-value, and out of scope for a normal change.

---

## 5. `Value.__bool__` raises — on purpose

**The rule.** IR `Value` objects are not truthy. `if some_value:` raises. Use the
explicit comparison/predicate builders (`b.cmp_lt(a, b)`, `b.select(...)`, etc.).

**Why.** A `Value` is a handle to an SSA result, not a Python bool. Allowing
`if value:` would silently take a host-side branch on a *compile-time* object,
producing a kernel that doesn't contain the comparison you meant. Raising forces
you to express control flow in the IR (`scf_if`, `select`) where it belongs.

**What breaks if you ignore it.** Without the guard you'd get kernels that compile
but compute the wrong thing, with no error. The raise converts a silent
correctness bug into an immediate, obvious one.

---

## 6. SSA values don't escape their region

**The rule.** A value assigned inside an `scf_if` / `scf_for` body does not flow
out of that region. To get a value *out*, thread it through the region's iter-args
(loops) or compute it unconditionally with `select` (conditionals).

**Why.** The IR is SSA with structured regions; a definition inside a region is not
in scope outside it. This mirrors MLIR's region semantics.

**What breaks if you ignore it.** Either a builder error, or you accidentally read
a stale/outer value. The classic case is a per-lane reduction or a masked load
where the tail value must be produced with `select(cmp, val, identity)` and always
combined, rather than assigned only inside an `if`.

---

## 7. Build on local disk; never trust a stale archive

**The rule.** Build into a local directory (e.g. `/tmp`), not NFS. And the provider
must link a **fresh** `librocke_core.a` built from current source — never a
checked-in or left-over archive.

**Why.** NFS makes the compiler and `comgr` pathologically slow. And the dominant
class of phantom "it's broken" failures in this codebase has been *a stale build
product used silently* — an old archive whose validator or lowering predates your
change, linked because a default path pointed at it.

**What breaks if you ignore it.** You debug a "bug" that is really the old binary;
you can lose a day to it. [`troubleshooting.md`](./troubleshooting.md) catalogs the
specific symptoms. `tools/check_byte_identity.py` and `BUILD.md` exist to make the
fresh-build path the only path.

---

## 8. Keep internal tracker nomenclature out of shipped artifacts

**The rule.** Code comments and user-facing docs describe **functionality and
usage only**. Don't write internal workstream/wave/RFC/task identifiers in
comments, READMEs, or docstrings.

**Why.** Those identifiers are meaningful only to the people running the current
program; to a future reader they are noise that ages badly. The architecture RFC
(and its status companion) is the single place that nomenclature belongs.

**What breaks if you ignore it.** Nothing functional — but the docs accrete
references that no one can resolve, which is exactly the "tribal knowledge" problem
this documentation set exists to fix.

---

## 9. Never auto-fix lint on emitter code — `ruff --fix` deletes side-effecting builder calls

**The rule.** **Do NOT run `ruff check --fix`** (and do not let an editor
auto-apply `F841` "unused variable" fixes) on emitter code — `rocke/core/`,
`rocke/helpers/`, `rocke/instances/`. Lint with `ruff check` *without* `--fix`;
for a genuinely-intentional unused handle add `# noqa: F841`; and **always re-run
the byte-identity gate after any lint or format pass** on those trees.

**Why.** In this DSL the IR builder is *side-effecting*: a statement like
`c8 = b.const_i32(8)` **emits an instruction into the kernel** even when the
Python handle `c8` is never read again. ruff's `F841` only sees an "unused
variable" and its autofix **deletes the whole assignment** — which removes the
emitted op and silently changes the kernel. The same hazard applies to `F401`
import removals when an import has load-time side effects.

**What breaks if you ignore it.** The lint "passes," the module still imports and
runs, but the emitted LLVM-IR is now different (a missing op / shifted SSA), so
the kernel is subtly wrong and the byte-identity gate goes red. This has already
bitten real kernels — a removed `b.const_i32(8)` and a removed
`b.mul(ml_base, ...)` were caught only because the byte-identity gate happened to
cover those families. Treat any lint/format pass on emitter trees as a change
that must be re-verified against the gate, never as a safe no-op.
