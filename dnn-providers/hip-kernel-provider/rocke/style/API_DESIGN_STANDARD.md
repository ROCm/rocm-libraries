# rocKE — API design standard

How functions are layered, named, exported, and tested in `rocke`, independent of
language. The [Python](PYTHON_STYLE.md) and [C++](CPP_STYLE.md) guides carry the
per-language mechanics; this doc carries the shared shape both must follow. Goal: a
reader can tell what a function takes, what it answers, and where it belongs — from the
signature alone — and a fix *composes* the existing pieces instead of re-implementing or
duplicating them.

> If a rule here conflicts with [`platform/AGENTS.md`](../platform/AGENTS.md), `AGENTS.md`
> wins — it owns the hard invariants (byte-identity, relative paths, cross-platform).

## Contents
- [How to read this](#how-to-read-this)
- [Part 1 — Layering & naming](#part-1--layering--naming)
- [Part 2 — What's public](#part-2--whats-public)
- [Part 3 — Finding your way](#part-3--finding-your-way)
- [Part 4 — rocke's hard protocols](#part-4--rockes-hard-protocols)
- [Part 5 — Testing what you build](#part-5--testing-what-you-build)
- [Part 6 — Self-check before done](#part-6--self-check-before-done)
- [Worked example](#worked-example--the-mma-validation-layer)
- [What is enforced](#what-is-enforced)

## How to read this

**The short version: today, one guard test actually fails the build — the one-way
dependency (plus a single-owner check). The other checkable rules are reviewer warnings
until a guard is added for each. The live-versus-planned list is in
[What is enforced](#what-is-enforced).**

Each rule is tagged **(checkable)** — a reviewer can decide pass/fail from the code — or
**(judgment)** — it needs discussion, not a mechanical check. A rule being checkable does
not mean the current tree already passes it; several rules the tree still breaks.

Two layer words recur: a **fundamental** does one indivisible piece of work on the base
type; a **wrapper** adapts a fundamental's interface and delegates with no logic of its
own; a **composed entry** runs several fundamentals in a fixed order and is the single
front door for the whole operation.

---

## Part 1 — Layering & naming

**1. A function takes the base type it operates on. (checkable)**
If the caller has to access fields inside a wrapper object to pass the function the real
value, the function was written at the wrong layer.
```
mma_operand_layout_sound(layout, canon, role)   # good — operates on a layout, takes one
mma_operand_layout_sound(plan, role)            # bad  — extracts plan.a_desc.layout itself
```
*Exception — factories.* A function that *builds* a new value takes the ingredients, not
the product, and checks no input: `canonical_layouts(traits, subtiles)` returns layouts,
so it cannot take one. A function that returns a *result about* an input is not a factory
— it operates on that input and obeys rule 1.

**2. Names reveal the thing and the check. (judgment)**
A reader should learn what a function handles, and for a check what it verifies, without
opening the body — you should be able to call it correctly with the body hidden.
- No bare noun as a name: `operand` alone says nothing. `mma_operand_layout_sound` says
  it — the *operand layout*, checked for *soundness*. ("operand" as a qualifier is fine;
  only standalone is banned.)
- A check names its subject and its result: `mma_pair_k_aligned`, not `check_mma`.
- Reserve the `test_` prefix for functions under a `tests/` tree (the one part of this rule
  a grep can enforce).

**3. A thin wrapper has zero logic. (checkable)**
Field access plus one delegating call — no conditional, loop, arithmetic, or validation.
The moment a wrapper grows a branch it is a fundamental in disguise; name and place it as
one.

**4. One composed entry; no caller re-sequences the fundamentals. (checkable)**
When a composed entry exists, no call site combines two or more of its fundamentals
manually. Manual assembly is how a step gets silently dropped: the day someone adds a
check, the callers that forgot it keep compiling. This is the accumulator-validation
regression — the issue point ran the operand checks that were coded there and never the
accumulator check, because no single entry ran the whole set.
*Exception — sanctioned lowest-layer code.* A few call sites deliberately author at the
rawest layer (the hand-sequenced raw-`b.mma` paths). They are exempt from rules 4 and 9 —
and take **zero** operand/accumulator validation by design. Exempt, not protected; the
exemption covers only paths explicitly sanctioned as lowest-layer code.

**5. No duplicate for an existing question. (judgment; the alias half is checkable)**
Do not add a function that answers what an existing one already answers. Never ship an
alias to keep an old name alive — rename every call site and delete the old name.
```
operand_sound = mma_operand_layout_sound   # bad — an alias that splits callers in two
```

**6. Name the producer/acceptor role at each call site. (judgment)**
The same layout math runs in two roles: a **producer** checks descriptors it just built
(and may skip checks guaranteed by construction); an **acceptor** checks values it was
handed (and may not). State the role so a later reader does not merge the two and weaken
one.

**7. Glass vessel — a helper calls the fundamental, never copies its body. (judgment)**
Convenience helpers are built on the fundamentals and expose the common path; a copied
body is a second source of truth that becomes outdated. ("All knobs public" is about
reachability, not about removing `_`-private internals.)

**8. Glass-box composite — expose the parts, never make them inaccessible. (checkable via per-case guard)**
When an object is assembled from lower-layer pieces, those pieces stay publicly reachable,
so a user can always drop a layer and work with them directly. `TileMma` exposes `.plan`
and `.driver`; it adds convenience on top, it does not hide them behind a closed interface.

**9. Build fixes by composing the fundamentals. (judgment)**
Add behaviour as a new check in the composed entry or a new fundamental it calls — not a
special-case at one call site. Its checkable part is rule 4.

**Validation return types. (return shape checkable; which bucket a check belongs in is judgment)**
A check whose caller must tell a reorder-fixable *warning* from a hard *error* returns a
three-tier diagnostic carrying a severity — not the plain `(ok, reason)` pair a simple
pass/fail check uses (see
[`PYTHON_STYLE.md §7`](PYTHON_STYLE.md#7-dataclasses-specs-and-validation)). Collapsing the
first into the second discards the distinction the caller needs.

---

## Part 2 — What's public

Mostly checkable. (Python mechanics — `__all__`, `_`-underscore, `@property` — in
[`PYTHON_STYLE.md §8`](PYTHON_STYLE.md#8-public-api-surface-__all__).)

- **The surface a module shares with its siblings is not the public author door. (judgment)**
  The package front door is a separate, deliberate opt-in — a name being importable by a
  sibling file does not make it author-facing.
- **Import from the layer's one door, never a submodule's internals.** Bypassing the door
  couples you to machinery the layer never promised to keep stable (the import-path form of
  rule 4). Sibling imports *within* a layer are fine; crossing *into* another layer's
  internals is not.
- **State is private; callers get a read-only accessor** — no public mutable field.
- **If you export a function, export the type its result is named with** (project-defined
  types, not builtins), so a caller can name the result without reaching into a submodule.

---

## Part 3 — Finding your way

- **Search before you add. (judgment)** Grep the owning area for the concept and the verb
  that would name it before writing a helper — this is the *how* that makes rule 5
  achievable. rocke names are predictable (verbs `build_`/`is_valid_`/`verify_`, concepts
  are nouns), so the search is cheap.
- **A function lives where its concept is owned, not where the caller sits. (judgment)**
  The tiling split is essential: `analysis/` stays importable without matplotlib *because*
  drawing lives in `visualization/`.
- **The package door is a readable index. (checkable: names listed)** Every public name is
  listed, and the `__init__` docstring carries one line per concept saying which file owns
  it, so a reader finds the right file without opening any.

---

## Part 4 — rocke's hard protocols

Design that quietly breaks a repo invariant is unshippable:
- **No signature takes or returns an absolute or escaping path.** Anchor on the asset
  resolver or `__file__`. (checkable)
- **A `platform/` API never imports toward `library/`** — the dependency is one-way,
  including a deferred import inside a function body. (checkable)
- **Optional heavy dependencies stay behind a guarded import**; public signatures name only
  the mandatory dependency's types (numpy, not torch). (checkable)
- **If a call's side effect is the point, name it so** (`emit_*`/`build_*`), so a reader —
  and the linter — knows the call is the point and will not delete it. (judgment)
- **An emitter API must be reproducible byte-for-byte in the C++ engine.** No
  single-engine-only tricks (reflection, runtime introspection, dict-ordering); the
  byte-identity gate is the separate proof. (judgment)

---

## Part 5 — Testing what you build

Rounded coverage that proves what the code does. All judgment as written; for emitted
kernels the binding form is the on-GPU numeric-parity gate — see
[`KERNEL_AUTHORING.md` Core DoD](../KERNEL_AUTHORING.md#core-dod--every-change) (don't
restate it). Python test mechanics are in [`PYTHON_STYLE.md §11`](PYTHON_STYLE.md#11-tests).

- **Prove it, don't just make it look tested.** Assert the real produced result, not the
  spec or shape. If you could return a constant and the test stayed green, it proves
  nothing. (A geometry-only test stayed green while the accumulator check was missing; a
  real-result test would have gone red.)
- **Real input, truthful reference value.** Push the actual object through the real path;
  base the expected value on something built a *different* way (a hand-computed value, a
  byte-identity equal, a frozen hash) — never the code predicting its own output.
- **Parametrize the axes that change behaviour** — shape (square *and* rectangular), dtype,
  atom count, k-stacking — not just one easy case. Square-only coverage is what allowed the
  rectangular-wave regression.
- **Choose cases by consequence** — the boundary (0, 1, first/last), the degenerate case,
  and the near-miss that looks identical but isn't (transposed, off-by-one stride, swapped
  axes). Three cases that matter beat thirty trivial ones.
- **Self-validating setup.** Assert the fixture really is in the state you claim, and isn't
  already trivially true, before you test the behaviour — so it can't pass for the wrong
  reason.
- **Test the other side** — the round trip where there's an inverse, and the *rejections*
  (every guard gets a `raises(match=…)`), not just the passing case.
- **"Passes" is not "done."** Name a bug your suite would still miss; cover it or write down
  why not.

---

## Part 6 — Self-check before done

Run these on yourself before calling a design done:
- **Type the call site first, then count the lookups.** Write the line a user would type
  before you write the function; if using it needs more than one lookup, the surface is
  wrong. This is the honest "would I use this myself?" test.
- **Rebuild from the spec, not a nearby kernel.** A borrowed constant you can't justify from
  *this* spec is a latent bug.
- **Name the layer of every new line** (rules 4/9). "I put it here because it was faster"
  means it belongs in the composed entry.
- **The honesty round.** Write what you *want* true, what you *know* true (verified), what
  you're *unsure* of; if want and know differ, you're not done.

And: **repeated redesign requests mean you're solving the wrong problem, not solving the
right one badly.** Stop editing, write down the fixed facts versus your assumptions, and
hand that back.

---

## Worked example — the MMA validation layer

The MMA issue path is the reference implementation of Parts 1–5.

```
  canonical_layouts(traits, subtiles)          factory: builds the canonical A/B/C reference layouts
     |
     +--> mma_operand_layout_sound(layout, canon, role)                 one operand vs the machine
     +--> mma_accumulator_flow_consistent(c, a, b, a_canon,b_canon,c_canon)  C from the PASSED a, b
          mma_pair_k_aligned(a, b)                          A vs B agree on K   (takes no canon)
          _assert_atom_contiguous(a) / (b)                  atom-slice contract (A and B only)
                         |
                         v
  _validate_mma_issue(a_frag, b_frag, accumulator, plan)    composed entry: runs them all, once,
                                                            stopping at the first failure
  not in the recipe (separate issue-time gates): operand dtype agreement, backend-op resolution
```

- Each fundamental takes the base layout, not the plan (rule 1); `canonical_layouts` is the
  factory and the single home for the canonical reference layouts (rule 7 — every consumer
  calls it).
- The driver makes **one** call to `_validate_mma_issue` (rule 4). The accumulator
  regression existed because the issue path ran *no* accumulator check at all; the composed
  entry is the structural fix that stops a future check from being silently dropped (rule 9).
- `mma_accumulator_flow_consistent` derives C from the **passed** operands, so a correct
  hand-built C that differs from the plan's own descriptor is still accepted — it is checked
  against what the operands produce, not against the plan's style.
- The plan (producer) and driver (acceptor) run the fundamentals their contract needs — the
  producer skips the pair and contiguity checks it guarantees by construction (rule 6).
- The recipe covers the layout checks and atom contiguity; its membership is owned by
  [`tiling_api_contract.md`](../platform/python/rocke/helpers/tiling/docs/tiling_api_contract.md),
  so this section only illustrates the rules on it.

---

## What is enforced

A style doc gets skimmed; a failing build does not. So the checkable rules are backed by
guard tests — and the honest state of that today:

- **Shipped:** the one-way dependency (`library/tests/test_library_layering.py`, an AST walk
  with a `KNOWN_VIOLATIONS` allowlist that is shrunk over time) and the single-owner grep
  guard (the sole fragment-slicer test).
- **To add — one guard per checkable rule:** import-from-the-door, machinery-off-the-author-door,
  export-the-return-type (project-defined types only), every-public-name-appears-in-the-index,
  no-escaping-paths, numpy-only signatures. Model each on the layering AST walk or the
  single-owner grep. The current tree breaks several, so **each new guard ships with an
  allowlist of today's violations that is shrunk over time (never added to)** — never a build
  that fails on day one. The index guard checks only that every public name is *listed*;
  whether the concept line reads true stays a reviewer call.
- **The style-guide reviewer** warns on the remaining checkable rules from the code.
- **Byte-identity is its own gate** (`tools/check_byte_identity.py`), not a guard test here.
- The **judgment** rules (Parts 5–6, much of 1 and 3) are raised in human review — not weaker
  for being unmechanized, but kept to one line each so they get read.
