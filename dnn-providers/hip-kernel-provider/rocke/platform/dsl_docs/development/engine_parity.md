# Engine Parity Mandate — every Python optimization must exist in C++ too

**Read this before adding or tuning any kernel.** It applies to humans and to AI
coding agents working in this repository.

## The rule

The CK DSL ships **two peer engines** that must stay equivalent:

- **Python** (`rocke/`) — the authoring frontend and the differential oracle.
- **C++** (`Cpp/`) — a first-class runtime engine. The hipDNN provider links
  it and can build *and* lower any kernel at runtime with **no Python present**.

Therefore:

> **Any optimization, kernel, instance, atom, op, fusion, arch-specific path, or
> performance tuning added to or changed in the Python engine MUST have the
> equivalent in the C++ engine, in the same change, proven byte-identical at the
> LLVM-IR level by the differential gate.**

A Python-only optimization is an **incomplete change**. If it isn't mirrored in
C++, the C++ runtime (what actually ships in the provider) emits a different —
usually slower or wrong — kernel than the one you authored and benchmarked in
Python. The whole value of the dual-engine design rests on the two emitting the
*same bytes*.

## What "an optimization / case" means here

All of these require a matching C++ change:

- A new or changed **instance** (`rocke/instances/**` ⇄ `Cpp/instances/**`).
- A new **MFMA/WMMA atom** or dtype, or an **arch-specific path** (e.g. an RDNA4
  `gfx1201` WMMA atom) — both engines, plus the arch tables.
- A new **op** in the IR (`core/ir.py` ⇄ the C lowering) or a **helper**.
- A **pipeline / epilogue / fusion** change, an **LDS layout / swizzle** change,
  a **scheduling** change — anything that alters emitted IR.
- A **correctness or arg-eval-order fix** in a builder (the C++ compiler evaluates
  args right-to-left vs Python left-to-right — the classic divergence).

If your change alters what the Python engine *emits*, it must alter the C++
engine identically.

## How to satisfy it

1. Make the change on **both** sides (the trees mirror each other:
   `rocke/...` ↔ `Cpp/...`).
2. Run the differential gate and confirm the affected families are byte-identical:
   ```bash
   tools/check_byte_identity.py        # builds the C engine + run_diff --mode ll
   ```
   Every family must be GREEN (C-emitted `.ll` == Python-emitted `.ll`). See
   [`engine_contributing.md`](./engine_contributing.md) for modes, the golden
   snapshot, and the per-flavor note.
3. For a new/changed instance family, validate the binding too:
   ```bash
   python Cpp/bindings/prove_parity_binding.py
   ```
4. If you *intend* to change emitted output, re-bless the golden snapshot in the
   same change and have the diff reviewed.

A change is **done** only when the byte-identity gate is green for everything you
touched.

## Directive for AI agents

If you add or tune a kernel/optimization in `rocke/` (Python), you **must** add
the equivalent in `Cpp/` (C++) in the same change and verify byte-identity
with the gate above. Do not report a Python-only optimization as complete. If you
cannot mirror it in C++, stop and surface that explicitly rather than leaving the
engines divergent. The same applies in reverse (a C++-only change that the Python
oracle doesn't match).

See also: [`invariants.md`](./invariants.md) (the byte-identity contract and the
arg-eval-order trap) and [`engine_contributing.md`](./engine_contributing.md)
(the full contributor workflow and the gate).
