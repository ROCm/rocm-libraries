# Engine Parity Mandate — every Python optimization must exist in C++ too

**Read this before adding or tuning any kernel.** It applies to humans and to AI
coding agents working in this repository.

## The rule

The CK DSL ships **two peer engines** that must stay equivalent:

- **Python** (`rocke/`) — the authoring frontend and the differential oracle.
- **C++** (`cpp/`) — a first-class runtime engine. The hipDNN provider links
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

- A new or changed **instance** (`rocke/instances/**` ⇄ `cpp/instances/**`).
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
   `rocke/...` ↔ `cpp/...`).
2. Run the differential gate and confirm the affected families are byte-identical:
   ```bash
   tools/check_byte_identity.py        # builds the C engine + run_diff --mode ll
   ```
   Every family must be GREEN (C-emitted `.ll` == Python-emitted `.ll`). See
   [`engine_contributing.md`](./engine_contributing.md) for modes, the golden
   snapshot, and the per-flavor note.
3. For a new/changed instance family, validate the binding too:
   ```bash
   python cpp/bindings/prove_parity_binding.py
   ```
4. Run the whole suite through the differential lane, which lowers every kernel
   the tests build with *both* engines and compares the bytes:
   ```bash
   ROCKE_BACKEND=both python -m pytest tests -rs
   ```
   This is broader than the gate: the gate covers a fixed family list, while
   `both` covers whatever the tests happen to build.
5. If you *intend* to change emitted output, re-bless the golden snapshot in the
   same change and have the diff reviewed.

A change is **done** only when the byte-identity gate is green for everything you
touched.

## The differential lane never falls back

`ROCKE_BACKEND=both` will **not** substitute the Python result when the C++
engine fails to lower a kernel. That would report an uncompared kernel as
parity-verified — the exact failure the lane exists to catch, and an invisible
one, since the caller still gets plausible IR and the suite still goes green.

So a cpp-side failure always propagates. There is one classification, not an
escape hatch: if the arch is named in `core/backend.py::CPP_UNPORTED_ARCHES`
the error is re-raised as `BackendCoverageGap`, which `tests/conftest.py` turns
into a **skip** so the gap is counted and named in the run summary. Anything
else stays a failure.

`CPP_UNPORTED_ARCHES` is the single place a gap is declared, and it is **empty**:
every arch the Python engine lowers, the C++ engine lowers too, so no kernel can
reach the skip path. gfx1250 was the last entry and came off when
`LL_BACKEND_GFX1250` landed in `cpp/core/lower_llvm/core.cpp`. Deleting an entry
is what "ported" means — the lane then holds that arch to full byte-identity.
Never add an entry to quiet a failure on an arch the C++ engine is supposed to
serve; that failure is a regression.

Note the scope of that claim: it is about the **lowerer**. The C++ *instance
builders* under `cpp/instances/` still have no `gfx1250/` directory, so the
gfx1250 kernels are authored in Python and lowered by the C++ engine — a
different axis from the instance-builder families, which compare a C++ builder
against its Python twin.

The lowerer axis is anchored in the byte-identity gate on its own terms. A
parity family does not have to come from an instance builder: any
`tests/instances/parity/<name>_emit.{c,py}` pair is picked up automatically, so
a family can build a kernel straight from each language's IR builder and
byte-compare the lowered `.ll`. `gfx1250_lowering` is that shape — one config
per place `Gfx1250Backend` diverges from its parent (K=32 and K=64 WMMA, the
element-typed `ds_load_tr16_b128`, the split barrier drains, the two wait
counters). Four of those divergences are a *choice* between two encodings, so
each is paired with its gfx950 twin: a lowering that ignored the backend and
hardcoded either form would pass one config and fail its partner.

## Directive for AI agents

If you add or tune a kernel/optimization in `rocke/` (Python), you **must** add
the equivalent in `cpp/` (C++) in the same change and verify byte-identity
with the gate above. Do not report a Python-only optimization as complete. If you
cannot mirror it in C++, stop and surface that explicitly rather than leaving the
engines divergent. The same applies in reverse (a C++-only change that the Python
oracle doesn't match).

See also: [`invariants.md`](./invariants.md) (the byte-identity contract and the
arg-eval-order trap) and [`engine_contributing.md`](./engine_contributing.md)
(the full contributor workflow and the gate).
