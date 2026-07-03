# Contributing to the engine (the dual-backend contract)

This is the guide for contributors who touch the **kernel-generation engine
itself** — the IR builder, the helpers, the instance builders, or the LLVM
lowering. If you only want to *author and tune kernels*, start with
[`onboarding.md`](./onboarding.md) and [`extending.md`](./extending.md); you may
never need this page. If you are about to edit anything under `rocke/core/`,
`rocke/helpers/`, `rocke/instances/`, or `Cpp/`, read this first — it
describes a contract that is easy to break silently and expensive to break.

## The one thing to know: there are two engines

The engine exists **twice**, on purpose:

| | Path | Role |
|---|------|------|
| **Python engine** | `rocke/` | The authoring frontend. You write specs and builders here; it lowers IR → LLVM text. |
| **C++ engine** | `Cpp/` | A faithful port of the same lowering, compiled into `librocke_core.a`. It is what the hipDNN provider links so kernels can be served with **no Python at runtime**. |

The two are kept **byte-for-byte identical at the LLVM-IR level**: for every
kernel family and every config, the `.ll` text the Python engine emits and the
`.ll` text the C++ engine emits are the *same bytes*. This is not an aspiration —
it is an enforced invariant, and it is the reason the provider can be trusted.

**The practical consequence for you:** a change to the Python lowering that is
not mirrored in the C++ side (or vice versa) breaks the contract. The build will
still pass. The Python tests will still pass. Nothing will look wrong — until the
differential gate (below) catches it, or worse, a provider ships a kernel that no
longer matches. So the definition of "done" for an engine change is not "my
tests pass"; it is **"the differential gate is still green."**

See [`invariants.md`](./invariants.md) for the full list of rules this contract
implies (including a genuinely sneaky compiler-evaluation-order trap that has
bitten this codebase before).

## Definition of done

An engine change is done when **all** of these hold:

1. The Python unit tests pass (see [`testing.md`](./testing.md)) — most run
   without a GPU; a handful of harness/timer tests need one.
2. The C++ engine builds clean.
3. The **differential `.ll` gate is green** — C++ emission matches Python
   emission, byte for byte, across every family. This is the contract.
4. If you *intended* to change emitted output, the golden snapshot is re-blessed
   in the same change, with the diff reviewed.

Steps 2–3 are a single command (next section). Make it a habit; run it before you
consider a change finished.

> ⚠️ **Do not `ruff check --fix` emitter code.** The IR builder is
> **side-effecting**: ruff's `F841` autofix deletes "unused" `b.const_i32(...)` /
> `b.mul(...)` assignments that actually emit ops, silently breaking byte-identity.
> Lint with `ruff check` (no `--fix`), add `# noqa: F841` for intentional handles,
> and **re-run the gate below after any lint or format pass** — treat a lint pass
> as a code change, never a safe no-op. (See [`invariants.md`](./invariants.md)
> rule 9.)

## The gate: prove your change is byte-safe

One script builds the C++ engine and runs the full cross-engine `.ll` diff:

```bash
tools/check_byte_identity.py
```

It prints a per-family GREEN/MISMATCH table and exits non-zero on any **real**
mismatch. A green run means the two engines still agree everywhere.

If you prefer to run the steps by hand (they are what the script does):

```bash
# From dnn-providers/hip-kernel-provider/rocKE/Python  (use the project venv with ROCm+PyTorch).
PY=$(pwd)                      # .../dnn-providers/hip-kernel-provider/rocKE/Python

# 1. Build the C++ engine archive into a LOCAL dir (never NFS — it is slow).
cmake -S "$PY/Cpp" -B /tmp/rocke_verify -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rocke_verify -j

# 2. Run the differential .ll gate (default archive is /tmp/rocke_verify/librocke_core.a).
PYTHONPATH="$PY" python3 -m tests/instances/differential/run_diff.py --mode ll
```

Expected: every family GREEN, 0 real mismatches. Two families
(`fmha_appendkv`, `gfx1151_wmma_gemm_iu8_dequant`) may report a benign
`RANGE_DRIFT` (the C emitter enumerates a slightly wider config range; the
in-range bytes are identical) — that is pre-existing and not a failure.

### When the gate is red

A red gate means your change altered emitted bytes on one side but not the other.
To localize and understand it:

```bash
# Narrow to the offending family.
PYTHONPATH="$PY" python3 -m tests/instances/differential/run_diff.py --mode ll --only gemm

# See *structural* (real) drift vs incidental SSA-numbering drift.
PYTHONPATH="$PY" python3 -m tests/instances/differential/run_diff.py --mode ir --canonical --only gemm
```

`--mode ir --canonical` is the diagnostic lens: it separates a genuine
divergence (`STRUCT_DRIFT`) from harmless SSA-id renumbering (`CANON_EQUAL`).
Most "I changed the builder and the `.ll` is identical but the IR differs"
surprises are the latter — see [`invariants.md`](./invariants.md#the-ir-builder-must-emit-operands-left-to-right).

### Transitional state: when "red" vs this tree is expected

While engine-reconciliation work is in flight, the C++ engine may intentionally
match a reference that has not yet been merged into this tree's Python. The
clearest case is a deliberately **suppressed scheduler-hint sequence**: the C++
side emits no `sched.group.barrier` for GEMM-family kernels, so the default gate
(C++ vs *this tree's* Python) reports those families as drifted. **During that
window this is expected, not a regression** — the contract being upheld is "C++
matches the reconciliation reference," not "C++ matches the not-yet-updated local
Python."

To check against the reference tree instead of this one:

```bash
tools/check_byte_identity.py --ref-pyroot <reference-tree-python> --ref-shim <stubs>
```

A green run there is the authoritative signal. Once the reference is merged into
this tree, the default gate becomes authoritative again. If you see GEMM-family
`sched.group.barrier`-only drift and nothing else, suspect this before you
suspect your change.

## The three gate modes

`run_diff` compares the two engines at three depths. Use the one that matches what
you changed:

| Mode | Compares | Use when you changed… |
|------|----------|------------------------|
| `--mode ll` | final LLVM-IR text (the contract) | anything in lowering or builders — this is the gate of record |
| `--mode ir` | the `ck.dsl.ir/v1` serialized IR (upstream of lowering) | a builder, to see drift *before* the lowerer absorbs it |
| `--mode verify` | the IR verifier's diagnostics | the verifier or validity gates |

Add `--canonical` to `--mode ir` to re-triage byte differences into "real" vs
"just SSA numbering". Add `--only <substr>` to focus on families.

## Golden snapshots (intentional output changes)

The gate also has a committed golden snapshot so output changes can't slip in
unnoticed:

```bash
# Fail if any family's emission differs from the blessed golden.
PYTHONPATH="$PY" python3 -m tests/instances/differential/run_diff.py --check-golden

# Re-bless — ONLY from a green state, as part of a reviewed, intended change.
PYTHONPATH="$PY" python3 -m tests/instances/differential/run_diff.py --record-golden
```

If you deliberately change what a kernel emits, re-blessing the golden is part of
the change, and the diff is what your reviewer reads. Re-blessing to make a red
gate go green *without understanding why it went red* is how the contract dies.

## The C++ binding catalog

The Python ↔ C++ engine equivalence is also exercised through the pybind binding
(`Cpp/bindings/`), which exposes each family's lowering to Python. If you add
or change an instance family, validate the binding still produces identical
emission:

```bash
python3 Cpp/bindings/prove_parity_binding.py   # per-family VALIDATED / mismatched
```

## Where a change lands on both sides

[`extending.md`](./extending.md) walks the Python side (a new IR op touches
`core/ir.py`, `core/ir_print.py`, `core/lower_llvm.py`, and tests). The contract
adds one rule: **the same lowering must exist in the C++ engine** under the
mirrored path in `Cpp/` (the directory layout mirrors the Python module
tree). The gate is what tells you the two sides agree.

## Building the full provider stack

To build the engine, the hipDNN provider plugin, and the end-to-end graph demos
in one go, use the canonical build script and its reference doc:

```bash
tools/check_byte_identity.py        # see BUILD.md for options
```

`BUILD.md` is the authority on build flags and artifact hygiene; it exists
because every historical "it doesn't build / it's broken" turned out to be a
stale or mis-flagged build product used silently. If something that should work
doesn't, read [`troubleshooting.md`](./troubleshooting.md) before you debug the
code — the cause is usually the build, not the source.

## Checklist before you call it done

- [ ] Python unit tests pass (`test_rocke.py`; most no-GPU, ~20 need a GPU).
- [ ] `tools/check_byte_identity.py` is green (engine builds + `.ll` gate).
- [ ] If output changed on purpose: golden re-blessed, diff reviewed.
- [ ] New/changed instance family: `prove_parity_binding.py` validates the binding.
- [ ] No new internal-tracker nomenclature in code comments or docs (functionality only).
