# rocKE — Python style guide

Conventions for the `rocke` Python code — the platform authoring frontend
(`platform/python/rocke/`) and the `library/` package. These rules
are **derived from the existing code**, so new code is indistinguishable from what is
already here — **except** for typing and other modern-language choices, where the
guide **prescribes the modern standard** (see the PEP baseline below and §3) even when
the current majority is an older form. The intent is that new code converges on modern
Python rather than perpetuating legacy idioms; we do **not** maintain backwards
compatibility with pre-3.10 forms.

This is the human-readable companion to the tooling. Formatting is owned by
**black** (wired into the repo-root `.pre-commit-config.yaml`); linting is done
with **ruff check** (see AGENTS.md — currently run manually, config not yet
committed). Wiring ruff into hooks is deferred; this document defines the rules
regardless of when enforcement lands.

Run locally before pushing (from `platform/`):

```bash
black python/rocke                      # format
ruff check python/rocke                 # lint — do NOT pass --fix here (see §9)
# full hook set over your PR range, from the repo root (see AGENTS.md):
pre-commit run --from-ref origin/develop --to-ref HEAD
```

> If a rule here ever conflicts with `AGENTS.md`, `AGENTS.md` wins — it carries the
> hard invariants (byte-identity, relative paths, cross-platform). This guide
> covers *style*; `AGENTS.md` covers *correctness*.

---

## Cardinal rules (never break)

These must never be broken; a violation is always Critical. The confidentiality rules
are owned by `AGENTS.md`; the engineering rules are repeated here because routine Python
edits violate them easily.

- **Compliance** (export controls, restricted data, NPI, product/marketing/code names,
  performance data, legal/marketing, internal links) — defined in AGENTS.md
  "Compliance", the source of truth. Not restated here.
- **Never autofix emitter code** (`core/`, `helpers/`, `instances/`). The IRBuilder is
  side-effecting — `b.const_i32(8)` emits an op even if its handle is unused — so
  `ruff --fix` / F841 autofix silently changes kernels. Lint without `--fix`; see §9.
- **Byte-identity.** Any change to emitted IR must be mirrored in the C++ engine and
  re-verified with `python tools/check_byte_identity.py`.
- **Relative paths only.** No absolute repo paths, nothing escaping `platform/`; anchor
  on `__file__` (enforced by `tests/run_all.py`). See §10.
- **Cross-platform, Python-only scripts.** No `.sh`/shell-only flows; use `pathlib`,
  `tempfile`, `shutil.which`. See §10.

## Modern Python baseline (PEPs we encourage)

The package targets **Python ≥ 3.10**. Prefer modern language/typing features; do not
add code that maintains compatibility with pre-3.10 idioms.

- **Style & docs:** PEP 8 (style — black/ruff enforce most), PEP 257 (docstrings),
  PEP 20 (the Zen — the tie-breaker for judgment calls).
- **Typing (enforced; see §3):** PEP 585 builtin generics (`list[int]`), PEP 604 unions
  (`X | None`), PEP 563 `from __future__ import annotations`, PEP 484/526 (hints &
  variable annotations). Prefer PEP 544 Protocols for structural interfaces and
  PEP 613 `TypeAlias` for named aliases.
- **Language features:** PEP 557 dataclasses, PEP 498 f-strings, PEP 634–636 structural
  pattern matching (`match`/`case`) where it reads better than an `if/elif` chain.
- **Packaging:** PEP 621 (`[project]` metadata), PEP 517/518 (build backend).
- **Not yet — require newer than 3.10; do not use until `requires-python` is bumped:**
  PEP 695 (`type` / `class Foo[T]`, 3.12), PEP 692 (`Unpack` kwargs, 3.11),
  PEP 698 (`@override`, 3.12).

---

## API design (the cross-language standard)

How functions are layered, named, exported, and tested follows the shared
[API design standard](API_DESIGN_STANDARD.md). That document owns the rules; this guide owns
how they are expressed in Python:

- **Exports and the author door** — a module's `__all__` is the surface shared with sibling
  modules; the package `__init__.py` `__all__` is the author door (§8). Also export the result
  type a function is named with (standard Part 2).
- **Naming, `_`-private state, read-only `@property` accessors** — §4, §7.
- **Validation return types** — `(ok, reason)` for a simple gate; a three-tier diagnostic when
  the caller must tell a reorder-fixable warning from a hard error (§7).
- **Test coverage** — §11.

Repo protocols the standard depends on (Part 4 — no paths in signatures, numpy-only public
signatures, `platform/` not importing `library/`) are owned by
[AGENTS.md](../platform/AGENTS.md). Enforcement of these rules is pending; the standard's
[What is enforced](API_DESIGN_STANDARD.md#what-is-enforced) is the single home for that status.

---

## 1. Formatting (owned by black — do not hand-fight it)

- **Line length 88**, **double quotes**, black defaults. No `[tool.black]` overrides
  exist; keep it that way so the repo-wide hook stays authoritative.
- Run `black` before committing; never manually reformat to something black would
  undo.
- Use **f-strings** for interpolation (they are the norm across emitters, e.g.
  `f"  {op.result.name} = call <4 x float> @llvm.amdgcn.mfma..."`). Avoid `%` and
  `str.format`.

## 2. Imports

- **`from __future__ import annotations`** at the top of modules is the prevailing
  convention — keep using it (it makes annotations lazy strings and lets you
  reference types without runtime import cost).
- **Relative imports within the package**, using explicit dot-depth to the target
  layer. This is universal here:
  ```python
  from ..ir import Op                       # core/isa/backend.py
  from ...core.arch import ArchTarget       # instances/common/mfma_gemm.py
  from ...helpers.atoms import MfmaAtom
  ```
  Do **not** use absolute `rocke.core...` imports inside the package, and never a
  hardcoded/`sys.path` hack (see AGENTS.md "relative paths only").
- Group imports stdlib → third-party (`numpy`, `torch`) → intra-package, matching
  black/ruff-isort ordering. Keep them at module top; local imports inside a
  function are reserved for **breaking import cycles** or **optional deps** (e.g.
  `from ..core.arch import ArchTarget` inside `is_valid_spec` to avoid a cycle) —
  this is an accepted, deliberate pattern, not a default.

## 3. Type annotations — modern only

The package targets **Python ≥ 3.10** and modules carry `from __future__ import
annotations` (PEP 563), so annotations are lazy strings and modern forms cost nothing
at runtime. **Write modern typing; do not maintain the legacy `typing`-module forms.**

- **Annotate every public function signature and dataclass field.** Untyped public
  API is out of place in this codebase.
- **Builtin generics (PEP 585), not `typing` aliases:** `list[int]`, `dict[str, int]`,
  `tuple[int, ...]`, `set[str]` — **never** `List`, `Dict`, `Tuple`, `Set`.
- **Union with `|` (PEP 604), not `Optional`/`Union`:** `X | None`, `X | Y` —
  **never** `Optional[X]` or `Union[X, Y]`.
  ```python
  def is_valid_spec(spec: MfmaGemmSpec, arch: str = "gfx950") -> tuple[bool, str]:
  def mfma_k_loop(..., initial_acc: Value | None = None) -> Value:
  ```
- **`collections.abc`, not `typing`, for ABCs:** `Callable`, `Iterable`, `Sequence`,
  `Mapping` come from `collections.abc`.
- **Legacy `typing.Optional/List/Dict/Tuple/Union/Set` is deprecated here.** The bulk
  of existing code predates this policy and still uses the capitalized forms; do **not**
  add new ones, and modernize opportunistically when you touch a file (don't do
  unrelated mass rewrites in a feature change).
- This is enforced by `ruff` pyupgrade rules `UP006` (PEP 585), `UP007`/`UP045`
  (PEP 604), `UP035` (deprecated `typing` imports) once the ruff config lands; until
  then it is a manual rule.
- Use the domain types (`Value`, `VectorType`, `PtrType`, `KernelDef`) rather than
  bare `Any` when a real type exists. `Any` is fine for genuine polymorphism
  (e.g. `mma(self, op: Any, ...)` accepting `MmaOp | str`).

## 4. Naming

| Kind | Convention | Example |
|---|---|---|
| Function / method / variable | `snake_case` | `build_mfma_gemm`, `lane_decode` |
| Class / dataclass | `PascalCase` | `MfmaGemmSpec`, `ISABackend`, `MfmaAtom` |
| Module-private helper | `_leading_underscore` | `_binop`, `_operand`, `_mma_c_frag_len` |
| Module-private constant | `_UPPER_SNAKE` | `_SUPPORTED_DTYPES`, `_CATALOG_DTYPE` |
| Public constant | `UPPER_SNAKE` | `F16`, `I32`, `BF16` |

Follow the established **verb prefixes** — they are load-bearing signals of a
function's role, and readers rely on them:

- `build_*` → constructs and returns a `KernelDef` (`build_mfma_gemm`).
- `is_valid_spec` / `validate_*` → validation; see §7 for the return contract.
- `lower_*` / `_op_<opname>` → lowering. Op handlers are named
  `_op_<op.name with '.'→'_'>` because dispatch is reflection:
  `getattr(self, f"_op_{op.name.replace('.', '_')}")` (`core/lower_llvm.py`). If you
  add an op named `tile.foo`, its handler **must** be `_op_tile_foo`.
- `emit_*` → writes IR/text for one construct (`ISABackend.emit_mma`, `atom.emit`).
- Factory `@classmethod`s return a configured instance and are named for the shape
  they produce (`MfmaAtom.f16_16x16x16()`), not `from_*`/`make_*` — match the
  neighbours in the same class.

### Variable naming (kernel-domain axes)

Name kernel-authoring locals after the **problem's own domain axes**, not generic
`M`/`N`/`K` — those are reserved for the MMA atom shape (`MMA_*`). `TILE_*` and `WAVE_*`
tile the **problem axes**, not `M`/`N`/`K`.

The example column below uses an **attention / FMHA** kernel, whose axes are query
tokens (`Q`), key/value tokens (`KV`), and head dimension (`D`). A different problem
(GEMM, conv, …) uses its own axis names following the same pattern.

| Prefix | Level | Meaning | Example (FMHA) |
|---|---|---|---|
| `Q_` / `KV_` / `D_` | problem axis | the problem's own axes; never `M`/`N`/`K` | `Q`, `KV`, `D` |
| `*_LEN` | tensor | full baked extent of an axis | `Q_LEN`, `KV_LEN`, `D_LEN` |
| `TILE_*` | CTA | per-CTA tile size **along a problem axis** | `TILE_Q`, `TILE_KV` |
| `WAVE_*` | wave | per-wave extent **along a problem axis** | `WAVE_Q`, `WAVE_KV` |
| `MMA_*` | hardware | the fixed MMA atom shape — the **only** place `M`/`N`/`K` appear | `MMA_M`, `MMA_N`, `MMA_K` |
| `N_*` | count | integer count of things | `N_WAVES`, `N_KV_TILES` |
| `*_MMAS` | count | number of MMA output subtiles covering an axis | `Q_MMAS` |
| `<GEMM>_STEPS` | count | number of MMA contraction steps in a named GEMM | `QK_STEPS`, `PV_STEPS` |
| lowercase | runtime | per-lane/per-wave SSA locals | `lane`, `wave` |

Head dimension: the axis is `D` and its full extent is `D_LEN`; `HEAD_D` is a common
synonym — pick one per kernel and keep it consistent.

## 5. Docstrings

Docstrings here are **substantive and explain the WHY**, not just the signature.
This is a deliberate house style for a codebase full of hardware subtleties.

- **Module docstring** on every non-trivial module: what it is and how it fits the
  `Spec → build_* → KernelDef → lower → .ll` pipeline.
- **Public functions, classes, dataclasses, and op handlers get docstrings.** State
  the contract, the units, and any hardware invariant a caller could get wrong.
- **Encode rationale and provenance.** The codebase routinely cites *why* a value is
  what it is — lane layouts, register widths, and correctness traps:
  ```python
  """K-packed f16 atom on gfx950+. K=32/atom in two halves.
  A lane `c4 = lane / 16` holds K = [c4 * 8 : c4 * 8 + 8]
  (NOT the flat-concat layout ...; the wrong packing compiles, runs, and
  validates within 1e-2 but fails at 1e-3)."""
  ```
- **Reference tags are allowed** where they add provenance a reader can actually
  resolve: runbook sections (`dsl_docs/optimization/runbook_compliance.md`) and
  change/phase IDs like `P53`/`P88` (rocke's internal work-item tags). Only use a tag
  if a reader can trace it; keep tags in the docstring/comment, not in identifiers.
- Trivial private one-liners may have a single-line docstring or none — don't pad.

## 6. Comments

- Default to **no comment**; add one only when the *why* is non-obvious (a hardware
  constraint, a subtle invariant, a correctness trap, a deliberate cycle-breaking
  import). The existing comments are almost all "why", e.g. the note that
  `b.const_i32(8)` emits an op even when its handle is unused.
- Don't describe *what* well-named code already says, and don't reference the
  current task/PR in a comment — that belongs in the commit message.

## 7. Dataclasses, specs, and validation

- **`@dataclass(frozen=True)` is the default** for specs and value objects
  (`MfmaGemmSpec`, `MfmaAtom`). Immutability is intentional — specs are hashed and
  used as cache/dispatch keys.
- Put **derived quantities behind `@property`**, not in `__init__`
  (`MfmaGemmSpec.atom`, `.tile_k`, `.block_size`). Keep stored fields minimal and
  compute the rest.
- **Validation returns `tuple[bool, str]`** — `(ok, reason)` with a structured,
  actionable reason string; builders call it and raise on failure:
  ```python
  ok, why = is_valid_spec(spec, arch=arch)
  if not ok:
      raise ValueError(f"invalid mfma_gemm spec for {arch}: {why}")
  ```
  Which exception to raise, how to word the message, and where `__post_init__` checks
  go versus `is_valid_spec` are owned by section 13.
- **Layered checks that distinguish a reorder-fixable warning from a hard error** return a
  three-tier diagnostic (a severity-carrying result) instead of `(ok, reason)`. That form is
  owned by the [API design standard](API_DESIGN_STANDARD.md#part-1--layering--naming), not
  restated here.

## 8. Public API surface (`__all__`)

- Modules that expose a public surface define **`__all__`** (used in ~90 modules).
  If you add a public function/class to such a module, add it to `__all__`; keep
  genuinely internal helpers out of it (and `_`-prefixed).
- `__init__.py` re-exports the layer's public names; wire new public API through the
  nearest package `__init__.py` (e.g. a new builder in `instances/common/` is exported
  from `instances/common/__init__.py`) rather than expecting callers to reach into
  submodules. Reusable kernels must also be wired into the registry/tests (AGENTS.md).

## 9. Emitter rules (core / helpers / instances) — read before linting

The IR builder is **side-effecting**: calling `b.const_i32(8)` appends an op to the
kernel even if you discard the returned handle. This changes how you lint and edit
emitter code:

- **Never run `ruff check --fix` (or any autofix) on `core/`, `helpers/`,
  `instances/`.** F841 "unused local" autofix will delete a binding whose *call*
  was the point, silently changing the emitted kernel. Lint with `ruff check`
  (no `--fix`) and fix findings by hand. (This is a hard rule from `AGENTS.md`.)
- When a bound value is intentionally unused (ABI params, ordering ops), mark it
  explicitly the way the code already does — `# noqa: F841 - ABI` — rather than
  removing it:
  ```python
  _M = b.param("M", I32)  # noqa: F841 - ABI
  ```
- **Byte-identity**: any change to emitted IR must be mirrored in the C++ engine in
  the same change and re-blessed against the gate. A "style" cleanup that alters
  emission is not a style change. When in doubt, run the gate from `platform/` (a built
  C++ engine is required first — see AGENTS.md "Build / test / run"):
  `python tools/check_byte_identity.py`.

## 10. Cross-platform & paths

- Python only — **no `.sh`/bash helper scripts** under `platform/`. Use `pathlib`,
  `tempfile`, `shutil.which`, `os.cpu_count()`; never hardcode `/tmp`, `nproc`,
  `sudo`, or shell-only flows.
- **No absolute repo paths and nothing escaping `platform/`.** Anchor on `__file__`.
  `tests/run_all.py` enforces this with a grep guard.

## 11. Tests

Test-file naming and the PR-bot test requirement are owned by AGENTS.md. The test-coverage
principles — prove the behaviour, choose cases by consequence, a self-validating setup — are in
the [API design standard](API_DESIGN_STANDARD.md#part-5--testing-what-you-build). The Python
mechanics:

- **`pytest`** with `@pytest.mark.parametrize` to sweep the axes that change behaviour (shape,
  dtype, atom count), not one case.
- **Prefer a real input over a replacement.** Push the actual object through the real path; a
  replaced dependency tests the replacement, not the code.
- **Most tests are CPU-only** (lowering/serialization/byte-identity); gate GPU-only numeric
  tests on real hardware being present rather than faking it.
- **Assert the rejections too.** Every guard gets an error-condition test, not just the
  passing case; what that test must check is in section 13.6.

## 12. A minimal conforming module

A new `instances/` builder that assembles the rules above (body elided):

```python
"""Foo GEMM builder — Spec -> build_* -> KernelDef (host authoring)."""
from __future__ import annotations

from dataclasses import dataclass

from ...core.ir import IRBuilder, KernelDef
from ...helpers.atoms import MfmaAtom

__all__ = ["FooSpec", "build_foo", "is_valid_spec"]


@dataclass(frozen=True)
class FooSpec:
    M: int
    N: int
    dtype: str = "f16"

    @property
    def atom(self) -> MfmaAtom:
        return MfmaAtom.f16_16x16x16()


def is_valid_spec(spec: FooSpec, arch: str = "gfx950") -> tuple[bool, str]:
    if spec.M % spec.atom.m:
        return False, f"M={spec.M} must be a multiple of atom.m={spec.atom.m}"
    return True, "ok"


def build_foo(spec: FooSpec, arch: str = "gfx950") -> KernelDef:
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid foo spec for {arch}: {why}")
    b = IRBuilder(f"foo_{spec.M}x{spec.N}_{spec.dtype}")
    # ... emit body via helpers (loads, mfma_k_loop, epilogue) ...
    return b.kernel  # the KernelDef
```

Then export `FooSpec`/`build_foo` from `instances/common/__init__.py` and add a
`test_foo.py`. Verify against the current API before copying — this is a shape guide,
not a working kernel.

## 13. Error handling

This section is policy: it says how new and changed code reports bad input. Bringing
existing code in line with it is tracked separately; do not mass-rewrite unrelated
checks or messages in a feature change. Each bullet is one self-contained rule, so it
can be applied (or extracted by a tool) on its own. A rule in its fullest form reads:

```text
RULE:  Raise ValueError, never `assert`, for a bad value reaching a public API.
WHEN:  A public function or dataclass gets an argument of the right type but an
       invalid value.
DO:    raise ValueError(f"split_k failed - tile_k={tile_k}, expected a multiple of 16")
DON'T: assert tile_k % 16 == 0
WHY:   `python -O` strips assert statements, so the check silently disappears.
```

### 13.1 Principles

- **Fail fast at the API boundary.** Reject bad input where it enters a public function
  or dataclass, before any IR is emitted, so the error names the caller's argument
  rather than surfacing later as a broken kernel or a lowering failure.
- **Validation costs nothing in the kernel.** Checks are host-side Python and must never
  emit IR: do not call the builder (`b.*`) inside a check (builder calls are
  side-effecting, section 9). Validate once, in the constructor, factory, or public
  entry point, not inside the loop that emits ops per element, per atom, or per K step.
- **Contracts are enforced at runtime.** A constraint stated in a docstring or a type
  hint is also checked in code if breaking it would produce a wrong kernel.

> **Warning -- adding a check can break a caller.** A kernel that happened to work with
> an input outside the documented contract will start raising once that contract is
> checked. This is an intended, accepted behaviour change: such a kernel was relying
> on undefined behaviour. Say so in the commit message and fix the caller; do not
> loosen the check to keep it passing.

### 13.2 Error messages

- **Use the template `{what failed} - {param}={bad value}, expected {constraint}`.**
  Name the offending parameter, show the value it had, and state what was required.
- **List the valid options when the choice is from a fixed set** (`expected one of
  ('workgroup', 'agent', 'system')`), so the fix is in the message.
- **Use `!r` for strings and other non-numeric values** so an empty or whitespace
  value is visible.
- **For an `(ok, reason)` gate (section 7)** the reason string carries
  `{param}={bad value}, expected {constraint}` and the builder that raises on it
  supplies `{what failed}`.

```python
# Good: names the parameter, the value, and the constraint.
raise ValueError(
    f"make_tile failed - tile_k={tile_k}, expected a multiple of atom.k={atom.k}"
)

# Banned: says something is wrong, not what or how to fix it.
raise ValueError("invalid shape")
```

### 13.3 Which exception

| Exception | Raise it when | Example |
|---|---|---|
| `ValueError` (the default) | The argument has the right type but a bad value. | `raise ValueError(f"make_tile failed - tile_k={tile_k}, expected > 0")` |
| `TypeError` | The argument is the wrong kind of object. | `raise TypeError(f"load_tile failed - view={type(view).__name__}, expected TensorView")` |
| `KeyError` | A name is looked up in a registry or table and is not there. | `raise KeyError(f"lookup failed - arch={arch!r}, expected one of {sorted(ARCHES)}")` |
| `NotImplementedError` | The input is valid but this code path is not written yet. | `raise NotImplementedError(f"no HIP lowering for op {op.name!r}")` |
| `RuntimeError` | Something outside the caller's arguments failed, or the engine broke its own invariant. Raise the existing subclass where there is one. | `raise ComgrError(...)` (a `RuntimeError` from `runtime/comgr.py`); `HipError` (`runtime/hip_module.py`) for a HIP API failure |

- **Never use `assert` to validate input to a public API.** `python -O` removes assert
  statements, so the check vanishes in optimized runs. `assert` is fine in tests and for
  internal sanity checks whose loss would not let bad input through.
- **When callers need to catch one failure specifically, subclass the matching
  built-in** exception (as `ComgrError(RuntimeError)` and
  `PinRefused(ValueError)` do), so a caller catching the built-in still catches it.

### 13.4 Where validation goes

- **Public functions validate every input at entry, before emitting IR.**
- **Dataclasses validate their own fields in `__post_init__`.** A spec that can be
  constructed is internally consistent (see `QkScaleSpec` in `helpers/qk_scale.py`).
  In a `frozen=True` dataclass, assign a normalized field with `object.__setattr__`.
- **Checks that need context the dataclass does not hold** (the target arch, another
  spec) go in `is_valid_spec` and are raised by the builder (section 7).
- **A `_`-prefixed private function may skip validation** only when its public caller
  has already checked the same input, when it runs inside an IR-emitting inner loop, or
  when its docstring states the precondition it relies on.
- **Rely on duck typing; add an `isinstance` check only where a wrong type would
  otherwise fail with a cryptic error far from the call** (an `AttributeError` deep in
  lowering, a silently wrong offset). No blanket `isinstance` guards on every argument.
- **Check values in public APIs, highest-risk first:** block and grid sizes, vector
  widths, tile sizes, and other numeric parameters that a wrong value turns into a
  wrong kernel rather than an exception.

### 13.5 Parity with the C++ engine

- **A check mirrored in the C++ engine raises the same message text in both.** The C++
  side of the contract (throw sites, the `extern "C"` boundary, status codes) is owned
  by [CPP_STYLE.md section 6](CPP_STYLE.md#6-error-model-the-engines-core-contract). The
  Python `raise` and the C++ call site look like this:

```python
if scope not in _FENCE_SCOPES:
    raise ValueError(f"unknown fence scope {scope!r}")
```

```cpp
if(!rocke_i_fence_scope_ok(scope))
{
    (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "unknown fence scope '%s'", scope);
    return;  // not reached: rocke_i_set_err throws (ckc::ValueError)
}
```

- **The exception type picks the status code:** `ValueError` -> `ROCKE_ERR_VALUE`,
  `TypeError` -> `ROCKE_ERR_TYPE`, `KeyError` -> `ROCKE_ERR_KEY`,
  `NotImplementedError` -> `ROCKE_ERR_NOTIMPL`. There is no `RuntimeError` code.
- **`{x!r}` on a string matches `'%s'` in C**, as long as the value holds no quote or backslash.
- **Changing a mirrored message changes both engines in the same change**, along with
  any test that matches on it.

### 13.6 Testing error conditions (mandatory)

- **Every added or hardened check gets a test** that asserts (a) the exception type and
  (b) a message that names the bad value. Use `pytest.raises(Exc, match=...)`; a bare
  `pytest.raises(Exception)` passes on any failure, including a typo in the test.
- **Cover each kind of check:** choice from a fixed set, wrong type, value or range, and
  `__post_init__` validation of a dataclass.
- **`match` is a regular expression** (`re.search`): escape `(`, `)`, `[`, `.`, or use
  `re.escape` on a literal message.

```python
# Before: passes on any exception, and never says which input was rejected.
def test_qk_scale_spec_rejects_bad_input():
    with pytest.raises(Exception):
        QkScaleSpec(layout="per_tile")


# After: pins the type and the message for each rejected input.
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"layout": "per_tile"}, r"layout must be 'per_head' or 'per_block', got 'per_tile'"),
        ({"layout": "per_block", "scale_block": 0}, r"scale_block must be > 0 \(got 0\)"),
    ],
)
def test_qk_scale_spec_rejects_bad_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        QkScaleSpec(**kwargs)
```

### 13.7 Type hints and runtime checks

- **Type hints are required on public signatures (section 3) and do not replace
  runtime checks.** Python does not enforce hints at runtime; a check from section 13.4
  still applies where a bad value would produce a wrong kernel.

---

## Quick checklist for a new Python change

- [ ] `black` clean (88 cols, double quotes, f-strings)
- [ ] `from __future__ import annotations`; relative intra-package imports
- [ ] Modern typing only — `list`/`dict`/`tuple`/`X | None`
- [ ] Public signatures typed; docstrings explain the *why* + any hw invariant
- [ ] Verb-prefix naming (`build_*`/`is_valid_*`/`lower_*`/`emit_*`/`_op_<name>`)
- [ ] Specs are `frozen=True`; derived fields via `@property`
- [ ] Validation returns `(ok, reason)`; exception type + message per section 13
- [ ] Every new or hardened check has a test asserting type and message (section 13.6)
- [ ] Public names added to `__all__` / `__init__.py`
- [ ] Emitter code: **no autofix**; `# noqa: F841 - ABI` for intentional unused ops
- [ ] Emission changed? mirror in C++ + re-run the byte-identity gate
- [ ] No absolute paths, no shell scripts
