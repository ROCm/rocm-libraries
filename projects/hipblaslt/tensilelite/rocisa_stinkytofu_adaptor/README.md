# rocisa_stinkytofu_adaptor

A transient Python shim that lets Tensilelite (`projects/hipblaslt/tensilelite/`)
talk to the [stinkytofu](../../../../shared/stinkytofu/) Python binding
(`_stinkytofu.so`) while pretending to be the older `rocisa` package.

## What

`rocisa_stinkytofu_adaptor` mirrors the public surface of
`projects/hipblaslt/tensilelite/rocisa/rocisa` (the nanobind C++ bindings)
so KernelWriter callers can keep using `from rocisa import ...` unchanged.

The dispatcher lives at `projects/hipblaslt/tensilelite/rocisa/rocisa/__init__.py`.
Set `ROCISA_BACKEND=stinkytofu` to activate this adapter; anything else (or
unset) keeps the original native bindings.

```bash
export ROCISA_BACKEND=stinkytofu
```

## Why this lives next to `tensilelite/rocisa/` (and not under `shared/stinkytofu/`)

This package is a *consumer* of the stinkytofu Python binding, not part of
stinkytofu itself. Two consequences drive the placement:

* **Dependency direction is one-way**: tensilelite → adapter →
  (native rocisa | stinkytofu). `shared/stinkytofu/` must stay neutral
  for any future consumer; it does not need to know that a rocisa-shaped
  adapter exists. Putting the adapter under `shared/stinkytofu/` would
  invert that direction.
* **Lifecycle is tensilelite-local**: when KernelWriter eventually calls
  stinkytofu directly, this folder gets deleted with no impact on
  `shared/stinkytofu/`. Sibling-of-rocisa makes it visually obvious it is
  the "alternative backend" to `tensilelite/rocisa/`.

## Scope

* gfx1250 only today. Other gfx generations stay on the native `_rocisa.so`.
* Transient. Expected to be deleted / drastically shrunk once gfx1250
  KernelWriter is rewritten against a real stinkytofu service API.

## Layout

```text
projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/
├── rocisa_stinkytofu_adaptor/     # Python package (mirrors rocisa submodules)
│   ├── __init__.py                # rocIsa singleton shell + re-exports
│   ├── base.py                    # Item, rocIsa state sink, isaToGfx
│   ├── caps.py / enum.py          # dynamic HW caps + IntEnum shims
│   ├── register.py                # RegisterPool (real)
│   ├── container.py               # RegisterContainer, vgpr/sgpr/… factories
│   ├── code.py                    # Module, Macro, Signature*, KernelBody, …
│   ├── label.py                   # LabelManager (real)
│   ├── instruction.py             # Instruction bases, MacroInstruction, VMovB32 (partial)
│   ├── functions.py               # ArgumentLoader offsets real; emit stubs
│   ├── macro.py                   # dummy macro builders only (Macro* lives in code.py)
│   ├── asmpass.py                 # rocIsaPass port (partial; some passes stub)
│   ├── stinky_interop.py          # toStinkyTofuModule + signature emit wrapper
│   └── _dummy.py                  # make_dummy_* factories
├── tests/
│   ├── test.sh                    # wrapper: PYTHONPATH + unittest discover
│   ├── test_base.py
│   ├── test_register.py
│   ├── test_container.py
│   ├── test_code.py
│   ├── test_label.py
│   ├── test_instruction.py
│   ├── test_functions.py
│   ├── test_macro.py
│   ├── test_asmpass.py
│   └── test_emission_consistency.py
└── README.md
```

## Tests — 1:1 shim ↔ test mapping

| Shim module | Test file |
|-------------|-----------|
| `base.py` | `test_base.py` |
| `register.py` | `test_register.py` |
| `container.py` | `test_container.py` |
| `code.py` | `test_code.py` |
| `label.py` | `test_label.py` |
| `instruction.py` | `test_instruction.py` |
| `functions.py` | `test_functions.py` |
| `macro.py` | `test_macro.py` |
| `asmpass.py` | `test_asmpass.py` |

`test_emission_consistency.py` is the cross-module integration suite (not tied
to a single shim file).

## Running the tests

Use `test.sh` only — it sets `PYTHONPATH` and discovers the binding build
so callers do not run `python3 test_*.py` or `pytest` directly.

From the tests directory (or any cwd):

```bash
cd projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests

./test.sh                 # all test_*.py (unittest discover)
./test.sh -v              # same, verbose
./test.sh test_register   # one file only — basename of test_*.py, no .py suffix
./test.sh --help
```

The second argument to `./test.sh` must be a **test file** name (`test_register`,
`test_code`, …), not an individual test method
(`test_kernarg_address_at_index_0` will not work).

`test.sh` handles the environment: it auto-discovers
`<tensilelite>/*build*/tensilelite/rocisa`, sets `PYTHONPATH` to
`<binding_root>:<tensilelite>`, then runs `python3 -m unittest discover`.
No manual `PYTHONPATH` or `STINKY_BUILD_DIR` is needed in the normal case.
See `./test.sh --help` for optional overrides.

### Python version must match the build

Extension modules are tagged by CPython ABI, e.g.
`_stinkytofu.cpython-312-x86_64-linux-gnu.so` only loads under **Python 3.12**.
If `python3` and the build disagree, integration tests are **skipped** (not
failed). After `./test.sh -v`, expect `OK (skipped=0)` when the binding matches.

## Smoke check (stinkytofu backend wired in)

```bash
ROCISA_BACKEND=stinkytofu PYTHONPATH=<build>/tensilelite/rocisa:<tensilelite> \
    python3 -c "import rocisa; print(rocisa)"
```

This exercises the dispatcher path:
`rocisa/__init__.py → rocisa_stinkytofu_adaptor`.
