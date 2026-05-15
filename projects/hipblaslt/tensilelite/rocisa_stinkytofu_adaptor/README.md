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
├── rocisa_stinkytofu_adaptor/        # Python package
│   ├── __init__.py                   # rocIsa / IsaInfo singletons
│   ├── base.py / caps.py / enum.py   # real implementations
│   ├── register.py / functions.py    # real implementations
│   ├── code.py                       # SrdUpperValue (gfx1250)
│   ├── _dummy.py                     # factory for not-yet-real shims
│   └── …                             # rest still dummies (see __init__.py)
├── tests/
│   ├── test_register.py
│   └── test_argument_loader.py
└── README.md                         # this file
```

No CMake or `pyproject.toml`: deliberately minimal infrastructure because
the package is transient (see *Scope* above) and lives on `PYTHONPATH`
managed by the rocisa dispatcher.

## Running the tests

```bash
python3 projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_register.py
python3 projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_argument_loader.py
```

Each test self-bootstraps `sys.path`, so no install / `PYTHONPATH` setup
is required. With pytest:

```bash
pytest projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/
```

## Smoke check (stinkytofu backend wired in)

```bash
ROCISA_BACKEND=stinkytofu PYTHONPATH=<rocisa_build_dir> \
    python3 -c "import rocisa; print(rocisa)"
```

This exercises the dispatcher path:
`rocisa/__init__.py → rocisa_stinkytofu_adaptor`.
