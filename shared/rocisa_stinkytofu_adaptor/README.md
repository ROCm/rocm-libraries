# rocisa_stinkytofu_adaptor

A transient Python shim that lets Tensilelite (`projects/hipblaslt/tensilelite/`)
talk to the [stinkytofu](../stinkytofu/) Python binding (`_stinkytofu.so`) while
pretending to be the older `rocisa` package.

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

## Why this lives outside `shared/stinkytofu/`

This package is a *consumer* of the stinkytofu Python binding, not part of
stinkytofu itself. Keeping it as a sibling of `shared/stinkytofu/` (rather
than nested inside `shared/stinkytofu/python_module/`) makes that direction
of dependency explicit and avoids the import-shadow trap where a sibling
`stinkytofu/` source directory would otherwise hide the user-built
`_stinkytofu.so` on `sys.path`.

## Scope

* gfx1250 only today. Other gfx generations stay on the native `_rocisa.so`.
* Transient. Expected to be deleted / drastically shrunk once gfx1250
  KernelWriter is rewritten against a real stinkytofu service API.

## Layout

```text
shared/rocisa_stinkytofu_adaptor/
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
python3 shared/rocisa_stinkytofu_adaptor/tests/test_register.py
python3 shared/rocisa_stinkytofu_adaptor/tests/test_argument_loader.py
```

Each test self-bootstraps `sys.path`, so no install / `PYTHONPATH` setup
is required. With pytest:

```bash
pytest shared/rocisa_stinkytofu_adaptor/tests/
```

## Smoke check (stinkytofu backend wired in)

```bash
ROCISA_BACKEND=stinkytofu PYTHONPATH=<rocisa_build_dir> \
    python3 -c "import rocisa; print(rocisa)"
```

This exercises the dispatcher path:
`rocisa/__init__.py → rocisa_stinkytofu_adaptor`.
