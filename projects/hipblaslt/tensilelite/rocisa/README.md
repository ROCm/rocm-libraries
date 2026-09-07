# rocisa

A Python/C++ code generator for ROCm ISA, built with nanobind.

## Developer Setup

Install rocisa as an editable package using the invoke task from the tensilelite root:

```bash
cd rocm-libraries/projects/hipblaslt/tensilelite
invoke rocisa
```

This compiles the C++ extension and installs it into your active venv so that
`import rocisa` works from anywhere — no `PYTHONPATH` required.

For compatibility, the editable install rebuilds on rocisa's first import in
each Python process. That behavior is deprecated because an import should not
perform a native build. Opt out now with:

```bash
invoke rocisa --no-rebuild-on-import
```

After the future default change, use `--rebuild-on-import` only when that
automatic incremental rebuild is explicitly desired.

> **Linux only.** The `invoke` dev workflow (`invoke rocisa`, `invoke build-client`)
> is supported on Linux (a ROCm dev container) only: it uses `amdclang`, defaults to
> `/opt/rocm`, and resolves dependencies via RPATH — none of which apply on Windows.
> On Windows, rocisa is built by the integrated CMake build (TheRock / CI), which is
> where its Windows dependent-DLL handling lives.

## Rebuilding after C++ changes

See [tensilelite/README.md — "Rebuilding rocisa after C++ changes"](../README.md#rebuilding-rocisa-after-c-changes)
for the full rebuild instructions, including which cmake build directory to use
depending on how rocisa was installed.

Importing rocisa with stale bindings raises an `ImportError` with a clear rebuild
hint, so you will not silently use an out-of-date extension.

## Building independently (without tensilelite)

```bash
cd rocisa
pip install -e .
```

scikit-build-core handles the cmake configuration and compilation automatically.
Requires the ROCm SDK (`amdclang++`) and `/opt/rocm` on the default search path,
or set `ROCM_PATH`.

For a direct editable install, disable the deprecated import rebuild with:

```bash
SKBUILD_EDITABLE_REBUILD=false pip install -e .
```

For more information, see `docs/`.
