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

## Rebuilding after C++ changes

`invoke rocisa` only needs to be re-run when `pyproject.toml` or `CMakeLists.txt`
change. For C++ source edits, rebuild the extension directly:

```bash
cmake --build <build_dir> --target _rocisa
```

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

For more information, see `docs/`.
