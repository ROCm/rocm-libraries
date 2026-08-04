<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite

TensileLite is hipBLASLt's Python generator, logic validator, and tuning
workflow. Released Python wheels are part of a matched ROCm
artifact set: the wheel's `+rocmA.B.C` version must match
`$ROCM_PATH/.info/version`, and ROCm owns `tensilelite-client`. `rocisa` is a
separately prepared Python dependency; TensileLite requires it to be importable
but does not prescribe its ABI or native-artifact layout.

## Supported interface

```bash
tensilelite create-library --help
tensilelite logic --help
tensilelite run --help

# Equivalent module form
python -m tensilelite --help
```

The default wheel exposes `import tensilelite`; it does not provide the legacy
`Tensile` namespace or `Tensile/bin` launchers. An optional
`tensilelite-tensile-compat` wheel supplies deprecated command aliases only.

## Released installation

Use the wheel index delivered with the target ROCm release, then select that
same ROCm installation:

```bash
export ROCM_PATH=/opt/rocm
python -m pip install --index-url <rocm-wheel-index> tensilelite
python -c 'import tensilelite, rocisa; print(tensilelite.__version__)'
```

Import fails deliberately when the wheel and ROCm release differ, when rocisa
cannot be imported, or when the ROCm-owned client is missing. Released wheels
have no runtime client-path override.

Optional runtime capabilities remain available as extras:

```bash
python -m pip install 'tensilelite[profile]'     # yappi profiling
python -m pip install 'tensilelite[hip-query]'   # hip-python GPU queries
python -m pip install 'tensilelite[orjson]'      # preferred JSON accelerator
python -m pip install 'tensilelite[ujson]'
python -m pip install 'tensilelite[simplejson]'
```

Only one JSON extra is needed. If multiple backends are installed, TensileLite
prefers orjson, then ujson, then simplejson, and finally the Python standard
library.

## Source development

From a Linux ROCm development environment, the one-command setup installs the
shared development requirements and editable rocisa, builds/stages
`tensilelite-client`, and installs TensileLite editably into the active Python
environment:

```bash
cd rocm-libraries/projects/hipblaslt/tensilelite
invoke install --gpu-targets gfx942
```

The editable installation records the staged client's absolute path. Python
source edits are immediately visible; rerun `invoke build-client` after client
source or CMake changes to rebuild and refresh that same staged path.

Each step remains available independently. A manual source install may bind any
existing client executable:

```bash
python -m pip install -r requirements-dev.txt
invoke build-client --gpu-targets gfx942

python -m pip install --no-build-isolation --no-deps -e . \
  --config-settings="tensilelite.client-path=$PWD/build_tmp/tensilelite-rocm/libexec/hipblaslt/tensilelite/tensilelite-client"
```

The config-setting value must be an absolute executable path. It is frozen into
that installation; runtime CLI, YAML, and environment overrides are not
supported. A stable symlink may be supplied and retargeted without reinstalling
the Python package.

The integrated CMake `BUILD` mode still creates a private environment using the
standard staged ROCm layout:

```bash
invoke build-client --gpu-targets gfx942
export ROCM_PATH="$PWD/build_tmp/tensilelite-rocm"
"$PWD/build_tmp/tensilelite-venv/bin/python" -m tensilelite --help
```

On Windows, use `build_tmp/tensilelite-venv/Scripts/python.exe`. A custom build
directory has the same `tensilelite-rocm` and `tensilelite-venv` children.

## Tests

Tox provisions the same staged runtime before importing either Python package:

```bash
tox -e unit -- tensilelite/Tests/unit
tox -e py3 -- tensilelite/Tests -m common
tox -e coverage-unit
tox -e coverage
```

Useful variables:

- `TENSILELITE_TEST_ARCH`: architecture used for unit-test staging (default
  `gfx942`).
- `TENSILE_NUM_PYTEST_WORKERS`: pytest worker count (default `4`).
- `TENSILELITE_CLIENT_ARGS`: extra arguments forwarded to `invoke build-client`
  by full/coverage tox environments.

The optional affected-tests hook is installed with:

```bash
uv sync
invoke build-client --gpu-targets gfx942
uv run invoke precommit-install
```

## CMake integration

`HIPBLASLT_TENSILELITE_PYTHON_MODE` selects the package environment:

- `BUILD` stages `tensilelite-client`, installs TensileLite editably, and
  inherits an already installed rocisa without implicit network access.
- `SYSTEM` validates the installed TensileLite wheel/client and importable
  rocisa at configure time.

Device-generation builds require Python 3.10. Host-only hipBLASLt builds that
do not build the client or Python artifacts retain their existing Python
requirements.

Relevant options:

- `TENSILELITE_ENABLE_HOST`
- `TENSILELITE_ENABLE_CLIENT`
- `TENSILELITE_BUILD_TESTING`
- `HIPBLASLT_TENSILELITE_PYTHON_MODE`
- `GPU_TARGETS`

## Design records

- `docs/Public.md`: original proposal.
- `docs/PackagingDecisions.md`: accepted choices and rationale.
- `docs/PackagingPlan.md`: implementation and acceptance plan.
