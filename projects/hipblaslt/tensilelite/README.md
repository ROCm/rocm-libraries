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
are unbound and resolve only
`$ROCM_PATH/libexec/hipblaslt/tensilelite/tensilelite-client` (with `.exe` on
Windows); `PATH` is never searched.

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

The editable installation records the built client's absolute path in the
current user's keyed `~/.tensilelite/bindings/` registry. Python source edits
are immediately visible; rerun `invoke build-client` after client source or
CMake changes.

Each step remains available independently. A manual source install may bind any
existing client executable:

```bash
python -m pip install -r requirements-dev-common.txt
invoke build-client --gpu-targets gfx942

python -m pip install --no-build-isolation --no-deps -e .
python -m tensilelite_configure_client \
  --client "$PWD/build_tmp/tensilelite/client/tensilelite-client"

# Remove only this installation's development binding.
python -m tensilelite_configure_client --reset
```

The client value must be an absolute executable whose exact `--version`
matches the installed distribution. A configured binding is exclusive: a
broken configured path never falls back to the production client. Configuration
does not alter the wheel, and the client selection is frozen for each importing
process. Use a fresh process after changing or resetting a binding.

## Tests

Tox builds the client, configures the active editable installation, and uses the
selected real ROCm SDK before importing either Python package:

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

Device generation builds the canonical controlled-artifact wheel, installs it
into the single CMake-selected Python with `--force-reinstall --no-deps`, and
uses only the in-tree raw rocisa package through a command-scoped `PYTHONPATH`.
Every generator command refreshes the keyed binding to its exact built client.
Do not run two configurations concurrently against one Python environment.

Device-generation builds require Python 3.10 and Python development headers;
stable-ABI rocisa builds require Python 3.12. A true host-only build does not
require TensileLite Python. Standalone Windows builds must set `ROCM_PATH` to
the SDK used for the build.

Relevant options:

- `TENSILELITE_ENABLE_HOST`
- `TENSILELITE_ENABLE_CLIENT`
- `TENSILELITE_BUILD_TESTING`
- `ROCISA_BUILD_PYTHON` (rocisa-only root configuration)
- `GPU_TARGETS`

## Design records

- `docs/Public.md`: original proposal.
- `docs/PackagingDecisions.md`: accepted choices and rationale.
- `docs/PackagingPlan.md`: implementation and acceptance plan.
- `PythonBuildGrillingDecisions.md`: current canonical Python-build decisions.
