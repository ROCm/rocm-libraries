<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite

TensileLite is hipBLASLt's Python generator, logic validator, and tuning
workflow. Released Python wheels are part of a matched ROCm SDK. The intended
final runtime contract has two installation models:

- **Python SDK packages:** the active `rocm-sdk-core.__version__` is the full
  ROCm publication identity. Core tool trampolines from that Python
  environment run the compiler and toolchain commands. The final client model
  uses a `rocm-sdk-libraries` trampoline for its library-owned native payload.
  Until that payload ships, a client request reports that it is unavailable.
- **Conventional prefix:** an explicitly selected prefix from `ROCM_PATH`,
  `/opt/rocm`, or a ROCm tool discovered on `PATH` supplies root-relative
  tools and the native client. Its `.info/version` is authoritative only for
  the base `A.B.C` compatibility line, so it cannot distinguish nightly, RC,
  or CI publications on that line.

The two models do not borrow paths from each other. `rocisa` is a
separately prepared Python dependency; TensileLite requires it to be importable
but does not prescribe its ABI or native-artifact layout.

At present, the Python-SDK path and strict wheel/ROCm version comparison are
implemented but deliberately gated off until TheRock forwards the required
package identity and SDK payload. Production uses conventional-prefix discovery
and bypasses the version comparison during that transition.

## Supported interface

```bash
tensilelite benchmark-cluster --help
tensilelite create-library --help
tensilelite generate-summations --help
tensilelite logic --help
tensilelite logic-to-yaml --help
tensilelite merge-library --help
tensilelite retune-library --help
tensilelite run --help
tensilelite update-library --help

# Equivalent module form
python -m tensilelite --help
```

The default wheel exposes `import tensilelite`; it does not provide the legacy
`Tensile` namespace or `Tensile/bin` launchers. An optional
`tensilelite-tensile-compat` wheel supplies deprecated command aliases only.

## Released installation

For a conventional ROCm prefix, use the wheel index delivered with the target
release and select that same prefix:

```bash
export ROCM_PATH=/opt/rocm
python -m pip install --index-url <rocm-wheel-index> tensilelite
python -c 'import tensilelite, rocisa; print(tensilelite.__version__)'
```

With TheRock's Python SDK, install the matching core, libraries, and device
payload in the same Python environment before installing TensileLite. TensileLite
uses the core package's full Python distribution version and its interpreter's
tool trampolines; it does not require `rocm[devel]` or a synthetic ROCm prefix:

```bash
python -m pip install --index-url <rocm-wheel-index> \
  'rocm[libraries,device-<target>]'
python -m pip install --index-url <rocm-wheel-index> tensilelite
```

When the strict validation gate is enabled, import fails when the wheel and ROCm
release differ or when rocisa cannot be imported. The client is resolved and
validated only when a
benchmark/validation workflow requests its path. A configured client binding
always wins for a conventional-prefix installation. The Python SDK model
currently reports that the client is unavailable because `rocm-sdk-libraries`
does not yet ship it; its final state uses that package's exact
`tensilelite-client` console script from the active Python environment. The
conventional-prefix model uses
`$ROCM_PATH/libexec/hipblaslt/tensilelite/tensilelite-client` (with `.exe` on
Windows). Neither model performs a broad client search on `PATH`.

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

TENSILELITE_ROCM_VERSION="$(<"${ROCM_PATH:-/opt/rocm}/.info/version")" \
  python -m pip install --no-build-isolation --no-deps -e .
python -m tensilelite_configure_client \
  --client "$PWD/build_tmp/tensilelite/client/tensilelite-client"

# Remove only this installation's development binding.
python -m tensilelite_configure_client --reset
```

The client value must be an absolute executable whose exact `--version`
matches the installed distribution. A configured binding is exclusive: a
broken configured path never falls back to the production client. Configuration
does not alter the wheel, and the client selection is frozen after its first
request in a process. Use a fresh process after changing or resetting a binding.

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
It does not resolve or bind `tensilelite-client`. Do not run two configurations
concurrently against one Python environment.

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
