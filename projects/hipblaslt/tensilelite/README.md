# TensileLite

TensileLite is hipBLASLt's Python generator, logic validator, and tuning
workflow. It ships as one pure-Python wheel, `tensilelite`, plus a native
benchmark client, `tensilelite-client`.

## Versions

| Version | Built by | Meaning |
| --- | --- | --- |
| `5.0.0+g<commit>` | `pip install <path>`, tox, `invoke install`, standalone CMake | A development build of that rocm-libraries commit. It makes no ROCm claim. |
| `5.0.0+g<commit>.dirty` | The same, from a checkout with uncommitted changes | As above; the exact changes are recorded in a digest in `_build_info.py`. |
| `5.0.0+rocm<identity>` | TheRock release, release-candidate and nightly builds | Released as part of that ROCm build. |
| `5.0.0+devrocm<identity>` | TheRock CI and development builds | Built in that TheRock development build. |

`VERSION` and `tensilelite.GENERATOR_VERSION` identify the generator format
written to logic and configuration files and must remain equal. The commit is
always recorded in the installed package (`tensilelite/_build_info.py`), also
for `+rocm`/`+devrocm` wheels.

## ROCm installations

TensileLite selects one ROCm installation when it is imported:

| Installation | How it is found | ROCm identity |
| --- | --- | --- |
| TheRock Python SDK (`pip install rocm[...]`) | `rocm_sdk_core` in the same Python | `rocm_sdk_core.__version__` (full) |
| TheRock native install (`/opt/rocm`, packages) | `ROCM_PATH`, `/opt/rocm`, or `hipconfig` on `PATH` | `share/therock/therock_manifest.json` (full); `.info/version` if the manifest is missing |
| Older ROCm install | same | `.info/version` (`X.Y.Z` only) |

Older installs are supported on a best-effort basis without CI: generation
works, and benchmarking needs a client you build against that ROCm and bind
(see [Source development](#source-development)). They ship no
`tensilelite-client`.

A `+rocm`/`+devrocm` wheel must match its ROCm installation, comparing full
identities (or `X.Y.Z` on an older install). A `+g<commit>` wheel makes no ROCm
claim and is not compared. `rocisa` is a separately prepared Python dependency;
TensileLite requires it to be importable but does not prescribe its layout.

The Python SDK is used whenever `rocm_sdk_core` is installed in the same
Python; there is no separate switch. Its compiler and tool commands are the
console scripts in that interpreter's scripts directory.

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

The wheel exposes `import tensilelite` and the `tensilelite` command family
only. The legacy `Tensile` namespace, `Tensile/bin` launchers and `Tensile*`
console scripts are gone, with no compatibility package:

| Legacy command | Use instead |
| --- | --- |
| `Tensile` | `tensilelite run` |
| `TensileCreateLibrary` | `tensilelite create-library` |
| `TensileLogic` | `tensilelite logic` |
| `TensileGetPath` | No command: read packaged files through `tensilelite.resources` |
| Other `Tensile<Name>` scripts | `tensilelite --help` lists the subcommands |


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

Import fails when a `+rocm`/`+devrocm` wheel and the selected ROCm differ, and
warns once for a `+g<commit>` wheel. The client is
resolved and validated only when a benchmark or validation workflow requests
its path. A configured client binding always wins; otherwise the lookup is
limited to the selected installation's executable paths: the graph-prepared
`PATH` for TheRock CI, interpreter script directories for Python SDK packages,
and `<prefix>/bin` plus `<prefix>/lib/llvm/bin` for a conventional prefix. If
no compatible `tensilelite-client` is available there, the requesting workflow
reports an error rather than searching broadly on `PATH`.

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

`invoke install` also installs the standalone stinkytofu binding used by the
gfx1250 backend when a gfx1250 GPU is detected. Pass `--stinkytofu on|off` to
force the decision, or `--rebuild-stinkytofu` to rebuild an already-installed
binding. After changing rocisa or stinkytofu C++ sources, rerun `invoke rocisa`
(or `invoke install`).

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

The package and the client are paired by the rocm-libraries commit they were
built from (`tensilelite-client --source-revision`):

- an installed package and its client must report exactly the same commit,
  including the `.dirty.<digest>` suffix of a build with uncommitted changes;
- an editable install computes its commit from the checkout when the client is
  first requested, so the client must be built from the same commit; local
  uncommitted changes on either side only warn;
- a package or client built without a commit never pairs. Build from a git
  checkout, or pass the commit with `TENSILELITE_SOURCE_REVISION` (pip) or
  `-DTENSILELITE_SOURCE_REVISION` (CMake).

The client also records the ROCm it was compiled against
(`tensilelite-client --build-rocm`). A different major ROCm version than the
selected installation is an error; other differences warn once.

The client value must be an absolute executable. A configured binding is
exclusive: a broken configured path never falls back to the production client. Configuration
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
invoke configure-client --build-dir build_tmp
uv run invoke precommit-install
```

Rebuild the client after committing; it must be built from the same commit as
the editable package.


## CMake integration

Device generation builds the `tensilelite` wheel, unpacks it into the build
tree (`<build>/tensilelite-wheel`) and runs it with the in-tree raw rocisa build
through a command-scoped `PYTHONPATH`. Nothing is installed into the selected
Python. It does not resolve or bind `tensilelite-client`.

The configure records the rocm-libraries commit for the wheel and the client,
and re-runs when `HEAD` moves. Building from a tree without git records no
commit and warns that client pairing is disabled; pass
`-DTENSILELITE_SOURCE_REVISION=<commit>` to record one.

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

<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->
