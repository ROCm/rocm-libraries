# Build options and environment, author-scoped

The rocKE surface spans the engine build, the ingestor build, the provider configuration
and the packaging-time environment, across five files. This page is the subset an author
needs. rocKE's own `dsl_docs/reference/env_flags.md` owns the full variable reference and
`rocke/BUILDING.md` owns the Python-only loop; neither gives the combined
author → descriptor → pack picture, which is why this page exists.

## CMake options

All default to the value shown. None is set by the `hip-kernel-provider` preset, so every
one of them is yours to pass.

| Option | Default | Why an author cares |
|---|---|---|
| `HIPKERNELPROVIDER_ENABLE_ROCKE` | OFF | Gates `add_subdirectory(rocke)` (`dnn-providers/hip-kernel-provider/CMakeLists.txt:49`, used at `:351`). Without it there is no rocKE engine and no `rocke-pyenv` |
| `HIPDNN_ENABLE_KERNEL_INGESTOR` | OFF | Gates the generic ingestor that consumes `kind: rocke` and `kind: kpack` descriptors (`dnn-providers/hip-kernel-provider/CMakeLists.txt:51-53`). Must be ON to integrate anything descriptor-backed |
| `HIPDNN_ENABLE_SDPA` | OFF | The SDPA **frontend** (`projects/hipdnn/CMakeLists.txt:55`), compiled in as a define at `projects/hipdnn/frontend/CMakeLists.txt:86`. **Required ON for any attention graph**, in the hipDNN SDK at `HIPDNN_ROOT` *and* the provider, or the plan silently DECLINEs with nothing naming the flag |
| `ENABLE_ASM_SDPA_ENGINE` | ON | A *competing* hand-written ASM engine (`dnn-providers/hip-kernel-provider/CMakeLists.txt:54`), unrelated to the frontend despite the similar name. Turn it **OFF** to get signal on a new rocKE attention engine |
| `HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE` | OFF | Lets the rocKE producer run over the production source root (`dnn-providers/hip-kernel-provider/descriptor-packaging/cmake/HkpPackaging.cmake:744-747`). Requires `HIPKERNELPROVIDER_ENABLE_ROCKE` ON and a `ROCKE_WHEEL_DIR`, or configure aborts with FATAL_ERROR (`:798-810`) |
| `ROCKE_BUILD_PYENV` | ON | Builds the build-local editable `rocke-pyenv` venv (`dnn-providers/hip-kernel-provider/rocke/CMakeLists.txt:26-28`). Leave it ON unless you manage the environment yourself |

`HIPDNN_ENABLE_SDPA=OFF` is the mistake that wastes a whole build: the provider compiles,
the engine enumerates, and every attention graph declines with no diagnostic pointing at
the flag. rocKE's own reference calls this out
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/reference/env_flags.md:35`).

`ENABLE_ASM_SDPA_ENGINE=ON` is the mistake that wastes a whole *test run*: the shared
suite exercises the winning engine for a graph, not every engine that could serve it, so a
new attention engine may never execute while the suite passes green on the incumbent. Two
ways out — build with the competitor absent, or assert on engine identity in the test. A
test that checks the output is correct proves *something* computed it; one that also
checks which engine was selected proves yours did.

## A build invocation that has been executed

Start from this rather than composing a new one. It is an executed gfx950 ingestor
build, and it builds off the `hip-kernel-provider` preset —
read the superbuild skill first, because the preset already solves the toolchain file and
the component scoping, and hand-rolling around it costs configures.

```bash
cmake --preset hip-kernel-provider -B build-exp \
  -DGPU_TARGETS=gfx950 \
  -DPython3_EXECUTABLE=<venv>/bin/python \
  -DENABLE_CLANG_TIDY=OFF \
  -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON \
  -DHIPDNN_ENABLE_SDPA=ON \
  -DENABLE_ASM_SDPA_ENGINE=OFF \
  -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON \
  -DHIPKERNELPROVIDER_ENABLE_TESTS=ON \
  -DHIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP=ON \
  -DHIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE=ON \
  -DHIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT=<repo>/dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors \
  -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR=<dir containing rocm_kpack/> \
  -DHIPKERNELPROVIDER_ROCKE_COMGR_LIB=/opt/rocm/lib/libamd_comgr.so.3 \
  -DHIPKERNELPROVIDER_KPACK_ALLOW_FETCH=OFF
```

Three things about it that are not obvious:

- **Configure from the repository root**, not the provider directory.
- **`Python3_EXECUTABLE` must itself import `msgpack` and `zstandard`.** This is distinct
  from `HIPKERNELPROVIDER_KPACK_PYTHON_DIR`, which only locates `rocm_kpack/`. The packager
  *runs* the interpreter, and the failure names `/usr/bin/python3` rather than the variable
  you set.
- **`gfx950` builds fine on a gfx942 or CPU-only host.** `arch` is a codegen target through
  comgr, not a host requirement. Only *running* the kernel needs the device.

## The two environment variables that change results

Everything else in rocKE's env reference changes logging or tooling. These two change what
you get:

**`ROCKE_BACKEND`** — `cpp` | `python` | `both`, default `cpp`
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/reference/env_flags.md:15`).
`cpp` **silently falls back** to the Python lowerer when the `rocke_engine` extension is
not built, which is the normal state in a wheel venv. Set `ROCKE_CPP_STRICT=1` to make that
raise instead
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/reference/env_flags.md:17`).
`both` runs both engines and asserts byte-identity.

The packager does not leave this to chance: it pins the backend to `python` explicitly and
verifies no fallback happened, through rocKE's own fallback ledger rather than by scraping
stderr — because a packaged kernel must not be produced by an accident of which engine
happened to be importable
(`dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/rocke_compile.py:24-36`).
Pin it the same way when you are reproducing a pack by hand.

**`ROCKE_LLVM_FLAVOR`** — `llvm22` | `llvm20` | `llvm23`, auto-resolved from the comgr
library that will actually load
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/reference/env_flags.md:16`).
It **materially affects performance**: MFMA scheduling and register allocation differ
between flavors, and attention prefill bodies that look register-bound or occupancy-
collapsed on one are clean on another. Forcing a flavor the loaded toolchain does not
match surfaces as `ComgrError: do_action(COMPILE_SOURCE_TO_BC): status=1`, an
intrinsic-signature mismatch
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/runtime/comgr_and_hipmodule.md:208-209`).
Leave it on auto unless you are deliberately comparing, and record which flavor resolved
alongside any number you report.

## The Python environment

`rocke/platform` and `rocke/library` are two independent `pyproject.toml` roots. They are
editable-installed **in that order**, and the order matters
(`dnn-providers/hip-kernel-provider/rocke/BUILDING.md:69-70`):

```sh
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --upgrade pip "setuptools>=61" wheel
pip install --config-settings editable_mode=compat -e rocke/platform
pip install --config-settings editable_mode=compat --no-deps -e rocke/library
```

CMake does exactly this for you when `HIPKERNELPROVIDER_ENABLE_ROCKE=ON`: the venv lands
in `build/rocke-pyenv/`, never your system or user site-packages, and it is created with
`--system-site-packages` so the ROCm PyTorch you already have is reused rather than
reinstalled (`dnn-providers/hip-kernel-provider/rocke/BUILDING.md:50-57`). Editable
installs point at live source, so code edits need no reinstall.

The generic names `kernels`, `builders` and `dispatch` come from `rocke/library` and are
confined to that venv. That is why `source` in a descriptor is a dotted path through
`kernels` and why the packer needs the wheel environment rather than a source path.

For the ad-hoc, non-CMake path rocKE documents for itself — `export PYTHONPATH=python` from
`rocke/platform/`, plus a separately installed ROCm PyTorch wheel — see
`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/development/setup_guide.md:38-47`.
`No module named 'rocke'` always means `PYTHONPATH` is not pointing at
`rocke/platform/python`
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/development/setup_guide.md:227-234`).

## Running the packager outside a build

`hkp_pack` has a CLI at
`dnn-providers/hip-kernel-provider/descriptor-packaging/tools/hkp_pack.py`, taking
`--source-root`, `--out-root`, `--arches` and `--hipcc`
(`dnn-providers/hip-kernel-provider/descriptor-packaging/tools/hkp_pack.py:79-104`).
**`--hipcc` is required even for a rocKE-only root**, because producer selection is
per-UKD and the argument is parsed before any descriptor is read; pass a real hipcc path
even when nothing will use it.

Three dependencies bite, in this order:

1. `rocm_kpack` must be importable — `--kpack-python-dir <.../rocm_kpack-src/python>`.
2. That package needs **`msgpack`** and **`zstandard`**. A bare venv has neither, and the
   failure surfaces as `unable to import rocm_kpack` naming the *inner* missing module.
3. The rocKE path needs comgr, and resolves `/opt/rocm/lib/libamd_comgr.so` on its own.

`PYTHONPATH` for a source tree is `<provider>/descriptor-packaging/python`,
`<provider>/rocke/library`, `<provider>/rocke/platform/python`.

Set `HKP_PACK_JOBS` explicitly when packing a small variant set: the default is a
compromise tuned for large packs, and a 200-variant pack is substantially faster on twelve
workers than on the default
(`dnn-providers/hip-kernel-provider/descriptor-packaging/tools/hkp_pack.py:41-77`).
`HKP_PACK_JOBS=1` forces serial execution, which is what you want while a compile is
failing: the traceback comes from the walk itself with nothing else in flight.
