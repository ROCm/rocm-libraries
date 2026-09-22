# Run the gfx950 convolution demo on another machine

This branch contains the source, descriptors, request catalog, tests, and demo
tools. Build the libraries and packaged kernels on the destination machine, or
use an installation already built from this branch. The demo compares the same
convolution through hipDNN and direct rocKE, checks correctness, and reports GPU
execution times. See [SUMMARY.md](SUMMARY.md) for the purpose and benefits and
[README.md](README.md) for the full integration contract.

## Get the branch and choose working directories

After the branch has been pushed to the shared remote:

```bash
git clone --branch users/aledudek/gfx950-conv-share \
    https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
export CONV_SOURCE="$PWD"
mkdir -p ../gfx950-conv-work
export CONV_WORK="$(realpath ../gfx950-conv-work)"
export CONV_BUILD="$CONV_WORK/build"
export CONV_INSTALL="$CONV_WORK/install"
export CONV_RESULTS="$CONV_WORK/results"
export CONV_GENERATOR="$CONV_SOURCE/projects/hipdnn/tools/IngestorGenerator"
export CONV_ROCM=/opt/rocm
export ROCKE_LLVM_FLAVOR=llvm22
export ROCKE_COMGR_LIB="$CONV_ROCM/lib/libamd_comgr.so"
export AMD_COMGR_CACHE_DIR="$CONV_WORK/comgr-cache"
```

Use a Linux machine with a **gfx950 GPU** and a working ROCm development
installation. The verified compiler is **ROCm 7.2.3 COMGR with LLVM 22**. Keep the
same COMGR library and LLVM flavor for packaging and direct comparisons; adjust
`CONV_ROCM` if that installation is elsewhere. Put `CONV_WORK` on local storage
when possible. Builds, dependencies, virtual environments, caches, and results
will live there, outside the checkout.

## If an installation already exists

Activate its Python environment and set `CONV_INSTALL` to its installation
prefix, then continue at **Run and read the results** below:

```bash
source /path/to/existing/venv/bin/activate
export CONV_INSTALL=/path/to/existing/install
```

That environment must contain this branch's `hipdnn_frontend` wheel, NumPy, and
ROCm-enabled PyTorch. The installation must include the HIP Kernel Provider,
its packaged gfx950 convolution descriptors/kpack, and
`bin/hipdnn_integration_tests`. An ordinary ROCm installation alone does not
contain this experimental engine. The runner uses rocKE sources from the
checkout and sets the library/plugin paths for its child processes.

## Build an installation

You need Git, Ninja, a C++ build toolchain, **CMake 3.26+**, **Python 3.12+ with
development headers and venv support**, and development packages for zstd,
msgpack-cxx, and Boost. On Ubuntu, the latter packages are `libzstd-dev`,
`libmsgpack-cxx-dev`, and `libboost-dev`. CMake and the Python packaging steps
download dependencies, so configure your company's approved package mirrors
and network access first.

Create a Python environment and install the build/demo dependencies:

```bash
python3.12 -m venv "$CONV_WORK/venv"
source "$CONV_WORK/venv/bin/activate"
python -m pip install 'cmake>=3.26' ninja build wheel numpy \
    'msgpack>=1.0.0' 'zstandard>=0.20.0' \
    -r "$CONV_GENERATOR/requirements.txt"
```

Install a **ROCm-enabled PyTorch** wheel approved for the dev image. Select its
version and index using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/)
or your company's ROCm environment instructions; set `CONV_TORCH_VERSION` and
`CONV_TORCH_INDEX` accordingly. The recorded demo used `2.10.0+rocm7.0`; this is
the tested package version, not a requirement to replace a working approved
environment. A default CPU wheel cannot run this demo.

```bash
python -m pip install "torch==${CONV_TORCH_VERSION:?set the approved ROCm wheel version}" \
    --index-url "${CONV_TORCH_INDEX:?set its approved ROCm wheel index}"
python -c 'import torch; assert torch.version.hip and torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

Build the pinned kpack dependency. Both its C++ installation and Python source
directory are needed:

```bash
mkdir -p "$CONV_WORK/deps"
git clone https://github.com/ROCm/rocm-systems.git "$CONV_WORK/deps/rocm-systems"
git -C "$CONV_WORK/deps/rocm-systems" checkout --detach \
    a022846cf553c2b135410a5168f97705f1b9c6ac
cmake -S "$CONV_WORK/deps/rocm-systems/shared/kpack" \
    -B "$CONV_WORK/deps/kpack-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX="$CONV_WORK/deps/kpack-install"
cmake --build "$CONV_WORK/deps/kpack-build" --parallel 8
cmake --install "$CONV_WORK/deps/kpack-build"
```

Configure and build from the rocm-libraries root. The dependency prelude supplies
the imported zstd/msgpack targets required by the pinned kpack CMake package.
The preset selects a release build; reduce the parallelism if memory is limited.
The recipe disables the optional formatting and static-analysis targets. For
C++ development, install clang-format 18 and clang-tidy 20, then enable
`ENABLE_CLANG_FORMAT` and `ENABLE_CLANG_TIDY` as needed.

```bash
cd "$CONV_SOURCE"
export PATH="$CONV_ROCM/bin:$PATH"
cmake --preset hip-kernel-provider -G Ninja -B "$CONV_BUILD" \
    -DROCM_PATH="$CONV_ROCM" \
    -DCMAKE_INSTALL_PREFIX="$CONV_INSTALL" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    '-DROCM_LIBS_ENABLE_COMPONENTS=hipdnn;hipdnn-python;hipdnn-integration-tests;hip-kernel-provider' \
    -DCMAKE_PROJECT_INCLUDE="$CONV_SOURCE/dnn-providers/hip-kernel-provider/docs/gfx950_conv_fwd/kpack_dependencies.cmake" \
    -DCMAKE_PREFIX_PATH="$CONV_WORK/deps/kpack-install;$CONV_ROCM" \
    -DPython_EXECUTABLE="$CONV_WORK/venv/bin/python" \
    -DPython3_EXECUTABLE="$CONV_WORK/venv/bin/python" \
    -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON \
    -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON \
    -DHIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE=ON \
    -DHIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP=ON \
    -DHIPKERNELPROVIDER_ROCKE_COMGR_LIB="$ROCKE_COMGR_LIB" \
    -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR="$CONV_WORK/deps/rocm-systems/shared/kpack/python" \
    -DHIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT="$CONV_SOURCE/dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors" \
    -DGPU_TARGETS=gfx950 -DAMDGPU_TARGETS=gfx950 \
    -DENABLE_CLANG_FORMAT=OFF \
    -DENABLE_CLANG_TIDY=OFF \
    -DENABLE_ASM_SDPA_ENGINE=OFF
cmake --build "$CONV_BUILD" --parallel 8
cmake --install "$CONV_BUILD"
```

`hipdnn-python` builds the native extension but does not install a Python package.
Pack and install its wheel in the active environment:

```bash
python "$CONV_SOURCE/projects/hipdnn/python/frontend_wheel_package/pack_frontend_wheel.py" \
    --build-dir "$CONV_BUILD" --wheel-dir "$CONV_WORK/wheel"
python -m pip install --force-reinstall "$CONV_WORK/wheel"/hipdnn_frontend-*.whl
export LD_LIBRARY_PATH="$CONV_INSTALL/lib:$CONV_WORK/deps/kpack-install/lib:$CONV_ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'import hipdnn_frontend; print("hipDNN Python import: PASS")'
```

If the machine has several ROCm versions, keep the selected compiler, HIP
headers, and runtime consistent. The original dev image needed
`HIPCC_COMPILE_FLAGS_APPEND="-isystem $CONV_ROCM/include"` to correct a stale HIP
header search path; use this only if that same problem occurs on your image.

## Run and read the results

Run these in the activated environment, with the variables above still set:

```bash
python "$CONV_GENERATOR/tools/run_conv_demo.py" smoke \
    --install-prefix "$CONV_INSTALL" --output-dir "$CONV_RESULTS" \
    --comgr-library "$ROCKE_COMGR_LIB" --llvm-flavor "$ROCKE_LLVM_FLAVOR"

python "$CONV_GENERATOR/tools/run_conv_demo.py" headline \
    --install-prefix "$CONV_INSTALL" --output-dir "$CONV_RESULTS" \
    --comgr-library "$ROCKE_COMGR_LIB" --llvm-flavor "$ROCKE_LLVM_FLAVOR"
```

- **Smoke:** 11 installed integration cases, followed by eight comparisons for
  FP16/BF16, forced `tile_k=64/128`, automatic selection, and reuse in another
  process. Expect 11 passing tests, 8/8 passing comparisons, and two passing
  cache restart checks.
- **Headline:** 51 packaged requests, each compared against direct rocKE and an
  independent reference, using the catalog's `tile_k=64` variant. Expect 51/51
  passing comparisons. This run includes substantially larger tensors.
- **Timings:** microseconds of GPU execution. The ratio is hipDNN time divided
  by direct rocKE time; near `1.0` means similar execution time. Compilation,
  plan creation, and the initial kernel-selection search are excluded from the
  steady-state timing. This ratio does not measure full application latency.

Each invocation creates a fresh `probe-smoke-*` or `probe-headline-*` directory
under `CONV_WORK/results` and prints its exact `summary.json` path. It contains
the per-request JSON, logs, and selection caches. To redisplay a completed run,
pass the printed path to the viewer (replace the example path):

```bash
python "$CONV_GENERATOR/tools/show_conv_results.py" /path/to/batch/summary.json
```

The viewer also accepts the batch directory itself and needs only Python's
standard library. Existing results do not require a GPU to inspect.

## Changing shapes or kernels

The engine supports **plain 2D forward cross-correlation, dense channels-last
storage, groups=1, symmetric padding, FP16/BF16 storage, and FP32 accumulation**.
Each request must have an exact compiled catalog entry; see
[graph_contract.md](graph_contract.md). The tracked request catalog is sufficient
for both demos; no external workload archive is needed.

Start with `tools/benchmark_conv_integration.py --help` under `CONV_GENERATOR` for
one-off probes. Save one request from `configs/gfx950_conv_fwd.requests.json` to
a separate file and use `--request-file` and `--tile-k 64`. Automatic selection
and cache reuse require a request with both tile candidates packaged.

To add a shape, update `configs/gfx950_conv_fwd.yaml` and its request catalog,
then generate a reviewable bundle outside the checkout:

```bash
python "$CONV_GENERATOR/generate.py" \
    --config "$CONV_GENERATOR/configs/gfx950_conv_fwd.yaml" \
    --output-dir "$CONV_WORK/generated-conv"
```

Merge the new descriptor entries into
`dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/rocKE/gfx950_conv_fwd`.
Preserve existing UUIDs and completed native implementations: generation emits
fresh UUIDs and native stubs. Follow the
[generator instructions](../../../../projects/hipdnn/tools/IngestorGenerator/README.md)
and [kernel constraints](mining.md), then reconfigure, rebuild, reinstall, and
rerun the demo plus a correctness probe for the added shape. Reconfiguration is
also required after changing integration-test bundle JSON. Repack the frontend
wheel if you change its bindings.
