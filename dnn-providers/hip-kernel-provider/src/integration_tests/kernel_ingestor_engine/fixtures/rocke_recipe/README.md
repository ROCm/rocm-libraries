# Native SDPA recipe integration example

This opt-in spike connects a real hipDNN frontend graph to an offline-authored
rocKE recipe. It depends on the JIT infrastructure from PR #10456 and demonstrates
downstream integration work; it is not a merge requirement for that PR.

Python records and rolls the recipe during the build. The installed descriptors
reference CBOR, which the native provider specializes to LLVM and compiles through
COMGR when preparing a plan. Repeated execution uses the prepared HIP module and
the caller's current buffers and stream. The application does not run Python or
supply a recipe key. Recipe production is offline; IR generation and GPU code
compilation happen at runtime.

## Supported fixture

| Property | Value |
| --- | --- |
| Engine | `hipkernel:RockeSdpaExample` |
| Device | gfx950 |
| Operation | One causal forward self-attention node, inference |
| Types | BF16 Q/K/V/O, FLOAT graph compute |
| Dimensions | B=1, Hq=Hkv=4, D=128, Sq=Skv=S |
| Sequence lengths | 512, 768, 1024 only |
| Layout | Logical BHSD with BSHD storage; strides `[S*512,128,512,1]` |
| Scale | Graph constant `1/sqrt(128)` |
| Tiling | block_m=256, block_n=64, nonpersistent |
| Workspace | Zero |

Statistics, dropout, bias, ragged/paged inputs, and other optional SDPA features
are outside this fixture. Q/K/V/O must have 16-byte alignment and nonoverlapping
storage. The graph matcher checks supported semantics and layout; candidate
admission checks the recipe's explicit guard before compilation.

The producer records S=512 and 1024 and verifies held-out S=768. A preferred
candidate has a guard admitting only S=512; the second candidate admits all three
sizes. This exercises selection after an ordinary guard refusal. The guard is
authored explicitly, not inferred from sampled traces or generated with SymPy.

## Source contract

The SDK parses this proposed descriptor representation:

```json
{
  "kind": "rocke_recipe",
  "bundle": "sdpa_dense.cbor",
  "recipe_key": "sdpa_dense_bf16_d128_causal"
}
```

`bundle` resolves relative to its declaring descriptor and must remain inside
the descriptor tree. `recipe_key` selects an entry; graph facts supply its `S`
parameter. Each dispatch handler advertises source kinds through
`supportsSourceKind`. The SDK acquires no rocKE or COMGR dependency.

The provider keeps one immutable byte snapshot per canonical bundle path, shared
by admission and preparation. Replacing loaded artifacts requires a new process.
This is not a persistent compilation cache or a digest-based update mechanism.
Missing entries and guard refusals exclude candidates. Missing/malformed bundles
produce admission diagnostics and make this engine unavailable through the
frontend. This example does not establish cross-provider fallback after a
compilation failure.

## Build and run

Use a checkout containing this change and its JIT dependency, a C++20 compiler,
CMake 3.25.2 or newer, uv, and a ROCm installation with COMGR. The validated
toolchain was ROCm 7.14 with the LLVM23 lowerer. On installations lacking ROCm
release metadata, set `ROCKE_LLVM_FLAVOR=llvm23` explicitly for that toolchain;
the provider does not change process-global compiler configuration in prepare.

From the repository root, create the build and Python environment outside the
checkout. Both projects must be built against the SDK headers from this change.
The existing provider packager can fetch its pinned kpack authoring dependency.

```bash
export SDPA_SOURCE="$PWD"
export SDPA_WORK="$(mktemp -d /tmp/rocke-sdpa.XXXXXX)"
export ROCM_PATH=/opt/rocm
uv venv --python 3.12 "$SDPA_WORK/venv"
uv pip install --python "$SDPA_WORK/venv/bin/python" \
  numpy==2.3.3 zstandard==0.25.0 msgpack==1.1.2 ninja==1.13.0
export PATH="$SDPA_WORK/venv/bin:$ROCM_PATH/bin:$PATH"

cmake -S "$SDPA_SOURCE/projects/hipdnn" -B "$SDPA_WORK/hipdnn" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SDPA_WORK/install" \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DROCM_PATH="$ROCM_PATH" \
  -DPython3_EXECUTABLE="$SDPA_WORK/venv/bin/python" \
  -DHIPDNN_ENABLE_SDPA=ON -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON \
  -DHIPDNN_SKIP_TESTS=ON -DHIPDNN_GENERATE_SDK_HEADERS=OFF \
  -DENABLE_CLANG_FORMAT=OFF -DENABLE_CLANG_TIDY=OFF
cmake --build "$SDPA_WORK/hipdnn" --parallel 8
cmake --install "$SDPA_WORK/hipdnn"

cmake -S "$SDPA_SOURCE/dnn-providers/hip-kernel-provider" \
  -B "$SDPA_WORK/provider" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SDPA_WORK/install" \
  -DCMAKE_PREFIX_PATH="$SDPA_WORK/install;$ROCM_PATH" -DROCM_PATH="$ROCM_PATH" \
  -DPython3_EXECUTABLE="$SDPA_WORK/venv/bin/python" \
  -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON \
  -DHIPKERNELPROVIDER_ENABLE_ROCKE_RECIPE_EXAMPLE=ON \
  -DHIPKERNELPROVIDER_KPACK_ALLOW_FETCH=ON -DENABLE_ASM_SDPA_ENGINE=OFF \
  -DHIPKERNELPROVIDER_ENABLE_TESTS=OFF -DBUILD_TESTING=OFF \
  -DROCKE_INSTALL_PYTHON_PACKAGE=OFF -DROCKE_INSTALL_TESTS=OFF \
  -DROCKE_INSTALL_PYTHON_TESTS=OFF \
  -DENABLE_CLANG_FORMAT=OFF -DENABLE_CLANG_TIDY=OFF
cmake --build "$SDPA_WORK/provider" --target test_frontend_sdpa --parallel 8
cmake --install "$SDPA_WORK/provider"

export LD_LIBRARY_PATH="$SDPA_WORK/install/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export ROCKE_LLVM_FLAVOR=llvm23 HIPDNN_LOG_LEVEL=info
env -u PYTHONPATH -u PYTHONHOME -u HIPDNN_DESCRIPTOR_DIR \
  -u HIPDNN_DESCRIPTOR_RUNTIME_DIR \
  "$SDPA_WORK/install/bin/test_frontend_sdpa" \
  "$SDPA_WORK/install/lib/hipdnn_plugins/engines"
```

Run the final command on gfx950. This manual opt-in test is not registered in
CTest and does not silently skip on an unsupported device. It reports three
selected-engine checks, three `rocKE recipe compiled` events, six numerical
results, and `frontend_result: PASS`. The compiled key ends in `_short` for
S=512 and uses the broader entry for S=768/1024.

The test uses an independent C++ causal-SDPA reference with double accumulation
and the same BF16 inputs. It does not use `CpuFpReferenceSdpa`. It checks new
buffers, nonconsecutive UIDs, a non-default stream, output sentinels, missing
UIDs, misaligned/overlapping buffers, unsupported sequence lengths, dtype,
layout, head size, and mask. On Linux it also checks that `libpython` is absent
from the process mappings.

For a native checkpoint independent of hipDNN, run `produce_sdpa.py OUTPUT` with
the same venv and `ROCKE_BACKEND=python ROCKE_LLVM_FLAVOR=llvm23`. Configure this
directory's standalone CMake project with
`-DROCKE_PLATFORM="$SDPA_SOURCE/dnn-providers/hip-kernel-provider/rocke/platform"`.
Run `test_native_sdpa OUTPUT --compile-only` for byte-identical LLVM comparison
and COMGR, or `--gpu` to add GPU numerical execution. The producer's reference
LLVM and manifest are test evidence; only CBOR and JSON descriptors are installed
as this example's kernel inputs.

## Validation and follow-up

The native checkpoint passed three LLVM comparisons and COMGR compilations.
Native and frontend GPU checks each passed six numerical executions. A relocated
frontend payload containing native binaries, JSON and CBOR also passed, without
Python, reference LLVM, or precompiled HSACO in that payload. Fresh processes
rejected missing CBOR, malformed CBOR and missing entries before compilation.
The two focused SDK test suites passed 134 tests with AddressSanitizer enabled.
A fresh feature-disabled provider build passed without a new COMGR dependency.

This is a bounded integration spike, not production SDPA coverage. Follow-up work
includes general graph-to-recipe bindings, symbolic guard authoring, more SDPA
features/targets, per-call compiler configuration, artifact compatibility and
failure-policy tests, persistent/coalesced compilation caching, multi-device and
capture lifetime rules, and automated test registration. Incompatible/newer-reader
artifact rejection has not been tested through this frontend. Performance and
general structural rolling are outside the demonstrated result.
