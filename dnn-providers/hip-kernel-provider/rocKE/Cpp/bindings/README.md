# ckc_engine — pybind11 binding for the C++ Cpp engine

This directory is the binding layer of the dual-backend path. It builds a
Python extension module, `ckc_engine`, that wraps the prebuilt C++ engine
archive (`libckc_core.a`) and exposes its public C API to Python.

It is **additive and isolated**: it adds only new files under `bindings/`, links
the engine archive read-only, and `#include`s the public headers in
`../include/ckc/`. It does **not** modify the engine `src/`/`include/`, the main
`CMakeLists.txt`, or the Python `ck_dsl` package.

## Module API

The first template family is the **universal GEMM**. The Python `spec` argument
is a dict carrying the `UniversalGemmSpec` fields. Keys may be flat
(`tile_m`, `pipeline`, `dtype_a`, …) or nested (`tile={...}`, `trait={...}`,
`data={...}`); every field defaults to the C engine default, so a minimal dict
`{name, tile_m, tile_n, tile_k, warp_m, warp_n}` works.

```python
import ckc_engine

spec = dict(
    name="test1",
    tile_m=128, tile_n=128, tile_k=32,
    warp_m=2, warp_n=2, warp_k=1,
    warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
    pipeline="compv3", epilogue="default",
    dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32",
    wave_size=64, block_size=256, batched=False,
)

ll  = ckc_engine.gemm_lower_llvm(spec, arch="gfx950")    # -> str  (AMDGPU .ll)
ir  = ckc_engine.gemm_serialize_ir(spec, arch="gfx950")  # -> str  (ck.dsl.ir/v1)
ds  = ckc_engine.gemm_verify(spec, arch="gfx950")        # -> list[str] (diagnostics; [] == well-formed)
ok, why = ckc_engine.gemm_is_valid(spec, arch="gfx950")  # -> (bool, str)
name    = ckc_engine.gemm_kernel_name(spec)              # -> str
```

| function | C engine entry point | returns |
|---|---|---|
| `gemm_lower_llvm(spec, arch)`   | `ckc_gemm_universal_lower_to_llvm` (build + lower) | `.ll` text |
| `gemm_serialize_ir(spec, arch)` | `ckc_build_universal_gemm_new` + `ckc_ir_serialize` | `ck.dsl.ir/v1` text |
| `gemm_verify(spec, arch)`       | `ckc_build_universal_gemm_new` + `ckc_verify` | `list[str]` |
| `gemm_is_valid(spec, arch)`     | `ckc_gemm_universal_is_valid_spec` | `(bool, reason)` |
| `gemm_kernel_name(spec)`        | `ckc_gemm_universal_kernel_name` | `str` |

`arch` defaults to `"gfx950"`.

### Error model

The engine uses a **sticky-error IRBuilder**. On a build/lower failure (e.g. an
invalid spec rejected by the validity gate) the binding raises a Python
`RuntimeError` carrying the `ckc_ir_builder_error()` / status text. Arena and
builder lifetime are managed inside each call (built, used, then
`ckc_ir_builder_free`'d before returning); the `char*` results malloc'd by the
engine are copied into a `std::string` and `free`'d.

## Build

1. Build the engine archive (read-only; never modify the engine):

   ```bash
   # run from the rocKE/ root; the top-level CMakeLists builds the ckc_core archive
   cmake -S . -B /tmp/ckc_pybind/engine -DCMAKE_BUILD_TYPE=Release
   cmake --build /tmp/ckc_pybind/engine --target ckc_core -j"$(nproc)"
   # -> /tmp/ckc_pybind/engine/libckc_core.a
   ```

2. Build the extension (its own CMake; points at the archive + headers + a
   python that has pybind11):

   ```bash
   cmake -S Cpp/bindings -B /tmp/ckc_pybind/build \
     -DCMAKE_BUILD_TYPE=Release \
     -DCKC_ENGINE_ARCHIVE=/tmp/ckc_pybind/engine/libckc_core.a \
     -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
     -DPYTHON_EXECUTABLE="$(which python)"
   cmake --build /tmp/ckc_pybind/build -j"$(nproc)"
   # -> /tmp/ckc_pybind/build/ckc_engine.cpython-3XX-*.so
   ```

   Use a python environment that has `pybind11` (>= 3.0) installed so that
   `find_package(pybind11)` resolves.

   The engine compiles as C++20; the binding sets `CMAKE_CXX_STANDARD 20`. The
   archive's symbols are reached through the `extern "C"` public headers, so it
   links cleanly (`target_link_libraries(... libckc_core.a m)`).

## Consistency proof

`/tmp/ckc_pybind/prove_parity.py` drives all the GEMM configs from
`tests/instances/parity/gemm_emit.py` through `ckc_engine` and compares to the Python
engine (`lower_kernel_to_llvm(build_universal_gemm(spec))` and
`ir_serialize.serialize`). Because the binding drives the **same** C++ engine the
differential harness already validates, valid configs are **byte-identical**
(sha equal) and invalid configs are **rejected identically** by both engines.

```
cfg   LL                  IR
test1 byte-identical       byte-identical
test2 both reject (LDS budget 196608 > 163840 cap on gfx950)
test3 both reject (unsupported bf16 warp_tile (32,32,8) on gfx950)
test4 byte-identical       byte-identical
test5 byte-identical       byte-identical
test6 byte-identical       byte-identical
test7 both reject (unsupported fp16 warp_tile (32,32,32) on gfx950)
```

## How this becomes the `CK_DSL_BACKEND=cpp` backend

This module is the engine side of the dual-backend switch, and `cpp` is now the
**default**: `ck_dsl.core.backend` routes lowering through this engine. For a
Python-authored kernel it serializes the IR and calls the family-agnostic
`ckc_engine.lower_serialized_ir(...)`; for cataloged families the per-family
`<fam>_lower_llvm` / `_serialize_ir` / `_verify` entry points are used. If
`ckc_engine` isn't importable, the chokepoint auto-falls back to the native
Python lowerer (byte-identical) and records why. Select explicitly with
`CK_DSL_BACKEND=python|cpp|both`.

**Environment flags:** this harness uses `CKC_PARITY_BUILD` (the `.so` dir) and
`CKC_PARITY_EMIT` (the prebuilt-emitter dir) — point both at the **same** fresh
build. The backend/flavor flags are `CK_DSL_BACKEND`, `CK_DSL_LLVM_FLAVOR`,
`CK_DSL_CPP_STRICT`. Full list:
[`../../dsl_docs/reference/env_flags.md`](../../dsl_docs/reference/env_flags.md).

**Parity rule:** every optimization in the Python engine must have its C++ twin,
proven byte-identical — see
[`../../dsl_docs/development/engine_parity.md`](../../dsl_docs/development/engine_parity.md).
