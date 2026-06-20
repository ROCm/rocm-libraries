# ckc_engine — pybind11 binding for the C++ ck_dsl_c engine

This directory is the binding layer of the dual-backend path. It builds a
Python extension module, `ckc_engine`, that wraps the prebuilt C++ engine
archive (`libckc_core.a`) and exposes its public C API to Python.

It is **additive and isolated**: it adds only new files under `bindings/`, links
the engine archive read-only, and `#include`s the public headers in
`../include/ckc/`. It does **not** modify the engine `src/`/`include/`, the main
`ck_dsl_c/CMakeLists.txt`, or the Python `ck_dsl` package.

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
   cmake -S ck_dsl_c -B /tmp/ckc_pybind/engine -DCMAKE_BUILD_TYPE=Release
   cmake --build /tmp/ckc_pybind/engine -j"$(nproc)"
   # -> /tmp/ckc_pybind/engine/libckc_core.a
   ```

2. Build the extension (its own CMake; points at the archive + headers + a
   python that has pybind11):

   ```bash
   cmake -S ck_dsl_c/bindings -B /tmp/ckc_pybind/build \
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

`/tmp/ckc_pybind/prove_parity.py` drives all 7 GEMM configs from
`tests/parity/gemm_emit.py` through `ckc_engine` and compares to the Python
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

This module is the engine side of the dual-backend switch: with the C++ engine
reachable from Python and proven byte-identical, a `CK_DSL_BACKEND=cpp` selector
can route GEMM lowering/serialization through `ckc_engine` instead of the pure
Python lowerer.

**Deeper integration (not yet done here):** wiring
`CK_DSL_BACKEND` into the `ck_dsl` `IRBuilder`/instances so the **same Python
authoring API** transparently routes to this C++ engine — edits `ck_dsl/core`.
It is intentionally **not** done in this task to avoid touching the shared
Python core concurrently with the hardening work. That step would: (1) read the
`CK_DSL_BACKEND` env var, (2) when `cpp`, translate the existing
`UniversalGemmSpec` into the spec dict and call `ckc_engine.*` from
`build_universal_gemm` / `lower_kernel_to_llvm`, and (3) extend the binding to
the remaining op families (the same `build_spec` + 3-entry-point pattern
generalizes to each `ckc_build_*` / `ckc_*_lower_to_llvm`).
