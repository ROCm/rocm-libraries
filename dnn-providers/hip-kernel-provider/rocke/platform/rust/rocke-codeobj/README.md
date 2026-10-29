# rocKE direct code-object compiler

`rocke-codeobj` builds `rocke_codeobj_ffi`, a Rust library that turns
serialized rocKE IR directly into a loadable HSACO. It has no LLVM dependency:
no LLVM, Clang, LLD, COMGR, assembler, external tool, compiler shared library,
precompiled-image template, or code-object cache participates. The crate
depends only on the Rust standard library.

Build and test out of tree:

```text
cargo test --locked --target-dir <build-dir>/rocke-codeobj
cargo build --locked --release --target-dir <build-dir>/rocke-codeobj
```

or through CMake with `-DROCKE_BUILD_RUST_CODEOBJ=ON`, which also registers the
`rocke_native_gemm_code_object`, `rocke_native_direct_code_object` and
`rocke_native_direct_code_object_session` CTests. C++ targets link the library
with `rocke_link_codeobj_ffi(<target>)`.

## General rocKE IR compiler

`rocke_codeobj_compile_rocke_ir_v2` compiles serialized rocKE IR for gfx1151 or
gfx950 (`rocke_codeobj_compile_rocke_ir_v1` for gfx1151), and the session entry
points compile repeatedly on the same worker threads or from a streamed input.
In Python, `compile_kernel(..., code_object_backend="direct")` or
`ROCKE_CODEOBJ_BACKEND=direct` selects it. Its usage, stages, limits, debug
controls and change gates are described in
[`dsl_docs/runtime/direct_compiler.md`](../../dsl_docs/runtime/direct_compiler.md).

## Direct gfx1151 GEMM compiler

`rocke_codeobj_compile_gfx1151_wmma_gemm_v1` accepts the serialized
`ck.dsl.ir/v1` produced by the rocKE C++ or Python builder, validates the
complete qualified f16 RCR 16x16x16 WMMA graph, encodes gfx1151 instructions,
and writes the ELF64/AMDHSA records in Rust. The result points directly at the
writer allocation until the caller disposes it with
`rocke_codeobj_compile_result_dispose_v1`. The C declarations are in
`platform/cpp/include/rocke/code_object_ffi.h`.

The Python helper is `compile_gfx1151_wmma_gemm_direct()`. It loads the library
from `ROCKE_CODEOBJ_DIRECT_LIB`, or from the platform library search path when
that variable is unset.

This entry point is a single-pattern compiler: M/N/K remain
runtime arguments and the builder may select `xm` or `xn` grid order and a
valid symbol base. Every other architecture, dtype, layout, tile, or IR graph
is rejected.

`src/gfx1151_wmma_gemm.hex` is compiled only into Rust tests. It is the
qualified oracle result used to check the generated instruction stream and
loader-visible sections; release code does not decode, copy, or otherwise read
it. Changing the encoder requires oracle comparison, ELF inspection, gfx1151
module-load and numeric gates, and a direct-backend version change.
