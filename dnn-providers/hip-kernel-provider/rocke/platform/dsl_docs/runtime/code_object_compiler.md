# Code-object compiler contract

rocKE has two separate backend choices for kernels that lower through LLVM:

1. the Python or C++ engine lowers `KernelDef` to byte-identical AMDGPU LLVM IR;
2. COMGR turns that LLVM IR into a loadable HSACO.

`ROCKE_BACKEND` selects the first choice. `ROCKE_CODEOBJ_BACKEND` names the
second; `comgr` is its only value. Keeping those controls separate is
required: the code-object compiler must never change the bytes emitted by
either lowering engine.

The direct gfx1151 WMMA GEMM path below is a different boundary. It consumes
serialized rocKE IR, not LLVM IR, and no LLVM component participates.

## COMGR results

`rocke.runtime.code_object` returns:

- the unmodified HSACO bytes;
- wall-clock timings for IR-to-bitcode, bitcode linking, code generation, and
  executable linking;
- compiler identity and provenance through `loaded_compiler_info()` and
  `compiler_identity()`.

Several modules are linked as LLVM bitcode and code-generated once. Linking
separate relocatables is not equivalent: it can leave several AMDGPU metadata
notes in one ELF image, while the HIP loader consumes one module metadata note.

Cache keys include the complete compiler identity as well as IR, target, and
options, so artifacts from different COMGR builds never alias.

## Direct gfx1151 WMMA GEMM

`compile_gfx1151_wmma_gemm_direct()` consumes the serialized rocKE graph for
the gfx1151 wave32 f16 RCR 16x16x16 WMMA GEMM. M, N, and K remain runtime
arguments and must be positive multiples of 16. Grid order and the symbol are
graph inputs; the ABI, register allocation, and instruction schedule are
qualified properties of this backend version.

The Rust `cdylib` (`rocke_codeobj_ffi`) returns its generated HSACO allocation
through the versioned C ABI in `cpp/include/rocke/code_object_ffi.h`. It does
not load COMGR or LLVM, invoke Clang/LLD/an assembler, launch a subprocess,
copy a precompiled image, or consult a code-object cache, and the crate has no
dependencies outside the Rust standard library. Unsupported architectures,
layouts, dtypes, tile shapes, or changed IR graphs fail rather than falling
back.

Python loads the library from `ROCKE_CODEOBJ_DIRECT_LIB`, or from the platform
library search path when that variable is unset.

The checked-in HSACO hex is test-only oracle data. Release code contains a
gfx1151 instruction encoder and an ELF/AMDHSA writer, not that image. Extending
the backend means adding reviewed instruction selection and encoding for
another explicit IR pattern and enrolling it in ELF, loader, numeric, and
performance gates; this slice is not a general AMDGPU compiler.

### Native C++ to Rust GEMM chain

The `rocke_native_gemm_code_object` integration target exercises the complete
Python-free gfx1151 GEMM path:

1. the rocKE C++ `WmmaGemmSpec` builder constructs the kernel;
2. C++ serializes the `KernelDef` as `ck.dsl.ir/v1`;
3. Rust strictly matches the complete qualified GEMM graph;
4. the Rust gfx1151 encoder selects the fixed-register wave32 schedule and
   writes instruction words, including label-derived branch displacements;
5. the Rust ELF writer builds the descriptor, MessagePack metadata, dynamic
   symbols and hashes, program headers, and section headers;
6. Rust validates the resulting ELF, metadata note, and generated kernel
   symbol.

Build and register the host integration test with:

```text
-DROCKE_BUILD_RUST_CODEOBJ=ON
```

The executable accepts `[HSACO_OUT] [ITERATIONS]`. It clears compiler-related
environment variables, compiles both `xm` and `xn` C++ graphs, and requires
distinct generated symbols and artifacts. Optional iteration output is
diagnostic only; no performance threshold is enforced by the correctness test.

### Native C++ to Rust chain for general rocKE IR

`rocke_codeobj_compile_rocke_ir_v1` (also declared in `code_object_ffi.h`)
compiles any serialized kernel the direct compiler supports; it takes the same
request layout as the GEMM entry point and rejects unsupported IR with a
diagnostic. The `rocke_native_direct_code_object` target drives it from the C++
instance builders with no Python: gfx1151 WMMA forward attention, implicit-GEMM
forward convolution, and the WMMA GEMM (through the GEMM entry point). For each
case it checks the image and its determinism, and it reports the median time of
the C++ build, the C++ serialization, and the Rust compile with the compiler's
own phase split:

```text
rocke_native_direct_code_object [--reps N] [--warmup N] [--only SUBSTR]
    [--expect-reject SUBSTR,...] [--dump DIR] [--json FILE] [--cpp-parse]
```

`--expect-reject` names cases the direct compiler is known to reject. Such a
case fails the run if it starts compiling, so the expectation is dropped when
the gap closes. The CTest registration lists convolution there: the C++ tile
loader emits `tile.buffer_load_vN_f16`, where the Python loader emits
`tile.buffer_load_vN`, and the direct compiler parses only the latter. Both
spellings lower to identical LLVM IR, which is why the `.ll` byte-identity gate
does not see the difference; `run_diff.py --mode ir --canonical` does.

## Qualification commands

```text
cargo test --locked --manifest-path platform/rust/rocke-codeobj/Cargo.toml
python -m pytest platform/tests/runtime/test_direct_gemm.py \
  platform/tests/runtime/test_code_object.py \
  platform/tests/runtime/test_code_object_elf.py
ctest -R "rocke_native_(gemm|direct)_code_object"
```

The GPU numeric case in `test_direct_gemm.py` runs only on a gfx1151 device.
Store measurements in CI artifacts, not in the repository.
