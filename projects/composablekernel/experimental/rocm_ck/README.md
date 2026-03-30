# rocm_ck — Schema-Driven API for CK Tile Kernels

Describe GPU kernels as data (Signature + Algorithm), validate at compile time, dispatch with a universal argument struct.

## Architecture

rocm_ck replaces CK Tile's template-parameter API with a data-driven schema. Kernels are defined by two pure-data structs — no template parameters at the user level:

- **Signature** (what to compute) — a directed compute graph where tensors are nodes, operators are edges, and scalars are named parameters. Expressed with C++ designated initializers.
- **Algorithm** (how to compute) — tile geometry, pipeline strategy, wavefront layout, padding. Independent of data types.

These flow through a compile-time pipeline:

```text
Signature (what) + Algorithm (how)
    → consteval resolve() + make_spec()    compile-time validation
    → NTTP kernel descriptor                 structural type, zero runtime cost
    → extern "C" __global__ fn(Args)         bridge: host C++ ↔ device CK Tile
    → host fills Args, launches kernel       runtime
```

**resolve()** is `consteval` — it propagates dtypes, validates SSA uniqueness, resolves rank and layout defaults. Invalid configurations fail at compile time with actionable error messages, not linker errors or runtime crashes.

**make_spec()** pattern-matches the operator sequence (e.g., `[GemmOp, AddOp, ReluOp]`) and returns a structural NTTP kernel descriptor that triggers the correct CK Tile template instantiation.

**Args** is a universal 1408-byte flat POD struct. It lives in the GPU's kernarg segment (device memory, not registers). Each wave issues `s_load` instructions for only the fields it reads — unused slots cost nothing. Any language or dispatcher that can fill a byte buffer can launch a kernel.

The **bridge pattern** (`extern "C" __global__` wrapper) cleanly separates compilation: device code compiles with CK Tile headers; the host binary only needs HIP runtime and the kpack loader.

### Complete Example

This is a full kernel definition — GEMM with fused bias addition and ReLU activation, accumulating in fp32, storing fp16 output:

```cpp
// gemm_fp16_add_relu.hip
#include "gemm_dev.hpp"

static constexpr rocm_ck::GemmSpec spec = rocm_ck::make_spec(
    rocm_ck::Signature{
        .dtype = rocm_ck::DataType::FP16,
        .ops = {rocm_ck::GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                rocm_ck::AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                rocm_ck::ReluOp{.in = "D", .out = "E"}}},
    rocm_ck::GemmAlgorithm{.block_tile  = {128, 128, 32},
                            .block_waves = {2, 2, 1},
                            .mfma_tile   = {16, 16, 16}});

extern "C" __global__ void gemm_fp16_add_relu(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}
```

Every architectural concept is visible: Signature declares the compute graph, Algorithm specifies the tile strategy, `make_spec` validates and resolves at compile time, and the bridge function takes universal `Args`.

## Compilation Boundary

Examples 03 and 04 use a three-file pattern that enforces clean separation
between metaprogramming, host runtime, and device code:

```text
                  _spec.hpp (metaprogramming)
                 /            \
        _api.hpp               _dev.hpp
       (host only)            (device only)
           |                       |
       main.cpp               *.hip files
```

**`_spec.hpp`** is pure metaprogramming. It contains `consteval` factories
(`make_spec`, `is_valid_mfma`), `constexpr` structural types
(`GemmSpec`, `ElementwiseSpec`), named accessors, and `static_assert` tests.
The compiler evaluates everything at compile time — no runtime code is generated
on either side. Both host (g++) and device (hipcc `--cuda-device-only`) passes
include this header.

**`_api.hpp`** is host-only. It contains runtime code that uses the standard
library freely: arg assembly (`make_args`), launch wrappers, runtime validation
with rich error messages. Guarded with `#ifdef __HIP_DEVICE_COMPILE__` / `#error`
to prevent accidental inclusion from `.hip` files.

**`_dev.hpp`** is device-only. It contains the CK Tile bridge — `__device__`
functions that wire structural NTTP descriptors to the CK Tile template stack.
Guarded with `#ifndef __HIP_DEVICE_COMPILE__` / `#error` to prevent inclusion
from host `.cpp` files.

The `.hip` files are compiled with `--cuda-device-only` (device pass only), so
`__HIP_DEVICE_COMPILE__` is always defined. `main.cpp` is compiled as plain C++
by g++, so `__HIP_DEVICE_COMPILE__` is never defined. The guards enforce
the correct boundaries at the preprocessor level.

## What This Changes

CK Tile is a powerful template metaprogramming library. Using it directly means wiring ~7 internal type layers (`TileGemmShape`, `TileGemmTraits`, `GemmPipelineProblem`, pipeline type, partitioner, epilogue, kernel) with dozens of positional template parameters. See [`gemm_dev.hpp`](examples/04_gemm/gemm_dev.hpp) for what that looks like — it's the device-side bridge that rocm_ck hides behind 6 named fields.

rocm_ck doesn't replace CK Tile — it provides a structured front-end. The same CK Tile kernels run underneath, but the user-facing API is pure data with compile-time validation.

## Composable Operators

Signatures describe computation as a graph of typed operators connected by named tensors. Each operator output gets a unique name (SSA form); graph edges are shared names between outputs and inputs.

**GEMM + bias + ReLU:**
```text
A, B → [GemmOp] → C → [AddOp] ← bias → D → [ReluOp] → E
```

**Available operators:**

| Category | Operators |
|----------|-----------|
| Compute | `GemmOp` |
| Binary | `AddOp`, `MulOp` |
| Unary | `ReluOp`, `FastGeluOp`, `GeluOp`, `SiluOp`, `SigmoidOp`, `SoftmaxOp` |
| Scalar | `ScaleOp` |

Adding a new operator: define a struct with named tensor slots, add it to the `Op` variant, add one line to `visit_op()` in `resolve.hpp`. If the struct satisfies `BinaryOpLike` or `UnaryOpLike` (C++20 concepts), generic rank/layout propagation works automatically.

See [SIGNATURE.md](SIGNATURE.md) for the full specification: dtype cascading, layout resolution, SSA validation rules, and fusion pattern-matching.

## Examples

Progressive examples, each building on the last:

| Example | Architecture concept |
|---------|---------------------|
| [01_hello_world](examples/01_hello_world/) | Multiarch pipeline baseline — hand-written HIP kernel, one binary per arch |
| [02_ck_tile_vector_add](examples/02_ck_tile_vector_add/) | Bridge pattern — `extern "C"` wrapper around CK Tile's `ElementWiseKernel` |
| [03_rocm_ck_vector_add](examples/03_rocm_ck_vector_add/) | Full schema — Signature/Algorithm split, `make_spec` validation, 12 compiled variants, registry-based selection |
| [04_gemm](examples/04_gemm/) | Composed operators — multi-type GEMM (fp32/fp16/bf16), fused epilogue, mixed-precision accumulation |

### Example 01: Hello World

Validates the end-to-end multiarch pipeline with a trivial hand-written vector-add kernel. One kernel source, compiled per architecture, packed into a single archive, loaded at runtime.

### Example 02: CK Tile Bridge

Proves that CK Tile kernels can be compiled into standalone `.hsaco` code objects via the bridge pattern. The device kernel (`ck_tile_add.hip`) wraps `ElementWiseKernel<Add>` behind an `extern "C"` entry point with flat arguments compatible with `hipModuleLaunchKernel`.

### Example 03: Full Tuning Surface

Production-ready pattern with:

- **Signature/Algorithm separation** — *what* (operator graph + data types) vs *how* (tile geometry, wavefront count, vector width, padding)
- **12 compiled variants** across FP32, FP16, BF16 with different block sizes, multi-wave, and mixed-precision
- **Constexpr validation** via `make_spec` that catches invalid configurations at compile time
- **Variant registry** with `find_variant(DataType, problem_size)` for automatic kernel selection
- **Archive metadata** — tuning parameters stored in the kpack TOC for tooling

See the [example 03 README](examples/03_rocm_ck_vector_add/README.md) for full details.

### Example 04: GEMM (Multi-Type)

Extends the schema to GEMM:

- **Operator-centric Signature** — `GemmOp`, `AddOp`, `ReluOp` compose the compute graph
- **Multi-type variants** — fp32, fp16, bf16 compiled from ~20-line `.hip` files via `run<S>` template
- **Mixed-precision epilogue** — all variants accumulate in fp32; CShuffleEpilogue handles output type conversion
- **Composed epilogue** — `GemmOp + AddOp + ReluOp` for fused bias + activation
- **Typed host buffers** — `float_to_typed` / `typed_to_float` for type-agnostic verification

See the [example 04 README](examples/04_gemm/README.md) for full details.

## Design for Integration

The schema is designed to map cleanly into dispatcher and tooling systems:

- **Signature** is a `constexpr` aggregate struct with no pointers or heap allocation. Its fields (dtypes, operators, tensor names) are a superset of a dispatcher's `KernelKey`. Serialization to JSON or Python dataclasses is mechanical.
- **Algorithm** maps to a dispatcher's `KernelInstance` — tile geometry, pipeline strategy, padding. Each field has a concrete numeric value.
- **Args** is trivially-copyable, standard-layout POD with `static_assert`-verified layout. Any language or runtime that can fill a 1408-byte buffer can launch a kernel.

Integration is intentionally deferred until the schema stabilizes. See `dispatcher-integration-strategy.md` for the planned approach.

## Pipeline

```text
*.hip kernel sources
    | (clang++ --cuda-device-only, per arch × per variant)
    v
*.hsaco code objects
    | (pack.py — variant-aware packer)
    v
kernels.kpack  (single archive: multiple architectures × multiple variants)
    | (loaded at runtime by kpack C API)
    v
host executable  (variant selection → hipModuleLoadData → hipModuleLaunchKernel)
```

## Directory Structure

```text
experimental/rocm_ck/
├── CMakeLists.txt                  # Top-level (delegates to include/ and examples)
├── README.md
├── include/                        # Shared headers (header-only)
│   ├── CMakeLists.txt              # INTERFACE library target "rocm_ck"
│   └── rocm_ck/
│       │ # Types — pure definitions, no runtime, no CK deps
│       ├── types.hpp               # index_t, wavefront_size
│       ├── datatype_utils.hpp      # DataType enum, data_type_bits(), data_type_name()
│       ├── layout.hpp              # Layout enum (Row, Col, Contiguous, Auto)
│       ├── tensor_desc.hpp         # TensorDesc: name, dtype, rank, layout
│       ├── physical_tensor.hpp     # TensorName, PhysicalTensor (consteval, NTTP-safe)
│       │
│       │ # ABI — shared host/device interface
│       ├── args.hpp                # Args, TensorArg, ScalarValue (kernarg ABI, 1408 bytes)
│       │
│       │ # Metaprogramming — compile-time logic, no runtime
│       ├── ops.hpp                 # Operator structs (GemmOp, AddOp, ...) + Op variant
│       ├── signature.hpp           # Tensor, Scalar, Signature (compute graph schema)
│       ├── resolve.hpp             # consteval resolve(): dtype cascade, rank/layout propagation
│       │
│       │ # Host-only — requires HIP runtime or C++ runtime features
│       ├── hip_check.hpp           # HIP_CHECK error-checking macro
│       ├── gpu_arch.hpp            # get_gpu_arch() — GPU architecture detection
│       ├── typed_buffer.hpp        # TypedBuffer: RAII device memory with type conversion
│       ├── datatype_convert.hpp    # float ↔ typed conversions, verification tolerances
│       ├── kpack_module.hpp        # KpackArchive, KpackKernel RAII wrappers
│       │
│       │ # Device-only — requires CK Tile headers (--cuda-device-only)
│       └── ck_type_map.hpp         # DataType → CK Tile C++ type mapping
├── rocm_kpack/                     # Vendored kpack C runtime library (from TheRock)
│   ├── CMakeLists.txt
│   ├── include/rocm_kpack/
│   │   ├── kpack.h
│   │   ├── kpack_types.h
│   │   └── kpack_export.h
│   └── src/
│       ├── kpack_internal.h
│       ├── isa_target_match.h
│       ├── kpack.cpp
│       ├── archive.cpp
│       ├── compression.cpp
│       ├── toc_parser.cpp
│       ├── loader.cpp
│       ├── path_resolution.cpp
│       └── isa_target_match.cpp
└── examples/
    ├── 01_hello_world/             # Minimal: hand-written HIP kernel
    │   ├── CMakeLists.txt
    │   ├── vector_add.hip
    │   ├── pack.py
    │   └── main.cpp
    ├── 02_ck_tile_vector_add/      # Bridge pattern: extern "C" + CK Tile
    │   ├── CMakeLists.txt
    │   ├── ck_tile_add.hip
    │   ├── pack.py
    │   └── main.cpp
    ├── 03_rocm_ck_vector_add/      # Full tuning surface: variants + registry
    │   ├── CMakeLists.txt
    │   ├── vector_add_variants.hpp     # Variant table + consteval lookup + findVariant
    │   ├── vector_add_*.hip            # 12 variant instantiations (~12 lines each)
    │   ├── pack.py                     # Variant-aware packer with metadata
    │   └── main.cpp                    # Variant selection demo + verify-all mode
    └── 04_gemm/                     # GEMM: multi-type via operator-centric Signature
        ├── CMakeLists.txt
        ├── gemm_variants.hpp           # Variant table + consteval lookup
        ├── gemm_*.hip                  # 6 variant instantiations (~12 lines each)
        ├── cpu_ref.hpp                 # CPU reference GEMM implementation
        ├── cpu_ref.cpp                 # CPU reference implementation
        ├── pack.py                     # Variant-aware packer with dtype metadata
        └── main.cpp                    # Multi-variant loop with typed buffers
```

## Dependencies

- **ROCm** with HIP support (`/opt/rocm`)
- **System packages**: `libmsgpack-cxx-dev`, `libzstd-dev`
- **Python packages**: `msgpack` (`pip install msgpack`)

## Build

Each example is standalone — build from its directory:

```bash
cd experimental/rocm_ck/examples/01_hello_world  # or 02_, 03_, 04_

cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS="gfx90a;gfx942"

ninja -C build
```

Examples 03 and 04 require C++20 for struct NTTPs and `consteval` validation.

## Run

On a machine with a supported GPU:

```bash
# Example 01
./build/kpack_hello_world build/kernels.kpack

# Example 02
./build/kpack_ck_tile_vector_add build/kernels.kpack

# Example 03
./build/kpack_rocm_ck_vector_add build/kernels.kpack

# Example 04
./build/kpack_gemm build/gemm.kpack
```

If the current GPU's architecture is not in the archive, the demo prints a clear error and exits.

## Appendix: Kpack Archive Format

Kpack (kernel pack) is the binary archive format for multiarch distribution. Instead of shipping fat binaries, kpack packages per-architecture `.hsaco` code objects into compressed archives loaded at runtime based on the detected GPU.

### Archive Pipeline

1. **Compile**: `clang++ --cuda-device-only` compiles `.hip` sources into per-architecture `.hsaco` code objects
2. **Pack**: `pack.py` concatenates the `.hsaco` blobs after a 16-byte KPAK header, then appends a MessagePack table of contents (TOC) recording each blob's offset, size, and architecture
3. **Load**: The host binary opens the archive via the kpack C API, queries the detected GPU's architecture, extracts the matching code object, and loads it via `hipModuleLoadData`

### Binary Format

```text
[0x00]  "KPAK"              4 bytes   Magic
[0x04]  version             4 bytes   Little-endian uint32 (currently 1)
[0x08]  toc_offset          8 bytes   Little-endian uint64
[0x10]  blob_0              variable  Raw .hsaco for first arch
        blob_1              variable  Raw .hsaco for second arch
        ...
[toc_offset]  MessagePack TOC     variable  Compression scheme, arch list, blob metadata, nested TOC
```

The TOC can also carry optional `variant_metadata` sections (used by example 03) containing tuning parameters for each kernel variant.

### Inspect the Archive

```bash
python3 -c "
import msgpack, struct
f = open('build/kernels.kpack', 'rb')
magic, version, toc_offset = struct.unpack('<4sIQ', f.read(16))
print(f'Magic: {magic}, Version: {version}, TOC offset: {toc_offset}')
f.seek(toc_offset)
toc = msgpack.unpack(f, raw=False)
for binary, archs in toc['toc'].items():
    for arch, meta in archs.items():
        print(f'  {binary}/{arch}: {meta[\"original_size\"]} bytes')
"
```

### Vendored Runtime

The `rocm_kpack/` directory contains a local copy of the kpack C runtime library, taken from [TheRock](https://github.com/ROCm/TheRock) (`base/rocm-kpack/runtime/`) for this experimental demo. This avoids requiring a full TheRock build as a dependency — the runtime is small and self-contained. Once kpack ships as part of a ROCm release, this vendored copy should be replaced with a proper `find_package(rocm_kpack)`.

It provides:

- `kpack_open` / `kpack_close` — archive lifecycle
- `kpack_get_architecture_count` / `kpack_get_architecture` — enumerate architectures
- `kpack_get_binary_count` / `kpack_get_binary` — enumerate binaries
- `kpack_get_kernel` / `kpack_free_kernel` — extract a code object by binary name + architecture
- NoOp and Zstd per-kernel decompression
- Higher-level loader/cache API for runtime integration
