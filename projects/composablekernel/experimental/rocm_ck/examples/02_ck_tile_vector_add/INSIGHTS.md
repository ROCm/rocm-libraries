# Insights: The Bridge Pattern

This example proves that CK Tile device code can be compiled standalone and loaded at runtime via kpack. The bridge pattern is the architectural foundation for all rocm_ck kernels.

## 1. The Bridge Pattern Is the Key Architectural Insight

An `extern "C" __global__` wrapper takes flat arguments (pointers, ints), constructs CK Tile types on-device, and delegates to the template kernel:

```cpp
extern "C" __global__ void ck_tile_vector_add(
    ck_tile::index_t n,
    const float* a,
    const float* b,
    float* c)
{
    auto lens    = ck_tile::make_tuple(n);
    auto strides = ck_tile::make_tuple(ck_tile::index_t{1});
    auto inputs  = ck_tile::make_tuple(a, b);
    Kernel{}(lens, strides, strides, inputs, c);
}
```

The wrapper is the clean separation boundary. Above it: the kpack/HIP runtime world with C linkage and flat types. Below it: CK Tile's template-heavy, C++-mangled kernel implementations. Production kernels must all follow this pattern.

## 2. CK Tile Headers Compile with --cuda-device-only

This was the gating risk for the entire kpack approach. Proven: CK Tile's template-heavy headers produce valid `.hsaco` code objects when compiled standalone with `--cuda-device-only`. No host stubs needed, no runtime initialization, no linking required.

The CMake command:
```
${CMAKE_HIP_COMPILER} --offload-arch=${arch} --cuda-device-only -x hip
    -I${CK_INCLUDE_DIR} -o ${hsaco_output} ck_tile_add.hip
```

Output is a clang offload bundle containing the raw GPU code object. `hipModuleLoadData` handles it directly — no unbundling step required.

> **Production implication**: Every CK Tile kernel can be compiled to a standalone `.hsaco` per architecture. The entire kpack model — compile separately, pack into archives, load at runtime — is viable because this compilation path works.

## 3. Host Code Is CK-Free

The host binary includes only `<hip/hip_runtime.h>`, `<rocm_kpack/kpack.h>`, and standard C++. Zero CK Tile headers in the host compilation. The host links against `hip::host` (not `hip`) and `rocm_kpack`.

This means rocm_ck consumers never see CK Tile internals. No 200+ template headers, no multi-minute compilations, no CK version coupling in consumer builds. The kpack archive is the only artifact that crosses the boundary.

## 4. CK Tile Types Are Constructed On-Device

`make_tuple(a, b)` and `make_tuple(n)` happen inside the kernel. The ABI boundary uses only fundamental types — pointers and ints. This is essential because CK Tile types (tuples of sequences, tile distributions) are not ABI-stable. Their layout depends on template instantiation details that can change between CK versions.

By constructing CK types on-device from flat arguments, the bridge pattern makes the ABI boundary version-independent. The host doesn't need to know how `ck_tile::tuple` is laid out — it just passes the raw values.

## 5. The extern "C" Symbol Is Required for hipModuleGetFunction

CK Tile kernels have mangled C++ symbols (e.g., `_Z21ck_tile_vector_addPKfS0_Pfi`). `hipModuleGetFunction` needs a predictable, stable entry point name. The `extern "C"` wrapper provides one:

```cpp
hipModuleGetFunction(&kernel_function, module, "ck_tile_vector_add");
```

The string `"ck_tile_vector_add"` is a stable contract. Without `extern "C"`, the symbol name would depend on the function signature, compiler version, and mangling scheme — all of which are fragile across builds and platforms.

> **Production implication**: Every compiled kernel variant needs a unique, stable `extern "C"` symbol. The naming convention (operation + types + tile config) must be defined once and enforced by the build system.

## 6. Block Size Is a Compile-Time Contract

`BlockTile = sequence<256>` is baked into the device code at template instantiation time. The host must launch with matching grid dimensions:

```cpp
const int BLOCK_SIZE = 256;  // Must match BlockTile
const int GRID_SIZE  = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
```

There is no runtime check for this — a mismatch produces wrong results silently. In Example 03, this coupling is managed by embedding the block size in the kernel descriptor, so the host reads `kernel.workgroup_size` instead of hardcoding a constant. That's the production model: the descriptor is the single source of truth for launch parameters.
