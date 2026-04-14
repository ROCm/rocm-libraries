# Utility Components

All utilities are in namespace `ck_tile::direct_conv` and located in
`projects/composablekernel/include/ck_tile/ops/direct_convolution/utils/`.

---

## conv_params.hpp

### Enums

- **`enum class DataType { fp16, bf16, fp32, fp8, bf8 }`** -- supported numeric formats.
- **`enum class Direction { Fprop = 1, Dgrad = 2, Wgrad = 4 }`** -- convolution pass direction.
- **`enum TensorOrder { NCHW, NHWC }`** -- tensor memory layout.

### `to_string(Direction)`

Returns the string name of a direction value (`"Fprop"`, `"Dgrad"`, `"Wgrad"`).

### `struct Conv2dParams`

Holds all parameters for a 2D convolution:

| Field       | Description                                    |
|-------------|------------------------------------------------|
| `direction` | `Direction` enum value                         |
| `n`         | batch size                                     |
| `h`, `w`    | input height and width                         |
| `c`         | input channels                                 |
| `k`         | output channels (filters)                      |
| `kh`, `kw`  | filter height and width                        |
| `pad`       | padding (height, width)                        |
| `stride`    | stride (height, width)                         |
| `dilation`  | dilation (height, width)                       |
| `p`, `q`    | output height and width                        |
| `groups`    | number of groups                               |
| `data types`| input, weight, and output data types           |
| `order`     | tensor order (NCHW or NHWC)                    |

**Methods:**

- `compute_output_size()` -- computes output dimensions `p` and `q` from input, filter, padding, stride, and dilation.
- `is_valid()` -- returns true if the parameter combination is valid.
- `channels_per_group()` -- returns `c / groups`.
- `filters_per_group()` -- returns `k / groups`.

### `template <Direction D> struct SizeView`

Provides a direction-aware view of convolution dimensions.

- **Fprop specialization** -- passes through all dimensions unchanged.
- **Dgrad specialization** -- swaps `h/w` with `p/q`, swaps `stride` with `dilation`, and inverts padding as `(kh - 1) * dilation_h - pad_h`.

---

## types.hpp

### Type Aliases

| Alias       | Underlying Type            |
|-------------|----------------------------|
| `fp16_t`    | 16-bit float               |
| `fp16x4_t`  | vector of 4 fp16 values    |
| `fp32x4_t`  | vector of 4 fp32 values    |
| `bf16_t`    | bfloat16                   |
| `fp32_t`    | 32-bit float               |
| `fp8_t`     | 8-bit float (E4M3)         |
| `bf8_t`     | 8-bit float (E5M2)         |

### `template <DataType> struct ToTypeImpl` / `ToType<DataType>`

Maps a `DataType` enum value to the corresponding C++ type at compile time.

### `mantissa_bits(DataType)`

Returns the number of mantissa bits for a given data type.

---

## mathutil.hpp

### `constexpr int maximum(int a, int b)`

Returns the maximum of two integers. Marked `constexpr` and usable on both host and device.

### `constexpr int divup(int x, int y)`

Ceiling division: returns `(x + y - 1) / y`.

---

## launch_params.hpp

### `struct LaunchParams`

```cpp
struct LaunchParams {
    dim3   grid;
    dim3   block_size;
    size_t dynamic_shared_bytes = 0;
};
```

Holds HIP kernel launch configuration.

---

## kernel_variant.hpp

### `struct KernelVariant`

Dispatch interface for convolution kernels. Contains function pointers:

| Function Pointer       | Description                                          |
|------------------------|------------------------------------------------------|
| `is_applicable`        | checks if the kernel supports the given parameters   |
| `config_is_compatible` | checks if a config index is valid for the parameters |
| `get_launch_params`    | returns `LaunchParams` for a given config            |
| `launch`               | launches the kernel on a HIP stream                  |
| `get_workspace_size`   | returns required workspace bytes                     |
| `num_configs`          | number of available configurations                   |

---

## matrix_layout.hpp

### `template <int M, int K, int B, typename T> struct MatrixLayout`

Maps MFMA register indices to matrix element coordinates. Template parameters:

| Parameter | Meaning                           |
|-----------|-----------------------------------|
| `M`       | outer-product dimension           |
| `K`       | inner-product dimension           |
| `B`       | batch dimension                   |
| `T`       | element type (determines packing) |

**Methods:**

- `items_per_register()` -- number of elements packed per register, based on `sizeof(T)`.
- `outer(lane)` -- returns `lane % M`, the outer-product coordinate.
- `inner(lane, idx)` -- inner-product coordinate accounting for register packing.
- `batch(lane)` -- returns `(lane / M) % B`, the batch index.

---

## swizzle.hpp

### `template <int C_> struct SwizzleT`

LDS swizzle pattern for NHWC data with 16-bit elements. Eliminates LDS bank conflicts during the Global -> LDS -> MFMA data movement pipeline.

**Methods:**

- `offset_uint2(x, c4)` -- computes the `uint2`-granularity offset in LDS.
- `offset_uint4(x, c8)` -- computes the `uint4`-granularity offset in LDS.
- `x(offset_uint4)` -- inverse mapping: extracts the spatial `x` coordinate from a `uint4` offset.
- `c8(offset_uint4)` -- inverse mapping: extracts the channel coordinate (in units of 8) from a `uint4` offset.

---

## transpose_lds_layout.hpp

### `template <int M, int K, int B> struct TransposeLDSLayout`

Lane-to-address mapping for the `DS_READ_B64_TR_B16` instruction on CDNA4 (gfx950). Used for Dgrad weight loading with transposed read.

**Methods:**

- `row(lane, read_idx)` -- source matrix row for a given lane and read index.
- `col(lane)` -- starting column (reads 4 consecutive elements).
- `batch(lane)` -- batch index.

---

## memory.hpp

*(Newly ported)*

### `constexpr vmcnt(int count)`

Computes the `vmcnt` flags for `__builtin_amdgcn_s_waitcnt`.

### `template <int Count> wait_vmcnt()`

Waits until all but the last `Count` outstanding vector memory loads have completed.

### `wait_vmcnt_all()`

Convenience wrapper for `wait_vmcnt<0>()` -- waits for all outstanding loads.

---

## detail.hpp

*(Newly ported)*

### `template <int N> static_for(F f)`

Compile-time unrolled loop that calls `f<0>()`, `f<1>()`, ..., `f<N-1>()`.

### `template <int N> dispatch(int idx, F f)`

Runtime-to-compile-time dispatch: calls `f<I>()` where `I` is the compile-time constant matching the runtime value `idx` (for `idx` in `[0, N)`).
