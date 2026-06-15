# libmxfp6gemm — MXFP6 GEMM for AMD CDNA4 (gfx950 / MI350X)

A standalone HIP library that computes

```
D[M,N] = A[M,K] · B[K,N]
```

where **A and B are MXFP6** (FP6 E2M3 values + one shared E8M0 scale per 32-element K-block) and
**D is FP32, FP16, or BF16**. One kernel paradigm, routed by shape; the public header is
device-free so a host translation unit can drive the library without a HIP compiler.

---

## Performance vs CK MXFP6

FP16 output, K=8192, on MI350X (gfx950), ROCm 7.0.2.1. Same-precision MXFP6 performance vs CK's
`mx_flatmm` (`-mx_prec=fp6xfp6`), measured on the same machine in the same session.

| M × N (K=8192) | CK FP6 (TFLOPs) | libmxfp6gemm (TFLOPs) | speedup |
|---|---:|---:|---:|
| 2048 × 4096 | 1050 | 1706 | **1.63×** |
| 2048 × 8192 | 1059 | 2176 | **2.06×** |
| 4096 × 4096 | 1059 | 2329 | **2.20×** |
| 4096 × 8192 | 1112 | 2352 | **2.12×** |
| 8192 × 8192 | 1115 | 2418 | **2.17×** |

Absolute numbers vary run-to-run; the **ratio** is the stable takeaway.

---

## Build

```bash
cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=gfx950
cmake --build build -j
ctest --test-dir build          # end-to-end correctness gate (vs CPU reference)
```

This produces the static library `libmxfp6gemm.a` and the test `test_gemm`.

To use it in your project: include `<mxfp6/gemm.hpp>`, link `mxfp6gemm`. The header `gemm.hpp`
is device-free (plain declarations); the host preprocessing helpers are in `mxfp6/preprocess.hpp`.

---

## The shape of the data

`gemm()` does **not** quantize and **does not** take your raw float matrices. It takes
**pre-quantized, pre-laid-out DEVICE buffers**. The host-side preprocessing (in
`mxfp6/preprocess.hpp`) turns float matrices into those buffers. The flow is:

```
float A[M][K]  ─ quantize_to_mxfp6 ──────────────► packed A  + A scales ─ pad_scales_k ─ preprocess_scale ─ tile_scale ─► tiled A scales
float B[K][N]  ─ preprocess_B (transpose+quant) ─► packed Bᵀ + B scales ──────────────── preprocess_scale ─ tile_scale ─► tiled B scales
                                                             └─ preshuffle_B ─► preshuffled B
```

You then upload the four buffers (packed A, preshuffled B, tiled A scales, tiled B scales) to the
device and call `gemm()`.

The quick start below uses the **default, recommended layout**: B (weights) is K-padded once,
and A (activations) stays in its **natural compact layout** — no per-row padding, just a tiny
end pad via `a_compact_end_pad`. See [K-padding](#k-padding--why-a-stays-compact) for why.

---

## Quick start

```cpp
#include <mxfp6/gemm.hpp>
#include <mxfp6/preprocess.hpp>
using namespace mxfp6;

// Inputs (host): A is row-major [M][K] (activations), B is row-major [K][N] (weights).
// M and N must be multiples of 256; K a multiple of 32 (see Requirements).
int M = 8192, N = 8192, K = 8192;

TileChoice tc = choose_tile(M, N);   // picks the tile + scale grouping for this shape
int Kp = kpad(K);                    // pad K up to a multiple of K_TILE (=192); here 8256

// --- 1. quantize (host) ---
// A: quantized in its NATURAL COMPACT K layout — no per-row padding, A_f32 is just M*K floats.
QuantizedMatrix Aq = quantize_to_mxfp6(A_f32, M, K);
// B (weights): padded to Kp with a ZERO K-tail. B_f32 is Kp*N floats with rows [K..Kp) = 0.
QuantizedMatrix Bq = preprocess_B(B_f32, Kp, N);        // transposes B[K][N]→Bᵀ[N][K], then quantizes

// --- 2. lay out for the kernel (host) ---
PreshuffledB Bsh = preshuffle_B(Bq);
// A scales: pad_scales_k extends them to Kp with a non-NaN tail (REQUIRED for compact A — see below).
TiledScale   sA  = tile_scale(preprocess_scale(pad_scales_k(Aq.scales.data(), M, K).data(), M, Kp),
                              tc.MPW, K_TILE / 64);
TiledScale   sB  = tile_scale(preprocess_scale(Bq.scales.data(), N, Kp), tc.NPW, K_TILE / 64);
//                                                                       ^^^^^^  ^^^^^^^^^^^^
//                                                          A uses tc.MPW, B uses tc.NPW; subs = K_TILE/64 = 3

// --- 3. upload to device ---
// A buffer = compact data + a small end pad for the last row's K-tail read (content irrelevant).
size_t Abytes = (size_t)M * Aq.packed_row_bytes + a_compact_end_pad(K);
void *dA, *dBsh; uint8_t *dsA, *dsB; void *dD;
hipMalloc(&dA,   Abytes);                     hipMemcpy(dA, Aq.packed_data.data(),
                                                        (size_t)M * Aq.packed_row_bytes, ...);
hipMalloc(&dBsh, Bsh.data.size());           hipMemcpy(dBsh, Bsh.data.data(), Bsh.data.size(), ...);
hipMalloc(&dsA,  sA.data.size());            hipMemcpy(dsA,  sA.data.data(),  sA.data.size(),  ...);
hipMalloc(&dsB,  sB.data.size());            hipMemcpy(dsB,  sB.data.data(),  sB.data.size(),  ...);
hipMalloc(&dD,   (size_t)M * N * sizeof(OutElem));   // OutElem = float / __half / __hip_bfloat16

// --- 4. launch ---
gemm(OutType::F16, M, N, Kp,
     dA, dBsh, dsA, dsB, dD,
     Aq.packed_row_bytes,        // A_row_bytes = packed(K)  (compact stride)
     Bq.packed_row_bytes);       // B_row_bytes = packed(Kp)
hipDeviceSynchronize();
```

`dD` now holds the `M×N` row-major result in the requested output type. Note A is never padded
per-row — only `a_compact_end_pad(K)` bytes are added at the very end of the whole buffer.

---

## API reference

### `mxfp6/gemm.hpp` (public, device-free)

| Symbol | Purpose |
|---|---|
| `enum class OutType { F32, F16, BF16 }` | Output element type. |
| `constexpr int K_TILE` (=192) | The kernel's K-tile depth. K is padded to a multiple of it. |
| `int kpad(int K)` | K rounded up to a multiple of `K_TILE` → pass as the kernel's `Kp`. |
| `struct TileChoice { int MT, NT, MPW, NPW; }` | Tile (MT×NT) + per-wave 32-block counts (MPW/NPW) for scale grouping. |
| `TileChoice choose_tile(int M, int N)` | Pick the tile/scale grouping for a shape. **Use its `MPW`/`NPW` for `tile_scale`.** |
| `void gemm(OutType, M, N, Kp, dA, dBsh, dsA, dsB, dD, A_row_bytes, B_row_bytes)` | Launch. All `d*` are device pointers. |

### `mxfp6/preprocess.hpp` (host)

| Symbol | Purpose |
|---|---|
| `quantize_to_mxfp6(const float* mat, rows, cols)` | float → `QuantizedMatrix` (packed FP6 + per-block E8M0 scales). |
| `preprocess_B(const float* B, K, N)` | Transpose `B[K][N]`→`Bᵀ[N][K]`, then quantize. |
| `preshuffle_B(const QuantizedMatrix& Bq)` | Re-pack Bᵀ for coalesced VMEM loads → `PreshuffledB`. |
| `preprocess_scale(const uint8_t* scales, dim, K)` | Re-order per-block scales to the MFMA lane layout. |
| `tile_scale(const PreprocessedScale&, group, subs)` | Group a wave's scales contiguous. `group = MPW` (A) / `NPW` (B); `subs = K_TILE/64`. |
| `a_compact_end_pad(int K)` | (compact-A recipe) bytes to over-allocate at the A-buffer end. |
| `pad_scales_k(const uint8_t* scales, dim, K)` | (compact-A recipe) extend scales to `kpad(K)` with a non-NaN tail. |
| `dequantize_mxfp6(const QuantizedMatrix&, float* out)` | MXFP6 → float (for debugging / reference). |

The data types and FP6/E8M0 conversions live in `mxfp6/types.hpp`; a CPU reference GEMM
(`mxfp6_gemm_ref`) lives in `tests/reference.hpp`.

---

## K-padding — why A stays compact

The kernel reads K in deep tiles, so K is padded to `kpad(K)` and the kernel computes over the
full `Kp`. The padded K-tail is harmless **as long as it is zero on B**, because `B[k]·anything = 0`.
The quick start above exploits this so you never pad your activations:

* **B (weights):** padded to `Kp` with a zero K-tail — done once, offline. `preprocess_B` over a
  zero-`Kp`-padded float already produces this.
* **A (activations):** stays in its natural compact layout (`A_row_bytes = packed(K)`). The kernel
  still reads `packed(Kp)` per row, so each row's K-tail read spills into the **next row's** real
  data — which is fine, because B's zero K-tail multiplies it away. Only two things are needed:
  1. **`a_compact_end_pad(K)`** extra bytes at the very end of the A buffer, so the *last* row's
     K-tail read stays in bounds (its content is irrelevant), and
  2. **`pad_scales_k`** to extend A's scales to `Kp` with a non-NaN tail — see the footgun below.

This is measured perf-neutral vs per-row padding and is gated by `verify_compact` in `test_gemm`.

> **Footgun — the one thing you must not skip.** The fp6 *data* tail is inert (`B·0 = 0`), but an
> E8M0 scale byte of `0xFF` decodes to **NaN**, and `0·NaN = NaN` poisons the *entire* output.
> Always build A's scale tail with `pad_scales_k` (or quantize A over a zero-`Kp`-padded float).

**Simpler alternative (if you don't mind padding activations):** quantize A over `Kp` directly
(`quantize_to_mxfp6(A_f32, M, Kp)` with the K-tail zeroed), pass `A_row_bytes = Aq.packed_row_bytes`
(now `packed(Kp)`), and skip both `a_compact_end_pad` and `pad_scales_k` (the zero-padded quant
gives a safe scale tail automatically). Same result, but A is padded per-row.

---

## Requirements & caveats

- **K must be a multiple of 32** (the MX block size). It is then padded internally to `kpad(K)`
  (a multiple of `K_TILE`=192) — pass `kpad(K)` as the kernel's `Kp`. ⚠️ A K that is not a
  multiple of 32 is silently mis-quantized in a release build (the `assert` is compiled out).
- **M and N must be multiples of 256** (the tile). The small-M path relaxes M to a multiple of
  128, but 256 is always safe. Pad M/N up otherwise — the grid is launched with integer division,
  so a non-divisible M/N silently drops the remainder rows/cols.
- **B is transposed by `preprocess_B`.** Pass B in its natural `[K][N]` row-major layout; the
  helper produces `Bᵀ[N][K]` and quantizes it.
- **Scale grouping must match the tile.** Use `choose_tile(M,N).MPW` for A's `tile_scale` and
  `.NPW` for B's. Mismatched grouping gives wrong results.
- **`A_row_bytes` / `B_row_bytes`** are the packed bytes per row (= `Quantized.packed_row_bytes`).
  For compact A this is `packed(K)`; for per-row-padded A it is `packed(Kp)`.
- **E8M0 scale `0xFF` is NaN** and will poison the **entire** output (`0·NaN = NaN`). This is the
  one real footgun of the pad-B-only/compact-A recipe — always extend A's scale tail with
  `pad_scales_k` (or quantize A over a zero-`Kp`-padded float). The fp6 *data* tail is inert.
- **Feed finite activations.** A `0xFF` scale only ever comes from uninitialized memory, never from
  `quantize_to_mxfp6` (it clamps to ≤254 and `std::max` ignores NaN). So a NaN/Inf in A's *real*
  data does **not** NaN-poison the output — but it is silently swallowed and **corrupts the
  affected 32-block's row** (wrong values, verified). The fp6 data encoding has no NaN/Inf, so A's
  data can never *introduce* NaN; only a `0xFF` scale can. Sanitize activations upstream.
- **The pad-B-only recipe requires B's K-tail to be exactly zero.** It does not work if you pad
  only A.
- **Inputs are device buffers.** `gemm()` only launches the kernel; you allocate, quantize on the
  host, and `hipMemcpy` the four buffers yourself. The output `dD` must be sized for the chosen
  `OutType`.
- **Targets gfx950 (CDNA4 / MI350X).** The kernel uses `v_mfma_scale_f32_32x32x64_f8f6f4`.

---

## Testing

`ctest` (i.e. `test_gemm`) is the correctness gate: fresh-allocated, `0x5A`-poisoned outputs
compared against a CPU reference, on both tile paths (256×256 and 128×256), including
non-square / partial-grid / `k_tiles==1`, and the compact-A recipe (`verify_compact`). It also
prints an indicative FP16 performance sweep (let it reach steady state before trusting the numbers).
