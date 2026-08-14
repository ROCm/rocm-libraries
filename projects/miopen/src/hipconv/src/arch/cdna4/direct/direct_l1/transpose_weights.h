#pragma once

#include "channels_last_layouts.h"
#include "weights_layout.h"
#include "transpose_weights_layout.h"
#include "transpose_lds_layout.h"
#include "memory.h"
#include "types.h"
#include "hipconv/conv2d_params.hpp"
#include <hip/hip_runtime.h>

#include <cstdint>

namespace hipconv::cdna4::direct_transpose_weights
{
constexpr int VEC_LEN    = 2;
constexpr int BLOCK_SIZE = 256;

constexpr int C_divisor = 64;

// LDS row stride (fp16), chosen bank-conflict-free for the b128 store read.
//
// The store reads each row with ds_read_b128, whose lanes fall into 4 irregular
// phases of 16. Modeling an access by its 16-byte quad q =
// ((L%16)*(stride/8) + L/16) % 16, stride 144 (stride/8 == 2 mod 16) is verified
// conflict-free across all four phases, where stride 136 still collided. 288-byte
// rows stay 16-B aligned (b128 requirement).
constexpr int LDS_K_STRIDE = 144;

// Transpose one K(Kwg) x C(64) source tile into a wavegroup's DirectL1 sub-tensor.
//
// One workgroup handles all k16 sub-tiles of one (kq, c64, kh, kw). The full-Kwg
// K span (vs one k16 tile per workgroup) keeps 4x the loads in flight and cuts the
// workgroup count 4x, hiding the HBM latency the tiny old workgroups exposed.
//
//   Load: 4 waves x 16 dword buffer_load_lds, one K-row each. 32 active lanes cover
//         the 64-fp16 row; lanes 32..63 are OOB and write zeros to the LDS pad.
//         K-OOB rows skip (LDS pre-zeroed); num_bytes also guards C-OOB lanes.
//   Store: wave w stores its k16 = w sub-tile. Lane L writes one b128 per c32 strip
//         straight to dst slot L (the operand: lane L holds K-row L%16, c8-group
//         L/16 = 8 contiguous fp16), a per-lane copy with no swizzle.
//
// blockIdx.z (0 .. WavesK-1) selects the K partition; all sub-tensors share shape,
// so it is a pure offset on src (+wg*Kwg) and dst (+wg*sub-tensor).
template <int Kh, int Kw, int Kwg, hipconv::DataType DT>
__device__ void transpose_weights_subtensor_impl(const ToType<DT>* __restrict__ src,
                                                 ToType<DT>* __restrict__ dst,
                                                 int K,
                                                 int C,
                                                 int C_padded,
                                                 int Kq,
                                                 int Kwg_total)
{
    using T = ToType<DT>;
    static_assert(sizeof(T) == 2, "transpose_weights only supports 2-byte element types");
    static_assert(Kwg % 16 == 0 && Kwg <= 64, "Kwg must be a multiple of 16, <= 64");
    using datatypex8_t = std::conditional_t<DT == hipconv::DataType::bf16, bf16x8_t, fp16x8_t>;

    constexpr int k16_count = Kwg / 16;

    // K partition (blockIdx.z): a source K offset (wg*Kwg) and a dest sub-tensor.
    //
    // The dest base is the layout's wave_group axis (stride = one sub-tensor).
    using DstSubLayout               = WeightsLayout<Kh, Kw, Kwg, 1>;
    const int wg                     = blockIdx.z;
    const int src_k_offset_in_stripe = wg * Kwg;
    ToType<DT>* sub_dst =
        dst + static_cast<size_t>(DstSubLayout(Kq * Kwg, C_padded).wave_group(wg).offset);

    // Group (blockIdx.y): a base-pointer shift on both src and dst (each contiguous).
    const int g = blockIdx.y;
    src += static_cast<size_t>(g) * K * Kh * Kw * C;
    sub_dst += static_cast<size_t>(g) * (static_cast<size_t>(Kq) * Kwg_total) * C_padded * Kh * Kw;

    // Decode (c64, kh, kw, kq) from blockIdx.x, c64 fastest (k16 is a wave axis now).
    const int c64_count = C_padded / 64;
    int b               = blockIdx.x;
    const int wg_c64    = b % c64_count;
    b /= c64_count;
    const int wg_kh = b % Kh;
    b /= Kh;
    const int wg_kw = b % Kw;
    b /= Kw;
    const int wg_kq = b;

    const int tid  = threadIdx.x;
    const int wave = tid / 64;
    const int lane = tid & 63;

    // K(Kwg=64) rows x LDS_K_STRIDE fp16.
    __shared__ T lds[64 * LDS_K_STRIDE];
    auto* lds_u4               = reinterpret_cast<uint4*>(lds);
    constexpr int lds_u4_count = (64 * LDS_K_STRIDE) / 8;
    for(int i = tid; i < lds_u4_count; i += BLOCK_SIZE)
        lds_u4[i] = uint4{0, 0, 0, 0};
    __syncthreads();

    const int c_base = wg_c64 * 64;

    // Load phase: 16 instructions per wave, K(1) each, covering the Kwg K-rows.
    constexpr int data_format = 1 << 15;
    const int valid_c         = C - c_base;
    const int valid_bytes     = valid_c > 0 ? valid_c * (int)sizeof(T) : 0;
    // Cap at 128 (32 lanes x 4 B) so lanes 32..63 are OOB and zero the LDS pad.
    const int c_vec_num_bytes = valid_bytes < 128 ? valid_bytes : 128;

    // Source canonical layout positioned at (kh, kw, c_base); k advances per row.
    auto src_layout         = ChannelsLastWeightsLayout<Kh, Kw>(K, C).kh(wg_kh).kw(wg_kw).c(c_base);
    const int k_stripe_base = wg_kq * Kwg_total + src_k_offset_in_stripe;

    // Strength-reduce the per-row source offset to a running scalar add.
    //
    // Consecutive K-rows differ by a constant stride (Kh*Kw*C). readfirstlane pins
    // the base and stride in SGPRs so the add stays scalar; without it the compiler
    // re-emits the full multiply for all 16 unrolled rows.
    const int row_stride = __builtin_amdgcn_readfirstlane(src_layout.strides().k());
    int src_k_off = __builtin_amdgcn_readfirstlane(src_layout.k(k_stripe_base + wave * 16).offset);

    for(int k_iter = 0; k_iter < 16; ++k_iter, src_k_off += row_stride)
    {
        const int k_local = wave * 16 + k_iter;
        if(k_stripe_base + k_local >= K)
            continue;

        const T* src_k_ptr = src + src_k_off;

        auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<T*>(src_k_ptr), /*stride=*/0, c_vec_num_bytes, data_format);

        T* lds_row = &lds[k_local * LDS_K_STRIDE];
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            rsrc, lds_row, /*bytes_per_lane=*/4, lane * 4, 0, 0, 0);
    }

    wait_vmcnt<0>();
    __syncthreads();

    // Store phase: wave w handles k16 = w.
    //
    // Lane L holds K-row = L%16 and c8-group = L/16; its 8 operand fp16 are the 8
    // contiguous channels lds[(k16*16 + L%16)][c32*32 + (L/16)*8 ..].
    if(wave >= k16_count)
        return;
    const int k16 = wave;

    auto base = WeightsLayout<Kh, Kw, Kwg, 8>(Kq * Kwg, C_padded)
                    .kq(wg_kq)
                    .c64(wg_c64)
                    .kw(wg_kw)
                    .kh(wg_kh)
                    .k16(k16);
    auto* dst_v8 = reinterpret_cast<datatypex8_t*>(sub_dst);

    const int k_row = k16 * 16 + (lane % 16);
    const int c8grp = lane / 16; // 0..3

#pragma unroll
    for(int c32_idx = 0; c32_idx < 2; ++c32_idx)
    {
        const int c = c32_idx * 32 + c8grp * 8;
        dst_v8[base.c32(c32_idx).offset + lane] =
            *reinterpret_cast<const datatypex8_t*>(&lds[k_row * LDS_K_STRIDE + c]);
    }
}

template <int Kh, int Kw, int Kwg, hipconv::DataType DT>
__global__ void transpose_weights_kernel(const ToType<DT>* __restrict__ src,
                                         ToType<DT>* __restrict__ dst,
                                         int K,
                                         int C,
                                         int C_padded,
                                         int Kq,
                                         int Kwg_total)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_raw_ptr_buffer_load_lds))
    {
        transpose_weights_subtensor_impl<Kh, Kw, Kwg, DT>(src, dst, K, C, C_padded, Kq, Kwg_total);
    }
}

// Dgrad weights pre-format: fprop format with operand axes swapped, filter flipped.
//
// Lets the dgrad compute kernel reuse the fprop weights loader and mma loop
// unchanged. Two transforms are baked in here:
//
//   1. Reduction swap C<->K. Dgrad reduces over the forward output channels (dout)
//      and produces din, so the dest operand is C(din) x K(dout), the transpose of
//      fprop's. The canonical source is din-contiguous (the opposite axis), so we
//      load it into LDS then DS_READ_B64_TR_B16 reads it back with din as the
//      operand M and dout as the reduction (as grouped_32c's dgrad does). The
//      8-per-lane tr-read fragment IS the operand, stored straight to dest slot lane*8.
//   2. Spatial 180-degree tap flip. Dgrad needs tap (Kh-1-kh, Kw-1-kw); baked in at
//      the source read so the compute loop's tap pairing is unchanged.
//
// The dest is WeightsLayout with its "K" (output) bound to din and "C" (reduction)
// to dout. The wavegroup split is over din; dout is shared by both wavegroups.

// LDS row pitch for the dgrad [dout][din] tile.
//
// Absorbs a full 64-lane buffer_load_lds (256 bytes) so one row's OOB lanes land
// in pad, not the next row; the +8-fp16 over 128 also staggers rows across banks
// for the tr-read (as in LDS_K_STRIDE).
constexpr int DGRAD_LDS_DOUT_STRIDE = 136;

template <int Kh, int Kw, int Kwg, hipconv::DataType DT>
__device__ void transpose_weights_dgrad_subtensor_impl(const ToType<DT>* __restrict__ src,
                                                       ToType<DT>* __restrict__ dst,
                                                       int K_dout,
                                                       int C_din,
                                                       int C_padded_dout,
                                                       int Kq,
                                                       int Kwg_total)
{
    using T = ToType<DT>;
    static_assert(sizeof(T) == 2, "transpose_weights only supports 2-byte element types");
    static_assert(Kwg % 16 == 0 && Kwg <= 64, "Kwg must be a multiple of 16, <= 64");
    using datatypex8_t = std::conditional_t<DT == hipconv::DataType::bf16, bf16x8_t, fp16x8_t>;
    using int16x4_t    = __attribute__((ext_vector_type(4))) short;
    using int16x8_t    = __attribute__((ext_vector_type(8))) short;

    constexpr int k16_count = Kwg / 16;

    // K partition (blockIdx.z): a source din offset (wg*Kwg) and a dest sub-tensor.
    //
    // The dest base is the layout's wave_group axis (stride = one sub-tensor).
    using DstSubLayout                 = WeightsLayout<Kh, Kw, Kwg, 1>;
    const int wg                       = blockIdx.z;
    const int src_din_offset_in_stripe = wg * Kwg;
    ToType<DT>* sub_dst =
        dst + static_cast<size_t>(DstSubLayout(Kq * Kwg, C_padded_dout).wave_group(wg).offset);

    // Group (blockIdx.y): a base-pointer shift on both src and dst (each contiguous).
    const int g = blockIdx.y;
    src += static_cast<size_t>(g) * K_dout * Kh * Kw * C_din;
    sub_dst +=
        static_cast<size_t>(g) * (static_cast<size_t>(Kq) * Kwg_total) * C_padded_dout * Kh * Kw;

    // Decode (c64, kh, kw, kq) from blockIdx.x, c64 fastest (k16 is a wave axis now).
    //
    // One workgroup covers a dout(64) x din(Kwg) tile. The full-Kwg din span makes
    // each dout row load one coalesced 128-byte line (din is contiguous, the k16
    // sub-tiles adjacent in it) and fans the 8 inner tiles across all 4 waves.
    const int c64_count = C_padded_dout / 64;
    int b               = blockIdx.x;
    const int wg_c64    = b % c64_count;
    b /= c64_count;
    const int wg_kh = b % Kh;
    b /= Kh;
    const int wg_kw = b % Kw;
    b /= Kw;
    const int wg_kq = b;

    const int tid  = threadIdx.x;
    const int wave = tid / 64;
    const int lane = tid & 63;

    // 64 dout rows x DGRAD_LDS_DOUT_STRIDE fp16.
    __shared__ T lds[64 * DGRAD_LDS_DOUT_STRIDE];
    auto* lds_u4               = reinterpret_cast<uint4*>(lds);
    constexpr int lds_u4_count = (64 * DGRAD_LDS_DOUT_STRIDE) / 8;
    for(int i = tid; i < lds_u4_count; i += BLOCK_SIZE)
        lds_u4[i] = uint4{0, 0, 0, 0};
    __syncthreads();

    // din base (dest "K"/output axis), the full Kwg width in this wavegroup's stripe.
    const int din_base = wg_kq * Kwg_total + src_din_offset_in_stripe;
    // dout base (dest "C"/reduction axis), shared by both wavegroups.
    const int dout_base = wg_c64 * 64;

    // Flipped filter taps.
    const int src_kh = Kh - 1 - wg_kh;
    const int src_kw = Kw - 1 - wg_kw;

    // Source canonical layout: w[K=dout][Kh][Kw][C=din], din innermost.
    auto src_layout = ChannelsLastWeightsLayout<Kh, Kw>(K_dout, C_din);

    // Load phase: one buffer_load_lds per dout row, 32 lanes over the din(Kwg) run.
    //
    // Cap num_bytes so din-OOB and idle lanes zero the row pad, not the next row.
    constexpr int data_format = 1 << 15;
    const int valid_din       = C_din - din_base;
    const int valid_bytes     = valid_din > 0 ? valid_din * (int)sizeof(T) : 0;
    // Kwg din * 2 B is the real row; cap there so the higher lanes are OOB.
    constexpr int row_bytes = Kwg * (int)sizeof(T);
    const int din_num_bytes = valid_bytes < row_bytes ? valid_bytes : row_bytes;

    // Strength-reduce the per-row source offset to a running scalar add.
    //
    // Consecutive dout rows differ by a constant stride (Kh*Kw*C_din); the taps and
    // din base are fixed, so only the dout term advances. readfirstlane keeps it scalar.
    const int row_stride = __builtin_amdgcn_readfirstlane(src_layout.strides().k());
    int src_off          = __builtin_amdgcn_readfirstlane(
        src_layout.k(dout_base + wave * 16).kh(src_kh).kw(src_kw).c(din_base).offset);

    // Each of four waves loads 16 of the 64 dout rows.
    for(int d_iter = 0; d_iter < 16; ++d_iter, src_off += row_stride)
    {
        const int dout_local = wave * 16 + d_iter;
        if(dout_base + dout_local >= K_dout)
            continue; // dout-OOB row: LDS stays zero (padded reduction).

        const T* src_ptr = src + src_off;

        auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<T*>(src_ptr), /*stride=*/0, din_num_bytes, data_format);

        T* lds_row = &lds[dout_local * DGRAD_LDS_DOUT_STRIDE];
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            rsrc, lds_row, /*bytes_per_lane=*/4, lane * 4, 0, 0, 0);
    }

    wait_vmcnt<0>();
    __syncthreads();

    // Store phase: tr-read the LDS [dout][din] tile as operands, store lane*8.
    //
    // wave w handles k16 = w; waves beyond k16_count (none for Kwg=64) sit out.
    if(wave >= k16_count)
        return;
    const int k16 = wave;

    using DstLayout = WeightsLayout<Kh, Kw, Kwg, 8>;
    auto base =
        DstLayout(Kq * Kwg, C_padded_dout).kq(wg_kq).c64(wg_c64).kw(wg_kw).kh(wg_kh).k16(k16);
    auto* dst_v8 = reinterpret_cast<datatypex8_t*>(sub_dst);

    // M = din (contiguous, via col()), reduction = dout (strided, via row()).
    //
    // This wave's k16 sub-tile occupies LDS din columns [k16*16, +16).
    using TR            = TransposeLDSLayout<16, 32>;
    const int din_col   = k16 * 16 + TR::col(lane);
    const int dout_row0 = TR::row(lane, 0);
    const int dout_row1 = TR::row(lane, 1);

#pragma unroll
    for(int c32_idx = 0; c32_idx < 2; ++c32_idx)
    {
        // c32_idx selects which 32-dout half of the 64-row LDS tile.
        const int dout_half = 32 * c32_idx;
        auto* a0            = reinterpret_cast<int16x4_t*>(
            &lds[(dout_row0 + dout_half) * DGRAD_LDS_DOUT_STRIDE + din_col]);
        auto* a1 = reinterpret_cast<int16x4_t*>(
            &lds[(dout_row1 + dout_half) * DGRAD_LDS_DOUT_STRIDE + din_col]);

        int16x4_t r0 = __builtin_amdgcn_ds_read_tr16_b64_v4i16(a0);
        int16x4_t r1 = __builtin_amdgcn_ds_read_tr16_b64_v4i16(a1);

        datatypex8_t frag = __builtin_bit_cast(
            datatypex8_t, (int16x8_t){r0[0], r0[1], r0[2], r0[3], r1[0], r1[1], r1[2], r1[3]});

        dst_v8[base.c32(c32_idx).offset + lane] = frag;
    }
}

template <int Kh, int Kw, int Kwg, hipconv::DataType DT>
__global__ void transpose_weights_dgrad_kernel(const ToType<DT>* __restrict__ src,
                                               ToType<DT>* __restrict__ dst,
                                               int K_dout,
                                               int C_din,
                                               int C_padded_dout,
                                               int Kq,
                                               int Kwg_total)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_read_tr16_b64_v4i16))
    {
        transpose_weights_dgrad_subtensor_impl<Kh, Kw, Kwg, DT>(
            src, dst, K_dout, C_din, C_padded_dout, Kq, Kwg_total);
    }
}

inline constexpr int make_divisible(int n, int divisor)
{
    int dm1 = divisor - 1;
    return (n + dm1) & ~dm1;
}

// Padded channel counts: K to a multiple of WavesK*Kwg, C to a multiple of 64.
//
// k_padded_override, when > 0, forces K_padded (the K-partition may pad the block
// count past the plain block_k rounding, adding pure-pad K-blocks the buffer must
// hold). It must be a multiple of WavesK*Kwg and >= the plain rounding.
template <int Kwg, int WavesK = 2>
inline void
get_padded_channels(int K, int C, int& K_padded, int& C_padded, int k_padded_override = 0)
{
    K_padded = k_padded_override > 0 ? k_padded_override : make_divisible(K, WavesK * Kwg);
    C_padded = make_divisible(C, C_divisor);
}

// Host launch wrapper (fprop).
//
// Caller owns device allocations:
//   src: K * Kh * Kw * C elements, channels-last (raw, unpadded).
//   dst: >= WavesK * sub_elts (sub_elts = Kq*Kwg*Kh*Kw*C_padded,
//        Kq = ceil(K / (WavesK*Kwg))). OOB src coords land 0 in dst.
// The WavesK sub-tensors live adjacently, partition i owning K rows
// [q*WavesK*Kwg + i*Kwg, +Kwg). K/C are per-group; groups format via blockIdx.y.
// WavesK defaults to 2; 4 packs K(256).
template <int Kh, int Kw, int Kwg, hipconv::DataType DT, int WavesK = 2>
inline void launch_transpose_weights(const ToType<DT>* d_src,
                                     ToType<DT>* d_dst,
                                     int K,
                                     int C,
                                     int groups            = 1,
                                     hipStream_t stream    = 0,
                                     int k_padded_override = 0)
{
    static_assert(Kwg % 16 == 0 && Kwg >= 16 && Kwg <= 64, "Kwg must be in {16, 32, 48, 64}");

    constexpr int Kwg_total = WavesK * Kwg;

    int K_padded;
    int C_padded;
    get_padded_channels<Kwg, WavesK>(K, C, K_padded, C_padded, k_padded_override);
    const int Kq = K_padded / Kwg_total;

    const dim3 block(BLOCK_SIZE);
    // x = Kq * (C/64) * Kw * Kh (c64 fastest); y = group; z = K partition (WavesK).
    const int total = Kq * (C_padded / 64) * Kw * Kh;

    transpose_weights_kernel<Kh, Kw, Kwg, DT><<<dim3(total, groups, WavesK), block, 0, stream>>>(
        d_src, d_dst, K, C, C_padded, Kq, Kwg_total);
}

// Dgrad host launch wrapper.
//
// Mirrors launch_transpose_weights but binds the dest "K" axis to din and "C" to
// dout, and runs the dgrad kernel (C<->K swap + tap flip).
//   d_src: canonical K_dout x Kh x Kw x C_din (raw, unpadded).
//   d_dst: >= WavesK * sub_elts (sub_elts = Kq*Kwg*Kh*Kw*C_padded_dout,
//          Kq = ceil(C_din / (WavesK*Kwg))). din-OOB / dout padding land 0.
//   Kwg: din output channels per K-partition wave.
// K_dout/C_din are per-group. WavesK defaults to 2; 4 packs din(256).
template <int Kh, int Kw, int Kwg, hipconv::DataType DT, int WavesK = 2>
inline void launch_transpose_weights_dgrad(const ToType<DT>* d_src,
                                           ToType<DT>* d_dst,
                                           int K_dout,
                                           int C_din,
                                           int groups            = 1,
                                           hipStream_t stream    = 0,
                                           int k_padded_override = 0)
{
    static_assert(Kwg % 16 == 0 && Kwg >= 16 && Kwg <= 64, "Kwg must be in {16, 32, 48, 64}");

    constexpr int Kwg_total = WavesK * Kwg;

    // dest "K" = din (split, padded to WavesK*Kwg); dest "C" = dout (padded to 64).
    // din_padded may be forced past the plain rounding by the K-partition (dgrad's
    // output/reduction swap makes din the padded output axis).
    int din_padded;
    int dout_padded;
    get_padded_channels<Kwg, WavesK>(C_din, K_dout, din_padded, dout_padded, k_padded_override);
    const int Kq = din_padded / Kwg_total;

    const dim3 block(BLOCK_SIZE);
    // x = Kq * (dout/64) * Kw * Kh (c64 fastest); y = group; z = K partition (WavesK).
    const int total = Kq * (dout_padded / 64) * Kw * Kh;

    transpose_weights_dgrad_kernel<Kh, Kw, Kwg, DT>
        <<<dim3(total, groups, WavesK), block, 0, stream>>>(
            d_src, d_dst, K_dout, C_din, dout_padded, Kq, Kwg_total);
}

} // namespace hipconv::cdna4::direct_transpose_weights
