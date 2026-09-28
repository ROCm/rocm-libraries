#pragma once

// Per-wavegroup global-memory weights loader for the direct_l1 kernel.
//
// Selects the wave's sub-tensor of the pre-formatted workspace, then issues b128
// buffer loads landing one MFMA operand tile (K(16) x C(32) = 1024 bytes) each;
// the cursor bumps 1024 bytes per tile. A step loads Kh x (Kwg/16) tiles.
//
// All waves_k sub-tensors are shape-identical and differ only in base, so one
// loader serves any partition: select the base by wave_k_idx (a scalar select)
// rather than a per-step branch that would double the load instruction count.

#include "config.h"
#include "layer_pars.h"
#include "weights_layout.h"
#include "types.h"

#include <hip/hip_runtime.h>
#include <type_traits>

namespace hipconv::cdna4::direct_l1
{

template <Config cfg, typename datatype_t>
class WeightsLoader
{
    static_assert(sizeof(datatype_t) == 2);

    static constexpr int rsrc_data_format = 1 << 15;
    static constexpr int Kh               = cfg.kh;
    static constexpr int Kwg              = cfg.wave_k();

    using Vector = __attribute__((ext_vector_type(4))) unsigned int;
    // Veclen=1 so offsets/strides are in fp16 elements (datatype_t units).
    using Layout = WeightsLayout<cfg.kh, cfg.kw, Kwg, 1>;

public:
    using datatypex8_t = std::conditional_t<std::is_same_v<datatype_t, bf16_t>, bf16x8_t, fp16x8_t>;

    // Number of K(16) x C(32) tiles per step, and the per-step k16 extent.
    static constexpr int max_k16        = Kwg / 16;
    static constexpr int tiles_per_step = Kh * (Kwg / 16);

    // 1024 bytes = one K(16) x C(32) f16 operand tile.
    static constexpr int bytes_per_tile = 1024;

    // K = the partition's K share; k_idx selects the workgroup, wave_k_idx the sub-tensor.
    __device__
    WeightsLoader(int K, int C_padded, int k_idx, const datatype_t* weights_global, int wave_k_idx)
        : cursor_bytes_(0)
    {
        const datatype_t* base = select_base(K, C_padded, k_idx, weights_global, wave_k_idx);
        rsrc_                  = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<datatype_t*>(base), 0, stripe_bytes(K, C_padded), rsrc_data_format);
    }

    // Load one step's tiles into the caller's register array.
    //
    // Advances the cursor by tiles_per_step; the schedule (step -> (Kw, c32%2)) is
    // the caller's responsibility.
    __device__ void load_step(datatypex8_t (&tiles)[Kh][Kwg / 16])
    {
        // Wave-uniform cursor routed through soffset; readfirstlane avoids a waterfall.
        const int base_bytes = __builtin_amdgcn_readfirstlane(cursor_bytes_);

        // Lane L reads [L*16, L*16+16), so 64 lanes cover one 1024-byte tile.
        const int lane_off = __lane_id() * 16;

#pragma unroll
        for(int kh = 0; kh < Kh; ++kh)
        {
#pragma unroll
            for(int k16 = 0; k16 < (Kwg / 16); ++k16)
            {
                const int tile_idx = kh * (Kwg / 16) + k16;
                const int soff     = base_bytes + tile_idx * bytes_per_tile;

                Vector raw     = __builtin_amdgcn_raw_buffer_load_b128(rsrc_, lane_off, soff, 0);
                tiles[kh][k16] = __builtin_bit_cast(datatypex8_t, raw);
            }
        }

        cursor_bytes_ = base_bytes + tiles_per_step * bytes_per_tile;
    }

private:
    __device__ static const datatype_t*
    select_base(int K, int C_padded, int k_idx, const datatype_t* weights_global, int wave_k_idx)
    {
        // wave_k_idx selects the sub-tensor, k_idx the workgroup along K within it.
        const int off = Layout(K, C_padded).wave_group(wave_k_idx).kq(k_idx).offset;
        // off differs between K partitions; readfirstlane pins the descriptor to SGPRs.
        return weights_global + __builtin_amdgcn_readfirstlane(off);
    }

    __device__ static int stripe_bytes(int K, int C_padded)
    {
        return Layout(K, C_padded).strides().kq() * static_cast<int>(sizeof(datatype_t));
    }

    __amdgpu_buffer_rsrc_t rsrc_;
    int cursor_bytes_;
};

} // namespace hipconv::cdna4::direct_l1
