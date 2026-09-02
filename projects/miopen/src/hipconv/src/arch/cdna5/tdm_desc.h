#pragma once

#include "bunnies_mi400.hpp"

namespace bunnies
{

// Reusable wrapper around the gfx1250 TDM (`tensor_load_to_lds`) descriptor set.
// The static fields (data_size, tensor_dim0, tile_dim0, tensor_stride1 and the
// optional inner-dim padding) are configured once via init(); only the per-load
// varying fields (global_addr, lds_addr, tensor_dim1, tile_dim1) are patched
// before each load(). This avoids rebuilding the full descriptor on every call.
//
// `tensor_dim{0,1}` are full-tensor element counts (positive-side OOB reads
// return zero); `tile_dim{0,1}` are the tile element counts; `row_stride_elems`
// is the per-row pitch in elements. When enabled, padding inserts
// (pad_amount + 1) dwords every 2^(pad_interval + 1) dwords of the inner dim.
struct TdmDesc
{
    arch_mi400::tdm_group0 d0{};
    arch_mi400::tdm_group1 d1{};
    arch_mi400::tdm_group2 d2{};
    arch_mi400::tdm_group3 d3{};
    arch_mi400::tdm_group4 d4{};

    __device__ __forceinline__ void init(unsigned data_size_bytes,
                                         unsigned tensor_dim0,
                                         unsigned tile_dim0,
                                         unsigned long long row_stride_elems,
                                         bool pad_enable       = false,
                                         unsigned pad_interval = 0,
                                         unsigned pad_amount   = 0)
    {
        d1.data_size = ilog2(data_size_bytes);
        d1.set_tensor_dim0(tensor_dim0);
        d1.tile_dim0 = tile_dim0;
        d1.set_tensor_stride1(row_stride_elems);
        d1.pad_enable   = pad_enable ? 1u : 0u;
        d1.pad_interval = pad_interval;
        d1.pad_amount   = pad_amount;
    }

    __device__ __forceinline__ void load(unsigned long long global_addr_bytes,
                                         unsigned lds_offset_bytes,
                                         unsigned tensor_dim1,
                                         unsigned tile_dim1)
    {
        d0.set_global_addr(static_cast<uintptr_t>(global_addr_bytes));
        d0.lds_addr = lds_offset_bytes;
        d1.set_tensor_dim1(tensor_dim1);
        d1.tile_dim1 = tile_dim1;
        __builtin_amdgcn_tensor_load_to_lds(d0.data, d1.data, d2.data, d3.data, d4.data, 0);
    }
};

} // namespace bunnies
