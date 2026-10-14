#pragma once

#include "bunnies_mi400.hpp"

namespace bunnies
{

// Reusable wrapper around the gfx1250 TDM descriptor set: init() pins the static fields and only
// global_addr, lds_addr and the outer extents are patched per load. tensor_dim* are element counts
// measured forward from global_addr, so a tile past one is zero-filled and a zero extent clears it.
struct TdmDesc
{
    arch_mi400::tdm_group0 d0{};
    arch_mi400::tdm_group1 d1{};
    arch_mi400::tdm_group2 d2{};
    arch_mi400::tdm_group3 d3{};
    arch_mi400::tdm_group4 d4{};

    // Padding, when enabled, inserts (pad_amount + 1) dwords every 2^(pad_interval + 1) dwords of
    // the inner dim.
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

    // Pin the outer-dim extents so load() need not re-pack them: they live in split 16-bit halves
    // of d1, so a caller whose extents are loop-invariant leaves d1 wholly constant by setting them
    // here instead of through the four-argument load().
    __device__ __forceinline__ void set_dim1(unsigned tensor_dim1, unsigned tile_dim1)
    {
        d1.set_tensor_dim1(tensor_dim1);
        d1.tile_dim1 = tile_dim1;
    }

    // Third mode, off by default (tile_dim2 == 0 is what makes a descriptor 2D). At tile_dim2 == 1
    // it is a per-load "does this slice exist" flag costing one store, since tensor_dim2 is a whole
    // word of d2 and tensor_stride2 is never stepped.
    __device__ __forceinline__ void set_tile_dim2(unsigned tile_dim2) { d1.tile_dim2 = tile_dim2; }
    __device__ __forceinline__ void set_dim2(unsigned tensor_dim2)
    {
        d2.set_tensor_dim2(tensor_dim2);
    }
    // Only needed once tile_dim2 > 1 actually steps the mode.
    __device__ __forceinline__ void set_stride2(unsigned long long stride_elems)
    {
        d1.set_tensor_stride2(stride_elems);
    }

    // One issue repeating the descriptor, stepping the global address and the LDS
    // address by a fixed increment each pass. Both increments count elements, not
    // bytes, and the count field holds passes minus one. Extents are re-applied per
    // pass rather than consumed, so a run of rows batches whatever its extents say.
    __device__ __forceinline__ void
    set_iterate(unsigned lds_inc_elems, unsigned long long global_inc_elems, unsigned passes)
    {
        d1.iterate_enable = passes > 1 ? 1u : 0u;
        d2.set_tensor_dim3(lds_inc_elems);
        d2.set_tensor_stride3(global_inc_elems);
        d2.tile_dim3 = passes - 1;
    }

    // Group 0 filled in, and nothing started. Split from the issue so that a caller with
    // a barrier between the two can put the addressing on one side and the transfer on
    // the other: working out where a tile comes from is free at any time, and starting
    // to write it is not free until the LDS it lands in is dead.
    __device__ __forceinline__ void arm(unsigned long long global_addr_bytes,
                                        unsigned lds_offset_bytes)
    {
        d0.set_global_addr(static_cast<uintptr_t>(global_addr_bytes));
        d0.lds_addr = lds_offset_bytes;
    }

    // The issue alone, against whatever arm() and the extent setters last left behind.
    __device__ __forceinline__ void fire_load()
    {
        arch_mi400::tensor_load_to_lds(d0, d1, d2, d3, d4);
    }

    // Issue a load against the extents already in the descriptor.
    __device__ __forceinline__ void load(unsigned long long global_addr_bytes,
                                         unsigned lds_offset_bytes)
    {
        arm(global_addr_bytes, lds_offset_bytes);
        fire_load();
    }

    __device__ __forceinline__ void load(unsigned long long global_addr_bytes,
                                         unsigned lds_offset_bytes,
                                         unsigned tensor_dim1,
                                         unsigned tile_dim1)
    {
        set_dim1(tensor_dim1, tile_dim1);
        load(global_addr_bytes, lds_offset_bytes);
    }

    // LDS -> global against the same descriptor set. A tile reaching past an extent is clipped
    // rather than zero-filled, which is what lets an output tile skip its per-lane predicate.
    __device__ __forceinline__ void store(unsigned long long global_addr_bytes,
                                          unsigned lds_offset_bytes)
    {
        d0.set_global_addr(static_cast<uintptr_t>(global_addr_bytes));
        d0.lds_addr = lds_offset_bytes;
        d0.is_store = 1; // the "must be 0" in tdm_group0 is the load direction's constraint
        arch_mi400::tensor_store_from_lds(d0, d1, d2, d3, d4);
    }
};

} // namespace bunnies
