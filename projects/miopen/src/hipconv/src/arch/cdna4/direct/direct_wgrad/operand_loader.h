#pragma once

// LDS-to-register operand loads for the direct_wgrad compute phase.
//
// Both operands take the channel axis as their non-reduction dimension, so both are read with
// ds_read_b64_tr_b16. bunnies::load_tile drives the lane and round mapping; this file supplies
// the coordinate-to-LDS-offset functor. The swizzle those offsets apply is derived in
// docs/algorithms/direct/direct-wgrad-cdna4-lds-swizzle.md.

#include "bunnies.hpp"
#include "bunnies_cdna4.hpp"
#include "config.h"
#include "lds_layout.h"
#include "types.h"

#include "hipconv/conv2d_params.hpp"

namespace hipconv::cdna4::direct_wgrad
{

template <hipconv::DataType DT>
constexpr bunnies::fpfmt operand_fmt =
    DT == hipconv::DataType::bf16 ? bunnies::fpfmt::e8m7 : bunnies::fpfmt::e5m10;

// S contributes the A operand, C x Q; delta the B operand, Q x K.
template <hipconv::DataType DT>
using mat_s = arch::matrix<operand_fmt<DT>, MFMA_M, MFMA_K, bunnies::use::A>;
template <hipconv::DataType DT>
using mat_delta = arch::matrix<operand_fmt<DT>, MFMA_K, MFMA_N, bunnies::use::B>;
using mat_wgrad = arch::matrix<bunnies::fpfmt::e8m23, MFMA_M, MFMA_N, bunnies::use::Acc>;

// One wave's operand and accumulator tiles.
template <Config cfg, hipconv::DataType DT>
using rt_s = bunnies::reg_tile<mat_s<DT>, cfg.wave_c16, 1>;
template <Config cfg, hipconv::DataType DT>
using rt_delta = bunnies::reg_tile<mat_delta<DT>, 1, cfg.wave_k16>;
template <Config cfg>
using rt_wgrad = bunnies::reg_tile<mat_wgrad, cfg.wave_c16, cfg.wave_k16>;

// Assigns a value-initialized matrix, so nothing writes the tile's storage vector through a
// narrower type than the one it was declared with.
template <typename RegTile>
__device__ void zero_tile(RegTile& tile)
{
#pragma unroll
    for(int mb = 0; mb < RegTile::row_blocks; ++mb)
#pragma unroll
        for(int nb = 0; nb < RegTile::col_blocks; ++nb)
            tile.block(mb, nb) = typename RegTile::matrix{};
}

// Load this wave's C x Q(32) slice of the S row held in `row_lds`.
//
// q_shift is the filter-column offset of the read window: the compute phase reads the same row
// at shifts 0..kw-1, which is why the S tile carries a kw-1 column halo. c_base is the wave's
// channel origin within the workgroup's C block.
//
// Column and shift pass separately because under the batch unfold the column selects the packed
// image and the shift then moves within it, possibly into its halo; a pre-added shift would read
// as the next image's column. `SRowLayout::ladder_pays` picks between this form and
// `load_s_ladder`.
template <Config cfg, hipconv::DataType DT, typename Layout>
__device__ void load_s(rt_s<cfg, DT>& tile, ToType<DT>* row_lds, int q_shift, int c_base)
{
    // row is the lane's channel within the 16-row block, 4-aligned so the group divide is
    // exact; col is the Q index.
    bunnies::load_tile<arch::ds_read_b64_tr_b16>(
        tile, row_lds, [=](int mb, int /*nb*/, int row, int col) {
        return Layout::elem_offset_shifted(col, q_shift, (c_base + mb * MFMA_M + row) / 4);
    });
}

// Load the kw shifted S tiles at once, as one ladder over the row.
//
// A shift whose rotation matches the unshifted read's is that read's address plus a constant, so
// the tiles share one address register and the constant rides in the instruction's offset field.
// The ladder section of docs/algorithms/direct/direct-wgrad-cdna4-lds-swizzle.md derives which
// displacements fold and why `RowLayout::ladder_pays` gates the form.
//
// This drives the load itself where `load_s` goes through `bunnies::load_tile`. The fold needs the
// round-0 coordinate hoisted out of the round loop and the displacement known as a constant, and a
// map-driven loader can supply neither without being handed the schedule -- which is the loader's
// business, not the layout's. There being one ladder in the tree, it costs less to write the two
// loops here than to carry a primitive that has to be told them.
//
// Going through `load_tile` per shift instead costs an address register per read: the swizzle is
// not linear in the column, so the compiler cannot recover that two reads sit a fixed distance
// apart, and a wide ladder on a deep ring spills.
//
// The shift is the outermost loop, and that is not cosmetic. Which reads share an address is a
// matter of common subexpressions and does not depend on the order, but the schedule does, and
// issuing a block's whole ladder before moving to the next block measured 1.27x slower on a config
// with nothing to gain from the fold in the first place.
template <Config cfg, hipconv::DataType DT, typename Layout>
__device__ void load_s_ladder(rt_s<cfg, DT> (&tiles)[cfg.kw], ToType<DT>* row_lds, int c_base)
{
    using Inst   = arch::ds_read_b64_tr_b16;
    using Tile   = rt_s<cfg, DT>;
    using Matrix = typename Tile::matrix;
    using ld_t   = typename Inst::type;

    constexpr int bpi        = bunnies::bits_per_item(Matrix::fmt);
    constexpr int per_round  = Inst::bits_per_load / bpi;
    constexpr int num_rounds = Matrix::num_items / per_round;

    // Hoisted, so every displacement that folds is this address plus an immediate. row is the
    // lane's channel within the 16-row block and col the Q index, as in load_s.
    const auto coord0 = Matrix::map(Inst::map(bunnies::lane_id(), 0, bpi));

    bunnies::static_unroll<cfg.kw>([&](auto shift) {
        constexpr int s = decltype(shift)::value;
#pragma unroll
        for(int mb = 0; mb < Tile::row_blocks; ++mb)
            bunnies::static_unroll<num_rounds>([&](auto rnd) {
                constexpr int r     = decltype(rnd)::value;
                constexpr int d     = r * Layout::cols_per_round + s;
                constexpr int fixed = Layout::displacement(d);

                const int c4 = (c_base + mb * MFMA_M + coord0[0]) / 4;
                int offset;
                if constexpr(fixed != Layout::no_displacement)
                    offset = Layout::elem_offset_shifted(coord0[1], 0, c4) + fixed;
                else
                    offset = Layout::elem_offset_shifted(coord0[1], d, c4);
                Inst::load(row_lds + offset,
                           reinterpret_cast<ld_t*>(&tiles[s].block(mb, 0).data) + r);
            });
    });
}

// Load this wave's Q(32) x K slice of the delta row held in `row_lds`.
//
// Delta is read at a single column offset, so there is no shift argument. k_base is the wave's
// channel origin within the workgroup's K block.
template <Config cfg, hipconv::DataType DT, typename Layout>
__device__ void load_delta(rt_delta<cfg, DT>& tile, ToType<DT>* row_lds, int k_base)
{
    // Mirror of load_s: row is the Q index and col the 4-aligned channel.
    bunnies::load_tile<arch::ds_read_b64_tr_b16>(
        tile, row_lds, [=](int /*mb*/, int nb, int row, int col) {
        return Layout::elem_offset(row, (k_base + nb * MFMA_N + col) / 4);
    });
}

} // namespace hipconv::cdna4::direct_wgrad
