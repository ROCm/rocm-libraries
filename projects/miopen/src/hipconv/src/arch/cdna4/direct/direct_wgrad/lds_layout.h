#pragma once

// LDS geometry for the direct_wgrad row buffers, and the swizzle their operand reads use.
//
// Both operands sit in LDS channel-contiguous, S as W x C and delta as Q x K, and both feed the
// MFMA transposed, so both are read with ds_read_b64_tr_b16.
// docs/algorithms/direct/direct-wgrad-cdna4-lds-swizzle.md derives the swizzle.

#include "bunnies.hpp"
#include "bunnies_cdna4.hpp"
#include "config.h"

#include <array>
#include <utility>

namespace hipconv::cdna4::direct_wgrad
{
using arch = bunnies::arch_cdna4;

// LDS swizzle for the ds_read_b64_tr_b16 operand read, which is the 16-wide operand of
// mfma_f32_16x16x32_f16 and its bf16 twin. The read's lane map, the bank-conflict-free condition a
// phase of it has to meet, which bits of the column the rotation reads at each width, and why
// swizzle.h's three swizzles do not serve it are in direct-wgrad-cdna4-lds-swizzle.md.
template <int C_, int WUnfold_ = MFMA_K, int Halo_ = 0, int NUnfold_ = 1>
struct TransposeSwizzle
{
    // Channels in one row of the tile, and the uint2 (4-channel) groups they form.
    static constexpr int C  = C_;
    static constexpr int C4 = C / 4;

    // Distinct rotations.
    static constexpr int G = C4 / 4;

    // The halo is kw - 1 on an S row and 0 on a delta row.
    static constexpr int w_unfold    = WUnfold_;
    static constexpr int halo        = Halo_;
    static constexpr int w_per_image = WUnfold_ + Halo_;
    static constexpr int n_unfold    = NUnfold_;

    // spread() and pack() branch on this, the compiler having no way to fold their
    // non-power-of-two divisor to the identity on its own.
    static constexpr bool packed = NUnfold_ > 1;

    // Column offsets the read window takes: the halo plus the unshifted read.
    static constexpr int shifts = Halo_ + 1;

    static_assert(C % 4 == 0, "TransposeSwizzle needs C divisible by 4 (uint2 = 4 fp16)");
    static_assert((C4 & (C4 - 1)) == 0, "TransposeSwizzle needs C4 a power of two");
    static_assert(C4 >= 8,
                  "TransposeSwizzle is bank-conflict-free only for C4 >= 8; a row of C(16) "
                  "has one rotation and cannot separate x from x + 8, and a rotation that "
                  "could is not a multiple of 4 and would split a lane's 16-byte write");
    static_assert(WUnfold_ > 0 && (WUnfold_ & (WUnfold_ - 1)) == 0,
                  "TransposeSwizzle needs the per-image column count a power of two, so the "
                  "read side splits an operand column into (image, column) with a shift");

    // Physical LDS column of the operand column `col`. The caller adds the filter shift after,
    // a shift being allowed into the image's halo and never into the next image.
    static constexpr int spread(int col)
    {
        if constexpr(!packed)
            return col;
        else
            return (col / w_unfold) * w_per_image + col % w_unfold;
    }

    // The packed column a physical column carries, which the rotation keys on.
    static constexpr int pack(int x_)
    {
        if constexpr(!packed)
            return x_;
        else
            return (x_ / w_per_image) * w_unfold + x_ % w_per_image;
    }

    // Candidate rotations: three bits of the packed column, lowest result bit first, masked to
    // G - 1. Only the C(32) rows at four images need anything but the first; see the doc.
    static constexpr int num_candidates   = 3;
    static constexpr int candidates[3][3] = {{3, 1, 0}, {1, 3, 0}, {0, 1, 3}};

    // Unsigned internals, so the shifts and the mask lower to single ops.
    static constexpr int g_of(const int (&bits)[3], int p_)
    {
        unsigned p = p_;
        return static_cast<int>(((p >> bits[0]) & 1u) | (((p >> bits[1]) & 1u) << 1) |
                                (((p >> bits[2]) & 1u) << 2)) &
               (G - 1);
    }

    // Which operand column each lane reads, composed from ds_read_b64_tr_b16's own lane map.
    //
    // The bank-conflict search below has to know which columns a phase names at once, and that is
    // a property of ds_read_b64_tr_b16 and of the operand matrix rather than of this swizzle.
    // Composing their two maps keeps one encoding of it. A copy written out here would be a second,
    // free to drift from the instruction the loader actually issues, and the drift is silent: the
    // same count of reads at the same addresses, naming the wrong columns.
    //
    // Both operands agree on the map. The operand column is coord[1] of the A matrix and coord[0]
    // of the B matrix, and the two reduce to the same function of (lane, round), so the S and delta
    // rows share it as they share the swizzle.
    using read_inst = arch::ds_read_b64_tr_b16;
    using read_mat  = arch::matrix<bunnies::fpfmt::e5m10, MFMA_M, MFMA_K, bunnies::use::A>;

    static constexpr int read_bpi        = bunnies::bits_per_item(read_mat::fmt);
    static constexpr int items_per_round = read_inst::bits_per_load / read_bpi;
    static constexpr int read_rounds     = read_mat::num_items / items_per_round;

    // Tabulated, because the searches below call read_col inside four-deep loops and the composed
    // map costs two calls and an array where the arithmetic it replaces cost none. Evaluated
    // directly it exhausts the compiler's constexpr step budget.
    static constexpr auto read_cols = [] {
        std::array<std::array<int, read_rounds>, arch::wave_size> t{};
        for(int lane = 0; lane < arch::wave_size; ++lane)
            for(int round = 0; round < read_rounds; ++round)
                t[lane][round] =
                    read_mat::map(read_inst::map(lane, round * items_per_round, read_bpi))[1];
        return t;
    }();

    static constexpr int read_col(int lane, int round) { return read_cols[lane][round]; }

    // Columns one round advances.
    static constexpr int cols_per_round = read_col(0, 1) - read_col(0, 0);

    // Every lane advances by the same columns from one round to the next, so a round displaces the
    // whole read by a fixed amount instead of moving each lane its own distance.
    static constexpr bool rounds_advance_uniformly()
    {
        for(int lane = 0; lane < arch::wave_size; ++lane)
            for(int round = 0; round < read_rounds; ++round)
                if(read_col(lane, round) - read_col(lane, 0) != round * cols_per_round)
                    return false;
        return true;
    }
    static_assert(rounds_advance_uniformly(),
                  "the transpose read's rounds do not advance every lane by the same columns, so "
                  "a round is not a fixed displacement of the read");


    // Whether every phase of the transpose read hits 32 distinct uint2 slots under `bits`.
    //
    // Only the S tile is read at more than one shift, and a nonzero halo marks it.
    static constexpr bool conflict_free_under(const int (&bits)[3], int num_shifts)
    {
        for(int c_base = 0; c_base < C4; c_base += 4)
            for(int shift = 0; shift < num_shifts; ++shift)
                for(int phase = 0; phase < 2; ++phase)
                    for(int round = 0; round < read_rounds; ++round)
                    {
                        unsigned seen = 0;
                        for(int lane = phase * 32; lane < phase * 32 + 32; ++lane)
                        {
                            const int col = read_col(lane, round);
                            const int x_  = spread(col) + shift;
                            const int off =
                                (x_ * C4 + ((lane % 4 + c_base) ^ (4 * g_of(bits, col + shift)))) %
                                32;
                            if(seen & (1u << off))
                                return false;
                            seen |= 1u << off;
                        }
                    }
        return true;
    }

    static constexpr int pick_rotation()
    {
        for(int i = 0; i < num_candidates; ++i)
            if(conflict_free_under(candidates[i], shifts))
                return i;
        return -1;
    }
    static constexpr int rotation = pick_rotation();
    static_assert(rotation >= 0,
                  "no candidate rotation is bank-conflict-free for this row geometry; the "
                  "read pattern has moved and the candidate list has to grow with it");

    // The chosen rotation's three bits, hoisted out of the candidate table.
    //
    // g() sits on the address path of every LDS read, so its shift amounts have to be immediates.
    // Indexing candidates[rotation] inside it keeps the array live and costs the folded kernel its
    // spill-free register allocation.
    static constexpr int bit_lo  = candidates[rotation][0];
    static constexpr int bit_mid = candidates[rotation][1];
    static constexpr int bit_hi  = candidates[rotation][2];

    static constexpr int g(int p_)
    {
        unsigned p = p_;
        return static_cast<int>(((p >> bit_lo) & 1u) | (((p >> bit_mid) & 1u) << 1) |
                                (((p >> bit_hi) & 1u) << 2)) &
               (G - 1);
    }

    // The layout: (physical column, 4-channel group) -> uint2 offset within the row buffer.
    static constexpr int offset_uint2(int x, int c4) { return x * C4 + (c4 ^ (4 * g(pack(x)))); }

    // The same slot, named the way the read side has it, which avoids dividing by w_per_image.
    static constexpr int offset_uint2_shifted(int col, int shift, int c4)
    {
        return (spread(col) + shift) * C4 + (c4 ^ (4 * g(col + shift)));
    }

    // ---- the read as a ladder ----
    //
    // Rung d is the round-0 unshifted read displaced d columns, and rungs that fold onto it share
    // one address register; see the ladder section of the doc and `load_s_ladder`.

    // Whether the read at (round, shift) is the round-0 read at rung 4 * round + shift.
    //
    // Survives the unfold because a round-0 column's low three bits are at most 3, so a round's
    // four columns stay inside one packed image.
    static constexpr bool rounds_are_rungs()
    {
        for(int round = 0; round < read_rounds; ++round)
            for(int shift = 0; shift < shifts; ++shift)
                for(int lane = 0; lane < 64; ++lane)
                    for(int c4_ = 0; c4_ < C4; ++c4_)
                        if(offset_uint2_shifted(read_col(lane, round), shift, c4_) !=
                           offset_uint2_shifted(
                               read_col(lane, 0), round * cols_per_round + shift, c4_))
                            return false;
        return true;
    }

    // The uint2 distance to the read displaced `d` columns, or no_rung where it is not fixed.
    //
    // The rotation moves with the column, so the distance is fixed only where every column the
    // read can name keeps its rotation that far on. Which displacements the ladder asks about is
    // the loader's business; this answers only whether one of them holds a distance.
    static constexpr int no_rung = -1;

    static constexpr int displacement(int d)
    {
        for(int lane = 0; lane < arch::wave_size; ++lane)
            for(int c4_ = 0; c4_ < C4; ++c4_)
            {
                const int col = read_col(lane, 0);
                if(offset_uint2_shifted(col, d, c4_) != offset_uint2_shifted(col, 0, c4_) + d * C4)
                    return no_rung;
            }
        return d * C4;
    }

    // Inverses, for the global-to-LDS side: a lane owns a fixed LDS slot and has to discover
    // which source element belongs there.
    static constexpr int x(int off) { return static_cast<unsigned>(off) / C4; }
    static constexpr int c4(int off) { return c4_at(off, pack(x(off))); }

    // The same inverse where the caller already knows the packed column, which the loader does:
    // it needs pack()'s non-power-of-two quotient anyway to find which image a slot reads from.
    static constexpr int c4_at(int off, int p)
    {
        return static_cast<int>(static_cast<unsigned>(off) % C4) ^ (4 * g(p));
    }

    static constexpr bool addressings_agree()
    {
        for(int col = 0; col < 32; ++col)
            for(int shift = 0; shift < shifts; ++shift)
                for(int c4_ = 0; c4_ < C4; ++c4_)
                    if(offset_uint2_shifted(col, shift, c4_) !=
                       offset_uint2(spread(col) + shift, c4_))
                        return false;
        return true;
    }

    // Whether the map permutes each column's channel groups, which the global-to-LDS write needs
    // or it would drop some slots and write others twice.
    static constexpr bool inverses_round_trip()
    {
        for(int x_ = 0; x_ < 64; ++x_)
            for(int c4_ = 0; c4_ < C4; ++c4_)
            {
                const int off = offset_uint2(x_, c4_);
                if(off / C4 != x_ || x(off) != x_ || c4(off) != c4_)
                    return false;
            }
        return true;
    }

    static_assert(inverses_round_trip(),
                  "TransposeSwizzle::offset_uint2 and x()/c4() are not mutual inverses");
    static_assert(addressings_agree(),
                  "TransposeSwizzle::offset_uint2_shifted does not reach the slot "
                  "offset_uint2 names for the same element");
    static_assert(rounds_are_rungs(),
                  "a round of the transpose read is not a filter shift of the round before "
                  "it, so the read is not a ladder and displacement() does not describe it");

    static constexpr bool is_bank_conflict_free(int num_shifts)
    {
        return conflict_free_under(candidates[rotation], num_shifts);
    }
    static_assert(is_bank_conflict_free(shifts));
};

// One spatial item's LDS row: Cols columns of Chans channels, swizzled. The ring geometry around
// it belongs to the row loader.
//
// Cols rounds the row up to a whole load round, so an item begins on a round boundary and a round
// never straddles two items. The loader forces every column at or past LiveCols out of bounds, so
// the padding costs LDS and the writes of the zeros it returns, and no global traffic.
//
// WUnfold and Halo divide the live columns into packed images, the halo being the columns only the
// filter shifts reach. The defaults are the delta row, one image with no halo.
template <int Cols_, int Chans_, int LiveCols_ = Cols_, int WUnfold_ = MFMA_K, int Halo_ = 0>
struct RowLayout
{
    static constexpr int w_unfold    = WUnfold_;
    static constexpr int w_per_image = WUnfold_ + Halo_;
    static constexpr int n_unfold    = LiveCols_ / w_per_image;
    static_assert(n_unfold * w_per_image == LiveCols_,
                  "the live columns must divide into whole packed images");

    using Swizzle = TransposeSwizzle<Chans_, WUnfold_, Halo_, n_unfold>;

    static constexpr int cols      = Cols_;
    static constexpr int chans     = Chans_;
    static constexpr int live_cols = LiveCols_;
    static_assert(live_cols <= cols);

    static constexpr int size_uint2 = cols * Swizzle::C4;
    static constexpr int size_elems = 4 * size_uint2;

    // Element offset of a (physical column, 4-channel group) within the row.
    __device__ __host__ static constexpr int elem_offset(int x, int c4)
    {
        return 4 * Swizzle::offset_uint2(x, c4);
    }

    // The same element, named the way the compute phase has it.
    __device__ __host__ static constexpr int elem_offset_shifted(int col, int shift, int c4)
    {
        return 4 * Swizzle::offset_uint2_shifted(col, shift, c4);
    }

    // The element distance to the read displaced `d` columns, or no_displacement where the rotation
    // moves and there is no fixed distance. This is the whole of what `load_s_ladder` asks of a
    // layout: the loader decides which displacements it will issue and what to do about the answer.
    static constexpr int no_displacement = Swizzle::no_rung;

    // Columns one round of the transpose read advances, which the loader counts displacements in.
    static constexpr int cols_per_round = Swizzle::cols_per_round;

    static constexpr int displacement(int d)
    {
        const int off = Swizzle::displacement(d);
        return off == Swizzle::no_rung ? no_displacement : 4 * off;
    }

    static constexpr int max_rung =
        (Swizzle::read_rounds - 1) * Swizzle::cols_per_round + Swizzle::shifts - 1;

    // Whether addressing the read as a ladder buys an address register at all. The compiler folds
    // two rungs unaided, and the ladder section of the doc has what a row folding no more costs.
    static constexpr bool ladder_pays()
    {
        int folded = 0;
        for(int d = 0; d <= max_rung; ++d)
            if(Swizzle::displacement(d) != Swizzle::no_rung)
                ++folded;
        return folded > 2;
    }
};

// A ring of row buffers, held as pointers to separate LDS objects.
//
// Slots are selected at compile time. A runtime select would merge the pointers into one value
// and lose the object identity the waitcnt disambiguation rests on.
template <typename T, int NumSlots>
struct RowRing
{
    T* slots[NumSlots];

    template <int Slot>
    __device__ T* get() const
    {
        static_assert(0 <= Slot && Slot < NumSlots);
        return slots[Slot];
    }
};

// Declare Count row buffers of Size elements and hand them to `body` as pointers.
//
// The recursion makes them separate LDS objects: each depth is its own instantiation and carries
// its own __shared__ declaration. A loop, an [N][SIZE] array, and a struct of array members all
// give one object instead, and the transpose read of any slot then drains the DMA to every other
// (see Config::row_buffers).
template <typename T, int Size, int Count, typename F, typename... Rows>
__device__ void with_rows(F&& body, Rows*... rows)
{
    if constexpr(sizeof...(Rows) == Count)
    {
        body(rows...);
    }
    else
    {
        __shared__ T row[Size];
        with_rows<T, Size, Count>(body, rows..., row);
    }
}

// Declare Count row buffers of Size elements as one LDS object, and hand `body` a ring into it.
//
// Serves the prologue's scratch rows, which are all written before its single wait and all read
// after it, so no read of theirs overlaps DMA to a row beside them and nothing needs with_rows'
// object identity.
//
// One object puts the rows a compile-time stride apart, so their reads share an address register
// where separate objects cost one each. Those addresses were spilling on the packed configs, and a
// spilled LDS address reloaded in the prologue can only be waited on with vmcnt(0), at the one
// point in the item where the prefetch is deliberately in flight.
//
// The alignment is the load width, which a row is a whole number of and the block's own base is
// not otherwise promised to be.
template <typename T, int Size, int Count, typename F>
__device__ void with_row_block(F&& body)
{
    __shared__ __align__(16) T block[Count * Size];

    const auto ring = [&]<int... I>(std::integer_sequence<int, I...>) {
        return RowRing<T, Count>{{(block + I * Size)...}};
    }(std::make_integer_sequence<int, Count>{});

    body(ring);
}

} // namespace hipconv::cdna4::direct_wgrad
