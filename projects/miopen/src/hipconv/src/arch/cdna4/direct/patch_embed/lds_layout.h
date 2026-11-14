#pragma once

// LDS layouts of the patch_embed tiles, and the swizzles that keep every access to them off a
// shared bank. test/cdna4/direct/patch_embed/swizzle_test.cpp checks every config's layouts
// against gfx950's bank model.

#include "bunnies_cdna4.hpp"
#include "config.h"

#include <array>

namespace hipconv::cdna4::patch_embed::detail
{

// A row-major tile of 16-bit elements, or one bf16 plane of a tf32 tile: rows of GRANULES 16-byte
// granules, granule g of a row stored at slot g ^ term(row). A four-granule row keys on bit 2 of
// the row, since ds_read_b128 serves rows r and r + 4 in one phase.
template <int GRANULES>
struct GranuleTile
{
    static_assert(GRANULES == 4 || GRANULES == 8 || GRANULES == 16,
                  "a row of 32, 64 or 128 elements; another width needs its own term");

    static constexpr int term(int row)
    {
        return GRANULES == 4 ? (row >> 1) & 2 : row & (GRANULES - 1);
    }

    // An involution, so the same map takes a slot back to the granule it holds.
    static constexpr int slot(int row, int g) { return g ^ term(row); }

    // Rows over which term repeats: a staging slot that walks rows a multiple of this apart
    // fetches one granule column throughout.
    static constexpr int period = GRANULES == 4 ? 8 : GRANULES;
};

// LDS layout for a tile ds_read_b64_tr_b16 reads transposed: dgrad's B, and both of wgrad's.
//
// The tile sits reduction-major with its output axis contiguous, which is the tensor's own layout
// in each case, so the stage stays a plain DTL copy and the transpose costs nothing but the read. A
// read phase would then name four banks, so a rotation keyed on the row spreads it. The candidate
// rotations and the conflict test are direct_wgrad's, specialised to a tile with no halo and no
// unfold; see docs/algorithms/details/direct/direct-wgrad-cdna4-lds-swizzle.md for the derivation.
template <int WIDTH>
struct TransposeTile
{
    using arch      = bunnies::arch_cdna4;
    using read_inst = arch::ds_read_b64_tr_b16;
    // Both 16-bit formats share one lane map, so the operand's own format does not matter here.
    using read_mat = arch::matrix<bunnies::fpfmt::e5m10, MFMA_N, MFMA_K, bunnies::use::A>;

    static constexpr int NG = WIDTH / 4; // four-element groups in a row
    static constexpr int G  = NG / 4;    // distinct rotations

    static constexpr int bpi    = bunnies::bits_per_item(read_mat::fmt);
    static constexpr int per_rd = read_inst::bits_per_load / bpi;
    static constexpr int rounds = read_mat::num_items / per_rd;

    // The reduction index each lane names, from the instruction's map and the operand's composed.
    static constexpr auto cols = [] {
        std::array<std::array<int, rounds>, arch::wave_size> t{};
        for(int lane = 0; lane < arch::wave_size; ++lane)
            for(int rnd = 0; rnd < rounds; ++rnd)
                t[lane][rnd] = read_mat::map(read_inst::map(lane, rnd * per_rd, bpi))[1];
        return t;
    }();

    static constexpr int candidates[3][3] = {{3, 1, 0}, {1, 3, 0}, {0, 1, 3}};

    static constexpr int rot_of(const int (&bits)[3], int p_)
    {
        unsigned p = p_;
        return static_cast<int>(((p >> bits[0]) & 1u) | (((p >> bits[1]) & 1u) << 1) |
                                (((p >> bits[2]) & 1u) << 2)) &
               (G - 1);
    }

    // Whether every phase of the read hits 32 distinct slots.
    static constexpr bool conflict_free(const int (&bits)[3])
    {
        for(int base = 0; base < NG; base += 4)
            for(int phase = 0; phase < 2; ++phase)
                for(int rnd = 0; rnd < rounds; ++rnd)
                {
                    unsigned seen = 0;
                    for(int lane = phase * 32; lane < phase * 32 + 32; ++lane)
                    {
                        const int col = cols[lane][rnd];
                        const int off =
                            (col * NG + ((lane % 4 + base) ^ (4 * rot_of(bits, col)))) % 32;
                        if(seen & (1u << off))
                            return false;
                        seen |= 1u << off;
                    }
                }
        return true;
    }

    static constexpr int pick()
    {
        for(int i = 0; i < 3; ++i)
            if(conflict_free(candidates[i]))
                return i;
        return -1;
    }
    static constexpr int rotation = pick();
    static_assert(rotation >= 0,
                  "no candidate rotation keeps the transposed read conflict-free; the read "
                  "pattern has moved and the candidate list has to grow with it");

    // Hoisted out of the table so the shifts lower to immediates on the address path.
    static constexpr int bit_lo  = candidates[rotation][0];
    static constexpr int bit_mid = candidates[rotation][1];
    static constexpr int bit_hi  = candidates[rotation][2];

    static constexpr int rot(int p_)
    {
        unsigned p = p_;
        return static_cast<int>(((p >> bit_lo) & 1u) | (((p >> bit_mid) & 1u) << 1) |
                                (((p >> bit_hi) & 1u) << 2)) &
               (G - 1);
    }

    // (reduction row, four-element group) -> element offset in the tile.
    static constexpr int elem(int n, int grp) { return (n * NG + (grp ^ (4 * rot(n)))) * 4; }

    // The group a staging slot owns, the inverse of elem()'s permutation.
    static constexpr int group_at(int n, int slot_grp) { return slot_grp ^ (4 * rot(n)); }
};

// The fp32 epilogue's stage: OUT_ROW elements a row, 16-byte chunks XORed by the row mod 16 in
// place of a pad, which an fp32 row has no room for at the widest tile.
template <int OUT_ROW>
struct OutStage
{
    static_assert(OUT_ROW % 64 == 0, "the term permutes sixteen chunks");

    static constexpr int elem(int r, int c)
    {
        return r * OUT_ROW + ((c / 4) ^ (r % 16)) * 4 + c % 4;
    }
};

} // namespace hipconv::cdna4::patch_embed::detail
