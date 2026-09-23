// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"

namespace ck_tile {

// Store out an M*N row-major matrix, staging it through LDS so that the global
// stores are coalesced.
//
// The accumulator arrives in the warp-GEMM C layout: on a wave32 machine a lane
// owns 8 contiguous elements every 16, so the 32 lanes of one store instruction
// land on 16 different rows, two 16-byte fragments each. Rows are stride_o apart
// (e.g. 512 B for BSHD with h=2, d=128), so a single instruction touches 16
// separate cache lines and writes a fraction of each. Staging through LDS
// (a warp-level transpose) lets every store instruction cover two whole rows
// and issue a single coalesced HBM write.
//
// This is the same idea as CShuffleEpilogue, but that one is tied to the XDL
// warp-GEMM shape (kMPerXdl / kNPerXdl / BlockedXDLN_PerWarp) and carries the
// D-tensor chaining machinery; a wave32 WMMA C layout does not fit its problem
// description, hence a separate, much smaller epilogue.

// NOTE: the warp count, not the block size, is the template parameter.
// get_warp_size() is constexpr but returns 64 on host (where the kernel symbol
// is formed) and 32 on device (wave32). Using kNumWarps keeps the type
// independent of the host/device boundary and avoids a host/device symbol mismatch.
template <typename AccDataType_,
          typename ODataType_,
          index_t kNumWarps_,
          index_t kMPerBlock_,
          index_t kNPerBlock_,
          bool kPadM_,
          bool kPadN_>
struct LdsShuffle2DEpilogueProblem
{
    using AccDataType                   = remove_cvref_t<AccDataType_>;
    using ODataType                     = remove_cvref_t<ODataType_>;
    static constexpr index_t kNumWarps  = kNumWarps_;
    static constexpr index_t kMPerBlock = kMPerBlock_;
    static constexpr index_t kNPerBlock = kNPerBlock_;
    static constexpr bool kPadM         = kPadM_;
    static constexpr bool kPadN         = kPadN_;
    static constexpr index_t NumDTensor = 0;
};

template <typename Problem_, typename Policy_ = void>
struct LdsShuffle2DEpilogue
{
    using Problem     = remove_cvref_t<Problem_>;
    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType   = remove_cvref_t<typename Problem::ODataType>;

    static constexpr index_t kNumWarps = Problem::kNumWarps;
    static constexpr index_t kM        = Problem::kMPerBlock;
    static constexpr index_t kN        = Problem::kNPerBlock;
    static constexpr bool kPadM        = Problem::kPadM;
    static constexpr bool kPadN        = Problem::kPadN;

    static_assert(Problem::NumDTensor == 0, "D-tensor chaining is not implemented");
    static_assert(sizeof(ODataType) <= 16, "ODataType wider than 16 B is not supported");

    // 16-byte vector access, the widest ds_/buffer_ op.
    static constexpr index_t kVectorMax = 16 / static_cast<index_t>(sizeof(ODataType));

    // A vectorized global store carries ONE out-of-bounds flag for the whole
    // access, taken from its first element (tensor_view::set_vectorized_elements),
    // so a vector that straddles the N boundary is written in full. When N is
    // padded the dispatcher only guarantees hdim_v % kPadNAlignment == 0
    // (fmha_fwd.py: the qr_tdm dvcheck), so the store vector must divide that,
    // or the tail access spills into the next row. Unpadded N is exact and
    // needs no clamp. kPadM needs no equivalent: the M vector length is 1,
    // rows being stride_o apart, so the row check is already per-element.
    static constexpr index_t kPadNAlignment = 8;
    static constexpr index_t kVector =
        kPadN ? (kVectorMax < kPadNAlignment ? kVectorMax : kPadNAlignment) : kVectorMax;

    // Pad the LDS row pitch so that row_bytes % 128 == 32. A wave writing the
    // warp-GEMM layout covers one contiguous 8-dword block per row; with the
    // pitch 32 bytes off a bank-cycle those 16 blocks start at four distinct
    // bank offsets, four rows each, which is the conflict-free minimum for the
    // 128 dwords a wave32 ds_write_b128 moves. An unpadded 256-byte pitch puts
    // every row on the same bank. Solve for the pad rather than assuming an
    // already-128-byte-aligned row, so the property survives any kN.
    static constexpr index_t kRowBytes = kN * static_cast<index_t>(sizeof(ODataType));
    static constexpr index_t kPadBytes = ((32 - kRowBytes) % 128 + 128) % 128;
    static constexpr index_t kNLds     = kN + kPadBytes / static_cast<index_t>(sizeof(ODataType));

    static_assert(kN % kVector == 0, "kN must be a multiple of the vector size");
    static_assert(kPadBytes % static_cast<index_t>(sizeof(ODataType)) == 0,
                  "LDS row pad must be a whole number of elements");
    static_assert((kRowBytes + kPadBytes) % 128 == 32, "LDS row pitch is not bank-staggered");
    static_assert((kNLds * sizeof(ODataType)) % 16 == 0,
                  "LDS row pitch must stay 16-byte aligned for b128 access");

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return static_cast<index_t>(kM * kNLds * sizeof(ODataType));
    }

    // The global store itself, in whichever layout, is Default2DEpilogue's. Its
    // raw store on padded windows matters: plain store_tile there lets the
    // compiler build the buffer resource in VGPRs and wrap every buffer_store in
    // a readfirstlane waterfall loop.
    using DirectStore =
        Default2DEpilogue<Default2DEpilogueProblem<AccDataType, ODataType, kPadM, kPadN>>;

    // Advertises that operator() takes a 4th (smem) argument; read through
    // ck_tile::detail::epilogue_uses_smem_v. The caller cannot probe the call
    // expression instead: Default2DEpilogue and others already have a defaulted
    // `void*` 4th parameter, so an arity probe matches them too.
    static constexpr bool kUsesSmem = true;

    // The LDS-shuffle path is selected on the *type* of the smem argument, not
    // its value: a caller with no spare LDS passes the `nullptr` literal
    // (std::nullptr_t) and gets the direct-store fallback, everyone else passes
    // a real pointer. A runtime `p_smem == nullptr` test here would make the
    // compiler emit both paths -- the dead one still participates in register
    // allocation, which on the gfx1250 FMHA kernel is 156 wasted instructions.
    //
    // The parameter is defaulted so that this epilogue stays substitutable for
    // Default2DEpilogue at the eleven existing three-argument call sites; those
    // keep the direct store, which is correct, just not coalesced.
    template <typename ODramWindowTmp,
              typename OAccTile,
              typename DsDramWindows,
              typename PSmem = std::nullptr_t>
    CK_TILE_DEVICE auto operator()(ODramWindowTmp& o_dram_window_tmp,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows&,
                                   PSmem p_smem = nullptr) const
    {
        if constexpr(std::is_same_v<PSmem, std::nullptr_t>)
        {
            // No spare LDS: store the accumulator layout straight out, scattered.
            DirectStore{}(o_dram_window_tmp, o_acc_tile, nullptr);
        }
        else
        {
            const auto lds_desc =
                make_naive_tensor_descriptor(make_tuple(number<kM>{}, number<kN>{}),
                                             make_tuple(number<kNLds>{}, number<1>{}),
                                             number<kVector>{},
                                             number<1>{});

            auto o_lds = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<ODataType*>(p_smem), lds_desc);

            auto in_lds_window = make_tile_window(o_lds,
                                                  make_tuple(number<kM>{}, number<kN>{}),
                                                  {0, 0},
                                                  o_acc_tile.get_tile_distribution());

            // The pipeline's last V TDM may still be on the tensorcnt counter;
            // drain both counters before writing LDS to avoid a WAW hazard.
            // Spell both: the single-arg form skips s_wait_dscnt by default.
            s_wait_tensorcnt_barrier<0, 0>();
            store_tile(in_lds_window, cast_tile<ODataType>(o_acc_tile));
            block_sync_lds();

            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<kNumWarps * get_warp_size(),
                                                      kM,
                                                      kN,
                                                      kVector,
                                                      tile_distribution_pattern::thread_raked>;

            constexpr auto dram_dist = TileEncodingPattern::make_2d_static_tile_distribution();

            auto out_lds_window =
                make_tile_window(o_lds, make_tuple(number<kM>{}, number<kN>{}), {0, 0}, dram_dist);

            DirectStore{}(o_dram_window_tmp, load_tile(out_lds_window), nullptr);
        }
    }
};

} // namespace ck_tile
