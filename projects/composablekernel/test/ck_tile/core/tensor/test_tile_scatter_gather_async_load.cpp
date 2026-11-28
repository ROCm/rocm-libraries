// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// tile_scatter_gather::async_load_with_offset on gfx1250: gathers the rows of a tile from global
// memory into LDS with async loads, then copies LDS out and compares with a host gather.
//
// The tile distribution has a gathered row dimension (Y0) with two values, so each thread loads
// two gathered rows. This catches LDS addresses taken per wave instead of per lane, LDS rows not
// stepped along the gather dimension, and gathered rows offset twice in DRAM. Both the stepped
// (static_move_ys = false) and the static-offset (static_move_ys = true) paths are tested. The
// copy-out reads LDS after s_wait_asynccnt and a barrier, so it also needs the async-load wait.

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <vector>

namespace {

using namespace ck_tile;

constexpr index_t kBlockSize = 128; // four waves of 32
constexpr index_t kRows      = 32;  // rows of the tile
constexpr index_t kRowBytes  = 128; // bytes per row
constexpr index_t kSrcRows   = 96;  // rows of the source the tile gathers from

// Rows (M0 x M1 x M2) = 2 x 4 waves x 4; bytes (K0 x K1 x K2) = 1 x 8 lanes x 16. Y0 = M0 is
// the gathered dimension: each thread loads two rows, 16 rows apart.
constexpr index_t kM2 = 32 / 8;
constexpr index_t kM1 = kBlockSize / 32;
constexpr index_t kM0 = kRows / (kM1 * kM2);
constexpr index_t kK1 = 8;
constexpr index_t kK2 = 16;
constexpr index_t kK0 = kRowBytes / (kK1 * kK2);

[[maybe_unused]] CK_TILE_DEVICE constexpr auto make_dstr()
{
    return make_static_tile_distribution(
        tile_distribution_encoding<sequence<1>,
                                   tuple<sequence<kM0, kM1, kM2>, sequence<kK0, kK1, kK2>>,
                                   tuple<sequence<1>, sequence<1, 2>>,
                                   tuple<sequence<1>, sequence<2, 1>>,
                                   sequence<1, 2, 2>,
                                   sequence<0, 0, 2>>{});
}

template <bool StaticMoveYs>
__global__ void __launch_bounds__(kBlockSize)
    gather_rows_to_lds(const uint8_t* src, const index_t* row_ids, uint8_t* dst)
{
#if defined(__gfx125__)
    __shared__ uint8_t smem[kRows * kRowBytes];

    constexpr auto dstr = make_dstr();
    const auto coord    = dstr.calculate_index();

    // One offset per value of Y0: the source row of this thread's tile row, in bytes.
    statically_indexed_array<index_t, kM0> offsets;
    static_for<0, kM0, 1>{}([&](auto m0) {
        offsets(m0) = row_ids[m0 * (kRows / kM0) + coord[number<0>{}]] * kRowBytes;
    });

    const auto src_view = make_naive_tensor_view<address_space_enum::global>(
        src, make_tuple(kSrcRows, kRowBytes), make_tuple(kRowBytes, 1), number<kK2>{}, number<1>{});
    const auto gather = make_tile_scatter_gather(
        src_view, make_tuple(number<kRows>{}, number<kRowBytes>{}), {0, 0}, dstr, offsets);

    const auto lds_view = make_naive_tensor_view<address_space_enum::lds>(
        smem,
        make_tuple(number<kRows>{}, number<kRowBytes>{}),
        make_tuple(number<kRowBytes>{}, number<1>{}),
        number<kK2>{},
        number<1>{});
    auto lds_window =
        make_tile_window(lds_view, make_tuple(number<kRows>{}, number<kRowBytes>{}), {0, 0});

    async_load_tile(lds_window, gather, number<-1>{}, true_type{}, bool_constant<StaticMoveYs>{});
    block_sync_lds_direct_load<0>();

    for(index_t i = threadIdx.x; i < kRows * kRowBytes; i += kBlockSize)
        dst[i] = smem[i];
#else
    ignore = src;
    ignore = row_ids;
    ignore = dst;
#endif
}

template <bool StaticMoveYs>
void run_gather(unsigned seed)
{
    if(!is_gfx125_supported())
        GTEST_SKIP() << "async global->LDS gather is tested on gfx1250 only";

    std::vector<uint8_t> src(kSrcRows * kRowBytes);
    for(index_t r = 0; r < kSrcRows; ++r)
        for(index_t c = 0; c < kRowBytes; ++c)
            src[r * kRowBytes + c] = static_cast<uint8_t>(r * 131 + c * 7 + 3);

    // Distinct source rows in random order, so a wrong row or a wrong LDS slot shows up.
    std::vector<index_t> row_ids(kSrcRows);
    std::iota(row_ids.begin(), row_ids.end(), 0);
    std::shuffle(row_ids.begin(), row_ids.end(), std::mt19937(seed));
    row_ids.resize(kRows);

    DeviceMem src_dev(src.size());
    DeviceMem ids_dev(row_ids.size() * sizeof(index_t));
    DeviceMem dst_dev(kRows * kRowBytes);
    src_dev.ToDevice(src.data());
    ids_dev.ToDevice(row_ids.data());
    dst_dev.SetZero();

    gather_rows_to_lds<StaticMoveYs>
        <<<1, kBlockSize>>>(static_cast<const uint8_t*>(src_dev.GetDeviceBuffer()),
                            static_cast<const index_t*>(ids_dev.GetDeviceBuffer()),
                            static_cast<uint8_t*>(dst_dev.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<uint8_t> dst(kRows * kRowBytes);
    dst_dev.FromDevice(dst.data());

    int wrong_rows = 0;
    for(index_t r = 0; r < kRows; ++r)
    {
        bool row_ok = true;
        for(index_t c = 0; c < kRowBytes; ++c)
            row_ok = row_ok && dst[r * kRowBytes + c] == src[row_ids[r] * kRowBytes + c];
        if(!row_ok)
        {
            ++wrong_rows;
            ADD_FAILURE() << "tile row " << r << " does not hold source row " << row_ids[r];
        }
    }
    EXPECT_EQ(wrong_rows, 0);
}

TEST(TileScatterGatherAsyncLoad, SteppedCoordinates)
{
    for(unsigned seed : {1u, 2u, 3u})
        run_gather<false>(seed);
}

TEST(TileScatterGatherAsyncLoad, StaticYOffsets)
{
    for(unsigned seed : {1u, 2u, 3u})
        run_gather<true>(seed);
}

} // namespace
