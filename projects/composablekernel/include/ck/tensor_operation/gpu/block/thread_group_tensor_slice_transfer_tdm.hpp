// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

namespace ck {

struct TdmLdsTileLayout
{
    bool valid;
    index_t rows_per_pad_group;
    index_t pad_elems;
};

// Evaluate each row separately so transformed descriptors do not exhaust the compiler's
// constexpr-step limit when checking every element of a large tile.
template <typename BlockDesc, index_t K0, index_t K1, index_t Row>
__host__ __device__ constexpr bool IsTdmLdsRowContiguous()
{
    constexpr auto desc       = BlockDesc{};
    constexpr auto row_offset = desc.CalculateOffset(make_multi_index(0, Row, 0));
    for(index_t k = 0; k < K0 * K1; ++k)
    {
        if(desc.CalculateOffset(make_multi_index(k / K1, Row, k % K1)) != row_offset + k)
            return false;
    }
    return true;
}

template <typename BlockDesc, index_t K0, index_t K1, index_t Row>
struct TdmLdsRow
{
    static constexpr index_t offset  = BlockDesc{}.CalculateOffset(make_multi_index(0, Row, 0));
    static constexpr bool contiguous = IsTdmLdsRowContiguous<BlockDesc, K0, K1, Row>();
};

// TDM can produce contiguous K rows with a regular padding gap between groups of rows.
// Verify every (K, row) offset, including transformed LDS layouts, before selecting TDM.
template <typename BlockDesc, index_t K0, index_t Rows, index_t K1>
__host__ __device__ constexpr TdmLdsTileLayout DeduceTdmLdsTileLayout()
{
    constexpr index_t KPerBlock = K0 * K1;
    auto offset = [](index_t r) { return BlockDesc{}.CalculateOffset(make_multi_index(0, r, 0)); };

    if(offset(0) != 0)
        return {false, 0, 0};

    index_t group = 1;
    while(group < Rows && offset(group) == group * KPerBlock)
        ++group;
    const index_t pad = group < Rows ? offset(group) - group * KPerBlock : 0;
    if(pad < 0)
        return {false, 0, 0};

    bool valid = true;
    static_for<0, Rows, 1>{}([&](auto row) {
        using Row = TdmLdsRow<BlockDesc, K0, K1, row.value>;
        valid     = valid && Row::contiguous &&
                Row::offset == row.value * KPerBlock + (row.value / group) * pad;
    });
    return {valid, group, pad};
}

// Ineligible padding falls back to DirectLoad; do not instantiate an unencodable descriptor.
template <typename DataType, index_t Rows, index_t KPerBlock>
__host__ __device__ constexpr bool IsTdmLdsTileLayoutSupported(TdmLdsTileLayout layout)
{
    constexpr index_t bytes = sizeof(DataType);
    if(!layout.valid || (bytes != 2 && bytes != 4) || Rows <= 0 || Rows >= 65536 ||
       KPerBlock <= 0 || KPerBlock >= 65536 || layout.rows_per_pad_group <= 0)
        return false;
    if(layout.pad_elems == 0)
        return true;
    const index_t interval = layout.rows_per_pad_group * KPerBlock * bytes / 4;
    const index_t amount   = layout.pad_elems * bytes / 4;
    return Rows % layout.rows_per_pad_group == 0 && KPerBlock * bytes % 4 == 0 &&
           layout.pad_elems * bytes % 4 == 0 && amount >= 1 && amount <= 128 && interval >= 2 &&
           interval <= 256 && (interval & (interval - 1)) == 0;
}

// B-only packed [N, K] global->LDS transfer. Wave zero issues one tensor instruction;
// all other load waves retain their existing A async loads. Remaining tensor extents
// supply positive-end OOB zero filling for both N padding and the final K tile.
template <typename ThreadGroup,
          typename DataType,
          index_t Rows,
          index_t K0,
          index_t K1,
          index_t RowsPerPadGroup,
          index_t PadElems,
          index_t NumIssueWaves>
struct ThreadGroupTensorSliceTransfer_Tdm2D
{
    static constexpr bool kUsesTensorDataMover = true;

    static constexpr index_t KPerBlock   = K0 * K1;
    static constexpr index_t RowsPerWave = Rows / NumIssueWaves;
    static constexpr index_t kDataBytes  = sizeof(DataType);

    static_assert(kDataBytes == 1 || kDataBytes == 2 || kDataBytes == 4 || kDataBytes == 8);
    static_assert(Rows % NumIssueWaves == 0);
    // every issuing wave must start at a padding-group boundary (TDM pad counter restarts at 0)
    static_assert(PadElems == 0 || RowsPerWave % RowsPerPadGroup == 0);
    static_assert(KPerBlock * kDataBytes % 4 == 0, "TDM padding needs a DWORD-multiple row");
    static_assert(Rows < 65536 && KPerBlock < 65536);

    static constexpr uint32_t kDataSizeCode =
        kDataBytes == 8 ? 3u : (kDataBytes == 4 ? 2u : (kDataBytes == 2 ? 1u : 0u));

    static constexpr index_t kPadIntervalDwords = RowsPerPadGroup * KPerBlock * kDataBytes / 4;
    static constexpr index_t kPadAmountDwords   = PadElems * kDataBytes / 4;

    static constexpr uint32_t log2_u32(index_t v)
    {
        uint32_t r = 0;
        while((index_t{1} << (r + 1)) <= v)
            ++r;
        return r;
    }

    static_assert(PadElems == 0 || (PadElems * kDataBytes % 4 == 0 && kPadAmountDwords >= 1 &&
                                    kPadAmountDwords <= 128 && kPadIntervalDwords >= 2 &&
                                    kPadIntervalDwords <= 256 &&
                                    (kPadIntervalDwords & (kPadIntervalDwords - 1)) == 0),
                  "LDS padding not expressible with D#.pad_interval/pad_amount");

    // D# group 1 word 0: data_size[17:16], pad_enable[20], pad_interval[24:22], pad_amount[31:25]
    static constexpr uint32_t kGroup1Word0 =
        (kDataSizeCode << 16) |
        (PadElems == 0 ? 0u
                       : ((1u << 20) | ((log2_u32(kPadIntervalDwords) - 1u) << 22) |
                          (static_cast<uint32_t>(kPadAmountDwords - 1) << 25)));

    // Byte offset of the first LDS element written by issuing wave `w`
    static constexpr index_t kLdsBytesPerWave =
        (RowsPerWave * KPerBlock + (RowsPerWave / RowsPerPadGroup) * PadElems) * kDataBytes;

    __device__ ThreadGroupTensorSliceTransfer_Tdm2D(const DataType* p_grid,
                                                    index_t rows_raw,
                                                    index_t k_raw,
                                                    index_t row_stride,
                                                    index_t row_origin,
                                                    index_t k_origin)
        : p_grid_{p_grid},
          rows_raw_{rows_raw},
          k_raw_{k_raw},
          row_stride_{row_stride},
          row_origin_{row_origin},
          k_origin_{k_origin}
    {
    }

    template <typename SrcDesc>
    __device__ void PrecomputeIdx(const SrcDesc&)
    {
    }

    template <typename SrcDesc, typename Index>
    __device__ void MoveSrcSliceWindow(const SrcDesc&, const Index& step)
    {
        k_origin_ += step[Number<0>{}] * K1;
    }

    template <typename SrcBuffer, typename DstDesc, typename DstBuffer>
    __device__ void Load(const SrcBuffer&, const DstDesc&, DstBuffer& dst_buf)
    {
#if defined(__gfx125__)
        const index_t wave =
            __builtin_amdgcn_readfirstlane(ThreadGroup::GetThreadId() / get_warp_size());
        if(wave < NumIssueWaves)
        {
            const index_t row0      = row_origin_ + wave * RowsPerWave;
            const index_t rows_left = math::max(rows_raw_ - row0, index_t{0});
            const index_t k_left    = math::max(k_raw_ - k_origin_, index_t{0});

            const uint64_t gaddr = reinterpret_cast<uint64_t>(
                p_grid_ + static_cast<long_index_t>(row0) * row_stride_ + k_origin_);
            const uint32_t lds_addr =
                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst_buf.p_data_)) +
                static_cast<uint32_t>(wave * kLdsBytesPerWave);
            const uint64_t stride0 = static_cast<uint64_t>(row_stride_);

            uint32x4_t g0;
            g0[0] = 1u; // count = 1: valid tensor
            g0[1] = __builtin_amdgcn_readfirstlane(lds_addr);
            g0[2] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(gaddr));
            g0[3] =
                __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(gaddr >> 32) & 0x1FFFFFFu) |
                (2u << 30); // type = 2

            const uint32_t td0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(k_left));
            const uint32_t td1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(rows_left));
            const uint32_t s0  = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(stride0));

            int32x8_t g1;
            g1[0] = static_cast<int32_t>(kGroup1Word0);
            g1[1] = static_cast<int32_t>((td0 & 0xFFFFu) << 16);
            g1[2] = static_cast<int32_t>((td0 >> 16) | ((td1 & 0xFFFFu) << 16));
            g1[3] = static_cast<int32_t>((td1 >> 16) | (static_cast<uint32_t>(KPerBlock) << 16));
            g1[4] = static_cast<int32_t>(RowsPerWave); // tile_dim1, tile_dim2 = 0 (unused)
            g1[5] = static_cast<int32_t>(s0);          // tensor_dim0_stride[31:0]
            g1[6] = static_cast<int32_t>(static_cast<uint32_t>(stride0 >> 32) & 0xFFFFu);
            g1[7] = 0;

            const int32x4_t z4 = {0, 0, 0, 0};
            const int32x8_t z8 = {0, 0, 0, 0, 0, 0, 0, 0};
            __builtin_amdgcn_tensor_load_to_lds(g0, g1, z4, z4, z8, 0);
        }
#else
        ignore = dst_buf;
#endif
    }

    private:
    const DataType* p_grid_;
    index_t rows_raw_;
    index_t k_raw_;
    index_t row_stride_;
    index_t row_origin_;
    index_t k_origin_;
};

// Tensor completion is unordered with ASYNCcnt/DScnt. Drain the issuing wave before
// the existing workgroup barrier publishes its LDS writes to math waves.
template <typename Transfer>
__device__ inline void wait_tdm_transfer()
{
#if defined(__gfx125__)
    if constexpr(requires { Transfer::kUsesTensorDataMover; })
        __builtin_amdgcn_s_wait_tensorcnt(0);
#endif
}

} // namespace ck
