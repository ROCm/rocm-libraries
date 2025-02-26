/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef ROCWMMA_DEVICE_LAYOUT_TRAITS_INT_TEST_HPP
#define ROCWMMA_DEVICE_LAYOUT_TRAITS_INT_TEST_HPP

#include <rocwmma/rocwmma.hpp>

#include "unit_test_traits.hpp"
#include "common.hpp"

static constexpr uint32_t ERROR_VALUE   = 7;
static constexpr uint32_t SUCCESS_VALUE = 0;

namespace rocwmma
{
    template <typename LayoutLhs,
              typename LayoutRhs,
              bool ExpectSame,
              bool ExpectOrthogonal,
              bool DebugOnFail>
    ROCWMMA_HOST bool
        testLayoutPair(const char* file, const char* line, std::ostream& stream = std::cout)
    {
        constexpr bool is_layout_same_result = rocwmma::is_layout_same_v<LayoutLhs, LayoutRhs>;
        constexpr bool is_layout_orthogonal_result
            = rocwmma::is_layout_orthogonal_v<LayoutLhs, LayoutRhs>;
        constexpr bool compare_result = ((is_layout_same_result == ExpectSame)
                                         && (is_layout_orthogonal_result == ExpectOrthogonal));

        if constexpr(DebugOnFail)
        {
            stream << "File: " << file << " L:" << line << std::endl;
            stream << "<Test Begin>" << std::endl;
            stream << "Lhs: " << LayoutLhs{} << std::endl;
            stream << rocwmma::layout_traits<LayoutLhs>{};
            stream << "Rhs: " << LayoutRhs{} << std::endl;
            stream << rocwmma::layout_traits<LayoutRhs>{};
            stream << "is_layout_same: " << is_layout_same_result << " Expected: " << ExpectSame
                   << std::endl;
            stream << "is_layout_orthogonal: " << is_layout_orthogonal_result
                   << " Expected: " << ExpectOrthogonal << std::endl;
            stream << "Result:" << (compare_result ? "PASS" : "FAIL") << std::endl;
            stream << "<Test End>" << std::endl;
        }

        return compare_result;
    }

    template<typename LayoutLhs, typename LayoutRhs>
    ROCWMMA_DEVICE constexpr void debugRegisterFormats()
    {
        if(isFirstThread())
        {
            using traits_lhs = layout_traits<LayoutLhs>;
            using traits_rhs = layout_traits<LayoutRhs>;
            printf("testCompatibleRegisterParams: %d\n", LayoutTraits_impl::testCompatibleRegisterParams<LayoutLhs, LayoutRhs>());
            printf("MmaDim: %d, MmaDim: %d\n", traits_lhs::MmaDim, traits_rhs::MmaDim);
            printf("DataFormat: %d, DataFormat: %d\n", (int)traits_lhs::Format, (int)traits_rhs::Format);
            printf("is_valid: %d, is_valid: %d\n", traits_lhs::is_valid, traits_rhs::is_valid);
            printf("is_same_dataT: %d\n", is_same_v<typename traits_lhs::DataT, typename traits_rhs::DataT>);
        }
    }

    template <typename LayoutLhs,
              typename LayoutRhs,
              bool ExpectSame,
              bool ExpectOrthogonal,
              bool DebugOnFail>
    ROCWMMA_DEVICE bool testLayoutPair(const char* file, uint32_t line)
    {
        constexpr bool is_layout_same_result = rocwmma::is_layout_same_v<LayoutLhs, LayoutRhs>;
        constexpr bool is_layout_orthogonal_result
            = rocwmma::is_layout_orthogonal_v<LayoutLhs, LayoutRhs>;
        constexpr bool compare_result = ((is_layout_same_result == ExpectSame)
                                         && (is_layout_orthogonal_result == ExpectOrthogonal));

        if(!compare_result && DebugOnFail && isFirstThread())
        {
            printf("File: %s L:%d\n", file, line);
            printf("<TestBegin>\n");
            printf("is_layout_same: %d (Expected: %d)\n", is_layout_same_result, ExpectSame);
            printf("is_layout_orthogonal: %d (Expected: %d)\n",
                   is_layout_orthogonal_result,
                   ExpectOrthogonal);
            printf("%s\n", (compare_result ? "PASS" : "FAIL"));
            printf("<Test End>\n");
        }

        return compare_result;
    }

#define ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(                                                      \
    LayoutLhs, LayoutRhs, ExpectSame, ExpectOrthogonal, DebugOnFail)                          \
    testLayoutPair<LayoutLhs, LayoutRhs, ExpectSame, ExpectOrthogonal, DebugOnFail>(__FILE__, \
                                                                                    __LINE__);

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct RegisterLayoutIntTestingSet
    {
        using ColInline = RegisterLayout::Storage<
            MatrixLayout::ColInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>,
            DataLayoutT>;
        using ColOrtho = RegisterLayout::Storage<
            MatrixLayout::ColOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>,
            DataLayoutT>;
        using RowInline = RegisterLayout::Storage<
            MatrixLayout::RowInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>,
            DataLayoutT>;
        using RowOrtho = RegisterLayout::Storage<
            MatrixLayout::RowOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>,
            DataLayoutT>;

        using MmaInput = RegisterLayout::MmaInput<MmaDim, DataT, true, (bool)ROCWMMA_ARCH_GFX11
                                                       ? RegisterLayout::Format::WMMA_INPUT_GFX11
                                                       : RegisterLayout::Format::SOA_INT>;
        using MmaAcc   = RegisterLayout::MmaAcc<MmaDim, DataT, true, (bool)ROCWMMA_ARCH_GFX11
                                         ? RegisterLayout::Format::WMMA_ACC_GFX11
                                         : RegisterLayout::Format::ACC_INT_A_MAJOR>;
    };

    template <typename StorageLayout>
    using MatrixLayout_t = typename layout_traits<StorageLayout>::MatrixLayout;

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool matrixLayoutTraitsTestInterleaved0()
    {
        constexpr bool debug_on_fail = true;

        // Testing MatrixLayout properties
        // MatrixLayouts are invariant to vector width
        using Set
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>;

        bool result = true;

        // Matrix <-> Matrix layout
        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::ColOrtho>, MatrixLayout_t<typename Set::ColOrtho>, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::ColOrtho>, MatrixLayout_t<typename Set::ColInline>, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::ColOrtho>, MatrixLayout_t<typename Set::RowOrtho>, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::ColOrtho>, MatrixLayout_t<typename Set::RowInline>, false, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::ColInline>, MatrixLayout_t<typename Set::ColOrtho>,  false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::ColInline>, MatrixLayout_t<typename Set::ColInline>, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::ColInline>, MatrixLayout_t<typename Set::RowOrtho>, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::ColInline>, MatrixLayout_t<typename Set::RowInline>, false, true, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::RowOrtho>, MatrixLayout_t<typename Set::ColOrtho>, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::RowOrtho>, MatrixLayout_t<typename Set::ColInline>, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::RowOrtho>, MatrixLayout_t<typename Set::RowOrtho>, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::RowOrtho>, MatrixLayout_t<typename Set::RowInline>, false, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::RowInline>, MatrixLayout_t<typename Set::ColOrtho>, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::RowInline>, MatrixLayout_t<typename Set::ColInline>, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::RowInline>, MatrixLayout_t<typename Set::RowOrtho>, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(MatrixLayout_t<typename Set::RowInline>, MatrixLayout_t<typename Set::RowInline>, true, false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <typename DataLayoutT>
    ROCWMMA_DEVICE constexpr bool testRowMajor()
    {
        return is_layout_same_v<DataLayoutT, row_major>;
    }

    template <typename DataLayoutT>
    ROCWMMA_DEVICE constexpr bool testColMajor()
    {
        return is_layout_same_v<DataLayoutT, col_major>;
    }

    template <uint32_t MmaDim, typename DataT>
    ROCWMMA_DEVICE constexpr bool testMmaDim()
    {
        return (MmaDim == 16u && (bool)ROCWMMA_BLOCK_DIM_16_SUPPORTED)
               || (MmaDim == 32u && (bool)ROCWMMA_BLOCK_DIM_32_SUPPORTED
                   && !is_same_v<DataT, float64_t>);
    }

    template <uint32_t BlockDim, uint32_t MmaDim>
    ROCWMMA_DEVICE constexpr uint32_t dimPerThread()
    {
        return BlockDim / MmaDim;
    }

    template <uint32_t BlockK,
              uint32_t MmaDim,
              uint32_t SplitK,
              uint32_t WaveSize = Constants::AMDGCN_WAVE_SIZE>
    ROCWMMA_DEVICE constexpr uint32_t kPerThread()
    {
        return BlockK * MmaDim / (WaveSize * SplitK);
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestInterleaved0()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // BlockDim = same
        // BlockK = same
        // DataT = same
        // DataLayoutT = same
        // MmaDim = same
        // SplitK = same
        // Checks identity quirk condition
        using Set0
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>;
        using Set1
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>;

        constexpr bool     is_row_mjr = testRowMajor<DataLayoutT>();
        constexpr bool     is_col_mjr = testColMajor<DataLayoutT>();
        constexpr bool     is_mma_dim = testMmaDim<MmaDim, DataT>();
        constexpr uint32_t dpt        = dimPerThread<BlockDim, MmaDim>();
        constexpr uint32_t kpt        = kPerThread<BlockK, MmaDim, SplitK>();

        // VW tests for quirks
        constexpr bool is_dpt_eq_1 = (dpt == 1u);
        constexpr bool is_kpt_eq_1 = (kpt == 1u);

        // Identity quirk where interleaved register layouts match
        // regardless of their formats (e.g., AOS / SOA)
        constexpr bool is_id_quirk = is_dpt_eq_1 || is_kpt_eq_1;

        constexpr bool is_row_mjr_dpt_1 = is_row_mjr && is_dpt_eq_1;
        constexpr bool is_row_mjr_kpt_1 = is_row_mjr && is_kpt_eq_1;
        constexpr bool is_col_mjr_dpt_1 = is_col_mjr && is_dpt_eq_1;
        constexpr bool is_col_mjr_kpt_1 = is_col_mjr && is_kpt_eq_1;

        bool result = true;

        if constexpr(!is_id_quirk)
        {
            return result;
        }

        // Storage <-> storage layout
        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, (is_row_mjr_dpt_1 || is_col_mjr_kpt_1 ), false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, is_kpt_eq_1, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, (is_row_mjr || (is_dpt_eq_1 && is_kpt_eq_1)) , false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, (is_row_mjr_dpt_1 || is_col_mjr_kpt_1), false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, (is_col_mjr || (is_dpt_eq_1 && is_kpt_eq_1)), false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, is_dpt_eq_1, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, is_kpt_eq_1, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline,typename Set1::RowOrtho, (is_col_mjr || (is_dpt_eq_1 && is_kpt_eq_1)), false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, (is_col_mjr_dpt_1 || is_row_mjr_kpt_1), false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, (is_row_mjr || (is_dpt_eq_1 && is_kpt_eq_1)), false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, is_dpt_eq_1, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, (is_col_mjr_dpt_1 || is_row_mjr_kpt_1), false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, true, false, debug_on_fail);

        // Storage <-> mma layouts
        // gfx11 have unique mma formats that must always be transformed from storage
        if constexpr ((bool)ROCWMMA_ARCH_GFX11)
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false, ((is_row_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false, ((is_col_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false, ((is_col_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false, ((is_row_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false, ((is_row_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false, ((is_col_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false, ((is_col_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false, ((is_row_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);
        }
        // other targets have mma formats that may overlap storage formats
        else
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, ((is_row_mjr || is_kpt_eq_1) && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput,((is_col_mjr || is_dpt_eq_1) && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, ((is_col_mjr || is_kpt_eq_1) && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput,((is_row_mjr || is_dpt_eq_1) && is_mma_dim),  false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, ((is_row_mjr || is_kpt_eq_1) && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline,((is_col_mjr || is_dpt_eq_1) && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, ((is_col_mjr || is_kpt_eq_1) && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline,((is_row_mjr || is_dpt_eq_1) && is_mma_dim),  false, debug_on_fail);
        }

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, false, ((is_row_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,((is_col_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, false, ((is_col_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,((is_row_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, false, ((is_row_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,((is_col_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, false, ((is_col_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,((is_row_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestInterleaved1()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // BlockDim = same
        // BlockK = same
        // DataT = same
        // DataLayoutT = same
        // MmaDim = same
        // SplitK = same
        // Checks non-identity quirk condition
        using Set0
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>;
        using Set1
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>;

        constexpr bool     is_row_mjr = testRowMajor<DataLayoutT>();
        constexpr bool     is_col_mjr = testColMajor<DataLayoutT>();
        constexpr bool     is_mma_dim = testMmaDim<MmaDim, DataT>();
        constexpr uint32_t dpt        = dimPerThread<BlockDim, MmaDim>();
        constexpr uint32_t kpt        = kPerThread<BlockK, MmaDim, SplitK>();

        // Identity quirk where interleaved register layouts match
        // regardless of their formats (e.g., AOS / SOA)
        constexpr bool is_id_quirk = (dpt == 1u) || (kpt == 1u);

        constexpr bool is_mma_row_mjr = is_row_mjr && is_mma_dim;
        constexpr bool is_mma_col_mjr = is_col_mjr && is_mma_dim;

        bool result = true;

        if constexpr(is_id_quirk)
        {
            return result;
        }

        // Storage <-> storage layout
        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, is_row_mjr, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, is_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, false, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline,typename Set1::RowOrtho, false, is_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, is_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, true, false, debug_on_fail);

        // Storage <-> mma layouts
        // gfx11 have unique mma formats that must always be transformed from storage
        if constexpr ((bool)ROCWMMA_ARCH_GFX11)
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_row_mjr, debug_on_fail);
        }
        // other targets have mma formats that may overlap storage formats
        else
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, is_mma_row_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, is_mma_col_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, is_mma_row_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, is_mma_col_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_row_mjr, debug_on_fail);
        }

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, false,  is_mma_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  is_mma_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, false,  is_mma_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  is_mma_row_mjr, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, false,  is_mma_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  is_mma_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, false,  is_mma_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  is_mma_row_mjr, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestInterleaved2()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // BlockDim = same
        // BlockK = same
        // DataT = same
        // DataLayoutT = orthogonal
        // MmaDim = same
        // SplitK = same
        // Checks identity quirk condition
        using Set0
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>;
        using Set1 = RegisterLayoutIntTestingSet<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 orthogonal_layout_t<DataLayoutT>,
                                                 MmaDim,
                                                 SplitK>;

        constexpr bool     is_row_mjr = testRowMajor<DataLayoutT>();
        constexpr bool     is_col_mjr = testColMajor<DataLayoutT>();
        constexpr bool     is_mma_dim = testMmaDim<MmaDim, DataT>();
        constexpr uint32_t dpt        = dimPerThread<BlockDim, MmaDim>();
        constexpr uint32_t kpt        = kPerThread<BlockK, MmaDim, SplitK>();

        // VW tests for quirks
        constexpr bool is_dpt_eq_1 = (dpt == 1u);
        constexpr bool is_kpt_eq_1 = (kpt == 1u);

        // Identity quirk where interleaved register layouts match
        // regardless of their formats (e.g., AOS / SOA)
        constexpr bool is_id_quirk = is_dpt_eq_1 || is_kpt_eq_1;

        constexpr bool is_row_mjr_dpt_1 = is_row_mjr && is_dpt_eq_1;
        constexpr bool is_row_mjr_kpt_1 = is_row_mjr && is_kpt_eq_1;
        constexpr bool is_col_mjr_dpt_1 = is_col_mjr && is_dpt_eq_1;
        constexpr bool is_col_mjr_kpt_1 = is_col_mjr && is_kpt_eq_1;

        bool result = true;

        if constexpr(!is_id_quirk)
        {
            return result;
        }

        // Storage <-> storage layout
        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, is_kpt_eq_1, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, (is_col_mjr || (is_dpt_eq_1 && is_kpt_eq_1)), false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, true , false, debug_on_fail); // Can be invalid in same way
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, (is_col_mjr_dpt_1 || is_row_mjr_kpt_1), false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, (is_row_mjr || (is_dpt_eq_1 && is_kpt_eq_1)), false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, is_dpt_eq_1, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, (is_col_mjr_dpt_1 || is_row_mjr_kpt_1), false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, true, false, debug_on_fail); // Can be invalid in same way

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, true, false, debug_on_fail); // Can be invalid in same way
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline,typename Set1::RowOrtho, (is_row_mjr_dpt_1 || is_col_mjr_kpt_1), false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, is_kpt_eq_1, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, (is_row_mjr || (is_dpt_eq_1 && is_kpt_eq_1)), false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, (is_row_mjr_dpt_1 || is_col_mjr_kpt_1), false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, true, false, debug_on_fail); // Can be invalid in same way
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline,  (is_col_mjr || (is_dpt_eq_1 && is_kpt_eq_1)), false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, is_dpt_eq_1, false, debug_on_fail);

        // Storage <-> mma layouts
        // gfx11 have unique mma formats that must always be transformed from storage
        if constexpr ((bool)ROCWMMA_ARCH_GFX11)
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false, ((is_row_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false, ((is_col_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false, ((is_col_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false, ((is_row_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false, ((is_col_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false, ((is_row_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false, ((is_row_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false, ((is_col_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);
        }
        // other targets have mma formats that may overlap storage formats
        else
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  ((is_row_mjr || is_kpt_eq_1) && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, ((is_col_mjr || is_dpt_eq_1) && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  ((is_col_mjr || is_kpt_eq_1) && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, ((is_row_mjr || is_dpt_eq_1) && is_mma_dim),  false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  ((is_col_mjr || is_kpt_eq_1) && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, ((is_row_mjr || is_dpt_eq_1) && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  ((is_row_mjr || is_kpt_eq_1) && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, ((is_col_mjr || is_dpt_eq_1) && is_mma_dim),  false, debug_on_fail);
        }

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc,  false, ((is_row_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false, ((is_col_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc,  false, ((is_col_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false, ((is_row_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho,  false, ((is_col_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false, ((is_row_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho,  false, ((is_row_mjr || is_kpt_eq_1) && is_mma_dim), debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false, ((is_col_mjr || is_dpt_eq_1) && is_mma_dim), debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestInterleaved3()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // BlockDim = same
        // BlockK = same
        // DataT = same
        // DataLayoutT = orthogonal
        // MmaDim = same
        // SplitK = same
        // Checks non-identity quirk condition
        using Set0
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>;
        using Set1 = RegisterLayoutIntTestingSet<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 orthogonal_layout_t<DataLayoutT>,
                                                 MmaDim,
                                                 SplitK>;

        constexpr bool     is_row_mjr = testRowMajor<DataLayoutT>();
        constexpr bool     is_col_mjr = testColMajor<DataLayoutT>();
        constexpr bool     is_mma_dim = testMmaDim<MmaDim, DataT>();
        constexpr uint32_t dpt        = dimPerThread<BlockDim, MmaDim>();
        constexpr uint32_t kpt        = kPerThread<BlockK, MmaDim, SplitK>();

        // VW tests for quirks
        constexpr bool is_dpt_eq_1 = (dpt == 1u);
        constexpr bool is_kpt_eq_1 = (kpt == 1u);

        // Identity quirk where interleaved register layouts match
        // regardless of their formats (e.g., AOS / SOA)
        constexpr bool is_id_quirk = is_dpt_eq_1 || is_kpt_eq_1;

        constexpr bool is_mma_row_mjr = is_row_mjr && is_mma_dim;
        constexpr bool is_mma_col_mjr = is_col_mjr && is_mma_dim;

        bool result = true;

        if constexpr(is_id_quirk)
        {
            return result;
        }

        // Storage <-> storage layout
        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, is_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, true, false, debug_on_fail); // Can be invalid in same way
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, is_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, true, false, debug_on_fail); // Can be invalid in same way

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, true, false, debug_on_fail); // Can be invalid in same way
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline,typename Set1::RowOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, is_row_mjr, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, true, false, debug_on_fail); // Can be invalid in same way
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, is_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, false, false, debug_on_fail);

        // Storage <-> mma layouts
        // gfx11 have unique mma formats that must always be transformed from storage
        if constexpr ((bool)ROCWMMA_ARCH_GFX11)
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_col_mjr, debug_on_fail);
        }
        // other targets have mma formats that may overlap storage formats
        else
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, is_mma_row_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, is_mma_col_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, is_mma_col_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, is_mma_row_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_col_mjr, debug_on_fail);
        }

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, false,  is_mma_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  is_mma_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, false,  is_mma_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  is_mma_row_mjr, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, false,  is_mma_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  is_mma_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, false,  is_mma_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  is_mma_col_mjr, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestInterleaved4()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // BlockDim = different
        // BlockK = different
        // DataT = same
        // DataLayoutT = same
        // MmaDim = same
        // SplitK = same
        constexpr uint32_t BlockDim0 = BlockDim;
        constexpr uint32_t BlockDim1 = BlockDim == 32u ? 64u : 32u;
        constexpr uint32_t BlockK0   = BlockK;
        constexpr uint32_t BlockK1   = BlockK == 32u ? 64u : 32u;
        using Set0
            = RegisterLayoutIntTestingSet<BlockDim0, BlockK0, DataT, DataLayoutT, MmaDim, SplitK>;
        using Set1
            = RegisterLayoutIntTestingSet<BlockDim1, BlockK1, DataT, DataLayoutT, MmaDim, SplitK>;

        constexpr bool     is_row_mjr = testRowMajor<DataLayoutT>();
        constexpr bool     is_col_mjr = testColMajor<DataLayoutT>();
        constexpr bool     is_mma_dim = testMmaDim<MmaDim, DataT>();
        constexpr uint32_t dpt0       = dimPerThread<BlockDim0, MmaDim>();
        constexpr uint32_t kpt0       = kPerThread<BlockK0, MmaDim, SplitK>();
        constexpr uint32_t dpt1       = dimPerThread<BlockDim1, MmaDim>();
        constexpr uint32_t kpt1       = kPerThread<BlockK1, MmaDim, SplitK>();

        // VW tests for quirks
        constexpr bool is_dpt0_eq_1 = (dpt0 == 1u);
        constexpr bool is_kpt0_eq_1 = (kpt0 == 1u);
        constexpr bool is_dpt1_eq_1 = (dpt1 == 1u);
        constexpr bool is_kpt1_eq_1 = (kpt1 == 1u);

        // Identity quirk where interleaved register layouts match
        // regardless of their formats (e.g., AOS / SOA)
        constexpr bool is_id_quirk0 = (is_dpt0_eq_1 || is_kpt0_eq_1);
        constexpr bool is_id_quirk1 = (is_dpt1_eq_1 || is_kpt1_eq_1);

        bool result = true;

        // Current test case deals with quirks validation
        if constexpr((is_id_quirk0 != is_id_quirk1) || !is_id_quirk0)
        {
            return result;
        }

        // Ensure MmaDim layout constraints are met
        if constexpr(BlockDim0 >= MmaDim && BlockDim1 >= MmaDim)
        {
            // Storage <-> storage layout
            // clang-format off
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline,typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, false, false, debug_on_fail);

            // Storage <-> mma layouts
            // Same MmaDim
            // gfx11 have unique mma formats that must always be transformed from storage
            if constexpr ((bool)ROCWMMA_ARCH_GFX11)
            {
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false, ((is_row_mjr || is_kpt0_eq_1) && is_mma_dim), debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false, ((is_col_mjr || is_dpt0_eq_1) && is_mma_dim), debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false, ((is_col_mjr || is_kpt0_eq_1) && is_mma_dim), debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false, ((is_row_mjr || is_dpt0_eq_1) && is_mma_dim), debug_on_fail);

                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false, ((is_row_mjr || is_kpt1_eq_1) && is_mma_dim), debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false, ((is_col_mjr || is_dpt1_eq_1) && is_mma_dim), debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false, ((is_col_mjr || is_kpt1_eq_1) && is_mma_dim), debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false, ((is_row_mjr || is_dpt1_eq_1) && is_mma_dim), debug_on_fail);
            }
            // other targets have mma formats that may overlap storage formats
            else
            {
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, ((is_row_mjr || is_kpt0_eq_1) && is_mma_dim),  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput,((is_col_mjr || is_dpt0_eq_1) && is_mma_dim),  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, ((is_col_mjr || is_kpt0_eq_1) && is_mma_dim),  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput,((is_row_mjr || is_dpt0_eq_1) && is_mma_dim),  false, debug_on_fail);

                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, ((is_row_mjr || is_kpt1_eq_1) && is_mma_dim),  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline,((is_col_mjr || is_dpt1_eq_1) && is_mma_dim),  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, ((is_col_mjr || is_kpt1_eq_1) && is_mma_dim),  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline,((is_row_mjr || is_dpt1_eq_1) && is_mma_dim),  false, debug_on_fail);
            }

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, false, ((is_row_mjr || is_kpt0_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,((is_col_mjr || is_dpt0_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, false, ((is_col_mjr || is_kpt0_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,((is_row_mjr || is_dpt0_eq_1) && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, false, ((is_row_mjr || is_kpt1_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,((is_col_mjr || is_dpt1_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, false, ((is_col_mjr || is_kpt1_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,((is_row_mjr || is_dpt1_eq_1) && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
            // clang-format on
        }

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestInterleaved5()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // BlockDim = different
        // BlockK = different
        // DataT = same
        // DataLayoutT = same
        // MmaDim = same
        // SplitK = same
        constexpr uint32_t BlockDim0 = BlockDim;
        constexpr uint32_t BlockDim1 = BlockDim == 32u ? 64u : 32u;
        constexpr uint32_t BlockK0   = BlockK;
        constexpr uint32_t BlockK1   = BlockK == 32u ? 64u : 32u;
        using Set0
            = RegisterLayoutIntTestingSet<BlockDim0, BlockK0, DataT, DataLayoutT, MmaDim, SplitK>;
        using Set1
            = RegisterLayoutIntTestingSet<BlockDim1, BlockK1, DataT, DataLayoutT, MmaDim, SplitK>;

        constexpr bool     is_row_mjr = testRowMajor<DataLayoutT>();
        constexpr bool     is_col_mjr = testColMajor<DataLayoutT>();
        constexpr bool     is_mma_dim = testMmaDim<MmaDim, DataT>();
        constexpr uint32_t dpt0       = dimPerThread<BlockDim0, MmaDim>();
        constexpr uint32_t kpt0       = kPerThread<BlockK0, MmaDim, SplitK>();
        constexpr uint32_t dpt1       = dimPerThread<BlockDim1, MmaDim>();
        constexpr uint32_t kpt1       = kPerThread<BlockK1, MmaDim, SplitK>();

        // VW tests for quirks
        constexpr bool is_dpt0_eq_1 = (dpt0 == 1u);
        constexpr bool is_kpt0_eq_1 = (kpt0 == 1u);
        constexpr bool is_dpt1_eq_1 = (dpt1 == 1u);
        constexpr bool is_kpt1_eq_1 = (kpt1 == 1u);

        // Identity quirk where interleaved register layouts match
        // regardless of their formats (e.g., AOS / SOA)
        constexpr bool is_id_quirk0 = (is_dpt0_eq_1 || is_kpt0_eq_1);
        constexpr bool is_id_quirk1 = (is_dpt1_eq_1 || is_kpt1_eq_1);

        constexpr bool is_mma_row_mjr = is_row_mjr && is_mma_dim;
        constexpr bool is_mma_col_mjr = is_col_mjr && is_mma_dim;

        bool result = true;

        // Other test case deals with quirks validation
        if constexpr((is_id_quirk0 != is_id_quirk1) || is_id_quirk0)
        {
            return result;
        }

        // Ensure MmaDim layout constraints are met
        if constexpr(BlockDim0 >= MmaDim && BlockDim1 >= MmaDim)
        {
            // Storage <-> storage layout
            // clang-format off
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline,typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, false, false, debug_on_fail);

            // Storage <-> mma layouts
            // Same MmaDim
            // gfx11 have unique mma formats that must always be transformed from storage
            if constexpr ((bool)ROCWMMA_ARCH_GFX11)
            {
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false,  is_mma_row_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_col_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false,  is_mma_col_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_row_mjr, debug_on_fail);

                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false,  is_mma_row_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_col_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false,  is_mma_col_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_row_mjr, debug_on_fail);
            }
            // other targets have mma formats that may overlap storage formats
            else
            {
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, is_mma_row_mjr,  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_col_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, is_mma_col_mjr,  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_row_mjr, debug_on_fail);

                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, is_mma_row_mjr,  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_col_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, is_mma_col_mjr,  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_row_mjr, debug_on_fail);
            }

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  is_mma_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  is_mma_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
            // clang-format on
        }

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestInterleaved6()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // BlockDim = same
        // BlockK = same
        // DataT = different, same size
        // DataLayoutT = same
        // MmaDim = same
        // SplitK = same
        using DataT0 = DataT;
        using DataT1 = conditional_t<
            sizeof(DataT) == 1u,
            int8_t,
            conditional_t<sizeof(DataT) == 2u,
                          int16_t,
                          conditional_t<sizeof(DataT) == 4u,
                                        int32_t,
                                        conditional_t<sizeof(DataT) == 8u, int64_t, DataT>>>>;
        using Set0
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT0, DataLayoutT, MmaDim, SplitK>;
        using Set1
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT1, DataLayoutT, MmaDim, SplitK>;

        bool result = true;

        // Already checked same types
        if constexpr(is_same_v<DataT0, DataT1>)
        {
            return result;
        }

        // Storage <-> storage layout
        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, false, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline,typename Set1::RowOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, false, false, debug_on_fail);

        // Storage <-> mma layouts
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestInterleaved7()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // BlockDim = same
        // BlockK = same
        // DataT = different, different size
        // DataLayoutT = same
        // MmaDim = same
        // SplitK = same
        using DataT0 = DataT;
        using DataT1 = conditional_t<
            sizeof(DataT) == 1u,
            int16_t,
            conditional_t<sizeof(DataT) == 2u,
                          int32_t,
                          conditional_t<sizeof(DataT) == 4u,
                                        int64_t,
                                        conditional_t<sizeof(DataT) == 8u, int32_t, DataT>>>>;
        using Set0
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT0, DataLayoutT, MmaDim, SplitK>;
        using Set1
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT1, DataLayoutT, MmaDim, SplitK>;

        bool result = true;

        // Already checked same types
        if constexpr(is_same_v<DataT0, DataT1>)
        {
            return result;
        }

        // Storage <-> storage layout
        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, false, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline,typename Set1::RowOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, false, false, debug_on_fail);

        // Storage <-> mma layouts
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestInterleaved8()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // BlockDim = same
        // BlockK = same
        // DataT = same
        // DataLayoutT = same
        // MmaDim = different, valid
        // SplitK = same
        constexpr uint32_t MmaDim0 = MmaDim;
        constexpr uint32_t MmaDim1 = MmaDim == 16 ? 32u : 16u;

        constexpr uint32_t dpt0 = dimPerThread<BlockDim, MmaDim0>();
        constexpr uint32_t kpt0 = kPerThread<BlockK, MmaDim0, SplitK>();
        constexpr uint32_t dpt1 = dimPerThread<BlockDim, MmaDim1>();
        constexpr uint32_t kpt1 = kPerThread<BlockK, MmaDim1, SplitK>();

        bool result = true;

        // Ensure MmaDim layout constraints are met
        if constexpr(dpt0 > 0u && kpt0 > 0u && dpt1 > 0u && kpt1 > 0u)
        {
            using Set0 = RegisterLayoutIntTestingSet<BlockDim,
                                                     BlockK,
                                                     DataT,
                                                     DataLayoutT,
                                                     MmaDim0,
                                                     SplitK>;
            using Set1 = RegisterLayoutIntTestingSet<BlockDim,
                                                     BlockK,
                                                     DataT,
                                                     DataLayoutT,
                                                     MmaDim1,
                                                     SplitK>;

            // Storage <-> storage layout
            // clang-format off
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline,typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, false, false, debug_on_fail);

            // Storage <-> mma layouts
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
            // clang-format on
        }

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestInterleaved9()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // BlockDim = same
        // BlockK = same
        // DataT = same
        // DataLayoutT = same
        // MmaDim = same
        // SplitK = different
        constexpr uint32_t SplitK0 = SplitK;
        constexpr uint32_t SplitK1 = SplitK == 4u ? 2u : 4u;

        using Set0
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK0>;
        using Set1
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK1>;

        constexpr bool     is_row_mjr = testRowMajor<DataLayoutT>();
        constexpr bool     is_col_mjr = testColMajor<DataLayoutT>();
        constexpr bool     is_mma_dim = testMmaDim<MmaDim, DataT>();
        constexpr uint32_t dpt0       = dimPerThread<BlockDim, MmaDim>();
        constexpr uint32_t kpt0       = kPerThread<BlockK, MmaDim, SplitK0>();
        constexpr uint32_t dpt1       = dimPerThread<BlockDim, MmaDim>();
        constexpr uint32_t kpt1       = kPerThread<BlockK, MmaDim, SplitK1>();

        // VW tests for quirks
        constexpr bool is_dpt0_eq_1 = (dpt0 == 1u);
        constexpr bool is_kpt0_eq_1 = (kpt0 == 1u);
        constexpr bool is_dpt1_eq_1 = (dpt1 == 1u);
        constexpr bool is_kpt1_eq_1 = (kpt1 == 1u);

        // Identity quirk where interleaved register layouts match
        // regardless of their formats (e.g., AOS / SOA)
        constexpr bool is_id_quirk0 = (is_dpt0_eq_1 || is_kpt0_eq_1);
        constexpr bool is_id_quirk1 = (is_dpt1_eq_1 || is_kpt1_eq_1);

        constexpr bool is_mma_row_mjr = is_row_mjr && is_mma_dim;
        constexpr bool is_mma_col_mjr = is_col_mjr && is_mma_dim;

        bool result = true;

        // Current test deals with quirk case validation
        if constexpr((is_id_quirk0 != is_id_quirk1) || !is_id_quirk0)
        {
            return result;
        }

        // Ensure layout requirements are satisfied
        if constexpr(dpt0 > 0u && kpt0 > 0u && dpt1 > 0u && kpt1 > 0u)
        {
            // Storage <-> storage layout
            // clang-format off
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline,typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, false, false, debug_on_fail);

            // Storage <-> mma layouts
            // gfx11 have unique mma formats that must always be transformed from storage
            if constexpr ((bool)ROCWMMA_ARCH_GFX11)
            {
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false, ((is_row_mjr || is_kpt0_eq_1) && is_mma_dim), debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false, ((is_col_mjr || is_dpt0_eq_1) && is_mma_dim), debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false, ((is_col_mjr || is_kpt0_eq_1) && is_mma_dim), debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false, ((is_row_mjr || is_dpt0_eq_1) && is_mma_dim), debug_on_fail);

                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false, ((is_row_mjr || is_kpt1_eq_1) && is_mma_dim), debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false, ((is_col_mjr || is_dpt1_eq_1) && is_mma_dim), debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false, ((is_col_mjr || is_kpt1_eq_1) && is_mma_dim), debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false, ((is_row_mjr || is_dpt1_eq_1) && is_mma_dim), debug_on_fail);
            }
            // other targets have mma formats that may overlap storage formats
            else
            {
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, ((is_row_mjr || is_kpt0_eq_1) && is_mma_dim),  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput,((is_col_mjr || is_dpt0_eq_1) && is_mma_dim),  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, ((is_col_mjr || is_kpt0_eq_1) && is_mma_dim),  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput,((is_row_mjr || is_dpt0_eq_1) && is_mma_dim),  false, debug_on_fail);

                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, ((is_row_mjr || is_kpt1_eq_1) && is_mma_dim),  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline,((is_col_mjr || is_dpt1_eq_1) && is_mma_dim),  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, ((is_col_mjr || is_kpt1_eq_1) && is_mma_dim),  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline,((is_row_mjr || is_dpt1_eq_1) && is_mma_dim),  false, debug_on_fail);
            }

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, false, ((is_row_mjr || is_kpt0_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,((is_col_mjr || is_dpt0_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, false, ((is_col_mjr || is_kpt0_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,((is_row_mjr || is_dpt0_eq_1) && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, false, ((is_row_mjr || is_kpt1_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,((is_col_mjr || is_dpt1_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, false, ((is_col_mjr || is_kpt1_eq_1) && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,((is_row_mjr || is_dpt1_eq_1) && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
            // clang-format on
        }

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestInterleaved10()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // BlockDim = same
        // BlockK = same
        // DataT = same
        // DataLayoutT = same
        // MmaDim = same
        // SplitK = different
        constexpr uint32_t SplitK0 = SplitK;
        constexpr uint32_t SplitK1 = SplitK == 4u ? 2u : 4u;

        using Set0
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK0>;
        using Set1
            = RegisterLayoutIntTestingSet<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK1>;

        constexpr bool     is_row_mjr = testRowMajor<DataLayoutT>();
        constexpr bool     is_col_mjr = testColMajor<DataLayoutT>();
        constexpr bool     is_mma_dim = testMmaDim<MmaDim, DataT>();
        constexpr uint32_t dpt0       = dimPerThread<BlockDim, MmaDim>();
        constexpr uint32_t kpt0       = kPerThread<BlockK, MmaDim, SplitK0>();
        constexpr uint32_t dpt1       = dimPerThread<BlockDim, MmaDim>();
        constexpr uint32_t kpt1       = kPerThread<BlockK, MmaDim, SplitK1>();

        // VW tests for quirks
        constexpr bool is_dpt0_eq_1 = (dpt0 == 1u);
        constexpr bool is_kpt0_eq_1 = (kpt0 == 1u);
        constexpr bool is_dpt1_eq_1 = (dpt1 == 1u);
        constexpr bool is_kpt1_eq_1 = (kpt1 == 1u);

        // Identity quirk where interleaved register layouts match
        // regardless of their formats (e.g., AOS / SOA)
        constexpr bool is_id_quirk0 = (is_dpt0_eq_1 || is_kpt0_eq_1);
        constexpr bool is_id_quirk1 = (is_dpt1_eq_1 || is_kpt1_eq_1);

        constexpr bool is_mma_row_mjr = is_row_mjr && is_mma_dim;
        constexpr bool is_mma_col_mjr = is_col_mjr && is_mma_dim;

        bool result = true;

        // Other test handles quirk case validation
        if constexpr((is_id_quirk0 != is_id_quirk1) || is_id_quirk0)
        {
            return result;
        }

        // Ensure layout requirements are satisfied
        if constexpr(dpt0 > 0u && kpt0 > 0u && dpt1 > 0u && kpt1 > 0u)
        {
            // Storage <-> storage layout
            // clang-format off
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline,typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, false, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, false, false, debug_on_fail);

            // Storage <-> mma layouts
            // gfx11 have unique mma formats that must always be transformed from storage
            if constexpr ((bool)ROCWMMA_ARCH_GFX11)
            {
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false,  is_mma_row_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_col_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false,  is_mma_col_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_row_mjr, debug_on_fail);

                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false,  is_mma_row_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_col_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false,  is_mma_col_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_row_mjr, debug_on_fail);
            }
            // other targets have mma formats that may overlap storage formats
            else
            {
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, is_mma_row_mjr,  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_col_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, is_mma_col_mjr,  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_row_mjr, debug_on_fail);

                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, is_mma_row_mjr,  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_col_mjr, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, is_mma_col_mjr,  false, debug_on_fail);
                result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_row_mjr, debug_on_fail);
            }


            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  is_mma_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  is_mma_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
            // clang-format on
        }

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool testBarrageInterleaved()
    {
        bool result = true;

        constexpr uint32_t dpt = dimPerThread<BlockDim, MmaDim>();
        constexpr uint32_t kpt = kPerThread<BlockK, MmaDim, SplitK>();

        // Must satisfy layout requirement
        if constexpr(dpt > 0u && kpt > 0u && BlockK >= kpt)
        {
            // clang-format off
            result &= matrixLayoutTraitsTestInterleaved0<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();
            result &= registerLayoutTraitsTestInterleaved0<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();
            result &= registerLayoutTraitsTestInterleaved1<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();
            result &= registerLayoutTraitsTestInterleaved2<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();
            result &= registerLayoutTraitsTestInterleaved3<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();
            result &= registerLayoutTraitsTestInterleaved4<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();
            result &= registerLayoutTraitsTestInterleaved5<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();
            result &= registerLayoutTraitsTestInterleaved6<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();
            result &= registerLayoutTraitsTestInterleaved7<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();
            result &= registerLayoutTraitsTestInterleaved8<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();
            result &= registerLayoutTraitsTestInterleaved9<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();
            result &= registerLayoutTraitsTestInterleaved10<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();
            // clang-format on
        }

        return result;
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool layoutTraitsTestA()
    {
        constexpr uint32_t BlockDim = BlockM;
        constexpr uint32_t BlockK   = BlockN;

        bool result = true;
        result &= testBarrageInterleaved<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();

        return result;
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool layoutTraitsTestB()
    {
        constexpr uint32_t BlockDim = BlockN;
        constexpr uint32_t BlockK   = BlockM;

        bool result = true;
        result &= testBarrageInterleaved<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();

        return result;
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    ROCWMMA_DEVICE bool layoutTraitsTestAcc()
    {
        // TODO: WaveCount
        constexpr uint32_t BlockDim = BlockN;
        constexpr uint32_t BlockK   = BlockM;

        bool result = true;
        result &= testBarrageInterleaved<BlockDim, BlockK, DataT, DataLayoutT, MmaDim, SplitK>();

        return result;
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    __global__ void layoutTraitsIntTest(uint32_t     m,
                                        uint32_t     n,
                                        DataT const* in,
                                        DataT*       out,
                                        uint32_t     ld,
                                        DataT        param1,
                                        DataT        param2)
    {
        __shared__ int32_t result;
        result = 0;
        synchronize_workgroup();

        bool success = true;

        success &= layoutTraitsTestA<BlockM, BlockN, DataT, DataLayoutT, MmaDim, SplitK>();
        success &= layoutTraitsTestB<BlockM, BlockN, DataT, DataLayoutT, MmaDim, SplitK>();
        success &= layoutTraitsTestAcc<BlockM, BlockN, DataT, DataLayoutT, MmaDim, SplitK>();

        // Reduce error count
        atomicAdd(&result, (int32_t)success);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? 7 : 0);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_LAYOUT_TRAITS_TEST_HPP
