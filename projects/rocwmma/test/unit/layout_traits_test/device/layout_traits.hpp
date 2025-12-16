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

#ifndef ROCWMMA_DEVICE_LAYOUT_TRAITS_TEST_HPP
#define ROCWMMA_DEVICE_LAYOUT_TRAITS_TEST_HPP

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
        constexpr bool is_layout_same_result = is_layout_same_v<LayoutLhs, LayoutRhs>;
        constexpr bool is_layout_orthogonal_result = is_layout_orthogonal_v<LayoutLhs, LayoutRhs>;
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

    template <typename LayoutLhs,
              typename LayoutRhs,
              bool ExpectSame,
              bool ExpectOrthogonal,
              bool DebugOnFail>
    ROCWMMA_DEVICE bool testLayoutPair(const char* file, uint32_t line)
    {
        constexpr bool is_layout_same_result = is_layout_same_v<LayoutLhs, LayoutRhs>;
        constexpr bool is_layout_orthogonal_result = is_layout_orthogonal_v<LayoutLhs, LayoutRhs>;
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

    template <typename DataLayoutT>
    ROCWMMA_DEVICE bool dataLayoutTraitsTest()
    {
        constexpr bool debug_on_fail = true;

        using rocwmma::DataLayout::ColMajor;
        using rocwmma::DataLayout::RowMajor;

        // DataLayouts are invariant of matrix layout properties
        // Test both the meta tags and functional classes
        using SameMeta  = conditional_t<is_same_v<DataLayoutT, row_major>, row_major, col_major>;
        using OrthoMeta = conditional_t<is_same_v<DataLayoutT, row_major>, col_major, row_major>;
        using SameFunc  = conditional_t<is_same_v<DataLayoutT, row_major>, RowMajor, ColMajor>;
        using OrthoFunc = conditional_t<is_same_v<DataLayoutT, row_major>, ColMajor, RowMajor>;

        bool result = true;

        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(DataLayoutT, SameMeta, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(DataLayoutT, OrthoMeta, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(DataLayoutT, SameFunc, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(DataLayoutT, OrthoFunc, false, true, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              typename DataLayout>
    struct RegisterLayoutTestingSet
    {
        using ColInline = RegisterLayout::Storage<
            MatrixLayout::ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
            DataLayout>;
        using ColOrtho = RegisterLayout::Storage<
            MatrixLayout::ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
            DataLayout>;
        using RowInline = RegisterLayout::Storage<
            MatrixLayout::RowInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
            DataLayout>;
        using RowOrtho = RegisterLayout::Storage<
            MatrixLayout::RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
            DataLayout>;

        using MmaInput = RegisterLayout::MmaInput<BlockDim, DataT, false, (bool)ROCWMMA_ARCH_GFX11
                                                       ? RegisterLayout::Format::WMMA_INPUT_GFX11
                                                       : RegisterLayout::Format::SOA>;
        using MmaAcc   = RegisterLayout::MmaAcc<BlockDim, DataT, false, (bool)ROCWMMA_ARCH_GFX11
                                                     ? RegisterLayout::Format::WMMA_ACC_GFX11
                                                     : RegisterLayout::Format::SOA>;
    };

    template <typename StorageLayout>
    using MatrixLayout_t = typename layout_traits<StorageLayout>::MatrixLayout;

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool matrixLayoutTraitsTestNonInterleaved0()
    {
        constexpr bool debug_on_fail = true;

        // Testing MatrixLayout properties
        // MatrixLayouts are invariant to vector width
        constexpr uint32_t VectorWidth = MaxVectorWidth;
        using Set                      = RegisterLayoutTestingSet<BlockDim,
                                             BlockK,
                                             DataT,
                                             VectorWidth,
                                             MaxVectorWidth,
                                             DataLayoutT>;

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

    template <uint32_t MaxVectorWidth, typename DataT>
    ROCWMMA_DEVICE constexpr bool testMmaAccVW()
    {
        return MaxVectorWidth
               == ((bool)ROCWMMA_ARCH_GFX12
                       ? 8u
                       : ((is_same_v<DataT, float64_t> || (bool)ROCWMMA_ARCH_GFX11) ? 1u : 4u));
    }

    template<typename LayoutLhs, typename LayoutRhs>
    ROCWMMA_DEVICE constexpr void debug()
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

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestNonInterleaved0()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // VW = MaxVW
        // datalayout = same
        constexpr uint32_t VectorWidth = MaxVectorWidth;
        using Set0                     = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              DataLayoutT>;
        using Set1                     = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              DataLayoutT>;

        constexpr bool is_row_mjr = testRowMajor<DataLayoutT>();
        constexpr bool is_col_mjr = testColMajor<DataLayoutT>();
        constexpr bool is_mma_dim = testMmaDim<BlockDim, DataT>();
        constexpr bool is_acc_vw  = testMmaAccVW<MaxVectorWidth, DataT>();

        constexpr bool is_mma_row_mjr     = is_row_mjr && is_mma_dim;
        constexpr bool is_mma_col_mjr     = is_col_mjr && is_mma_dim;
        constexpr bool is_mma_acc_row_mjr = is_row_mjr && is_mma_dim && is_acc_vw;
        constexpr bool is_mma_acc_col_mjr = is_col_mjr && is_mma_dim && is_acc_vw;

        bool result = true;

        // Covered in another test case
        if constexpr(VectorWidth == 1u)
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

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc,  false,  is_mma_acc_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc,  false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  is_mma_acc_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho,  false,  is_mma_acc_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho,  false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  is_mma_acc_row_mjr, debug_on_fail);
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

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, is_mma_acc_row_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, is_mma_acc_col_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  is_mma_acc_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, is_mma_acc_row_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, is_mma_acc_col_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  is_mma_acc_row_mjr, debug_on_fail);
        }

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestNonInterleaved1()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // VW = MaxVW
        // datalayout = orthogonal
        constexpr uint32_t VectorWidth = MaxVectorWidth;
        using Set0                     = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              DataLayoutT>;
        using Set1                     = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              rocwmma::orthogonal_layout_t<DataLayoutT>>;

        constexpr bool is_row_mjr = testRowMajor<DataLayoutT>();
        constexpr bool is_col_mjr = testColMajor<DataLayoutT>();
        constexpr bool is_mma_dim = testMmaDim<BlockDim, DataT>();
        constexpr bool is_acc_vw  = testMmaAccVW<MaxVectorWidth, DataT>();

        constexpr bool is_mma_row_mjr     = is_row_mjr && is_mma_dim;
        constexpr bool is_mma_col_mjr     = is_col_mjr && is_mma_dim;
        constexpr bool is_mma_acc_row_mjr = is_row_mjr && is_mma_dim && is_acc_vw;
        constexpr bool is_mma_acc_col_mjr = is_col_mjr && is_mma_dim && is_acc_vw;

        bool result = true;

        // Covered in another test case
        if constexpr(VectorWidth == 1u)
        {
            return result;
        }

        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, is_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, is_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, true, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, is_row_mjr, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, true, false, debug_on_fail);
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
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, false,   is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_col_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc,  false,  is_mma_acc_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc,  false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  is_mma_acc_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho,  false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  is_mma_acc_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho,  false,  is_mma_acc_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  is_mma_acc_col_mjr, debug_on_fail);
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

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, is_mma_acc_row_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, is_mma_acc_col_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  is_mma_acc_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, is_mma_acc_col_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  is_mma_acc_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, is_mma_acc_row_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  is_mma_acc_col_mjr, debug_on_fail);
        }

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestNonInterleaved2()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // VW = 1u
        // datalayout = same
        constexpr uint32_t VectorWidth = 1u;
        using Set0                     = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              DataLayoutT>;
        using Set1                     = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              DataLayoutT>;

        constexpr bool is_mma_dim = testMmaDim<BlockDim, DataT>();
        constexpr bool is_acc_vw  = testMmaAccVW<MaxVectorWidth, DataT>();

        bool result = true;

        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, true, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, true, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowOrtho, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, true, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, true, false, debug_on_fail);

        // Storage <-> mma layouts
        // gfx11 have unique mma formats that must always be transformed from storage
        if constexpr ((bool)ROCWMMA_ARCH_GFX11)
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc,  false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc,  false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho,  false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho,  false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
        }
        // other targets have mma formats that may overlap storage formats
        else
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, (is_acc_vw && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, (is_acc_vw && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, (is_acc_vw && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, (is_acc_vw && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
        }

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestNonInterleaved3()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // VW = 1u
        // datalayout = orthogonal
        constexpr uint32_t VectorWidth = 1u;
        using Set0                     = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              DataLayoutT>;
        using Set1                     = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              rocwmma::orthogonal_layout_t<DataLayoutT>>;

        constexpr bool is_mma_dim = testMmaDim<BlockDim, DataT>();
        constexpr bool is_acc_vw  = testMmaAccVW<MaxVectorWidth, DataT>();

        bool result = true;

        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, true, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, true, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowOrtho, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, true, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, true, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, true, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, true, false, debug_on_fail);

        // Storage <-> mma layouts
        // gfx11 have unique mma formats that must always be transformed from storage
        if constexpr ((bool)ROCWMMA_ARCH_GFX11)
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc,  false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc,  false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho,  false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho,  false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
        }
        // other targets have mma formats that may overlap storage formats
        else
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, (is_acc_vw && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, (is_acc_vw && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, (is_acc_vw && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, (is_acc_vw && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
        }

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestNonInterleaved4()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // VW0 = 1u
        // VW1 = MaxVW
        // datalayout = same
        constexpr uint32_t VectorWidth0 = 1u;
        constexpr uint32_t VectorWidth1 = MaxVectorWidth;
        using Set0                      = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth0,
                                              MaxVectorWidth,
                                              DataLayoutT>;
        using Set1                      = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth1,
                                              MaxVectorWidth,
                                              DataLayoutT>;

        constexpr bool is_row_mjr = testRowMajor<DataLayoutT>();
        constexpr bool is_col_mjr = testColMajor<DataLayoutT>();
        constexpr bool is_mma_dim = testMmaDim<BlockDim, DataT>();
        constexpr bool is_acc_vw  = testMmaAccVW<MaxVectorWidth, DataT>();

        constexpr bool is_mma_row_mjr     = is_row_mjr && is_mma_dim;
        constexpr bool is_mma_col_mjr     = is_col_mjr && is_mma_dim;
        constexpr bool is_mma_acc_row_mjr = is_row_mjr && is_mma_dim && is_acc_vw;
        constexpr bool is_mma_acc_col_mjr = is_col_mjr && is_mma_dim && is_acc_vw;

        bool result = true;

        // Case tested in #0,1,2,3
        if constexpr(VectorWidth0 == VectorWidth1)
        {
            return result;
        }

        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, is_row_mjr, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, is_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, is_row_mjr, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, is_row_mjr, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, is_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, is_col_mjr, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, is_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, is_col_mjr, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, is_col_mjr, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowOrtho, false, is_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, is_col_mjr, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, is_col_mjr, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, is_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, is_row_mjr, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, is_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, is_row_mjr, false, debug_on_fail);

        // Storage <-> mma layouts
        // gfx11 have unique mma formats that must always be transformed from storage
        if constexpr ((bool)ROCWMMA_ARCH_GFX11)
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc,  false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc,  false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho,  false,  is_mma_acc_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho,  false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  is_mma_acc_row_mjr, debug_on_fail);
        }
        // other targets have mma formats that may overlap storage formats
        else
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, is_mma_row_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, is_mma_col_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_row_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, (is_acc_vw && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, (is_acc_vw && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, is_mma_acc_row_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, is_mma_acc_col_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  is_mma_acc_row_mjr, debug_on_fail);
        }

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestNonInterleaved5()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // VW0 = 1u
        // VW1 = MaxVW
        // datalayout = orthogonal
        constexpr uint32_t VectorWidth0 = 1u;
        constexpr uint32_t VectorWidth1 = MaxVectorWidth;
        using Set0                      = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth0,
                                              MaxVectorWidth,
                                              DataLayoutT>;
        using Set1                      = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth1,
                                              MaxVectorWidth,
                                              rocwmma::orthogonal_layout_t<DataLayoutT>>;

        constexpr bool is_row_mjr = testRowMajor<DataLayoutT>();
        constexpr bool is_col_mjr = testColMajor<DataLayoutT>();
        constexpr bool is_mma_dim = testMmaDim<BlockDim, DataT>();
        constexpr bool is_acc_vw  = testMmaAccVW<MaxVectorWidth, DataT>();

        constexpr bool is_mma_row_mjr     = is_row_mjr && is_mma_dim;
        constexpr bool is_mma_col_mjr     = is_col_mjr && is_mma_dim;
        constexpr bool is_mma_acc_row_mjr = is_row_mjr && is_mma_dim && is_acc_vw;
        constexpr bool is_mma_acc_col_mjr = is_col_mjr && is_mma_dim && is_acc_vw;

        bool result = true;

        // Case tested in #0,1,2,3
        if constexpr(VectorWidth0 == VectorWidth1)
        {
            return result;
        }

        // clang-format off
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColOrtho, is_col_mjr, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColOrtho, false, is_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColOrtho, is_col_mjr, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColOrtho, false, is_col_mjr, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::ColInline, false, is_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::ColInline, is_row_mjr, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::ColInline, false, is_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::ColInline, is_row_mjr, false, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowOrtho, is_row_mjr, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowOrtho, false, is_row_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowOrtho, is_row_mjr, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowOrtho, false, is_row_mjr, debug_on_fail);

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::RowInline, false, is_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowInline, is_col_mjr, false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::RowInline, false, is_col_mjr, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::RowInline, is_col_mjr, false, debug_on_fail);

        // Storage <-> mma layouts
        // gfx11 have unique mma formats that must always be transformed from storage
        if constexpr ((bool)ROCWMMA_ARCH_GFX11)
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false,  is_mma_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_col_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc,  false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc,  false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho,  false,  is_mma_acc_col_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  is_mma_acc_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho,  false,  is_mma_acc_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  is_mma_acc_col_mjr, debug_on_fail);

        }
        // other targets have mma formats that may overlap storage formats
        else
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, is_mma_col_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, is_mma_row_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_col_mjr, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, (is_acc_vw && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, (is_acc_vw && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  (is_acc_vw && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, is_mma_acc_col_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  is_mma_acc_row_mjr, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, is_mma_acc_row_mjr,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  is_mma_acc_col_mjr, debug_on_fail);
        }

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestNonInterleaved6()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // VW = 1
        // MaxVW0 = 1
        // MaxVW1 = MaxVW
        // datalayout = same
        constexpr uint32_t VectorWidth     = 1u;
        constexpr uint32_t MaxVectorWidth0 = MaxVectorWidth == 1u ? 4u : 1u;
        constexpr uint32_t MaxVectorWidth1 = MaxVectorWidth;
        using Set0                         = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth0,
                                              DataLayoutT>;
        using Set1                         = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth1,
                                              DataLayoutT>;

        constexpr bool is_mma_dim = testMmaDim<BlockDim, DataT>();
        constexpr bool is_acc_vw0 = testMmaAccVW<MaxVectorWidth0, DataT>();
        constexpr bool is_acc_vw1 = testMmaAccVW<MaxVectorWidth1, DataT>();

        bool result = true;

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
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowOrtho, false, false, debug_on_fail);
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
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc,  false,  (is_acc_vw0 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  (is_acc_vw0 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc,  false,  (is_acc_vw0 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  (is_acc_vw0 && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho,  false,  (is_acc_vw1 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  (is_acc_vw1 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho,  false,  (is_acc_vw1 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  (is_acc_vw1 && is_mma_dim), debug_on_fail);
        }
        // other targets have mma formats that may overlap storage formats
        else
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, (is_acc_vw0 && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  (is_acc_vw0 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, (is_acc_vw0 && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  (is_acc_vw0 && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, (is_acc_vw1 && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  (is_acc_vw1 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, (is_acc_vw1 && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  (is_acc_vw1 && is_mma_dim), debug_on_fail);
        }

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestNonInterleaved7()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // VW = 1
        // MaxVW0 = 1
        // MaxVW1 = MaxVW
        // datalayout = orthogonal
        constexpr uint32_t VectorWidth     = 1u;
        constexpr uint32_t MaxVectorWidth0 = MaxVectorWidth == 1u ? 4u : 1u;
        constexpr uint32_t MaxVectorWidth1 = MaxVectorWidth;
        using Set0                         = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth0,
                                              DataLayoutT>;
        using Set1                         = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth1,
                                              rocwmma::orthogonal_layout_t<DataLayoutT>>;

        constexpr bool is_mma_dim = testMmaDim<BlockDim, DataT>();
        constexpr bool is_acc_vw0 = testMmaAccVW<MaxVectorWidth0, DataT>();
        constexpr bool is_acc_vw1 = testMmaAccVW<MaxVectorWidth1, DataT>();

        bool result = true;

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
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowOrtho, false, false, debug_on_fail);
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
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho,  false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc,  false,  (is_acc_vw0 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  (is_acc_vw0 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc,  false,  (is_acc_vw0 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  (is_acc_vw0 && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho,  false,  (is_acc_vw1 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  (is_acc_vw1 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho,  false,  (is_acc_vw1 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  (is_acc_vw1 && is_mma_dim), debug_on_fail);
        }
        // other targets have mma formats that may overlap storage formats
        else
        {
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaInput, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaInput, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaInput, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColOrtho, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::ColInline, false,  is_mma_dim, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowOrtho, is_mma_dim,  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::RowInline, false,  is_mma_dim, debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColOrtho, typename Set1::MmaAcc, (is_acc_vw0 && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::MmaAcc, false,  (is_acc_vw0 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowOrtho, typename Set1::MmaAcc, (is_acc_vw0 && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::RowInline, typename Set1::MmaAcc, false,  (is_acc_vw0 && is_mma_dim), debug_on_fail);

            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColOrtho, (is_acc_vw1 && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::ColInline, false,  (is_acc_vw1 && is_mma_dim), debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowOrtho, (is_acc_vw1 && is_mma_dim),  false, debug_on_fail);
            result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::RowInline, false,  (is_acc_vw1 && is_mma_dim), debug_on_fail);
        }

        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaInput, typename Set1::MmaAcc, false,  false, debug_on_fail);
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::MmaAcc, typename Set1::MmaInput, false,  false, debug_on_fail);
        // clang-format on

        return result;
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestNonInterleaved8()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // VW = MaxVW
        // datalayout = same
        // Different BlockDim / BlockK
        constexpr uint32_t VectorWidth = MaxVectorWidth;
        constexpr uint32_t BlockDim0   = BlockDim;
        constexpr uint32_t BlockDim1   = BlockDim == 32u ? 64u : 32u;
        constexpr uint32_t BlockK0     = BlockK;
        constexpr uint32_t BlockK1     = BlockK == 32u ? 64u : 32u;
        using Set0                     = RegisterLayoutTestingSet<BlockDim0,
                                              BlockK0,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              DataLayoutT>;
        using Set1                     = RegisterLayoutTestingSet<BlockDim1,
                                              BlockK1,
                                              DataT,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              DataLayoutT>;

        bool result = true;

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
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowOrtho, false, false, debug_on_fail);
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
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestNonInterleaved9()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // VW = MaxVW
        // datalayout = same
        // Different size DataT
        constexpr uint32_t VectorWidth = MaxVectorWidth;
        using DataT0                   = DataT;
        using DataT1                   = conditional_t<
            sizeof(DataT) == 1u,
            int16_t,
            conditional_t<sizeof(DataT) == 2u,
                          int32_t,
                          conditional_t<sizeof(DataT) == 4u,
                                        int64_t,
                                        conditional_t<sizeof(DataT) == 8u, int32_t, DataT>>>>;

        using Set0 = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT0,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              DataLayoutT>;
        using Set1 = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT1,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              DataLayoutT>;

        bool result = true;

        // Already checked same types
        if constexpr(is_same_v<DataT0, DataT1>)
        {
            return result;
        }

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
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowOrtho, false, false, debug_on_fail);
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
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool registerLayoutTraitsTestNonInterleaved10()
    {
        constexpr bool debug_on_fail = true;

        // Non-interleaved
        // VW = MaxVW
        // datalayout = same
        // Same size DataT
        constexpr uint32_t VectorWidth = MaxVectorWidth;
        using DataT0                   = DataT;
        using DataT1                   = conditional_t<
            sizeof(DataT) == 1u,
            int8_t,
            conditional_t<sizeof(DataT) == 2u,
                          int16_t,
                          conditional_t<sizeof(DataT) == 4u,
                                        int32_t,
                                        conditional_t<sizeof(DataT) == 8u, int64_t, DataT>>>>;

        using Set0 = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT0,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              DataLayoutT>;
        using Set1 = RegisterLayoutTestingSet<BlockDim,
                                              BlockK,
                                              DataT1,
                                              VectorWidth,
                                              MaxVectorWidth,
                                              DataLayoutT>;

        bool result = true;

        // Already tested same type
        if constexpr(is_same_v<DataT0, DataT1>)
        {
            return result;
        }

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
        result &= ROCWMMA_TEST_LAYOUT_TRAITS_PAIR(typename Set0::ColInline, typename Set1::RowOrtho, false, false, debug_on_fail);
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
              uint32_t MaxVectorWidth>
    ROCWMMA_DEVICE bool testBarrageNonInterleaved()
    {
        bool result = true;

        // clang-format off
        result &= matrixLayoutTraitsTestNonInterleaved0<BlockDim, BlockK, DataT, DataLayoutT, MaxVectorWidth>();
        result &= registerLayoutTraitsTestNonInterleaved0<BlockDim, BlockK, DataT, DataLayoutT, MaxVectorWidth>();
        result &= registerLayoutTraitsTestNonInterleaved1<BlockDim, BlockK, DataT, DataLayoutT, MaxVectorWidth>();
        result &= registerLayoutTraitsTestNonInterleaved2<BlockDim, BlockK, DataT, DataLayoutT, MaxVectorWidth>();
        result &= registerLayoutTraitsTestNonInterleaved3<BlockDim, BlockK, DataT, DataLayoutT, MaxVectorWidth>();
        result &= registerLayoutTraitsTestNonInterleaved4<BlockDim, BlockK, DataT, DataLayoutT, MaxVectorWidth>();
        result &= registerLayoutTraitsTestNonInterleaved5<BlockDim, BlockK, DataT, DataLayoutT, MaxVectorWidth>();
        result &= registerLayoutTraitsTestNonInterleaved6<BlockDim, BlockK, DataT, DataLayoutT, MaxVectorWidth>();
        result &= registerLayoutTraitsTestNonInterleaved7<BlockDim, BlockK, DataT, DataLayoutT, MaxVectorWidth>();
        result &= registerLayoutTraitsTestNonInterleaved8<BlockDim, BlockK, DataT, DataLayoutT, MaxVectorWidth>();
        result &= registerLayoutTraitsTestNonInterleaved9<BlockDim, BlockK, DataT, DataLayoutT, MaxVectorWidth>();
        result &= registerLayoutTraitsTestNonInterleaved10<BlockDim, BlockK, DataT, DataLayoutT, MaxVectorWidth>();
        // clang-format on

        return result;
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayoutT>
    ROCWMMA_DEVICE bool layoutTraitsTestA()
    {
        // TODO: WaveCount
        constexpr uint32_t WaveCount = 1u;
        constexpr uint32_t BlockDim  = BlockM;
        constexpr uint32_t BlockK    = BlockN;
        constexpr uint32_t MaxVW = detail::MaxVWSelector<matrix_a, BlockDim, BlockK, DataT, DataLayoutT, WaveCount>::Result;

        bool result = true;
        result &= dataLayoutTraitsTest<DataLayoutT>();
        result &= testBarrageNonInterleaved<BlockDim, BlockK, DataT, DataLayoutT, MaxVW>();

        return result;
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayoutT>
    ROCWMMA_DEVICE bool layoutTraitsTestB()
    {
        // TODO: WaveCount
        constexpr uint32_t WaveCount = 1u;
        constexpr uint32_t BlockDim  = BlockN;
        constexpr uint32_t BlockK    = BlockM;
        constexpr uint32_t MaxVW     = rocwmma::detail::
            MaxVWSelector<matrix_b, BlockDim, BlockK, DataT, DataLayoutT, WaveCount>::Result;

        bool result = true;
        result &= dataLayoutTraitsTest<DataLayoutT>();
        result &= testBarrageNonInterleaved<BlockDim, BlockK, DataT, DataLayoutT, MaxVW>();

        return result;
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayoutT>
    ROCWMMA_DEVICE bool layoutTraitsTestAcc()
    {
        // TODO: WaveCount
        constexpr uint32_t WaveCount = 1u;
        constexpr uint32_t BlockDim  = BlockN;
        constexpr uint32_t BlockK    = BlockM;
        constexpr uint32_t MaxVW     = rocwmma::detail::
            MaxVWSelector<accumulator, BlockDim, BlockK, DataT, DataLayoutT, WaveCount>::Result;

        bool result = true;
        result &= dataLayoutTraitsTest<DataLayoutT>();
        result &= testBarrageNonInterleaved<BlockDim, BlockK, DataT, DataLayoutT, MaxVW>();

        return result;
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayoutT>
    __global__ void layoutTraitsTest(uint32_t     m,
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

        success &= layoutTraitsTestA<BlockM, BlockN, DataT, DataLayoutT>();
        success &= layoutTraitsTestB<BlockM, BlockN, DataT, DataLayoutT>();
        success &= layoutTraitsTestAcc<BlockM, BlockN, DataT, DataLayoutT>();

        // Reduce error count
        atomicAdd(&result, (int32_t)success);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? ERROR_VALUE : SUCCESS_VALUE);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_LAYOUT_TRAITS_TEST_HPP
