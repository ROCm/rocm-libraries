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

#ifndef ROCWMMA_DETAIL_IO_SHAPE_HPP
#define ROCWMMA_DETAIL_IO_SHAPE_HPP

#include "unit_kernel_base.hpp"

namespace rocwmma
{
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    struct IOShapeKernel final : public UnitKernelBase<BlockM, BlockN, DataT, DataLayoutT>
    {
    private:
        using Base = UnitKernelBase<BlockM, BlockN, DataT, DataLayoutT>;

    public:
        IOShapeKernel()        = default;
        ~IOShapeKernel() final = default;

        void setupImpl(typename Base::DataStorage::ProblemSize const& probsize) final
        {
            // Need at least 1 element for the result
            auto& dataInstance = Base::DataStorage::instance();
            dataInstance->resizeStorage(probsize);

            dataInstance->hostOut().get()[0] = static_cast<DataT>(ERROR_VALUE);
        }

        template <uint32_t WaveCount>
        bool waveTest()
        {
            bool err = false;

            // Accum requires WaveCount > 1
            if constexpr(!std::is_same_v<MatrixT, accumulator> || WaveCount == 1)
            {
                constexpr auto BlockDim = std::is_same_v<MatrixT, matrix_a> ? BlockM : BlockN;
                constexpr auto KDim     = std::is_same_v<MatrixT, accumulator> ? BlockM : BlockK;

                using detail::MaxVWSelector;
                using detail::MmaDimSelector;

                constexpr auto ExpectMaxVW
                    = MaxVWSelector<MatrixT, BlockDim, KDim, DataT, DataLayoutT, WaveCount>::Result;

                constexpr auto ExpectVW
                    = std::is_same_v<MatrixT, matrix_a>
                          ? std::is_same<DataLayoutT, row_major>::value || BlockDim > 32
                                ? ExpectMaxVW
                                : 1u
                      : std::is_same_v<MatrixT, matrix_b>
                          ? (std::is_same<DataLayoutT, col_major>::value || BlockDim > 32
                                 ? ExpectMaxVW
                                 : 1u)
                          : (std::is_same<DataLayoutT, col_major>::value ? ExpectMaxVW : 1u);

                constexpr auto ExpectMmaDim = MmaDimSelector<BlockDim, DataT>::Result;

                using IOLayout = IOLayout<MatrixT, BlockDim, KDim, DataT, DataLayoutT, WaveCount>;
                using IOLayoutInt
                    = IOLayoutInt<MatrixT, BlockDim, KDim, DataT, DataLayoutT, WaveCount>;
                using ExpectDataLayout = DataLayout::template Array1d<DataLayoutT>;

                err |= (IOLayout::MaxVW != ExpectMaxVW);
                err |= (IOLayout::VW != ExpectVW);
                err |= (IOLayoutInt::MmaDim != ExpectMmaDim);
                err |= (!std::is_same_v<typename IOLayout::DataLayout, ExpectDataLayout>);
            }

            return err;
        }

        void exec() final
        {
            if(Base::mRunFlag)
            {
                bool err = false;

                constexpr auto BlockDim    = std::is_same_v<MatrixT, matrix_a> ? BlockM : BlockN;
                constexpr auto KDim        = std::is_same_v<MatrixT, accumulator> ? BlockM : BlockK;
                constexpr auto BlockHeight = std::is_same_v<MatrixT, matrix_b> ? BlockK : BlockM;
                constexpr auto BlockWidth  = std::is_same_v<MatrixT, matrix_a> ? BlockK : BlockN;

                using IOShape = IOShape<MatrixT, BlockM, BlockN, BlockK>;

                // Sanity check on matrix shape properties
                err |= (IOShape::BlockDim != BlockDim);
                err |= (IOShape::KDim != KDim);
                err |= (IOShape::BlockHeight != BlockHeight);
                err |= (IOShape::BlockWidth != BlockWidth);

                // Test on all supported wave counts
                err |= waveTest<1>();
                err |= waveTest<2>();
                err |= waveTest<4>();

                if(!err)
                {
                    auto& dataInstance               = Base::DataStorage::instance();
                    dataInstance->hostOut().get()[0] = static_cast<DataT>(SUCCESS_VALUE);
                }
            }
        }

        void validateResultsImpl() final
        {
            auto& dataInstance = Base::DataStorage::instance();

            // Check the single output result
            Base::mValidationResult = (dataInstance->hostOut().get()[0] == DataT(SUCCESS_VALUE));
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(nullptr);
        }
    };

    // This is the GeneratorImpl class
    struct IOShapeGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            MatrixT     = 0,
            BlockMN     = 1,
            BlockK      = 2,
            DataT       = 3,
            DataLayoutT = 4,
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = IOShapeKernel<std::tuple_element_t<MatrixT, TestParamsT>, // MatrixT
                                std::tuple_element_t<BlockMN, TestParamsT>::value, // BlockM
                                std::tuple_element_t<BlockMN, TestParamsT>::value, // BlockN
                                std::tuple_element_t<BlockK, TestParamsT>::value, // BlockK
                                std::tuple_element_t<DataT, TestParamsT>, // DataT
                                std::tuple_element_t<DataLayoutT, TestParamsT> // DataLayoutT
                                >;

            return std::make_shared<KernelT>();
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_IO_SHAPE_HPP
