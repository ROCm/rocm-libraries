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

#ifndef ROCWMMA_DETAIL_MAP_MATRIX_TO_DATA_HPP
#define ROCWMMA_DETAIL_MAP_MATRIX_TO_DATA_HPP

#include "device/map_matrix_to_data.hpp"
#include "unit_kernel_base.hpp"

namespace rocwmma
{

    // Wrapper into the actual device function
    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct MapMatrixToDataKernel final : public UnitKernelBase<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = UnitKernelBase<BlockM, BlockN, DataT, Layout>;

    public:
        MapMatrixToDataKernel()        = default;
        ~MapMatrixToDataKernel() final = default;

        void setupImpl(typename Base::DataStorage::ProblemSize const& probsize) final
        {
            auto& dataInstance = Base::DataStorage::instance();

            // Initialize matrix storage
            const int64_t sizeD = Base::mM * Base::mN;
            dataInstance->resizeStorage(probsize);

            // Initialize matrix data on host
            MatrixUtil<Layout>::fillLaunchKernel(
                dataInstance->deviceIn().get(), Base::mM, Base::mN);
            MatrixUtil<Layout>::fillValLaunchKernel(dataInstance->deviceOut().get(),
                                                    Base::mM,
                                                    Base::mN,
                                                    std::numeric_limits<DataT>::signaling_NaN());
        }

        void validateResultsImpl() final
        {
            auto& dataInstance = Base::DataStorage::instance();

            const int64_t sizeD = Base::mM * Base::mN;

            double errorTolerance = 10.0;

            std::tie(Base::mValidationResult, Base::mMaxRelativeError)
                = compareEqualLaunchKernel<DataT, DataT, Layout, Layout>(
                    dataInstance->deviceIn().get(),
                    dataInstance->deviceOut().get(),
                    Base::mM,
                    Base::mN,
                    errorTolerance);
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(MapMatrixToData<BlockM, BlockN, DataT, Layout>);
        }
    };

    // This is the GeneratorImpl class
    struct MapMatrixToDataGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT  = 0,
            BlockM = 1,
            BlockN = 2,
            Layout = 3
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = MapMatrixToDataKernel<std::tuple_element_t<BlockM, TestParamsT>::value, // BlockM
                                        std::tuple_element_t<BlockN, TestParamsT>::value, // BlockN
                                        std::tuple_element_t<DataT, TestParamsT>, // DataT
                                        std::tuple_element_t<Layout, TestParamsT> // Layout
                                        >;

            return std::make_shared<KernelT>();
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_MAP_MATRIX_TO_DATA_HPP
