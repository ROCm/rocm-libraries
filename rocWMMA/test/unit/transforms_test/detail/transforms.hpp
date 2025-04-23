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

#ifndef ROCWMMA_DETAIL_TRANSFORMS_TEST_HPP
#define ROCWMMA_DETAIL_TRANSFORMS_TEST_HPP

#include "common.hpp"
#include "device/transforms.hpp"
#include "unit_kernel_base.hpp"

namespace rocwmma
{
    template <typename DataT>
    using TransformsKernelBase
        = UnitKernelBase<1,
                         1,
                         DataT,
                         col_major>; // BlockM, BlockN, DataLayout are redundant for this test

    // Wrapper into the actual device function
    template <uint32_t BlockDim, uint32_t VW, typename DataT>
    struct TransformsKernel : public TransformsKernelBase<DataT>
    {
    protected:
        using Base = TransformsKernelBase<DataT>;

    public:
        TransformsKernel()  = default;
        ~TransformsKernel() = default;

        bool checkSizes() const override
        {
            return true;
        }

        void setupImpl(typename Base::DataStorage::ProblemSize const& probsize) final
        {
            // Need at least 1 element for the result
            auto& dataInstance = Base::DataStorage::instance();
            dataInstance->resizeStorage(probsize);

            dataInstance->hostOut().get()[0] = static_cast<DataT>(ERROR_VALUE);
            dataInstance->copyData(dataInstance->deviceOut(), dataInstance->hostOut(), 1);
        }

        void validateResultsImpl() final
        {
            auto& dataInstance = Base::DataStorage::instance();

            // Cache current kernel result from device
            dataInstance->copyData(dataInstance->hostOut(), dataInstance->deviceOut(), 1);

            // Check the single output result
            Base::mValidationResult = (dataInstance->hostOut().get()[0] == DataT(SUCCESS_VALUE));
        }

        std::ostream& printHeader(std::ostream& stream = std::cout) const override
        {
            return stream << "WSize, DataT, VW, BlockDim" << std::endl;
        }

        std::ostream& printKernel(std::ostream& stream = std::cout) const override
        {
            using DeviceInfo = HipDevice;
            stream << "w" << DeviceInfo::instance()->warpSize() << ", " << dataTypeToString<DataT>()
                   << ", " << VW << ", " << BlockDim << std::endl;

            return stream;
        }
    };

    template <uint32_t BlockDim, uint32_t VW, typename DataT>
    struct AossoaKernel : public TransformsKernel<BlockDim, VW, DataT>
    {
    protected:
        using Base = TransformsKernelBase<DataT>;

    public:
        typename Base::KernelFunc kernelImpl() const override
        {
            return typename Base::KernelFunc(aossoaTest<DataT, VW, BlockDim>);
        }
    };

    template <uint32_t BlockDim, uint32_t VW, typename DataT>
    struct SoaaosKernel : public TransformsKernel<BlockDim, VW, DataT>
    {
    protected:
        using Base = TransformsKernelBase<DataT>;

    public:
        typename Base::KernelFunc kernelImpl() const override
        {
            return typename Base::KernelFunc(soaaosTest<DataT, VW, BlockDim>);
        }
    };

    // This is the GeneratorImpl class
    template <template <uint32_t BlockDim, uint32_t VW, typename DataT> typename Func>
    struct TransformsGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            BlockDim = 0,
            VW       = 1,
            DataT    = 2,
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT     = Func<std::tuple_element_t<BlockDim, TestParamsT>::value, // BlockDim
                                 std::tuple_element_t<VW, TestParamsT>::value, // VW
                                 std::tuple_element_t<DataT, TestParamsT> // DataT
                                 >;

            return std::make_shared<KernelT>();
        }
    };
    using AossoaGenerator = TransformsGenerator<AossoaKernel>;
    using SoaaosGenerator = TransformsGenerator<SoaaosKernel>;

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_TRANSFORMS_TEST_HPP
