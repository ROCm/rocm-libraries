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

#ifndef ROCWMMA_DETAIL_VECTOR_ITERATOR_HPP
#define ROCWMMA_DETAIL_VECTOR_ITERATOR_HPP

#include "common.hpp"
#include "device/vector_iterator.hpp"
#include "unit_kernel_base.hpp"

namespace rocwmma
{

    // Wrapper into the actual device function
    template <uint32_t VecSize, typename DataT>
    struct VectorIteratorKernel final
        : public UnitKernelBase<1,
                                1,
                                DataT,
                                col_major> // BlockM, BlockN, DataLayout are redundant for this test
    {
    private:
        using Base = UnitKernelBase<1, 1, DataT, col_major>;

    public:
        VectorIteratorKernel()        = default;
        ~VectorIteratorKernel() final = default;

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

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(vectorIteratorTest<DataT, VecSize>);
        }
    };

    // This is the GeneratorImpl class
    struct VectorIteratorGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            VecSize = 0,
            DataT   = 1,
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = VectorIteratorKernel<std::tuple_element_t<VecSize, TestParamsT>::value, // VecSize
                                       std::tuple_element_t<DataT, TestParamsT> // DataT
                                       >;

            return std::make_shared<KernelT>();
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_VECTOR_ITERATOR_HPP
