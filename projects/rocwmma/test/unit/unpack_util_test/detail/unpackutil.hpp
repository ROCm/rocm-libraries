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

#include "device/unpackutil.hpp"
#include "references/register_bank.hpp"
#include "unit_kernel_base.hpp"
#include <numeric>

namespace rocwmma
{

    /*************************************************************
     *        Kernels
     *************************************************************/

    template <typename DataT>
    using UnpackKernelBase
        = UnitKernelBase<1,
                         1,
                         DataT,
                         col_major>; // BlockM, BlockN, DataLayout are redundant for this test

    // Wrapper into the actual device function
    template <uint32_t VW, typename DataT>
    struct UnpackKernel : public UnpackKernelBase<DataT>
    {
    protected:
        using Base = UnpackKernelBase<DataT>;

        virtual void     manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) = 0;
        virtual uint32_t resultSize() const                                                    = 0;

    public:
        UnpackKernel()  = default;
        ~UnpackKernel() = default;

        bool checkSizes() const override
        {
            return true;
        }

        void setupImpl(typename Base::DataStorage::ProblemSize const&) final
        {
            // Need at least 1 element for the result
            auto& dataInstance = Base::DataStorage::instance();
            auto  blockX       = static_cast<uint32_t>(this->blockDim().x);
            dataInstance->resizeStorage(std::make_tuple(blockX, VW));

            auto hostInData = dataInstance->hostIn().get();
            auto dataSize   = blockX * VW;
            std::iota(hostInData, hostInData + dataSize, 0);
            dataInstance->copyData(dataInstance->deviceIn(), dataInstance->hostIn(), dataSize);
        }

        void validateResultsImpl() final
        {
            auto& dataInstance = Base::DataStorage::instance();

            auto               blockX     = static_cast<uint32_t>(this->blockDim().x);
            auto               resultSize = this->resultSize();
            std::vector<DataT> expectedData(resultSize);

            test::references::RegisterBank<DataT> m(blockX, VW);
            m.setData(dataInstance->hostIn().get());
            manipulateRegisterBankOnCpu(m);
            m.copyTo(expectedData.data());

            // Cache current kernel result from device
            dataInstance->copyData(dataInstance->hostOut(), dataInstance->deviceOut(), resultSize);

            // Check the single output result
            Base::mValidationResult
                = 0 == memcmp(expectedData.data(), dataInstance->hostOut().get(), resultSize);
        }

        std::ostream& printHeader(std::ostream& stream = std::cout) const override
        {
            return stream << "WSize, DataT, VW" << std::endl;
        }
        std::ostream& printKernel(std::ostream& stream = std::cout) const override
        {
            using DeviceInfo = HipDevice;
            stream << "w" << DeviceInfo::instance()->warpSize() << ", " << dataTypeToString<DataT>()
                   << ", " << VW;

            return stream;
        }
    };

    // Wrapper into the actual device function
    template <uint32_t VW, typename DataT>
    struct UnpackLo2Kernel final : public UnpackKernel<VW, DataT>
    {
    private:
        using Base = typename UnpackKernel<VW, DataT>::Base;

    public:
        UnpackLo2Kernel()        = default;
        ~UnpackLo2Kernel() final = default;

        void manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) override
        {
            m.unpackLo2();
        }

        uint32_t resultSize() const override
        {
            auto blockX = static_cast<uint32_t>(this->blockDim().x);
            return blockX * VW / 2;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(unpackLo2Test<DataT, VW>);
        }
    };

    template <uint32_t VW, typename DataT>
    struct UnpackLo4Kernel final : public UnpackKernel<VW, DataT>
    {
    private:
        using Base = typename UnpackKernel<VW, DataT>::Base;

    public:
        UnpackLo4Kernel()        = default;
        ~UnpackLo4Kernel() final = default;

        void manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) override
        {
            m.unpackLo4();
        }

        uint32_t resultSize() const override
        {
            auto blockX = static_cast<uint32_t>(this->blockDim().x);
            return blockX * VW / 2;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(unpackLo4Test<DataT, VW>);
        }
    };

    template <uint32_t VW, typename DataT>
    struct UnpackLo8Kernel final : public UnpackKernel<VW, DataT>
    {
    private:
        using Base = typename UnpackKernel<VW, DataT>::Base;

    public:
        UnpackLo8Kernel()        = default;
        ~UnpackLo8Kernel() final = default;

        void manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) override
        {
            m.unpackLo8();
        }

        uint32_t resultSize() const override
        {
            auto blockX = static_cast<uint32_t>(this->blockDim().x);
            return blockX * VW / 2;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(unpackLo8Test<DataT, VW>);
        }
    };

    template <uint32_t VW, typename DataT>
    struct UnpackHi2Kernel final : public UnpackKernel<VW, DataT>
    {
    private:
        using Base = typename UnpackKernel<VW, DataT>::Base;

    public:
        UnpackHi2Kernel()        = default;
        ~UnpackHi2Kernel() final = default;

        void manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) override
        {
            m.unpackHi2();
        }

        uint32_t resultSize() const override
        {
            auto blockX = static_cast<uint32_t>(this->blockDim().x);
            return blockX * VW / 2;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(unpackHi2Test<DataT, VW>);
        }
    };

    template <uint32_t VW, typename DataT>
    struct UnpackHi4Kernel final : public UnpackKernel<VW, DataT>
    {
    private:
        using Base = typename UnpackKernel<VW, DataT>::Base;

    public:
        UnpackHi4Kernel()        = default;
        ~UnpackHi4Kernel() final = default;

        void manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) override
        {
            m.unpackHi4();
        }

        uint32_t resultSize() const override
        {
            auto blockX = static_cast<uint32_t>(this->blockDim().x);
            return blockX * VW / 2;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(unpackHi4Test<DataT, VW>);
        }
    };

    template <uint32_t VW, typename DataT>
    struct UnpackHi8Kernel final : public UnpackKernel<VW, DataT>
    {
    private:
        using Base = typename UnpackKernel<VW, DataT>::Base;

    public:
        UnpackHi8Kernel()        = default;
        ~UnpackHi8Kernel() final = default;

        void manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) override
        {
            m.unpackHi8();
        }

        uint32_t resultSize() const override
        {
            auto blockX = static_cast<uint32_t>(this->blockDim().x);
            return blockX * VW / 2;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(unpackHi8Test<DataT, VW>);
        }
    };

    template <uint32_t VW, typename DataT>
    struct UnpackLoHi1Kernel final : public UnpackKernel<VW, DataT>
    {
    private:
        using Base = typename UnpackKernel<VW, DataT>::Base;

    public:
        UnpackLoHi1Kernel()        = default;
        ~UnpackLoHi1Kernel() final = default;

        void manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) override
        {
            m.unpackLoHi1();
        }

        uint32_t resultSize() const override
        {
            auto blockX = static_cast<uint32_t>(this->blockDim().x);
            return blockX * VW;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(unpackLoHi1Test<DataT, VW>);
        }
    };

    template <uint32_t VW, typename DataT>
    struct UnpackLoHi2Kernel final : public UnpackKernel<VW, DataT>
    {
    private:
        using Base = typename UnpackKernel<VW, DataT>::Base;

    public:
        UnpackLoHi2Kernel()        = default;
        ~UnpackLoHi2Kernel() final = default;

        void manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) override
        {
            m.unpackLoHi2();
        }

        uint32_t resultSize() const override
        {
            auto blockX = static_cast<uint32_t>(this->blockDim().x);
            return blockX * VW;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(unpackLoHi2Test<DataT, VW>);
        }
    };

    template <uint32_t VW, typename DataT>
    struct UnpackLoHi4Kernel final : public UnpackKernel<VW, DataT>
    {
    private:
        using Base = typename UnpackKernel<VW, DataT>::Base;

    public:
        UnpackLoHi4Kernel()        = default;
        ~UnpackLoHi4Kernel() final = default;

        void manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) override
        {
            m.unpackLoHi4();
        }

        uint32_t resultSize() const override
        {
            auto blockX = static_cast<uint32_t>(this->blockDim().x);
            return blockX * VW;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(unpackLoHi4Test<DataT, VW>);
        }
    };

    template <uint32_t VW, typename DataT>
    struct UnpackLoHi8Kernel final : public UnpackKernel<VW, DataT>
    {
    private:
        using Base = typename UnpackKernel<VW, DataT>::Base;

    public:
        UnpackLoHi8Kernel()        = default;
        ~UnpackLoHi8Kernel() final = default;

        void manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) override
        {
            m.unpackLoHi8();
        }

        uint32_t resultSize() const override
        {
            auto blockX = static_cast<uint32_t>(this->blockDim().x);
            return blockX * VW;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(unpackLoHi8Test<DataT, VW>);
        }
    };

    template <uint32_t VW, typename DataT>
    struct UnpackLoHi16Kernel final : public UnpackKernel<VW, DataT>
    {
    private:
        using Base = typename UnpackKernel<VW, DataT>::Base;

    public:
        UnpackLoHi16Kernel()        = default;
        ~UnpackLoHi16Kernel() final = default;

        void manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) override
        {
            m.unpackLoHi16();
            // m.print();
        }

        uint32_t resultSize() const override
        {
            auto blockX = static_cast<uint32_t>(this->blockDim().x);
            return blockX * VW;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(unpackLoHi16Test<DataT, VW>);
        }
    };

    template <uint32_t VW, typename DataT>
    struct UnpackLoHi32Kernel final : public UnpackKernel<VW, DataT>
    {
    private:
        using Base = typename UnpackKernel<VW, DataT>::Base;

    public:
        UnpackLoHi32Kernel()        = default;
        ~UnpackLoHi32Kernel() final = default;

        void manipulateRegisterBankOnCpu(test::references::RegisterBank<DataT>& m) override
        {
            m.unpackLoHi32();
            // m.print();
        }

        bool checkSizes() const override
        {
            return Base::DeviceInfo::instance()->warpSize() > 32;
        }

        uint32_t resultSize() const override
        {
            auto blockX = static_cast<uint32_t>(this->blockDim().x);
            return blockX * VW;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(unpackLoHi32Test<DataT, VW>);
        }
    };

    /*************************************************************
     *       Generator
     *************************************************************/

    // This is the GeneratorImpl class
    template <template <uint32_t VW, typename DataT> typename Func>
    struct UnpackGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            VW    = 0,
            DataT = 1,
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT     = Func<std::tuple_element_t<VW, TestParamsT>::value, // VW
                                 std::tuple_element_t<DataT, TestParamsT> // DataT
                                 >;

            return std::make_shared<KernelT>();
        }
    };

    using UnpackLo2Generator    = UnpackGenerator<UnpackLo2Kernel>;
    using UnpackLo4Generator    = UnpackGenerator<UnpackLo4Kernel>;
    using UnpackLo8Generator    = UnpackGenerator<UnpackLo8Kernel>;
    using UnpackHi2Generator    = UnpackGenerator<UnpackHi2Kernel>;
    using UnpackHi4Generator    = UnpackGenerator<UnpackHi4Kernel>;
    using UnpackHi8Generator    = UnpackGenerator<UnpackHi8Kernel>;
    using UnpackLoHi1Generator  = UnpackGenerator<UnpackLoHi1Kernel>;
    using UnpackLoHi2Generator  = UnpackGenerator<UnpackLoHi2Kernel>;
    using UnpackLoHi4Generator  = UnpackGenerator<UnpackLoHi4Kernel>;
    using UnpackLoHi8Generator  = UnpackGenerator<UnpackLoHi8Kernel>;
    using UnpackLoHi16Generator = UnpackGenerator<UnpackLoHi16Kernel>;
    using UnpackLoHi32Generator = UnpackGenerator<UnpackLoHi32Kernel>;
} // namespace rocwmma

#endif // ROCWMMA_DETAIL_TRANSFORMS_TEST_HPP
