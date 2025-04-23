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

#ifndef ROCWMMA_DETAIL_CROSS_LANE_OPS_HPP
#define ROCWMMA_DETAIL_CROSS_LANE_OPS_HPP

#include "device/cross_lane_ops.hpp"
#include "reference.hpp"
#include "unit_kernel_base.hpp"
#include <rocwmma/internal/pack_util.hpp>

namespace rocwmma
{

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask  = 0xF,
              uint32_t WriteBankMask = 0xF,
              bool     BoundCtrl     = false>
    struct CrossLaneOpsKernelBase
        : public UnitKernelBase<1,
                                1,
                                DataT,
                                col_major> // BlockM, BlockN, DataLayout are redundant for this test
    {
    protected:
        using Base   = UnitKernelBase<1, 1, DataT, col_major>;
        using Layout = col_major;

    public:
        CrossLaneOpsKernelBase()          = default;
        virtual ~CrossLaneOpsKernelBase() = default;

        bool checkSizes() const final
        {
            return (Base::mTBlockY == 1);
        }

        bool checkDevice() const final
        {
            auto& deviceInfo = Base::DeviceInfo::instance();

            auto deviceArch = deviceInfo->getGcnArch();

            // gfx908 doesn't support dpp BCast16
            bool dppBCast16Check
                = !((deviceArch == Base::DeviceInfo::GFX908)
                    && (CrossLaneOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_DPP)
                    && (CrossLaneOp::opId() == CrossLaneOps::Properties::OP_ID_BCAST)
                    && (CrossLaneOp::groupSize() == 16u));

            // gfx11 doesn't support dpp wave shift / rotate, bcast15x16 or bcast31x32
            bool isGfx11 = (deviceArch == Base::DeviceInfo::GFX1100
                            || deviceArch == Base::DeviceInfo::GFX1101
                            || deviceArch == Base::DeviceInfo::GFX1102);

            bool isGfx12 = (deviceArch == Base::DeviceInfo::GFX1200)
                           || (deviceArch == Base::DeviceInfo::GFX1201);

            bool dppWaveShiftCheck
                = !((isGfx11 || isGfx12)
                    && (CrossLaneOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_DPP)
                    && (CrossLaneOp::opId() == CrossLaneOps::Properties::OP_ID_SHIFT)
                    && (CrossLaneOp::groupSize() == CrossLaneOps::Properties::OP_GROUP_SIZE_WARP));

            bool dppWaveRotateCheck
                = !((isGfx11 || isGfx12)
                    && (CrossLaneOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_DPP)
                    && (CrossLaneOp::opId() == CrossLaneOps::Properties::OP_ID_ROTATE)
                    && (CrossLaneOp::groupSize() == CrossLaneOps::Properties::OP_GROUP_SIZE_WARP));

            bool dppWaterfallBCastCheck
                = !((isGfx11 || isGfx12)
                    && (CrossLaneOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_DPP)
                    && (CrossLaneOp::opId() == CrossLaneOps::Properties::OP_ID_WFALL_BCAST));

            return Base::checkDevice() && dppBCast16Check && dppWaveShiftCheck && dppWaveRotateCheck
                   && dppWaterfallBCastCheck;
        }

        std::ostream& printHeader(std::ostream& stream = std::cout) const final
        {
            return stream << "DataT,"
                          << "Op_Id, "
                          << "Op_Impl, "
                          << "Wave_Size, "
                          << "Group_Size, "
                          << "WriteRowMask, "
                          << "WriteBankMask, "
                          << "BoundCtrl, "
                          << "Result" << std::endl;
        }
        std::ostream& printKernel(std::ostream& stream = std::cout) const final
        {
            std::ios_base::fmtflags f(stream.flags()); // save flags state
            stream << dataTypeToString<DataT>() << ", " << CrossLaneOp::opId() << ", "
                   << CrossLaneOp::opImpl() << ", ";
            stream.flags(f);
            stream << "w" << Base::DeviceInfo::instance()->warpSize() << ", "
                   << CrossLaneOp::groupSize() << ", ";
            stream << std::showbase << std::hex << WriteRowMask << ", " << WriteBankMask << ", ";
            stream.flags(f);
            stream << BoundCtrl << ", ";

            if(!Base::mRunFlag)
            {
                stream << "SKIPPED" << std::endl;
            }
            else
            {
                stream << (Base::mValidationResult ? "PASSED" : "FAILED") << std::endl;
            }

            stream.flags(f);
            return stream;
        }

        virtual void setupImpl(typename Base::DataStorage::ProblemSize const& /*probsize*/)
        {
            // Allocate storage
            auto& dataInstance = Base::DataStorage::instance();

            // Need only one entry to save the test result
            dataInstance->resizeStorage({1, 1});

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

        virtual typename Base::KernelFunc kernelImpl() const = 0;
    };

    template <typename DataT, typename CrossLaneOp>
    struct BlendOpsKernel final : public CrossLaneOpsKernelBase<DataT, CrossLaneOp>
    {
        using Base = CrossLaneOpsKernelBase<DataT, CrossLaneOp>;

    public:
        BlendOpsKernel()  = default;
        ~BlendOpsKernel() = default;

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(blendOpsTest<DataT, CrossLaneOp>);
        }

        typename Base::DataStorage::HostPtrT mSrc1;
    };

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    struct DppOpsKernel final
        : public CrossLaneOpsKernelBase<DataT, CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>
    {
        static_assert((CrossLaneOp::GROUP_SIZE & (CrossLaneOp::GROUP_SIZE - 1)) == 0,
                      "SubGroupSize must be power of 2.");
        using Base = UnitKernelBase<1, 1, DataT, col_major>;

    public:
        DppOpsKernel()  = default;
        ~DppOpsKernel() = default;

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(
                dppOpsTest<DataT, CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>);
        }
    };

    template <typename DataT, typename CrossLaneOp>
    struct SwizzleOpsKernel final : public CrossLaneOpsKernelBase<DataT, CrossLaneOp>
    {
        using Base = UnitKernelBase<1, 1, DataT, col_major>;

    public:
        SwizzleOpsKernel()  = default;
        ~SwizzleOpsKernel() = default;

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(swizzleOpsTest<DataT, CrossLaneOp>);
        }
    };

    template <typename DataT, typename CrossLaneOp>
    struct PermuteOpsKernel final : public CrossLaneOpsKernelBase<DataT, CrossLaneOp>
    {
        using Base = UnitKernelBase<1, 1, DataT, col_major>;

    public:
        PermuteOpsKernel()  = default;
        ~PermuteOpsKernel() = default;

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(permuteOpsTest<DataT, CrossLaneOp>);
        }
    };

    // This is the GeneratorImpl class
    struct DppOpsGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT         = 0,
            CrossLaneOp   = 1,
            WriteRowMask  = 2,
            WriteBankMask = 3,
            BoundCtrl     = 4,
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT     = DppOpsKernel<
                std::tuple_element_t<DataT, TestParamsT>, // DataT
                std::tuple_element_t<CrossLaneOp, TestParamsT>, // CrossLaneOp
                std::tuple_element_t<WriteRowMask, TestParamsT>::value, // WriteRowMask
                std::tuple_element_t<WriteBankMask, TestParamsT>::value, // WriteBankMask
                std::tuple_element_t<BoundCtrl, TestParamsT>::value // BoundCtrl
                >;

            return std::make_shared<KernelT>();
        }
    };

    struct SwizzleOpsGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT       = 0,
            CrossLaneOp = 1
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = SwizzleOpsKernel<std::tuple_element_t<DataT, TestParamsT>, // DataT
                                   std::tuple_element_t<CrossLaneOp, TestParamsT> // CrossLaneOp
                                   >;

            return std::make_shared<KernelT>();
        }
    };

    struct PermuteOpsGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT       = 0,
            CrossLaneOp = 1
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = PermuteOpsKernel<std::tuple_element_t<DataT, TestParamsT>, // DataT
                                   std::tuple_element_t<CrossLaneOp, TestParamsT> // CrossLaneOp
                                   >;

            return std::make_shared<KernelT>();
        }
    };

    struct BlendOpsGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT       = 0,
            CrossLaneOp = 1
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = BlendOpsKernel<std::tuple_element_t<DataT, TestParamsT>, // DataT
                                 std::tuple_element_t<CrossLaneOp, TestParamsT> // CrossLaneOp
                                 >;

            return std::make_shared<KernelT>();
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_CROSS_LANE_OPS_HPP
