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

#ifndef ROCWMMA_DETAIL_MAP_MATRIX_TO_DATA_OVERRIDE_HPP
#define ROCWMMA_DETAIL_MAP_MATRIX_TO_DATA_OVERRIDE_HPP

#include "device/map_matrix_to_data_override.hpp"
#include "helper_macros.hpp"
#include "unit_kernel_base.hpp"

namespace rocwmma
{

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct MapMatrixToDataOverrideKernel : public UnitKernelBase<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = UnitKernelBase<BlockM, BlockN, DataT, Layout>;

        template <uint32_t WaveSize, uint32_t ArchId>
        using TestGuard = FragSize_guard<BlockM, BlockN, DataT, Layout, WaveSize, ArchId>;

    protected:
        enum : int32_t
        {
            OVERRIDE_M,
            OVERRIDE_N

        } mOverride;

    public:
        MapMatrixToDataOverrideKernel(decltype(mOverride) overrideContext = OVERRIDE_M)
            : Base()
            , mOverride(overrideContext){};
        virtual ~MapMatrixToDataOverrideKernel() = default;

        void setupImpl(typename Base::DataStorage::ProblemSize const& probsize) final
        {
            auto& dataInstance = Base::DataStorage::instance();
            srand((unsigned)time(0));

            // Determine override param:
            // Base::mParam1 is stored as DataT, so we must make sure we
            // can store an accurate integer representation for override.
            // must be between [0, min(maxInt, M) - BlockM) for block rw.
            auto maxInt    = maxExactInteger<DataT>();
            using MaxIntT  = decltype(maxInt);
            auto blockSize = mOverride == OVERRIDE_M ? BlockM : BlockN;
            auto matrixDim = mOverride == OVERRIDE_M ? Base::mM : Base::mN;
            auto upperBound
                = std::max(static_cast<MaxIntT>(0),
                           std::min(maxInt, static_cast<MaxIntT>(matrixDim - blockSize)));

            // Modulate if non zero
            upperBound = upperBound ? rand() % upperBound : upperBound;

            // Finally, assign back to param1
            Base::mParam1 = static_cast<DataT>(static_cast<float32_t>(upperBound));

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

            // Allocate and cache device-initialized input data
            dataInstance->copyData(dataInstance->hostIn(), dataInstance->deviceIn(), sizeD);

            // Allocate host memory for reference calcs
            auto& hostInput  = dataInstance->hostIn();
            auto  hostResult = dataInstance->template allocHost<DataT>(sizeD);

            /// Initialize reference for validation

            // Setup addressing for DataLayout
            auto rowMjrOffset = [](std::pair<uint32_t, uint32_t> const& matrixCoord, uint32_t ld) {
                return get<0>(matrixCoord) * ld + get<1>(matrixCoord);
            };
            auto colMjrOffset = [](std::pair<uint32_t, uint32_t> const& matrixCoord, uint32_t ld) {
                return get<1>(matrixCoord) * ld + get<0>(matrixCoord);
            };
            auto ld          = std::is_same<Layout, row_major>::value ? Base::mN : Base::mM;
            auto arrayOffset = std::is_same<Layout, row_major>::value ? rowMjrOffset : colMjrOffset;

            // Scale from workgroup grid to wave grid
            auto waveGridDim = std::make_pair(
                Base::gridDim().x * (Base::blockDim().x / Base::DeviceInfo::instance()->warpSize()),
                Base::gridDim().y * Base::blockDim().y);

// Calculate the expected output from the inputs
#pragma omp parallel for
            for(auto row = 0u; row < get<0>(waveGridDim); row++)
            {
#pragma omp parallel for
                for(auto col = 0u; col < get<1>(waveGridDim); col++)
                {
                    // Setup read / write matrix coords for block
                    auto overrideVal = static_cast<uint32_t>(static_cast<float32_t>(Base::mParam1));
                    auto readBase
                        = std::make_pair(mOverride == OVERRIDE_M ? overrideVal : row * BlockM,
                                         mOverride == OVERRIDE_N ? overrideVal : col * BlockN);

                    auto writeBase = std::make_pair(row * BlockM, col * BlockN);

                    // Loop through entire read block and copy to dest block
                    for(auto rowOffset = 0u; rowOffset < BlockM; rowOffset++)
                    {
                        for(auto colOffset = 0u; colOffset < BlockN; colOffset++)
                        {
                            auto matrixRead  = readBase + std::make_pair(rowOffset, colOffset);
                            auto matrixWrite = writeBase + std::make_pair(rowOffset, colOffset);

                            *(hostResult.get() + arrayOffset(matrixWrite, ld))
                                = *(hostInput.get() + arrayOffset(matrixRead, ld));
                        }
                    }
                }
            }

            auto reference = dataInstance->template allocDevice<DataT>(sizeD);
            dataInstance->copyData(reference, hostResult, sizeD);

            double errorTolerance = 1.0;
            std::tie(Base::mValidationResult, Base::mMaxRelativeError)
                = compareEqualLaunchKernel<DataT, DataT, Layout, Layout>(
                    dataInstance->deviceOut().get(),
                    reference.get(),
                    Base::mM,
                    Base::mN,
                    errorTolerance);
        }

        bool checkQuirks() const final
        {
            auto waveSize   = Base::DeviceInfo::instance()->warpSize();
            auto deviceArch = Base::DeviceInfo::instance()->getGcnArch();

            // The test guard for this class requires 2 values at runtime.
            auto dispatchGuard = [waveSize, deviceArch]() {
                bool dispatchResult = false;

#define CASE_IMPL_ASSIGN2(WAVE_SIZE, ARCH_ID) \
    dispatchResult = TestGuard<WAVE_SIZE, ARCH_ID>::enable();

#define SWITCH_BODY_WAVE_SIZE(ARCH_ID) \
    ROCWMMA_SWITCH_BODY2_ARG2(         \
        waveSize, CASE_IMPL_ASSIGN2, HipDevice::Wave32, HipDevice::Wave64, ARCH_ID)

#define DISPATCH_GUARD_BODY                          \
    ROCWMMA_SWITCH_BODY9_ARG1(deviceArch,            \
                              SWITCH_BODY_WAVE_SIZE, \
                              HipDevice::GFX908,     \
                              HipDevice::GFX90A,     \
                              HipDevice::GFX942,     \
                              HipDevice::GFX950,     \
                              HipDevice::GFX1100,    \
                              HipDevice::GFX1101,    \
                              HipDevice::GFX1102,    \
                              HipDevice::GFX1200,    \
                              HipDevice::GFX1201)

                DISPATCH_GUARD_BODY

#undef CASE_IMPL_ASSIGN2
#undef SWITCH_BODY_WAVE_SIZE
#undef DISPATCH_GUARD_BODY

                return dispatchResult;
            };

            return Base::checkQuirks() && dispatchGuard();
        }
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct MapMatrixToDataOverrideMKernel final
        : public MapMatrixToDataOverrideKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = MapMatrixToDataOverrideKernel<BlockM, BlockN, DataT, Layout>;

    public:
        MapMatrixToDataOverrideMKernel()
            : Base(Base::OVERRIDE_M){};
        ~MapMatrixToDataOverrideMKernel() final = default;
        typename Base::KernelFunc kernelImpl() const final
        {
            return
                typename Base::KernelFunc(MapMatrixToDataOverrideM<BlockM, BlockN, DataT, Layout>);
        }
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct MapMatrixToDataOverrideNKernel final
        : public MapMatrixToDataOverrideKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = MapMatrixToDataOverrideKernel<BlockM, BlockN, DataT, Layout>;

    public:
        MapMatrixToDataOverrideNKernel()
            : Base(Base::OVERRIDE_N){};
        ~MapMatrixToDataOverrideNKernel() final = default;
        typename Base::KernelFunc kernelImpl() const final
        {
            return
                typename Base::KernelFunc(MapMatrixToDataOverrideN<BlockM, BlockN, DataT, Layout>);
        }
    };

    // This is the GeneratorImpl class
    template <template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
              class OverrideClass>
    struct MapMatrixToDataOverrideGenerator
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
                = OverrideClass<std::tuple_element_t<BlockM, TestParamsT>::value, // BlockM
                                std::tuple_element_t<BlockN, TestParamsT>::value, // BlockN
                                std::tuple_element_t<DataT, TestParamsT>, // DataT
                                std::tuple_element_t<Layout, TestParamsT> // Layout
                                >;

            return std::make_shared<KernelT>();
        }
    };

    using MapMatrixToDataOverrideMGenerator
        = MapMatrixToDataOverrideGenerator<MapMatrixToDataOverrideMKernel>;
    using MapMatrixToDataOverrideNGenerator
        = MapMatrixToDataOverrideGenerator<MapMatrixToDataOverrideNKernel>;

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_MAP_MATRIX_TO_DATA_OVERRIDE_HPP
