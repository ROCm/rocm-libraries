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

#ifndef ROCWMMA_KERNEL_BASE_IMPL_HPP
#define ROCWMMA_KERNEL_BASE_IMPL_HPP

#include <cmath>
#include <tuple>

#include <hip/hip_ext.h>
#include <hip/hip_runtime_api.h>

#include <gtest/gtest.h>

// Library includes
#include <rocwmma/internal/constants.hpp>
#include <rocwmma/internal/utils.hpp>

#include "common.hpp"
#include "gemm_kernel_base.hpp"
#include "performance.hpp"

#if ROCWMMA_VALIDATION_TESTS
#include "reference.hpp" // Vanilla CPU kernel
#endif // ROCWMMA_VALIDATION_TESTS

#if ROCWMMA_ROCBLAS_INTEGRATION
#include "rocblas_reference.hpp" // rocBLAS GPU kernel
#endif // ROCWMMA_ROCBLAS_INTEGRATION

namespace rocwmma
{

    // Using Cpu reference kernel if:
    // - Not using rocBLAS OR
    // - Using rocBLAS and it cannot solve the problem
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    constexpr bool GemmKernelBase<BlockM,
                                  BlockN,
                                  BlockK,
                                  InputT,
                                  OutputT,
                                  ComputeT,
                                  LayoutA,
                                  LayoutB,
                                  LayoutC,
                                  LayoutD>::mIsCpuRef
        = !(bool)ROCWMMA_ROCBLAS_INTEGRATION
          || ((bool)ROCWMMA_ROCBLAS_INTEGRATION
              && !quirks::rocblas_supported<InputT, OutputT, ComputeT>::value);

    // Prepare / run reference kernel if:
    // - Validation mode OR
    // - Benchmarking with rocBLAS and it can solve the problem
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    constexpr bool GemmKernelBase<BlockM,
                                  BlockN,
                                  BlockK,
                                  InputT,
                                  OutputT,
                                  ComputeT,
                                  LayoutA,
                                  LayoutB,
                                  LayoutC,
                                  LayoutD>::mRunRefFlag
        = (bool)ROCWMMA_VALIDATION_TESTS || mBenchRef;

    // Benchmark the reference if:
    // - Benchmarking with rocBLAS AND
    // - rocBLAS can solve the problem
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    constexpr bool GemmKernelBase<BlockM,
                                  BlockN,
                                  BlockK,
                                  InputT,
                                  OutputT,
                                  ComputeT,
                                  LayoutA,
                                  LayoutB,
                                  LayoutC,
                                  LayoutD>::mBenchRef
        = ((bool)ROCWMMA_BENCHMARK_TESTS && ROCWMMA_BENCHMARK_WITH_ROCBLAS
           && quirks::rocblas_supported<InputT, OutputT, ComputeT>::value);

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    GemmKernelBase<BlockM,
                   BlockN,
                   BlockK,
                   InputT,
                   OutputT,
                   ComputeT,
                   LayoutA,
                   LayoutB,
                   LayoutC,
                   LayoutD>::GemmKernelBase()
    {
        reset();
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    GemmKernelBase<BlockM,
                   BlockN,
                   BlockK,
                   InputT,
                   OutputT,
                   ComputeT,
                   LayoutA,
                   LayoutB,
                   LayoutC,
                   LayoutD>::~GemmKernelBase()
    {
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    uint32_t GemmKernelBase<BlockM,
                            BlockN,
                            BlockK,
                            InputT,
                            OutputT,
                            ComputeT,
                            LayoutA,
                            LayoutB,
                            LayoutC,
                            LayoutD>::ldsUsage() const
    {
        return 0;
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    dim3 GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::gridDim() const
    {
        return dim3(ceilDiv(mM, BlockM * mTBlockX / DeviceInfo::instance()->warpSize()),
                    ceilDiv(mN, BlockN * mTBlockY));
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    dim3 GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::blockDim() const
    {
        return dim3(mTBlockX, mTBlockY);
    }

    // Kernel run checks. Virtual as different GEMM kernels have different requirements
    // True = run test
    // False = skip test
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    bool GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::checkDevice() const
    {
        auto& deviceInfo = DeviceInfo::instance();
        auto  deviceArch = deviceInfo->getGcnArch();

        // No unsupported devices
        return !(deviceArch == DeviceInfo::UNSUPPORTED_ARCH);
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    bool GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::checkSizes() const
    {
        // gridDim() takes the upper bound of block coverage.
        // In case of uneven division, this might put us out of bounds.
        // Forfeit the run because there is no tail for cleanup of remainders.
        auto tileSize = std::make_pair(BlockM * mTBlockX / DeviceInfo::instance()->warpSize(),
                                       BlockN * mTBlockY);
        auto gridDims = gridDim();
        return (gridDims.x * std::get<0>(tileSize) == mM)
               && (gridDims.y * std::get<1>(tileSize) == mN) && (mK % BlockK == 0) && BlockK <= mK;
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    bool GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::checkLds() const
    {
        return ldsUsage() <= DeviceInfo::instance()->sharedMemSize();
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    bool GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::checkQuirks() const
    {
        // Historically, there have been some bugs that are elicited under certain conditions
        // and produce 'quirky' failures that are beyond WMMA's control.
        // E.g. Previous compiler issue with h16 unsanitized register packing.
        // We can choose to ignore the quirks and focus on WMMA specific failures here.
        return true;
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    void GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::reset()
    {
        mTBlockX = mTBlockY = 0u;
        mM = mN = mK = 0u;
        mLda = mLdb = mLdc = mLdd = 0u;
        mAlpha = mBeta = static_cast<ComputeT>(0u);

        mColdRuns = (bool)(ROCWMMA_VALIDATION_TESTS) ? 0u : 1u;
        mHotRuns  = (bool)(ROCWMMA_VALIDATION_TESTS) ? 1u : 5u;

        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;

        mElapsedTimeMs = mTotalGFlops = mMeasuredTFlopsPerSec = 0.0;
        mEfficiency                                           = -1;

        mMeasuredTFlopsPerSec = 0.0;
        mRefEfficiency        = -1;
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    HipResource* GemmKernelBase<BlockM,
                                BlockN,
                                BlockK,
                                InputT,
                                OutputT,
                                ComputeT,
                                LayoutA,
                                LayoutB,
                                LayoutC,
                                LayoutD>::getResource() const
    {
        return DataStorage::instance().get();
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    std::ostream& GemmKernelBase<BlockM,
                                 BlockN,
                                 BlockK,
                                 InputT,
                                 OutputT,
                                 ComputeT,
                                 LayoutA,
                                 LayoutB,
                                 LayoutC,
                                 LayoutD>::printHeader(std::ostream& stream /* = std::cout */) const
    {
        return stream << "TBlkX, TBlkY, "
                      << "BlkM, BlkN, BlkK, "
                      << "MatM, MatN, MatK, "
                      << "alpha, lda, ldb, beta, ldc, ldd, "
                      << "LytA_LytB_LytC_LytD, "
                      << "Ti_To_Tc, "
                      << "elapsedMs, "
                      << "Problem Size(GFlops), "
                      << "TFlops/s, "
                      << "Efficiency(%), "
                      << (mBenchRef ? "rocBLAS TFlops/s(%), rocBLAS Efficiency(%), " : "")
                      << "Result" << std::endl;
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    std::ostream& GemmKernelBase<BlockM,
                                 BlockN,
                                 BlockK,
                                 InputT,
                                 OutputT,
                                 ComputeT,
                                 LayoutA,
                                 LayoutB,
                                 LayoutC,
                                 LayoutD>::printKernel(std::ostream& stream) const
    {
        stream << mTBlockX << ", " << mTBlockY << ", " << BlockM << ", " << BlockN << ", " << BlockK
               << ", " << mM << ", " << mN << ", " << mK << ", " << mAlpha << ", " << mLda << ", "
               << mLdb << ", " << mBeta << ", " << mLdc << ", " << mLdd << ", "
               << dataTypeToString<LayoutA>() << "_" << dataTypeToString<LayoutB>() << "_"
               << dataTypeToString<LayoutC>() << "_" << dataTypeToString<LayoutD>() << ", "
               << dataTypeToString<InputT>() << "_" << dataTypeToString<OutputT>() << "_"
               << dataTypeToString<ComputeT>() << ", ";

        if(!mRunFlag)
        {
            stream << "n/a"
                   << ", "
                   << "n/a"
                   << ", "
                   << "n/a"
                   << ", "
                   << "n/a"
                   << ", " << (mBenchRef ? "n/a, n/a, " : "") << "SKIPPED" << std::endl;
        }
        else
        {

            stream << mElapsedTimeMs << ", " << mTotalGFlops << ", " << mMeasuredTFlopsPerSec
                   << ", " << mEfficiency << ", "
                   << (mBenchRef ? (std::to_string(mRefMeasuredTFlopsPerSec) + ", "
                                    + std::to_string(mRefEfficiency) + ", ")
                                 : "")
                   << ((bool)ROCWMMA_VALIDATION_TESTS ? (mValidationResult ? "PASSED" : "FAILED")
                                                      : "BENCH")
                   << std::endl;
        }

        return stream;
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    void GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::setup(ProblemParams const& problem)
    {
        // Reset the flags in case of multiple runs
        mRunFlag          = true;
        mValidationResult = false;

        // Format incoming problem parameters
        std::tie(mTBlockX, mTBlockY)
            = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.threadBlockSize)),
                       static_cast<uint32_t const&>(std::get<1>(problem.threadBlockSize)));
        std::tie(mM, mN, mK)
            = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.problemSize)),
                       static_cast<uint32_t const&>(std::get<1>(problem.problemSize)),
                       static_cast<uint32_t const&>(std::get<2>(problem.problemSize)));
        std::tie(mAlpha, mBeta) = std::tie((ComputeT const&)static_cast<ComputeT>(problem.alpha),
                                           (ComputeT const&)static_cast<ComputeT>(problem.beta));
        std::tie(mLda, mLdb, mLdc, mLdd)
            = std::tie((std::is_same<LayoutA, row_major>::value ? mK : mM),
                       (std::is_same<LayoutB, row_major>::value ? mN : mK),
                       (std::is_same<LayoutC, row_major>::value ? mN : mM),
                       (std::is_same<LayoutC, row_major>::value ? mN : mM));

        // Clear the kernel to run
        mRunFlag &= checkDevice();
        mRunFlag &= checkSizes();
        mRunFlag &= checkLds();
        mRunFlag &= checkQuirks();

        if(mRunFlag)
        {
            auto& dataInstance = DataStorage::instance();

            // Initialize matrix storage
            dataInstance->resizeStorage(problem.problemSize);

            // Initialize matrix data on device
            MatrixUtil<LayoutA>::fillLaunchKernel(dataInstance->deviceA().get(), mM, mK);
            MatrixUtil<LayoutB>::fillLaunchKernel(dataInstance->deviceB().get(), mK, mN);
            MatrixUtil<LayoutC>::fillLaunchKernel(dataInstance->deviceC().get(), mM, mN);
            MatrixUtil<LayoutD>::fillValLaunchKernel(dataInstance->deviceD().get(),
                                                     mM,
                                                     mN,
                                                     std::numeric_limits<OutputT>::signaling_NaN());

            // Initialize the host data if we are to use Cpu validation.
            if constexpr(mRunRefFlag && mIsCpuRef)
            {
                dataInstance->copyDeviceToHostAll();
            }
        }
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    void GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::exec()
    {
        if(mRunFlag)
        {
            ///
            /// Run ROCWMMA kernel
            ///

            auto rocwmmaKernel = [this]() {
                auto& dataInstance = DataStorage::instance();
                hipExtLaunchKernelGGL((this->kernelImpl()), // Kernel to launch
                                      (this->gridDim()), // Wg grid size
                                      (this->blockDim()), // Thread block size
                                      (this->ldsUsage()), // sharedMemBytes
                                      0, // stream
                                      nullptr, // Event start
                                      nullptr, // event stop
                                      0, // flags
                                      this->mM, // M
                                      this->mN, // N
                                      this->mK, // K
                                      dataInstance->deviceA().get(), // A*
                                      dataInstance->deviceB().get(), // B*
                                      dataInstance->deviceC().get(), // C*
                                      dataInstance->deviceD().get(), // D*
                                      this->mLda, // lda
                                      this->mLdb, // ldb
                                      this->mLdc, // ldc
                                      this->mLdd, // ldd
                                      this->mAlpha, // alpha
                                      this->mBeta); // beta
            };

            hipEvent_t startEvent, stopEvent;
            CHECK_HIP_ERROR(hipEventCreate(&startEvent));
            CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

            // Cold runs for frequency warm-up
            for(uint32_t i = 0; i < mColdRuns; ++i)
            {
                rocwmmaKernel();
            }

            // Finish cold runs
            CHECK_HIP_ERROR(hipEventRecord(stopEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

            // Use the hot runs for timing. Ensure sequential execution.
            mElapsedTimeMs = 0.0;
            for(uint32_t i = 0; i < mHotRuns; ++i)
            {
                CHECK_HIP_ERROR(hipEventRecord(startEvent));
                rocwmmaKernel();
                CHECK_HIP_ERROR(hipEventRecord(stopEvent));
                CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
                auto timeMs = 0.0f;
                CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));
                mElapsedTimeMs += timeMs;
            }

            // Calculate efficiency
            auto& deviceInfo = DeviceInfo::instance();

            auto devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<InputT>();
            mTotalGFlops                = calculateGFlops(mM, mN, mK);
            mMeasuredTFlopsPerSec       = calculateTFlopsPerSec(mM, mN, mK, mElapsedTimeMs)
                                    * static_cast<float64_t>(mHotRuns);

            mEfficiency = round(mMeasuredTFlopsPerSec / devicePeakGFlopsPerSec * 100000.0);

            CHECK_HIP_ERROR(hipEventDestroy(startEvent));
            CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

            if constexpr(mRunRefFlag)
            {
                // Reference kernel selection
                std::function<void()> refKernel;

                if constexpr(mIsCpuRef)
                {

#if ROCWMMA_VALIDATION_TESTS

                    // Define fallback CPU kernel
                    auto cpuKernel = [this]() {
                        auto& dataInstance = DataStorage::instance();
                        gemm_CPU<InputT, OutputT, ComputeT, LayoutA, LayoutB, LayoutC, LayoutD>(
                            this->mM,
                            this->mN,
                            this->mK,
                            dataInstance->hostA().get(),
                            dataInstance->hostB().get(),
                            dataInstance->hostC().get(),
                            dataInstance->hostD().get(),
                            this->mAlpha,
                            this->mBeta);
                    };

                    // Assign cpu func
                    refKernel = cpuKernel;

#endif // ROCWMMA_VALIDATION_TESTS
                }
                else
                {

#if ROCWMMA_ROCBLAS_INTEGRATION

                    auto rocBlasKernel = [this]() {
                        // Create a rocBLAS handle to be used with rocBLAS API
                        rocblas_handle handle;
                        CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

                        auto& dataInstance = DataStorage::instance();

                        static_assert((!std::is_same_v<InputT, float8_t>
                                       && !std::is_same_v<InputT, bfloat8_t>)
                                          || std::is_same_v<ComputeT, float32_t>,
                                      "f8 types must have f32 compute type");

                        CHECK_ROCBLAS_ERROR(
                            dispatch_rocBLAS(handle,
                                             rocblas_layout<LayoutA>::operation(), // opA
                                             rocblas_layout<LayoutB>::operation(), // opB
                                             this->mM, // M
                                             this->mN, // N
                                             this->mK, // K
                                             &(this->mAlpha), // alpha,
                                             dataInstance->deviceA().get(), // A*,
                                             rocblas_types<InputT>::type(), // a_type
                                             this->mLda, // lda
                                             dataInstance->deviceB().get(), // B*,
                                             rocblas_types<InputT>::type(), // b_type
                                             this->mLdb, // ldb
                                             &(this->mBeta), // beta
                                             dataInstance->deviceC().get(), // C*
                                             rocblas_types<OutputT>::type(), // c_type
                                             this->mM, // ldc (col major output only)
                                             dataInstance->deviceD().get(), // D*
                                             rocblas_types<OutputT>::type(), // d_type
                                             this->mM, // ldd (col major output only)
                                             rocblas_types<ComputeT>::type(), // compute_type
                                             rocblas_gemm_algo_standard, // algo
                                             0, // solution_index
                                             0)); // flags

                        rocblas_destroy_handle(handle);
                    };

                    // Assign cpu func
                    refKernel = rocBlasKernel;

#endif // ROCWMMA_ROCBLAS_INTEGRATION
                }

                // Sanity check that a reference was selected
                if(!refKernel)
                {
                    std::cout << "No ref kernel\n";
                    return;
                }

                // Prepare inputs for the reference kernel
                auto& dataInstance = DataStorage::instance();

                // rocBLAS ref
                auto rocWMMACacheD = DataStorage::template allocDevice<OutputT>(0);
                if constexpr(!mIsCpuRef)
                {
                    // A, B, C & D are cached on on device pointers from the rocWMMA run.
                    // Need to cache rocWMMA D result device memory, then re-initialize
                    // C and D as necessary for the reference device run.

                    // Cache rocWMMA result on device only if we are validating
                    if constexpr(!mBenchRef)
                    {
                        dataInstance->template reallocDevice<OutputT>(rocWMMACacheD, mM * mN);
                        dataInstance->copyData(rocWMMACacheD, dataInstance->deviceD(), mM * mN);
                    }

                    // rocBLAS matrix C is always in col_major, so adjust it if needed
                    if(!std::is_same<LayoutC, col_major>::value)
                    {
                        MatrixUtil<col_major>::fillLaunchKernel(
                            dataInstance->deviceC().get(), mM, mN);
                    }

                    // Reset device D with NaN
                    MatrixUtil<LayoutD>::fillValLaunchKernel(
                        dataInstance->deviceD().get(),
                        mM,
                        mN,
                        std::numeric_limits<OutputT>::signaling_NaN());
                }

                hipEvent_t startEvent, stopEvent;
                CHECK_HIP_ERROR(hipEventCreate(&startEvent));
                CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

                // Cold runs for frequency warm-up
                for(uint32_t i = 0; i < mColdRuns; ++i)
                {
                    refKernel();
                }

                // Finish cold runs
                CHECK_HIP_ERROR(hipEventRecord(stopEvent));
                CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

                // Hot runs for timing
                auto elapsedTimeMs = 0.0;
                for(uint32_t i = 0; i < mHotRuns; ++i)
                {
                    CHECK_HIP_ERROR(hipEventRecord(startEvent));
                    refKernel();
                    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
                    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
                    auto timeMs = 0.0f;
                    CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));
                    elapsedTimeMs += timeMs;
                }

                CHECK_HIP_ERROR(hipEventDestroy(startEvent));
                CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

                // Calculate reference efficiency
                if constexpr(mBenchRef)
                {
                    auto& deviceInfo             = DeviceInfo::instance();
                    auto  devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<InputT>();

                    auto measuredTFlopsPerSec = calculateTFlopsPerSec(mM, mN, mK, elapsedTimeMs)
                                                * static_cast<float64_t>(mHotRuns);

                    mRefMeasuredTFlopsPerSec = measuredTFlopsPerSec;
                    mRefEfficiency
                        = round(measuredTFlopsPerSec / devicePeakGFlopsPerSec * 100000.0);
                }

                // Prepare data for validation
                if constexpr((bool)ROCWMMA_VALIDATION_TESTS)
                {
                    if constexpr(mIsCpuRef)
                    {
                        // A, B, C & D from rocWMMA run are cached on device pointers.
                        // A, B, C & D from reference are cached on host pointers.
                        // Copy the reference host D result to C device pointer so we
                        // can validate the reference (device C) vs rocWMMA (device D).
                        dataInstance->copyData(
                            dataInstance->deviceC(), dataInstance->hostD(), mM * mN);
                    }
                    else
                    {
                        // A, B, C & D from reference run are cached on device pointers.
                        // D from rocWMMA is cached in local device pointer.
                        // Copy the rocWMMA local result to C device pointer so we can
                        // validate the reference (device D) vs rocWMMA (device C).
                        dataInstance->copyData(dataInstance->deviceC(), rocWMMACacheD, mM * mN);
                    }
                }
            }
        }
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    void GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::validateResults()
    {

        if(mRunFlag && (bool)ROCWMMA_VALIDATION_TESTS)
        {
            // If CPU reference, result layout is LayoutD, otherwise rocBLAS ref is always in col_major;
            using DeviceRefLayout = typename std::conditional_t<mIsCpuRef, LayoutD, col_major>;

            auto& dataInstance = DataStorage::instance();

            // If CPU ref, the rocWMMA result is in device D, otherwise device C
            auto* rocWMMAResult
                = mIsCpuRef ? dataInstance->deviceD().get() : dataInstance->deviceC().get();

            // If CPU ref, the reference result is in device C, otherwise device D
            auto* refResult
                = mIsCpuRef ? dataInstance->deviceC().get() : dataInstance->deviceD().get();

            // Give more error tolerance to ComputeT = fp16,
            // due to MFMA output is always fp32. We downcast the MFMA result to fp16, which
            // will introduce an error compared to native fp16 MAC. The tolerance would be a function
            // of max / min values and number of operations propagating the error.
            // Note that integer values between [-2048, 2048 ] are exactly representable by fp16,
            // and significant rounding errors occur thereafter to the nearest multiple of 2.
            // The input generator for GEMM uses integer values within a certain range, therefore
            // FMA operations will be very prone to significant errors.
            double errorTolerance = sizeof(ComputeT) < sizeof(float32_t) ? 100.0 : 10.0;

            std::tie(mValidationResult, mMaxRelativeError)
                = compareEqualLaunchKernel<OutputT, OutputT, LayoutD, DeviceRefLayout>(
                    rocWMMAResult, refResult, mM, mN, errorTolerance);

            EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;
        }
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    void GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::reportResults(std::ostream& stream,
                                                bool          omitHeader,
                                                bool          omitSkipped,
                                                bool          omitFailed,
                                                bool          omitPassed)
    {
        // Print header to std::cout
        if(!omitHeader)
        {
            printHeader(stream);
        }

        // Conditionally print kernel outputs
        if((mRunFlag || !omitSkipped) && (mValidationResult || !omitFailed)
           && (!mValidationResult || !omitPassed))
        {
            printKernel(stream);
        }
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    void GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::tearDown()
    {
    }

} // namespace rocwmma

#endif // ROCWMMA_KERNEL_BASE_IMPL_HPP
