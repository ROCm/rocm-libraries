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

#ifndef ROCWMMA_KERNEL_BASE_HPP
#define ROCWMMA_KERNEL_BASE_HPP

#include <iostream>
#include <sstream>
#include <string>

#include "gemm_resource.hpp"
#include "hip_device.hpp"

namespace rocwmma
{

    // Basic structure to hold runtime problem
    // parameters
    struct ProblemParams
    {
        std::pair<int64_t, int64_t>           threadBlockSize;
        std::tuple<int64_t, int64_t, int64_t> problemSize;
        double                                alpha;
        double                                beta;
    };

    // Typeless Kernel interface to use with testing harness.
    struct KernelI
    {
        KernelI() {}
        virtual ~KernelI(){};

        virtual void setup(ProblemParams const& problem) = 0;
        virtual void exec()                              = 0;
        virtual void validateResults()                   = 0;
        virtual void reportResults(std::ostream& stream,
                                   bool          omitHeader,
                                   bool          omitSkipped,
                                   bool          omitFailed,
                                   bool          omitPassed)
            = 0;
        virtual void          tearDown()                              = 0;
        virtual HipResource*  getResource() const                     = 0;
        virtual std::ostream& printHeader(std::ostream& stream) const = 0;
        virtual std::ostream& printKernel(std::ostream& stream) const = 0;

        static bool sHeaderPrinted;
    };

    inline std::ostream& operator<<(std::ostream& stream, KernelI const& kernel)
    {
        kernel.printHeader(stream);
        kernel.printKernel(stream);
        return stream;
    }

    // Typed GEMM kernel that provides the basis for GEMM tests.
    // This class provides common implementation code.
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD = LayoutC>
    struct GemmKernelBase : public KernelI
    {
    protected: // Types
        // Shared access to Gemm storage
        using DataStorage = GemmResource<InputT, OutputT>;

        // Using Hip device backend
        using DeviceInfo = HipDevice;

        // Interface to device kernel
        using KernelFunc = void (*)(uint32_t, // M
                                    uint32_t, // N
                                    uint32_t, // K
                                    InputT const*, // A
                                    InputT const*, // B
                                    OutputT const*, // C
                                    OutputT*, // D
                                    uint32_t, // lda
                                    uint32_t, // ldb
                                    uint32_t, // ldc
                                    uint32_t, // ldd
                                    ComputeT, // Alpha
                                    ComputeT); // Beta

    protected:
        GemmKernelBase();
        virtual ~GemmKernelBase();

        // Kernels MUST provide the device kernel function.
        virtual KernelFunc kernelImpl() const = 0;

        // Launch parameters.
        // Base calculations for grid and block dimensions
        // assume one output block per wave.
        virtual uint32_t ldsUsage() const;
        virtual dim3     gridDim() const;
        virtual dim3     blockDim() const;

        // Kernel run checks.
        // True = run test
        // False = skip test
        virtual bool checkDevice() const;
        virtual bool checkSizes() const;
        virtual bool checkLds() const;
        virtual bool checkQuirks() const;

        // Reset all members to default values
        virtual void reset();

        // Helper function to dispatch kernel guards
        // with runtime TBlockX, TBlockY, WaveSize and Device Arch
        template <template <uint32_t, uint32_t, uint32_t, uint32_t> class TestGuard>
        bool dispatchGuard() const;

        template <template <uint32_t, uint32_t, uint32_t, uint32_t> class KernelClass>
        KernelFunc dispatchKernelFunc() const;

    public:
        // KernelI interface fulfillment
        virtual void          setup(ProblemParams const& problem) override;
        virtual void          exec() override;
        virtual void          validateResults() override;
        virtual void          reportResults(std::ostream& stream,
                                            bool          omitHeader,
                                            bool          omitSkipped,
                                            bool          omitFailed,
                                            bool          omitPassed) override;
        virtual void          tearDown() override;
        virtual HipResource*  getResource() const override;
        virtual std::ostream& printHeader(std::ostream& stream) const override;
        virtual std::ostream& printKernel(std::ostream& stream) const override;

    protected:
        // Problem params for kernel
        uint32_t mTBlockX, mTBlockY;
        uint32_t mM, mN, mK;
        uint32_t mLda, mLdb, mLdc, mLdd;
        ComputeT mAlpha, mBeta;

        // Execution flow control
        uint32_t mColdRuns;
        uint32_t mHotRuns;
        bool     mRunFlag          = true;
        bool     mValidationResult = false;
        double   mMaxRelativeError;

        // Performance
        float64_t mElapsedTimeMs, mTotalGFlops, mMeasuredTFlopsPerSec;
        int32_t   mEfficiency;

        // Reference
        float64_t         mRefMeasuredTFlopsPerSec;
        int32_t           mRefEfficiency;
        static const bool mIsCpuRef;
        static const bool mRunRefFlag;
        static const bool mBenchRef;
    };

} // namespace rocwmma

#include "gemm_kernel_base_dispatch_impl.hpp"

#endif // ROCWMMA_KERNEL_BASE_HPP
