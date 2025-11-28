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

#ifndef ROCWMMA_UNIT_KERNEL_BASE_HPP
#define ROCWMMA_UNIT_KERNEL_BASE_HPP

#include <iostream>
#include <sstream>
#include <string>

#include "hip_device.hpp"
#include "unit_resource.hpp"

namespace rocwmma
{

    // Basic structure to hold runtime problem
    // parameters
    struct ProblemParams
    {
        std::pair<int64_t, int64_t> threadBlockSize;
        std::pair<int64_t, int64_t> problemSize;
        double                      param1;
        double                      param2;
    };

    // Typeless Kernel interface to use with testing harness.
    struct KernelI
    {
        KernelI()          = default;
        virtual ~KernelI() = default;

        virtual void          setup(ProblemParams const& problem)                 = 0;
        virtual void          validateResults()                                   = 0;
        virtual void          reportResults()                                     = 0;
        virtual void          tearDown()                                          = 0;
        virtual void          exec()                                              = 0;
        virtual std::ostream& printHeader(std::ostream& stream = std::cout) const = 0;
        virtual std::ostream& printKernel(std::ostream& stream = std::cout) const = 0;

        bool runFlag() const
        {
            return mRunFlag;
        }
        bool validationResult() const
        {
            return mValidationResult;
        }

    protected:
        static bool sHeaderPrinted;
        bool        mRunFlag          = true;
        bool        mValidationResult = false;
    };

    inline std::ostream& operator<<(std::ostream& stream, KernelI const& kernel)
    {
        kernel.printHeader(stream);
        kernel.printKernel(stream);
        return stream;
    }

    // Typed Unit test kernel that provides the basis for Unit tests.
    // This class provides common implementation code.
    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct UnitKernelBase : public KernelI
    {
    protected: // Types
        // Shared access to Test storage
        using DataStorage = UnitResource<DataT>;

        // Using Hip device backend
        using DeviceInfo = HipDevice;

        // Interface to device kernel
        using KernelFunc = void (*)(uint32_t, // M
                                    uint32_t, // N
                                    DataT const*, // In
                                    DataT*, // Out
                                    uint32_t, // ld
                                    DataT, // param1
                                    DataT); // param2

    protected:
        UnitKernelBase();
        virtual ~UnitKernelBase();

        // Kernels MUST provide the device kernel function.
        virtual void       setupImpl(typename DataStorage::ProblemSize const& size) = 0;
        virtual KernelFunc kernelImpl() const                                       = 0;
        virtual void       validateResultsImpl()                                    = 0;

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

    public:
        // KernelI interface fulfillment
        virtual void          setup(ProblemParams const& problem) override;
        virtual void          exec() override;
        virtual void          validateResults() override;
        virtual void          reportResults() override;
        virtual void          tearDown() override;
        virtual std::ostream& printHeader(std::ostream& stream = std::cout) const override;
        virtual std::ostream& printKernel(std::ostream& stream = std::cout) const override;

    protected:
        // Problem params for kernel
        uint32_t mTBlockX, mTBlockY;
        uint32_t mM, mN;
        uint32_t mLd;
        DataT    mParam1, mParam2;

        // Execution flow control
        double mMaxRelativeError;

        // Performance
        float64_t mTotalGFlops, mMeasuredTFlopsPerSec;
        float64_t mElapsedTimeMs;
        int32_t   mEfficiency;
    };

} // namespace rocwmma

#include "unit_kernel_base_impl.hpp"

#endif // ROCWMMA_UNIT_KERNEL_BASE_HPP
