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

#ifndef DLRM_KERNEL_BASE_HPP
#define DLRM_KERNEL_BASE_HPP

#include <iostream>
#include <sstream>
#include <string>

#include <rocwmma/internal/constants.hpp>

#include "common.hpp"
#include "dlrm_resource.hpp"
#include "hip_device.hpp"

namespace rocwmma
{

    // Training pass direction
    enum class DlrmDirection_t : bool
    {
        Forward,
        Backward
    };

    // Basic structure to hold runtime problem
    // parameters
    struct ProblemParams
    {
        std::pair<int64_t, int64_t>           threadBlockSize;
        std::tuple<int64_t, int64_t, int64_t> problemSize;
        DlrmDirection_t                       passDirection;
    };

    // Typeless Kernel interface to use with testing harness.
    struct KernelI
    {
        KernelI() {}
        virtual ~KernelI(){};

        virtual void          setup(ProblemParams const& problem)                 = 0;
        virtual void          exec()                                              = 0;
        virtual void          validateResults()                                   = 0;
        virtual void          reportResults()                                     = 0;
        virtual void          tearDown()                                          = 0;
        virtual HipResource*  getResource()                                       = 0;
        virtual std::ostream& printHeader(std::ostream& stream = std::cout) const = 0;
        virtual std::ostream& printKernel(std::ostream& stream = std::cout) const = 0;

        static bool sHeaderPrinted;
    };

    inline std::ostream& operator<<(std::ostream& stream, KernelI const& kernel)
    {
        kernel.printHeader(stream);
        kernel.printKernel(stream);
        return stream;
    }

    // Typed DLRM kernel that provides the basis for DLRM tests.
    // This class provides common implementation code.
    template <uint32_t TileSize, typename DataT>
    struct DlrmKernelBase : public KernelI
    {
    protected: // Types
        // Shared access to DLRM storage
        using DataStorage = DlrmResource<DataT>;
        // Using Hip device backend
        using DeviceInfo = HipDevice;

        // Interface to forward device kernel
        using KernelFwdFunc = void (*)(const DataT* __restrict, // input
                                       DataT* __restrict, // output
                                       float*, // acc
                                       uint32_t, // m
                                       uint32_t, // k
                                       uint32_t, // b
                                       uint32_t, // inputBatchOffset
                                       uint32_t, // outputBatchOffset
                                       uint32_t); // accBatchOffset

        // Interface to backwards device kernels
        using KernelBwdFunc = void (*)(const DataT* __restrict, // input
                                       const DataT* __restrict, // upstreamGrad
                                       DataT* __restrict, // grad
                                       DataT* __restrict, // bottomMlpGrad
                                       DataT* __restrict, // acc
                                       uint32_t, // m
                                       uint32_t, // k
                                       uint32_t, // b
                                       uint32_t, // inputBatchOffset
                                       uint32_t, // upstreamBatchOffset
                                       uint32_t); // accBatchOffset

        using KernelTrilFunc = void (*)(const DataT* __restrict, // upstreamGrad
                                        DataT* __restrict, // acc
                                        uint32_t, // m
                                        uint32_t, // k
                                        uint32_t, // b
                                        uint32_t, // upstreamBatchOffset
                                        uint32_t); // accBatchOffset

    protected:
        DlrmKernelBase();
        virtual ~DlrmKernelBase();

        // Kernels MUST provide the device kernel function.
        virtual KernelFwdFunc  kernelFwdImpl() const  = 0;
        virtual KernelBwdFunc  kernelBwdImpl() const  = 0;
        virtual KernelTrilFunc kernelTrilImpl() const = 0;

        // Kernel launch parameters
        virtual uint32_t ldsUsage() const;
        virtual dim3     gridDim() const;
        virtual dim3     blockDim() const;

        // Kernel run checks.
        // True = run test
        // False = skip test
        virtual bool checkDevice() const;
        virtual bool checkSizes() const;
        virtual bool checkLds() const;

        // Reset all members to default values
        virtual void reset();

    public:
        // KernelI interface fulfillment
        virtual void          setup(ProblemParams const& problem) override;
        virtual void          exec() override;
        virtual void          validateResults() override;
        virtual void          reportResults() override;
        virtual void          tearDown() override;
        virtual HipResource*  getResource() override;
        virtual std::ostream& printHeader(std::ostream& stream = std::cout) const override;
        virtual std::ostream& printKernel(std::ostream& stream = std::cout) const override;

    protected:
        // Problem params for kernel
        uint32_t mTBlockX, mTBlockY;
        uint32_t mM, mK, mB;

        // Padded problem params
        uint32_t mMPadded, mKPadded;

        // Execution flow control
        uint32_t mRepeats;
        bool     mRunFlag          = true;
        bool     mValidationResult = false;
        double   mMaxRelativeError;

        DlrmDirection_t passDirection = DlrmDirection_t::Forward;

        // Performance
        float64_t mTotalGFlops, mMeasuredTFlopsPerSec;
        float64_t mElapsedTimeMs;
        int32_t   mEfficiency;
    };

} // namespace rocwmma

#include "dlrm_kernel_base_impl.hpp"

#endif // DLRM_KERNEL_BASE_HPP
