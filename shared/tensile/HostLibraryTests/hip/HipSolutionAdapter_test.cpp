/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>

#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Utils.hpp>

#include "TestData.hpp"
#include "TestUtils.hpp"

#include <random>

using namespace Tensile;

TEST(HipSolutionAdapterTest, BetaOnlyKernel_Zero)
{
    TensorDescriptor desc(DataType::Float, {43, 13, 65}, {1, 50, 50 * 16});

    float* d_d = nullptr;
    float* c_d = nullptr;

    HIP_CHECK_EXC(hipMalloc(&c_d, desc.totalAllocatedBytes()));
    HIP_CHECK_EXC(hipMemset(c_d, 0x33, desc.totalAllocatedBytes()));

    HIP_CHECK_EXC(hipMalloc(&d_d, desc.totalAllocatedBytes()));
    HIP_CHECK_EXC(hipMemset(d_d, 0x22, desc.totalAllocatedBytes()));

    KernelInvocation k;

    k.kernelName      = "Cijk_S";
    k.workGroupSize.x = 256;
    k.workGroupSize.y = 1;
    k.workGroupSize.z = 1;

    k.numWorkGroups.x = CeilDivide(desc.totalLogicalElements(), k.workGroupSize.x);
    k.numWorkGroups.y = 1;
    k.numWorkGroups.z = 1;

    k.numWorkItems.x = k.workGroupSize.x * k.numWorkGroups.x;
    k.numWorkItems.y = k.workGroupSize.y * k.numWorkGroups.y;
    k.numWorkItems.z = k.workGroupSize.z * k.numWorkGroups.z;

    k.args.append<float*>("D", d_d);
    k.args.append<float const*>("C", c_d);
    k.args.append<unsigned int>("strideD1", desc.strides()[1]);
    k.args.append<unsigned int>("strideD2", desc.strides()[2]);
    k.args.append<unsigned int>("strideC1", desc.strides()[1]);
    k.args.append<unsigned int>("strideC2", desc.strides()[2]);
    k.args.append<unsigned int>("size0", desc.sizes()[0]);
    k.args.append<unsigned int>("size1", desc.sizes()[1]);
    k.args.append<unsigned int>("size2", desc.sizes()[2]);
    k.args.append<float>("beta", 0.0f);

    hip::SolutionAdapter adapter(false);
    adapter.loadEmbeddedCodeObjects("kernels_lite_mixed");

    HIP_CHECK_EXC(adapter.launchKernel(k));

    std::vector<float> d_h(desc.totalAllocatedElements());

    HIP_CHECK_EXC(hipMemcpy(d_h.data(), d_d, desc.totalAllocatedBytes(), hipMemcpyDeviceToHost));

    HIP_CHECK_EXC(hipFree(c_d));
    HIP_CHECK_EXC(hipFree(d_d));

    std::vector<float> d_ref_h(desc.totalAllocatedElements());

    memset(d_ref_h.data(), 0x22, desc.totalAllocatedBytes());
    for(int k = 0; k < desc.sizes()[2]; k++)
        for(int j = 0; j < desc.sizes()[1]; j++)
            for(int i = 0; i < desc.sizes()[0]; i++)
            {
                d_ref_h[desc.index(i, j, k)] = 0.0f;
            }

    for(int i = 0; i < d_ref_h.size(); i++)
    {
        ASSERT_FLOAT_EQ(d_h[i], d_ref_h[i]) << i;
    }
}

TEST(HipSolutionAdapterTest, BetaOnlyKernel_Nonzero)
{
    TensorDescriptor desc(DataType::Float, {43, 13, 65}, {1, 50, 50 * 16});

    float beta = 1.9f;

    float* c_d = nullptr;
    float* d_d = nullptr;

    HIP_CHECK_EXC(hipMalloc(&c_d, desc.totalAllocatedBytes()));
    HIP_CHECK_EXC(hipMalloc(&d_d, desc.totalAllocatedBytes()));

    HIP_CHECK_EXC(hipMemset(c_d, 0x22, desc.totalAllocatedBytes()));
    HIP_CHECK_EXC(hipMemset(d_d, 0x33, desc.totalAllocatedBytes()));

    float c_initial_value;
    HIP_CHECK_EXC(hipMemcpy(&c_initial_value, c_d, sizeof(float), hipMemcpyDeviceToHost));
    float d_final_value = c_initial_value * beta;

    KernelInvocation k;

    k.kernelName      = "Cijk_S";
    k.workGroupSize.x = 256;
    k.workGroupSize.y = 1;
    k.workGroupSize.z = 1;

    k.numWorkGroups.x = CeilDivide(desc.totalLogicalElements(), k.workGroupSize.x);
    k.numWorkGroups.y = 1;
    k.numWorkGroups.z = 1;

    k.numWorkItems.x = k.workGroupSize.x * k.numWorkGroups.x;
    k.numWorkItems.y = k.workGroupSize.y * k.numWorkGroups.y;
    k.numWorkItems.z = k.workGroupSize.z * k.numWorkGroups.z;

    k.args.append<float*>("D", d_d);
    k.args.append<float const*>("C", c_d);
    k.args.append<unsigned int>("strideD1", desc.strides()[1]);
    k.args.append<unsigned int>("strideD2", desc.strides()[2]);
    k.args.append<unsigned int>("strideC1", desc.strides()[1]);
    k.args.append<unsigned int>("strideC2", desc.strides()[2]);
    k.args.append<unsigned int>("size0", desc.sizes()[0]);
    k.args.append<unsigned int>("size1", desc.sizes()[1]);
    k.args.append<unsigned int>("size2", desc.sizes()[2]);
    k.args.append<float>("beta", beta);

    hip::SolutionAdapter adapter(false);
    adapter.loadEmbeddedCodeObjects("kernels_lite_mixed");

    HIP_CHECK_EXC(adapter.launchKernel(k));

    std::vector<float> d_h(desc.totalAllocatedElements());

    HIP_CHECK_EXC(hipMemcpy(d_h.data(), d_d, desc.totalAllocatedBytes(), hipMemcpyDeviceToHost));

    std::vector<float> d_ref_h(desc.totalAllocatedElements());

    memset(d_ref_h.data(), 0x33, desc.totalAllocatedBytes());
    for(int k = 0; k < desc.sizes()[2]; k++)
        for(int j = 0; j < desc.sizes()[1]; j++)
            for(int i = 0; i < desc.sizes()[0]; i++)
            {
                d_ref_h[desc.index(i, j, k)] = d_final_value;
            }

    for(int i = 0; i < d_ref_h.size(); i++)
    {
        ASSERT_FLOAT_EQ(d_h[i], d_ref_h[i]) << i;
    }
}
