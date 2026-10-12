// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_vector.hpp"
#include "utility.hpp"
#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <string>
#include <utility>
#include <vector>

namespace
{
    enum class AlphaPath
    {
        Descriptor,
        GroupedDescriptor,
        LegacyVectorInputs,
        ScalarInputs,
        ScalarInputsShort,
        HostInputs,
        GroupedInputs,
        InvalidInputs
    };

    void checkExtensionAlpha(AlphaPath path, const float* hostAlpha = nullptr, uint16_t splitK = 0)
    {
        int             device = 0;
        hipDeviceProp_t properties{};
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        ASSERT_EQ(hipGetDeviceProperties(&properties, device), hipSuccess);
        if(std::string(properties.gcnArchName).find("gfx1250") != 0)
            GTEST_SKIP() << "Requires the gfx1250 HSS device-alpha logic";

        constexpr int                n        = 128;
        constexpr int                elements = n * n;
        hipblaslt_local_handle       handle;
        hipblaslt_local_matmul_descr desc(
            HIPBLAS_OP_T, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblaslt_local_matrix_layout abLayout(n, n, n, HIP_R_16F);
        hipblaslt_local_matrix_layout cdLayout(n, n, n, HIP_R_32F);
        device_vector<_Float16>       a(elements), b(elements);
        device_vector<float>          c(elements), d(elements);
        // Scalar mode deliberately receives only one element, not an M-element vector.
        device_vector<float> scale(path == AlphaPath::LegacyVectorInputs ? n : 1);
        ASSERT_EQ(a.memcheck(), hipSuccess);
        ASSERT_EQ(b.memcheck(), hipSuccess);
        ASSERT_EQ(c.memcheck(), hipSuccess);
        ASSERT_EQ(d.memcheck(), hipSuccess);
        ASSERT_EQ(scale.memcheck(), hipSuccess);
        std::vector<_Float16> ones(elements, 1);
        std::vector<float>    hostC(elements, 3), output(elements);
        ASSERT_EQ(hipMemcpy(a, ones.data(), elements * sizeof(_Float16), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(b, ones.data(), elements * sizeof(_Float16), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(c, hostC.data(), elements * sizeof(float), hipMemcpyHostToDevice),
                  hipSuccess);
        float alpha = 1, beta = 1;
        auto  mode = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_SCALAR_BETA_HOST_EXT;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)),
                  HIPBLAS_STATUS_SUCCESS);

        if(path == AlphaPath::GroupedDescriptor)
        {
            hipblaslt_ext::GroupedGemm           grouped(handle,
                                               HIPBLAS_OP_T,
                                               HIPBLAS_OP_N,
                                               HIP_R_16F,
                                               HIP_R_16F,
                                               HIP_R_32F,
                                               HIP_R_32F,
                                               HIPBLAS_COMPUTE_32F);
            std::vector<hipblasLtMatmulDesc_t>   descriptors{desc};
            std::vector<hipblasLtMatrixLayout_t> abLayouts{abLayout}, cdLayouts{cdLayout};
            std::vector<void*>                   alphas{static_cast<float*>(scale)}, betas{&beta};
            std::vector<void*> as{static_cast<_Float16*>(a)}, bs{static_cast<_Float16*>(b)};
            std::vector<void*> cs{static_cast<float*>(c)}, ds{static_cast<float*>(d)};
            // The existing public status conversion maps rocblaslt_status_not_implemented
            // to HIPBLAS_STATUS_INTERNAL_ERROR.
            EXPECT_EQ(grouped.setProblem(descriptors,
                                         alphas,
                                         as,
                                         abLayouts,
                                         bs,
                                         abLayouts,
                                         betas,
                                         cs,
                                         cdLayouts,
                                         ds,
                                         cdLayouts),
                      HIPBLAS_STATUS_INTERNAL_ERROR);
            return;
        }

        hipblaslt_ext::Gemm gemm(handle,
                                 HIPBLAS_OP_T,
                                 HIPBLAS_OP_N,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIP_R_32F,
                                 HIP_R_32F,
                                 HIPBLAS_COMPUTE_32F);
        if(path == AlphaPath::Descriptor)
        {
            ASSERT_EQ(gemm.setProblem(
                          desc, scale, a, abLayout, b, abLayout, &beta, c, cdLayout, d, cdLayout),
                      HIPBLAS_STATUS_SUCCESS);
        }
        else
        {
            hipblaslt_ext::GemmInputs inputs;
            inputs.setA(a);
            inputs.setB(b);
            inputs.setC(c);
            inputs.setD(d);
            inputs.setAlpha(path == AlphaPath::ScalarInputs || path == AlphaPath::ScalarInputsShort
                                ? hostAlpha
                                : &alpha);
            inputs.setBeta(&beta);
            inputs.setScaleAlphaScalar(scale);
            if(path == AlphaPath::LegacyVectorInputs)
                inputs.setScaleAlphaVec(scale);
            else if(path == AlphaPath::HostInputs)
                inputs.setScaleAlphaVec(nullptr);
            hipblaslt_ext::GemmEpilogue    epilogue;
            hipblaslt_ext::GemmProblemType type(HIPBLAS_OP_T,
                                                HIPBLAS_OP_N,
                                                HIP_R_16F,
                                                HIP_R_16F,
                                                HIP_R_32F,
                                                HIP_R_32F,
                                                HIPBLAS_COMPUTE_32F);
            if(path == AlphaPath::GroupedInputs)
            {
                hipblaslt_ext::GroupedGemm               grouped(handle,
                                                   HIPBLAS_OP_T,
                                                   HIPBLAS_OP_N,
                                                   HIP_R_16F,
                                                   HIP_R_16F,
                                                   HIP_R_32F,
                                                   HIP_R_32F,
                                                   HIPBLAS_COMPUTE_32F);
                std::vector<int64_t>                     sizes{n}, batches{1};
                std::vector<hipblaslt_ext::GemmEpilogue> epilogues{epilogue};
                std::vector<hipblaslt_ext::GemmInputs>   groupedInputs{inputs};
                EXPECT_EQ(
                    grouped.setProblem(sizes, sizes, sizes, batches, epilogues, groupedInputs),
                    HIPBLAS_STATUS_INTERNAL_ERROR);
                return;
            }
            if(path == AlphaPath::InvalidInputs)
            {
                inputs.setScaleAlphaScalar(nullptr);
                EXPECT_EQ(gemm.setProblem(n, n, n, 1, epilogue, inputs),
                          HIPBLAS_STATUS_INVALID_VALUE);
                inputs.setScaleAlphaScalar(scale);
                float zero = 0;
                inputs.setAlpha(&zero);
                inputs.setA(nullptr);
                EXPECT_EQ(gemm.setProblem(n, n, n, 1, epilogue, inputs),
                          HIPBLAS_STATUS_INVALID_VALUE);
                inputs.setA(a);
                inputs.setB(nullptr);
                EXPECT_EQ(gemm.setProblem(n, n, n, 1, epilogue, inputs),
                          HIPBLAS_STATUS_INVALID_VALUE);
                inputs.setB(b);
                inputs.setBeta(nullptr);
                EXPECT_EQ(gemm.setProblem(n, n, n, 1, epilogue, inputs),
                          HIPBLAS_STATUS_INVALID_VALUE);
                return;
            }
            // Exercise deep copies as well as the short and explicit-stride overloads.
            hipblaslt_ext::GemmInputs copied(inputs);
            inputs.setScaleAlphaVec(nullptr);
            if(path == AlphaPath::ScalarInputsShort)
            {
                ASSERT_EQ(gemm.setProblem(n, n, n, 1, epilogue, copied), HIPBLAS_STATUS_SUCCESS);
            }
            else
            {
                ASSERT_EQ(gemm.setProblem(n,
                                          n,
                                          n,
                                          1,
                                          n,
                                          n,
                                          n,
                                          n,
                                          elements,
                                          elements,
                                          elements,
                                          elements,
                                          epilogue,
                                          copied,
                                          type),
                          HIPBLAS_STATUS_SUCCESS);
            }
        }

        hipblaslt_ext::GemmPreference preference;
        preference.setMaxWorkspaceBytes(32 * 1024 * 1024);
        std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
        ASSERT_EQ(gemm.algoGetHeuristic(1, preference, algorithms), HIPBLAS_STATUS_SUCCESS);
        ASSERT_FALSE(algorithms.empty());
        hipblaslt_ext::GemmTuning tuning;
        tuning.setSplitK(splitK);
        size_t workspaceBytes = 0;
        ASSERT_EQ(gemm.isAlgoSupported(algorithms[0].algo, tuning, workspaceBytes),
                  HIPBLAS_STATUS_SUCCESS);
        device_vector<char> workspace(workspaceBytes + 1);
        ASSERT_EQ(workspace.memcheck(), hipSuccess);
        gemm.setMaxWorkspaceBytes(workspaceBytes);
        ASSERT_EQ(gemm.initialize(algorithms[0].algo, tuning, workspace), HIPBLAS_STATUS_SUCCESS);

        // Updating device memory after initialize must affect every later launch.
        for(float value : {2.0f, -1.0f, 0.0f})
        {
            std::vector<float> factors(path == AlphaPath::LegacyVectorInputs ? n : 1, value);
            if(path == AlphaPath::LegacyVectorInputs)
                for(int row = 0; row < n; ++row)
                    factors[row] += row % 3;
            ASSERT_EQ(
                hipMemcpy(
                    scale, factors.data(), factors.size() * sizeof(float), hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(gemm.run(nullptr), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipMemcpy(output.data(), d, elements * sizeof(float), hipMemcpyDeviceToHost),
                      hipSuccess);
            for(int index = 0; index < elements; ++index)
            {
                const float factor
                    = path == AlphaPath::HostInputs
                          ? alpha
                          : factors[path == AlphaPath::LegacyVectorInputs ? index % n : 0];
                ASSERT_EQ(output[index], n * factor + 3) << "index=" << index << " alpha=" << value;
            }
        }
    }
}

TEST(pre_checkin_device_scalar_alpha_ext, DescriptorGemm)
{
    checkExtensionAlpha(AlphaPath::Descriptor);
}

TEST(pre_checkin_device_scalar_alpha_ext, GroupedDescriptorRejected)
{
    checkExtensionAlpha(AlphaPath::GroupedDescriptor);
}

TEST(pre_checkin_device_scalar_alpha_ext, GemmInputsRetainsVectorSemantics)
{
    checkExtensionAlpha(AlphaPath::LegacyVectorInputs);
}

TEST(pre_checkin_device_scalar_alpha_ext, GemmInputsScalar)
{
    const float zero = 0, negative = -1;
    for(uint16_t splitK : {1, 2})
        for(const float* hostAlpha : {static_cast<const float*>(nullptr), &zero, &negative})
        {
            SCOPED_TRACE(::testing::Message() << "splitK=" << splitK << " hostAlpha=" << hostAlpha);
            checkExtensionAlpha(AlphaPath::ScalarInputs, hostAlpha, splitK);
        }
}

TEST(pre_checkin_device_scalar_alpha_ext, GemmInputsScalarShortOverload)
{
    checkExtensionAlpha(AlphaPath::ScalarInputsShort);
}

TEST(pre_checkin_device_scalar_alpha_ext, GemmInputsReturnsToHostAlpha)
{
    checkExtensionAlpha(AlphaPath::HostInputs);
}

TEST(pre_checkin_device_scalar_alpha_ext, GroupedGemmInputsRejected)
{
    checkExtensionAlpha(AlphaPath::GroupedInputs);
}

TEST(pre_checkin_device_scalar_alpha_ext, GemmInputsInvalidPointers)
{
    checkExtensionAlpha(AlphaPath::InvalidInputs);
}

TEST(pre_checkin_device_scalar_alpha_ext, GemmInputsModeCopyAndMove)
{
    // Setters store pointers without reading memory, so this test needs no device allocation.
    float                     scalar = 2, vector[2] = {3, 4};
    hipblaslt_ext::GemmInputs inputs;
    EXPECT_EQ(inputs.getScaleAlphaScalar(), nullptr);
    EXPECT_EQ(inputs.getScaleAlphaVec(), nullptr);
    inputs.setScaleAlphaVec(vector);
    EXPECT_EQ(inputs.getScaleAlphaVec(), vector);
    EXPECT_EQ(inputs.getScaleAlphaScalar(), nullptr);
    inputs.setScaleAlphaScalar(&scalar);
    EXPECT_EQ(inputs.getScaleAlphaScalar(), &scalar);
    EXPECT_EQ(inputs.getScaleAlphaVec(), nullptr);

    hipblaslt_ext::GemmInputs copied(inputs), assigned;
    assigned = inputs;
    inputs.setScaleAlphaVec(vector);
    EXPECT_EQ(inputs.getScaleAlphaVec(), vector);
    EXPECT_EQ(inputs.getScaleAlphaScalar(), nullptr);
    EXPECT_EQ(copied.getScaleAlphaScalar(), &scalar);
    EXPECT_EQ(assigned.getScaleAlphaScalar(), &scalar);
    hipblaslt_ext::GemmInputs moved(std::move(copied)), moveAssigned;
    moveAssigned = std::move(assigned);
    EXPECT_EQ(moved.getScaleAlphaScalar(), &scalar);
    EXPECT_EQ(moveAssigned.getScaleAlphaScalar(), &scalar);
    moved.setScaleAlphaVec(nullptr);
    EXPECT_EQ(moved.getScaleAlphaVec(), nullptr);
    EXPECT_EQ(moved.getScaleAlphaScalar(), nullptr);
}
