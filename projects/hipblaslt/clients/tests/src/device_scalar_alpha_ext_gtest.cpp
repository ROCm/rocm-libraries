// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_vector.hpp"
#include "utility.hpp"
#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstring>
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
            hipblaslt_local_matmul_descr         hostDesc(
                HIPBLAS_OP_T, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
            std::vector<hipblasLtMatmulDesc_t>   descriptors{desc, hostDesc};
            std::vector<hipblasLtMatrixLayout_t> abLayouts(2, abLayout), cdLayouts(2, cdLayout);
            std::vector<void*> alphas{static_cast<float*>(scale), &alpha}, betas(2, &beta);
            std::vector<void*> as(2, static_cast<_Float16*>(a)), bs(2, static_cast<_Float16*>(b));
            std::vector<void*> cs(2, static_cast<float*>(c)), ds(2, static_cast<float*>(d));
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
                      HIPBLAS_STATUS_INVALID_VALUE);
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
                std::vector<int64_t>                     sizes(2, n), batches(2, 1);
                std::vector<hipblaslt_ext::GemmEpilogue> epilogues{epilogue};
                std::vector<hipblaslt_ext::GemmInputs>   groupedInputs{inputs, inputs};
                groupedInputs[1].setScaleAlphaVec(nullptr);
                EXPECT_EQ(
                    grouped.setProblem(sizes, sizes, sizes, batches, epilogues, groupedInputs),
                    HIPBLAS_STATUS_INVALID_VALUE);
                groupedInputs[1].setScaleAlphaVec(scale);
                EXPECT_EQ(
                    grouped.setProblem(sizes, sizes, sizes, batches, epilogues, groupedInputs),
                    HIPBLAS_STATUS_INVALID_VALUE);
                std::swap(groupedInputs[0], groupedInputs[1]);
                EXPECT_EQ(
                    grouped.setProblem(sizes, sizes, sizes, batches, epilogues, groupedInputs),
                    HIPBLAS_STATUS_INVALID_VALUE);
                groupedInputs[0].setScaleAlphaScalar(nullptr);
                EXPECT_EQ(
                    grouped.setProblem(sizes, sizes, sizes, batches, epilogues, groupedInputs),
                    HIPBLAS_STATUS_INVALID_VALUE);
                groupedInputs[0].setScaleAlphaScalar(scale);
                groupedInputs[0].setA(nullptr);
                EXPECT_EQ(
                    grouped.setProblem(sizes, sizes, sizes, batches, epilogues, groupedInputs),
                    HIPBLAS_STATUS_INVALID_VALUE);
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

TEST(pre_checkin_device_scalar_alpha_ext, DescriptorRejectsNonFp32Scale)
{
    // Rejection happens before kernel selection, regardless of the GPU architecture.
    constexpr int                 n = 128;
    hipblaslt_local_handle        handle;
    hipblaslt_local_matrix_layout abLayout(n, n, n, HIP_R_16F);
    hipblaslt_local_matrix_layout cdLayout(n, n, n, HIP_R_32F);
    device_vector<_Float16>       a(n * n), b(n * n);
    device_vector<float>          c(n * n), d(n * n), alpha(2);
    ASSERT_EQ(a.memcheck(), hipSuccess);
    ASSERT_EQ(b.memcheck(), hipSuccess);
    ASSERT_EQ(c.memcheck(), hipSuccess);
    ASSERT_EQ(d.memcheck(), hipSuccess);
    ASSERT_EQ(alpha.memcheck(), hipSuccess);
    ASSERT_EQ(hipMemset(a, 0, n * n * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMemset(b, 0, n * n * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMemset(c, 0, n * n * sizeof(float)), hipSuccess);
    // Zero storage is valid for every tested scale type; pad to the largest one.
    ASSERT_EQ(hipMemset(alpha, 0, 2 * sizeof(float)), hipSuccess);
    double                       beta = 0;
    auto                         mode = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_SCALAR_BETA_HOST_EXT;
    hipblaslt_local_matmul_descr validDesc(
        HIPBLAS_OP_T, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                  validDesc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)),
              HIPBLAS_STATUS_SUCCESS);

    for(auto types : {std::make_pair(HIPBLAS_COMPUTE_16F, HIP_R_16F),
                      std::make_pair(HIPBLAS_COMPUTE_32F, HIP_R_16F),
                      std::make_pair(HIPBLAS_COMPUTE_32F, HIP_R_64F),
                      std::make_pair(HIPBLAS_COMPUTE_32F, HIP_R_32I),
                      std::make_pair(HIPBLAS_COMPUTE_32F, HIP_C_32F)})
    {
        SCOPED_TRACE(::testing::Message()
                     << "compute=" << types.first << " scale=" << types.second);
        hipblaslt_local_matmul_descr desc(HIPBLAS_OP_T, HIPBLAS_OP_N, types.first, types.second);
        // Setting and reading the mode remains independent of datatype validation.
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtPointerMode_t readMode = HIPBLASLT_POINTER_MODE_HOST;
        size_t                 bytes    = 0;
        ASSERT_EQ(
            hipblasLtMatmulDescGetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &readMode, sizeof(readMode), &bytes),
            HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(readMode, mode);
        EXPECT_EQ(bytes, sizeof(mode));

        hipblaslt_local_preference       preference;
        hipblasLtMatmulHeuristicResult_t result{};
        int                              returned = -1;
        EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                  desc,
                                                  abLayout,
                                                  abLayout,
                                                  cdLayout,
                                                  cdLayout,
                                                  preference,
                                                  1,
                                                  &result,
                                                  &returned),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(returned, 0);
        size_t workspaceBytes = 0;
        EXPECT_EQ(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                       desc,
                                                       alpha,
                                                       abLayout,
                                                       abLayout,
                                                       &beta,
                                                       cdLayout,
                                                       cdLayout,
                                                       result.algo,
                                                       workspaceBytes),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(
            hipblaslt_ext::matmulIsTuned(handle, desc, abLayout, abLayout, cdLayout, cdLayout), -1);

        EXPECT_EQ(hipblasLtMatmul(handle,
                                  desc,
                                  alpha,
                                  a,
                                  abLayout,
                                  b,
                                  abLayout,
                                  &beta,
                                  c,
                                  cdLayout,
                                  d,
                                  cdLayout,
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);

        hipblaslt_ext::Gemm gemm(handle,
                                 HIPBLAS_OP_T,
                                 HIPBLAS_OP_N,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIP_R_32F,
                                 HIP_R_32F,
                                 HIPBLAS_COMPUTE_32F);
        EXPECT_EQ(
            gemm.setProblem(desc, alpha, a, abLayout, b, abLayout, &beta, c, cdLayout, d, cdLayout),
            HIPBLAS_STATUS_INVALID_VALUE);

        for(int invalidIndex : {0, 1})
        {
            SCOPED_TRACE(::testing::Message() << "invalid grouped descriptor=" << invalidIndex);
            hipblaslt_ext::GroupedGemm           grouped(handle,
                                               HIPBLAS_OP_T,
                                               HIPBLAS_OP_N,
                                               HIP_R_16F,
                                               HIP_R_16F,
                                               HIP_R_32F,
                                               HIP_R_32F,
                                               HIPBLAS_COMPUTE_32F);
            std::vector<hipblasLtMatmulDesc_t>   descriptors(2, validDesc);
            std::vector<hipblasLtMatrixLayout_t> abLayouts(2, abLayout), cdLayouts(2, cdLayout);
            std::vector<void*> alphas(2, static_cast<float*>(alpha)), betas(2, &beta);
            std::vector<void*> as(2, static_cast<_Float16*>(a)), bs(2, static_cast<_Float16*>(b));
            std::vector<void*> cs(2, static_cast<float*>(c)), ds(2, static_cast<float*>(d));
            descriptors[invalidIndex] = desc;
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
                      HIPBLAS_STATUS_INVALID_VALUE);
        }
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST(pre_checkin_device_scalar_alpha_ext, GroupedDescriptorMixedModesRejected)
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

TEST(pre_checkin_device_scalar_alpha_ext, GroupedGemmInputsMixedModesRejected)
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

namespace
{
    // Three GEMMs have different K and alpha. No scalar may be reused across records.
    void checkGroupedAlpha(bool descriptor, bool userArgs, uint16_t splitK, int alphaMode = 0)
    {
        int             device = 0;
        hipDeviceProp_t properties{};
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        ASSERT_EQ(hipGetDeviceProperties(&properties, device), hipSuccess);
        if(std::string(properties.gcnArchName).find("gfx1250") != 0)
            GTEST_SKIP() << "Requires the gfx1250 grouped HSS SAV logic";

        constexpr int           count = 3, n = 128, elements = n * n;
        hipblaslt_local_handle  handle;
        device_vector<_Float16> a(elements * count), b(elements * count);
        device_vector<float>    c(elements), d(elements * count);
        device_vector<float>    scale(alphaMode == 2 ? n * count : count);
        ASSERT_EQ(a.memcheck(), hipSuccess);
        ASSERT_EQ(b.memcheck(), hipSuccess);
        ASSERT_EQ(c.memcheck(), hipSuccess);
        ASSERT_EQ(d.memcheck(), hipSuccess);
        ASSERT_EQ(scale.memcheck(), hipSuccess);
        std::vector<_Float16> ones(elements * count, 1);
        std::vector<float>    hostC(elements, 3), output(elements * count);
        ASSERT_EQ(hipMemcpy(a, ones.data(), ones.size() * sizeof(_Float16), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(b, ones.data(), ones.size() * sizeof(_Float16), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(c, hostC.data(), elements * sizeof(float), hipMemcpyHostToDevice),
                  hipSuccess);
        float              ignoredAlpha = 0, one = 1, beta = 1;
        std::vector<float> hostAlpha{2, -1, 0};
        // Host alpha=0 becomes K=0 and cannot meet the split-K kernel's minimum K.
        // Keep that case at GSU=1; use nonzero host values for conversion coverage.
        if(alphaMode == 1 && splitK > 1)
            hostAlpha.back() = 3;
        hipblaslt_ext::GroupedGemm               grouped(handle,
                                           HIPBLAS_OP_T,
                                           HIPBLAS_OP_N,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_32F,
                                           HIP_R_32F,
                                           HIPBLAS_COMPUTE_32F);
        std::vector<hipblaslt_ext::GemmInputs>   inputs(count);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogues(1);
        std::vector<int64_t> sizes(count, n), ks{n, 2 * n, 3 * n}, batches(count, 1);
        for(int i = 0; i < count; ++i)
        {
            inputs[i].setA(a);
            inputs[i].setB(b);
            inputs[i].setC(c);
            inputs[i].setD(static_cast<float*>(d) + i * elements);
            inputs[i].setAlpha(alphaMode == 1   ? &hostAlpha[i]
                               : alphaMode == 2 ? &one
                                                : &ignoredAlpha);
            inputs[i].setBeta(&beta);
            if(alphaMode == 0)
            {
                inputs[i].setScaleAlphaScalar(static_cast<float*>(scale) + i);
                if(i == count - 1)
                    inputs[i].setAlpha(nullptr);
            }
            else if(alphaMode == 2)
                inputs[i].setScaleAlphaVec(static_cast<float*>(scale) + i * n);
        }
        if(descriptor)
        {
            hipblaslt_local_matmul_descr desc(
                HIPBLAS_OP_T, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
            auto mode = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_SCALAR_BETA_HOST_EXT;
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                          desc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)),
                      HIPBLAS_STATUS_SUCCESS);
            hipblaslt_local_matrix_layout        ab0(n, n, n, HIP_R_16F);
            hipblaslt_local_matrix_layout        ab1(2 * n, n, 2 * n, HIP_R_16F);
            hipblaslt_local_matrix_layout        ab2(3 * n, n, 3 * n, HIP_R_16F);
            hipblaslt_local_matrix_layout        cd(n, n, n, HIP_R_32F);
            std::vector<hipblasLtMatmulDesc_t>   descriptors(count, desc);
            std::vector<hipblasLtMatrixLayout_t> abs{ab0, ab1, ab2}, cds(count, cd);
            std::vector<void*> alphas, betas(count, &beta), as(count, static_cast<_Float16*>(a)),
                bs(count, static_cast<_Float16*>(b)), cs(count, static_cast<float*>(c)), ds;
            for(int i = 0; i < count; ++i)
            {
                alphas.push_back(static_cast<float*>(scale) + i);
                ds.push_back(static_cast<float*>(d) + i * elements);
            }
            ASSERT_EQ(
                grouped.setProblem(descriptors, alphas, as, abs, bs, abs, betas, cs, cds, ds, cds),
                HIPBLAS_STATUS_SUCCESS);
        }
        else
        {
            ASSERT_EQ(grouped.setProblem(sizes, sizes, ks, batches, epilogues, inputs),
                      HIPBLAS_STATUS_SUCCESS);
        }

        hipblaslt_ext::GemmPreference preference;
        preference.setMaxWorkspaceBytes(32 * 1024 * 1024);
        std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
        ASSERT_EQ(grouped.algoGetHeuristic(32, preference, algorithms), HIPBLAS_STATUS_SUCCESS);
        ASSERT_FALSE(algorithms.empty());
        hipblaslt_ext::GemmTuning tuning;
        tuning.setSplitK(splitK);
        size_t workspaceBytes = 0;
        // Host mode may rank a GSU=1-only kernel first. Find a candidate supporting
        // the requested tuning rather than assuming every candidate supports it.
        auto selected = algorithms.end();
        for(auto candidate = algorithms.begin(); candidate != algorithms.end(); ++candidate)
            if(grouped.isAlgoSupported(candidate->algo, tuning, workspaceBytes)
               == HIPBLAS_STATUS_SUCCESS)
            {
                selected = candidate;
                break;
            }
        ASSERT_NE(selected, algorithms.end());
        device_vector<char> workspace(workspaceBytes + 1);
        ASSERT_EQ(workspace.memcheck(), hipSuccess);
        grouped.setMaxWorkspaceBytes(workspaceBytes);
        if(userArgs && splitK > 1 && alphaMode == 0)
        {
            EXPECT_EQ(grouped.initialize(selected->algo, tuning, workspace, true),
                      HIPBLAS_STATUS_INVALID_VALUE);
            return;
        }
        ASSERT_EQ(grouped.initialize(selected->algo, tuning, workspace, userArgs),
                  HIPBLAS_STATUS_SUCCESS);

        std::vector<hipblaslt_ext::UserArguments> records(count);
        device_vector<char> deviceRecords(count * sizeof(hipblaslt_ext::UserArguments));
        ASSERT_EQ(deviceRecords.memcheck(), hipSuccess);
        if(userArgs)
        {
            ASSERT_EQ(grouped.getDefaultValueForDeviceUserArguments(records.data()),
                      HIPBLAS_STATUS_SUCCESS);
            // Values in fixed alpha slots must be ignored in scalar mode.
            if(alphaMode == 0)
                for(auto& record : records)
                    std::memcpy(record.alpha, &ignoredAlpha, sizeof(float));
            ASSERT_EQ(hipMemcpy(deviceRecords,
                                records.data(),
                                count * sizeof(records[0]),
                                hipMemcpyHostToDevice),
                      hipSuccess);
        }
        for(int iteration = 0; iteration < 2; ++iteration)
        {
            std::vector<float> factors(alphaMode == 2 ? n * count : count);
            for(int i = 0; i < count; ++i)
                for(int row = 0; row < (alphaMode == 2 ? n : 1); ++row)
                    factors[i * (alphaMode == 2 ? n : 1) + row]
                        = hostAlpha[(i + iteration) % count] + (alphaMode == 2 ? row % 3 : 0);
            ASSERT_EQ(
                hipMemcpy(
                    scale, factors.data(), factors.size() * sizeof(float), hipMemcpyHostToDevice),
                hipSuccess);
            // Device-user-args own the pointers used at launch, not the original GemmInputs.
            if(userArgs && alphaMode == 0 && iteration == 1)
            {
                for(int i = 0; i < count; ++i)
                    records[i].scaleAlphaVec = static_cast<float*>(scale) + count - 1 - i;
                ASSERT_EQ(hipMemcpy(deviceRecords,
                                    records.data(),
                                    count * sizeof(records[0]),
                                    hipMemcpyHostToDevice),
                          hipSuccess);
            }
            ASSERT_EQ(userArgs ? grouped.run(deviceRecords, nullptr) : grouped.run(nullptr),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(
                hipMemcpy(output.data(), d, output.size() * sizeof(float), hipMemcpyDeviceToHost),
                hipSuccess);
            for(int i = 0; i < count; ++i)
                for(int j = 0; j < elements; ++j)
                {
                    float factor = alphaMode == 1 ? hostAlpha[i]
                                                  : factors[i * (alphaMode == 2 ? n : 1)
                                                            + (alphaMode == 2 ? j % n : 0)];
                    if(userArgs && alphaMode == 0 && iteration == 1)
                        factor = factors[count - 1 - i];
                    ASSERT_EQ(output[i * elements + j], ks[i] * factor + 3)
                        << "GEMM=" << i << " element=" << j << " iteration=" << iteration;
                }
        }
    }
}

TEST(pre_checkin_device_scalar_alpha_ext, GroupedScalarPackedArgs)
{
    for(bool descriptor : {false, true})
        for(uint16_t splitK : {1, 2})
        {
            SCOPED_TRACE(::testing::Message() << "descriptor=" << descriptor << " GSU=" << splitK);
            checkGroupedAlpha(descriptor, false, splitK);
        }
}

TEST(pre_checkin_device_scalar_alpha_ext, GroupedScalarDeviceUserArgs)
{
    for(bool descriptor : {false, true})
        checkGroupedAlpha(descriptor, true, 1);
}

TEST(pre_checkin_device_scalar_alpha_ext, GroupedScalarDeviceUserArgsRejectsSplitK)
{
    checkGroupedAlpha(false, true, 2);
}

TEST(pre_checkin_device_scalar_alpha_ext, GroupedHostVectorRegression)
{
    for(int mode : {1, 2})
        for(bool userArgs : {false, true})
            for(uint16_t splitK : {1, 2})
            {
                // Device-user-args conversion is outside this change's support scope.
                if(userArgs && splitK > 1)
                    continue;
                SCOPED_TRACE(::testing::Message()
                             << "mode=" << mode << " userArgs=" << userArgs << " GSU=" << splitK);
                checkGroupedAlpha(false, userArgs, splitK, mode);
            }
}
