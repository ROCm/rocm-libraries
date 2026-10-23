// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_validation.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <roc/hostnumerics/generation.hpp>
#include <roc/hostnumerics/periodic_matmul.hpp>
#include <set>
#include <sstream>
#include <stdexcept>

namespace
{
    namespace hn = roc::hostnumerics;
    using Clock  = std::chrono::steady_clock;

    void requireHip(hipError_t status)
    {
        if(status != hipSuccess)
            throw std::runtime_error(hipGetErrorString(status));
    }
    void requireBlas(hipblasStatus_t status)
    {
        if(status != HIPBLAS_STATUS_SUCCESS)
            throw std::runtime_error("hipBLASLt status " + std::to_string(int(status)));
    }
    double seconds(Clock::time_point begin)
    {
        return std::chrono::duration<double>(Clock::now() - begin).count();
    }

    struct DeviceBytes
    {
        void* data = nullptr;
        explicit DeviceBytes(size_t bytes)
        {
            requireHip(hipMalloc(&data, bytes));
        }
        ~DeviceBytes()
        {
            (void)hipFree(data);
        }
        DeviceBytes(const DeviceBytes&) = delete;
        DeviceBytes& operator=(const DeviceBytes&) = delete;
    };

    struct Descriptors
    {
        hipblasLtHandle_t                      handle     = nullptr;
        hipblasLtMatmulDesc_t                  operation  = nullptr;
        hipblasLtMatmulPreference_t            preference = nullptr;
        std::array<hipblasLtMatrixLayout_t, 4> layouts{};
        ~Descriptors()
        {
            for(auto layout : layouts)
                if(layout)
                    (void)hipblasLtMatrixLayoutDestroy(layout);
            if(preference)
                (void)hipblasLtMatmulPreferenceDestroy(preference);
            if(operation)
                (void)hipblasLtMatmulDescDestroy(operation);
            if(handle)
                (void)hipblasLtDestroy(handle);
        }
    };

    // Read all padding on the device. Numerical validation separately visits
    // every logical output; neither scan requires a padded host output buffer.
    __global__ void checkPadding(const uint32_t* d,
                                 size_t          count,
                                 size_t          m,
                                 size_t          n,
                                 size_t          ld,
                                 size_t          stride,
                                 unsigned*       bad)
    {
        for(size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < count;
            x += size_t(gridDim.x) * blockDim.x)
        {
            const size_t inBatch = x % stride;
            if((inBatch / ld >= n || inBatch % ld >= m) && d[x] != 0xffffffffU)
                atomicExch(bad, 1U);
        }
    }

    hn::Tensor matrix(hn::ScalarType type,
                      size_t         batches,
                      size_t         rows,
                      size_t         cols,
                      ptrdiff_t      rowStride,
                      ptrdiff_t      columnStride,
                      size_t         batchStride)
    {
        auto storage = hn::generate(
            type,
            hn::Shape{batches * batchStride},
            hn::GenerationRecipe::realOnly(hn::GenerationRecipe::constant({.value = 40}), {}));
        return storage.shareStorageWithLayout(hn::Layout(
            hn::Shape{batches, rows, cols}, {ptrdiff_t(batchStride), rowStride, columnStride}, 0));
    }

    hn::PeriodicMatmulPlan requirePlan(hn::PeriodicMatmulPreparation preparation)
    {
        if(const auto* failure = std::get_if<hn::VerificationFailure>(&preparation))
            throw std::runtime_error(failure->reason);
        return std::get<hn::PeriodicMatmulPlan>(std::move(preparation));
    }

    enum class CheckMethod
    {
        Periodic,
        DenseBaseline
    };

    void runAcceptance(bool        floating,
                       bool        large,
                       bool        transA,
                       bool        transB,
                       CheckMethod method = CheckMethod::Periodic)
    {
        const size_t  m = large ? 4099 : 131, n = large ? 4093 : 67, k = large ? 4097 : 201;
        const int32_t batches = large ? 2 : 3;
        const size_t  lda = (transA ? k : m) + 4, ldb = (transB ? n : k) + 6;
        const size_t  ldc = m + 8, ldd = m + 12;
        const size_t  strideA = lda * (transA ? m : k) + 13;
        const size_t  strideB = ldb * (transB ? k : n) + 17;
        const size_t  strideC = ldc * n + 19, strideD = ldd * n + 23;
        // Host operands remain compact even though all device matrices have
        // padded leading dimensions and batch strides.
        const size_t                hostStrideA = m * k, hostStrideB = k * n, hostStrideC = m * n;
        const ptrdiff_t             ar = transA ? ptrdiff_t(k) : 1, ac = transA ? 1 : ptrdiff_t(m);
        const ptrdiff_t             br = transB ? ptrdiff_t(n) : 1, bc = transB ? 1 : ptrdiff_t(k);
        const std::array<size_t, 2> periodsA
            = large ? std::array<size_t, 2>{67, 71} : std::array<size_t, 2>{11, 13};
        const std::array<size_t, 2> periodsB
            = large ? std::array<size_t, 2>{73, 79} : std::array<size_t, 2>{7, 11};
        auto begin = Clock::now();
        auto a     = matrix(hn::ScalarType::BFloat16, batches, m, k, ar, ac, hostStrideA);
        auto b     = matrix(hn::ScalarType::BFloat16, batches, k, n, br, bc, hostStrideB);
        auto c     = matrix(hn::ScalarType::Float32, batches, m, n, 1, ptrdiff_t(m), hostStrideC);
        std::vector<std::vector<hn::PeriodicMatmulSegment>> definitions(batches);
        for(int batch = 0; batch < batches; ++batch)
        {
            for(size_t s = 0; s < 2U; ++s)
            {
                const size_t first  = s == 0 ? 0 : k / 2;
                const size_t length = s == 0 ? k / 2 : k - k / 2;
                auto         recipe = [&](uint64_t seed) {
                    return hn::GenerationRecipe::realOnly(
                        floating ? hn::GenerationRecipe::uniformReal({.lower = -1, .upper = 1})
                                 : hn::GenerationRecipe::uniformInteger({.lower = -2, .upper = 2}),
                        {.seed = seed});
                };
                auto at = hn::generate(hn::ScalarType::BFloat16,
                                       hn::Shape{periodsA[s], length},
                                       recipe(31 + 101 * batch + 17 * s));
                auto bt = hn::generate(hn::ScalarType::BFloat16,
                                       hn::Shape{length, periodsB[s]},
                                       recipe(73 + 211 * batch + 23 * s));
                definitions[batch].push_back(
                    {first,
                     hn::PeriodicInitialization(at, hn::Shape{m, length}),
                     hn::PeriodicInitialization(bt, hn::Shape{length, n})});
                const auto& definition = definitions[batch].back();
                // Chunk boundaries cross the input periods. The logical origin
                // is local to this segment; the destination layout selects K.
                for(size_t row = 0; row < m; row += 37)
                    definition.a.fillInto(a.shareStorageWithLayout(hn::Layout(
                                              hn::Shape{std::min(size_t(37), m - row), length},
                                              {ar, ac},
                                              ptrdiff_t(batch * hostStrideA) + ptrdiff_t(row) * ar
                                                  + ptrdiff_t(first) * ac)),
                                          std::array<size_t, 2>{row, 0});
                for(size_t col = 0; col < n; col += 31)
                    definition.b.fillInto(b.shareStorageWithLayout(hn::Layout(
                                              hn::Shape{length, std::min(size_t(31), n - col)},
                                              {br, bc},
                                              ptrdiff_t(batch * hostStrideB) + ptrdiff_t(first) * br
                                                  + ptrdiff_t(col) * bc)),
                                          std::array<size_t, 2>{0, col});
            }
            for(size_t j = 0; j < n; ++j)
                for(size_t i = 0; i < m; ++i)
                    c.storeFrom({size_t(batch), i, j},
                                float(int((7 * i + 11 * j + 13 * batch) % 17) - 8)
                                    / (floating ? 31.0f : 1.0f));
        }
        hn::Tensor bias(hn::ScalarType::Float32, hn::Shape{large ? 0U : m, 1});
        for(size_t i = 0; i < (large ? 0U : m); ++i)
            bias.storeFrom({i, 0}, float(int((i * 7) % 13) - 6) / (floating ? 8.0f : 1.0f));
        const double generationSeconds = seconds(begin);
        begin                          = Clock::now();
        std::vector<hn::PeriodicMatmulPlan> plans;
        size_t                              referenceWork = 0, productBytes = 0;
        if(method == CheckMethod::Periodic)
            for(const auto& definition : definitions)
            {
                plans.push_back(requirePlan(hn::preparePeriodicMatmul(
                    hn::Shape{m, n},
                    k,
                    definition,
                    hn::MatmulOptions{roc::hostnumerics::ScalarType::Float32},
                    floating ? hn::PeriodicMatmulMode::FloatingPoint
                             : hn::PeriodicMatmulMode::ExactInteger,
                    {})));
                referenceWork += plans.back().statistics().referenceMultiplyAdds;
                productBytes += plans.back().statistics().productBytes;
                EXPECT_GT(plans.back().statistics().combinedRowPeriod, m);
                EXPECT_GT(plans.back().statistics().combinedColumnPeriod, n);
            }
        std::vector<hn::Tensor> fullReference;
        size_t                  fullReferenceCalls = 0;
        if(method == CheckMethod::DenseBaseline)
        {
            ASSERT_FALSE(floating); // paired exact baseline uses the same F64 arithmetic
            for(size_t batch = 0; batch < size_t(batches); ++batch)
            {
                auto product
                    = hn::matmul(a.shareStorageWithLayout(
                                     hn::Layout(hn::Shape{m, k}, {ar, ac}, batch * hostStrideA)),
                                 b.shareStorageWithLayout(
                                     hn::Layout(hn::Shape{k, n}, {br, bc}, batch * hostStrideB)),
                                 hn::ScalarType::Float64,
                                 hn::MatmulOptions(hn::ScalarType::Float64),
                                 std::nullopt,
                                 roc::hostnumerics::GemmBackend::Automatic);
                ++fullReferenceCalls;
                hn::EpilogueOptions epilogue(hn::ScalarType::Float64);
                epilogue.inputScale  = hn::Tensor(2.0);
                epilogue.addendScale = hn::Tensor(-1.0);
                epilogue.addend      = c.shareStorageWithLayout(
                    hn::Layout(hn::Shape{m, n}, {1, ptrdiff_t(m)}, batch * hostStrideC));
                if(!large)
                    epilogue.bias = bias;
                hn::referenceEpilogueInto(product, {product}, epilogue);
                productBytes += product.rawEncodedBackingStorage().size();
                referenceWork += m * n * k;
                fullReference.push_back(std::move(product));
            }
        }
        const double preparationSeconds = seconds(begin);

        // Independent dense oracle only for the small cross-check. It consumes
        // the materialized encoded inputs, without segment products or modulo.
        std::vector<double> dense;
        if(!large)
        {
            dense.resize(size_t(batches) * m * n);
            for(size_t batch = 0; batch < size_t(batches); ++batch)
                for(size_t i = 0; i < m; ++i)
                    for(size_t j = 0; j < n; ++j)
                    {
                        double sum = 0;
                        for(size_t z = 0; z < k; ++z)
                            sum += a.loadAs<double>({batch, i, z})
                                   * b.loadAs<double>({batch, z, j});
                        dense[(batch * m + i) * n + j] = 2 * sum - c.loadAs<double>({batch, i, j})
                                                         + bias.loadAs<double>({i, 0});
                    }
        }
        DeviceBytes da(size_t(batches) * strideA * sizeof(uint16_t)),
            db(size_t(batches) * strideB * sizeof(uint16_t));
        DeviceBytes dc(size_t(batches) * strideC * sizeof(float)),
            dd(size_t(batches) * strideD * sizeof(float));
        DeviceBytes      deviceBias(std::max(size_t(1), bias.rawEncodedBackingStorage().size()));
        constexpr size_t workspaceBytes = 64 * 1024 * 1024;
        DeviceBytes      workspace(workspaceBytes), badPadding(sizeof(unsigned));
        begin       = Clock::now();
        auto upload = [&](const hn::Tensor& input,
                          DeviceBytes&      device,
                          hipDataType       type,
                          size_t            storedRows,
                          size_t            storedCols,
                          size_t            ld,
                          size_t            stride) {
            const size_t bytes = device_element_size(type);
            DeviceMatrix layout{device.data,
                                type,
                                int64_t(storedRows),
                                int64_t(storedCols),
                                int64_t(ld),
                                int64_t(stride)};
            requireHip(poison_padding_device(layout, batches, batches * stride, nullptr));
            for(size_t batch = 0; batch < size_t(batches); ++batch)
                requireHip(
                    hipMemcpy2D(static_cast<std::byte*>(device.data) + batch * stride * bytes,
                                ld * bytes,
                                input.rawEncodedBackingStorage().data()
                                    + batch * storedRows * storedCols * bytes,
                                storedRows * bytes,
                                storedRows * bytes,
                                storedCols,
                                hipMemcpyHostToDevice));
        };
        upload(a, da, HIP_R_16BF, transA ? k : m, transA ? m : k, lda, strideA);
        upload(b, db, HIP_R_16BF, transB ? n : k, transB ? k : n, ldb, strideB);
        upload(c, dc, HIP_R_32F, m, n, ldc, strideC);
        if(!large)
            requireHip(hipMemcpy(deviceBias.data,
                                 bias.rawEncodedBackingStorage().data(),
                                 bias.rawEncodedBackingStorage().size(),
                                 hipMemcpyHostToDevice));

        const double uploadSeconds = seconds(begin);

        Descriptors descriptors;
        requireBlas(hipblasLtCreate(&descriptors.handle));
        requireBlas(
            hipblasLtMatmulDescCreate(&descriptors.operation, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        const hipblasOperation_t ta = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        const hipblasOperation_t tb = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        requireBlas(hipblasLtMatmulDescSetAttribute(
            descriptors.operation, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        requireBlas(hipblasLtMatmulDescSetAttribute(
            descriptors.operation, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        if(!large)
        {
            const hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
            const hipDataType         biasType = HIP_R_32F;
            requireBlas(hipblasLtMatmulDescSetAttribute(descriptors.operation,
                                                        HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                        &epilogue,
                                                        sizeof(epilogue)));
            requireBlas(hipblasLtMatmulDescSetAttribute(descriptors.operation,
                                                        HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                        &biasType,
                                                        sizeof(biasType)));
            requireBlas(hipblasLtMatmulDescSetAttribute(descriptors.operation,
                                                        HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                        &deviceBias.data,
                                                        sizeof(deviceBias.data)));
        }
        const std::array<size_t, 4>  rows{transA ? k : m, transB ? n : k, m, m};
        const std::array<size_t, 4>  cols{transA ? m : k, transB ? k : n, n, n};
        const std::array<size_t, 4>  leading{lda, ldb, ldc, ldd};
        const std::array<int64_t, 4> strides{
            int64_t(strideA), int64_t(strideB), int64_t(strideC), int64_t(strideD)};
        for(size_t i = 0; i < 4; ++i)
        {
            requireBlas(hipblasLtMatrixLayoutCreate(&descriptors.layouts[i],
                                                    i < 2 ? HIP_R_16BF : HIP_R_32F,
                                                    rows[i],
                                                    cols[i],
                                                    leading[i]));
            requireBlas(hipblasLtMatrixLayoutSetAttribute(descriptors.layouts[i],
                                                          HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                          &batches,
                                                          sizeof(batches)));
            requireBlas(
                hipblasLtMatrixLayoutSetAttribute(descriptors.layouts[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &strides[i],
                                                  sizeof(strides[i])));
        }
        requireBlas(hipblasLtMatmulPreferenceCreate(&descriptors.preference));
        requireBlas(hipblasLtMatmulPreferenceSetAttribute(descriptors.preference,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &workspaceBytes,
                                                          sizeof(workspaceBytes)));
        std::array<hipblasLtMatmulHeuristicResult_t, 4> solutions{};
        int                                             count = 0;
        requireBlas(hipblasLtMatmulAlgoGetHeuristic(descriptors.handle,
                                                    descriptors.operation,
                                                    descriptors.layouts[0],
                                                    descriptors.layouts[1],
                                                    descriptors.layouts[2],
                                                    descriptors.layouts[3],
                                                    descriptors.preference,
                                                    solutions.size(),
                                                    solutions.data(),
                                                    &count));
        ASSERT_EQ(count, 4) << "The acceptance fixture requires four supported solutions.";
        std::set<int> ids;
        for(auto& solution : solutions)
        {
            ASSERT_EQ(solution.state, HIPBLAS_STATUS_SUCCESS);
            ids.insert(hipblaslt_ext::getIndexFromAlgo(solution.algo));
        }
        ASSERT_EQ(ids.size(), 4);
        const float alpha = 2, beta = -1;
        size_t      launches = 0, expectedFailures = 0, compared = 0, readbackBytes = 0;
        double      gemmSeconds = 0, checkSeconds = 0, readbackSeconds = 0, hostCheckSeconds = 0,
               projectionSeconds = 0, maxTolerance = 0;
        size_t exceptionElements = 0, diagnosticElements = 0, verificationDeviceBytes = 0,
               checkTensorBytes = 0;
        // First a clean run, then an injection run. References are prepared once.
        for(bool inject : {false, true})
            for(auto& solution : solutions)
                for(int iteration = 0; iteration < 4; ++iteration)
                {
                    SCOPED_TRACE(::testing::Message()
                                 << "solution=" << hipblaslt_ext::getIndexFromAlgo(solution.algo)
                                 << " iteration=" << iteration << " inject=" << inject
                                 << " floating=" << floating);
                    requireHip(hipMemset(dd.data, 0xff, size_t(batches) * strideD * sizeof(float)));
                    begin = Clock::now();
                    requireBlas(hipblasLtMatmul(descriptors.handle,
                                                descriptors.operation,
                                                &alpha,
                                                da.data,
                                                descriptors.layouts[0],
                                                db.data,
                                                descriptors.layouts[1],
                                                &beta,
                                                dc.data,
                                                descriptors.layouts[2],
                                                dd.data,
                                                descriptors.layouts[3],
                                                &solution.algo,
                                                workspace.data,
                                                workspaceBytes,
                                                nullptr));
                    requireHip(hipDeviceSynchronize());
                    gemmSeconds += seconds(begin);
                    ++launches;
                    const bool corrupted = inject && iteration == 2;
                    if(corrupted)
                    {
                        const size_t offset = (batches - 1) * strideD + (n - 3) * ldd + m - 2;
                        float        value;
                        requireHip(hipMemcpy(&value,
                                             static_cast<float*>(dd.data) + offset,
                                             sizeof(value),
                                             hipMemcpyDeviceToHost));
                        const float original = value;
                        value += floating ? 16.0f : 1.0f;
                        ASSERT_NE(value, original);
                        requireHip(hipMemcpy(static_cast<float*>(dd.data) + offset,
                                             &value,
                                             sizeof(value),
                                             hipMemcpyHostToDevice));
                    }
                    begin = Clock::now();
                    requireHip(hipMemset(badPadding.data, 0, sizeof(unsigned)));
                    hipLaunchKernelGGL(checkPadding,
                                       dim3(256),
                                       dim3(256),
                                       0,
                                       nullptr,
                                       static_cast<const uint32_t*>(dd.data),
                                       size_t(batches) * strideD,
                                       m,
                                       n,
                                       ldd,
                                       strideD,
                                       static_cast<unsigned*>(badPadding.data));
                    requireHip(hipGetLastError());
                    unsigned paddingErrors = 0;
                    requireHip(hipMemcpy(&paddingErrors,
                                         badPadding.data,
                                         sizeof(paddingErrors),
                                         hipMemcpyDeviceToHost));
                    ASSERT_EQ(paddingErrors, 0U);
                    size_t mismatches = 0;
                    for(size_t batch = 0; batch < size_t(batches); ++batch)
                        for(size_t col = 0; col < n; col += 61)
                        {
                            const size_t width = std::min(size_t(61), n - col);
                            hn::Tensor   observed(
                                hn::ScalarType::Float32,
                                hn::Layout(hn::Shape{m, width}, {1, ptrdiff_t(m)}, 0));
                            auto readbackBegin = Clock::now();
                            requireHip(hipMemcpy2D(observed.rawEncodedBackingStorage().data(),
                                                   m * sizeof(float),
                                                   static_cast<float*>(dd.data) + batch * strideD
                                                       + col * ldd,
                                                   ldd * sizeof(float),
                                                   m * sizeof(float),
                                                   width,
                                                   hipMemcpyDeviceToHost));
                            readbackSeconds += seconds(readbackBegin);
                            readbackBytes += m * width * sizeof(float);
                            const auto              hostBegin = Clock::now();
                            hn::PeriodicMatmulCheck report;
                            report.status = hn::VerificationStatus::Passed;
                            if(method == CheckMethod::Periodic)
                            {
                                hn::EpilogueOptions epilogue(hn::ScalarType::Float32);
                                epilogue.inputScale  = hn::Tensor(alpha);
                                epilogue.addendScale = hn::Tensor(beta);
                                epilogue.addend      = c.shareStorageWithLayout(
                                    hn::Layout(hn::Shape{m, width},
                                               {1, ptrdiff_t(m)},
                                               batch * hostStrideC + col * m));
                                if(!large)
                                    epilogue.bias = bias;
                                report = plans[batch].checkRegion(
                                    observed, std::array<size_t, 2>{0, col}, epilogue);
                                // Two F64 working regions plus the F32 observation;
                                // the plan's 8 MiB checker budget covers the F64 regions.
                                EXPECT_LE(2 * m * width * sizeof(double), 8U * 1024 * 1024);
                                checkTensorBytes
                                    = std::max(checkTensorBytes,
                                               m * width * (2 * sizeof(double) + sizeof(float)));
                            }
                            else if(method == CheckMethod::DenseBaseline)
                            {
                                const auto expected
                                    = fullReference[batch].shareStorageWithLayout(hn::Layout(
                                        hn::Shape{m, width}, {ptrdiff_t(n), 1}, ptrdiff_t(col)));
                                report.comparison
                                    = hn::compare(observed, expected, hn::nearComparisonOptions(0));
                                for(auto& mismatch : report.comparison.reportedMismatches)
                                    mismatch.coordinates[1] += col;
                                report.status = report.comparison.passed()
                                                    ? hn::VerificationStatus::Passed
                                                    : hn::VerificationStatus::Mismatch;
                                checkTensorBytes
                                    = std::max(checkTensorBytes, m * width * sizeof(float));
                            }
                            ASSERT_TRUE(report.status == hn::VerificationStatus::Passed
                                        || report.status == hn::VerificationStatus::Mismatch)
                                << report.reason;
                            compared += report.comparison.compared;
                            mismatches += report.comparison.mismatches;
                            maxTolerance = std::max(maxTolerance, report.absoluteTolerance);
                            for(const auto& mismatch : report.comparison.reportedMismatches)
                            {
                                EXPECT_TRUE(corrupted);
                                EXPECT_EQ(batch, size_t(batches - 1));
                                EXPECT_EQ(mismatch.coordinates,
                                          (std::vector<size_t>{m - 2, n - 3}));
                            }
                            if(!large)
                                for(size_t i = 0; i < m; ++i)
                                    for(size_t j = 0; j < width; ++j)
                                    {
                                        const bool injectedHere = corrupted
                                                                  && batch == size_t(batches - 1)
                                                                  && i == m - 2 && col + j == n - 3;
                                        const double error
                                            = std::abs(observed.loadAs<double>({i, j})
                                                       - dense[(batch * m + i) * n + col + j]);
                                        EXPECT_EQ(error <= report.absoluteTolerance, !injectedHere);
                                    }
                            hostCheckSeconds += seconds(hostBegin);
                        }
                    EXPECT_EQ(mismatches, corrupted ? 1U : 0U);
                    expectedFailures += mismatches != 0;
                    checkSeconds += seconds(begin);
                }
        EXPECT_EQ(launches, 32U);
        EXPECT_EQ(expectedFailures, 4U);
        EXPECT_EQ(compared, launches * batches * m * n);
        EXPECT_LT(maxTolerance, floating ? 16.0 : 1.0);
        if(large)
            EXPECT_TRUE(dense.empty());
        if(method != CheckMethod::DenseBaseline)
        {
            EXPECT_EQ(fullReferenceCalls, 0U);
            EXPECT_TRUE(fullReference.empty());
        }
        std::ostringstream evidence;
        const char* methodName = method == CheckMethod::Periodic ? "periodic" : "dense_baseline";
        evidence << std::setprecision(9) << methodName << " acceptance: floating=" << floating
                 << " shape=" << m << 'x' << n << 'x' << k << " batches=" << batches
                 << " preparations=" << (plans.size() + fullReference.size())
                 << " full_reference_calls=" << fullReferenceCalls << " launches=" << launches
                 << " injected_mismatches=" << expectedFailures << " compared=" << compared
                 << " reference_multiply_adds=" << referenceWork
                 << " product_bytes=" << productBytes << " readback_bytes=" << readbackBytes
                 << " max_tolerance=" << maxTolerance << " generation_s=" << generationSeconds
                 << " preparation_s=" << preparationSeconds << " upload_s=" << uploadSeconds
                 << " gemm_s=" << gemmSeconds << " check_s=" << checkSeconds
                 << " projection_s=" << projectionSeconds
                 << " output_readback_s=" << readbackSeconds << " host_check_s=" << hostCheckSeconds
                 << " exception_elements=" << exceptionElements
                 << " diagnostic_elements=" << diagnosticElements
                 << " check_tensor_bytes=" << checkTensorBytes
                 << " verification_device_bytes=" << verificationDeviceBytes << " solutions=";
        for(int id : ids)
            evidence << id << ',';
        std::cout << evidence.str() << '\n';
    }

    TEST(HostNumericsPeriodicGemm_pre_checkin, ExactSmall)
    {
        for(bool a : {false, true})
            for(bool b : {false, true})
                runAcceptance(false, false, a, b);
    }
    TEST(HostNumericsPeriodicGemm_pre_checkin, FloatingSmall)
    {
        for(bool a : {false, true})
            for(bool b : {false, true})
                runAcceptance(true, false, a, b);
    }
    TEST(HostNumericsPeriodicGemm, DISABLED_LargeExact)
    {
        runAcceptance(false, true, false, true);
    }
    TEST(HostNumericsPeriodicGemm, DISABLED_LargeFloating)
    {
        runAcceptance(true, true, false, true);
    }
    // Separately selected paired baseline. Its dense reference is prepared once
    // and reused for the same solutions/iterations as the large periodic checks.
    TEST(HostNumericsPeriodicGemm, DISABLED_LargeDenseBaseline)
    {
        runAcceptance(false, true, false, true, CheckMethod::DenseBaseline);
    }

}
