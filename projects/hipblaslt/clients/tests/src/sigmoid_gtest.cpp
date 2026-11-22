// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "sigmoid_test_observer.hpp"
#include "hipblaslt_data.hpp"
#include "hipblaslt_test.hpp"
#include "hipBuffer.hpp"
#include "norm.hpp"
#include "utility.hpp"
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

// Defined by matmul_gtest.cpp's unchanged shared client implementation.
void testing_matmul(const Arguments& arg);

thread_local SigmoidEpilogueObservation* sigmoid_epilogue_observation = nullptr;

hipblasStatus_t sigmoid_test_set_attribute(hipblasLtMatmulDesc_t           desc,
                                          hipblasLtMatmulDescAttributes_t attr,
                                          const void*                     value,
                                          size_t                          size)
{
    const auto status = hipblasLtMatmulDescSetAttribute(desc, attr, value, size);
    if(status == HIPBLAS_STATUS_SUCCESS && sigmoid_epilogue_observation
       && attr == HIPBLASLT_MATMUL_DESC_EPILOGUE && size == sizeof(hipblasLtEpilogue_t))
    {
        sigmoid_epilogue_observation->sent = true;
        std::memcpy(&sigmoid_epilogue_observation->epilogue, value, size);
    }
    return status;
}

namespace
{
    struct ObserveEpilogue
    {
        SigmoidEpilogueObservation result;
        ObserveEpilogue() { sigmoid_epilogue_observation = &result; }
        ~ObserveEpilogue() { sigmoid_epilogue_observation = nullptr; }
    };

    struct sigmoid_test : RocBlasLt_Test<sigmoid_test, hipblaslt_test_valid>
    {
        static bool type_filter(const Arguments& arg)
        {
            return hipblaslt_client_global_filters(arg) && arg.a_type == HIP_R_32F
                   && arg.b_type == HIP_R_32F && arg.c_type == HIP_R_32F
                   && arg.d_type == HIP_R_32F && arg.compute_type == HIPBLAS_COMPUTE_32F;
        }
        static bool function_filter(const Arguments& arg)
        {
            return !std::strcmp(arg.function, "sigmoid_client")
                   || !std::strcmp(arg.function, "sigmoid_api");
        }
        static std::string name_suffix(const Arguments& arg)
        {
            RocBlasLt_TestName<sigmoid_test> name(arg.name);
            name << arg.M[0] << '_' << arg.N[0] << '_' << arg.K[0] << '_' << arg.beta;
            return std::move(name);
        }
    };

    void check_direct_sigmoid(const Arguments& arg)
    {
        const int64_t m = arg.M[0], n = arg.N[0], k = arg.K[0];
        const float alpha = arg.alpha, beta = arg.beta;
        std::vector<float> a(m * k, 0), b(k * n, 0), c(m * n), d(m * n);
        for(int64_t i = 0; i < m; ++i)
            a[i] = static_cast<float>(i % 17 - 8);
        for(int64_t j = 0; j < n; ++j)
        {
            b[j * k] = j % 2 ? -1.0f : 1.0f;
            for(int64_t i = 0; i < m; ++i)
                c[i + j * m] = static_cast<float>(j % 3 - 1) * 0.5f;
        }

        HipDeviceBuffer da(HIP_R_32F, a.size()), db(HIP_R_32F, b.size());
        HipDeviceBuffer dc(HIP_R_32F, c.size()), dd(HIP_R_32F, d.size());
        for(auto* buffer : {&da, &db, &dc, &dd})
            ASSERT_EQ(buffer->memcheck(), hipSuccess);
        ASSERT_EQ(hipMemcpy(da.buf(), a.data(), da.getNumBytes(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(db.buf(), b.data(), db.getNumBytes(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dc.buf(), c.data(), dc.getNumBytes(), hipMemcpyHostToDevice), hipSuccess);

        hipblaslt_local_handle handle;
        hipblaslt_local_matmul_descr desc(HIPBLAS_OP_N, HIPBLAS_OP_N,
                                         HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblaslt_local_matrix_layout la(m, k, m, HIP_R_32F), lb(k, n, k, HIP_R_32F);
        hipblaslt_local_matrix_layout ld(m, n, m, HIP_R_32F);
        hipblaslt_local_preference pref;
        ASSERT_EQ(desc.status(), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(la.status(), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(lb.status(), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(ld.status(), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(pref.status(), HIPBLAS_STATUS_SUCCESS);
        const hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_SIGMOID;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtEpilogue_t readback = HIPBLASLT_EPILOGUE_DEFAULT;
        size_t written = 0;
        ASSERT_EQ(hipblasLtMatmulDescGetAttribute(desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                &readback, sizeof(readback), &written),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(readback, HIPBLASLT_EPILOGUE_SIGMOID);
        const size_t workspace_bytes = 32 * 1024 * 1024;
        ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspace_bytes, sizeof(workspace_bytes)),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatmulHeuristicResult_t result{};
        int count = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, ld, ld, pref,
                                                1, &result, &count), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(count, 1) << "No sigmoid solution in the loaded stock library";
        ASSERT_EQ(result.state, HIPBLAS_STATUS_SUCCESS);
        HipDeviceBuffer workspace(HIP_R_8I, workspace_bytes);
        ASSERT_EQ(workspace.memcheck(), hipSuccess);
        ASSERT_EQ(hipblasLtMatmul(handle, desc, &alpha, da.buf(), la, db.buf(), lb,
                                 &beta, dc.buf(), ld, dd.buf(), ld, &result.algo,
                                 workspace.buf(), workspace_bytes, nullptr), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(hipMemcpy(d.data(), dd.buf(), dd.getNumBytes(), hipMemcpyDeviceToHost), hipSuccess);

        // Independent full scalar reference. No client activation helper or GPU
        // result participates in the oracle; negative, zero and positive values
        // distinguish sigmoid from DEFAULT, relu and swish.
        double squared_error = 0, squared_reference = 0;
        for(int64_t j = 0; j < n; ++j)
            for(int64_t i = 0; i < m; ++i)
            {
                double sum = 0;
                for(int64_t p = 0; p < k; ++p)
                    sum += static_cast<double>(a[i + p * m]) * b[p + j * k];
                const double x = alpha * sum + beta * c[i + j * m];
                const double reference = 1.0 / (1.0 + std::exp(-x));
                const double delta = static_cast<double>(d[i + j * m]) - reference;
                squared_error += delta * delta;
                squared_reference += reference * reference;
            }
        const double norm_error = std::sqrt(squared_error / squared_reference);
        const int solution_index = hipblaslt_ext::getIndexFromAlgo(result.algo);
        hipblaslt_cout << "sigmoid_api solution_index=" << solution_index
                      << " norm_error=" << norm_error << " tolerance=" << norm_tolerance(HIP_R_32F)
                      << std::endl;
        ASSERT_TRUE(std::isfinite(norm_error)) << "sigmoid norm_error=" << norm_error;
        ASSERT_TRUE(norm_check(norm_error, HIP_R_32F, HIPBLAS_COMPUTE_32F, HIP_R_32F, HIP_R_32F))
            << "sigmoid norm_error=" << norm_error << " tolerance=" << norm_tolerance(HIP_R_32F);
    }

    TEST_P(sigmoid_test, dense_forward)
    {
        const auto& arg = GetParam();
        ASSERT_EQ(arg.activation_type, hipblaslt_activation_type::sigmoid);
        ASSERT_EQ(arg.bias_vector, 0);
        ASSERT_EQ(arg.norm_check, 1);
        ASSERT_EQ(arg.norm_check_assert, 1);
        ASSERT_EQ(arg.unit_check, 0);
        ASSERT_EQ(arg.a_type, HIP_R_32F);
        ASSERT_EQ(arg.batch_count, 1);
        if(!std::strcmp(arg.function, "sigmoid_client"))
        {
            // Feature: the real client must send sigmoid and its own reference
            // must apply it. Removing either dispatch branch breaks this test.
            ObserveEpilogue observation;
            testing_matmul(arg);
            ASSERT_TRUE(observation.result.sent) << "Client never sent a sigmoid epilogue";
            ASSERT_EQ(observation.result.epilogue, HIPBLASLT_EPILOGUE_SIGMOID)
                << "Client sent DEFAULT/BIAS instead of sigmoid";
        }
        else
        {
            // Preservation: the runtime sigmoid enum already reaches a stock
            // activation kernel. Mapping it to None breaks this scalar verdict.
            check_direct_sigmoid(arg);
        }
    }
    INSTANTIATE_TEST_CATEGORIES(sigmoid_test);
} // namespace
