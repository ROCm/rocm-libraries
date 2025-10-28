/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#include <miopen/datatype.hpp>
#include <gtest/gtest.h>

#include "get_handle.hpp"
#include "verify.hpp"

#define PERF_ENABLE 1
#if PERF_ENABLE
#include "perf_helper.hpp"
#endif

struct TensorsConfig
{
    std::vector<int> aclens;
    std::vector<int> acstrides;
    std::vector<int> blens;
    std::vector<int> bstrides;
};

template <typename T>
std::vector<TensorsConfig> TensorsConfigs()
{
    std::vector<TensorsConfig> configs;
#define MiB (1024ul * 1024ul)

#if PERF_ENABLE
    for(int N = (1 * MiB); N <= (1024 * MiB); N *= 2)
    {
        configs.push_back({{1, 1, N}, {N, N, 1}, {1, 1, N}, {N, N, 1}});
    }
    return configs;
#else
    int C = 4;
    int N = 20 * 1024;
    configs.push_back({{1, C, N}, {N * C, N, 1}, {1, C, N}, {N * C, N, 1}});
    C = 1;
    N = 64 * MiB;
    configs.push_back({{1, C, N}, {N * C, N, 1}, {1, C, N}, {N * C, N, 1}});
    N = 256 * MiB;
    configs.push_back({{1, C, N}, {N * C, N, 1}, {1, C, N}, {N * C, N, 1}});
    N = 1024 * MiB;
    configs.push_back({{1, C, N}, {N * C, N, 1}, {1, C, N}, {N * C, N, 1}});
    return configs;
#endif
}

template <typename T>
struct Op2dTensorLiteTest
    : public ::testing::TestWithParam<std::tuple<TensorsConfig, float, float, float>>
{
protected:
    void SetUp() override
    {
        auto&& handle                                 = get_handle();
        std::tie(tensorsConfig, alpha0, alpha1, beta) = GetParam();

        data_type = miopen_type<T>{};

        // Generate elements in tensors
        tensA = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides}.generate(
            tensor_elem_gen_integer{17});
        tensB = tensor<T>{tensorsConfig.blens, tensorsConfig.bstrides}.generate(
            tensor_elem_gen_integer{17});
        tensC = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides}.generate(
            [](auto...) { return 1; });

        // Write the device tensors
        tensA_dev = handle.Write(tensA.data);
        tensB_dev = handle.Write(tensB.data);

        // Allocate output tensors for OCL and HIP
        tensC_ocl = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides};
        tensC_hip = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides};

        // Prepare all parameters needed for kernel
        auto first_not_one = std::find_if(
            tensorsConfig.blens.rbegin(), tensorsConfig.blens.rend(), [](int i) { return i != 1; });
        auto d = std::distance(tensorsConfig.blens.begin(), first_not_one.base());

        int num_wg = first_not_one != tensorsConfig.blens.rend()
                         ? static_cast<int>(*first_not_one == 0 ? 1 : *first_not_one)
                         : 1;
        for(int i = (d - 2); i >= 0; i--)
        {
            if(tensorsConfig.blens[i] != 1)
            {
                num_wg *= tensorsConfig.blens[i];
            }
        }

        long max_num_wg = 4096;
        num_wg          = num_wg > max_num_wg ? max_num_wg : num_wg;

        auto len     = tensorsConfig.aclens[2];
        auto RD_BLCK = (len % 4 == 0) ? 4 : (len % 2 == 0) ? 2 : 1;

        const std::string MIOPEN_TYPE = miopen::GetDataType(data_type);
        const std::string READ_TYPE =
            (RD_BLCK == 1) ? MIOPEN_TYPE : MIOPEN_TYPE + std::to_string(RD_BLCK);

        params = " -DMIOPEN_TYPE=" + MIOPEN_TYPE + " -DREAD_TYPE=" + READ_TYPE +
                 " -DRD_BLCK=" + std::to_string(RD_BLCK);
        params += " -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE";

        total_work         = std::max(len / RD_BLCK, 1);
        long local_threads = 256;
        long grp_sz        = (total_work + local_threads - 1) / local_threads;
        grp_sz             = std::min(max_num_wg, grp_sz);
        long glb_sz        = local_threads * grp_sz;

        total_work2         = tensorsConfig.aclens[1];
        long local_threads2 = 64;
        long grp_sz2        = (total_work2 + local_threads2 - 1) / local_threads2;
        grp_sz2             = std::min((max_num_wg / grp_sz), grp_sz2);
        long glb_sz2        = local_threads2 * grp_sz2;

        vld = {local_threads, 1, 1};
        vgd = {glb_sz, glb_sz2, 1};

        network_config += std::to_string(data_type) + "-miopenTensorOpAdd-";

        use_beta = !miopen::float_equal(beta, 0);
        use_bias = (tensorsConfig.blens[1] == 1);
    }

    void runOCL() // run OCL kernel
    {
        auto&& handle = get_handle();
        // Write data to device tensor
        tensC_dev = handle.Write(tensC.data);

        std::string paramsOCL =
            params + " " + miopen::GetDataTypeKBP(data_type).GenerateFor(miopen::kbp::OpenCL{});

        std::string program_name       = "MIOpenTensorKernels.cl";
        std::string network_config_ocl = network_config + "-ocl";

        handle.AddKernel("Op2dTensorLite",
                         network_config_ocl,
                         program_name,
                         "Op2dTensorLite",
                         vld,
                         vgd,
                         paramsOCL)(tensA_dev.get(),
                                    tensorsConfig.acstrides[0],
                                    tensB_dev.get(),
                                    tensorsConfig.bstrides[0],
                                    tensC_dev.get(),
                                    tensorsConfig.acstrides[0],
                                    alpha0,
                                    alpha1,
                                    beta,
                                    uint64_t(0),
                                    uint64_t(0),
                                    uint64_t(0),
                                    total_work,
                                    total_work2,
                                    use_beta,
                                    use_bias);

        tensC_ocl.data = handle.Read<T>(tensC_dev, tensC_ocl.data.size());

#if PERF_ENABLE
        ph.perfTest(handle,
                    "Op2dTensorLite",
                    network_config_ocl,
                    tensA_dev.get(),
                    tensorsConfig.acstrides[0],
                    tensB_dev.get(),
                    tensorsConfig.bstrides[0],
                    tensC_dev.get(),
                    tensorsConfig.acstrides[0],
                    alpha0,
                    alpha1,
                    beta,
                    uint64_t(0),
                    uint64_t(0),
                    uint64_t(0),
                    total_work,
                    total_work2,
                    use_beta,
                    use_bias);
#endif
    }

    void runHIP() // run HIP kernel
    {
        auto&& handle = get_handle();
        tensC_dev     = handle.Write(tensC.data);

        std::string paramsHIP =
            params + " " + miopen::GetDataTypeKBP(data_type).GenerateFor(miopen::kbp::HIP{});

        std::string program_name       = "MIOpenTensorKernelsHip.cpp";
        std::string network_config_hip = network_config + "-hip";

        handle.AddKernel("Op2dTensorLite",
                         network_config_hip,
                         program_name,
                         "Op2dTensorLite",
                         vld,
                         vgd,
                         paramsHIP)(tensA_dev.get(),
                                    tensorsConfig.acstrides[0],
                                    tensB_dev.get(),
                                    tensorsConfig.bstrides[0],
                                    tensC_dev.get(),
                                    tensorsConfig.acstrides[0],
                                    alpha0,
                                    alpha1,
                                    beta,
                                    uint64_t(0),
                                    uint64_t(0),
                                    uint64_t(0),
                                    total_work,
                                    total_work2,
                                    use_beta,
                                    use_bias);

        tensC_hip.data = handle.Read<T>(tensC_dev, tensC_hip.data.size());

#if PERF_ENABLE
        ph.perfTest(handle,
                    "Op2dTensorLite",
                    network_config_hip,
                    tensA_dev.get(),
                    tensorsConfig.acstrides[0],
                    tensB_dev.get(),
                    tensorsConfig.bstrides[0],
                    tensC_dev.get(),
                    tensorsConfig.acstrides[0],
                    alpha0,
                    alpha1,
                    beta,
                    uint64_t(0),
                    uint64_t(0),
                    uint64_t(0),
                    total_work,
                    total_work2,
                    use_beta,
                    use_bias);
#endif
    }

    void verify()
    {
        auto error = miopen::rms_range(tensC_ocl, tensC_hip);
        EXPECT_TRUE(error == 0) << "GPU outputs do not match each other. Error: " << error;
    }

    void TearDown() override
    {
#if PERF_ENABLE
        std::string stats{};
        stats += "_aclens_" + std::to_string(tensorsConfig.aclens[0]) + "_" +
                 std::to_string(tensorsConfig.aclens[1]) + "_" +
                 std::to_string(tensorsConfig.aclens[2]) + "_" +
                 std::to_string(tensorsConfig.aclens[3]) + "_acstrides_" +
                 std::to_string(tensorsConfig.acstrides[0]) + "_" +
                 std::to_string(tensorsConfig.acstrides[1]) + "_" +
                 std::to_string(tensorsConfig.acstrides[2]) + "_" +
                 std::to_string(tensorsConfig.acstrides[3]);
        stats += "_blens_" + std::to_string(tensorsConfig.blens[0]) + "_" +
                 std::to_string(tensorsConfig.blens[1]) + "_" +
                 std::to_string(tensorsConfig.blens[2]) + "_" +
                 std::to_string(tensorsConfig.blens[3]) + "_bstrides_" +
                 std::to_string(tensorsConfig.bstrides[0]) + "_" +
                 std::to_string(tensorsConfig.bstrides[1]) + "_" +
                 std::to_string(tensorsConfig.bstrides[2]) + "_" +
                 std::to_string(tensorsConfig.bstrides[3]);
        stats += "_alpha0_" + std::to_string(alpha0) + "_alpha1_" + std::to_string(alpha1) +
                 "_beta_" + std::to_string(beta) + "_" + miopen::GetDataType(data_type);

        ph.writeStatsToCSV("tensor_2d_lite.csv", stats);
#endif
    }

    std::string network_config{};
    std::string params{};
    std::vector<size_t> vld, vgd;

    tensor<T> tensA;
    tensor<T> tensB;
    tensor<T> tensC;
    tensor<T> tensC_ocl;
    tensor<T> tensC_hip;

    miopenDataType_t data_type;

    miopen::Allocator::ManageDataPtr tensA_dev;
    miopen::Allocator::ManageDataPtr tensB_dev;
    miopen::Allocator::ManageDataPtr tensC_dev;

    TensorsConfig tensorsConfig;
    T alpha0, alpha1, beta;
    long total_work, total_work2;
    int use_beta, use_bias;

#if PERF_ENABLE
    PerfHelper ph;
#endif
};

using GPU_Op2dTensorLiteTest_FP32 = Op2dTensorLiteTest<float>;

TEST_P(GPU_Op2dTensorLiteTest_FP32, PortTest)
{
    // run OCL kernel
    runOCL();
    // run HIP kernel
    runHIP();
    // verify if the output tensors are same
    verify();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Op2dTensorLiteTest_FP32,
                         testing::Combine(testing::ValuesIn(TensorsConfigs<float>()),
                                          testing::Values(1.0f),
                                          testing::Values(1.0f),
                                          testing::Values(0.0f, 1.0f)));
