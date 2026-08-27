/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#include "utility.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/host_validation/GroupedGemmDataInitialization.hpp>
#include <hipblaslt/host_validation/Types.hpp>
#include <hipblaslt_arguments.hpp>
#include <hipblaslt_vector.hpp>
#include <iostream>
#include <limits>
#include <roc/host_validation/validation.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif

// default sizes
#define DIM1 1024
#define DIM2 1024
#define DIM3 1024
#define BATCH_COUNT 1
#define ALPHA 1
#define BETA 0

typedef enum _ActivationType
{
    NONE  = 0,
    RELU  = 1,
    GELU  = 2,
    SWISH = 3,
    CLAMP = 4,
} ActivationType;

inline const char* ToString(ActivationType act)
{
    switch(act)
    {
    case NONE:
        return "none";
    case RELU:
        return "relu";
    case GELU:
        return "gelu";
    case SWISH:
        return "swish";
    case CLAMP:
        return "clamp";
    default:
        return "[Unknown Activation Type]";
    }
}

roc::host_validation::Activation toHostValidationActivation(ActivationType activation)
{
    switch(activation)
    {
    case ActivationType::NONE:
        return roc::host_validation::Activation::None;
    case ActivationType::RELU:
        return roc::host_validation::Activation::Relu;
    case ActivationType::GELU:
        return roc::host_validation::Activation::Gelu;
    case ActivationType::SWISH:
        return roc::host_validation::Activation::Swish;
    case ActivationType::CLAMP:
        return roc::host_validation::Activation::Clamp;
    }

    throw std::invalid_argument("Unsupported grouped GEMM activation.");
}

// cppcheck-suppress constParameter
static void show_usage(char* argv[])
{
    std::cerr
        << "Usage: " << argv[0] << " <options>\n"
        << "options:\n"
        << "\t-h, --help\t\t\t\tShow this help message\n"
        << "\t-v, --verbose\t\t\t\tVerbose output\n"
        << "\t-V, --validate\t\t\t\tVerify results\n"
        << "\t--initialization \t\t\t\tInitialize matrix data. Options: rand_int, trig_float, "
           "hpl(floating), special, zero. (default is hpl)\n"
        << "\t--bench_count\t\t\t\tNumber of benchmark runs (default is 1)\n"
        << "\t--sync_count\t\t\t\tNumber of sync runs (default is 1)\n"
        << "\t--in_datatype \t\tdatatype \tGEMM_STRIDED argument in: fp32, fp16, bf16 (default is "
           "fp32)\n"
        << "\t--out_datatype \t\tdatatype \tGEMM_STRIDED argument out: fp32, fp16, bf16 (default "
           "is fp32)\n"
        << "\t--trans_a \t\ttrans_a \tGEMM_STRIDED argument trans_a: N or T (default is N)\n"
        << "\t--trans_b \t\ttrans_b \tGEMM_STRIDED argument trans_b: N or T (default is N)\n"
        << "\t-m \t\t\tm \t\tGEMM_STRIDED argument m (Since mk is fixed, only the first m is "
           "taken)\n"
        << "\t-n \t\t\tn \t\tGEMM_STRIDED argument n\n"
        << "\t-k \t\t\tk \t\tGEMM_STRIDED argument k (Since mk is fixed, only the first m is "
           "taken)\n"
        << "\t--batch_count \t\tbatch_count\tGEMM_STRIDED argument batch_count\n"
        << "\t--lda \t\t\tlda \t\tGEMM_STRIDED argument lda\n"
        << "\t--ldb \t\t\tldb \t\tGEMM_STRIDED argument ldb\n"
        << "\t--ldc \t\t\tldc \t\tGEMM_STRIDED argument ldc\n"
        << "\t--ldd \t\t\tldd \t\tGEMM_STRIDED argument ldd\n"
        << "\t--stride_a \t\tstride_a \tGEMM_STRIDED argument stride_a\n"
        << "\t--stride_b \t\tstride_b \tGEMM_STRIDED argument stride_b\n"
        << "\t--stride_c \t\tstride_c \tGEMM_STRIDED argument stride_c\n"
        << "\t--stride_d \t\tstride_d \tGEMM_STRIDED argument stride_d\n"
        << "\t--alpha \t\talpha \t\tGEMM_STRIDED argument alpha (default is 1)\n"
        << "\t--beta \t\t\tbeta \t\tGEMM_STRIDED argument beta (default is 0)\n"
        << "\t--act \t\t\tact \t\tGEMM_STRIDED set activation type: relu, gelu, swish, clamp, none "
           "(default is "
           "none)\n"
        << "\t--bias \t\t\tbias \t\tGEMM_STRIDED set bias: 0 or 1 (default is 0)\n"
        << std::endl;
}

static int parse_arguments(int                          argc,
                           char*                        argv[],
                           hipDataType&                 in_datatype,
                           hipDataType&                 out_datatype,
                           std::vector<int64_t>&        m,
                           std::vector<int64_t>&        n,
                           std::vector<int64_t>&        k,
                           std::vector<int64_t>&        lda,
                           std::vector<int64_t>&        ldb,
                           std::vector<int64_t>&        ldc,
                           std::vector<int64_t>&        ldd,
                           std::vector<int64_t>&        stride_a,
                           std::vector<int64_t>&        stride_b,
                           std::vector<int64_t>&        stride_c,
                           std::vector<int64_t>&        stride_d,
                           std::vector<int64_t>&        batch_count,
                           std::vector<float>&          alpha,
                           std::vector<float>&          beta,
                           hipblasOperation_t&          trans_a,
                           hipblasOperation_t&          trans_b,
                           std::vector<bool>&           enable_bias,
                           std::vector<ActivationType>& actType,
                           int32_t&                     bench_count,
                           int32_t&                     sync_count,
                           hipblaslt_initialization&    initialization,
                           bool&                        verbose,
                           bool&                        validate)
{
    if(argc >= 2)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
            {
                if((arg == "-h") || (arg == "--help"))
                {
                    return EXIT_FAILURE;
                }
                if((arg == "-v") || (arg == "--verbose"))
                {
                    verbose = true;
                }
                else if((arg == "-V") || (arg == "--validate"))
                {
                    validate = true;
                }
                else if(arg == "--initialization")
                {
                    std::string initializationStr = argv[++i];
                    initialization = string2hipblaslt_initialization(initializationStr);
                }
                else if(arg == "--sync_count")
                {
                    sync_count = atoi(argv[++i]);
                }
                else if(arg == "--bench_count")
                {
                    bench_count = atoi(argv[++i]);
                }
                else if((arg == "-m") && (i + 1 < argc))
                {
                    if(m.size() == 0)
                    {
                        m.push_back(atoi(argv[++i]));
                    }
                }
                else if((arg == "-n") && (i + 1 < argc))
                {
                    n.push_back(atoi(argv[++i]));
                }
                else if((arg == "-k") && (i + 1 < argc))
                {
                    if(k.size() == 0)
                    {
                        k.push_back(atoi(argv[++i]));
                    }
                }
                else if((arg == "--batch_count") && (i + 1 < argc))
                {
                    batch_count.push_back(atoi(argv[++i]));
                }
                else if((arg == "--lda") && (i + 1 < argc))
                {
                    lda.push_back(atoi(argv[++i]));
                }
                else if((arg == "--ldb") && (i + 1 < argc))
                {
                    ldb.push_back(atoi(argv[++i]));
                }
                else if((arg == "--ldc") && (i + 1 < argc))
                {
                    ldc.push_back(atoi(argv[++i]));
                }
                else if((arg == "--ldd") && (i + 1 < argc))
                {
                    ldd.push_back(atoi(argv[++i]));
                }
                else if((arg == "--stride_a") && (i + 1 < argc))
                {
                    stride_a.push_back(atoi(argv[++i]));
                }
                else if((arg == "--stride_b") && (i + 1 < argc))
                {
                    stride_b.push_back(atoi(argv[++i]));
                }
                else if((arg == "--stride_c") && (i + 1 < argc))
                {
                    stride_c.push_back(atoi(argv[++i]));
                }
                else if((arg == "--stride_d") && (i + 1 < argc))
                {
                    stride_d.push_back(atoi(argv[++i]));
                }
                else if((arg == "--alpha") && (i + 1 < argc))
                {
                    alpha.push_back(atof(argv[++i]));
                }
                else if((arg == "--beta") && (i + 1 < argc))
                {
                    beta.push_back(atof(argv[++i]));
                }
                else if((arg == "--bias") && (i + 1 < argc))
                {
                    enable_bias.push_back(atoi(argv[++i]));
                }
                else if((arg == "--act") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "relu", 4) == 0)
                        actType.push_back(ActivationType::RELU);
                    else if(strncmp(argv[i], "gelu", 4) == 0)
                        actType.push_back(ActivationType::GELU);
                    else if(strncmp(argv[i], "swish", 5) == 0)
                        actType.push_back(ActivationType::SWISH);
                    else if(strncmp(argv[i], "clamp", 5) == 0)
                        actType.push_back(ActivationType::CLAMP);
                    else if(strncmp(argv[i], "none", 4) == 0)
                        actType.push_back(ActivationType::NONE);
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--trans_a") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans_a = HIPBLAS_OP_N;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_a = HIPBLAS_OP_T;
                    }
                    else if(strncmp(argv[i], "C", 1) == 0 || strncmp(argv[i], "c", 1) == 0)
                    {
                        trans_a = HIPBLAS_OP_C;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--trans_b") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans_b = HIPBLAS_OP_N;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_b = HIPBLAS_OP_T;
                    }
                    else if(strncmp(argv[i], "C", 1) == 0 || strncmp(argv[i], "c", 1) == 0)
                    {
                        trans_b = HIPBLAS_OP_C;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--in_datatype") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "fp32", 4) == 0)
                    {
                        in_datatype = HIP_R_32F;
                    }
                    else if(strncmp(argv[i], "fp16", 4) == 0)
                    {
                        in_datatype = HIP_R_16F;
                    }
                    else if(strncmp(argv[i], "bf16", 4) == 0)
                    {
                        in_datatype = HIP_R_16BF;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--out_datatype") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "fp32", 4) == 0)
                    {
                        out_datatype = HIP_R_32F;
                    }
                    else if(strncmp(argv[i], "fp16", 4) == 0)
                    {
                        out_datatype = HIP_R_16F;
                    }
                    else if(strncmp(argv[i], "bf16", 4) == 0)
                    {
                        out_datatype = HIP_R_16BF;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else
                {
                    std::cerr << "error with " << arg << std::endl;
                    std::cerr << "do not recognize option" << std::endl << std::endl;
                    return EXIT_FAILURE;
                }
            }
            else
            {
                std::cerr << "error with " << arg << std::endl;
                std::cerr << "option must start with - or --" << std::endl << std::endl;
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

bool bad_argument(hipblasOperation_t trans_a,
                  hipblasOperation_t trans_b,
                  int64_t            m,
                  int64_t            n,
                  int64_t            k,
                  int64_t            lda,
                  int64_t            ldb,
                  int64_t            ldc,
                  int64_t            ldd,
                  int64_t            stride_a,
                  int64_t            stride_b,
                  int64_t            stride_c,
                  int64_t            stride_d,
                  int64_t            batch_count)
{
    bool argument_error = false;
    if((trans_a == HIPBLAS_OP_N) && (lda < m))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << lda << " < " << m << std::endl;
    }
    if((trans_a != HIPBLAS_OP_N) && (lda < k))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << lda << " < " << k << std::endl;
    }
    if((trans_b == HIPBLAS_OP_N) && (ldb < k))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldb = " << ldb << " < " << k << std::endl;
    }
    if((trans_b != HIPBLAS_OP_N) && (ldb < n))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldb = " << ldb << " < " << n << std::endl;
    }
    if(stride_a < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_a < 0" << std::endl;
    }
    if(stride_b < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_b < 0" << std::endl;
    }
    if(ldc < m)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldc = " << ldc << " < " << m << std::endl;
    }
    if(stride_c < n * ldc)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_c = " << stride_c << " < " << n * ldc << std::endl;
    }
    if(ldd < m)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldc = " << ldd << " < " << m << std::endl;
    }
    if(stride_d < n * ldd)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_c = " << stride_d << " < " << n * ldd << std::endl;
    }
    if(batch_count == 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument batch_count = " << batch_count << std::endl;
    }

    return argument_error;
}

__global__ void kernelUpdateN(uint32_t gemm_count, void* userArgs, int32_t* sizes_n)
{
    uint64_t id = hipBlockIdx_x * 256 + hipThreadIdx_x;

    if(id >= gemm_count)
        return;

    hipblaslt_ext::UserArguments* dUAFloat = static_cast<hipblaslt_ext::UserArguments*>(userArgs);
    dUAFloat[id].n                         = sizes_n[id];
}

template <typename Tin, typename Tout>
int test_hipblaslt(hipDataType                 in_datatype,
                   hipDataType                 out_datatype,
                   hipblasOperation_t          trans_a,
                   hipblasOperation_t          trans_b,
                   std::vector<int64_t>        m,
                   std::vector<int64_t>        n,
                   std::vector<int64_t>        k,
                   std::vector<int64_t>        lda,
                   std::vector<int64_t>        ldb,
                   std::vector<int64_t>        ldc,
                   std::vector<int64_t>        ldd,
                   std::vector<int64_t>        stride_a,
                   std::vector<int64_t>        stride_b,
                   std::vector<int64_t>        stride_c,
                   std::vector<int64_t>        stride_d,
                   std::vector<int64_t>        batch_count,
                   std::vector<float>          alpha,
                   std::vector<float>          beta,
                   std::vector<bool>           enable_bias,
                   std::vector<ActivationType> actType,
                   int32_t                     gemm_count,
                   int32_t                     bench_count,
                   int32_t                     sync_count,
                   hipblaslt_initialization    initialization,
                   bool                        validate,
                   bool                        verbose)
{
    int                  status = EXIT_SUCCESS;
    std::vector<int64_t> a_stride_1(gemm_count), a_stride_2(gemm_count), b_stride_1(gemm_count),
        b_stride_2(gemm_count);
    std::vector<int> size_a1(gemm_count), size_b1(gemm_count), size_c1(gemm_count),
        size_d1(gemm_count);

    std::vector<int> size_a(gemm_count), size_b(gemm_count), size_c(gemm_count), size_d(gemm_count),
        size_bias(gemm_count);
    std::vector<void*> da(gemm_count), db(gemm_count), dc(gemm_count), dd(gemm_count),
        d_bias(gemm_count);
    std::vector<std::vector<Tin>>   ha(gemm_count), hb(gemm_count);
    std::vector<std::vector<Tout>>  hc(gemm_count), hd(gemm_count), hd_gold(gemm_count);
    std::vector<std::vector<float>> h_bias(gemm_count);

    hipblasLtHandle_t handle;
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

    std::vector<hipblasLtEpilogue_t> epilogue(gemm_count);

    //////
    // simulate feeding arguments to groupedgemm by previous kernel
    // 0. collect sum of N
    // 1. set problem to {Ms, {sum of N, 1, 1, 1, ...}, Ks}
    // 2. get default DeviceUserArguments from grouped gemm objects
    // 3. initialize if algo changes
    // 4. launch kernel to modify Ns (simulate this by memcpy Ns to device)
    // 5. launch groupedGemm kernel
    /////
    // step 0: collect sum of n
    int                  sum_of_n = 0;
    std::vector<int64_t> sum_of_n_vec;
    std::vector<int32_t> n_vec;
    for(int i = 0; i < gemm_count; i++)
    {
        sum_of_n += n[i];
        n_vec.push_back(n[i]);
        sum_of_n_vec.push_back(1);
    }
    sum_of_n_vec[0] = sum_of_n;

    int32_t* d_n_vec = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_n_vec, sizeof(int32_t) * gemm_count));
    CHECK_HIP_ERROR(
        hipMemcpy(d_n_vec, n_vec.data(), sizeof(int32_t) * gemm_count, hipMemcpyHostToDevice));

    for(int i = 0; i < gemm_count; i++)
    {
        size_c1[i] = ldc[i] * sum_of_n;
        size_d1[i] = ldd[i] * sum_of_n;
        if(trans_a == HIPBLAS_OP_N)
        {
            a_stride_1[i] = 1;
            a_stride_2[i] = lda[i];
            size_a1[i]    = lda[i] * k[i];
        }
        else
        {
            a_stride_1[i] = lda[i];
            a_stride_2[i] = 1;
            size_a1[i]    = lda[i] * m[i];
        }
        if(trans_b == HIPBLAS_OP_N)
        {
            b_stride_1[i] = 1;
            b_stride_2[i] = ldb[i];
            size_b1[i]    = ldb[i] * sum_of_n;
        }
        else
        {
            b_stride_1[i] = ldb[i];
            b_stride_2[i] = 1;
            size_b1[i]    = ldb[i] * k[i];
        }

        size_a[i]    = size_a1[i] + stride_a[i] * (batch_count[i] - 1);
        size_b[i]    = size_b1[i] + stride_b[i] * (batch_count[i] - 1);
        size_c[i]    = size_c1[i] + stride_c[i] * (batch_count[i] - 1);
        size_d[i]    = size_d1[i] + stride_d[i] * (batch_count[i] - 1);
        size_bias[i] = enable_bias[i] ? m[i] : 0;

        // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
        ha[i].resize(size_a[i]);
        hb[i].resize(size_b[i]);
        hc[i].resize(size_c[i]);
        hd[i].resize(size_d[i]);
        hd_gold[i].resize(size_d[i]);
        h_bias[i].resize(size_bias[i]);

        // initial data on host
        hipblaslt::host_validation::initializeGroupedGemm(ha[i],
                                                          size_a[i],
                                                          hb[i],
                                                          size_b[i],
                                                          hc[i],
                                                          size_c[i],
                                                          h_bias[i],
                                                          size_bias[i],
                                                          initialization);

        CHECK_HIP_ERROR(hipMalloc(&da[i], size_a[i] * sizeof(Tin)));
        CHECK_HIP_ERROR(hipMalloc(&db[i], size_b[i] * sizeof(Tin)));
        CHECK_HIP_ERROR(hipMalloc(&dc[i], size_c[i] * sizeof(Tout)));
        CHECK_HIP_ERROR(hipMalloc(&dd[i], size_d[i] * sizeof(Tout)));
        if(enable_bias[i])
            CHECK_HIP_ERROR(hipMalloc(&d_bias[i], size_bias[i] * sizeof(float)));

        // copy matrices from host to device
        CHECK_HIP_ERROR(
            hipMemcpy(da[i], ha[i].data(), sizeof(Tin) * size_a[i], hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(db[i], hb[i].data(), sizeof(Tin) * size_b[i], hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dc[i], hc[i].data(), sizeof(Tout) * size_c[i], hipMemcpyHostToDevice));
        if(enable_bias[i])
            CHECK_HIP_ERROR(hipMemcpy(
                d_bias[i], h_bias[i].data(), sizeof(float) * size_bias[i], hipMemcpyHostToDevice));
    }

    // Set User Preference attributes
    uint64_t max_workspace_size = 128 * 1024 * 1024;
    void*    d_workspace;
    CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Get Heuristic results
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;

    // Get all algorithms
    hipblaslt_ext::GemmType gemmType = hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                     gemmType,
                                                     trans_a,
                                                     trans_b,
                                                     in_datatype,
                                                     in_datatype,
                                                     out_datatype,
                                                     out_datatype,
                                                     HIPBLAS_COMPUTE_32F,
                                                     heuristicResult));

    std::vector<int> validIdx;
    int              returnedAlgoCount = heuristicResult.size();

    hipblaslt_ext::GroupedGemm groupedGemm(handle,
                                           trans_a,
                                           trans_b,
                                           in_datatype,
                                           in_datatype,
                                           out_datatype,
                                           out_datatype,
                                           HIPBLAS_COMPUTE_32F);

    std::cout << "index, transAB, M, N, K, lda, ldb, ldc, stride_a, stride_b, "
                 "stride_c, batch_count, alpha, beta, bias, activationType"
              << std::endl;

    for(int i = 0; i < gemm_count; i++)
    {
        std::cout << i << ", " << hipblas_operation_to_string(trans_a)
                  << hipblas_operation_to_string(trans_b) << ", " << m[i] << ", " << n[i] << ", "
                  << k[i] << ", " << lda[i] << ", " << ldb[i] << ", " << ldc[i] << ", "
                  << stride_a[i] << ", " << stride_b[i] << ", " << stride_c[i] << ", "
                  << batch_count[i] << ", " << alpha[i] << ", " << beta[i] << ", " << enable_bias[i]
                  << ", " << ToString(actType[i]) << std::endl;
    }

    std::vector<hipblaslt_ext::GemmEpilogue> gemmEpilogue(gemm_count);
    std::vector<hipblaslt_ext::GemmInputs>   gemmInputs(gemm_count);
    for(size_t i = 0; i < gemm_count; i++)
    {
        if(enable_bias[i] && actType[i] == ActivationType::NONE)
            epilogue[i] = HIPBLASLT_EPILOGUE_BIAS;
        else if(enable_bias[i] && actType[i] == ActivationType::RELU)
            epilogue[i] = HIPBLASLT_EPILOGUE_RELU_BIAS;
        else if(enable_bias[i] && actType[i] == ActivationType::GELU)
            epilogue[i] = HIPBLASLT_EPILOGUE_GELU_BIAS;
        else if(enable_bias[i] && actType[i] == ActivationType::SWISH)
            epilogue[i] = HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT;
        else if(enable_bias[i] && actType[i] == ActivationType::CLAMP)
            epilogue[i] = HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT;
        else if(!enable_bias[i] && actType[i] == ActivationType::NONE)
            epilogue[i] = HIPBLASLT_EPILOGUE_DEFAULT;
        else if(!enable_bias[i] && actType[i] == ActivationType::RELU)
            epilogue[i] = HIPBLASLT_EPILOGUE_RELU;
        else if(!enable_bias[i] && actType[i] == ActivationType::GELU)
            epilogue[i] = HIPBLASLT_EPILOGUE_GELU;
        else if(!enable_bias[i] && actType[i] == ActivationType::SWISH)
            epilogue[i] = HIPBLASLT_EPILOGUE_SWISH_EXT;
        else if(!enable_bias[i] && actType[i] == ActivationType::CLAMP)
            epilogue[i] = HIPBLASLT_EPILOGUE_CLAMP_EXT;
        gemmEpilogue[i].setMode(epilogue[i]);
        gemmEpilogue[i].setBiasDataType(static_cast<hipDataType>(HIP_R_32F));
        if(actType[i] == ActivationType::CLAMP)
        {
            gemmEpilogue[i].setAct0(-1.f);
            gemmEpilogue[i].setAct1(1.f);
        }
        gemmInputs[i].setA(da[i]);
        gemmInputs[i].setB(db[i]);
        gemmInputs[i].setC(dc[i]);
        gemmInputs[i].setD(dd[i]);
        gemmInputs[i].setAlpha(static_cast<void*>(&alpha[i]));
        gemmInputs[i].setBeta(static_cast<void*>(&beta[i]));
        gemmInputs[i].setBias(d_bias[i]);
    }

    auto gemmProblemType = hipblaslt_ext::GemmProblemType(trans_a,
                                                          trans_b,
                                                          in_datatype,
                                                          in_datatype,
                                                          out_datatype,
                                                          out_datatype,
                                                          HIPBLAS_COMPUTE_32F);

    // step 1: set problem to {Ms, {sum of N, 1, 1, 1, ...}, Ks}
    CHECK_HIPBLASLT_ERROR(groupedGemm.setProblem(m,
                                                 sum_of_n_vec,
                                                 k,
                                                 batch_count,
                                                 lda,
                                                 ldb,
                                                 ldc,
                                                 ldd,
                                                 stride_a,
                                                 stride_b,
                                                 stride_c,
                                                 stride_d,
                                                 gemmEpilogue,
                                                 gemmInputs,
                                                 gemmProblemType));

    for(int i = 0; i < returnedAlgoCount; i++)
    {
        size_t workspace_size = 0;
        if(groupedGemm.isAlgoSupported(heuristicResult[i].algo, workspace_size)
           == HIPBLAS_STATUS_SUCCESS)
        {
            if(workspace_size <= max_workspace_size)
                validIdx.push_back(i);
        }
    }

    std::cout << "Is supported " << validIdx.size() << " / Total solutions: " << returnedAlgoCount
              << std::endl;
    if(validIdx.empty())
    {
        std::cerr << "No Solution found!" << std::endl;
        CHECK_HIP_ERROR(hipFree(d_n_vec));
        CHECK_HIP_ERROR(hipFree(d_workspace));
        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));

        for(int i = 0; i < gemm_count; i++)
        {
            CHECK_HIP_ERROR(hipFree(da[i]));
            CHECK_HIP_ERROR(hipFree(db[i]));
            CHECK_HIP_ERROR(hipFree(dc[i]));
            CHECK_HIP_ERROR(hipFree(dd[i]));
            if(enable_bias[i])
                CHECK_HIP_ERROR(hipFree(d_bias[i]));
        }
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
        return EXIT_SUCCESS;
    }

    // step2: get default DeviceUserArguments from grouped gemm objects
    hipblaslt_ext::UserArguments* dUAFloat = new hipblaslt_ext::UserArguments[gemm_count];
    groupedGemm.getDefaultValueForDeviceUserArguments((void*)dUAFloat);
    // Once you get the default value here, you can make several copies and change the values
    // from the host

    // Next Copy them to the device memory
    hipblaslt_ext::UserArguments* d_dUAFloat = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_dUAFloat, sizeof(hipblaslt_ext::UserArguments) * gemm_count));
    CHECK_HIP_ERROR(hipMemcpy(d_dUAFloat,
                              dUAFloat,
                              sizeof(hipblaslt_ext::UserArguments) * gemm_count,
                              hipMemcpyHostToDevice));

    float bestMs = std::numeric_limits<float>::max();
    for(int sol = 0; sol < validIdx.size(); sol++)
    {
        // step3: Initialize
        groupedGemm.setMaxWorkspaceBytes(max_workspace_size);
        CHECK_HIPBLASLT_ERROR(
            groupedGemm.initialize(heuristicResult[validIdx[sol]].algo, d_workspace));

        // step 4: launch kernel to modify Ns
        int threads = 256;
        int blocks  = ceil((double)gemm_count / threads);
        hipLaunchKernelGGL(
            kernelUpdateN, dim3(blocks), dim3(threads), 0, stream, gemm_count, d_dUAFloat, d_n_vec);

        float      eventMs;
        hipEvent_t start, stop;
        static_cast<void>(hipEventCreate(&start));
        static_cast<void>(hipEventCreate(&stop));
        static_cast<void>(hipEventRecord(start, stream));

        for(int sync = 0; sync < sync_count; sync++)
        {
            for(int bench = 0; bench < bench_count; bench++)
            {

                // step 5: launch grouped gemm kernel
                CHECK_HIPBLASLT_ERROR(groupedGemm.run(d_dUAFloat, stream));
            }
            static_cast<void>(hipDeviceSynchronize());
        }

        static_cast<void>(hipEventRecord(stop, stream));
        static_cast<void>(hipEventSynchronize(stop));
        static_cast<void>(hipEventElapsedTime(&eventMs, start, stop));
        static_cast<void>(hipEventDestroy(start));
        static_cast<void>(hipEventDestroy(stop));

        eventMs /= (bench_count * sync_count);
        bestMs       = std::min(bestMs, eventMs);
        double flops = 0;
        for(int i = 0; i < gemm_count; i++)
            flops += 2 * m[i] * n[i] * k[i] * batch_count[i];
        double tflops = flops / eventMs / 1000000000;

        std::cout << "      Sol " << sol << ": Perf: " << std::to_string(eventMs) << " ms, "
                  << std::to_string(tflops) << " Tflops";

        if(bestMs == eventMs)
            std::cout << " *" << std::endl;
        else
            std::cout << std::endl;

        // calculate golden or correct result
        if(validate)
        {
            std::cout << "Start to validate: " << std::endl;
            for(int i = 0; i < gemm_count; i++)
            {
                std::cout << "GEMM " << i;
                // copy output from device to CPU
                CHECK_HIP_ERROR(hipMemcpy(
                    hd[i].data(), dd[i], sizeof(Tout) * size_c[i], hipMemcpyDeviceToHost));
                auto*  a_ptr = &ha[i][0];
                auto*  b_ptr = &hb[i][0];
                auto*  c_ptr = &hc[i][0];
                auto*  d_ptr = &hd_gold[i][0];
                float* bias_ptr;
                if(enable_bias[i])
                    bias_ptr = &h_bias[i][0];
                else
                    bias_ptr = nullptr;

                bool passed = true;
                for(int i3 = 0; i3 < batch_count[i]; i3++)
                {
                    using namespace roc::host_validation;
                    using namespace hipblaslt::host_validation;
                    auto storageElements = [](size_t    rows,
                                              size_t    columns,
                                              ptrdiff_t rowStride,
                                              ptrdiff_t columnStride) {
                        if(rows == 0 || columns == 0)
                            return size_t(0);
                        return size_t(1) + (rows - 1) * size_t(std::abs(rowStride))
                               + (columns - 1) * size_t(std::abs(columnStride));
                    };

                    const size_t aElements
                        = storageElements(size_t(m[i]), size_t(k[i]), a_stride_1[i], a_stride_2[i]);
                    const size_t bElements
                        = storageElements(size_t(k[i]), size_t(n[i]), b_stride_1[i], b_stride_2[i]);
                    const size_t cElements = storageElements(size_t(m[i]), size_t(n[i]), 1, ldc[i]);
                    const size_t dElements = storageElements(size_t(m[i]), size_t(n[i]), 1, ldd[i]);

                    auto referenceOutput = tensorFromMutableStorage(
                        d_ptr + i3 * stride_d[i],
                        dElements,
                        Layout(Shape{size_t(m[i]), size_t(n[i])}, {1, ldd[i]}));
                    GemmRequest problem(
                        GemmOperand(tensorFromStorage(a_ptr + i3 * stride_a[i],
                                                      aElements,
                                                      Layout(Shape{size_t(m[i]), size_t(k[i])},
                                                             {a_stride_1[i], a_stride_2[i]}))),
                        GemmOperand(tensorFromStorage(b_ptr + i3 * stride_b[i],
                                                      bElements,
                                                      Layout(Shape{size_t(k[i]), size_t(n[i])},
                                                             {b_stride_1[i], b_stride_2[i]}))),
                        tensorFromStorage(c_ptr + i3 * stride_c[i],
                                          cElements,
                                          Layout(Shape{size_t(m[i]), size_t(n[i])}, {1, ldc[i]})),
                        referenceOutput,
                        ScalarType::Float32);
                    problem.epilogue.alpha      = static_cast<double>(alpha[i]);
                    problem.epilogue.beta       = static_cast<double>(beta[i]);
                    problem.epilogue.activation = toHostValidationActivation(actType[i]);
                    if(bias_ptr)
                        problem.epilogue.bias
                            = VectorBinding{Tensor::copyNativeStorage<float>(
                                                Layout::contiguousLastDimensionFastest(Shape{size_t(m[i])}),
                                                std::span<const float>(bias_ptr, size_t(m[i]))),
                                            MatrixAxis::Row};
                    if(actType[i] == ActivationType::SWISH)
                        problem.epilogue.activationParameter0 = 1.0;
                    else if(actType[i] == ActivationType::CLAMP)
                    {
                        problem.epilogue.activationParameter0 = -1.0;
                        problem.epilogue.activationParameter1 = 1.0;
                    }
                    referenceGemm(problem);
                    copyTensorStorageTo(d_ptr + i3 * stride_d[i], dElements, referenceOutput);

                    ComparisonOptions comparisonOptions{
                        .symmetricRelativeTolerance = std::nextafter(0.001, 0.0),
                        .maxReportedMismatches      = 10};
                    comparisonOptions.selection.indexOrder
                        = IndexOrder::FirstDimensionFastest;
                    const Layout comparisonLayout(
                        Shape{size_t(m[i]), size_t(n[i])}, {1, ldd[i]});
                    const auto comparison = roc::host_validation::compare(
                        hipblaslt::host_validation::tensorFromStorage(
                            hd[i].data() + i3 * stride_d[i], dElements, comparisonLayout),
                        hipblaslt::host_validation::tensorFromStorage(
                            d_ptr + i3 * stride_d[i], dElements, comparisonLayout),
                        comparisonOptions);
                    passed = passed && comparison.passed();
                    for(const auto& mismatch : comparison.reportedMismatches)
                    {
                        printf("Err: Index %ld: %f vs %f\n",
                               static_cast<long>(mismatch.index + i3 * stride_d[i]),
                               mismatch.expected,
                               mismatch.observed);
                    }
                }
                if(!passed)
                {
                    std::cout << " FAIL" << std::endl;
                    status = EXIT_FAILURE;
                }
                else
                {
                    std::cout << " PASS" << std::endl;
                }
            }
        }
    }

    delete[] dUAFloat;
    CHECK_HIP_ERROR(hipFree(d_dUAFloat));
    CHECK_HIP_ERROR(hipFree(d_n_vec));

    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipFree(d_workspace));

    for(int i = 0; i < gemm_count; i++)
    {
        CHECK_HIP_ERROR(hipFree(da[i]));
        CHECK_HIP_ERROR(hipFree(db[i]));
        CHECK_HIP_ERROR(hipFree(dc[i]));
        CHECK_HIP_ERROR(hipFree(dd[i]));
        if(enable_bias[i])
            CHECK_HIP_ERROR(hipFree(d_bias[i]));
    }
    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    return status;
}

int main(int argc, char* argv[])
{
    // initialize parameters with default values
    hipblasOperation_t trans_a      = HIPBLAS_OP_N;
    hipblasOperation_t trans_b      = HIPBLAS_OP_N;
    hipDataType        in_datatype  = HIP_R_32F;
    hipDataType        out_datatype = HIP_R_32F;

    std::vector<int64_t> m, lda, stride_a;
    std::vector<int64_t> n, ldb, stride_b;
    std::vector<int64_t> k, ldc, stride_c;
    std::vector<int64_t> ldd, stride_d;
    std::vector<int64_t> batch_count;

    std::vector<float>          alpha;
    std::vector<float>          beta;
    std::vector<bool>           enable_bias;
    std::vector<ActivationType> actType;

    bool    verbose     = false;
    bool    validate    = false;
    int32_t bench_count = 1;
    int32_t sync_count  = 1;

    hipblaslt_initialization initialization = hipblaslt_initialization::hpl;

    if(parse_arguments(argc,
                       argv,
                       in_datatype,
                       out_datatype,
                       m,
                       n,
                       k,
                       lda,
                       ldb,
                       ldc,
                       ldd,
                       stride_a,
                       stride_b,
                       stride_c,
                       stride_d,
                       batch_count,
                       alpha,
                       beta,
                       trans_a,
                       trans_b,
                       enable_bias,
                       actType,
                       bench_count,
                       sync_count,
                       initialization,
                       verbose,
                       validate))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    // mk is fixed. Broadcast it.
    auto m_0 = m[0];
    auto k_0 = k[0];
    m.resize(n.size(), m_0);
    k.resize(n.size(), k_0);

    int32_t         gemm_count = m.size();
    int             deviceId;
    hipDeviceProp_t deviceProperties;
    static_cast<void>(hipGetDevice(&deviceId));
    static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
    if(gpu_arch_match(deviceProperties.gcnArchName, "11\\d{2}"))
        return EXIT_SUCCESS;

    // when arguments not specified, set to default values
    for(int i = 0; i < gemm_count; i++)
    {
        if(i == n.size())
            n.push_back(DIM2);
        if(i == k.size())
            k.push_back(DIM3);
        if(i == lda.size())
            lda.push_back(trans_a == HIPBLAS_OP_N ? m[i] : k[i]);
        if(i == ldb.size())
            ldb.push_back(trans_b == HIPBLAS_OP_N ? k[i] : n[i]);
        if(i == ldc.size())
            ldc.push_back(m[i]);
        if(i == ldd.size())
            ldd.push_back(m[i]);
        if(i == stride_a.size())
            stride_a.push_back(trans_a == HIPBLAS_OP_N ? lda[i] * k[i] : lda[i] * m[i]);
        if(i == stride_b.size())
            stride_b.push_back(trans_b == HIPBLAS_OP_N ? ldb[i] * n[i] : ldb[i] * k[i]);
        if(i == stride_c.size())
            stride_c.push_back(ldc[i] * n[i]);
        if(i == stride_d.size())
            stride_d.push_back(ldd[i] * n[i]);
        if(i == batch_count.size())
            batch_count.push_back(BATCH_COUNT);
        if(i == alpha.size())
            alpha.push_back(ALPHA);
        if(i == beta.size())
            beta.push_back(BETA);
        if(i == enable_bias.size())
            enable_bias.push_back(0);
        if(i == actType.size())
            actType.push_back(ActivationType::NONE);

        if(bad_argument(trans_a,
                        trans_b,
                        m[i],
                        n[i],
                        k[i],
                        lda[i],
                        ldb[i],
                        ldc[i],
                        ldd[i],
                        stride_a[i],
                        stride_b[i],
                        stride_c[i],
                        stride_d[i],
                        batch_count[i]))
        {
            std::cerr << "GEMM idx: " << i << std::endl;
            show_usage(argv);
            return EXIT_FAILURE;
        }
    }

    int status = EXIT_SUCCESS;
    if(in_datatype == HIP_R_32F && out_datatype == HIP_R_32F)
        status = test_hipblaslt<hipblasLtFloat, hipblasLtFloat>(in_datatype,
                                                                out_datatype,
                                                                trans_a,
                                                                trans_b,
                                                                m,
                                                                n,
                                                                k,
                                                                lda,
                                                                ldb,
                                                                ldc,
                                                                ldd,
                                                                stride_a,
                                                                stride_b,
                                                                stride_c,
                                                                stride_d,
                                                                batch_count,
                                                                alpha,
                                                                beta,
                                                                enable_bias,
                                                                actType,
                                                                gemm_count,
                                                                bench_count,
                                                                sync_count,
                                                                initialization,
                                                                validate,
                                                                verbose);
    else if(in_datatype == HIP_R_16F && out_datatype == HIP_R_32F)
        status = test_hipblaslt<hipblasLtHalf, hipblasLtFloat>(in_datatype,
                                                               out_datatype,
                                                               trans_a,
                                                               trans_b,
                                                               m,
                                                               n,
                                                               k,
                                                               lda,
                                                               ldb,
                                                               ldc,
                                                               ldd,
                                                               stride_a,
                                                               stride_b,
                                                               stride_c,
                                                               stride_d,
                                                               batch_count,
                                                               alpha,
                                                               beta,
                                                               enable_bias,
                                                               actType,
                                                               gemm_count,
                                                               bench_count,
                                                               sync_count,
                                                               initialization,
                                                               validate,
                                                               verbose);
    else if(in_datatype == HIP_R_16F && out_datatype == HIP_R_16F)
        status = test_hipblaslt<hipblasLtHalf, hipblasLtHalf>(in_datatype,
                                                              out_datatype,
                                                              trans_a,
                                                              trans_b,
                                                              m,
                                                              n,
                                                              k,
                                                              lda,
                                                              ldb,
                                                              ldc,
                                                              ldd,
                                                              stride_a,
                                                              stride_b,
                                                              stride_c,
                                                              stride_d,
                                                              batch_count,
                                                              alpha,
                                                              beta,
                                                              enable_bias,
                                                              actType,
                                                              gemm_count,
                                                              bench_count,
                                                              sync_count,
                                                              initialization,
                                                              validate,
                                                              verbose);
    else if(in_datatype == HIP_R_16BF && out_datatype == HIP_R_16BF)
        status = test_hipblaslt<hipblasLtBfloat16, hipblasLtBfloat16>(in_datatype,
                                                                      out_datatype,
                                                                      trans_a,
                                                                      trans_b,
                                                                      m,
                                                                      n,
                                                                      k,
                                                                      lda,
                                                                      ldb,
                                                                      ldc,
                                                                      ldd,
                                                                      stride_a,
                                                                      stride_b,
                                                                      stride_c,
                                                                      stride_d,
                                                                      batch_count,
                                                                      alpha,
                                                                      beta,
                                                                      enable_bias,
                                                                      actType,
                                                                      gemm_count,
                                                                      bench_count,
                                                                      sync_count,
                                                                      initialization,
                                                                      validate,
                                                                      verbose);

    return status;
}
