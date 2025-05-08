/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hipsparselt/hipsparselt.h>
#include <iostream>
#include <limits>
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
        throw EXIT_FAILURE;                       \
    }
#endif

#ifndef CHECK_HIPSPARSELT_ERROR
#define CHECK_HIPSPARSELT_ERROR(error)                           \
    if(error != HIPSPARSE_STATUS_SUCCESS)                        \
    {                                                            \
        fprintf(stderr, "hipSPARSELt error(Err=%d) : ", error);  \
        if(error == HIPSPARSE_STATUS_NOT_INITIALIZED)            \
            fprintf(stderr, "HIPSPARSE_STATUS_NOT_INITIALIZED"); \
        if(error == HIPSPARSE_STATUS_INTERNAL_ERROR)             \
            fprintf(stderr, " HIPSPARSE_STATUS_INTERNAL_ERROR"); \
        if(error == HIPSPARSE_STATUS_INVALID_VALUE)              \
            fprintf(stderr, "HIPSPARSE_STATUS_INVALID_VALUE");   \
        if(error == HIPSPARSE_STATUS_ALLOC_FAILED)               \
            fprintf(stderr, "HIPSPARSE_STATUS_ALLOC_FAILED");    \
        if(error == HIPSPARSE_STATUS_ARCH_MISMATCH)              \
            fprintf(stderr, "HIPSPARSE_STATUS_ARCH_MISMATCH");   \
        fprintf(stderr, "\n");                                   \
        throw EXIT_FAILURE;                                      \
    }
#endif

template <typename T, typename To>
class SMART_DESTROYER
{
    typedef To (*destroy_func)(T*);

public:
    SMART_DESTROYER(T* ptr, destroy_func func)
    {
        _ptr  = ptr;
        _func = func;
    }
    ~SMART_DESTROYER()
    {
        if(_ptr != nullptr)
            static_cast<void>(_func(_ptr));
    }

    T*           _ptr = nullptr;
    destroy_func _func;
};

template <typename T, typename To>
class SMART_DESTROYER_NON_PTR
{
    typedef To (*destroy_func)(T);

public:
    SMART_DESTROYER_NON_PTR(T* ptr, destroy_func func)
    {
        _ptr  = ptr;
        _func = func;
    }
    ~SMART_DESTROYER_NON_PTR()
    {
        if(_ptr != nullptr)
            static_cast<void>(_func(*_ptr));
    }

    T*           _ptr = nullptr;
    destroy_func _func;
};

// default sizes
#define DIM1 127
#define DIM2 128
#define DIM3 129
#define BATCH_COUNT 10

template <typename Ti, typename Tc>
inline float norm1(Ti a, Ti b)
{
    auto ac = static_cast<Tc>(a);
    auto bc = static_cast<Tc>(b);

    return static_cast<Tc>(abs(ac) + abs(bc));
}

template <typename Ti, typename Tc>
void prune_strip(const Ti* in,
                 Ti*       out,
                 int64_t   m,
                 int64_t   n,
                 int64_t   stride1,
                 int64_t   stride2,
                 int       num_batches,
                 int64_t   stride_b)
{
    for(int b = 0; b < num_batches; b++)
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j += 4)
            {
                size_t pos[4];
                for(int k = 0; k < 4; k++)
                {
                    pos[k] = b * stride_b + i * stride1 + (j + k) * stride2;
                }

                auto max_norm1 = static_cast<Tc>(-1.0f);
                int  pos_a, pos_b;
                for(int a = 0; a < 4; a++)
                {
                    for(int b = a + 1; b < 4; b++)
                    {
                        auto norm1_v = norm1<Ti, Tc>(in[pos[a]], in[pos[b]]);
                        if(norm1_v > max_norm1)
                        {
                            pos_a     = a;
                            pos_b     = b;
                            max_norm1 = norm1_v;
                        }
                    }
                }

                for(int k = 0; k < 4; k++)
                {
                    if(k == pos_a || k == pos_b)
                    {
                        if(in != out)
                            out[pos[k]] = in[pos[k]];
                    }
                    else
                        out[pos[k]] = static_cast<Ti>(0.0f);
                }
            }
        }
}

template <typename T>
void validate(T* A, T* B, int64_t n1, int64_t n2, int64_t n3, int64_t s1, int64_t s2, int64_t s3)
{
    bool correct = true;
    using c_type = std::conditional_t<std::is_same<__half, T>::value, float, T>;
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    for(int i3 = 0; i3 < n3; i3++)
    {
        for(int i1 = 0; i1 < n1; i1++)
        {
            for(int i2 = 0; i2 < n2; i2++)
            {
                auto pos     = (i1 * s1) + (i2 * s2) + (i3 * s3);
                T    value_a = A[pos];
                T    value_b = B[pos];
                if(static_cast<c_type>(value_a) != static_cast<c_type>(value_b))
                {
                    // direct floating point comparison is not reliable
                    std::printf("(%d, %d, %d):\t%f vs. %f\n",
                                i3,
                                i1,
                                i2,
                                static_cast<double>(value_a),
                                static_cast<double>(value_b));
                    correct = false;
                    //break;
                }
            }
        }
    }
    if(correct)
        std::printf("prune strip test PASSED\n");
    else
        std::printf("prune strip test FAILED: wrong result\n");
}

template <typename T>
void test_prune_check(hipsparseLtHandle_t*           handle,
                      hipsparseLtMatmulDescriptor_t* matmul,
                      T*                             d,
                      hipStream_t                    stream,
                      bool                           expected)
{
    int* d_valid;
    int  h_valid = 0;
    CHECK_HIP_ERROR(hipMalloc(&d_valid, sizeof(int)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtSpMMAPruneCheck(handle, matmul, d, d_valid, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(&h_valid, d_valid, sizeof(int), hipMemcpyDeviceToHost, stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    auto s = expected ? "passed" : "falied";
    if((h_valid == 0) == expected)
        std::printf("expected %s, prune check PASSED\n", s);
    else
        std::printf("expected %s, prune check FAILED: wrong result\n", s);
    CHECK_HIP_ERROR(hipFree(d_valid));
}

template <typename T>
void print_strided_batched(
    const char* name, T* A, int64_t n1, int64_t n2, int64_t n3, int64_t s1, int64_t s2, int64_t s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    printf("---------- %s ----------\n", name);
    int max_size = 128;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                printf("%8.1f ", static_cast<float>(A[(i1 * s1) + (i2 * s2) + (i3 * s3)]));
            }
            printf("\n");
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            printf("\n");
    }
}

// cppcheck-suppress constParameter
static void show_usage(char* argv[])
{
    std::cerr
        << "Usage: " << argv[0] << " <options>\n"
        << "options:\n"
        << "\t-h, --help\t\t\t\tShow this help message\n"
        << "\t-v, --verbose\t\t\t\tverbose output\n"
        << "\t-m \t\t\tm\t\tGEMM_STRIDED_BATCHED argument m\n"
        << "\t-n \t\t\tn\t\tGEMM_STRIDED_BATCHED argument n\n"
        << "\t--ld \t\t\tld \t\tGEMM_STRIDED_BATCHED argument lda\n"
        << "\t--trans \t\ttrans \tGEMM_STRIDED_BATCHED argument trans_a\n"
        << "\t--stride \t\tstride \tGEMM_STRIDED_BATCHED argument stride_a\n"
        << "\t--batch_count \t\tbatch_count \tGEMM_STRIDED_BATCHED argument batch count\n"
        << "\t--sparse_b \t\tsparse_b \t\tStructurted Sparsity Matrix B (A is Dense Matrix)\n"
        << "\t-r \t\tprecision \tGEMM_STRIDED_BATCHED argument precsion, etc., h=half, b=bfloat16\n"
        << "\t--header \t\theader \t\tprint header for output\n"
        << std::endl;
}

static int parse_arguments(int                   argc,
                           char*                 argv[],
                           int64_t&              m,
                           int64_t&              n,
                           int64_t&              ld,
                           int64_t&              stride,
                           int&                  batch_count,
                           hipsparseOperation_t& trans,
                           hipDataType&          type,
                           bool&                 header,
                           bool&                 sparse_b,
                           bool&                 verbose)
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
                else if(arg == "--header")
                {
                    header = true;
                }
                else if((arg == "-m") && (i + 1 < argc))
                {
                    m = atoi(argv[++i]);
                }
                else if((arg == "-n") && (i + 1 < argc))
                {
                    n = atoi(argv[++i]);
                }
                else if((arg == "--batch_count") && (i + 1 < argc))
                {
                    batch_count = atoi(argv[++i]);
                }
                else if((arg == "--ld") && (i + 1 < argc))
                {
                    ld = atoi(argv[++i]);
                }
                else if((arg == "--stride") && (i + 1 < argc))
                {
                    stride = atoi(argv[++i]);
                }
                else if((arg == "--trans") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans = HIPSPARSE_OPERATION_NON_TRANSPOSE;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans = HIPSPARSE_OPERATION_TRANSPOSE;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "-r") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "h", 1) == 0)
                    {
                        type = type;
                    }
                    else if(strncmp(argv[i], "b", 1) == 0)
                    {
                        type = HIP_R_16BF;
                    }
                    else if(strncmp(argv[i], "i8", 1) == 0)
                    {
                        type = HIP_R_8I;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if(arg == "--sparse_b")
                {
                    sparse_b = true;
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

bool bad_argument(hipsparseOperation_t trans,
                  int64_t              m,
                  int64_t              n,
                  int64_t              ld,
                  int64_t              stride,
                  int64_t              batch_count)
{
    bool argument_error = false;
    if((trans == HIPSPARSE_OPERATION_NON_TRANSPOSE) && (ld < m))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << ld << " < " << m << std::endl;
    }
    if((trans == HIPSPARSE_OPERATION_TRANSPOSE) && (ld < n))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << ld << " < " << n << std::endl;
    }
    if(stride < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride < 0" << std::endl;
    }
    if(batch_count < 1)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument batch_count = " << batch_count << " < 1" << std::endl;
    }

    return argument_error;
}

template <typename T>
void initialize_a(std::vector<T>& ha, int64_t size_a)
{
    auto CAST = [](auto x) {
#if defined(__HIP_PLATFORM_AMD__)
        return static_cast<T>(x);
#else
        return static_cast<T>(static_cast<float>(x));
#endif
    };
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = CAST(rand() % 17);
    }
}

template <typename T>
void run(int64_t              m,
         int64_t              n,
         int64_t              ld,
         int64_t              stride,
         int                  batch_count,
         hipsparseOperation_t trans,
         hipDataType          type,
         bool                 sparse_b,
         bool                 verbose)
{
    int64_t stride_1, stride_2;
    int64_t row, col;
    int     size_1;
    if(trans == HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        std::cout << ", N";
        row      = m;
        col      = n;
        stride_1 = 1;
        stride_2 = ld;
        size_1   = ld * n;
    }
    else
    {
        std::cout << ", T";
        row      = n;
        col      = m;
        stride_1 = ld;
        stride_2 = 1;
        size_1   = ld * m;
    }

    std::cout << ", " << m << ", " << n << ", " << ld << ", " << stride << ", " << batch_count
              << std::endl;

    int size = stride == 0 ? size_1 * batch_count : stride * batch_count;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<T> hp(size);
    std::vector<T> hp_test(size);
    std::vector<T> hp_gold(size);

    // initial data on host
    initialize_a(hp, size);

    if(verbose)
    {
        printf("\n");
        if(trans == HIPSPARSE_OPERATION_NON_TRANSPOSE)
        {
            print_strided_batched("host initial", &hp[0], m, n, batch_count, 1, ld, stride);
        }
        else
        {
            print_strided_batched("host initial", &hp[0], m, n, batch_count, ld, 1, stride);
        }
    }

    // allocate memory on device
    T*          d;
    T*          d_test;
    void*       d_wworkspace;
    int         num_streams = 0;
    hipStream_t stream      = nullptr;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    SMART_DESTROYER_NON_PTR<hipStream_t, hipError_t> sS(&stream, hipStreamDestroy);

    CHECK_HIP_ERROR(hipMalloc(&d, size * sizeof(T)));
    SMART_DESTROYER<void, hipError_t> sD(d, hipFree);

    CHECK_HIP_ERROR(hipMalloc(&d_test, size * sizeof(T)));
    SMART_DESTROYER<void, hipError_t> sDt(d_test, hipFree);
    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(d, hp.data(), sizeof(T) * size, hipMemcpyHostToDevice));

    hipsparseLtHandle_t           handle;
    hipsparseLtMatDescriptor_t    matA, matB, matC, matD;
    hipsparseLtMatmulDescriptor_t matmul;

    CHECK_HIPSPARSELT_ERROR(hipsparseLtInit(&handle));
    SMART_DESTROYER<const hipsparseLtHandle_t, hipsparseStatus_t> sH(&handle, hipsparseLtDestroy);

    if(!sparse_b)
    {
        CHECK_HIPSPARSELT_ERROR(
            hipsparseLtStructuredDescriptorInit(&handle,
                                                &matA,
                                                row,
                                                col,
                                                ld,
                                                16,
                                                type,
                                                HIPSPARSE_ORDER_COL,
                                                HIPSPARSELT_SPARSITY_50_PERCENT));
    }
    else
    {
        CHECK_HIPSPARSELT_ERROR(
            hipsparseLtDenseDescriptorInit(&handle, &matA, n, m, n, 16, type, HIPSPARSE_ORDER_COL));
    }
    SMART_DESTROYER<const hipsparseLtMatDescriptor_t, hipsparseStatus_t> smA(
        &matC, hipsparseLtMatDescriptorDestroy);

    if(!sparse_b)
    {
        CHECK_HIPSPARSELT_ERROR(
            hipsparseLtDenseDescriptorInit(&handle, &matB, n, m, n, 16, type, HIPSPARSE_ORDER_COL));
    }
    else
    {
        CHECK_HIPSPARSELT_ERROR(
            hipsparseLtStructuredDescriptorInit(&handle,
                                                &matB,
                                                row,
                                                col,
                                                ld,
                                                16,
                                                type,
                                                HIPSPARSE_ORDER_COL,
                                                HIPSPARSELT_SPARSITY_50_PERCENT));
    }
    SMART_DESTROYER<const hipsparseLtMatDescriptor_t, hipsparseStatus_t> smB(
        &matC, hipsparseLtMatDescriptorDestroy);

    auto tmp_m = (!sparse_b) ? m : n;

    CHECK_HIPSPARSELT_ERROR(hipsparseLtDenseDescriptorInit(
        &handle, &matC, tmp_m, tmp_m, tmp_m, 16, type, HIPSPARSE_ORDER_COL));
    SMART_DESTROYER<const hipsparseLtMatDescriptor_t, hipsparseStatus_t> smC(
        &matC, hipsparseLtMatDescriptorDestroy);

    CHECK_HIPSPARSELT_ERROR(hipsparseLtDenseDescriptorInit(
        &handle, &matD, tmp_m, tmp_m, tmp_m, 16, type, HIPSPARSE_ORDER_COL));
    SMART_DESTROYER<const hipsparseLtMatDescriptor_t, hipsparseStatus_t> smD(
        &matC, hipsparseLtMatDescriptorDestroy);

    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matA, HIPSPARSELT_MAT_NUM_BATCHES, &batch_count, sizeof(batch_count)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matB, HIPSPARSELT_MAT_NUM_BATCHES, &batch_count, sizeof(batch_count)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matC, HIPSPARSELT_MAT_NUM_BATCHES, &batch_count, sizeof(batch_count)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matD, HIPSPARSELT_MAT_NUM_BATCHES, &batch_count, sizeof(batch_count)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, sparse_b ? &matB : &matA, HIPSPARSELT_MAT_BATCH_STRIDE, &stride, sizeof(stride)));

    auto compute_type = type == HIP_R_8I ? HIPSPARSELT_COMPUTE_32I :
#ifdef __HIP_PLATFORM_AMD__
                                         HIPSPARSELT_COMPUTE_32F;
#else
                                         HIPSPARSELT_COMPUTE_16F;
#endif

    CHECK_HIPSPARSELT_ERROR(
        hipsparseLtMatmulDescriptorInit(&handle,
                                        &matmul,
                                        sparse_b ? HIPSPARSE_OPERATION_NON_TRANSPOSE : trans,
                                        sparse_b ? trans : HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                        &matA,
                                        &matB,
                                        &matC,
                                        &matD,
                                        compute_type));

    CHECK_HIPSPARSELT_ERROR(
        hipsparseLtSpMMAPrune(&handle, &matmul, d, d_test, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    CHECK_HIP_ERROR(hipMemcpy(hp_test.data(), d_test, sizeof(T) * size, hipMemcpyDeviceToHost));

    if(!sparse_b)
        prune_strip<T, float>(
            hp.data(), hp_gold.data(), m, n, stride_1, stride_2, batch_count, stride);
    else
        prune_strip<T, float>(
            hp.data(), hp_gold.data(), n, m, stride_2, stride_1, batch_count, stride);

    if(verbose)
    {
        auto stride_r = stride == 0 ? size_1 : stride;
        print_strided_batched(
            "hp_gold calculated", &hp_gold[0], m, n, batch_count, stride_1, stride_2, stride_r);
        print_strided_batched(
            "hp_test calculated", &hp_test[0], m, n, batch_count, stride_1, stride_2, stride_r);
    }

    validate<T>(&hp_gold[0], &hp_test[0], m, n, batch_count, stride_1, stride_2, stride);

    test_prune_check<T>(&handle, &matmul, d_test, stream, true);
    test_prune_check<T>(&handle, &matmul, d, stream, false);
}

int main(int argc, char* argv[])
{
    try
    {
        // initialize parameters with default values
        hipsparseOperation_t trans = HIPSPARSE_OPERATION_NON_TRANSPOSE;

        // invalid int and float for hipsparselt spmm int and float arguments
        int64_t invalid_int64 = std::numeric_limits<int64_t>::min() + 1;
        int     invalid_int   = std::numeric_limits<int>::min() + 1;
        float   invalid_float = std::numeric_limits<float>::quiet_NaN();

        // initialize to invalid value to detect if values not specified on command line
        int64_t m = invalid_int64, n = invalid_int64, ld = invalid_int64, stride = invalid_int64;

        int         batch_count = invalid_int;
        hipDataType type        = HIP_R_16F;

        bool verbose  = false;
        bool header   = false;
        bool sparse_b = false;

        if(parse_arguments(
               argc, argv, m, n, ld, stride, batch_count, trans, type, header, sparse_b, verbose))
        {
            show_usage(argv);
            return EXIT_FAILURE;
        }

        // when arguments not specified, set to default values
        if(m == invalid_int64)
            m = DIM1;
        if(n == invalid_int64)
            n = DIM2;
        if(ld == invalid_int64)
            ld = trans == HIPSPARSE_OPERATION_NON_TRANSPOSE ? m : n;
        if(stride == invalid_int64)
            stride = trans == HIPSPARSE_OPERATION_NON_TRANSPOSE ? ld * n : ld * m;
        if(batch_count == invalid_int)
            batch_count = BATCH_COUNT;

        if(bad_argument(trans, m, n, ld, stride, batch_count))
        {
            show_usage(argv);
            return EXIT_FAILURE;
        }

        if(header)
        {
            std::cout << "type,trans,M,N,ld,stride,batch_count,"
                         "result,error";
            std::cout << std::endl;
        }

        switch(type)
        {
        case HIP_R_16F:
            std::cout << "H";
            run<__half>(m, n, ld, stride, batch_count, trans, type, sparse_b, verbose);
            break;
        case HIP_R_16BF:
            std::cout << "BF16";
            run<hip_bfloat16>(m, n, ld, stride, batch_count, trans, type, sparse_b, verbose);
            break;
        case HIP_R_8I:
            std::cout << "I8";
            run<int8_t>(m, n, ld, stride, batch_count, trans, type, sparse_b, verbose);
            break;
        default:
            break;
        }

        return EXIT_SUCCESS;
    }
    catch(int i)
    {
        return EXIT_FAILURE;
    }
}
