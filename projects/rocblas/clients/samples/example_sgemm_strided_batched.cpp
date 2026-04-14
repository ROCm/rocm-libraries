/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>
#include <rocblas/rocblas.h>
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

#ifndef CHECK_ROCBLAS_ERROR
#define CHECK_ROCBLAS_ERROR(error)                              \
    if(error != rocblas_status_success)                         \
    {                                                           \
        fprintf(stderr, "rocBLAS error: ");                     \
        if(error == rocblas_status_invalid_handle)              \
            fprintf(stderr, "rocblas_status_invalid_handle");   \
        if(error == rocblas_status_not_implemented)             \
            fprintf(stderr, " rocblas_status_not_implemented"); \
        if(error == rocblas_status_invalid_pointer)             \
            fprintf(stderr, "rocblas_status_invalid_pointer");  \
        if(error == rocblas_status_invalid_size)                \
            fprintf(stderr, "rocblas_status_invalid_size");     \
        if(error == rocblas_status_memory_error)                \
            fprintf(stderr, "rocblas_status_memory_error");     \
        if(error == rocblas_status_internal_error)              \
            fprintf(stderr, "rocblas_status_internal_error");   \
        fprintf(stderr, "\n");                                  \
        exit(EXIT_FAILURE);                                     \
    }
#endif

// default sizes
#define DIM1 127
#define DIM2 128
#define DIM3 129
#define BATCH_COUNT 10
#define ALPHA 2
#define BETA 3

void printMatrix(const char* name, float* A, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    printf("---------- %s ----------\n", name);
    int max_size = 3;
    for(int i = 0; i < m && i < max_size; i++)
    {
        for(int j = 0; j < n && j < max_size; j++)
        {
            printf("%f ", A[i + j * lda]);
        }
        printf("\n");
    }
}

void print_strided_batched(const char* name,
                           float*      A,
                           rocblas_int n1,
                           rocblas_int n2,
                           rocblas_int n3,
                           rocblas_int s1,
                           rocblas_int s2,
                           rocblas_int s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    printf("---------- %s ----------\n", name);
    int max_size = 3;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                printf("%8.1f ", A[(i1 * s1) + (i2 * s2) + (i3 * s3)]);
            }
            printf("\n");
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            printf("\n");
    }
}

template <typename T>
void mat_mat_mult(T        alpha,
                  T        beta,
                  int      M,
                  int      N,
                  int      K,
                  const T* A,
                  int      As1,
                  int      As2,
                  const T* B,
                  int      Bs1,
                  int      Bs2,
                  T*       C,
                  int      Cs1,
                  int      Cs2)
{
    for(int i1 = 0; i1 < M; i1++)
    {
        for(int i2 = 0; i2 < N; i2++)
        {
            T t = 0.0;
            for(int i3 = 0; i3 < K; i3++)
            {
                t += A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2];
            }
            C[i1 * Cs1 + i2 * Cs2] = beta * C[i1 * Cs1 + i2 * Cs2] + alpha * t;
        }
    }
}

// cppcheck-suppress constParameter
static void show_usage(char* argv[])
{
    std::cerr << "Usage: " << argv[0] << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\t\tShow this help message\n"
              << "\t-v, --verbose\t\t\t\tverbose output\n"
              << "\t-m \t\t\tm\t\tGEMM_STRIDED_BATCHED argument m\n"
              << "\t-n \t\t\tn\t\tGEMM_STRIDED_BATCHED argument n\n"
              << "\t-k \t\t\tk \t\tGEMM_STRIDED_BATCHED argument k\n"
              << "\t--lda \t\t\tlda \t\tGEMM_STRIDED_BATCHED argument lda\n"
              << "\t--ldb \t\t\tldb \t\tGEMM_STRIDED_BATCHED argument ldb\n"
              << "\t--ldc \t\t\tldc \t\tGEMM_STRIDED_BATCHED argument ldc\n"
              << "\t--trans_a \t\ttrans_a \tGEMM_STRIDED_BATCHED argument trans_a\n"
              << "\t--trans_b \t\ttrans_b \tGEMM_STRIDED_BATCHED argument trans_b\n"
              << "\t--stride_a \t\tstride_a \tGEMM_STRIDED_BATCHED argument stride_a\n"
              << "\t--stride_b \t\tstride_b \tGEMM_STRIDED_BATCHED argument stride_b\n"
              << "\t--stride_c \t\tstride_c \tGEMM_STRIDED_BATCHED argument stride_c\n"
              << "\t--batch_count \t\tbatch_count \tGEMM_STRIDED_BATCHED argument batch count\n"
              << "\t--alpha \t\talpha \t\tGEMM_STRIDED_BATCHED argument alpha\n"
              << "\t--beta \t\t\tbeta \t\tGEMM_STRIDED_BATCHED argument beta\n"
              << "\t--header \t\theader \t\tprint header for output\n"
              << std::endl;
}

static int parse_arguments(int                argc,
                           char*              argv[],
                           int&               m,
                           int&               n,
                           int&               k,
                           int&               lda,
                           int&               ldb,
                           int&               ldc,
                           int&               stride_a,
                           int&               stride_b,
                           int&               stride_c,
                           int&               batch_count,
                           float&             alpha,
                           float&             beta,
                           rocblas_operation& trans_a,
                           rocblas_operation& trans_b,
                           bool&              header,
                           bool&              verbose)
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
                else if((arg == "-k") && (i + 1 < argc))
                {
                    k = atoi(argv[++i]);
                }
                else if((arg == "--batch_count") && (i + 1 < argc))
                {
                    batch_count = atoi(argv[++i]);
                }
                else if((arg == "--lda") && (i + 1 < argc))
                {
                    lda = atoi(argv[++i]);
                }
                else if((arg == "--ldb") && (i + 1 < argc))
                {
                    ldb = atoi(argv[++i]);
                }
                else if((arg == "--ldc") && (i + 1 < argc))
                {
                    ldc = atoi(argv[++i]);
                }
                else if((arg == "--stride_a") && (i + 1 < argc))
                {
                    stride_a = atoi(argv[++i]);
                }
                else if((arg == "--stride_b") && (i + 1 < argc))
                {
                    stride_b = atoi(argv[++i]);
                }
                else if((arg == "--stride_c") && (i + 1 < argc))
                {
                    stride_c = atoi(argv[++i]);
                }
                else if((arg == "--alpha") && (i + 1 < argc))
                {
                    alpha = atof(argv[++i]);
                }
                else if((arg == "--beta") && (i + 1 < argc))
                {
                    beta = atof(argv[++i]);
                }
                else if((arg == "--trans_a") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans_a = rocblas_operation_none;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_a = rocblas_operation_transpose;
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
                        trans_b = rocblas_operation_none;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_b = rocblas_operation_transpose;
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

bool bad_argument(rocblas_operation trans_a,
                  rocblas_operation trans_b,
                  rocblas_int       m,
                  rocblas_int       n,
                  rocblas_int       k,
                  rocblas_int       lda,
                  rocblas_int       ldb,
                  rocblas_int       ldc,
                  rocblas_int       stride_a,
                  rocblas_int       stride_b,
                  rocblas_int       stride_c,
                  rocblas_int       batch_count)
{
    bool argument_error = false;
    if((trans_a == rocblas_operation_none) && (lda < m))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << lda << " < " << m << std::endl;
    }
    if((trans_a == rocblas_operation_transpose) && (lda < k))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << lda << " < " << k << std::endl;
    }
    if((trans_b == rocblas_operation_none) && (ldb < k))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldb = " << ldb << " < " << k << std::endl;
    }
    if((trans_b == rocblas_operation_transpose) && (ldb < n))
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
    if(batch_count < 1)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument batch_count = " << batch_count << " < 1" << std::endl;
    }

    return argument_error;
}

void initialize_a_b_c(std::vector<float>& ha,
                      rocblas_int         size_a,
                      std::vector<float>& hb,
                      rocblas_int         size_b,
                      std::vector<float>& hc,
                      std::vector<float>& hc_gold,
                      rocblas_int         size_c)
{
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = rand() % 17;
        //      ha[i] = i;
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = rand() % 17;
        //      hb[i] = 1.0;
    }
    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = rand() % 17;
        //      hc[i] = 1.0;
    }
    hc_gold = hc;
}

int main(int argc, char* argv[])
{
    // initialize parameters with default values
    rocblas_operation trans_a = rocblas_operation_none;
    rocblas_operation trans_b = rocblas_operation_transpose;

    // invalid int and float for rocblas_sgemm_strided_batched int and float arguments
    rocblas_int invalid_int   = std::numeric_limits<rocblas_int>::min() + 1;
    float       invalid_float = std::numeric_limits<float>::quiet_NaN();

    // initialize to invalid value to detect if values not specified on command line
    rocblas_int m = invalid_int, lda = invalid_int, stride_a = invalid_int;
    rocblas_int n = invalid_int, ldb = invalid_int, stride_b = invalid_int;
    rocblas_int k = invalid_int, ldc = invalid_int, stride_c = invalid_int;

    rocblas_int batch_count = invalid_int;

    float alpha = invalid_float;
    float beta  = invalid_float;

    bool verbose = false;
    bool header  = false;

    if(parse_arguments(argc,
                       argv,
                       m,
                       n,
                       k,
                       lda,
                       ldb,
                       ldc,
                       stride_a,
                       stride_b,
                       stride_c,
                       batch_count,
                       alpha,
                       beta,
                       trans_a,
                       trans_b,
                       header,
                       verbose))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    // when arguments not specified, set to default values
    if(m == invalid_int)
        m = DIM1;
    if(n == invalid_int)
        n = DIM2;
    if(k == invalid_int)
        k = DIM3;
    if(lda == invalid_int)
        lda = trans_a == rocblas_operation_none ? m : k;
    if(ldb == invalid_int)
        ldb = trans_b == rocblas_operation_none ? k : n;
    if(ldc == invalid_int)
        ldc = m;
    if(stride_a == invalid_int)
        stride_a = trans_a == rocblas_operation_none ? lda * k : lda * m;
    if(stride_b == invalid_int)
        stride_b = trans_b == rocblas_operation_none ? ldb * n : ldb * k;
    if(stride_c == invalid_int)
        stride_c = ldc * n;
    if(alpha != alpha)
        alpha = ALPHA; // check for alpha == invalid_float == NaN
    if(beta != beta)
        beta = BETA; // check for beta == invalid_float == NaN
    if(batch_count == invalid_int)
        batch_count = BATCH_COUNT;

    if(bad_argument(
           trans_a, trans_b, m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_count))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    if(header)
    {
        std::cout << "transAB,M,N,K,lda,ldb,ldc,stride_a,stride_b,stride_c,batch_count,alpha,beta,"
                     "result,error";
        std::cout << std::endl;
    }

    int a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    int size_a1, size_b1, size_c1 = ldc * n;
    if(trans_a == rocblas_operation_none)
    {
        std::cout << "N";
        a_stride_1 = 1;
        a_stride_2 = lda;
        size_a1    = lda * k;
    }
    else
    {
        std::cout << "T";
        a_stride_1 = lda;
        a_stride_2 = 1;
        size_a1    = lda * m;
    }
    if(trans_b == rocblas_operation_none)
    {
        std::cout << "N, ";
        b_stride_1 = 1;
        b_stride_2 = ldb;
        size_b1    = ldb * n;
    }
    else
    {
        std::cout << "T, ";
        b_stride_1 = ldb;
        b_stride_2 = 1;
        size_b1    = ldb * k;
    }

    std::cout << m << ", " << n << ", " << k << ", " << lda << ", " << ldb << ", " << ldc << ", "
              << stride_a << ", " << stride_b << ", " << stride_c << ", " << batch_count << ", "
              << alpha << ", " << beta << ", ";

    int size_a = batch_count == 0 ? size_a1 : size_a1 + stride_a * (batch_count - 1);
    int size_b = batch_count == 0 ? size_b1 : size_b1 + stride_b * (batch_count - 1);
    int size_c = batch_count == 0 ? size_c1 : size_c1 + stride_c * (batch_count - 1);

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<float> ha(size_a);
    std::vector<float> hb(size_b);
    std::vector<float> hc(size_c);
    std::vector<float> hc_gold(size_c);

    // initial data on host
    initialize_a_b_c(ha, size_a, hb, size_b, hc, hc_gold, size_c);

    if(verbose)
    {
        printf("\n");
        if(trans_a == rocblas_operation_none)
        {
            print_strided_batched("ha initial", &ha[0], m, k, batch_count, 1, lda, stride_a);
        }
        else
        {
            print_strided_batched("ha initial", &ha[0], m, k, batch_count, lda, 1, stride_a);
        }
        if(trans_b == rocblas_operation_none)
        {
            print_strided_batched("hb initial", &hb[0], k, n, batch_count, 1, ldb, stride_b);
        }
        else
        {
            print_strided_batched("hb initial", &hb[0], k, n, batch_count, ldb, 1, stride_b);
        }
        print_strided_batched("hc initial", &hc[0], m, n, batch_count, 1, ldc, stride_c);
    }

    // allocate memory on device
    float *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    CHECK_ROCBLAS_ERROR(rocblas_sgemm_strided_batched(handle,
                                                      trans_a,
                                                      trans_b,
                                                      m,
                                                      n,
                                                      k,
                                                      &alpha,
                                                      da,
                                                      lda,
                                                      stride_a,
                                                      db,
                                                      ldb,
                                                      stride_b,
                                                      &beta,
                                                      dc,
                                                      ldc,
                                                      stride_c,
                                                      batch_count));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(float) * size_c, hipMemcpyDeviceToHost));

    // calculate golden or correct result
    for(int i = 0; i < batch_count; i++)
    {
        float* a_ptr = &ha[i * stride_a];
        float* b_ptr = &hb[i * stride_b];
        float* c_ptr = &hc_gold[i * stride_c];
        mat_mat_mult<float>(alpha,
                            beta,
                            m,
                            n,
                            k,
                            a_ptr,
                            a_stride_1,
                            a_stride_2,
                            b_ptr,
                            b_stride_1,
                            b_stride_2,
                            c_ptr,
                            1,
                            ldc);
    }

    if(verbose)
    {
        print_strided_batched(
            "hc_gold calculated", &hc_gold[0], m, n, batch_count, 1, ldc, stride_c);
        print_strided_batched("hc calculated", &hc[0], m, n, batch_count, 1, ldc, stride_c);
    }

    float max_relative_error = std::numeric_limits<float>::min();
    for(int i = 0; i < size_c; i++)
    {
        float relative_error
            = hc_gold[i] == 0 ? hc_gold[i] - hc[i] : (hc_gold[i] - hc[i]) / hc_gold[i];
        relative_error = relative_error >= 0 ? relative_error : -relative_error;
        max_relative_error
            = relative_error < max_relative_error ? max_relative_error : relative_error;
    }
    float eps       = std::numeric_limits<float>::epsilon();
    float tolerance = 10;
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        std::cout << "FAIL, " << max_relative_error << std::endl;
    }
    else
    {
        std::cout << "PASS, " << max_relative_error << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}
