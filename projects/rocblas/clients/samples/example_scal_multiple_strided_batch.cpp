/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "client_utility.hpp"
#include "host_alloc.hpp"
#include "rocblas_vector.hpp"

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
#define BATCH_COUNT 10
#define ALPHA 2

void print_strided_batched_vector(
    const char* name, float* x, rocblas_int n1, rocblas_int n2, rocblas_int s1, rocblas_int s2)
{
    // n1, n2 are vector dimensions, sometimes called n, batch_count
    // s1, s1 are vector strides, sometimes called incx, stride_x
    printf("---------- %s ----------\n", name);
    int max_size = 4;

    for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            printf("%8.1f ", x[(i1 * s1) + (i2 * s2)]);
        }
        printf("\n");
    }
}

void print_multiple_strided_batched_vector(const char* name,
                                           float*      x,
                                           rocblas_int n1,
                                           rocblas_int n2,
                                           rocblas_int n3,
                                           rocblas_int s1,
                                           rocblas_int s2,
                                           rocblas_int s3)
{
    // n1, n2 are vector dimensions, sometimes called n, batch_count
    // s1, s1 are vector strides, sometimes called incx, stride_x
    printf("---------- %s ----------\n", name);
    int max_size = 4;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
        {
            for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
            {
                printf("%8.1f ", x[(i1 * s1) + (i2 * s2) + (i3 * s3)]);
            }
            printf("\n");
        }
        printf("---------------------------------------\n");
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
        << "\t-n \t\t\tn\t\tSCAL_STRIDED_BATCHED argument n\n"
        << "\t--incx \t\t\tincx \t\tSCAL_STRIDED_BATCHED argument lda\n"
        << "\t--stride_x \t\tstride_x \tSCAL_STRIDED_BATCHED argument stride_x\n"
        << "\t--batch_count \t\tbatch_count \tSCAL_STRIDED_BATCHED argument batch count\n"
        << "\t--multiple_count \tmultiple_count \tSCAL_STRIDED_BATCHED argument multiple count\n"
        << "\t--alpha \t\talpha \t\tSCAL_STRIDED_BATCHED argument alpha\n"
        << "\t--header \t\theader \t\tprint header for output\n"
        << std::endl;
}

static int parse_arguments(int    argc,
                           char*  argv[],
                           int&   n,
                           int&   incx,
                           int&   stride_x,
                           int&   batch_count,
                           int&   multiple_count,
                           float& alpha,
                           bool&  header,
                           bool&  verbose)
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
                else if((arg == "-n") && (i + 1 < argc))
                {
                    n = atoi(argv[++i]);
                }
                else if((arg == "--batch_count") && (i + 1 < argc))
                {
                    batch_count = atoi(argv[++i]);
                }
                else if((arg == "--multiple_count") && (i + 1 < argc))
                {
                    multiple_count = atoi(argv[++i]);
                }
                else if((arg == "--incx") && (i + 1 < argc))
                {
                    incx = atoi(argv[++i]);
                }
                else if((arg == "--stride_x") && (i + 1 < argc))
                {
                    stride_x = atoi(argv[++i]);
                }
                else if((arg == "--alpha") && (i + 1 < argc))
                {
                    alpha = atof(argv[++i]);
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

bool bad_argument(rocblas_int n, rocblas_int incx, rocblas_int stride_x, rocblas_int batch_count)
{
    bool argument_error = false;
    if(stride_x < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_x < 0" << std::endl;
    }
    if(n < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument n = " << n << " < "
                  << "0" << std::endl;
    }
    if(incx < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument incx = " << incx << " < "
                  << "0" << std::endl;
    }
    if(batch_count < 1)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument batch_count = " << batch_count << " < 1" << std::endl;
    }

    return argument_error;
}

void initialize_x(float* hx, float* hx_gold, int size_x)
{
    srand(1);
    for(int i = 0; i < size_x; i++)
    {
        hx[i] = hx_gold[i] = rand() % 3;
    }
}

int main(int argc, char* argv[])
{
    // invalid int and float for rocblas_sgemm_strided_batched int and float arguments
    rocblas_int invalid_int   = std::numeric_limits<rocblas_int>::min() + 1;
    float       invalid_float = std::numeric_limits<float>::quiet_NaN();

    // initialize to invalid value to detect if values not specified on command line
    rocblas_int n = invalid_int, incx = invalid_int, stride_x = invalid_int;

    rocblas_int batch_count    = invalid_int;
    rocblas_int multiple_count = 2;

    float alpha = invalid_float;

    bool verbose = false;
    bool header  = false;

    if(parse_arguments(
           argc, argv, n, incx, stride_x, batch_count, multiple_count, alpha, header, verbose))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    // when arguments not specified, set to default values
    if(n == invalid_int)
        n = DIM1;
    if(incx == invalid_int)
        incx = 1;
    if(stride_x == invalid_int)
        stride_x = n * incx;
    if(alpha != alpha)
        alpha = ALPHA; // check for alpha == invalid_float == NaN
    if(batch_count == invalid_int)
        batch_count = BATCH_COUNT;

    if(bad_argument(n, incx, stride_x, batch_count))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    host_strided_batch_vector<float>   hx(n, incx, stride_x, batch_count);
    host_strided_batch_vector<float>   hx_gold(n, incx, stride_x, batch_count);
    device_strided_batch_vector<float> dx(n, incx, stride_x, batch_count);
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hx_gold.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());

    int size_x = hx.nmemb();
    initialize_x(hx.data(), hx_gold.data(), size_x);

    if(verbose)
    {
        print_strided_batched_vector("hx initial", hx, n, batch_count, incx, stride_x);
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    CHECK_ROCBLAS_ERROR(
        rocblas_sscal_strided_batched(handle, n, &alpha, dx, incx, stride_x, batch_count));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hx.transfer_from(dx));

    // calculate golden or correct result
    int s1 = incx, s2 = stride_x;
    for(int i2 = 0; i2 < batch_count; i2++)
    {
        for(int i1 = 0; i1 < n; i1++)
        {
            (hx_gold[i2])[i1 * incx] *= alpha;
        }
    }

    if(verbose)
    {
        print_strided_batched_vector("hx calculated", hx, n, batch_count, incx, stride_x);
        print_strided_batched_vector("hx_gold calculated", hx_gold, n, batch_count, incx, stride_x);
    }

    float max_relative_error = std::numeric_limits<float>::min();
    for(int i = 0; i < size_x; i++)
    {
        float relative_error = (hx_gold.data())[i] == 0
                                   ? (hx.data())[i]
                                   : ((hx_gold.data())[i] - (hx.data())[i]) / (hx_gold.data())[i];
        relative_error       = relative_error >= 0 ? relative_error : -relative_error;
        max_relative_error
            = relative_error < max_relative_error ? max_relative_error : relative_error;
    }
    float eps       = std::numeric_limits<float>::epsilon();
    float tolerance = 10;

    if(header)
    {
        std::cout << "N,incx,stride_x,batch_count,multiple_count,alpha" << std::endl;
    }
    std::cout << n << ", " << incx << ", " << stride_x << ", " << batch_count << ", "
              << multiple_count << ", " << alpha;

    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        std::cout << " : FAIL, " << max_relative_error << std::endl;
    }
    else
    {
        std::cout << " : PASS, " << max_relative_error << std::endl;
    }

    host_multiple_strided_batch_vector<float> hx_multiple(
        n, incx, stride_x, batch_count, multiple_count);
    device_multiple_strided_batch_vector<float> dx_multiple(
        n, incx, stride_x, batch_count, multiple_count);

    CHECK_HIP_ERROR(dx_multiple.memcheck());

    CHECK_HIP_ERROR(dx_multiple.broadcast_one_strided_batch_vector_from(hx));

    CHECK_HIP_ERROR(hx_multiple.transfer_from(dx_multiple));
    if(verbose)
    {
        print_multiple_strided_batched_vector("hx_multiple before scal",
                                              hx_multiple,
                                              n,
                                              batch_count,
                                              multiple_count,
                                              incx,
                                              stride_x,
                                              hx_multiple.multiple_stride());
    }

    for(int i = 0; i < multiple_count; i++)
    {
        CHECK_ROCBLAS_ERROR(rocblas_sscal_strided_batched(
            handle, n, &alpha, dx_multiple[i], incx, stride_x, batch_count));
    }

    CHECK_HIP_ERROR(hx_multiple.transfer_from(dx_multiple));

    if(verbose)
    {
        print_multiple_strided_batched_vector("hx_multiple after scal",
                                              hx_multiple,
                                              n,
                                              batch_count,
                                              multiple_count,
                                              incx,
                                              stride_x,
                                              hx_multiple.multiple_stride());
    }

    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}
