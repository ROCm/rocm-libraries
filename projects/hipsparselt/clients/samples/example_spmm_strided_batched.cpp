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

// default sizes
#define DIM1 127
#define DIM2 128
#define DIM3 129
#define BATCH_COUNT 10
#define ALPHA 2
#define BETA 3

template <typename T>
inline bool AlmostEqual(T a, T b)
{
    return false;
}

template <>
inline bool AlmostEqual(__half a, __half b)
{
#if defined(__HIP_PLATFORM_AMD__)
    union _HALF
    {
        uint16_t x;
        _Float16 data;
    };

    _HALF a_half    = {__half_raw(a).x};
    _HALF b_half    = {__half_raw(b).x};
    _HALF zero_half = {__half_raw(static_cast<__half>(0)).x};
    _HALF one_half  = {__half_raw(static_cast<__half>(1)).x};
    _HALF e_n2_half = {__half_raw(static_cast<__half>(0.01)).x};

    auto  a_data = a_half.data;
    auto  b_data = b_half.data;
    auto  zero   = zero_half.data;
    auto  one    = one_half.data;
    auto  e_n2   = e_n2_half.data;
#else
    auto a_data = a;
    auto b_data = b;
    auto zero = __half(0);
    auto one  = __half(1);
    auto e_n2 = __half(0.01);
#endif
    auto absA = (a_data > zero) ? a_data : static_cast<decltype(a_data)>(-a_data);
    auto absB = (b_data > zero) ? b_data : static_cast<decltype(b_data)>(-b_data);
    // this avoids NaN when inf is compared against inf in the alternative code
    // path
    if(static_cast<float>(absA) == std::numeric_limits<float>::infinity()
       || // numeric_limits is yet to
       // support _Float16 type
       // properly;
       static_cast<float>(absB)
           == std::numeric_limits<float>::infinity()) // however promoting it to
    // float works just as fine
    {
        return a_data == b_data;
    }
    auto absDiff = (a_data - b_data > zero) ? a_data - b_data : b_data - a_data;
    return absDiff / (absA + absB + one) < e_n2;
}

inline void extract_metadata(unsigned metadata, int& a, int& b, int& c, int& d)
{
    a = metadata & 0x03;
    b = (metadata >> 2) & 0x03;
    c = ((metadata >> 4) & 0x03);
    d = ((metadata >> 6) & 0x03);
}

std::string metadata_to_bits_str(unsigned char meta)
{
    std::string str;
    for(int b = 0; b < 8; b++)
    {
        str.append(std::to_string(meta >> (7 - b) & 0x01));
    }
    return str;
}

void print_strided_batched_meta(const char*    name,
                                unsigned char* A,
                                int64_t        n1,
                                int64_t        n2,
                                int64_t        n3,
                                int64_t        s1,
                                int64_t        s2,
                                int64_t        s3)
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
                auto meta = A[(i1 * s1) + (i2 * s2) + (i3 * s3)];
                auto str  = metadata_to_bits_str(meta);
                int  a, b, c, d;
                extract_metadata(meta, a, b, c, d);
                std::printf("[%ld][bits=%s]%02x%02x%02x%02x\t",
                            (i1 * s1) + (i2 * s2) + (i3 * s3),
                            str.c_str(),
                            a,
                            b,
                            c + 4,
                            d + 4);
            }
            printf("\n");
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            printf("\n");
    }
}

void print_strided_batched(const char* name,
                           __half*     A,
                           int64_t     n1,
                           int64_t     n2,
                           int64_t     n3,
                           int64_t     s1,
                           int64_t     s2,
                           int64_t     s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    printf("---------- %s (MxN=%ldx%ld,batch=%ld,stride0=%ld, stride1=%ld)----------\n",
           name,
           n1,
           n2,
           n3,
           s1,
           s2);
    int max_size = 128;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                printf("[%ld]\t%8.1f\t",
                       (i1 * s1) + (i2 * s2) + (i3 * s3),
                       static_cast<float>(A[(i1 * s1) + (i2 * s2) + (i3 * s3)]));
            }
            printf("\n");
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            printf("\n");
    }
}

template <typename Ti, typename To, typename Tc>
void mat_mat_mult(Tc        alpha,
                  Tc        beta,
                  int       M,
                  int       N,
                  int       K,
                  const Ti* A,
                  int       As1,
                  int       As2,
                  const Ti* B,
                  int       Bs1,
                  int       Bs2,
                  const To* C,
                  int       Cs1,
                  int       Cs2,
                  To*       D,
                  int       Ds1,
                  int       Ds2)
{
    for(int i1 = 0; i1 < M; i1++)
    {
        for(int i2 = 0; i2 < N; i2++)
        {
            Tc t = static_cast<Tc>(0);
            for(int i3 = 0; i3 < K; i3++)
            {
                t += static_cast<Tc>(A[i1 * As1 + i3 * As2])
                     * static_cast<Tc>(B[i3 * Bs1 + i2 * Bs2]);
            }
            D[i1 * Ds1 + i2 * Ds2]
                = static_cast<To>(beta * static_cast<Tc>(C[i1 * Cs1 + i2 * Cs2]) + alpha * t);
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
              << "\t--ldd \t\t\tldd \t\tGEMM_STRIDED_BATCHED argument ldc\n"
              << "\t--trans_a \t\ttrans_a \tGEMM_STRIDED_BATCHED argument trans_a\n"
              << "\t--trans_b \t\ttrans_b \tGEMM_STRIDED_BATCHED argument trans_b\n"
              << "\t--stride_a \t\tstride_a \tGEMM_STRIDED_BATCHED argument stride_a\n"
              << "\t--stride_b \t\tstride_b \tGEMM_STRIDED_BATCHED argument stride_b\n"
              << "\t--stride_c \t\tstride_c \tGEMM_STRIDED_BATCHED argument stride_c\n"
              << "\t--stride_d \t\tstride_d \tGEMM_STRIDED_BATCHED argument stride_d\n"
              << "\t--batch_count \t\tbatch_count \tGEMM_STRIDED_BATCHED argument batch count\n"
              << "\t--alpha \t\talpha \t\tGEMM_STRIDED_BATCHED argument alpha\n"
              << "\t--beta \t\t\tbeta \t\tGEMM_STRIDED_BATCHED argument beta\n"
              << "\t--sparse_b \t\tsparse_b \t\tStructurted Sparsity Matrix B (A is Dense Matrix)\n"
              << "\t--header \t\theader \t\tprint header for output\n"
              << std::endl;
}

static int parse_arguments(int                   argc,
                           char*                 argv[],
                           int64_t&              m,
                           int64_t&              n,
                           int64_t&              k,
                           int64_t&              lda,
                           int64_t&              ldb,
                           int64_t&              ldc,
                           int64_t&              ldd,
                           int64_t&              stride_a,
                           int64_t&              stride_b,
                           int64_t&              stride_c,
                           int64_t&              stride_d,
                           int&                  batch_count,
                           float&                alpha,
                           float&                beta,
                           hipsparseOperation_t& trans_a,
                           hipsparseOperation_t& trans_b,
                           bool&                 sparse_b,
                           bool&                 header,
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
                else if((arg == "--ldd") && (i + 1 < argc))
                {
                    ldd = atoi(argv[++i]);
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
                else if((arg == "--stride_d") && (i + 1 < argc))
                {
                    stride_d = atoi(argv[++i]);
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
                        trans_a = HIPSPARSE_OPERATION_NON_TRANSPOSE;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_a = HIPSPARSE_OPERATION_TRANSPOSE;
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
                        trans_b = HIPSPARSE_OPERATION_NON_TRANSPOSE;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_b = HIPSPARSE_OPERATION_TRANSPOSE;
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

bool bad_argument(hipsparseOperation_t trans_a,
                  hipsparseOperation_t trans_b,
                  int64_t              m,
                  int64_t              n,
                  int64_t              k,
                  int64_t              lda,
                  int64_t              ldb,
                  int64_t              ldc,
                  int64_t              ldd,
                  int64_t              stride_a,
                  int64_t              stride_b,
                  int64_t              stride_c,
                  int64_t              stride_d,
                  int64_t              batch_count)
{
    bool argument_error = false;
    if((trans_a == HIPSPARSE_OPERATION_NON_TRANSPOSE) && (lda < m))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << lda << " < " << m << std::endl;
    }
    if((trans_a == HIPSPARSE_OPERATION_TRANSPOSE) && (lda < k))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << lda << " < " << k << std::endl;
    }
    if((trans_b == HIPSPARSE_OPERATION_NON_TRANSPOSE) && (ldb < k))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldb = " << ldb << " < " << k << std::endl;
    }
    if((trans_b == HIPSPARSE_OPERATION_TRANSPOSE) && (ldb < n))
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
    if(batch_count < 1)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument batch_count = " << batch_count << " < 1" << std::endl;
    }

    return argument_error;
}

void initialize_a_b_c(std::vector<__half>& ha,
                      int64_t              size_a,
                      std::vector<__half>& hb,
                      int64_t              size_b,
                      std::vector<__half>& hc,
                      int64_t              size_c)
{
    auto CAST = [](auto x) {
#if defined(__HIP_PLATFORM_AMD__)
        return static_cast<__half>(x);
#else
        return static_cast<__half>(static_cast<float>(x));
#endif
    };

    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = CAST((rand() % 7) - 3);
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = CAST((rand() % 7) - 3);
    }
    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = CAST((rand() % 7) - 3);
    }
}

int main(int argc, char* argv[])
{
    try
    {
        // initialize parameters with default values
        hipsparseOperation_t trans_a = HIPSPARSE_OPERATION_NON_TRANSPOSE;
        hipsparseOperation_t trans_b = HIPSPARSE_OPERATION_TRANSPOSE;

        // invalid int and float for hipsparselt spmm int and float arguments
        int64_t invalid_int64 = std::numeric_limits<int64_t>::min() + 1;
        int     invalid_int   = std::numeric_limits<int>::min() + 1;
        float   invalid_float = std::numeric_limits<float>::quiet_NaN();

        // initialize to invalid value to detect if values not specified on command line
        int64_t m = invalid_int64, lda = invalid_int64, stride_a = invalid_int64;
        int64_t n = invalid_int64, ldb = invalid_int64, stride_b = invalid_int64;
        int64_t k = invalid_int64, ldc = invalid_int64, stride_c = invalid_int64;
        int64_t ldd = invalid_int64, stride_d = invalid_int64;

        int batch_count = invalid_int;

        float alpha = invalid_float;
        float beta  = invalid_float;

        bool sparse_b = false;
        bool verbose  = false;
        bool header   = false;

        if(parse_arguments(argc,
                           argv,
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
                           sparse_b,
                           header,
                           verbose))
        {
            show_usage(argv);
            return EXIT_FAILURE;
        }

        // when arguments not specified, set to default values
        if(m == invalid_int64)
            m = DIM1;
        if(n == invalid_int64)
            n = DIM2;
        if(k == invalid_int64)
            k = DIM3;
        if(lda == invalid_int64)
            lda = trans_a == HIPSPARSE_OPERATION_NON_TRANSPOSE ? m : k;
        if(ldb == invalid_int64)
            ldb = trans_b == HIPSPARSE_OPERATION_NON_TRANSPOSE ? k : n;
        if(ldc == invalid_int64)
            ldc = m;
        if(ldd == invalid_int64)
            ldd = m;
        if(stride_a == invalid_int64)
            stride_a = trans_a == HIPSPARSE_OPERATION_NON_TRANSPOSE ? lda * k : lda * m;
        if(stride_b == invalid_int64)
            stride_b = trans_b == HIPSPARSE_OPERATION_NON_TRANSPOSE ? ldb * n : ldb * k;
        if(stride_c == invalid_int64)
            stride_c = ldc * n;
        if(stride_d == invalid_int64)
            stride_d = ldd * n;
        if(alpha != alpha)
            alpha = ALPHA; // check for alpha == invalid_float == NaN
        if(beta != beta)
            beta = BETA; // check for beta == invalid_float == NaN
        if(batch_count == invalid_int)
            batch_count = BATCH_COUNT;

        if(bad_argument(trans_a,
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
                        batch_count))
        {
            show_usage(argv);
            return EXIT_FAILURE;
        }

        if(header)
        {
            std::cout
                << "transAB,M,N,K,lda,ldb,ldc,stride_a,stride_b,stride_c,batch_count,alpha,beta,"
                   "result,error";
            std::cout << std::endl;
        }

        int64_t a_stride_1, a_stride_2, b_stride_1, b_stride_2;
        int64_t row_a, col_a, row_b, col_b, row_c, col_c;
        int     size_a1, size_b1, size_c1 = ldc * n, size_d1 = ldd * n;
        if(trans_a == HIPSPARSE_OPERATION_NON_TRANSPOSE)
        {
            std::cout << "N";
            row_a      = m;
            col_a      = k;
            a_stride_1 = 1;
            a_stride_2 = lda;
            size_a1    = lda * k;
        }
        else
        {
            std::cout << "T";
            row_a      = k;
            col_a      = m;
            a_stride_1 = lda;
            a_stride_2 = 1;
            size_a1    = lda * m;
        }
        if(trans_b == HIPSPARSE_OPERATION_NON_TRANSPOSE)
        {
            std::cout << "N, ";
            row_b      = k;
            col_b      = n;
            b_stride_1 = 1;
            b_stride_2 = ldb;
            size_b1    = ldb * n;
        }
        else
        {
            std::cout << "T, ";
            row_b      = n;
            col_b      = k;
            b_stride_1 = ldb;
            b_stride_2 = 1;
            size_b1    = ldb * k;
        }
        row_c = m;
        col_c = n;

        std::cout << m << ", " << n << ", " << k << ", " << lda << ", " << ldb << ", " << ldc
                  << ", " << ldd << ", " << stride_a << ", " << stride_b << ", " << stride_c << ", "
                  << stride_d << ", " << batch_count << ", " << alpha << ", " << beta << ", ";
        int64_t stride_a_r = stride_a == 0 ? size_a1 : stride_a;
        int64_t stride_b_r = stride_b == 0 ? size_b1 : stride_b;
        int64_t stride_c_r = stride_c == 0 ? size_c1 : stride_c;
        int64_t stride_d_r = stride_d == 0 ? size_d1 : stride_d;

        int64_t size_a = stride_a_r * (stride_a == 0 ? 1 : batch_count);
        int64_t size_b = stride_b_r * (stride_b == 0 ? 1 : batch_count);
        int64_t size_c = stride_c_r * (stride_c == 0 ? 1 : batch_count);
        int64_t size_d = stride_d_r * (stride_d == 0 ? 1 : batch_count);
        // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
        std::vector<__half> ha(size_a);
        std::vector<__half> h_prune(sparse_b ? size_b : size_a);
        std::vector<__half> hb(size_b);
        std::vector<__half> hc(size_c);
        std::vector<__half> hd(size_d);
        std::vector<__half> hd_gold(size_d);

        // initial data on host
        initialize_a_b_c(ha, size_a, hb, size_b, hc, size_c);

        if(verbose)
        {
            printf("\n");
            if(trans_a == HIPSPARSE_OPERATION_NON_TRANSPOSE)
            {
                print_strided_batched("ha initial", &ha[0], m, k, batch_count, 1, lda, stride_a_r);
            }
            else
            {
                print_strided_batched("ha initial", &ha[0], m, k, batch_count, lda, 1, stride_a_r);
            }
            if(trans_b == HIPSPARSE_OPERATION_NON_TRANSPOSE)
            {
                print_strided_batched("hb initial", &hb[0], k, n, batch_count, 1, ldb, stride_b_r);
            }
            else
            {
                print_strided_batched("hb initial", &hb[0], k, n, batch_count, ldb, 1, stride_b_r);
            }
            print_strided_batched("hc initial", &hc[0], m, n, batch_count, 1, ldc, stride_c_r);
        }

        // allocate memory on device
        __half *    da, *dp, *db, *dc, *dd, *d_compressed, *d_compressBuffer;
        void*       d_workspace = nullptr;
        int         num_streams = 1;
        hipStream_t stream      = nullptr;
        hipStream_t streams[1]  = {stream};

        CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(__half)));
        SMART_DESTROYER<void, hipError_t> sDA(da, hipFree);
        CHECK_HIP_ERROR(hipMalloc(&dp, (sparse_b ? size_b : size_a) * sizeof(__half)));
        SMART_DESTROYER<void, hipError_t> sDP(dp, hipFree);
        CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(__half)));
        SMART_DESTROYER<void, hipError_t> sDB(db, hipFree);
        CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(__half)));
        SMART_DESTROYER<void, hipError_t> sDC(dc, hipFree);
        CHECK_HIP_ERROR(hipMalloc(&dd, size_d * sizeof(__half)));
        SMART_DESTROYER<void, hipError_t> sDD(dd, hipFree);
        // copy matrices from host to device
        CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(__half) * size_a, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(__half) * size_b, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(__half) * size_c, hipMemcpyHostToDevice));

        hipsparseLtHandle_t             handle;
        hipsparseLtMatDescriptor_t      matA, matB, matC, matD;
        hipsparseLtMatmulDescriptor_t   matmul;
        hipsparseLtMatmulAlgSelection_t alg_sel;
        hipsparseLtMatmulPlan_t         plan;

        CHECK_HIPSPARSELT_ERROR(hipsparseLtInit(&handle));
        SMART_DESTROYER<const hipsparseLtHandle_t, hipsparseStatus_t> sH(&handle,
                                                                         hipsparseLtDestroy);

        if(!sparse_b)
        {
            CHECK_HIPSPARSELT_ERROR(
                hipsparseLtStructuredDescriptorInit(&handle,
                                                    &matA,
                                                    row_a,
                                                    col_a,
                                                    lda,
                                                    16,
                                                    HIP_R_16F,
                                                    HIPSPARSE_ORDER_COL,
                                                    HIPSPARSELT_SPARSITY_50_PERCENT));
        }
        else
        {
            CHECK_HIPSPARSELT_ERROR(hipsparseLtDenseDescriptorInit(
                &handle, &matA, row_a, col_a, lda, 16, HIP_R_16F, HIPSPARSE_ORDER_COL));
        }
        SMART_DESTROYER<const hipsparseLtMatDescriptor_t, hipsparseStatus_t> smA(
            &matA, hipsparseLtMatDescriptorDestroy);

        if(!sparse_b)
        {
            CHECK_HIPSPARSELT_ERROR(hipsparseLtDenseDescriptorInit(
                &handle, &matB, row_b, col_b, ldb, 16, HIP_R_16F, HIPSPARSE_ORDER_COL));
        }
        else
        {
            CHECK_HIPSPARSELT_ERROR(
                hipsparseLtStructuredDescriptorInit(&handle,
                                                    &matB,
                                                    row_b,
                                                    col_b,
                                                    ldb,
                                                    16,
                                                    HIP_R_16F,
                                                    HIPSPARSE_ORDER_COL,
                                                    HIPSPARSELT_SPARSITY_50_PERCENT));
        }
        SMART_DESTROYER<const hipsparseLtMatDescriptor_t, hipsparseStatus_t> smB(
            &matB, hipsparseLtMatDescriptorDestroy);

        CHECK_HIPSPARSELT_ERROR(hipsparseLtDenseDescriptorInit(
            &handle, &matC, row_c, col_c, ldc, 16, HIP_R_16F, HIPSPARSE_ORDER_COL));
        SMART_DESTROYER<const hipsparseLtMatDescriptor_t, hipsparseStatus_t> smC(
            &matC, hipsparseLtMatDescriptorDestroy);

        CHECK_HIPSPARSELT_ERROR(hipsparseLtDenseDescriptorInit(
            &handle, &matD, row_c, col_c, ldd, 16, HIP_R_16F, HIPSPARSE_ORDER_COL));
        SMART_DESTROYER<const hipsparseLtMatDescriptor_t, hipsparseStatus_t> smD(
            &matD, hipsparseLtMatDescriptorDestroy);

        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
            &handle, &matA, HIPSPARSELT_MAT_NUM_BATCHES, &batch_count, sizeof(batch_count)));
        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
            &handle, &matA, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_a, sizeof(stride_a)));
        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
            &handle, &matB, HIPSPARSELT_MAT_NUM_BATCHES, &batch_count, sizeof(batch_count)));
        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
            &handle, &matB, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_b, sizeof(stride_b)));
        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
            &handle, &matC, HIPSPARSELT_MAT_NUM_BATCHES, &batch_count, sizeof(batch_count)));
        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
            &handle, &matC, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_c, sizeof(stride_c)));
        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
            &handle, &matD, HIPSPARSELT_MAT_NUM_BATCHES, &batch_count, sizeof(batch_count)));
        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
            &handle, &matD, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_d, sizeof(stride_d)));

        auto compute_type =
#ifdef __HIP_PLATFORM_AMD__
            HIPSPARSELT_COMPUTE_32F;
#else
            HIPSPARSELT_COMPUTE_16F;
#endif

        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmulDescriptorInit(
            &handle, &matmul, trans_a, trans_b, &matA, &matB, &matC, &matD, compute_type));

        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmulAlgSelectionInit(
            &handle, &alg_sel, &matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT));

        CHECK_HIPSPARSELT_ERROR(hipsparseLtSpMMAPrune(
            &handle, &matmul, sparse_b ? db : da, dp, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream));

        size_t workspace_size, compressed_size, compress_buffer_size;
        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));
        SMART_DESTROYER<const hipsparseLtMatmulPlan_t, hipsparseStatus_t> sP(
            &plan, hipsparseLtMatmulPlanDestroy);

        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));

        CHECK_HIPSPARSELT_ERROR(hipsparseLtSpMMACompressedSize(
            &handle, &plan, &compressed_size, &compress_buffer_size));

        CHECK_HIP_ERROR(hipMalloc(&d_compressed, compressed_size));
        SMART_DESTROYER<void, hipError_t> sd_compressed(d_compressed, hipFree);

        CHECK_HIP_ERROR(hipMalloc(&d_compressBuffer, compress_buffer_size));
        SMART_DESTROYER<void, hipError_t> sd_compressBuffer(d_compressBuffer, hipFree);

        CHECK_HIPSPARSELT_ERROR(
            hipsparseLtSpMMACompress(&handle, &plan, dp, d_compressed, d_compressBuffer, stream));
        if(workspace_size > 0)
        {
            CHECK_HIP_ERROR(hipMalloc(&d_workspace, workspace_size));
        }
        SMART_DESTROYER<void, hipError_t> sd_workspace(d_workspace, hipFree);

        CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmul(&handle,
                                                  &plan,
                                                  &alpha,
                                                  sparse_b ? da : d_compressed,
                                                  sparse_b ? d_compressed : db,
                                                  &beta,
                                                  dc,
                                                  dd,
                                                  d_workspace,
                                                  &streams[0],
                                                  num_streams));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hd.data(), dd, sizeof(__half) * size_c, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(h_prune.data(),
                                  dp,
                                  sizeof(__half) * (sparse_b ? size_b : size_a),
                                  hipMemcpyDeviceToHost));
        // calculate golden or correct result
        for(int i = 0; i < batch_count; i++)
        {
            __half* a_ptr = sparse_b ? &ha[i * stride_a] : &h_prune[i * stride_a];
            __half* b_ptr = sparse_b ? &h_prune[i * stride_b] : &hb[i * stride_b];
            __half* c_ptr = &hc[i * stride_c];
            __half* d_ptr = &hd_gold[i * stride_d];
            mat_mat_mult<__half, __half, float>(alpha,
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
                                                ldc,
                                                d_ptr,
                                                1,
                                                ldd);
        }
        if(verbose)
        {
            std::vector<__half> h_compressed(compressed_size);
            CHECK_HIP_ERROR(
                hipMemcpy(&h_compressed[0], d_compressed, compressed_size, hipMemcpyDeviceToHost));

            auto batch_count_c = ((sparse_b ? stride_b : stride_a) == 0) ? 1 : batch_count;

            int64_t c_stride_1, c_stride_2, c_stride_b, c_stride_b_r;
            int64_t m_stride_1, m_stride_2, m_stride_b, m_stride_b_r;
            if(!sparse_b)
            {
                c_stride_1   = (trans_a == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? 1 : k / 2;
                c_stride_2   = (trans_a == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : 1;
                c_stride_b_r = (trans_a == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k / 2 * c_stride_2
                                                                              : m * c_stride_1;

                m_stride_1   = k / 8;
                m_stride_2   = 1;
                m_stride_b_r = m * m_stride_1;
            }
            else
            {
                c_stride_1   = (trans_b == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? 1 : n;
                c_stride_2   = (trans_b == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k / 2 : 1;
                c_stride_b_r = (trans_b == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n * c_stride_2
                                                                              : k / 2 * c_stride_1;

                m_stride_1   = 1;
                m_stride_2   = k / 8;
                m_stride_b_r = n * m_stride_2;
            }

            c_stride_b = (sparse_b ? stride_b : stride_a) == 0 ? 0 : c_stride_b_r;
            m_stride_b = (sparse_b ? stride_b : stride_a) == 0 ? 0 : m_stride_b_r;

            if(!sparse_b)
            {
                print_strided_batched("device compress calculated",
                                      &h_compressed[0],
                                      m,
                                      k / 2,
                                      batch_count_c,
                                      c_stride_1,
                                      c_stride_2,
                                      c_stride_b_r);
                print_strided_batched_meta(
                    "device metadata calculated",
                    reinterpret_cast<unsigned char*>(&h_compressed[c_stride_b_r * batch_count_c]),
                    m,
                    k / 8,
                    batch_count_c,
                    m_stride_1,
                    m_stride_2,
                    m_stride_b_r);
            }
            else
            {
                print_strided_batched("device compress calculated",
                                      &h_compressed[0],
                                      k / 2,
                                      n,
                                      batch_count_c,
                                      c_stride_1,
                                      c_stride_2,
                                      c_stride_b_r);
                print_strided_batched_meta(
                    "device metadata calculated",
                    reinterpret_cast<unsigned char*>(&h_compressed[c_stride_b_r * batch_count_c]),
                    k / 8,
                    n,
                    batch_count_c,
                    m_stride_1,
                    m_stride_2,
                    m_stride_b_r);
            }
            print_strided_batched(
                "hc_gold calculated", &hd_gold[0], m, n, batch_count, 1, ldd, stride_d_r);
            print_strided_batched("hd calculated", &hd[0], m, n, batch_count, 1, ldd, stride_d_r);
        }

        bool passed = true;
        for(int i = 0; i < size_c; i++)
        {
            if(!AlmostEqual(hd_gold[i], hd[i]))
            {
                printf(
                    "Err: %f vs %f\n", static_cast<float>(hd_gold[i]), static_cast<float>(hd[i]));
                passed = false;
            }
        }
        if(!passed)
        {
            std::cout << "FAIL" << std::endl;
        }
        else
        {
            std::cout << "PASS" << std::endl;
        }

        return EXIT_SUCCESS;
    }
    catch(int)
    {
        return EXIT_FAILURE;
    }
}
