/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas_arguments.hpp"
#include "../../library/src/include/tuple_helper.hpp"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <istream>
#include <ostream>
#include <utility>

/*! \brief device matches pattern */
bool gpu_arch_match(const std::string& gpu_arch, const char pattern[4])
{
    int         gpu_len = gpu_arch.length();
    const char* gpu     = gpu_arch.c_str();

    // gpu is currently "gfx" followed by 3 or 4 characters, followed by optional ":" sections
    int prefix_len = 3;
    for(int i = 0; i < 4; i++)
    {
        if(!pattern[i])
            break;
        else if(pattern[i] == '?')
            continue;
        else if(prefix_len + i >= gpu_len || pattern[i] != gpu[prefix_len + i])
            return false;
    }
    return true;
};

void Arguments::init()
{
    // match python in rocblas_common.py

    function[0] = 0;
    strcpy(name, "rocblas-bench");
    category[0]            = 0;
    known_bug_platforms[0] = 0;

    // 64bit

    alpha  = 1.0;
    alphai = 0.0;
    beta   = 0.0;
    betai  = 0.0;

    stride_a = 0;
    stride_b = 0;
    stride_c = 0;
    stride_d = 0;
    stride_x = 0;
    stride_y = 0;

    user_allocated_workspace = 0;

    // 64bit

    M = 128;
    N = 128;
    K = 128;

    KL = 128;
    KU = 128;

    lda = 0;
    ldb = 0;
    ldc = 0;
    ldd = 0;

    incx = 0;
    incy = 0;

    batch_count = 1;

    scan   = c_scan_value; // member used to set multiple args set with scan_value
    scan_2 = c_scan_value_2; // 2d scanning support

    // 32bit

    iters      = 10;
    cold_iters = 2;

    algo           = 0;
    solution_index = 0;

    geam_ex_op = rocblas_geam_ex_operation_min_plus;

    flags = rocblas_gemm_flags_none;

    a_type       = rocblas_datatype_f32_r;
    b_type       = rocblas_datatype_f32_r;
    c_type       = rocblas_datatype_f32_r;
    d_type       = rocblas_datatype_f32_r;
    compute_type = rocblas_datatype_f32_r;

    initialization = rocblas_initialization::hpl;

    atomics_mode = rocblas_atomics_not_allowed;

    math_mode = rocblas_default_math;

    os_flags = rocblas_client_os::ALL;

    gpu_arch[0] = 0; // 4 chars so 32bit

    api = rocblas_client_api::C;

    // memory padding for testing write out of bounds
    pad = 4096;

    // 16 bit
    threads = 0;
    streams = 0;

    // bytes
    devices = 0;

    norm_check = 0;
    unit_check = 1;
    timing     = 0;

    transA = '*';
    transB = '*';
    side   = '*';
    uplo   = '*';
    diag   = '*';

    pointer_mode_host   = true;
    pointer_mode_device = true;
    outofplace          = false;
    HMM                 = false;
    graph_test          = false;
    repeatability_check = false;

    use_hipblaslt = -1;

    cleanup = true;
}

bool Arguments::validate()
{
// c_scan_value must matching value in rocblas_common.yaml definition
#define SCAN_VALUE_CHECK(arg_)      \
    if(arg_ == c_scan_value)        \
        arg_ = scan;                \
    else if(arg_ == c_scan_value_2) \
    arg_ = scan_2

    if(scan != c_scan_value)
    {
        // int64_t variables all same as scan

        SCAN_VALUE_CHECK(stride_a);
        SCAN_VALUE_CHECK(stride_b);
        SCAN_VALUE_CHECK(stride_c);
        SCAN_VALUE_CHECK(stride_d);
        SCAN_VALUE_CHECK(stride_x);
        SCAN_VALUE_CHECK(stride_y);

        SCAN_VALUE_CHECK(M);
        SCAN_VALUE_CHECK(N);
        SCAN_VALUE_CHECK(K);

        SCAN_VALUE_CHECK(KL);
        SCAN_VALUE_CHECK(KU);

        SCAN_VALUE_CHECK(lda);
        SCAN_VALUE_CHECK(ldb);
        SCAN_VALUE_CHECK(ldc);
        SCAN_VALUE_CHECK(ldd);

        SCAN_VALUE_CHECK(incx);
        SCAN_VALUE_CHECK(incy);

        SCAN_VALUE_CHECK(batch_count);
    }

#undef SCAN_VALUE_CHECK

    // future use, for now everything is valid after rules appled
    return true;
}

static Arguments& getDefaultArgs()
{
    static Arguments defaultArguments;
    static int       once = (defaultArguments.init(), 1);
    return defaultArguments;
}
static Arguments& gDefArgs = getDefaultArgs();

template <typename T>
bool member_different(const T& a, T& b)
{
    if constexpr(std::is_same_v<T, char[4]> || std::is_same_v<T, char[64]>)
        return true;
    else
        return a != b;
}

// Function to print Arguments out to stream in YAML format
rocblas_internal_ostream& operator<<(rocblas_internal_ostream& os, const Arguments& arg)
{
    // delim starts as "{ " and becomes ", " afterwards
    auto print_pair = [&, delim = "{ "](const char* name, const auto& value) mutable {
        os << delim << std::make_pair(name, value);
        delim = ", ";
    };

    // Print each (name, value) tuple pair if not default value
#define NAME_VALUE_PAIR(NAME)                     \
    if(member_different(arg.NAME, gDefArgs.NAME)) \
    print_pair(#NAME, arg.NAME)

    // cppcheck-suppress unknownMacro
    FOR_EACH_ARGUMENT(NAME_VALUE_PAIR, ;);

    // Closing brace
    return os << " }\n";
}

// Google Tests uses this automatically with std::ostream to dump parameters
std::ostream& operator<<(std::ostream& os, const Arguments& arg)
{
    rocblas_internal_ostream oss;
    // Print to rocblas_internal_ostream, then transfer to std::ostream
    return os << (oss << arg);
}

// Function to read Structures data from stream
std::istream& operator>>(std::istream& is, Arguments& arg)
{
    is.read(reinterpret_cast<char*>(&arg), sizeof(arg));
    return is;
}

// Error message about incompatible binary file format
static void validation_error [[noreturn]] (const char* name)
{
    rocblas_cerr << "Arguments field \"" << name
                 << "\" does not match format.\n\n"
                    "Fatal error: Binary test data does match input format.\n"
                    "Ensure that rocblas_arguments.hpp and rocblas_common.yaml\n"
                    "define exactly the same Arguments, that rocblas_gentest.py\n"
                    "generates the data correctly, and that endianness is the same."
                 << std::endl;
    rocblas_abort();
}

// rocblas_gentest.py is expected to conform to this format.
// rocblas_gentest.py uses rocblas_common.yaml to generate this format.
void Arguments::validate(std::istream& ifs)
{
    char      header[8]{}, trailer[8]{};
    Arguments arg{};

    ifs.read(header, sizeof(header));
    ifs >> arg;
    ifs.read(trailer, sizeof(trailer));

    if(strcmp(header, "rocBLAS"))
        validation_error("header");

    if(strcmp(trailer, "ROCblas"))
        validation_error("trailer");

    auto check_func = [sig = 0u](const char* name, const auto& value) mutable {
        static_assert(sizeof(value) <= 256,
                      "Fatal error: Arguments field is too large (greater than 256 bytes).");
        for(size_t i = 0; i < sizeof(value); ++i)
        {
            if(reinterpret_cast<const unsigned char*>(&value)[i] ^ sig ^ i)
                validation_error(name);
        }
        sig = (sig + 89) % 256;
    };

    // Apply check_func to each pair (name, value) of Arguments as a tuple
#define CHECK_FUNC(NAME) check_func(#NAME, arg.NAME)
    FOR_EACH_ARGUMENT(CHECK_FUNC, ;);
}
