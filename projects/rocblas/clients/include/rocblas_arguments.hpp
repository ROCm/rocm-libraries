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

#pragma once

#include "../../library/src/include/rocblas_ostream.hpp"
#include "rocblas.h"
#include "rocblas_datatype2string.hpp"
#include "rocblas_math.hpp"
#include <cstddef>
#include <istream>
#include <map>
#include <ostream>
#include <tuple>

// Predeclare enumerator
enum rocblas_argument : int;

// bit mask rocblas_client_api_
const uint32_t c_API_64       = 1;
const uint32_t c_API_FORTRAN  = 2;
const uint32_t c_API_INTERNAL = 4;

// last bit set for _64 API
typedef enum rocblas_client_api_
{
    C           = 0,
    C_64        = 1,
    FORTRAN     = 2,
    FORTRAN_64  = 3,
    INTERNAL    = 4,
    INTERNAL_64 = 5
} rocblas_client_api;

// bitmask
typedef enum rocblas_client_os_
{
    LINUX   = 1,
    WINDOWS = 2,
    ALL     = 3
} rocblas_client_os;

/*! \brief device matches pattern */
bool gpu_arch_match(const std::string& gpu_arch, const char pattern[4]);

/***************************************************************************
 *! \brief Class used to parse command arguments in both client & gtest    *
 * WARNING: If this data is changed, then rocblas_common.yaml must also be *
 * changed.                                                                *
 ***************************************************************************/
struct Arguments
{
    static constexpr int64_t c_scan_value   = -999;
    static constexpr int64_t c_scan_value_2 = -998;

    /*************************************************************************
     *                    Beginning Of Arguments                             *
     *************************************************************************/

    char function[64];
    char name[64];
    char category[64];
    char known_bug_platforms[64];

    // 64bit

    double alpha;
    double alphai;
    double beta;
    double betai;

    rocblas_stride stride_a; //  stride_a > transA == 'N' ? lda * K : lda * M
    rocblas_stride stride_b; //  stride_b > transB == 'N' ? ldb * N : ldb * K
    rocblas_stride stride_c; //  stride_c > ldc * N
    rocblas_stride stride_d; //  stride_d > ldd * N
    rocblas_stride stride_x;
    rocblas_stride stride_y;

    size_t user_allocated_workspace;

    // 64bit

    int64_t M;
    int64_t N;
    int64_t K;

    int64_t KL;
    int64_t KU;

    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldd;

    int64_t incx;
    int64_t incy;

    int64_t batch_count;

    int64_t scan;
    int64_t scan_2;

    int32_t iters;
    int32_t cold_iters;

    uint32_t algo;
    int32_t  solution_index;

    rocblas_geam_ex_operation geam_ex_op;

    rocblas_gemm_flags flags;

    rocblas_datatype a_type;
    rocblas_datatype b_type;
    rocblas_datatype c_type;
    rocblas_datatype d_type;
    rocblas_datatype compute_type;

    rocblas_initialization initialization;

    rocblas_atomics_mode atomics_mode;

    rocblas_client_os os_flags;

    // the gpu arch string after "gfx" for which the test is valid
    // '?' is wildcard char, empty string is default as valid on all
    char gpu_arch[4];

    rocblas_client_api api;

    // memory padding for testing write out of bounds
    uint32_t pad;

    uint32_t math_mode;

    // number of copies of arrays to allocate for context sensitive benchmarking
    uint64_t flush_batch_count;
    uint64_t flush_memory_size;

    // 16 bit
    uint16_t threads;
    uint16_t streams;

    // bytes
    uint8_t devices;

    int8_t norm_check;
    int8_t unit_check;
    int8_t res_check;
    int8_t timing;

    char transA;
    char transB;
    char side;
    char uplo;
    char diag;

    bool pointer_mode_host;
    bool pointer_mode_device;
    bool stochastic_rounding;
    bool c_noalias_d;
    bool outofplace;
    bool HMM; // xnack+
    bool graph_test;
    bool repeatability_check;

    int use_hipblaslt;

    bool cleanup;

    /*************************************************************************
     *                     End Of Arguments                                  *
     *************************************************************************/

    // we don't have a constructor as the python generated data is used for memory initializer for testing
    // thus this is for other use where we want defaults to match those specified in rocblas_common.yaml
    void init();

    // This should be called before use and may internally modify values to match rules
    // if the arguments don't make sense it will return false
    bool validate();

    // clang-format off

// Generic macro which operates over the list of arguments in order of declaration
#define FOR_EACH_ARGUMENT(OPER, SEP) \
    OPER(function) SEP               \
    OPER(name) SEP                   \
    OPER(category) SEP               \
    OPER(known_bug_platforms) SEP    \
    OPER(alpha) SEP                  \
    OPER(alphai) SEP                 \
    OPER(beta) SEP                   \
    OPER(betai) SEP                  \
    OPER(stride_a) SEP               \
    OPER(stride_b) SEP               \
    OPER(stride_c) SEP               \
    OPER(stride_d) SEP               \
    OPER(stride_x) SEP               \
    OPER(stride_y) SEP               \
    OPER(user_allocated_workspace) SEP \
    OPER(M) SEP                      \
    OPER(N) SEP                      \
    OPER(K) SEP                      \
    OPER(KL) SEP                     \
    OPER(KU) SEP                     \
    OPER(lda) SEP                    \
    OPER(ldb) SEP                    \
    OPER(ldc) SEP                    \
    OPER(ldd) SEP                    \
    OPER(incx) SEP                   \
    OPER(incy) SEP                   \
    OPER(batch_count) SEP            \
    OPER(scan) SEP                   \
    OPER(scan_2) SEP                 \
    OPER(iters) SEP                  \
    OPER(cold_iters) SEP             \
    OPER(algo) SEP                   \
    OPER(solution_index) SEP         \
    OPER(geam_ex_op) SEP             \
    OPER(flags) SEP                  \
    OPER(a_type) SEP                 \
    OPER(b_type) SEP                 \
    OPER(c_type) SEP                 \
    OPER(d_type) SEP                 \
    OPER(compute_type) SEP           \
    OPER(initialization) SEP         \
    OPER(atomics_mode) SEP           \
    OPER(os_flags) SEP               \
    OPER(gpu_arch) SEP               \
    OPER(api) SEP                    \
    OPER(pad) SEP                    \
    OPER(math_mode) SEP              \
    OPER(flush_batch_count) SEP      \
    OPER(flush_memory_size) SEP      \
    OPER(threads) SEP                \
    OPER(streams) SEP                \
    OPER(devices) SEP                \
    OPER(norm_check) SEP             \
    OPER(unit_check) SEP             \
    OPER(res_check) SEP              \
    OPER(timing) SEP                 \
    OPER(transA) SEP                 \
    OPER(transB) SEP                 \
    OPER(side) SEP                   \
    OPER(uplo) SEP                   \
    OPER(diag) SEP                   \
    OPER(pointer_mode_host) SEP      \
    OPER(pointer_mode_device) SEP    \
    OPER(stochastic_rounding) SEP    \
    OPER(c_noalias_d) SEP            \
    OPER(outofplace) SEP             \
    OPER(HMM) SEP                    \
    OPER(graph_test) SEP             \
    OPER(repeatability_check) SEP    \
    OPER(use_hipblaslt) SEP          \
    OPER(cleanup)
    // clang-format on

    // Validate input format.
    static void validate(std::istream& ifs);

    // Function to print Arguments out to stream in YAML format
    friend rocblas_internal_ostream& operator<<(rocblas_internal_ostream& str,
                                                const Arguments&          arg);

    // Google Tests uses this with std:ostream automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg);

    // Function to read Arguments data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg);

    template <typename T>
    bool alpha_isnan() const
    {
        return rocblas_isnan(alpha) || (rocblas_is_complex<T> && rocblas_isnan(alphai));
    }

    template <typename T>
    bool beta_isnan() const
    {
        return rocblas_isnan(beta) || (rocblas_is_complex<T> && rocblas_isnan(betai));
    }

private:
    // conversion helpers
    template <typename T>
    inline static T convert_alpha_beta(double r, double i)
    {
        return T(r);
    }

    template <>
    inline rocblas_half convert_alpha_beta(double r, double i)
    {
        // constructor with double silently converted to zero without cast to float
        return rocblas_half((float)r);
    }

    template <>
    inline rocblas_float_complex convert_alpha_beta(double r, double i)
    {
        return rocblas_float_complex(r, i);
    }

    template <>
    inline rocblas_double_complex convert_alpha_beta(double r, double i)
    {
        return rocblas_double_complex(r, i);
    }

public:
    // Convert (alpha, alphai) and (beta, betai) to a particular type
    // Return alpha, beta adjusted to 0 for when they are NaN
    template <typename T>
    T get_alpha() const
    {
        return alpha_isnan<T>() ? T(0) : convert_alpha_beta<T>(alpha, alphai);
    }

    template <typename T>
    T get_beta() const
    {
        return beta_isnan<T>() ? T(0) : convert_alpha_beta<T>(beta, betai);
    }
};

// We make sure that the Arguments struct is C-compatible
static_assert(std::is_standard_layout<Arguments>{},
              "Arguments is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<Arguments>{},
              "Arguments is not a trivial type, and thus is "
              "incompatible with C.");

// Arguments enumerators
// Create
//     enum rocblas_argument : int {e_M, e_N, e_K, e_KL, ... };
// There is an enum value for each case in FOR_EACH_ARGUMENT.
//
#define CREATE_ENUM(NAME) e_##NAME,
enum rocblas_argument : int
{
    FOR_EACH_ARGUMENT(CREATE_ENUM, )
};
#undef CREATE_ENUM

// ArgumentsHelper contains a templated lambda apply<> where there is a template
// specialization for each line in the CPP macro FOR_EACH_ARGUMENT. For example,
// the first lambda is:  apply<e_M> = [](auto&& func, const Arguments& arg, auto){func("M", arg.m);};
// This lambda can be used to print "M" and arg.m.
//
// alpha and beta are specialized separately, because they need to use get_alpha() or get_beta().
// To prevent multiple definitions of specializations for alpha and beta, the rocblas_argument
// enum for alpha and beta are changed to rocblas_argument(-1) and rocblas_argument(-2) during
// the FOR_EACH_ARGUMENT loop. Those out-of-range enum values are not used except here, and are
// only used so that the FOR_EACH_ARGUMENT loop can be used to loop over all of the arguments.

#if __cplusplus >= 201703L
// C++17
// ArgumentsHelper contains a templated lambda apply<> where there is a template
// specialization for each line in the CPP macro FOR_EACH_ARGUMENT. For example,
// the first lambda is:  apply<e_M> = [](auto&& func, const Arguments& arg, auto){func("M", arg.m)}
// This lambda can be used to print "M" and arg.m
namespace ArgumentsHelper
{
    template <rocblas_argument>
    inline constexpr auto apply = nullptr;

    // Macro defining specializations for specific arguments
    // e_alpha and e_beta get turned into negative sentinel value specializations
    // clang-format off
#define APPLY(NAME)                                                                         \
    template <>                                                                             \
    inline constexpr auto                                                     \
        apply<e_##NAME == e_alpha ? rocblas_argument(-1)                                    \
                                  : e_##NAME == e_beta ? rocblas_argument(-2) : e_##NAME> = \
            [](auto&& func, const Arguments& arg, auto) { func(#NAME, arg.NAME); }

    // Specialize apply for each Argument
    FOR_EACH_ARGUMENT(APPLY, ;);

    // Specialization for e_alpha
    template <>
    inline constexpr auto apply<e_alpha> =
        [](auto&& func, const Arguments& arg, auto T) {
            func("alpha", arg.get_alpha<decltype(T)>());
        };

    // Specialization for e_beta
    template <>
    inline constexpr auto apply<e_beta> =
        [](auto&& func, const Arguments& arg, auto T) {
            func("beta", arg.get_beta<decltype(T)>());
        };
};
    // clang-format on

#else

// C++14. TODO: Remove when C++17 is used
// clang-format off
namespace ArgumentsHelper
{
#define APPLY(NAME)                                             \
    template <>                                                 \
    struct apply<e_##NAME == e_alpha ? rocblas_argument(-1) :   \
                 e_##NAME == e_beta  ? rocblas_argument(-2) :   \
                 e_##NAME>                                      \
    {                                                           \
        auto operator()()                                       \
        {                                                       \
            return                                              \
                [](auto&& func, const Arguments& arg, auto)     \
                {                                               \
                    func(#NAME, arg.NAME);                      \
                };                                              \
        }                                                       \
    };

    template <rocblas_argument>
    struct apply
    {
    };

    // Go through every argument and define specializations
    FOR_EACH_ARGUMENT(APPLY, ;);

    // Specialization for e_alpha
    template <>
    struct apply<e_alpha>
    {
        auto operator()()
        {
            return
                [](auto&& func, const Arguments& arg, auto T)
                {
                    func("alpha", arg.get_alpha<decltype(T)>());
                };
        }
    };

    // Specialization for e_beta
    template <>
    struct apply<e_beta>
    {
        auto operator()()
        {
            return
                [](auto&& func, const Arguments& arg, auto T)
                {
                    func("beta", arg.get_beta<decltype(T)>());
                };
        }
    };
};
// clang-format on
#endif

#undef APPLY
