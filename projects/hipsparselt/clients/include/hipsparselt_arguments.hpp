/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

#pragma once

#include "hipsparselt_datatype2string.hpp"
#include "hipsparselt_math.hpp"
#include "hipsparselt_ostream.hpp"
#include <cstddef>
#include <hipsparselt/hipsparselt.h>
#include <istream>
#include <map>
#include <ostream>
#include <tuple>

// Predeclare enumerator
enum hipsparselt_argument : int;

/***************************************************************************
 *! \brief Class used to parse command arguments in both client & gtest    *
 * WARNING: If this data is changed, then hipsparselt_common.yaml must also be *
 * changed.                                                                *
 ***************************************************************************/
struct Arguments
{
    /*************************************************************************
     *                    Beginning Of Arguments                             *
     *************************************************************************/

    char function[64];
    char name[64];
    char category[64];
    char known_bug_platforms[64];

    // 32bit
    float alpha;
    float beta;

    int64_t stride_a; //  stride_a > transA == 'N' ? lda * K : lda * M
    int64_t stride_b; //  stride_b > transB == 'N' ? ldb * N : ldb * K
    int64_t stride_c; //  stride_c > ldc * N
    int64_t stride_d; //  stride_d > ldd * N

    size_t user_allocated_workspace;

    int64_t M;
    int64_t N;
    int64_t K;

    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldd;

    int32_t batch_count;

    int32_t iters;
    int32_t cold_iters;

    uint32_t algo;
    int32_t  solution_index;
    uint32_t prune_algo;

    hipDataType              a_type;
    hipDataType              b_type;
    hipDataType              c_type;
    hipDataType              d_type;
    hipsparseLtComputetype_t compute_type;

    hipsparselt_initialization initialization;

    // the gpu arch string after "gfx" for which the test is valid
    // '?' is wildcard char, empty string is default as valid on all
    char gpu_arch[16];

    // memory padding for testing write out of bounds
    uint32_t pad;

    // 16 bit
    uint16_t threads;
    uint16_t streams;

    // bytes
    uint8_t devices;

    int8_t norm_check;
    int8_t unit_check;
    bool norm_check_assert;

    int8_t timing;

    char transA;
    char transB;

    hipsparselt_activation_type activation_type;
    float                       activation_arg1; // threshold when activation type is relu
    float                       activation_arg2; // upperbound when activation type is relu

    bool        bias_vector;
    int64_t     bias_stride;
    hipDataType bias_type;

    bool c_noalias_d;
    bool HMM;

    bool    search;
    int32_t search_iters;

    bool sparse_b;
    int  func_version;

    bool alpha_vector_scaling;

    char orderA;
    char orderB;
    char orderC;
    char orderD;

    bool inEqualOut;
    int logging;
    /*************************************************************************
     *                     End Of Arguments                                  *
     *************************************************************************/

    // we don't have a constructor as the python generated data is used for memory initializer for testing
    // thus this is for other use where we want defaults to match those specified in hipsparselt_common.yaml
    void init();

    // clang-format off

// Generic macro which operates over the list of arguments in order of declaration
#define FOR_EACH_ARGUMENT(OPER, SEP) \
    OPER(function) SEP               \
    OPER(name) SEP                   \
    OPER(category) SEP               \
    OPER(known_bug_platforms) SEP    \
    OPER(alpha) SEP                  \
    OPER(beta) SEP                   \
    OPER(stride_a) SEP               \
    OPER(stride_b) SEP               \
    OPER(stride_c) SEP               \
    OPER(stride_d) SEP               \
    OPER(user_allocated_workspace) SEP \
    OPER(M) SEP                      \
    OPER(N) SEP                      \
    OPER(K) SEP                      \
    OPER(lda) SEP                    \
    OPER(ldb) SEP                    \
    OPER(ldc) SEP                    \
    OPER(ldd) SEP                    \
    OPER(batch_count) SEP            \
    OPER(iters) SEP                  \
    OPER(cold_iters) SEP             \
    OPER(algo) SEP                   \
    OPER(solution_index) SEP         \
    OPER(prune_algo) SEP             \
    OPER(a_type) SEP                 \
    OPER(b_type) SEP                 \
    OPER(c_type) SEP                 \
    OPER(d_type) SEP                 \
    OPER(compute_type) SEP           \
    OPER(initialization) SEP         \
    OPER(gpu_arch) SEP               \
    OPER(pad) SEP                    \
    OPER(threads) SEP                \
    OPER(streams) SEP                \
    OPER(devices) SEP                \
    OPER(norm_check) SEP             \
    OPER(unit_check) SEP             \
    OPER(norm_check_assert) SEP      \
    OPER(timing) SEP                 \
    OPER(transA) SEP                 \
    OPER(transB) SEP                 \
    OPER(activation_type) SEP        \
    OPER(activation_arg1) SEP        \
    OPER(activation_arg2) SEP        \
    OPER(bias_vector) SEP            \
    OPER(bias_stride) SEP            \
    OPER(bias_type) SEP              \
    OPER(c_noalias_d) SEP            \
    OPER(HMM) SEP                    \
    OPER(search) SEP                 \
    OPER(search_iters) SEP           \
    OPER(sparse_b) SEP               \
    OPER(func_version) SEP           \
    OPER(alpha_vector_scaling) SEP   \
    OPER(orderA) SEP                 \
    OPER(orderB) SEP                 \
    OPER(orderC) SEP                 \
    OPER(orderD) SEP                 \
    OPER(inEqualOut) SEP             \
    OPER(logging) SEP
    // clang-format on

    // Validate input format.
    static void validate(std::istream& ifs);

    // Function to print Arguments out to stream in YAML format
    friend hipsparselt_internal_ostream& operator<<(hipsparselt_internal_ostream& str,
                                                    const Arguments&              arg);

    // Google Tests uses this with std:ostream automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg);

    // Function to read Arguments data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg);

#ifdef WIN32
    // Clang specific code
    template <typename T>
    friend hipsparselt_internal_ostream& operator<<(hipsparselt_internal_ostream& os,
                                                    std::pair<char const*, T>     p);

    friend hipsparselt_internal_ostream& operator<<(hipsparselt_internal_ostream&       os,
                                                    std::pair<char const*, hipDataType> p);

    friend hipsparselt_internal_ostream&
        operator<<(hipsparselt_internal_ostream&                      os,
                   std::pair<char const*, hipsparselt_initialization> p);

    friend hipsparselt_internal_ostream& operator<<(hipsparselt_internal_ostream& os,
                                                    std::pair<char const*, bool>  p);
// End of Clang specific code
#endif

    // Convert (alpha, alphai) and (beta, betai) to a particular type
    // Return alpha, beta adjusted to 0 for when they are NaN
    template <typename T>
    T get_alpha() const
    {
        return alpha_isnan<T>() ? T(0) : convert_alpha_beta<T>(alpha);
    }

    template <typename T>
    T get_beta() const
    {
        return beta_isnan<T>() ? T(0) : convert_alpha_beta<T>(beta);
    }

    template <typename T>
    bool alpha_isnan() const
    {
        return hipsparselt_isnan(alpha);
    }

    template <typename T>
    bool beta_isnan() const
    {
        return hipsparselt_isnan(beta);
    }

private:
    template <typename T, typename U>
    static T convert_alpha_beta(U r)
    {
        return T(r);
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
//     enum hipsparselt_argument : int {e_M, e_N, e_K, e_KL, ... };
// There is an enum value for each case in FOR_EACH_ARGUMENT.
//
#define CREATE_ENUM(NAME) e_##NAME,
enum hipsparselt_argument : int
{
    FOR_EACH_ARGUMENT(CREATE_ENUM, )
};
#undef CREATE_ENUM

#if __clang__
#define HIPSPARSELT_CLANG_STATIC static
#else
#define HIPSPARSELT_CLANG_STATIC
#endif

// ArgumentsHelper contains a templated lambda apply<> where there is a template
// specialization for each line in the CPP macro FOR_EACH_ARGUMENT. For example,
// the first lambda is:  apply<e_M> = [](auto&& func, const Arguments& arg, auto){func("M", arg.m);};
// This lambda can be used to print "M" and arg.m.
//
// alpha and beta are specialized separately, because they need to use get_alpha() or get_beta().
// To prevent multiple definitions of specializations for alpha and beta, the hipsparselt_argument
// enum for alpha and beta are changed to hipsparselt_argument(-1) and hipsparselt_argument(-2) during
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
    template <hipsparselt_argument>
    static constexpr auto apply = nullptr;

    // Macro defining specializations for specific arguments
    // e_alpha and e_beta get turned into negative sentinel value specializations
    // clang-format off
#define APPLY(NAME)                                                                         \
    template <>                                                                             \
    HIPSPARSELT_CLANG_STATIC constexpr auto                                                     \
        apply<e_##NAME == e_alpha ? hipsparselt_argument(-1)                                    \
                                  : e_##NAME == e_beta ? hipsparselt_argument(-2) : e_##NAME> = \
            [](auto&& func, const Arguments& arg, auto) { func(#NAME, arg.NAME); }

    // Specialize apply for each Argument
    FOR_EACH_ARGUMENT(APPLY, ;);

    // Specialization for e_alpha
    template <>
    HIPSPARSELT_CLANG_STATIC constexpr auto apply<e_alpha> =
        [](auto&& func, const Arguments& arg, auto T) {
            func("alpha", arg.get_alpha<decltype(T)>());
        };

    // Specialization for e_beta
    template <>
    HIPSPARSELT_CLANG_STATIC constexpr auto apply<e_beta> =
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
    struct apply<e_##NAME == e_alpha ? hipsparselt_argument(-1) :   \
                 e_##NAME == e_beta  ? hipsparselt_argument(-2) :   \
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

    template <hipsparselt_argument>
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
