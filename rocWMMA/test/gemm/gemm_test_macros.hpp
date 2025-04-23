/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_GEMM_TEST_MACROS_HPP
#define ROCWMMA_GEMM_TEST_MACROS_HPP

#include "gemm_test.hpp"
#include "kernel_generator.hpp"

///
/// Test suite parameters definition
/// @params
/// test_params_name : name of the resulting class
/// common_base_params : base parameter class holding common symbols
/// kernel_generator_impl : kernel generator implementation class
/// ... (__VA_ARGS__) : a list of kernel parameters that will be combined to create a set of test kernels.
///
#define ROCWMMA_GENERATE_GEMM_GTEST_SUITE_PARAMS(                                                     \
    test_params_name, common_base_params, kernel_generator_impl, ...)                                 \
    struct test_params_name : public common_base_params                                               \
    {                                                                                                 \
        /* Use combinatorial logic to generate a set of kernel params from the input. */              \
        using KernelParams    = typename CombineLists<__VA_ARGS__>::Result;                           \
        using KernelGenerator = KernelGenerator<KernelParams, kernel_generator_impl>;                 \
                                                                                                      \
        /* Sanity check to make sure the generator produces kernels expected by the test interface */ \
        static_assert(std::is_same<typename kernel_generator_impl::ResultT,                           \
                                   typename common_base_params::KernelT>::value,                      \
                      "Kernels from this generator do not match testing interface");                  \
                                                                                                      \
        /* Generate the set of kernels to be tested */                                                \
        static inline typename KernelGenerator::ResultT kernels()                                     \
        {                                                                                             \
            return KernelGenerator::generate();                                                       \
        }                                                                                             \
    };

///
/// Test suite instantiation (gtest integration)
/// @params
/// test_suite_prefix: context describing the test suite (e.g. gemm_tests)
/// test_suite_name: name of the specific test suite (e.g. gemm_kernel_5_NN_Layout)
/// test_interface: base gtest interface class
/// test_invoke: name of the test function to invoke on the test suite
/// test_param_triage: triage of parameters delivered to tests (e.g macro to match test_interface with runtime params)
/// test_params: testing parameters used to generate the test suite
///
#define ROCWMMA_INSTANTIATE_GTEST_SUITE(test_suite_prefix, \
                                        test_suite_name,   \
                                        test_interface,    \
                                        test_invoke,       \
                                        test_param_triage, \
                                        test_params)       \
    class test_suite_name : public test_interface          \
    {                                                      \
    };                                                     \
                                                           \
    TEST_P(test_suite_name, test_invoke)                   \
    {                                                      \
        this->test_invoke();                               \
    }                                                      \
                                                           \
    INSTANTIATE_TEST_SUITE_P(test_suite_prefix, test_suite_name, test_param_triage(test_params));

///
/// Triage of test parameters, specific to GEMM gtests.
/// Uses GTest combinatorial function to instantiate all possible
/// combinations of given parameters from each context.
/// @params
/// test_params : the class generated by ROCWMMA_GENERATE_GEMM_GTEST_SUITE_PARAMS,
/// which fulfills the rocwmma::GemmTest interface.
///
#define ROCWMMA_GEMM_GTEST_PARAM_TRIAGE(test_params)                     \
    ::testing::Combine(::testing::ValuesIn(test_params::kernels()),      \
                       ::testing::ValuesIn(test_params::threadBlocks()), \
                       ::testing::ValuesIn(test_params::problemSizes()), \
                       ::testing::ValuesIn(test_params::alphas()),       \
                       ::testing::ValuesIn(test_params::betas()))

///
/// Specific to GEMM gtest interface of rocwmma::GemmTest
/// @params
/// test_suite_prefix = used as the general test context (e.g. gemm_kernel_tests)
/// test_suite_name = specific test context (e.g. gemm_my_kernel_NN_32x32_2x1)
/// test_params = the object generated by ROCWMMA_GENERATE_GEMM_GTEST_SUITE_PARAMS
/// Note: The rocwmma::GemmTest interface is paired here explicitly with the
/// ROCWMMA_GEMM_GTEST_PARAM_TRIAGE macro to ensure matching of gtest parameters.
/// Invokes the RunKernel() function in rocwmma::GemmTest object.
///
#define ROCWMMA_INSTANTIATE_GEMM_GTEST_SUITE(test_suite_prefix, test_suite_name, test_params) \
    ROCWMMA_INSTANTIATE_GTEST_SUITE(test_suite_prefix,                                        \
                                    test_suite_name,                                          \
                                    rocwmma::GemmTest,                                        \
                                    RunKernel,                                                \
                                    ROCWMMA_GEMM_GTEST_PARAM_TRIAGE,                          \
                                    test_params)

///
/// Specific to GEMM gtest interface of rocwmma::GemmTest
/// @params
/// test_suite_prefix = used as the general test context (e.g. gemm_kernel_tests)
/// test_suite_name = specific test context (e.g. gemm_my_kernel_NN_32x32_2x1)
/// test_params = the object generated by ROCWMMA_GENERATE_GEMM_GTEST_SUITE_PARAMS
/// Note: The rocwmma::GemmTest interface is paired here explicitly with the
/// ROCWMMA_GEMM_GTEST_PARAM_TRIAGE macro to ensure matching of gtest parameters.
/// Invokes the RunKernelWithoutWarmup() function in rocwmma::GemmTest object.
///
#define ROCWMMA_INSTANTIATE_GEMM_GTEST_SUITE_NO_WARMUP(              \
    test_suite_prefix, test_suite_name, test_params)                 \
    ROCWMMA_INSTANTIATE_GTEST_SUITE(test_suite_prefix,               \
                                    test_suite_name,                 \
                                    rocwmma::GemmTest,               \
                                    RunKernelWithoutWarmup,          \
                                    ROCWMMA_GEMM_GTEST_PARAM_TRIAGE, \
                                    test_params)

#endif // ROCWMMA_GEMM_TEST_MACROS_HPP
