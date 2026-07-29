/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

// Verifies that all vectorized math functions in miopen_math.hpp and the cast()
// function in vector_types.hpp compile successfully via HIPRTC for every
// registered vector width (1, 2, 4, 8). This catches missing vector-width
// support before it surfaces as a runtime HIPRTC_ERROR_COMPILATION in production.

#define WORKAROUND_SWDEV_257056_PCH_MISSING_MACROS 1
#define MIOPEN_WORKAROUND_COMPILER_CHANGE 1

#include <miopen/config.h>
#include <miopen/handle.hpp>
#include <miopen/execution_context.hpp>

#if WORKAROUND_SWDEV_257056_PCH_MISSING_MACROS
#include <miopen/hip_build_utils.hpp>
#endif

#include <string>
#include <sstream>

#include "get_handle.hpp"

namespace {

// Generates a HIP kernel source that instantiates all vectorized math functions
// and cast() for a given vector width. The kernel itself is a no-op — we only
// care that HIPRTC compilation succeeds.
std::string MakeVectorMathKernel(int vec_size)
{
    // The scalar type name and vector type expression for ext_vector_type
    // For vec_size == 1, we use plain float (no ext_vector_type).
    std::string vec_type;
    std::string vec_type_half;
    std::string vec_type_ushort;
    if(vec_size == 1)
    {
        vec_type        = "float";
        vec_type_half   = "_Float16";
        vec_type_ushort = "ushort";
    }
    else
    {
        vec_type = "float __attribute__((ext_vector_type(" + std::to_string(vec_size) + ")))";
        vec_type_half =
            "_Float16 __attribute__((ext_vector_type(" + std::to_string(vec_size) + ")))";
        vec_type_ushort =
            "ushort __attribute__((ext_vector_type(" + std::to_string(vec_size) + ")))";
    }

    std::ostringstream src;

    // HIPRTC preamble — kernel headers check this macro
    src << "#define MIOPEN_USE_FP32 1\n";
    src << "#define MIOPEN_USE_FP16 0\n";
    src << "#define MIOPEN_USE_BFP16 0\n";
    src << "#define MIOPEN_USE_FP8 0\n";
    src << "#define MIOPEN_USE_BFP8 0\n";
    src << "#define MIOPEN_USE_FPMIX 0\n";
    src << "#define MIOPEN_USE_BFPMIX 0\n";

    src << "#include \"miopen_math.hpp\"\n";
    src << "#include \"vector_types.hpp\"\n";

    // Kernel that instantiates all math functions for the given vector type
    src << "extern \"C\" {\n";
    src << "__global__ void test_vector_math_float_v" << vec_size << "() {\n";
    src << "    using VecType = " << vec_type << ";\n";
    src << "    VecType a{}, b{}, c{};\n";

    // Unary functions
    src << "    (void)miopen::exp(a);\n";
    src << "    (void)miopen::log(a);\n";
    src << "    (void)miopen::sqrt(a);\n";
    src << "    (void)miopen::rsqrt(a);\n";
    src << "    (void)miopen::tanh(a);\n";
    src << "    (void)miopen::fabs(a);\n";

    // Binary functions
    src << "    (void)miopen::fmax(a, b);\n";
    src << "    (void)miopen::fmin(a, b);\n";
    src << "    (void)miopen::pow(a, b);\n";

    // Ternary function
    src << "    (void)miopen::fma(a, b, c);\n";

    // cast: float vec -> float vec (identity cast, exercises the template)
    src << "    (void)miopen::cast<VecType>(a);\n";

    src << "}\n";

    // Also test half-precision vector math
    src << "__global__ void test_vector_math_half_v" << vec_size << "() {\n";
    src << "    using VecType = " << vec_type_half << ";\n";
    src << "    VecType a{}, b{}, c{};\n";
    src << "    (void)miopen::exp(a);\n";
    src << "    (void)miopen::log(a);\n";
    src << "    (void)miopen::sqrt(a);\n";
    src << "    (void)miopen::rsqrt(a);\n";
    src << "    (void)miopen::fabs(a);\n";
    src << "    (void)miopen::fmax(a, b);\n";
    src << "    (void)miopen::fmin(a, b);\n";
    src << "    (void)miopen::fma(a, b, c);\n";
    src << "    (void)miopen::cast<VecType>(a);\n";
    src << "}\n";

    // Also test bfloat16 (ushort) vector math
    src << "__global__ void test_vector_math_ushort_v" << vec_size << "() {\n";
    src << "    using VecType = " << vec_type_ushort << ";\n";
    src << "    VecType a{}, b{};\n";
    src << "    (void)miopen::exp(a);\n";
    src << "    (void)miopen::log(a);\n";
    src << "    (void)miopen::sqrt(a);\n";
    src << "    (void)miopen::rsqrt(a);\n";
    src << "    (void)miopen::fabs(a);\n";
    src << "    (void)miopen::fmax(a, b);\n";
    src << "    (void)miopen::fmin(a, b);\n";
    src << "    (void)miopen::fma(a, b, a);\n";
    src << "    (void)miopen::cast<VecType>(a);\n";
    src << "}\n";

    src << "} // extern \"C\"\n";

    return src.str();
}

} // namespace

struct GPU_VectorMathCompile_NONE : testing::TestWithParam<int>
{
};

TEST_P(GPU_VectorMathCompile_NONE, GPU_VectorMathCompiles_FP32)
{
    auto&& handle = get_handle();
    if(!miopen::IsHipKernelsEnabled())
    {
        GTEST_SKIP() << "HIP kernels are not enabled";
    }

    const int vec_size      = GetParam();
    const std::string src   = MakeVectorMathKernel(vec_size);
    const std::string fname = "test_vec_math_v" + std::to_string(vec_size) + ".cpp";

    // Trigger HIPRTC compilation for float kernels.
    // If any math function is missing support for this vector width,
    // AddKernel will throw with HIPRTC_ERROR_COMPILATION.
    EXPECT_NO_THROW(handle.AddKernel("NoAlgo",
                                     "",
                                     fname,
                                     "test_vector_math_float_v" + std::to_string(vec_size),
                                     {1, 1, 1},
                                     {1, 1, 1},
                                     "",
                                     0,
                                     src))
        << "HIPRTC compilation failed for float with VecSize=" << vec_size;
}

TEST_P(GPU_VectorMathCompile_NONE, GPU_VectorMathCompiles_FP16)
{
    auto&& handle = get_handle();
    if(!miopen::IsHipKernelsEnabled())
    {
        GTEST_SKIP() << "HIP kernels are not enabled";
    }

    const int vec_size      = GetParam();
    const std::string src   = MakeVectorMathKernel(vec_size);
    const std::string fname = "test_vec_math_half_v" + std::to_string(vec_size) + ".cpp";

    EXPECT_NO_THROW(handle.AddKernel("NoAlgo",
                                     "",
                                     fname,
                                     "test_vector_math_half_v" + std::to_string(vec_size),
                                     {1, 1, 1},
                                     {1, 1, 1},
                                     "",
                                     0,
                                     src))
        << "HIPRTC compilation failed for _Float16 with VecSize=" << vec_size;
}

TEST_P(GPU_VectorMathCompile_NONE, GPU_VectorMathCompiles_BFP16)
{
    auto&& handle = get_handle();
    if(!miopen::IsHipKernelsEnabled())
    {
        GTEST_SKIP() << "HIP kernels are not enabled";
    }

    const int vec_size      = GetParam();
    const std::string src   = MakeVectorMathKernel(vec_size);
    const std::string fname = "test_vec_math_ushort_v" + std::to_string(vec_size) + ".cpp";

    EXPECT_NO_THROW(handle.AddKernel("NoAlgo",
                                     "",
                                     fname,
                                     "test_vector_math_ushort_v" + std::to_string(vec_size),
                                     {1, 1, 1},
                                     {1, 1, 1},
                                     "",
                                     0,
                                     src))
        << "HIPRTC compilation failed for ushort (bf16) with VecSize=" << vec_size;
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_VectorMathCompile_NONE,
                         testing::Values(1, 2, 4, 8),
                         [](const auto& info) { return "VecSize_" + std::to_string(info.param); });

// Negative test: verify that an unsupported vector width (e.g. 3) is rejected
// at HIPRTC compile time by the static_assert allowlist in each math function.
// This confirms the guard works as intended through the JIT path.
static std::string MakeUnsupportedVecKernel()
{
    std::ostringstream src;

    src << "#define MIOPEN_USE_FP32 1\n";
    src << "#define MIOPEN_USE_FP16 0\n";
    src << "#define MIOPEN_USE_BFP16 0\n";
    src << "#define MIOPEN_USE_FP8 0\n";
    src << "#define MIOPEN_USE_BFP8 0\n";
    src << "#define MIOPEN_USE_FPMIX 0\n";
    src << "#define MIOPEN_USE_BFPMIX 0\n";

    src << "#include \"vector_types.hpp\"\n";
    src << "#include \"miopen_math.hpp\"\n";

    // Register float3 as a valid mapped_vector_type — this succeeds,
    // but the math functions should reject VecSize==3 via static_assert.
    src << "namespace miopen {\n";
    src << "template <>\n";
    src << "struct mapped_vector_type<float, 3>\n";
    src << "{\n";
    src << "    using type = float __attribute__((ext_vector_type(3)));\n";
    src << "};\n";
    src << "template <>\n";
    src << "struct mapped_vector_info<float __attribute__((ext_vector_type(3)))>\n";
    src << "{\n";
    src << "    using UnderlyingType         = float;\n";
    src << "    static constexpr size_t size = 3;\n";
    src << "};\n";
    src << "} // namespace miopen\n";

    src << "extern \"C\" {\n";
    src << "__global__ void test_unsupported_vec3() {\n";
    src << "    using Vec3 = float __attribute__((ext_vector_type(3)));\n";
    src << "    Vec3 a{}, b{};\n";
    src << "    (void)miopen::exp(a);\n";
    src << "}\n";
    src << "} // extern \"C\"\n";

    return src.str();
}

TEST(GPU_VectorMathCompileNegative_FP32, GPU_UnsupportedVecSize_FailsCompilation)
{
    auto&& handle = get_handle();
    if(!miopen::IsHipKernelsEnabled())
    {
        GTEST_SKIP() << "HIP kernels are not enabled";
    }

    const std::string src = MakeUnsupportedVecKernel();

    // HIPRTC compilation must fail because VecSize==3 is not in the
    // static_assert allowlist (1, 2, 4, 8).
    EXPECT_ANY_THROW(handle.AddKernel("NoAlgo",
                                      "",
                                      "test_unsupported_vec3.cpp",
                                      "test_unsupported_vec3",
                                      {1, 1, 1},
                                      {1, 1, 1},
                                      "",
                                      0,
                                      src))
        << "Expected HIPRTC compilation to fail for unsupported VecSize=3, "
           "but it succeeded — the static_assert guard may be missing";
}
