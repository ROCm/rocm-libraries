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

/// SWDEV-257056?focusedCommentId=6654244&page=com.atlassian.jira.plugin.system.issuetabpanels%3Acomment-tabpanel#comment-6654244
/// \todo Create dedicated ticket and rename macro.
#define WORKAROUND_SWDEV_257056_PCH_MISSING_MACROS 1

// https://gerrit-git.amd.com/c/compute/ec/clr/+/972441
// Issue #2600 is not fixed in 6.0 and still persists in 6.1 release candidates.
// We are expecting it to be fixed in 6.1 RC after week 4 in 2024.
#define WORKAROUND_ISSUE_2600                                                                    \
    ((HIP_PACKAGE_VERSION_FLAT >= 6000000000ULL && HIP_PACKAGE_VERSION_FLAT <= 6000999999ULL) || \
     (HIP_PACKAGE_VERSION_FLAT >= 6001000000ULL && HIP_PACKAGE_VERSION_FLAT <= 6001024049ULL))

// Temporarily disable warnings as errors for kernel builds to see real breaks with compiler changes
#define MIOPEN_WORKAROUND_COMPILER_CHANGE 1

#include <miopen/config.h>
#include <miopen/handle.hpp>
#include <miopen/execution_context.hpp>

#if WORKAROUND_SWDEV_257056_PCH_MISSING_MACROS
#include <miopen/hip_build_utils.hpp>
#endif

#include <vector>
#include <thread>

#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "test_parameter_name_generator.hpp"

namespace {

enum kernel_type_t
{
    miopenHIPKernelType,
    miopenOpenCLKernelType
};

using TestCase = NamedParameter<bool>;

static std::string Write2s(kernel_type_t kern_type)
{
    if(kern_type == miopenHIPKernelType)
    {
        return "#ifndef MIOPEN_HIP_RUNTIME_COMPILE\n"
               "#include <hip/hip_runtime.h>\n"
#if WORKAROUND_SWDEV_257056_PCH_MISSING_MACROS
               "#else\n"
               "#ifdef hipThreadIdx_x\n"
               "#undef hipThreadIdx_x\n"
               "#endif\n"
               "#define hipThreadIdx_x threadIdx.x\n"
               "\n"
               "#ifdef hipBlockDim_x\n"
               "#undef hipBlockDim_x\n"
               "#endif\n"
               "#define hipBlockDim_x blockDim.x\n"
               "\n"
               "#ifdef hipBlockIdx_x\n"
               "#undef hipBlockIdx_x\n"
               "#endif\n"
               "#define hipBlockIdx_x blockIdx.x\n"
#endif
               "#endif\n"
               "extern \"C\" {\n"
               "__global__ void write(int* data) {\n"
               "    int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;\n"
               "    data[num] *= 2;\n"
               "}\n"
               "}\n";
    }
    else if(kern_type == miopenOpenCLKernelType)
    {
        return "__kernel void write(__global int* data) { data[get_global_id(0)] *= 2; }\n";
    }
    else
    {
        MIOPEN_THROW("Unsupported kernel type");
    }
}

static void run2s(const miopen::Handle& h, std::size_t n, kernel_type_t kern_type)
{
    std::vector<int> data_in(n, 1);
    auto data_dev = h.Write(data_in);
    if(kern_type == miopenOpenCLKernelType)
    {
        h.AddKernel("NoAlgo",
                    "",
                    "test_ocl.cl",
                    "write",
                    {n, 1, 1},
                    {n, 1, 1},
                    "",
                    0,
                    Write2s(miopenOpenCLKernelType))(data_dev.get());
    }
    else if(kern_type == miopenHIPKernelType)
    {
        h.AddKernel("NoAlgo",
                    "",
                    "test_hip.cpp",
                    "write",
                    {n, 1, 1},
                    {n, 1, 1},
                    "",
                    0,
                    Write2s(miopenHIPKernelType))(data_dev.get());
    }
    else
    {
        MIOPEN_THROW("Unsupported kernel type");
    }
    std::fill(data_in.begin(), data_in.end(), 2);

    auto data_out = h.Read<int>(data_dev, n);
    EXPECT_EQ(data_out, data_in);
}

static void test_multithreads(kernel_type_t kern_type, const bool with_stream = false)
{
    auto&& h1 = get_handle();
    auto&& h2 = get_handle_with_stream(h1);
    std::thread([&] { run2s(with_stream ? h2 : h1, 16, kern_type); }).join();
    std::thread([&] { run2s(with_stream ? h2 : h1, 32, kern_type); }).join();
    std::thread([&] {
        std::thread([&] { run2s(with_stream ? h2 : h1, 64, kern_type); }).join();
    }).join();
    run2s(with_stream ? h2 : h1, 4, kern_type);
}

static std::string WriteError(kernel_type_t kern_type)
{
    if(kern_type == miopenOpenCLKernelType)
    {
        return "__kernel void write(__global int* data) { data[i] = 0; }\n";
    }
    else if(kern_type == miopenHIPKernelType)
    {
        return "#ifndef MIOPEN_HIP_RUNTIME_COMPILE\n"
               "#include <hip/hip_runtime.h>\n"
               "#endif\n"
               "extern \"C\" {\n"
               "__global__ void write(int* data) {\n"
               "    data[num] *= 2;\n"
               "}\n"
               "}\n";
    }
    else
    {
        MIOPEN_THROW("Unsupported kernel type");
    }
}

static void test_errors(kernel_type_t kern_type)
{
    auto&& h = get_handle();
    if(kern_type == miopenOpenCLKernelType)
    {
        EXPECT_ANY_THROW(h.AddKernel("NoAlgo",
                                     "",
                                     "error_ocl.cl",
                                     "write",
                                     {1, 1, 1},
                                     {1, 1, 1},
                                     "",
                                     0,
                                     WriteError(kern_type)));

        try
        {
            h.AddKernel("NoAlgo",
                        "",
                        "error_ocl.cl",
                        "write",
                        {1, 1, 1},
                        {1, 1, 1},
                        "",
                        0,
                        WriteError(kern_type));
        }
        catch(miopen::Exception& e)
        {
            EXPECT_FALSE(std::string(e.what()).empty());
        }
    }
    else if(kern_type == miopenHIPKernelType)
    {
        EXPECT_ANY_THROW(h.AddKernel("NoAlgo",
                                     "",
                                     "error_hip.cpp",
                                     "write",
                                     {1, 1, 1},
                                     {1, 1, 1},
                                     "",
                                     0,
                                     WriteError(miopenHIPKernelType)));

        try
        {
            h.AddKernel("NoAlgo",
                        "",
                        "error_hip.cpp",
                        "write",
                        {1, 1, 1},
                        {1, 1, 1},
                        "",
                        0,
                        WriteError(miopenHIPKernelType));
        }
        catch(miopen::Exception& e)
        {
            EXPECT_FALSE(std::string(e.what()).empty());
        }
    }
}

#if MIOPEN_BUILD_DEV && !WORKAROUND_ISSUE_2600 && !MIOPEN_WORKAROUND_COMPILER_CHANGE
static std::string WriteNop(kernel_type_t kern_type)
{
    if(kern_type == miopenOpenCLKernelType)
    {
        return "__kernel void write(__global int* data) {}\n";
    }
    else if(kern_type == miopenHIPKernelType)
    {
        return "#ifndef MIOPEN_HIP_RUNTIME_COMPILE\n"
               "#include <hip/hip_runtime.h>\n"
               "#endif\n"
               "extern \"C\" {\n"
               "__global__ void write(int* data) {\n"
               "}\n"
               "}\n";
    }
    else
    {
        MIOPEN_THROW("Unsupported kernel type");
    }
}
#endif

static void test_warnings([[maybe_unused]] kernel_type_t kern_type)
{
#if MIOPEN_BUILD_DEV && !WORKAROUND_ISSUE_2600 && !MIOPEN_WORKAROUND_COMPILER_CHANGE
    auto&& h = get_handle();
    if(kern_type == miopenOpenCLKernelType)
    {
        EXPECT_ANY_THROW(h.AddKernel(
            "NoAlgo", "", "nop_ocl.cl", "write", {1, 1, 1}, {1, 1, 1}, "", 0, WriteNop(kern_type)));
    }
    else if(kern_type == miopenHIPKernelType)
    {
        EXPECT_ANY_THROW(h.AddKernel("NoAlgo",
                                     "",
                                     "nop_hip.cpp",
                                     "write",
                                     {1, 1, 1},
                                     {1, 1, 1},
                                     "",
                                     0,
                                     WriteNop(kern_type)));
    }
#endif
}

static void test_arch_name()
{
    auto&& h = get_handle();

    const auto known_arch = {"gfx908",
                             "gfx90a",
                             "gfx906",
                             "gfx900",
                             "gfx942",
                             "gfx950",
                             "gfx803",
                             "gfx1030",
                             "gfx1031",
                             "gfx1100",
                             "gfx1101",
                             "gfx1102",
                             "gfx1103",
                             "gfx1150",
                             "gfx1151",
                             "gfx1152",
                             "gfx1153",
                             "gfx1200",
                             "gfx1201"};

    const auto this_arch = h.GetDeviceName();

    EXPECT_TRUE(std::any_of(
        known_arch.begin(), known_arch.end(), [&](std::string arch) { return arch == this_arch; }));
}

static inline auto GenCases() { return MakeNamedParameterValues<bool>("with_stream", false, true); }

static inline auto GetCases()
{
    static const auto cases = GenCases();
    return cases;
}

} // namespace

struct HandleTest : testing::TestWithParam<TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        with_stream = GetParam();
    }

    void RunHIP()
    {
        auto&& h = get_handle();

        if(h.GetDeviceName() != "gfx803" && miopen::IsHipKernelsEnabled())
        {
            test_multithreads(miopenHIPKernelType, with_stream);
            test_errors(miopenHIPKernelType);
            // Warnings currently don't work in OpenCL
#if !MIOPEN_BACKEND_OPENCL
            test_warnings(miopenHIPKernelType);
#endif
        }

        test_arch_name();
    }

    void RunCL()
    {
        test_multithreads(miopenOpenCLKernelType, with_stream);
        test_errors(miopenOpenCLKernelType);
        test_arch_name();
        // Warnings currently don't work in OpenCL
#if !MIOPEN_BACKEND_OPENCL
        test_warnings(miopenOpenCLKernelType);
#endif
    }

private:
    bool with_stream{};
};

struct TestNameGenerator
{
    std::string operator()(const auto& info)
    {
        const auto& with_stream = info.param;
        std::stringstream ss;
        std::string str;

        ss << "with_stream_" << std::boolalpha << with_stream() << "_test_id_" << info.index;

        str = ss.str();

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) -> char {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }
};

using CPU_Handle_NONE = HandleTest;

TEST_P(CPU_Handle_NONE, TestHIP) { this->RunHIP(); }

TEST_P(CPU_Handle_NONE, TestCL) { this->RunCL(); }

INSTANTIATE_TEST_SUITE_P(Smoke, CPU_Handle_NONE, GetCases(), TestNameGenerator{});
