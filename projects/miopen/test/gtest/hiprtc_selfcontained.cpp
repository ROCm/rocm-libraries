// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test: runtime-compiled MIOpen kernels must stay self-contained
// when the runtime compiler cannot reach a host C++ standard library (the
// AMD Windows ROCm wheels compile HIPRTC without one; kernels then failed
// with "'type_traits' file not found", see miopen_type_traits.hpp). An
// inline kernel using the STL-like parts added for that case must compile,
// launch and produce correct values while the host STL is unreachable.
// A unique per-run option keeps the options-keyed kernel binary cache from
// satisfying any compile below.

#include <gtest/gtest.h>

#include <miopen/handle.hpp>

#include "get_handle.hpp"

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

namespace {

// Isolation candidates, strongest first: full -nostdinc is required on
// msvc-triple clang, whose automatic MSVC STL detection bypasses
// -nostdinc++; -nostdinc++ keeps libc headers for toolchains whose
// preincluded HIP headers need them.
constexpr const char* kIsolationCandidates[] = {"-nostdinc", "-nostdinc++"};

const char* kTrivialKernel = "#ifndef MIOPEN_HIP_RUNTIME_COMPILE\n"
                             "#include <hip/hip_runtime.h>\n"
                             "#endif\n"
                             "extern \"C\" __global__ void gtest_rtc_trivial() {}\n";

// Fails to compile (via #error) exactly when any of the standard-library
// headers the self-contained fallbacks gate on is reachable.
const char* kStlProbeKernel = "#ifndef MIOPEN_HIP_RUNTIME_COMPILE\n"
                              "#include <hip/hip_runtime.h>\n"
                              "#endif\n"
                              "#if __has_include(<type_traits>) || __has_include(<utility>) || "
                              "__has_include(<limits>) || __has_include(<initializer_list>)\n"
                              "#error MIOPEN_GTEST_STL_REACHABLE\n"
                              "#endif\n"
                              "extern \"C\" __global__ void gtest_rtc_probe() {}\n";

// Compiles exactly when the ambient environment can use the host standard
// library. Inverted polarity versus kStlProbeKernel so that a uniformly
// broken probe cannot be mistaken for an STL-reachable host.
const char* kStlUseKernel = "#ifndef MIOPEN_HIP_RUNTIME_COMPILE\n"
                            "#include <hip/hip_runtime.h>\n"
                            "#endif\n"
                            "#include <type_traits>\n"
                            "static_assert(std::is_integral<int>::value, \"stl usable\");\n"
                            "extern \"C\" __global__ void gtest_rtc_stl_use() {}\n";

// Inline kernel exercising the self-contained fallbacks: type traits
// (static_asserts, conditional_t), perfect forwarding (std::forward) and
// braced-init-list construction (tensor_layout_t ctor taking
// std::initializer_list).
const char* kSelfContainedKernel =
    "#ifndef MIOPEN_HIP_RUNTIME_COMPILE\n"
    "#include <hip/hip_runtime.h>\n"
    "#endif\n"
    "#include \"miopen_type_traits.hpp\"\n"
    "#include \"miopen_utility.hpp\"\n"
    "#include \"tensor_view.hpp\"\n"
    "static_assert(std::is_same<std::conditional_t<true, uint32_t, float>, uint32_t>::value &&\n"
    "                  std::is_same<std::conditional_t<false, uint32_t, float>, float>::value,\n"
    "              \"conditional_t must select on B\");\n"
    "static_assert(std::is_trivially_copyable_v<float>, \"float must be trivially copyable\");\n"
    "template <typename T>\n"
    "__device__ uint64_t Forwarded(T&& value)\n"
    "{\n"
    "    return static_cast<uint64_t>(std::forward<T>(value));\n"
    "}\n"
    "extern \"C\" __global__ void gtest_rtc_selfcontained(uint64_t* out)\n"
    "{\n"
    "    tensor_layout_t<2> layout({3u, 5u});\n"
    "    uint64_t lvalue = 9;\n"
    "    out[0]          = layout.layout[0];\n"
    "    out[1]          = layout.layout[1];\n"
    "    out[2]          = Forwarded(lvalue);\n"
    "    out[3]          = Forwarded(11u);\n"
    "}\n";

// Generated at execution time (never at build time) so that rebuilds in the
// same build directory cannot reuse an earlier run's cache entries.
std::string NonceOption()
{
    static const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    static std::atomic<unsigned> seq{0};
    return "-DMIOPEN_GTEST_RTC_NONCE=" + std::to_string(stamp) + "_" + std::to_string(seq++);
}

// Compile `src` through the production runtime-compilation path. AddKernel
// compiles, loads the module and resolves the kernel symbol eagerly.
bool Compiles(miopen::Handle& h,
              const std::string& program,
              const std::string& kernel,
              const std::string& params,
              const std::string& src)
{
    try
    {
        auto invoke = h.AddKernel("gtest_rtc_regression",
                                  "",
                                  program,
                                  kernel,
                                  {1, 1, 1},
                                  {1, 1, 1},
                                  params + " " + NonceOption(),
                                  0,
                                  src);
        (void)invoke;
        return true;
    }
    catch(const std::exception&)
    {
        return false;
    }
}

// Launch the inline kernel and check the values it writes: the layout
// fields verify real initializer_list lowering, the forwarded values verify
// std::forward for an lvalue and an rvalue.
void RunSelfContainedKernel(miopen::Handle& h, const std::string& params)
{
    std::vector<uint64_t> out;
    try
    {
        auto out_dev = h.Write(std::vector<uint64_t>(4, 0));
        h.AddKernel("gtest_rtc_regression",
                    "",
                    "gtest_rtc_selfcontained.cpp",
                    "gtest_rtc_selfcontained",
                    {1, 1, 1},
                    {1, 1, 1},
                    params + " " + NonceOption(),
                    0,
                    kSelfContainedKernel)(out_dev.get());
        out = h.Read<uint64_t>(out_dev, 4);
    }
    catch(const std::exception&)
    {
        FAIL() << "self-contained kernel failed to compile or launch";
        return;
    }
    ASSERT_EQ(out.size(), 4u);
    EXPECT_EQ(out[0], 3u) << "initializer_list element 0";
    EXPECT_EQ(out[1], 5u) << "initializer_list element 1";
    EXPECT_EQ(out[2], 9u) << "forwarded lvalue";
    EXPECT_EQ(out[3], 11u) << "forwarded rvalue";
}

} // namespace

TEST(CPU_HipRtcSelfContained_NONE, RtCompilesInlineKernelWithoutHostStl)
{
    auto&& h = get_handle();

    RecordProperty("device", h.GetTargetProperties().Name());

    // P0: the toolchain must compile a trivial kernel under isolation.
    std::string isolation;
    for(const char* candidate : kIsolationCandidates)
    {
        if(Compiles(h, "gtest_rtc_trivial.cpp", "gtest_rtc_trivial", candidate, kTrivialKernel))
        {
            isolation = candidate;
            break;
        }
    }
    ASSERT_FALSE(isolation.empty()) << "toolchain broken under -nostdinc/-nostdinc++";
    RecordProperty("isolation", isolation);

    // P1: prove the isolation removes the host STL before any verdict.
    const bool stl_unreachable =
        Compiles(h, "gtest_rtc_stl_probe.cpp", "gtest_rtc_probe", isolation, kStlProbeKernel);

    // P2: classify the ambient environment. P1 failing while P2 compiles
    // means the runtime-compiler environment injects the STL even under
    // isolation (comgr on Linux): the no-STL condition is not testable
    // here, and the run reports itself as skipped, never as a pass.
    const bool ambient_stl_usable =
        Compiles(h, "gtest_rtc_stl_use.cpp", "gtest_rtc_stl_use", "", kStlUseKernel);

    if(!stl_unreachable)
    {
        ASSERT_TRUE(ambient_stl_usable) << "inconsistent probe results, refusing to classify";
        GTEST_SKIP() << "host STL still reachable under " << isolation
                     << ": no-STL condition NOT tested";
    }

    // The STL-like kernel headers must work with no host STL reachable.
    RunSelfContainedKernel(h, isolation);

    // With-STL control, where the ambient environment can use an STL.
    if(ambient_stl_usable)
    {
        RunSelfContainedKernel(h, "");
    }
}
