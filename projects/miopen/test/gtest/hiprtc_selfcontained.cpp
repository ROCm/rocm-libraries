// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test: runtime-compiled MIOpen kernels must stay self-contained
// when the host C++ standard library is unusable by the runtime compiler:
// either unreachable (the AMD Windows ROCm wheels compile HIPRTC without
// one; kernels then failed with "'type_traits' file not found", see
// miopen_type_traits.hpp) or reachable but unparseable by the device
// compiler (libstdc++ 12 headers mixed with GCC 13's bits/c++config.h). An
// inline kernel using the STL-like parts added for that case must compile,
// launch and produce correct values without the host STL.
// A unique per-run option keeps the options-keyed kernel binary cache from
// satisfying any compile below.

#include <gtest/gtest.h>

#include <miopen/config.h>
#include <miopen/handle.hpp>

#include "get_handle.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
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

// Fails to compile (via #error) exactly when any host standard-library
// header is reachable.
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

// Real host STL headers placed before the kernel headers.
const std::string kPreincludeHostStl = "#include <type_traits>\n"
                                       "#include <utility>\n"
                                       "#include <initializer_list>\n";

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
void RunSelfContainedKernel(miopen::Handle& h,
                            const std::string& params,
                            const std::string& preamble = "")
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
                    preamble + kSelfContainedKernel)(out_dev.get());
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

// A directory whose <type_traits>, <utility> and <initializer_list> fail to
// compile, standing in for a host standard library the device compiler cannot
// parse (libstdc++ 12 headers mixed with GCC 13's bits/c++config.h failed
// inside <type_traits>). Searched first via -isystem, so any kernel
// header that still reaches the host STL fails the compile.
class BrokenHostStlDir
{
public:
    BrokenHostStlDir() : path_(MakePath())
    {
        std::filesystem::create_directories(path_);
        for(const char* header : {"type_traits", "utility", "initializer_list"})
            std::ofstream(path_ / header)
                << "#error \"runtime-compiled kernel reached the host STL\"\n";
    }
    ~BrokenHostStlDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    BrokenHostStlDir(const BrokenHostStlDir&)            = delete;
    BrokenHostStlDir& operator=(const BrokenHostStlDir&) = delete;

    // Joined form: MIOpen glues a separate "-isystem <dir>" pair into one
    // argument, which HIPRTC silently ignores.
    std::string IsystemOption() const { return "-isystem" + path_.string(); }

    // Build options are split on whitespace without quoting, so a path with a
    // space cannot be passed to the compiler.
    bool IsUsable() const { return path_.string().find(' ') == std::string::npos; }

private:
    static std::filesystem::path MakePath()
    {
        const std::string nonce = NonceOption();
        return std::filesystem::temp_directory_path() /
               ("miopen_gtest_broken_stl_" + nonce.substr(nonce.rfind('=') + 1));
    }

    std::filesystem::path path_;
};

} // namespace

using CPU_HipRtcSelfContained_NONE = testing::TestWithParam<int>;

// Regression test: the kernel headers must not depend on the
// host STL being parseable, even where one is reachable (so this runs on Linux
// comgr too, unlike the -nostdinc test below).
TEST_P(CPU_HipRtcSelfContained_NONE, RtIgnoresUnusableHostStl)
{
    auto&& h = get_handle();
    RecordProperty("device", h.GetTargetProperties().Name());

#if !MIOPEN_USE_HIPRTC
    // Without HIPRTC the kernel is built offline with the host STL by design.
    GTEST_SKIP() << "runtime compilation is not HIPRTC";
#endif

    const BrokenHostStlDir broken_stl;
    if(!broken_stl.IsUsable())
        GTEST_SKIP() << "temporary directory path contains a space";
    RunSelfContainedKernel(h, broken_stl.IsystemOption());
}

// A real host STL header that was included before the kernel headers must be
// respected: defining the built-in fallbacks as well would redefine std types.
TEST_P(CPU_HipRtcSelfContained_NONE, RtRespectsPreincludedHostStl)
{
    auto&& h = get_handle();
    RecordProperty("device", h.GetTargetProperties().Name());

#if !MIOPEN_USE_HIPRTC
    GTEST_SKIP() << "runtime compilation is not HIPRTC";
#endif

    if(!Compiles(h, "gtest_rtc_stl_use.cpp", "gtest_rtc_stl_use", "", kStlUseKernel))
        GTEST_SKIP() << "host STL is not usable by the runtime compiler: nothing to pre-include";

    RunSelfContainedKernel(h, "", kPreincludeHostStl);

    // A single header that transitively declares only part of what the kernel headers provide
    // (e.g. <string> declares std::forward without defining the <utility> include guard).
    for(const char* header : {"<string>", "<type_traits>"})
    {
        SCOPED_TRACE(header);
        RunSelfContainedKernel(h, "", std::string("#include ") + header + "\n");
    }
}

TEST_P(CPU_HipRtcSelfContained_NONE, RtCompilesInlineKernelWithoutHostStl)
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

INSTANTIATE_TEST_SUITE_P(Quick, CPU_HipRtcSelfContained_NONE, testing::Values(0));
