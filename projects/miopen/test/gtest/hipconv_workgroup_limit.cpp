// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/hipoc_kernel.hpp>
#include <gtest/gtest.h>

#if defined(MIOPEN_USE_HIPCONV) && MIOPEN_USE_HIPCONV

// hipconv is vendored and exports nothing from libMIOpen.so, so these tests link the static
// libhipconv into the test binary and reach its internal headers through a per-source include
// path (see test/gtest/CMakeLists.txt).
#include "algorithm.h"
#include "arch_registry.h"
#include "conv_kernel.h"
#include "hip_util.h"
#include "hipconv/hipconv.hpp"
#include "launch_params.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kItemsPerYWorkgroup = 8;
constexpr std::size_t kItemsPerZWorkgroup = 4;
// Largest X workgroup count a ParamsFor() layer can ask for.
constexpr std::size_t kHugeX = 2'000'000'000;
// Y and Z workgroup counts far above any architecture's limit that ParamsFor() can still express.
constexpr std::size_t kHugeYZ = 200'000'000;

int g_launches = 0;

void CountLaunch(const LaunchParams&,
                 const hipconv::Device&,
                 const hipconv::ConvParams&,
                 const void*,
                 const void*,
                 void*,
                 void*,
                 hipStream_t)
{
    ++g_launches;
}

bool AnyParams(const hipconv::ConvParams&) { return true; }

std::uint32_t CeilDiv(int value, std::size_t divisor)
{
    return static_cast<std::uint32_t>((static_cast<std::size_t>(value) + divisor - 1) / divisor);
}

// A kernel that accepts every layer and launches one X workgroup per input column, one Y
// workgroup per kItemsPerYWorkgroup batch items and one Z workgroup per kItemsPerZWorkgroup
// output channels, the last of each rounded up. It is the real ConvKernel underneath: grid_fits(),
// launch(), and the hipconv filters see it exactly as they see a shipped kernel.
class FakeKernel final : public hipconv::ConvKernel
{
public:
    FakeKernel() : ConvKernel(&CountLaunch) {}

    std::string_view name() const override { return "fake"; }

    hipconv::Algorithm algorithm() const override { return hipconv::Algorithm::Direct; }

    bool is_applicable(const hipconv::Device&, const hipconv::ConvParams&) const override
    {
        return true;
    }

    bool is_valid_config(const hipconv::Device&, const hipconv::ConvParams&) const override
    {
        return true;
    }

    LaunchParams get_launch_params(const hipconv::Device&,
                                   const hipconv::ConvParams& par) const override
    {
        LaunchParams lp;
        lp.grid       = dim3(static_cast<std::uint32_t>(par.w),
                       CeilDiv(par.n, kItemsPerYWorkgroup),
                       CeilDiv(par.k, kItemsPerZWorkgroup));
        lp.block_size = dim3(256);
        return lp;
    }

    float get_weighted_throughput_index(const hipconv::Device&,
                                        const hipconv::ConvParams&) const override
    {
        return 1.0f;
    }
};

// The layer FakeKernel launches \p x, \p y and \p z workgroups for. With \p partial the last Y and
// Z workgroups hold a single item each.
hipconv::ConvParams ParamsFor(std::size_t x, std::size_t y, std::size_t z, bool partial = false)
{
    hipconv::ConvParams par{};
    par.w                     = static_cast<int>(x);
    const std::size_t y_slack = partial ? kItemsPerYWorkgroup - 1 : std::size_t{0};
    const std::size_t z_slack = partial ? kItemsPerZWorkgroup - 1 : std::size_t{0};
    par.n                     = static_cast<int>(y * kItemsPerYWorkgroup - y_slack);
    par.k                     = static_cast<int>(z * kItemsPerZWorkgroup - z_slack);
    return par;
}

// FakeKernel as the only kernel of an architecture named \p arch_name, wired up the way the
// generated registry wires a shipped one. The name need not be in the registry: only the Y/Z
// workgroup-count limit reads it.
struct Rig
{
    explicit Rig(std::string_view arch_name)
        : kernels{&kernel},
          group{kernels, nullptr},
          groups{&group},
          algorithm(&AnyParams, groups),
          algorithms{hipconv::ArchAlgorithm{hipconv::Algorithm::Direct, &algorithm}},
          arch{arch_name, algorithms},
          device{&arch, 80}
    {
    }

    Rig(const Rig&)            = delete;
    Rig& operator=(const Rig&) = delete;

    FakeKernel kernel;
    std::array<hipconv::ConvKernel*, 1> kernels;
    hipconv::ConvKernelGroup group;
    std::array<const hipconv::ConvKernelGroup*, 1> groups;
    hipconv::ConvAlgorithm algorithm;
    std::array<hipconv::ArchAlgorithm, 1> algorithms;
    hipconv::ArchEntry arch;
    hipconv::Device device;
};

std::size_t HipconvLimit(std::string_view arch_name)
{
    const hipconv::ArchEntry arch{arch_name, {}};
    return hipconv::max_workgroups_yz(hipconv::Device{&arch, 1});
}

// Whether grid_fits(), the algorithm's filter, the public listings and is_applicable() all take
// \p par on \p arch_name when \p accepted, and all reject it otherwise.
void ExpectFilter(std::string_view arch_name, const hipconv::ConvParams& par, bool accepted)
{
    SCOPED_TRACE(std::string{arch_name} + " w=" + std::to_string(par.w) +
                 " n=" + std::to_string(par.n) + " k=" + std::to_string(par.k));

    Rig rig{arch_name};
    const std::size_t listed = accepted ? 1 : 0;

    EXPECT_EQ(rig.kernel.grid_fits(rig.device, par), accepted);
    EXPECT_EQ(rig.algorithm.get_valid_configs(rig.device, par, hipconv::ALL_RANKED_CONFIGS).size(),
              listed);
    EXPECT_EQ(hipconv::get_valid_configs(
                  rig.device, par, hipconv::Algorithm::Direct, hipconv::ALL_RANKED_CONFIGS)
                  .size(),
              listed);
    EXPECT_EQ(hipconv::get_valid_configs(rig.device, par, hipconv::ALL_RANKED_CONFIGS).size(),
              listed);
    EXPECT_EQ(hipconv::is_applicable(&rig.kernel, rig.device, par), accepted);
    EXPECT_EQ(hipconv::ConvLaunch::make(&rig.kernel, rig.device, par).has_value(), accepted);
}

struct LimitedArch
{
    std::string_view name;
    std::size_t limit;
};

// Architectures with a Y/Z workgroup-count limit: gfx1200/gfx1201 wrap a count of 65537 or more,
// gfx1250 a count of 65536 or more.
constexpr LimitedArch limited_archs[] = {
    {"gfx1200", 65536}, {"gfx1201", 65536}, {"gfx1250", 65535}, {"gfx1250-strict", 65535}};

constexpr std::string_view unlimited_archs[] = {"gfx942", "gfx950", "gfx1100", "gfx1151", "gfx90a"};

// The HipError code launch() throws for \p f, or hipSuccess if it does not throw one.
template <class F>
hipError_t ThrownHipError(F&& f)
{
    try
    {
        f();
    }
    catch(const HipError& e)
    {
        return e.code;
    }
    return hipSuccess;
}

} // namespace

TEST(CPU_hipconv_workgroup_limit_NONE, arch_limits)
{
    EXPECT_EQ(HipconvLimit("gfx1200"), 65536u);
    EXPECT_EQ(HipconvLimit("gfx1201"), 65536u);
    EXPECT_EQ(HipconvLimit("gfx1250"), 65535u);
    // The name gcnArchName reports for the gfx1250 silicon revision, and feature suffixes.
    EXPECT_EQ(HipconvLimit("gfx1250-strict"), 65535u);
    EXPECT_EQ(HipconvLimit("gfx1201:sramecc+:xnack-"), 65536u);
    for(const auto arch : unlimited_archs)
        EXPECT_EQ(HipconvLimit(arch), hipconv::no_workgroup_limit) << arch;
}

// The limit follows the registry's own architecture entries, not just a hand-built one.
TEST(CPU_hipconv_workgroup_limit_NONE, registry_arch_limits)
{
    const struct
    {
        std::string_view name;
        std::size_t limit;
    } expected[] = {{"gfx942", hipconv::no_workgroup_limit},
                    {"gfx950", hipconv::no_workgroup_limit},
                    {"gfx1250", 65535},
                    {"gfx1250-strict", 65535}};

    const FakeKernel kernel;
    for(const auto& e : expected)
    {
        const auto arch = hipconv::resolve_arch(e.name);
        ASSERT_TRUE(arch.has_value()) << e.name;
        const hipconv::Device device{*arch, 80};
        EXPECT_EQ(hipconv::max_workgroups_yz(device), e.limit) << e.name;
        // 65536 Y workgroups fit exactly where the limit is 65536 or absent.
        EXPECT_EQ(kernel.grid_fits(device, ParamsFor(1, 65536, 1)), e.limit >= 65536u) << e.name;
    }
}

// hipconv has no MIOpen headers, so its limit table is a copy of miopen::GetMaxWorkgroupsYZ().
// This keeps the two equal for every architecture hipconv knows and for every name that either
// table special-cases.
TEST(CPU_hipconv_workgroup_limit_NONE, limits_agree_with_miopen)
{
    std::vector<std::string> names;
    for(const auto name : hipconv::arch_names())
        names.emplace_back(name);
    ASSERT_FALSE(names.empty());
    for(const char* name : {"gfx1200",
                            "gfx1201",
                            "gfx1250",
                            "gfx1250-strict",
                            "gfx1201:sramecc+:xnack-",
                            "gfx1250:xnack-",
                            "gfx942:sramecc+:xnack-",
                            "gfx1030",
                            "gfx1100",
                            "gfx1151"})
        names.emplace_back(name);
    for(const auto name : unlimited_archs)
        names.emplace_back(name);

    for(const auto& name : names)
        EXPECT_EQ(HipconvLimit(name), miopen::GetMaxWorkgroupsYZ(name)) << name;
}

TEST(CPU_hipconv_workgroup_limit_NONE, grid_fits_boundaries)
{
    // The limit's neighbours, in absolute counts, along Y and along Z.
    const struct
    {
        std::string_view arch;
        std::size_t workgroups;
        bool fits;
    } boundaries[] = {{"gfx1200", 65535, true},
                      {"gfx1200", 65536, true},
                      {"gfx1200", 65537, false},
                      {"gfx1201", 65535, true},
                      {"gfx1201", 65536, true},
                      {"gfx1201", 65537, false},
                      {"gfx1250", 65535, true},
                      {"gfx1250", 65536, false},
                      {"gfx1250", 65537, false},
                      {"gfx1250-strict", 65535, true},
                      {"gfx1250-strict", 65536, false},
                      {"gfx1250-strict", 65537, false}};
    for(const auto& b : boundaries)
    {
        ExpectFilter(b.arch, ParamsFor(1, b.workgroups, 1), b.fits);
        ExpectFilter(b.arch, ParamsFor(1, 1, b.workgroups), b.fits);
        ExpectFilter(b.arch, ParamsFor(1, b.workgroups, b.workgroups), b.fits);
    }

    for(const auto& a : limited_archs)
    {
        const std::size_t limit = a.limit;
        ExpectFilter(a.name, ParamsFor(1, limit - 1, limit - 1), true);
        ExpectFilter(a.name, ParamsFor(1, limit, limit), true);
        ExpectFilter(a.name, ParamsFor(1, limit + 1, limit), false);
        ExpectFilter(a.name, ParamsFor(1, limit, limit + 1), false);
        ExpectFilter(a.name, ParamsFor(1, limit + 1, limit + 1), false);
    }
}

// The limit applies to the rounded-up workgroup count, so one item past a full last workgroup is
// already over it.
TEST(CPU_hipconv_workgroup_limit_NONE, grid_fits_counts_partial_workgroups)
{
    for(const auto& a : limited_archs)
    {
        const std::size_t limit = a.limit;
        ExpectFilter(a.name, ParamsFor(1, limit, 1, /*partial=*/true), true);
        ExpectFilter(a.name, ParamsFor(1, 1, limit, /*partial=*/true), true);
        ExpectFilter(a.name, ParamsFor(1, limit + 1, 1, /*partial=*/true), false);
        ExpectFilter(a.name, ParamsFor(1, 1, limit + 1, /*partial=*/true), false);

        auto one_item_over_y = ParamsFor(1, limit, 1);
        one_item_over_y.n += 1;
        ExpectFilter(a.name, one_item_over_y, false);

        auto one_item_over_z = ParamsFor(1, 1, limit);
        one_item_over_z.k += 1;
        ExpectFilter(a.name, one_item_over_z, false);
    }
}

TEST(CPU_hipconv_workgroup_limit_NONE, grid_fits_does_not_limit_x)
{
    for(const auto& a : limited_archs)
    {
        ExpectFilter(a.name, ParamsFor(kHugeX, 1, 1), true);
        ExpectFilter(a.name, ParamsFor(kHugeX, a.limit, a.limit), true);
        ExpectFilter(a.name, ParamsFor(kHugeX, a.limit + 1, 1), false);
    }
}

// An architecture without a limit takes every grid, however large.
TEST(CPU_hipconv_workgroup_limit_NONE, grid_fits_passes_through_without_a_limit)
{
    for(const auto arch : unlimited_archs)
    {
        ExpectFilter(arch, ParamsFor(1, 65537, 65537), true);
        ExpectFilter(arch, ParamsFor(kHugeX, kHugeYZ, kHugeYZ), true);
    }
}

// launch() rejects an over-limit grid before it runs the kernel or any HIP call, so this needs no
// device. A grid that fits does reach HIP (hipGetLastError), so it is not exercised here.
TEST(CPU_hipconv_workgroup_limit_NONE, launch_rejects_over_limit_grid)
{
    for(const auto& a : limited_archs)
    {
        const std::size_t limit                = a.limit;
        const hipconv::ConvParams over_limit[] = {ParamsFor(1, limit + 1, 1),
                                                  ParamsFor(1, 1, limit + 1),
                                                  ParamsFor(1, limit + 1, limit + 1),
                                                  ParamsFor(1, limit, limit + 1, /*partial=*/true)};
        for(const auto& par : over_limit)
        {
            SCOPED_TRACE(std::string{a.name} + " n=" + std::to_string(par.n) +
                         " k=" + std::to_string(par.k));

            Rig rig{a.name};
            const auto lp = rig.kernel.get_launch_params(rig.device, par);
            g_launches    = 0;

            EXPECT_EQ(ThrownHipError([&] {
                          rig.kernel.launch(
                              lp, rig.device, par, nullptr, nullptr, nullptr, nullptr, nullptr);
                      }),
                      hipErrorInvalidConfiguration);
            EXPECT_EQ(g_launches, 0);

            // The public entry point turns the same HipError into its return code.
            EXPECT_EQ(hipconv::launch(&rig.kernel, rig.device, par, nullptr, nullptr, nullptr),
                      hipErrorInvalidConfiguration);
            EXPECT_EQ(g_launches, 0);
        }
    }
}

#endif // MIOPEN_USE_HIPCONV
