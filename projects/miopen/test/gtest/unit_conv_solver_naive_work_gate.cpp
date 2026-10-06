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

// Pure-CPU unit tests for the naive-conv work-size threshold
// (ConvDirectNaiveConvExceedsWorkLimit). Naive stays *applicable* at any size
// (it is the universal fallback); the threshold is how the three naive solvers
// answer SolverInterface::GetSpeedClass: over the limit they report
// SolverSpeedClass::ExceedsLaunchBudget, i.e. "applicable, but a single launch
// on this shape and device is estimated to outlast the OS GPU watchdog". The
// selection layers consume that answer -- FindCore/EvaluateInvokers
// (src/conv/solver_finders.cpp) for Find's mini-benchmark and
// GetSolutionsFallback (src/hip/convolutionocl.cpp) for immediate mode -- and
// benchmark only the best class present, so the un-tiled naive kernel is not
// *executed* while any better-classed alternative applies, not even a merely
// Slow one. See src/solver/conv/conv_direct_naive_conv.cpp for the metric.
//
// Two halves are tested separately because they answer different questions:
//
//   * ConvDirectNaiveConvWork(problem)  -- shape only. MAC count; direction- and
//     layout-invariant, group-aware.
//   * ConvDirectNaiveConvWorkLimit(cus, wavefront, clock_khz) -- device only. A
//     wall-time budget converted to MACs via device throughput, so the limit
//     scales across the CDNA/RDNA span instead of being one hand-tuned constant.
//
// These tests exercise both directly on ProblemDescriptions and synthetic device
// capabilities. They never construct a Handle and never launch a kernel, so they
// are safe on machines where the huge shapes would otherwise TDR.

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <miopen/conv/problem_description.hpp>
#include <miopen/solver.hpp>
#include <miopen/solver/conv_direct_naive_conv.hpp>

#include "unit_TensorDescriptor.hpp"
#include "unit_conv_ConvolutionDescriptor.hpp"
#include "lib_env_var.hpp"
#include "gtest_common.hpp"

namespace {

using miopen::unit_tests::ConvolutionDescriptorParams;
using miopen::unit_tests::TensorDescriptorParams;
using Direction = miopen::conv::Direction;

// Mirrors of the env overrides read by the limit model (both declared UINT64 in
// the library). MAX_WORK pins the limit outright; MAX_TIME_MS retunes the
// wall-time budget the limit is derived from.
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_WORK)
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_TIME_MS)

// Build a 2D conv ProblemDescription with square spatial dims, odd square
// filter, unit stride/dilation and 'same' padding (so out spatial == in
// spatial). Layout/dtype are irrelevant to the work metric, so we use the
// simplest (NCHW, fp32); lens are always in NCHW logical order.
miopen::conv::ProblemDescription Make2dProblem(
    std::size_t n, std::size_t c, std::size_t hw, std::size_t k, std::size_t fyx, Direction dir)
{
    const int pad = static_cast<int>(fyx / 2);
    TensorDescriptorParams in{miopenFloat, {n, c, hw, hw}};
    TensorDescriptorParams wei{miopenFloat, {k, c, fyx, fyx}};
    TensorDescriptorParams out{miopenFloat, {n, k, hw, hw}};
    ConvolutionDescriptorParams conv{{pad, pad}, {1, 1}, {1, 1}};
    return miopen::conv::ProblemDescription{in.GetTensorDescriptor(),
                                            wei.GetTensorDescriptor(),
                                            out.GetTensorDescriptor(),
                                            conv.GetConvolutionDescriptor(),
                                            dir};
}

// 3D variant (NCDHW), same conventions.
miopen::conv::ProblemDescription Make3dProblem(
    std::size_t n, std::size_t c, std::size_t dhw, std::size_t k, std::size_t fzyx, Direction dir)
{
    const int pad = static_cast<int>(fzyx / 2);
    TensorDescriptorParams in{miopenFloat, {n, c, dhw, dhw, dhw}};
    TensorDescriptorParams wei{miopenFloat, {k, c, fzyx, fzyx, fzyx}};
    TensorDescriptorParams out{miopenFloat, {n, k, dhw, dhw, dhw}};
    ConvolutionDescriptorParams conv{{pad, pad, pad}, {1, 1, 1}, {1, 1, 1}};
    return miopen::conv::ProblemDescription{in.GetTensorDescriptor(),
                                            wei.GetTensorDescriptor(),
                                            out.GetTensorDescriptor(),
                                            conv.GetConvolutionDescriptor(),
                                            dir};
}

// Depthwise 2D conv (group_count == in_channels == out_channels), so
// C_per_group == 1. Same spatial conventions as Make2dProblem; weights are
// {k, C_per_group=1, fyx, fyx}. Because the work metric folds in the group count
// (below), every real depthwise conv lands far under the limit, so the selection
// gate never work-defers it -- Naive keeps competing on merit with no special
// case. These tests lock that safety invariant in.
miopen::conv::ProblemDescription
MakeDepthwise2dProblem(std::size_t n, std::size_t c, std::size_t hw, std::size_t fyx, Direction dir)
{
    const int pad   = static_cast<int>(fyx / 2);
    const int group = static_cast<int>(c); // depthwise: group == c == k
    TensorDescriptorParams in{miopenFloat, {n, c, hw, hw}};
    TensorDescriptorParams wei{miopenFloat, {c, 1, fyx, fyx}}; // {k, C_per_group, fyx, fyx}
    TensorDescriptorParams out{miopenFloat, {n, c, hw, hw}};
    ConvolutionDescriptorParams conv{{pad, pad}, {1, 1}, {1, 1}, group};
    return miopen::conv::ProblemDescription{in.GetTensorDescriptor(),
                                            wei.GetTensorDescriptor(),
                                            out.GetTensorDescriptor(),
                                            conv.GetConvolutionDescriptor(),
                                            dir};
}

// Representative device capabilities, as (hw_cus, wavefront_width, clock_khz) --
// exactly the triple the limit model consumes. Using synthetic values keeps these
// tests CPU-only while still covering the CDNA/RDNA span we have to scale across.
// hw_cus is GetMaxHardwareComputeUnits(), i.e. already CU-per-WGP doubled for gfx1*.
//
// At the 1000 ms default budget these yield roughly:
//   MI300X  304 CU  x wave64 x 2.1 GHz -> ~123 GMAC
//   iGPU     32 WGP x wave32 x 2.9 GHz -> ~8.9 GMAC
// a ~14x spread, which is the whole point of deriving the limit instead of fixing it.
constexpr std::size_t kMi300xCus = 304, kMi300xWave = 64, kMi300xClockKhz = 2100000;
constexpr std::size_t kIgpuCus = 32, kIgpuWave = 32, kIgpuClockKhz = 2900000;

// The fixed limit used when a device reports incomplete capabilities. Mirrors
// NAIVE_CONV_FALLBACK_MAX_WORK in the library, which is deliberately not exported:
// duplicating the value here means a change to it has to be made twice on purpose.
constexpr std::size_t kFallbackMaxWork = static_cast<std::size_t>(16) * 1000 * 1000 * 1000;

std::size_t WorkLimit(std::size_t cus, std::size_t wave, std::size_t clock_khz)
{
    return miopen::solver::conv::ConvDirectNaiveConvWorkLimit(cus, wave, clock_khz);
}

std::size_t Work(const miopen::conv::ProblemDescription& p)
{
    return miopen::solver::conv::ConvDirectNaiveConvWork(p);
}

// Stand-in for the combiner, which needs a live device. The gate is
// Work(problem) > WorkLimit(device caps); these tests drive both halves directly so
// they never construct a Handle and never launch a kernel, which is what makes them
// safe on machines where the huge shapes would otherwise TDR.
bool ExceedsWorkLimitOn(const miopen::conv::ProblemDescription& p,
                        std::size_t cus,
                        std::size_t wave,
                        std::size_t clock_khz)
{
    return Work(p) > WorkLimit(cus, wave, clock_khz);
}

// Most shape-metric tests only care about the relative ordering of work counts, not
// which device is running; pin them to the small end (the iGPU) so a shape that is
// "too big" is too big for everything we ship.
bool ExceedsWorkLimit(const miopen::conv::ProblemDescription& p)
{
    return ExceedsWorkLimitOn(p, kIgpuCus, kIgpuWave, kIgpuClockKhz);
}

} // namespace

// The confirmed-TDR SDXL VAE decode conv (c128 k128 768^2 3x3, N=1) is ~87 GMAC,
// an order of magnitude above the iGPU's ~8.9 GMAC limit -> classified
// ExceedsLaunchBudget (naive skipped in the mini-bench whenever any better-classed
// alternative also applies).
TEST(CPU_ConvNaiveWorkGate_NONE, HugeShapeExceedsLimit)
{
    EXPECT_TRUE(ExceedsWorkLimit(Make2dProblem(1, 128, 768, 128, 3, Direction::Forward)));
}

// A resnet50-class 3x3 (~0.1 GMAC) is orders of magnitude below the limit -> under
// threshold (naive runs normally in the mini-bench).
TEST(CPU_ConvNaiveWorkGate_NONE, SmallShapeWithinLimit)
{
    EXPECT_FALSE(ExceedsWorkLimit(Make2dProblem(1, 64, 56, 64, 3, Direction::Forward)));
}

// A large 3D conv (~98 GMAC) is over threshold too -- the metric folds in output
// depth and the 3D filter volume.
TEST(CPU_ConvNaiveWorkGate_NONE, HugeShape3dExceedsLimit)
{
    EXPECT_TRUE(ExceedsWorkLimit(Make3dProblem(1, 64, 96, 64, 3, Direction::Forward)));
}

// The MAC total is identical for fwd/bwd/wrw, so one constant governs all three
// directions: the huge shape is over threshold and the small shape is not,
// regardless of direction.
TEST(CPU_ConvNaiveWorkGate_NONE, DirectionInvariant)
{
    for(const auto dir : {Direction::Forward, Direction::BackwardData, Direction::BackwardWeights})
    {
        EXPECT_TRUE(ExceedsWorkLimit(Make2dProblem(1, 128, 768, 128, 3, dir)));
        EXPECT_FALSE(ExceedsWorkLimit(Make2dProblem(1, 64, 56, 64, 3, dir)));
    }
}

// Depthwise safety invariant for the selection gate: because the work metric folds
// in the group count (C_per_group == 1 for depthwise), the same c/k/hw/filter that
// is far *over* the limit as a dense conv is ~C times smaller as depthwise and lands
// well *under* it. This is why Naive is never work-deferred for depthwise and keeps
// competing (TDR-safe). A large-spatial depthwise (c=k=g=1024, 64^2, 3x3 ~= 38 MMAC)
// stays under threshold; the matching dense conv (~38 GMAC) is over.
TEST(CPU_ConvNaiveWorkGate_NONE, DepthwiseFarUnderLimit)
{
    EXPECT_FALSE(ExceedsWorkLimit(MakeDepthwise2dProblem(1, 1024, 64, 3, Direction::Forward)));
}

TEST(CPU_ConvNaiveWorkGate_NONE, DenseCounterpartOverLimit)
{
    // Identical c/k/hw/filter as the depthwise above but dense (group == 1): ~C
    // times more work, pushing it over the limit -- confirms the metric is
    // group-aware and the depthwise case is not merely small by coincidence.
    EXPECT_TRUE(ExceedsWorkLimit(Make2dProblem(1, 1024, 64, 1024, 3, Direction::Forward)));
}

// The env override raises the limit; a huge shape that is normally over threshold
// falls under it when the cap is set above its work count (how a user with a raised
// TdrDelay opts naive back into the mini-bench on big shapes).
TEST(CPU_ConvNaiveWorkGate_NONE, EnvOverrideRaisesLimit)
{
    const auto huge = Make2dProblem(1, 128, 768, 128, 3, Direction::Forward);
    ASSERT_TRUE(ExceedsWorkLimit(huge)); // over threshold with the default limit
    ScopedEnvironment<std::uint64_t> raise(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_WORK,
                                           std::uint64_t{1} << 60);
    EXPECT_FALSE(ExceedsWorkLimit(huge));
}

// The env override also lowers the limit: a small shape that is normally under
// threshold goes over it when the cap is dropped below its work count. Confirms the
// override is honored in both directions rather than only disabling the check.
TEST(CPU_ConvNaiveWorkGate_NONE, EnvOverrideLowersLimit)
{
    const auto small = Make2dProblem(1, 64, 56, 64, 3, Direction::Forward);
    ASSERT_FALSE(ExceedsWorkLimit(small)); // under threshold with the default limit
    ScopedEnvironment<std::uint64_t> lower(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_WORK,
                                           std::uint64_t{1000});
    EXPECT_TRUE(ExceedsWorkLimit(small));
}

// ---------------------------------------------------------------------------
// Device scaling. The tests above pin the device so they measure the shape half;
// these pin the shape so they measure the device half.
// ---------------------------------------------------------------------------

// A bigger part earns a bigger budget, monotonically in every capability the model
// consumes. Stated as an ordering rather than against magic numbers so the test
// survives a recalibration of the throughput constant.
TEST(CPU_ConvNaiveWorkGate_NONE, LimitScalesWithDeviceCapability)
{
    const auto igpu   = WorkLimit(kIgpuCus, kIgpuWave, kIgpuClockKhz);
    const auto mi300x = WorkLimit(kMi300xCus, kMi300xWave, kMi300xClockKhz);
    EXPECT_GT(mi300x, igpu);

    // Each capability independently raises the limit: double it, the limit doubles.
    EXPECT_EQ(WorkLimit(2 * kIgpuCus, kIgpuWave, kIgpuClockKhz), 2 * igpu);
    EXPECT_EQ(WorkLimit(kIgpuCus, 2 * kIgpuWave, kIgpuClockKhz), 2 * igpu);
    EXPECT_EQ(WorkLimit(kIgpuCus, kIgpuWave, 2 * kIgpuClockKhz), 2 * igpu);
}

// The payoff of deriving the limit instead of fixing it: the *same* shape is
// last-resort on a small part and a normal candidate on a big one. The SDXL
// VAE-decode conv (~87 GMAC) is a confirmed TDR on a laptop-class part but sits
// comfortably inside a 304-CU MI300X's ~123 GMAC second, where naive should stay in
// the running rather than being gated off a machine that can afford it.
TEST(CPU_ConvNaiveWorkGate_NONE, SameShapeSlowOnSmallDeviceNotOnLarge)
{
    const auto sdxl = Make2dProblem(1, 128, 768, 128, 3, Direction::Forward);
    EXPECT_TRUE(ExceedsWorkLimitOn(sdxl, kIgpuCus, kIgpuWave, kIgpuClockKhz));
    EXPECT_FALSE(ExceedsWorkLimitOn(sdxl, kMi300xCus, kMi300xWave, kMi300xClockKhz));
}

// A device that cannot report a capability gives the model nothing to scale from, so
// the limit must degrade to the fixed pre-device-aware value rather than deriving a
// zero budget -- which would multiply out to 0 and gate *every* shape, including the
// tiny ones naive normally wins.
TEST(CPU_ConvNaiveWorkGate_NONE, IncompleteDeviceCapabilitiesUseFixedFallback)
{
    EXPECT_EQ(WorkLimit(0, kIgpuWave, kIgpuClockKhz), kFallbackMaxWork);
    EXPECT_EQ(WorkLimit(kIgpuCus, 0, kIgpuClockKhz), kFallbackMaxWork);
    EXPECT_EQ(WorkLimit(kIgpuCus, kIgpuWave, 0), kFallbackMaxWork);

    // And the fallback is a real working limit, not a disable: a small shape still
    // passes under it while the huge one still trips it.
    const auto small = Make2dProblem(1, 64, 56, 64, 3, Direction::Forward);
    const auto huge  = Make2dProblem(1, 128, 768, 128, 3, Direction::Forward);
    EXPECT_LT(Work(small), kFallbackMaxWork);
    EXPECT_GT(Work(huge), kFallbackMaxWork);
}

// ---------------------------------------------------------------------------
// Time budget. The limit is a wall-time budget expressed in MACs, so the budget is
// the knob a user reaches for after changing their watchdog timeout.
// ---------------------------------------------------------------------------

// Halving the budget halves the limit, on any device: the conversion is linear in
// time, which is what makes "set this to your TdrDelay" a usable instruction.
TEST(CPU_ConvNaiveWorkGate_NONE, TimeBudgetScalesLimitLinearly)
{
    const auto baseline = WorkLimit(kIgpuCus, kIgpuWave, kIgpuClockKhz); // 1000 ms default
    {
        ScopedEnvironment<std::uint64_t> half(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_TIME_MS,
                                              std::uint64_t{500});
        EXPECT_EQ(WorkLimit(kIgpuCus, kIgpuWave, kIgpuClockKhz), baseline / 2);
    }
    {
        ScopedEnvironment<std::uint64_t> doubled(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_TIME_MS,
                                                 std::uint64_t{2000});
        EXPECT_EQ(WorkLimit(kIgpuCus, kIgpuWave, kIgpuClockKhz), baseline * 2);
    }
}

// A user who raised their watchdog can raise the budget to match, putting a shape
// that was last-resort back into the running -- the same opt-in as the absolute
// MAX_WORK override but expressed in the units the watchdog actually uses.
TEST(CPU_ConvNaiveWorkGate_NONE, TimeBudgetRaisesShapeBackIntoContention)
{
    const auto sdxl = Make2dProblem(1, 128, 768, 128, 3, Direction::Forward);
    ASSERT_TRUE(ExceedsWorkLimit(sdxl));
    // ~87 GMAC vs ~8.9 GMAC/s on the iGPU needs ~10 s; 30 s clears it with margin.
    ScopedEnvironment<std::uint64_t> budget(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_TIME_MS,
                                            std::uint64_t{30000});
    EXPECT_FALSE(ExceedsWorkLimit(sdxl));
}

// The budget is clamped to a ceiling. This exists only so an outlandish env value
// cannot overflow the multiply into a small (or zero) limit, which would silently
// invert the override's meaning and gate everything.
TEST(CPU_ConvNaiveWorkGate_NONE, TimeBudgetIsClampedToCeiling)
{
    std::size_t at_ceiling = 0;
    {
        ScopedEnvironment<std::uint64_t> ceiling(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_TIME_MS,
                                                 std::uint64_t{60} * 1000);
        at_ceiling = WorkLimit(kIgpuCus, kIgpuWave, kIgpuClockKhz);
    }
    ScopedEnvironment<std::uint64_t> absurd(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_TIME_MS,
                                            std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(WorkLimit(kIgpuCus, kIgpuWave, kIgpuClockKhz), at_ceiling);
}

// MAX_WORK wins over MAX_TIME_MS. They are not composable -- one pins the limit and
// the other retunes a derivation -- so the absolute override short-circuits first
// and the budget is never consulted.
TEST(CPU_ConvNaiveWorkGate_NONE, AbsoluteOverrideBeatsTimeBudget)
{
    ScopedEnvironment<std::uint64_t> budget(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_TIME_MS,
                                            std::uint64_t{30000});
    ScopedEnvironment<std::uint64_t> absolute(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_WORK,
                                              std::uint64_t{12345});
    EXPECT_EQ(WorkLimit(kIgpuCus, kIgpuWave, kIgpuClockKhz), 12345u);
    // ...including on a device whose derived limit would be far larger.
    EXPECT_EQ(WorkLimit(kMi300xCus, kMi300xWave, kMi300xClockKhz), 12345u);
}

// The consumers of this gate express the whole last-resort policy as a comparison on
// SolverSpeedClass: benchmark the best (lowest) class that has an applicable member,
// defer everything worse. Reordering the enumerators would therefore silently invert
// the policy -- ExceedsLaunchBudget would outrank Slow, and a naive kernel this gate
// has just classified as a watchdog risk would be launched in preference to a solver
// that is merely off the pace. Pin the order so that stays a compile-visible change.
TEST(CPU_ConvNaiveWorkGate_NONE, SpeedClassOrderingPutsWatchdogRiskLast)
{
    using miopen::solver::SolverSpeedClass;
    EXPECT_LT(SolverSpeedClass::Normal, SolverSpeedClass::Slow);
    EXPECT_LT(SolverSpeedClass::Slow, SolverSpeedClass::ExceedsLaunchBudget);
}
