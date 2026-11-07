// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The per-problem CK caches answer a query from an earlier problem's result whenever their keys
// match, so a key that misses something the answer depends on silently gives one problem another
// problem's answer. These tests build problems that differ from a base problem in exactly one
// field and require every key to tell them apart, and check the cache's bound and its switch.

#include <miopen/conv/heuristics/default_config_cache.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/env.hpp>
#include <miopen/solver/ck_problem_cache.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/tensor.hpp>

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include "../../src/ck_impl/ck_grouped_conv_impl_helpers.hpp"
#endif

#include <gtest/gtest.h>

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace {

using miopen::conv::Direction;
using miopen::conv::ProblemDescription;

/// A 2D grouped convolution. The base has a 10x10 image, a 3x3 filter, stride 5 and no padding,
/// so pad 1, stride 6 and dilation 2 (each in one dimension) keep both the 2x2 output and the
/// adjusted right padding of 0: those cases differ from the base only in the field itself, not in
/// anything derived from it.
struct Spec
{
    miopenDataType_t type       = miopenBFloat16;
    miopenTensorLayout_t layout = miopenTensorNHWC;
    std::size_t n = 2, c = 8, h = 10, w = 10, k = 16, y = 3, x = 3, groups = 1;
    int pad_h = 0, pad_w = 0, stride_h = 5, stride_w = 5, dil_h = 1, dil_w = 1;
    /// Extra channels in x's channel stride (NHWC only): a non-packed input.
    std::size_t c_stride_extra = 0;
    double beta                = 0.0;
};

miopen::TensorDescriptor MakeX(const Spec& s)
{
    if(s.c_stride_extra == 0)
        return {s.type, s.layout, {s.n, s.c, s.h, s.w}};
    // Named vectors: only the const& overload is exported from MIOpen.
    const std::size_t cs                   = s.c + s.c_stride_extra;
    const std::vector<std::size_t> lens    = {s.n, s.c, s.h, s.w};
    const std::vector<std::size_t> strides = {s.h * s.w * cs, 1, s.w * cs, cs};
    return {s.type, miopenTensorNHWC, lens, strides};
}

ProblemDescription MakeProblem(const Spec& s, Direction dir = Direction::BackwardWeights)
{
    const miopen::TensorDescriptor x = MakeX(s);
    const miopen::TensorDescriptor weights{s.type, s.layout, {s.k, s.c / s.groups, s.y, s.x}};
    const miopen::ConvolutionDescriptor conv{2,
                                             miopenConvolution,
                                             miopenPaddingDefault,
                                             {s.pad_h, s.pad_w},
                                             {s.stride_h, s.stride_w},
                                             {s.dil_h, s.dil_w},
                                             {0, 0},
                                             static_cast<int>(s.groups),
                                             1.0f};
    const miopen::TensorDescriptor yd = conv.GetForwardOutputTensor(x, weights, s.type);
    const miopen::Scalar alpha{1.0};
    const miopen::Scalar beta{s.beta};
    if(dir == Direction::Forward)
        return {x, weights, yd, conv, dir, 0, alpha, beta};
    return {yd, weights, x, conv, dir, 0, alpha, beta};
}

struct Variation
{
    const char* field;
    std::function<void(Spec&)> apply;
    /// The tensor lengths and the adjusted right padding stay those of the base problem.
    bool same_output = false;
    /// The field reaches CK's arguments. The layout does not: the CK grouped solvers run NCHW
    /// problems through the NHWC kernels after a transpose, so CKArgsSplitK gives a packed NCHW
    /// problem the strides of the packed NHWC one, and CK's answer is the same for both.
    bool ck_argument = true;
};

/// One case per problem field that reaches the solvers' keys.
std::vector<Variation> SingleFieldVariations()
{
    return {
        {"data type", [](Spec& s) { s.type = miopenHalf; }},
        {"layout", [](Spec& s) { s.layout = miopenTensorNCHW; }, false, false},
        {"batch", [](Spec& s) { s.n = 3; }},
        {"input channels", [](Spec& s) { s.c = 16; }},
        {"output channels", [](Spec& s) { s.k = 24; }},
        {"height", [](Spec& s) { s.h = 12; }},
        {"width", [](Spec& s) { s.w = 12; }},
        {"filter height", [](Spec& s) { s.y = 1; }},
        {"filter width", [](Spec& s) { s.x = 1; }},
        {"groups", [](Spec& s) { s.groups = 2; }},
        {"pad height", [](Spec& s) { s.pad_h = 1; }, true},
        {"pad width", [](Spec& s) { s.pad_w = 1; }, true},
        {"stride height", [](Spec& s) { s.stride_h = 6; }, true},
        {"stride width", [](Spec& s) { s.stride_w = 6; }, true},
        {"dilation height", [](Spec& s) { s.dil_h = 2; }, true},
        {"dilation width", [](Spec& s) { s.dil_w = 2; }, true},
        {"non-packed input", [](Spec& s) { s.c_stride_extra = 8; }, true},
        {"alpha/beta case", [](Spec& s) { s.beta = 1.0; }, true},
    };
}

Spec Varied(const Variation& v)
{
    Spec s;
    v.apply(s);
    return s;
}

/// The tensor lengths and the adjusted right padding, which derive from the other fields: a
/// same-output case must leave all of them unchanged.
std::vector<std::size_t> Derived(const ProblemDescription& p)
{
    using miopen::solver::ProblemInterpreter;
    auto values     = p.GetIn().GetLengths();
    const auto& out = p.GetOut().GetLengths();
    values.insert(values.end(), out.begin(), out.end());
    values.push_back(static_cast<std::size_t>(ProblemInterpreter::GetAdjustedInputRightPadH(p)));
    values.push_back(static_cast<std::size_t>(ProblemInterpreter::GetAdjustedInputRightPadW(p)));
    return values;
}

std::string Key(const ProblemDescription& problem,
                const std::string& device = "gfx942",
                bool ranked_defaults      = true,
                const std::string& solver = "ConvHipImplicitGemmGroupWrwXdlops")
{
    return miopen::solver::conv::DefaultConfigKey(solver, device, problem, ranked_defaults);
}

/// Restores MIOPEN_DEBUG_CK_PROBLEM_CACHE's effective value when the test ends.
class CacheSwitch
{
public:
    explicit CacheSwitch(bool on) : previous(!miopen::env::disabled(MIOPEN_DEBUG_CK_PROBLEM_CACHE))
    {
        miopen::env::update(MIOPEN_DEBUG_CK_PROBLEM_CACHE, on);
    }
    ~CacheSwitch() { miopen::env::update(MIOPEN_DEBUG_CK_PROBLEM_CACHE, previous); }
    CacheSwitch(const CacheSwitch&)            = delete;
    CacheSwitch& operator=(const CacheSwitch&) = delete;

private:
    bool previous;
};

using IntCache = miopen::solver::CKProblemCache<std::map<int, int>>;

/// Looks `key` up, counting computations; the computed answer is key * 10.
std::optional<int> Query(IntCache& cache, int key, int& computed)
{
    return cache.GetOrCompute(key, [&] {
        ++computed;
        return std::optional<int>{key * 10};
    });
}

} // namespace

TEST(CPU_CKProblemCache_NONE, ComputesEachProblemOnce)
{
    const CacheSwitch on{true};
    IntCache cache;
    int computed = 0;
    EXPECT_EQ(Query(cache, 1, computed), 10);
    EXPECT_EQ(Query(cache, 1, computed), 10);
    EXPECT_EQ(Query(cache, 2, computed), 20);
    EXPECT_EQ(computed, 2);
    EXPECT_EQ(cache.Size(), 2u);
}

TEST(CPU_CKProblemCache_NONE, KeepsTheFirstAnswer)
{
    const CacheSwitch on{true};
    IntCache cache;
    EXPECT_EQ(cache.GetOrCompute(1, [] { return std::optional<int>{5}; }), 5);
    EXPECT_EQ(cache.GetOrCompute(1, [] { return std::optional<int>{6}; }), 5);
}

TEST(CPU_CKProblemCache_NONE, DoesNotStoreAMissingAnswer)
{
    const CacheSwitch on{true};
    IntCache cache;
    int computed  = 0;
    const auto no = [&] {
        ++computed;
        return std::optional<int>{};
    };
    EXPECT_FALSE(cache.GetOrCompute(1, no));
    EXPECT_FALSE(cache.GetOrCompute(1, no));
    EXPECT_EQ(computed, 2);
    EXPECT_EQ(cache.Size(), 0u);
}

TEST(CPU_CKProblemCache_NONE, StartsOverWhenFull)
{
    const CacheSwitch on{true};
    IntCache cache{2};
    int computed = 0;
    Query(cache, 1, computed);
    Query(cache, 2, computed);
    // A known problem is answered without growing or clearing the full cache.
    EXPECT_EQ(Query(cache, 2, computed), 20);
    EXPECT_EQ(cache.Size(), 2u);
    EXPECT_EQ(computed, 2);

    // A new problem clears it first, so the cache never holds more than its capacity.
    EXPECT_EQ(Query(cache, 3, computed), 30);
    EXPECT_EQ(cache.Size(), 1u);
    EXPECT_EQ(Query(cache, 1, computed), 10);
    EXPECT_EQ(computed, 4);
    EXPECT_EQ(cache.Size(), 2u);
}

TEST(CPU_CKProblemCache_NONE, SwitchedOffComputesEveryTime)
{
    IntCache cache;
    int computed = 0;
    {
        const CacheSwitch off{false};
        EXPECT_EQ(Query(cache, 1, computed), 10);
        EXPECT_EQ(Query(cache, 1, computed), 10);
        EXPECT_EQ(computed, 2);
        EXPECT_EQ(cache.Size(), 0u);
    }
    const CacheSwitch on{true};
    Query(cache, 1, computed);
    Query(cache, 1, computed);
    EXPECT_EQ(computed, 3);
}

TEST(CPU_DefaultConfigKey_NONE, SameProblemSameKey)
{
    EXPECT_EQ(Key(MakeProblem(Spec{})), Key(MakeProblem(Spec{})));
}

TEST(CPU_DefaultConfigKey_NONE, EveryProblemFieldChangesTheKey)
{
    const ProblemDescription base = MakeProblem(Spec{});
    for(const auto& v : SingleFieldVariations())
    {
        const ProblemDescription varied = MakeProblem(Varied(v));
        if(v.same_output)
            EXPECT_EQ(Derived(varied), Derived(base)) << v.field;
        EXPECT_NE(Key(varied), Key(base)) << v.field;
    }
}

TEST(CPU_DefaultConfigKey_NONE, DeviceSolverAndRankingChangeTheKey)
{
    const ProblemDescription p = MakeProblem(Spec{});
    const std::string base     = Key(p);
    EXPECT_NE(Key(p, "gfx950"), base) << "device";
    EXPECT_NE(Key(p, "gfx942", false), base) << "ranked defaults";
    EXPECT_NE(Key(p, "gfx942", true, "ConvHipImplicitGemm3DGroupWrwXdlops"), base) << "solver";
}

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

namespace {

/// The CK argument fields shared by the backward-data and backward-weights solvers.
/// Key() and the fields do not depend on the direction-specific MakeArgPtr().
struct KeyProbe : CKArgsSplitK<KeyProbe>
{
    using CKArgsSplitK<KeyProbe>::CKArgsSplitK;
};

} // namespace

TEST(CPU_CKArgsSplitKKey_NONE, SameProblemSameKey)
{
    EXPECT_EQ(KeyProbe{MakeProblem(Spec{})}.Key(), KeyProbe{MakeProblem(Spec{})}.Key());
}

TEST(CPU_CKArgsSplitKKey_NONE, EveryProblemFieldChangesTheKey)
{
    for(const Direction dir : {Direction::BackwardWeights, Direction::BackwardData})
    {
        const ProblemDescription base = MakeProblem(Spec{}, dir);
        for(const auto& v : SingleFieldVariations())
        {
            if(!v.ck_argument)
                continue;
            const ProblemDescription varied = MakeProblem(Varied(v), dir);
            if(v.same_output)
                EXPECT_EQ(Derived(varied), Derived(base)) << v.field;
            EXPECT_NE(KeyProbe{varied}.Key(), KeyProbe{base}.Key())
                << v.field << (dir == Direction::BackwardData ? " (bwd data)" : " (bwd weights)");
        }
    }
}

#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
