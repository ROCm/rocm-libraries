// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceGdn.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using GateDomain = hipdnn_flatbuffers_sdk::data_objects::GateDomain;

namespace
{

constexpr float K_TOLERANCE = 1e-5f;

// Advances `index` to the next row-major position of `dims`.
void nextIndex(std::vector<int64_t>& index, const std::vector<int64_t>& dims)
{
    for(size_t d = dims.size(); d-- > 0;)
    {
        if(++index[d] < dims[d])
        {
            return;
        }
        index[d] = 0;
    }
}

template <typename Type>
Tensor<Type> makeTensor(const std::vector<int64_t>& dims, const std::vector<float>& values)
{
    Tensor<Type> tensor(dims, generateStrides(dims));
    EXPECT_EQ(static_cast<size_t>(tensor.elementCount()), values.size());
    std::vector<int64_t> index(dims.size(), 0);
    for(const auto value : values)
    {
        tensor.setHostValue(static_cast<Type>(value), index);
        nextIndex(index, dims);
    }
    return tensor;
}

template <typename Type>
Tensor<Type> makeFilled(const std::vector<int64_t>& dims, float value)
{
    Tensor<Type> tensor(dims, generateStrides(dims));
    tensor.fillTensorWithValue(value);
    return tensor;
}

template <typename Type>
std::vector<float> valuesOf(const TensorBase<Type>& tensor)
{
    std::vector<float> result;
    const auto& dims = tensor.dims();
    std::vector<int64_t> index(dims.size(), 0);
    for(size_t i = 0; i < tensor.elementCount(); ++i)
    {
        result.push_back(static_cast<float>(tensor.getHostValue(index)));
        nextIndex(index, dims);
    }
    return result;
}

std::vector<float> randomValues(size_t count, std::mt19937& rng, float low, float high)
{
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<float> values(count);
    for(auto& value : values)
    {
        value = distribution(rng);
    }
    return values;
}

void expectNear(const std::vector<float>& actual,
                const std::vector<float>& expected,
                float tolerance = K_TOLERANCE)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "at flat index " << i;
    }
}

// A single-head, single-sequence problem (H = HV = 1) with K = V = 2.
struct SmallProblem
{
    SmallProblem(const std::vector<float>& qValues,
                 const std::vector<float>& kValues,
                 const std::vector<float>& vValues,
                 const std::vector<float>& gValues,
                 const std::vector<float>& betaValues)
        : tokens(static_cast<int64_t>(gValues.size()))
        , q(makeTensor<float>({tokens, 1, 2}, qValues))
        , k(makeTensor<float>({tokens, 1, 2}, kValues))
        , v(makeTensor<float>({tokens, 1, 2}, vValues))
        , g(makeTensor<float>({tokens, 1}, gValues))
        , beta(makeTensor<float>({tokens, 1}, betaValues))
        , cuSeqlens(makeTensor<int32_t>({2}, {0.0f, static_cast<float>(tokens)}))
        , o(makeFilled<float>({tokens, 1, 2}, 0.0f))
        , finalState(makeFilled<float>({1, 1, 2, 2}, 0.0f))
    {
    }

    GdnTensors<float> tensors()
    {
        GdnTensors<float> t;
        t.q = &q;
        t.k = &k;
        t.v = &v;
        t.g = &g;
        t.beta = &beta;
        t.cuSeqlens = &cuSeqlens;
        t.o = &o;
        t.finalState = &finalState;
        return t;
    }

    int64_t tokens;
    Tensor<float> q;
    Tensor<float> k;
    Tensor<float> v;
    Tensor<float> g;
    Tensor<float> beta;
    Tensor<int32_t> cuSeqlens;
    Tensor<float> o;
    Tensor<float> finalState;
};

GdnConfig unitScale(GateDomain domain = GateDomain::LINEAR)
{
    GdnConfig config;
    config.scale = 1.0f;
    config.gateDomain = domain;
    return config;
}

// A random multi-head, multi-sequence problem with K = 3, V = 2. Heads: q H, k HK, v HV.
struct RandomProblem
{
    RandomProblem(const std::vector<int32_t>& offsets,
                  int64_t headsQ,
                  int64_t headsK,
                  int64_t headsV,
                  unsigned seed,
                  int64_t stateRows = -1)
        : tokens(offsets.back())
        , sequences(static_cast<int64_t>(offsets.size()) - 1)
        , headsOut(std::max(headsQ, headsV))
        , rng(seed)
        , q(makeTensor<float>({tokens, headsQ, K_KEY},
                              randomValues(size(tokens * headsQ * K_KEY), rng, -1.0f, 1.0f)))
        , k(makeTensor<float>({tokens, headsK, K_KEY},
                              randomValues(size(tokens * headsK * K_KEY), rng, -1.0f, 1.0f)))
        , v(makeTensor<float>({tokens, headsV, K_VALUE},
                              randomValues(size(tokens * headsV * K_VALUE), rng, -1.0f, 1.0f)))
        , g(makeTensor<float>({tokens, headsOut},
                              randomValues(size(tokens * headsOut), rng, -0.7f, -0.05f)))
        , beta(makeTensor<float>({tokens, headsOut},
                                 randomValues(size(tokens * headsOut), rng, 0.1f, 0.9f)))
        , cuSeqlens(makeTensor<int32_t>({sequences + 1}, toFloat(offsets)))
        , initialState(
              makeTensor<float>({stateRows < 0 ? sequences : stateRows, headsOut, K_VALUE, K_KEY},
                                randomValues(size((stateRows < 0 ? sequences : stateRows) * headsOut
                                                  * K_VALUE * K_KEY),
                                             rng,
                                             -0.5f,
                                             0.5f)))
    {
    }

    static size_t size(int64_t count)
    {
        return static_cast<size_t>(count);
    }

    static std::vector<float> toFloat(const std::vector<int32_t>& values)
    {
        return std::vector<float>(values.begin(), values.end());
    }

    GdnTensors<float> tensors(Tensor<float>& o)
    {
        GdnTensors<float> t;
        t.q = &q;
        t.k = &k;
        t.v = &v;
        t.g = &g;
        t.beta = &beta;
        t.cuSeqlens = &cuSeqlens;
        t.o = &o;
        return t;
    }

    Tensor<float> makeOutput() const
    {
        return makeFilled<float>({tokens, headsOut, K_VALUE}, 0.0f);
    }

    static constexpr int64_t K_KEY = 3;
    static constexpr int64_t K_VALUE = 2;

    int64_t tokens;
    int64_t sequences;
    int64_t headsOut;
    std::mt19937 rng;
    Tensor<float> q;
    Tensor<float> k;
    Tensor<float> v;
    Tensor<float> g;
    Tensor<float> beta;
    Tensor<int32_t> cuSeqlens;
    Tensor<float> initialState;
};

// The elements [first, first + count) of `values`.
std::vector<float> slice(const std::vector<float>& values, size_t first, size_t count)
{
    const auto begin = values.begin() + static_cast<std::ptrdiff_t>(first);
    return std::vector<float>(begin, begin + static_cast<std::ptrdiff_t>(count));
}

// Copies the token rows [first, first + count) of a token-major tensor into a new tensor.
Tensor<float> sliceTokens(const Tensor<float>& source, int64_t first, int64_t count)
{
    auto dims = source.dims();
    dims[0] = count;
    const auto rowSize = source.elementCount() / static_cast<size_t>(source.dims()[0]);
    return makeTensor<float>(dims,
                             slice(valuesOf(source),
                                   static_cast<size_t>(first) * rowSize,
                                   static_cast<size_t>(count) * rowSize));
}

} // namespace

// Hand-checked three-token recurrence with a LINEAR gate and unit scale:
//   t0: k=[1,0] v=[2,3] alpha=0.5 beta=1   -> S=[[2,0],[3,0]],         o=S[1,1]=[2,3]
//   t1: k=[0,1] v=[1,1] alpha=0.5 beta=0.5 -> S=[[1,.5],[1.5,.5]],      o=S[1,2]=[2,2.5]
//   t2: k=[1,1] v=[0,0] alpha=1   beta=1   -> S k=[1.5,2], S=[[-.5,-1],[-.5,-1.5]], o=S[1,0]
TEST(TestCpuFpReferenceGdn, HandCheckedRecurrence)
{
    SmallProblem p({1, 1, 1, 2, 1, 0},
                   {1, 0, 0, 1, 1, 1},
                   {2, 3, 1, 1, 0, 0},
                   {0.5f, 0.5f, 1.0f},
                   {1.0f, 0.5f, 1.0f});

    CpuFpReferenceGdn::forward(p.tensors(), unitScale());

    expectNear(valuesOf(p.o), {2, 3, 2, 2.5f, -0.5f, -0.5f});
    expectNear(valuesOf(p.finalState), {-0.5f, -1.0f, -0.5f, -1.5f});
}

// LOG domain with g = ln(alpha) reproduces the LINEAR-domain result.
TEST(TestCpuFpReferenceGdn, LogAndLinearGateDomainsAgree)
{
    SmallProblem linear({1, 1, 1, 2, 1, 0},
                        {1, 0, 0, 1, 1, 1},
                        {2, 3, 1, 1, 0, 0},
                        {0.5f, 0.5f, 1.0f},
                        {1.0f, 0.5f, 1.0f});
    SmallProblem log({1, 1, 1, 2, 1, 0},
                     {1, 0, 0, 1, 1, 1},
                     {2, 3, 1, 1, 0, 0},
                     {std::log(0.5f), std::log(0.5f), 0.0f},
                     {1.0f, 0.5f, 1.0f});

    CpuFpReferenceGdn::forward(linear.tensors(), unitScale(GateDomain::LINEAR));
    CpuFpReferenceGdn::forward(log.tensors(), unitScale(GateDomain::LOG));

    expectNear(valuesOf(log.o), valuesOf(linear.o));
    expectNear(valuesOf(log.finalState), valuesOf(linear.finalState));
}

// The LINEAR gate floors alpha at 1e-10: alpha = 0 behaves as (almost) a full reset.
TEST(TestCpuFpReferenceGdn, LinearGateIsFloored)
{
    SmallProblem p({1, 0}, {0, 0}, {0, 0}, {0.0f}, {0.0f});
    auto initial = makeTensor<float>({1, 1, 2, 2}, {1, 2, 3, 4});
    auto t = p.tensors();
    t.initialState = &initial;

    CpuFpReferenceGdn::forward(t, unitScale());

    // k = 0: S = alpha S with alpha = 1e-10.
    expectNear(valuesOf(p.finalState), {1e-10f, 2e-10f, 3e-10f, 4e-10f}, 1e-15f);
}

// Initial state, LOG gate g = 0 (alpha = 1), beta = 0.5:
//   S0=[[1,2],[3,4]], k=[1,0], v=0: S k=[1,3], S=[[0.5,2],[1.5,4]], o=S[0,1]=[2,4].
// With the default scale 1 / sqrt(K), o is divided by sqrt(2).
TEST(TestCpuFpReferenceGdn, InitialStateAndDefaultScale)
{
    SmallProblem p({0, 1}, {1, 0}, {0, 0}, {0.0f}, {0.5f});
    auto initial = makeTensor<float>({1, 1, 2, 2}, {1, 2, 3, 4});
    auto t = p.tensors();
    t.initialState = &initial;

    CpuFpReferenceGdn::forward(t, unitScale(GateDomain::LOG));
    expectNear(valuesOf(p.o), {2, 4});
    expectNear(valuesOf(p.finalState), {0.5f, 2, 1.5f, 4});

    GdnConfig defaultScale;
    CpuFpReferenceGdn::forward(t, defaultScale);
    const auto invSqrt2 = 1.0f / std::sqrt(2.0f);
    expectNear(valuesOf(p.o), {2 * invSqrt2, 4 * invSqrt2});
}

// Without initial_state the state starts at zero: one token gives S = beta v k^T.
TEST(TestCpuFpReferenceGdn, ZeroStateDefault)
{
    SmallProblem p({1, 0}, {0, 1}, {3, 5}, {0.25f}, {2.0f});

    CpuFpReferenceGdn::forward(p.tensors(), unitScale());

    expectNear(valuesOf(p.finalState), {0, 6, 0, 10});
    expectNear(valuesOf(p.o), {0, 0});
}

// use_beta_sigmoid: beta logit 0 -> 0.5; allow_neg_eigval doubles it to 1 (full erase).
TEST(TestCpuFpReferenceGdn, BetaSigmoidAndNegativeEigenvalues)
{
    SmallProblem p({1, 0}, {1, 0}, {0, 0}, {1.0f}, {0.0f});
    auto initial = makeTensor<float>({1, 1, 2, 2}, {1, 2, 3, 4});
    auto t = p.tensors();
    t.initialState = &initial;

    auto config = unitScale();
    config.useBetaSigmoid = true;
    CpuFpReferenceGdn::forward(t, config);
    expectNear(valuesOf(p.o), {0.5f, 1.5f});

    config.allowNegEigval = true;
    CpuFpReferenceGdn::forward(t, config);
    expectNear(valuesOf(p.o), {0, 0});
    expectNear(valuesOf(p.finalState), {0, 2, 0, 4});
}

// safe_gate: alpha = exp(-exp(a_log) * softplus(g + dt_bias)). With k = 0 the state only decays.
//   no a_log / dt_bias, g = 0:            alpha = exp(-ln 2) = 0.5
//   a_log = ln 2, dt_bias = -1, g = 1:     alpha = exp(-2 ln 2) = 0.25
TEST(TestCpuFpReferenceGdn, SafeGate)
{
    SmallProblem p({1, 0}, {0, 0}, {0, 0}, {0.0f}, {1.0f});
    auto initial = makeTensor<float>({1, 1, 2, 2}, {1, 2, 3, 4});
    auto t = p.tensors();
    t.initialState = &initial;

    auto config = unitScale(GateDomain::LOG);
    config.safeGate = true;
    CpuFpReferenceGdn::forward(t, config);
    expectNear(valuesOf(p.o), {0.5f, 1.5f});

    auto shifted = makeTensor<float>({1, 1}, {1.0f});
    auto aLog = makeTensor<float>({1}, {std::log(2.0f)});
    auto dtBias = makeTensor<float>({1}, {-1.0f});
    t.g = &shifted;
    t.aLog = &aLog;
    t.dtBias = &dtBias;
    CpuFpReferenceGdn::forward(t, config);
    expectNear(valuesOf(p.o), {0.25f, 0.75f});
}

// use_qk_l2norm: k=[3,4] -> [0.6,0.8], q=[0,2] -> [0,1] (epsilon 1e-6 is negligible here);
// S = v k^T = [[0.6,0.8],[1.2,1.6]], o = S q = [0.8,1.6].
TEST(TestCpuFpReferenceGdn, QkL2Norm)
{
    SmallProblem p({0, 2}, {3, 4}, {1, 2}, {1.0f}, {1.0f});

    auto config = unitScale();
    config.useQkL2norm = true;
    CpuFpReferenceGdn::forward(p.tensors(), config);

    expectNear(valuesOf(p.finalState), {0.6f, 0.8f, 1.2f, 1.6f});
    expectNear(valuesOf(p.o), {0.8f, 1.6f});
}

// The L2 epsilon matters for tiny rows: x * rsqrt(|x|^2 + 1e-6).
TEST(TestCpuFpReferenceGdn, QkL2NormEpsilon)
{
    SmallProblem p({1e-3f, 0}, {1e-3f, 0}, {1, 0}, {1.0f}, {1.0f});

    auto config = unitScale();
    config.useQkL2norm = true;
    CpuFpReferenceGdn::forward(p.tensors(), config);

    const auto normalized = 1e-3f / std::sqrt(1e-6f + 1e-6f);
    expectNear(valuesOf(p.finalState), {normalized, 0, 0, 0}, 1e-6f);
    expectNear(valuesOf(p.o), {normalized * normalized, 0}, 1e-6f);
}

// GQA head sharing: H = HK = 1 query/key head replicated onto HV = HO = 2 value heads,
// and H = 2 query heads over one shared value head (HK = HV = 1).
TEST(TestCpuFpReferenceGdn, GroupedHeadSharing)
{
    {
        auto q = makeTensor<float>({1, 1, 2}, {1, 0});
        auto k = makeTensor<float>({1, 1, 2}, {1, 0});
        auto v = makeTensor<float>({1, 2, 2}, {1, 2, 3, 4});
        auto g = makeTensor<float>({1, 2}, {1, 1});
        auto beta = makeTensor<float>({1, 2}, {1, 0.5f});
        auto cu = makeTensor<int32_t>({2}, {0, 1});
        auto o = makeFilled<float>({1, 2, 2}, 0.0f);
        GdnTensors<float> t;
        t.q = &q;
        t.k = &k;
        t.v = &v;
        t.g = &g;
        t.beta = &beta;
        t.cuSeqlens = &cu;
        t.o = &o;
        CpuFpReferenceGdn::forward(t, unitScale());
        expectNear(valuesOf(o), {1, 2, 1.5f, 2});
    }
    {
        auto q = makeTensor<float>({1, 2, 2}, {1, 0, 0, 1});
        auto k = makeTensor<float>({1, 1, 2}, {1, 1});
        auto v = makeTensor<float>({1, 1, 2}, {2, 4});
        auto g = makeTensor<float>({1, 2}, {1, 1});
        auto beta = makeTensor<float>({1, 2}, {1, 1});
        auto cu = makeTensor<int32_t>({2}, {0, 1});
        auto o = makeFilled<float>({1, 2, 2}, 0.0f);
        GdnTensors<float> t;
        t.q = &q;
        t.k = &k;
        t.v = &v;
        t.g = &g;
        t.beta = &beta;
        t.cuSeqlens = &cu;
        t.o = &o;
        CpuFpReferenceGdn::forward(t, unitScale());
        // S = v k^T = [[2,2],[4,4]] for both heads; head 0 reads q=[1,0], head 1 q=[0,1].
        expectNear(valuesOf(o), {2, 4, 2, 4});
    }
}

// Running a sequence in two chunks, carrying the state, equals one pass.
TEST(TestCpuFpReferenceGdn, ChunkSplitInvariance)
{
    RandomProblem whole({0, 6}, 2, 2, 2, 7);
    auto oWhole = whole.makeOutput();
    auto finalWhole = makeFilled<float>({1, 2, 2, 3}, 0.0f);
    auto t = whole.tensors(oWhole);
    t.initialState = &whole.initialState;
    t.finalState = &finalWhole;
    auto config = unitScale(GateDomain::LOG);
    config.useQkL2norm = true;
    CpuFpReferenceGdn::forward(t, config);

    std::vector<float> chunkedOutputs;
    auto carried = makeTensor<float>({1, 2, 2, 3}, valuesOf(whole.initialState));
    for(const int64_t first : {int64_t{0}, int64_t{4}})
    {
        const int64_t count = first == 0 ? 4 : 2;
        auto q = sliceTokens(whole.q, first, count);
        auto k = sliceTokens(whole.k, first, count);
        auto v = sliceTokens(whole.v, first, count);
        auto g = sliceTokens(whole.g, first, count);
        auto beta = sliceTokens(whole.beta, first, count);
        auto cu = makeTensor<int32_t>({2}, {0, static_cast<float>(count)});
        auto o = makeFilled<float>({count, 2, 2}, 0.0f);
        auto next = makeFilled<float>({1, 2, 2, 3}, 0.0f);
        GdnTensors<float> chunk;
        chunk.q = &q;
        chunk.k = &k;
        chunk.v = &v;
        chunk.g = &g;
        chunk.beta = &beta;
        chunk.cuSeqlens = &cu;
        chunk.o = &o;
        chunk.initialState = &carried;
        chunk.finalState = &next;
        CpuFpReferenceGdn::forward(chunk, config);
        const auto values = valuesOf(o);
        chunkedOutputs.insert(chunkedOutputs.end(), values.begin(), values.end());
        carried = makeTensor<float>({1, 2, 2, 3}, valuesOf(next));
    }

    expectNear(chunkedOutputs, valuesOf(oWhole));
    expectNear(valuesOf(carried), valuesOf(finalWhole));
}

// Packed sequences are independent: a batch equals each sequence run on its own, and an empty
// sequence's final state is its initial state.
TEST(TestCpuFpReferenceGdn, PackedSequencesAreIndependent)
{
    RandomProblem batch({0, 3, 3, 5}, 1, 1, 2, 11);
    auto oBatch = batch.makeOutput();
    auto finalBatch = makeFilled<float>({3, 2, 2, 3}, 0.0f);
    auto t = batch.tensors(oBatch);
    t.initialState = &batch.initialState;
    t.finalState = &finalBatch;
    const auto config = unitScale(GateDomain::LOG);
    CpuFpReferenceGdn::forward(t, config);

    const auto initialValues = valuesOf(batch.initialState);
    const auto finalValues = valuesOf(finalBatch);
    const auto oValues = valuesOf(oBatch);
    const size_t stateSize = 2 * 2 * 3;
    const size_t tokenSize = 2 * 2;

    // Sequence 1 is empty.
    expectNear(slice(finalValues, stateSize, stateSize),
               slice(initialValues, stateSize, stateSize));

    const std::vector<std::pair<int64_t, int64_t>> spans = {{0, 3}, {3, 2}};
    const std::vector<size_t> stateRowsOf = {0, 2};
    for(size_t s = 0; s < spans.size(); ++s)
    {
        const auto first = spans[s].first;
        const auto count = spans[s].second;
        auto q = sliceTokens(batch.q, first, count);
        auto k = sliceTokens(batch.k, first, count);
        auto v = sliceTokens(batch.v, first, count);
        auto g = sliceTokens(batch.g, first, count);
        auto beta = sliceTokens(batch.beta, first, count);
        auto cu = makeTensor<int32_t>({2}, {0, static_cast<float>(count)});
        auto o = makeFilled<float>({count, 2, 2}, 0.0f);
        const auto row = stateRowsOf[s];
        auto initial
            = makeTensor<float>({1, 2, 2, 3}, slice(initialValues, row * stateSize, stateSize));
        auto finalOne = makeFilled<float>({1, 2, 2, 3}, 0.0f);
        GdnTensors<float> single;
        single.q = &q;
        single.k = &k;
        single.v = &v;
        single.g = &g;
        single.beta = &beta;
        single.cuSeqlens = &cu;
        single.o = &o;
        single.initialState = &initial;
        single.finalState = &finalOne;
        CpuFpReferenceGdn::forward(single, config);

        expectNear(valuesOf(o),
                   slice(oValues,
                         static_cast<size_t>(first) * tokenSize,
                         static_cast<size_t>(count) * tokenSize));
        expectNear(valuesOf(finalOne), slice(finalValues, row * stateSize, stateSize));
    }
}

// state_indices addresses a pool: sequence n reads and writes row stateIndices[n]; other rows of
// final_state are left untouched. Writing final_state into initial_state's own buffer (in place)
// gives the same result, because every state is read before any is written.
TEST(TestCpuFpReferenceGdn, StatePoolAndInPlaceUpdate)
{
    RandomProblem pool({0, 2, 5}, 2, 2, 2, 19, 3);
    auto indices = makeTensor<int32_t>({2}, {2, 0});

    auto oPool = pool.makeOutput();
    auto finalPool = makeFilled<float>({3, 2, 2, 3}, 42.0f);
    auto t = pool.tensors(oPool);
    t.initialState = &pool.initialState;
    t.stateIndices = &indices;
    t.finalState = &finalPool;
    const auto config = unitScale(GateDomain::LOG);
    CpuFpReferenceGdn::forward(t, config);

    // Row 1 is not addressed and keeps its contents.
    const auto finalValues = valuesOf(finalPool);
    const size_t stateSize = 2 * 2 * 3;
    for(size_t i = stateSize; i < 2 * stateSize; ++i)
    {
        EXPECT_EQ(finalValues[i], 42.0f);
    }

    // The dense equivalent: initial rows gathered in sequence order.
    const auto initialValues = valuesOf(pool.initialState);
    std::vector<float> gathered = slice(initialValues, 2 * stateSize, stateSize);
    const auto firstRow = slice(initialValues, 0, stateSize);
    gathered.insert(gathered.end(), firstRow.begin(), firstRow.end());
    auto denseInitial = makeTensor<float>({2, 2, 2, 3}, gathered);
    auto denseFinal = makeFilled<float>({2, 2, 2, 3}, 0.0f);
    auto oDense = pool.makeOutput();
    auto dense = pool.tensors(oDense);
    dense.initialState = &denseInitial;
    dense.finalState = &denseFinal;
    CpuFpReferenceGdn::forward(dense, config);

    expectNear(valuesOf(oPool), valuesOf(oDense));
    const auto denseFinalValues = valuesOf(denseFinal);
    expectNear(slice(finalValues, 2 * stateSize, stateSize), slice(denseFinalValues, 0, stateSize));
    expectNear(slice(finalValues, 0, stateSize), slice(denseFinalValues, stateSize, stateSize));

    // In place: final_state is initial_state's tensor.
    auto inPlace = makeTensor<float>({3, 2, 2, 3}, initialValues);
    auto oInPlace = pool.makeOutput();
    auto aliased = pool.tensors(oInPlace);
    aliased.initialState = &inPlace;
    aliased.stateIndices = &indices;
    aliased.finalState = &inPlace;
    CpuFpReferenceGdn::forward(aliased, config);

    expectNear(valuesOf(oInPlace), valuesOf(oPool));
    const auto inPlaceValues = valuesOf(inPlace);
    expectNear(slice(inPlaceValues, 2 * stateSize, stateSize),
               slice(finalValues, 2 * stateSize, stateSize));
    expectNear(slice(inPlaceValues, 0, stateSize), slice(finalValues, 0, stateSize));
    // The unaddressed row keeps its initial contents.
    expectNear(slice(inPlaceValues, stateSize, stateSize),
               slice(initialValues, stateSize, stateSize));
}

// Checkpoints every n = 2 tokens over sequences of lengths [5, 0, 2]: (5-1)/2+1 = 3 rows for
// sequence 0 (after 0, 2, 4 tokens), none for the empty sequence, 1 row (its incoming state) for
// sequence 2. C = max(7 / 2 + 3, 1) = 6 rows are allocated; rows 4 and 5 stay untouched.
TEST(TestCpuFpReferenceGdn, CheckpointRows)
{
    RandomProblem problem({0, 5, 5, 7}, 1, 1, 1, 23);
    ASSERT_EQ(CpuFpReferenceGdn::checkpointCapacity(7, 3, 2), 6);

    auto o = problem.makeOutput();
    auto checkpoints = makeFilled<float>({6, 1, 2, 3}, 7.0f);
    auto t = problem.tensors(o);
    t.initialState = &problem.initialState;
    t.stateCheckpoints = &checkpoints;
    auto config = unitScale(GateDomain::LOG);
    config.checkpointEveryNTokens = 2;
    CpuFpReferenceGdn::forward(t, config);

    const auto initialValues = valuesOf(problem.initialState);
    const auto rows = valuesOf(checkpoints);
    const size_t stateSize = 2 * 3;
    const auto row = [&](size_t r) { return slice(rows, r * stateSize, stateSize); };

    // Row 0 of each sequence is its incoming state.
    expectNear(row(0), slice(initialValues, 0, stateSize));
    expectNear(row(3), slice(initialValues, 2 * stateSize, stateSize));

    // Rows 1 and 2 are the state after the first 2 and 4 tokens of sequence 0.
    for(const int64_t prefix : {int64_t{2}, int64_t{4}})
    {
        auto q = sliceTokens(problem.q, 0, prefix);
        auto k = sliceTokens(problem.k, 0, prefix);
        auto v = sliceTokens(problem.v, 0, prefix);
        auto g = sliceTokens(problem.g, 0, prefix);
        auto beta = sliceTokens(problem.beta, 0, prefix);
        auto cu = makeTensor<int32_t>({2}, {0, static_cast<float>(prefix)});
        auto prefixOut = makeFilled<float>({prefix, 1, 2}, 0.0f);
        auto initial = makeTensor<float>({1, 1, 2, 3}, slice(initialValues, 0, stateSize));
        auto finalPrefix = makeFilled<float>({1, 1, 2, 3}, 0.0f);
        GdnTensors<float> prefixTensors;
        prefixTensors.q = &q;
        prefixTensors.k = &k;
        prefixTensors.v = &v;
        prefixTensors.g = &g;
        prefixTensors.beta = &beta;
        prefixTensors.cuSeqlens = &cu;
        prefixTensors.o = &prefixOut;
        prefixTensors.initialState = &initial;
        prefixTensors.finalState = &finalPrefix;
        auto prefixConfig = config;
        prefixConfig.checkpointEveryNTokens = 0;
        CpuFpReferenceGdn::forward(prefixTensors, prefixConfig);
        expectNear(row(static_cast<size_t>(prefix / 2)), valuesOf(finalPrefix));
    }

    for(size_t r = 4; r < 6; ++r)
    {
        for(const auto value : row(r))
        {
            EXPECT_EQ(value, 7.0f);
        }
    }
}

// HALF and BFLOAT16 io runs track the FLOAT run; the state stays in FP32.
TEST(TestCpuFpReferenceGdn, ReducedPrecisionIo)
{
    RandomProblem problem({0, 4}, 2, 2, 2, 31);
    auto oFloat = problem.makeOutput();
    auto t = problem.tensors(oFloat);
    t.initialState = &problem.initialState;
    const auto config = unitScale(GateDomain::LOG);
    CpuFpReferenceGdn::forward(t, config);

    const auto runWith = [&](auto tag, float tolerance) {
        using IoType = decltype(tag);
        auto q = makeTensor<IoType>(problem.q.dims(), valuesOf(problem.q));
        auto k = makeTensor<IoType>(problem.k.dims(), valuesOf(problem.k));
        auto v = makeTensor<IoType>(problem.v.dims(), valuesOf(problem.v));
        auto o = makeFilled<IoType>(oFloat.dims(), 0.0f);
        GdnTensors<IoType> reduced;
        reduced.q = &q;
        reduced.k = &k;
        reduced.v = &v;
        reduced.g = &problem.g;
        reduced.beta = &problem.beta;
        reduced.cuSeqlens = &problem.cuSeqlens;
        reduced.o = &o;
        reduced.initialState = &problem.initialState;
        CpuFpReferenceGdn::forward(reduced, config);
        expectNear(valuesOf(o), valuesOf(oFloat), tolerance);
    };
    runWith(half{}, 1e-2f);
    runWith(bfloat16{}, 5e-2f);
}

TEST(TestCpuFpReferenceGdn, RejectsInvalidConfigurations)
{
    SmallProblem p({1, 0}, {1, 0}, {1, 0}, {0.5f}, {0.5f});
    auto initial = makeTensor<float>({1, 1, 2, 2}, {0, 0, 0, 0});
    auto aLog = makeTensor<float>({1}, {0.0f});

    auto linearSafeGate = unitScale(GateDomain::LINEAR);
    linearSafeGate.safeGate = true;
    EXPECT_THROW(CpuFpReferenceGdn::forward(p.tensors(), linearSafeGate), std::invalid_argument);

    auto withALog = p.tensors();
    withALog.aLog = &aLog;
    EXPECT_THROW(CpuFpReferenceGdn::forward(withALog, unitScale()), std::invalid_argument);

    auto negEigval = unitScale();
    negEigval.allowNegEigval = true;
    EXPECT_THROW(CpuFpReferenceGdn::forward(p.tensors(), negEigval), std::invalid_argument);

    // state_indices without initial_state; with checkpoints; out of range.
    auto indices = makeTensor<int32_t>({1}, {0});
    auto noInitial = p.tensors();
    noInitial.stateIndices = &indices;
    EXPECT_THROW(CpuFpReferenceGdn::forward(noInitial, unitScale()), std::invalid_argument);

    auto checkpoints = makeFilled<float>({2, 1, 2, 2}, 0.0f);
    auto withCheckpoints = p.tensors();
    withCheckpoints.initialState = &initial;
    withCheckpoints.stateIndices = &indices;
    withCheckpoints.stateCheckpoints = &checkpoints;
    auto checkpointConfig = unitScale();
    checkpointConfig.checkpointEveryNTokens = 1;
    EXPECT_THROW(CpuFpReferenceGdn::forward(withCheckpoints, checkpointConfig),
                 std::invalid_argument);

    auto outOfRange = makeTensor<int32_t>({1}, {1});
    auto badRow = p.tensors();
    badRow.initialState = &initial;
    badRow.stateIndices = &outOfRange;
    EXPECT_THROW(CpuFpReferenceGdn::forward(badRow, unitScale()), std::out_of_range);

    // Checkpoints need n > 0 and C >= max(T / n + N, 1) rows.
    EXPECT_THROW(CpuFpReferenceGdn::forward(p.tensors(), checkpointConfig), std::invalid_argument);
    auto tooFew = makeFilled<float>({1, 1, 2, 2}, 0.0f);
    auto shortCheckpoints = p.tensors();
    shortCheckpoints.stateCheckpoints = &tooFew;
    EXPECT_THROW(CpuFpReferenceGdn::forward(shortCheckpoints, checkpointConfig),
                 std::invalid_argument);

    // cu_seqlens must end at T.
    auto badOffsets = makeTensor<int32_t>({2}, {0, 2});
    auto badCu = p.tensors();
    badCu.cuSeqlens = &badOffsets;
    EXPECT_THROW(CpuFpReferenceGdn::forward(badCu, unitScale()), std::invalid_argument);
}
