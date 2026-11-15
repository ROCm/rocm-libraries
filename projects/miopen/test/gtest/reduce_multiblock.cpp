// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/errors.hpp>
#include <miopen/reducetensor.hpp>
#include <miopen/tensor.hpp>

#include <half/half.hpp>

#include <gtest/gtest.h>

#include "get_handle.hpp"
#include "workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

// ALMIOPEN-3025: when a reduction is split over several blocks, every block stores a partial
// result in the workspace and a second kernel combines the partials. The partials have to be kept
// in the compute type; keeping them in the input type overflows (FP16 AVG of 100.0 over 2048
// elements returned inf) or loses precision.
//
// The expected values are computed on the CPU in double from the (rounded) input values. The data
// patterns are built so that they break the reduction for ANY split into blocks (every block of
// 1024 or more elements sees the same kind of data), so the tests do not depend on how the
// elements are assigned to blocks.
//
// The shapes are chosen from the thresholds in ReductionKernelConfigurator (src/reducetensor.cpp)
// so that every second-pass kernel (threadwise, warpwise, blockwise) and both the "reduce all
// dims" and "reduce some dims" variants of the first-pass kernel run:
//   - more than 1024 reduced elements selects the multi-block first pass;
//   - the second pass reduces one partial per block: at most 16 partials (threadwise), up to 256
//     (warpwise) or more (blockwise). With several outputs a reduction gets at most 32 blocks;
//     with one output it gets one block per 256 elements.
namespace {

enum class Pattern
{
    Const100,        // every element is 100
    Const300,        // every element is 300
    Const1000,       // every element is 1000
    Const12,         // every element is 12
    Const1,          // every element is 1
    SignedHalves100, // +100 in the first half of the reduced elements, -100 in the second
    SignedHalves300, // +300 in the first half of the reduced elements, -300 in the second
    CancelPrecision, // +/-(20 + (j % 7) / 64): exact in FP32, but not in FP16 once summed
    MulUnderOver,    // 0.97 in the first half, 1 / 0.97 in the second: blocks under/overflow
    ExtremeIndices,  // 0..63 repeating, 100 planted at 3K/4, -120 planted at K/3
};

struct Shape
{
    std::string name;
    std::vector<std::size_t> lengths;
    std::vector<int> reduceDims;
    std::vector<std::size_t> inStrides = {}; // empty: the input is packed
};

struct MultiBlockCase
{
    std::string name;
    miopenDataType_t inType;
    miopenDataType_t compType;
    miopenReduceTensorOp_t op;
    miopenReduceTensorIndices_t indices;
    Shape shape;
    Pattern pattern;
    float alpha      = 1.0f;
    float beta       = 0.0f;
    float outputInit = 0.0f; // initial value of the output elements (matters if beta != 0)
    // the NaN propagation selects another instance of the kernels (CK_PARAM_NAN_PROPAGATE)
    bool propagateNan = false;
    // compare with the expected value rounded to the output type, with no slack
    bool exactOutput = false;
    // the blend with the prior output is made in the compute type: the reference rounds the prior
    // output and beta to it
    bool computeBlend = false;

    friend std::ostream& operator<<(std::ostream& os, const MultiBlockCase& c)
    {
        return os << c.name;
    }
};

// Element type of the tensors
template <miopenDataType_t type>
struct ElemT;
template <>
struct ElemT<miopenHalf>
{
    using type = half_float::half;
};
template <>
struct ElemT<miopenFloat>
{
    using type = float;
};
template <>
struct ElemT<miopenInt8>
{
    using type = int8_t;
};

template <class T>
T FromDouble(double v)
{
    if constexpr(std::is_same_v<T, half_float::half>)
        return half_float::half(static_cast<float>(v));
    else
        return static_cast<T>(v);
}

template <class T>
double ToDouble(T v)
{
    return static_cast<double>(static_cast<float>(v));
}

// Visits the elements of a packed input tensor in memory order. `outOffset` is the offset of the
// output element the input element is reduced into, `pos` is the position of the element within
// its reduction (reduced elements are visited in row-major order of the reduced dimensions).
void ForEachElement(const std::vector<std::size_t>& inLengths,
                    const std::vector<std::size_t>& outLengths,
                    const std::function<void(std::size_t, std::size_t, std::size_t)>& fn)
{
    const std::size_t rank = inLengths.size();
    std::vector<std::size_t> outStrides(rank, 1);
    for(std::size_t i = rank - 1; i > 0; i--)
        outStrides[i - 1] = outStrides[i] * outLengths[i];

    const std::size_t total =
        std::accumulate(inLengths.begin(), inLengths.end(), std::size_t{1}, std::multiplies<>{});
    const std::size_t outTotal =
        std::accumulate(outLengths.begin(), outLengths.end(), std::size_t{1}, std::multiplies<>{});
    std::vector<std::size_t> posCounter(outTotal, 0);

    std::vector<std::size_t> idx(rank, 0);
    for(std::size_t lin = 0; lin < total; lin++)
    {
        std::size_t outOffset = 0;
        for(std::size_t i = 0; i < rank; i++)
            if(outLengths[i] != 1)
                outOffset += idx[i] * outStrides[i];

        fn(lin, outOffset, posCounter[outOffset]++);

        for(std::size_t i = rank; i > 0; i--)
        {
            if(++idx[i - 1] < inLengths[i - 1])
                break;
            idx[i - 1] = 0;
        }
    }
}

double PatternValue(Pattern p, std::size_t j, std::size_t k)
{
    const bool firstHalf = j < k / 2;
    switch(p)
    {
    case Pattern::Const100: return 100.0;
    case Pattern::Const300: return 300.0;
    case Pattern::Const1000: return 1000.0;
    case Pattern::Const12: return 12.0;
    case Pattern::Const1: return 1.0;
    case Pattern::SignedHalves100: return firstHalf ? 100.0 : -100.0;
    case Pattern::SignedHalves300: return firstHalf ? 300.0 : -300.0;
    case Pattern::CancelPrecision:
        return (firstHalf ? 1.0 : -1.0) * (20.0 + static_cast<double>(j % 7) / 64.0);
    case Pattern::MulUnderOver: return firstHalf ? 0.97 : 1.0 / 0.97;
    case Pattern::ExtremeIndices:
        if(j == k * 3 / 4)
            return 100.0;
        if(j == k / 3)
            return -120.0;
        return static_cast<double>(j % 64);
    }
    return 0.0;
}

double RoundToType(double v, miopenDataType_t t);

struct Reference
{
    std::vector<double> values;
    std::vector<int> indices;
};

Reference ComputeReference(const MultiBlockCase& c,
                           const std::vector<double>& input,
                           const std::vector<std::size_t>& outLengths,
                           const std::vector<double>& outputInit)
{
    const std::size_t outTotal = outputInit.size();
    const double zero          = 0.0;

    std::vector<double> sum(outTotal, zero), sumAbs(outTotal, zero), sumSq(outTotal, zero);
    std::vector<double> prod(outTotal, 1.0);
    std::vector<double> mn(outTotal, std::numeric_limits<double>::infinity());
    std::vector<double> mx(outTotal, -std::numeric_limits<double>::infinity());
    std::vector<double> amx(outTotal, zero);
    std::vector<int> idxMin(outTotal, 0), idxMax(outTotal, 0), idxAmax(outTotal, 0);
    std::vector<std::size_t> count(outTotal, 0);

    ForEachElement(
        c.shape.lengths, outLengths, [&](std::size_t lin, std::size_t o, std::size_t pos) {
            const double v = input[lin];
            sum[o] += v;
            sumAbs[o] += std::abs(v);
            sumSq[o] += v * v;
            prod[o] *= v;
            if(v < mn[o])
            {
                mn[o]     = v;
                idxMin[o] = static_cast<int>(pos);
            }
            if(v > mx[o])
            {
                mx[o]     = v;
                idxMax[o] = static_cast<int>(pos);
            }
            if(std::abs(v) > amx[o])
            {
                amx[o]     = std::abs(v);
                idxAmax[o] = static_cast<int>(pos);
            }
            count[o]++;
        });

    Reference ref;
    ref.values.resize(outTotal);
    ref.indices.assign(outTotal, 0);
    for(std::size_t o = 0; o < outTotal; o++)
    {
        double r = 0.0;
        switch(c.op)
        {
        case MIOPEN_REDUCE_TENSOR_ADD: r = sum[o]; break;
        case MIOPEN_REDUCE_TENSOR_AVG: r = sum[o] / static_cast<double>(count[o]); break;
        case MIOPEN_REDUCE_TENSOR_NORM1: r = sumAbs[o]; break;
        case MIOPEN_REDUCE_TENSOR_NORM2: r = std::sqrt(sumSq[o]); break;
        case MIOPEN_REDUCE_TENSOR_MUL: r = prod[o]; break;
        case MIOPEN_REDUCE_TENSOR_MIN:
            r              = mn[o];
            ref.indices[o] = idxMin[o];
            break;
        case MIOPEN_REDUCE_TENSOR_MAX:
            r              = mx[o];
            ref.indices[o] = idxMax[o];
            break;
        case MIOPEN_REDUCE_TENSOR_AMAX:
            r              = amx[o];
            ref.indices[o] = idxAmax[o];
            break;
        }
        const double prior =
            c.computeBlend ? RoundToType(outputInit[o], c.compType) : outputInit[o];
        const double beta =
            c.computeBlend ? RoundToType(static_cast<double>(c.beta), c.compType) : c.beta;
        ref.values[o] = static_cast<double>(c.alpha) * r + beta * prior;
    }
    return ref;
}

// Largest finite value of the type the reduction result is stored in
double MaxFinite(miopenDataType_t t)
{
    if(t == miopenHalf)
        return 65504.0;
    if(t == miopenInt8)
        return 127.0;
    return std::numeric_limits<float>::max();
}

// One ULP-ish of relative slack for the output type, plus an absolute floor
std::pair<double, double> Tolerance(miopenDataType_t t)
{
    if(t == miopenHalf)
        return {1.5e-3, 1e-3};
    if(t == miopenInt8)
        return {0.0, 0.0};
    return {2e-5, 1e-8};
}

// The value rounded to the nearest value of the type (ties to even)
double RoundToType(double v, miopenDataType_t t)
{
    if(t == miopenHalf)
    {
        int e = 0;
        std::frexp(v, &e);
        const double scale = std::ldexp(1.0, 11 - e); // 11 significant bits
        return std::nearbyint(v * scale) / scale;
    }
    if(t == miopenInt8)
        return std::nearbyint(v);
    return static_cast<double>(static_cast<float>(v));
}

constexpr std::size_t canaryBytes = 512;
constexpr std::uint8_t canaryByte = 0xA5;

template <class T>
void RunCase(const MultiBlockCase& c)
{
    auto&& handle = get_handle();

    // the output has length 1 in every reduced dimension
    auto outLengths = c.shape.lengths;
    for(const int d : c.shape.reduceDims)
        outLengths[d] = 1;

    const std::size_t inTotal = std::accumulate(
        c.shape.lengths.begin(), c.shape.lengths.end(), std::size_t{1}, std::multiplies<>{});
    const std::size_t outTotal =
        std::accumulate(outLengths.begin(), outLengths.end(), std::size_t{1}, std::multiplies<>{});
    const std::size_t reducedTotal = inTotal / outTotal;

    // input, rounded to the input type (the reference sees what the GPU sees)
    std::vector<T> input(inTotal);
    std::vector<double> inputD(inTotal);
    ForEachElement(c.shape.lengths, outLengths, [&](std::size_t lin, std::size_t, std::size_t pos) {
        input[lin]  = FromDouble<T>(PatternValue(c.pattern, pos, reducedTotal));
        inputD[lin] = ToDouble(input[lin]);
    });

    std::vector<T> output(outTotal, FromDouble<T>(c.outputInit));
    std::vector<double> outputInitD(outTotal, ToDouble(output[0]));

    const auto ref = ComputeReference(c, inputD, outLengths, outputInitD);

    // the test data must not make the correct result unrepresentable, or it would not prove
    // anything about the partial results
    for(const double v : ref.values)
        ASSERT_LE(std::abs(v), MaxFinite(c.inType)) << "bad test data, result out of range";

    // A strided input is laid out in a larger buffer; the elements the tensor does not address
    // hold a value that would change the result if they were read.
    std::vector<T> deviceInput;
    if(c.shape.inStrides.empty())
    {
        deviceInput = input;
    }
    else
    {
        const auto& lens   = c.shape.lengths;
        const auto& strd   = c.shape.inStrides;
        std::size_t extent = 1;
        for(std::size_t i = 0; i < lens.size(); i++)
            extent += (lens[i] - 1) * strd[i];
        deviceInput.assign(extent, FromDouble<T>(7.0));
        for(std::size_t lin = 0; lin < inTotal; lin++)
        {
            std::size_t rest = lin, offset = 0;
            for(std::size_t i = lens.size(); i > 0; i--)
            {
                offset += (rest % lens[i - 1]) * strd[i - 1];
                rest /= lens[i - 1];
            }
            deviceInput[offset] = input[lin];
        }
    }

    const miopen::TensorDescriptor inDesc =
        c.shape.inStrides.empty()
            ? miopen::TensorDescriptor{c.inType, c.shape.lengths}
            : miopen::TensorDescriptor{c.inType, c.shape.lengths, c.shape.inStrides};
    const miopen::TensorDescriptor outDesc{c.inType, outLengths};
    const auto nanOpt = c.propagateNan ? MIOPEN_PROPAGATE_NAN : MIOPEN_NOT_PROPAGATE_NAN;
    const miopen::ReduceTensorDescriptor reduceDesc{
        c.op, c.compType, nanOpt, c.indices, MIOPEN_32BIT_INDICES};

    const std::size_t wsBytes  = reduceDesc.GetWorkspaceSize(handle, inDesc, outDesc);
    const std::size_t idxBytes = reduceDesc.GetIndicesSize(inDesc, outDesc);

    // The workspace has exactly the size ReduceTensor() asks for; the bytes behind it must not
    // be touched (the host-side layout has to agree with the kernels).
    Workspace wspace{wsBytes + canaryBytes};
    wspace.Write(std::vector<std::uint8_t>(wsBytes + canaryBytes, canaryByte));
    Workspace idxSpace{idxBytes};

    auto input_dev  = handle.Write(deviceInput);
    auto output_dev = handle.Write(output);

    reduceDesc.ReduceTensor(handle,
                            idxBytes > 0 ? idxSpace.ptr() : nullptr,
                            idxBytes,
                            wspace.ptr(),
                            wsBytes,
                            &c.alpha,
                            inDesc,
                            input_dev.get(),
                            &c.beta,
                            outDesc,
                            output_dev.get());

    const auto gpu = handle.Read<T>(output_dev, outTotal);

    const auto [rel, abs] = c.exactOutput ? std::pair{0.0, 0.0} : Tolerance(c.inType);
    for(std::size_t o = 0; o < outTotal; o++)
    {
        const double got = ToDouble(gpu[o]);
        const double exp = c.exactOutput ? RoundToType(ref.values[o], c.inType) : ref.values[o];
        EXPECT_LE(std::abs(got - exp), rel * std::abs(exp) + abs)
            << "output " << o << ": got " << got << ", expected " << exp;
    }

    if(idxBytes > 0)
    {
        std::vector<int> gpuIdx(outTotal);
        idxSpace.ReadTo(gpuIdx);
        for(std::size_t o = 0; o < outTotal; o++)
            EXPECT_EQ(gpuIdx[o], ref.indices[o]) << "index of output " << o;
    }

    std::vector<std::uint8_t> wsAfter(wsBytes + canaryBytes);
    wspace.ReadTo(wsAfter);
    EXPECT_TRUE(std::all_of(
        wsAfter.begin() + wsBytes, wsAfter.end(), [](auto b) { return b == canaryByte; }))
        << "the reduction wrote behind the end of the workspace of " << wsBytes << " bytes";
}

void RunCaseDispatch(const MultiBlockCase& c)
{
    if(c.inType == miopenHalf)
        RunCase<ElemT<miopenHalf>::type>(c);
    else if(c.inType == miopenFloat)
        RunCase<ElemT<miopenFloat>::type>(c);
    else if(c.inType == miopenInt8)
        RunCase<ElemT<miopenInt8>::type>(c);
    else
        FAIL() << "unsupported type";
}

// The shapes; see the comment at the top of the file for what each one exercises.
const Shape thread2Partial{"Thread2Partial", {2, 2048}, {1}};      // 2 partials per output
const Shape thread2All{"Thread2All", {1, 2048}, {1}};              // 8 partials, 1 output
const Shape warp2Partial{"Warp2Partial", {2, 20480}, {1}};         // 20 partials per output
const Shape warp2All{"Warp2All", {1, 8192}, {1}};                  // 32 partials, 1 output
const Shape block2All{"Block2All", {1, 131072}, {1}};              // 512 partials, 1 output
const Shape nchwPartial{"NchwPartial", {1, 4, 64, 64}, {0, 2, 3}}; // 4 partials per output

MultiBlockCase MakeCase(const std::string& what,
                        miopenDataType_t inType,
                        miopenDataType_t compType,
                        miopenReduceTensorOp_t op,
                        const Shape& shape,
                        Pattern pattern,
                        miopenReduceTensorIndices_t indices = MIOPEN_REDUCE_TENSOR_NO_INDICES)
{
    return {what + "_" + shape.name, inType, compType, op, indices, shape, pattern};
}

std::vector<MultiBlockCase> Fp16CompFp32Smoke()
{
    // the reproducer from the ticket and its NORM2 sibling
    return {
        MakeCase("Avg100",
                 miopenHalf,
                 miopenFloat,
                 MIOPEN_REDUCE_TENSOR_AVG,
                 thread2Partial,
                 Pattern::Const100),
        MakeCase("Norm2_12",
                 miopenHalf,
                 miopenFloat,
                 MIOPEN_REDUCE_TENSOR_NORM2,
                 thread2All,
                 Pattern::Const12),
    };
}

std::vector<MultiBlockCase> Fp16CompFp32Standard()
{
    constexpr auto hf = miopenHalf;
    constexpr auto fl = miopenFloat;

    std::vector<MultiBlockCase> cases;

    for(const auto* s :
        {&thread2Partial, &thread2All, &warp2Partial, &warp2All, &block2All, &nchwPartial})
    {
        // a block of a one-output reduction sums 512 elements, a block of any other 1024
        cases.push_back(
            MakeCase("Avg300", hf, fl, MIOPEN_REDUCE_TENSOR_AVG, *s, Pattern::Const300));
        cases.push_back(
            MakeCase("Norm2_12", hf, fl, MIOPEN_REDUCE_TENSOR_NORM2, *s, Pattern::Const12));
        cases.push_back(MakeCase(
            "AddSigned300", hf, fl, MIOPEN_REDUCE_TENSOR_ADD, *s, Pattern::SignedHalves300));
        cases.push_back(
            MakeCase("AddCancel", hf, fl, MIOPEN_REDUCE_TENSOR_ADD, *s, Pattern::CancelPrecision));
        // the product of the 0.97 half and the 1/0.97 half underflows for long reductions
        if(s != &block2All)
            cases.push_back(MakeCase(
                "MulUnderOver", hf, fl, MIOPEN_REDUCE_TENSOR_MUL, *s, Pattern::MulUnderOver));
        cases.push_back(MakeCase(
            "MaxNoIndices", hf, fl, MIOPEN_REDUCE_TENSOR_MAX, *s, Pattern::ExtremeIndices));
    }

    // the reproducer from the ticket
    cases.push_back(
        MakeCase("Avg100", hf, fl, MIOPEN_REDUCE_TENSOR_AVG, thread2Partial, Pattern::Const100));

    // MIN/MAX/AMAX with flattened indices: the workspace holds the partial values and the
    // partial indices
    for(const auto* s : {&thread2Partial, &warp2All, &block2All})
    {
        constexpr auto ind = MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES;
        cases.push_back(
            MakeCase("MinIdx", hf, fl, MIOPEN_REDUCE_TENSOR_MIN, *s, Pattern::ExtremeIndices, ind));
        cases.push_back(
            MakeCase("MaxIdx", hf, fl, MIOPEN_REDUCE_TENSOR_MAX, *s, Pattern::ExtremeIndices, ind));
        cases.push_back(MakeCase(
            "AmaxIdx", hf, fl, MIOPEN_REDUCE_TENSOR_AMAX, *s, Pattern::ExtremeIndices, ind));
    }

    // alpha / beta blend of the combined partials
    auto blend =
        MakeCase("AvgBlend", hf, fl, MIOPEN_REDUCE_TENSOR_AVG, thread2Partial, Pattern::Const100);
    blend.alpha      = 0.5f;
    blend.beta       = 0.5f;
    blend.outputInit = 10.0f;
    cases.push_back(blend);

    return cases;
}

// The input type is the compute type, nothing may change for these
std::vector<MultiBlockCase> SameCompTypeStandard()
{
    return {
        MakeCase("HalfCompAvg1",
                 miopenHalf,
                 miopenHalf,
                 MIOPEN_REDUCE_TENSOR_AVG,
                 thread2Partial,
                 Pattern::Const1),
        MakeCase("HalfCompAvg1",
                 miopenHalf,
                 miopenHalf,
                 MIOPEN_REDUCE_TENSOR_AVG,
                 warp2All,
                 Pattern::Const1),
        MakeCase("HalfCompMaxIdx",
                 miopenHalf,
                 miopenHalf,
                 MIOPEN_REDUCE_TENSOR_MAX,
                 thread2Partial,
                 Pattern::ExtremeIndices,
                 MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES),
    };
}

// The reduction the RNN code runs for the gradient of the bias (see rnn_util.hpp): the sum over
// all the rows of the workspace (batch size times sequence length of them) of the bias columns
// of the rows, accumulated into the weights (beta = 1), FP32 compute, NaN propagation on. The rows
// are longer than the bias, so the input is strided.
std::vector<MultiBlockCase> RnnBiasGradientStandard()
{
    // 1 x rows x biasCount, rows are rowStride apart
    constexpr std::size_t rows = 2200, biasCount = 64, rowStride = 96;
    const Shape shape{"RnnBias", {1, rows, biasCount}, {0, 1}, {rows * rowStride, rowStride, 1}};

    auto c       = MakeCase("AddSigned100Accumulate",
                      miopenHalf,
                      miopenFloat,
                      MIOPEN_REDUCE_TENSOR_ADD,
                      shape,
                      Pattern::SignedHalves100);
    c.beta       = 1.0f;
    c.outputInit = 5.0f;

    // a constant gradient: the average fits FP16, the sum of a block does not
    auto avg       = MakeCase("AvgAccumulate",
                        miopenHalf,
                        miopenFloat,
                        MIOPEN_REDUCE_TENSOR_AVG,
                        shape,
                        Pattern::Const300);
    avg.beta       = 1.0f;
    avg.outputInit = 5.0f;

    std::vector<MultiBlockCase> cases{c, avg};
    for(auto& rnnCase : cases)
        rnnCase.propagateNan = true;
    return cases;
}

// The kernels of the reductions that propagate NaN are another instance than the ones of the
// cases above (the RNN-shaped cases excepted). One case per first-pass variant (all dims, some
// dims) and per second-pass wrapper that a shape can reach (threadwise and warpwise over all dims
// and some dims, blockwise over all dims).
std::vector<MultiBlockCase> NanPropagationStandard()
{
    constexpr auto hf = miopenHalf;
    constexpr auto fl = miopenFloat;

    std::vector<MultiBlockCase> cases;
    for(const auto* s : {&thread2Partial, &thread2All, &warp2Partial, &warp2All, &block2All})
    {
        cases.push_back(
            MakeCase("Avg300Nan", hf, fl, MIOPEN_REDUCE_TENSOR_AVG, *s, Pattern::Const300));
        cases.back().propagateNan = true;
    }
    return cases;
}

// alpha is converted to the compute type in every reduction, in a single pass or in several
// blocks: the same data reduced over 16, 128 or 512 elements (one pass) and over 2048 elements
// (blocks) gives the same result. An alpha converted to the input type first would change the
// one-pass result. One single-pass shape per kernel class: with 2 outputs, 16 elements run on a
// thread (limit warpSize, 64 or 32), 128 on a warp (limit 256) and 512 on a block (limit 1024),
// on both wave sizes; more than 1024 elements take blocks.
const Shape singlePass512{"SinglePass512", {2, 512}, {1}};
const Shape singlePass128{"SinglePass128", {2, 128}, {1}};
const Shape singlePass16{"SinglePass16", {2, 16}, {1}};
const std::vector<const Shape*> singlePassShapes = {&singlePass512, &singlePass128, &singlePass16};
const std::vector<const Shape*> alphaShapes      = [] {
    auto shapes = singlePassShapes;
    shapes.push_back(&thread2Partial);
    return shapes;
}();

// 1000 * 1.0004 is 1000.4 in FP32 and rounds to 1000.5 in FP16; with alpha rounded to FP16 first
// (1.0) the result would be 1000
std::vector<MultiBlockCase> AlphaConversionFp16()
{
    constexpr auto hf = miopenHalf;
    constexpr auto fl = miopenFloat;

    std::vector<MultiBlockCase> cases;
    for(const auto* s : alphaShapes)
    {
        cases.push_back(
            MakeCase("AlphaFp16", hf, fl, MIOPEN_REDUCE_TENSOR_AVG, *s, Pattern::Const1000));
        cases.back().alpha       = 1.0004f;
        cases.back().exactOutput = true;
    }
    return cases;
}

// the average of 100 is 100 and alpha 0.5 gives 50; an alpha converted to INT8 first would be 0
std::vector<MultiBlockCase> AlphaConversionInt8()
{
    constexpr auto i8 = miopenInt8;
    constexpr auto fl = miopenFloat;

    std::vector<MultiBlockCase> cases;
    for(const auto* s : alphaShapes)
    {
        cases.push_back(
            MakeCase("AlphaInt8", i8, fl, MIOPEN_REDUCE_TENSOR_AVG, *s, Pattern::Const100));
        cases.back().alpha = 0.5f;
    }
    return cases;
}

// MIN/MAX/AMAX with flattened indices run another instance of the single-pass kernels: the
// maximum is 100 and alpha 0.5 gives 50
std::vector<MultiBlockCase> AlphaConversionInt8Indices()
{
    std::vector<MultiBlockCase> cases;
    for(const auto* s : singlePassShapes)
    {
        cases.push_back(MakeCase("AlphaInt8MaxIdx",
                                 miopenInt8,
                                 miopenFloat,
                                 MIOPEN_REDUCE_TENSOR_MAX,
                                 *s,
                                 Pattern::ExtremeIndices,
                                 MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES));
        cases.back().alpha = 0.5f;
    }
    return cases;
}

// beta is converted to the compute type as well, and the blend with the prior output is done in
// the compute type; the result is rounded to the output type once. The second pass of a multi-block
// reduction blends too, and there the output type differs from the compute type: the shapes add
// the second-pass classes (2 partials threadwise, 20 warpwise, 512 blockwise) and the all-dims
// wrappers (one output: 512 elements in a single pass, 8 and 32 partials, 512 partials).
const Shape singlePass512All{"SinglePass512All", {1, 512}, {1}};
const std::vector<const Shape*> betaShapes = [] {
    auto shapes = alphaShapes;
    for(const auto* s : {&warp2Partial, &block2All, &thread2All, &warp2All, &singlePass512All})
        shapes.push_back(s);
    return shapes;
}();

// FP16 output: 12 + 1000 * 1.0004 is 1012.4 in FP32 and rounds to 1012.5 in FP16; with beta
// rounded to FP16 first (1.0) the result would be 1012.
std::vector<MultiBlockCase> BetaConversionFp16()
{
    std::vector<MultiBlockCase> cases;
    for(const auto* s : betaShapes)
    {
        cases.push_back(MakeCase(
            "BetaFp16", miopenHalf, miopenFloat, MIOPEN_REDUCE_TENSOR_AVG, *s, Pattern::Const12));
        cases.back().beta        = 1.0004f;
        cases.back().outputInit  = 1000.0f;
        cases.back().exactOutput = true;
    }
    return cases;
}

// 0.5 * 100 + 0.25 * 20 is 55; a beta converted to INT8 first would be 0 and the result 50, and a
// beta and an alpha taken one for the other would give 0.25 * 100 + 0.5 * 20 = 35
std::vector<MultiBlockCase> BetaConversionInt8()
{
    std::vector<MultiBlockCase> cases;
    for(const auto* s : betaShapes)
    {
        cases.push_back(MakeCase(
            "BetaInt8", miopenInt8, miopenFloat, MIOPEN_REDUCE_TENSOR_AVG, *s, Pattern::Const100));
        cases.back().alpha      = 0.5f;
        cases.back().beta       = 0.25f;
        cases.back().outputInit = 20.0f;
    }
    return cases;
}

// the same blend for MAX with flattened indices (the maximum is 100), in the first-call kernels
// with indices and in the second pass over the indices
std::vector<MultiBlockCase> BetaConversionInt8Indices()
{
    std::vector<MultiBlockCase> cases;
    for(const auto* s : betaShapes)
    {
        cases.push_back(MakeCase("BetaInt8MaxIdx",
                                 miopenInt8,
                                 miopenFloat,
                                 MIOPEN_REDUCE_TENSOR_MAX,
                                 *s,
                                 Pattern::ExtremeIndices,
                                 MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES));
        cases.back().alpha      = 0.5f;
        cases.back().beta       = 0.25f;
        cases.back().outputInit = 20.0f;
    }
    return cases;
}

// The prior output and beta are converted to the compute type before the blend, so with an output
// type wider than the compute type they are rounded to it: FP32 output and FP16 compute, beta of
// 1.0004 is 1 in FP16 and the maximum 100 plus 700 * 1 is 800 (develop blended in FP32 and gave
// 800.28; a beta left unrounded would give 800.5 in FP16). The values are exact in FP16; the
// maximum is used because a sum of the larger shapes would overflow FP16.
std::vector<MultiBlockCase> BetaWiderOutput()
{
    std::vector<MultiBlockCase> cases;
    for(const auto* s : betaShapes)
    {
        cases.push_back(MakeCase("BetaFp32Fp16CompMax",
                                 miopenFloat,
                                 miopenHalf,
                                 MIOPEN_REDUCE_TENSOR_MAX,
                                 *s,
                                 Pattern::ExtremeIndices));
        cases.push_back(MakeCase("BetaFp32Fp16CompMaxIdx",
                                 miopenFloat,
                                 miopenHalf,
                                 MIOPEN_REDUCE_TENSOR_MAX,
                                 *s,
                                 Pattern::ExtremeIndices,
                                 MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES));
    }
    for(auto& c : cases)
    {
        c.beta         = 1.0004f;
        c.outputInit   = 700.0f;
        c.exactOutput  = true;
        c.computeBlend = true;
    }
    return cases;
}

std::vector<MultiBlockCase> Fp32Standard()
{
    return {
        MakeCase("Avg100",
                 miopenFloat,
                 miopenFloat,
                 MIOPEN_REDUCE_TENSOR_AVG,
                 thread2Partial,
                 Pattern::Const100),
        MakeCase("AmaxIdx",
                 miopenFloat,
                 miopenFloat,
                 MIOPEN_REDUCE_TENSOR_AMAX,
                 warp2All,
                 Pattern::ExtremeIndices,
                 MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES),
    };
}

// int8 partials do not even fit a sum of 1024 elements of 100
std::vector<MultiBlockCase> Int8Standard()
{
    return {
        MakeCase("Avg100",
                 miopenInt8,
                 miopenFloat,
                 MIOPEN_REDUCE_TENSOR_AVG,
                 thread2Partial,
                 Pattern::Const100),
        MakeCase("Avg100",
                 miopenInt8,
                 miopenFloat,
                 MIOPEN_REDUCE_TENSOR_AVG,
                 warp2Partial,
                 Pattern::Const100),
    };
}

// The size of the workspace is a part of the interface: the partial results are stored in the
// compute type.
struct WorkspaceSizeCase
{
    std::string name;
    miopenDataType_t inType;
    miopenDataType_t compType;
    std::vector<std::size_t> lengths;
    std::vector<std::size_t> outLengths;
    std::size_t partials;  // number of partial results, 0 if the reduction has one pass
    bool rejected = false; // the size query throws miopenStatusBadParm

    friend std::ostream& operator<<(std::ostream& os, const WorkspaceSizeCase& c)
    {
        return os << c.name;
    }
};

std::size_t TypeSize(miopenDataType_t t)
{
    if(t == miopenHalf)
        return 2;
    if(t == miopenFloat)
        return 4;
    if(t == miopenDouble)
        return 8;
    return 1; // miopenInt8
}

std::vector<WorkspaceSizeCase> GetWorkspaceSizeCases()
{
    return {
        {"Fp16CompFp32_2x2048", miopenHalf, miopenFloat, {2, 2048}, {2, 1}, 4},
        {"Fp16CompFp16_2x2048", miopenHalf, miopenHalf, {2, 2048}, {2, 1}, 4},
        {"Fp32CompFp32_2x2048", miopenFloat, miopenFloat, {2, 2048}, {2, 1}, 4},
        {"Fp32CompFp64_2x2048", miopenFloat, miopenDouble, {2, 2048}, {2, 1}, 4},
        {"Int8CompFp32_2x2048", miopenInt8, miopenFloat, {2, 2048}, {2, 1}, 4},
        {"Fp16CompFp32_1x8192", miopenHalf, miopenFloat, {1, 8192}, {1, 1}, 32},
        {"Fp16CompFp32_2x1024_SinglePass", miopenHalf, miopenFloat, {2, 1024}, {2, 1}, 0},
        // the compute type is narrower than the input type: the partials are smaller than before
        {"Fp32CompFp16_2x2048", miopenFloat, miopenHalf, {2, 2048}, {2, 1}, 4},
        {"Fp64CompFp32_2x2048", miopenDouble, miopenFloat, {2, 2048}, {2, 1}, 4},
        // an odd number of partials of a narrower compute type: the size in bytes is not a multiple
        // of the size of the input type (4106 and 4116 bytes), which callers must not assume
        {"Fp32CompFp16_1x1280", miopenFloat, miopenHalf, {1, 1280}, {1, 1}, 5},
        {"Fp64CompFp32_1x1280", miopenDouble, miopenFloat, {1, 1280}, {1, 1}, 5},
        // a compute type without a size matters only if there are partial results
        {"Fp16CompInt64_2x1024_SinglePass", miopenHalf, miopenInt64, {2, 1024}, {2, 1}, 0},
        {"Fp16CompInt64_2x2048_Rejected", miopenHalf, miopenInt64, {2, 2048}, {2, 1}, 4, true},
        // an input type without a size is rejected, with or without partial results
        {"Int64CompFp32_2x1024_Rejected", miopenInt64, miopenFloat, {2, 1024}, {2, 1}, 0, true},
        {"Int64CompFp32_2x2048_Rejected", miopenInt64, miopenFloat, {2, 2048}, {2, 1}, 4, true},
    };
}

} // anonymous namespace

class GPU_ReduceMultiBlock_FP16 : public ::testing::TestWithParam<MultiBlockCase>
{
};
class GPU_ReduceMultiBlock_FP32 : public ::testing::TestWithParam<MultiBlockCase>
{
};
class GPU_ReduceMultiBlock_I8 : public ::testing::TestWithParam<MultiBlockCase>
{
};
class GPU_ReduceMultiBlockWorkspace_FP16 : public ::testing::TestWithParam<WorkspaceSizeCase>
{
};

TEST_P(GPU_ReduceMultiBlock_FP16, PartialResultsKeepComputeType) { RunCaseDispatch(GetParam()); }
TEST_P(GPU_ReduceMultiBlock_FP32, PartialResultsKeepComputeType) { RunCaseDispatch(GetParam()); }
TEST_P(GPU_ReduceMultiBlock_I8, PartialResultsKeepComputeType) { RunCaseDispatch(GetParam()); }

TEST_P(GPU_ReduceMultiBlockWorkspace_FP16, PartialsAreStoredInComputeType)
{
    const auto& c = GetParam();
    auto&& handle = get_handle();
    const miopen::TensorDescriptor inDesc{c.inType, c.lengths};
    const miopen::TensorDescriptor outDesc{c.inType, c.outLengths};
    const miopen::ReduceTensorDescriptor reduceDesc{MIOPEN_REDUCE_TENSOR_ADD,
                                                    c.compType,
                                                    MIOPEN_NOT_PROPAGATE_NAN,
                                                    MIOPEN_REDUCE_TENSOR_NO_INDICES,
                                                    MIOPEN_32BIT_INDICES};

    if(c.rejected)
    {
        try
        {
            std::ignore = reduceDesc.GetWorkspaceSize(handle, inDesc, outDesc);
            ADD_FAILURE() << "no exception thrown";
        }
        catch(const miopen::Exception& e)
        {
            EXPECT_EQ(e.status, miopenStatusBadParm);
        }
        return;
    }

    // one page for the tensor descriptors of the kernels, then the partial results
    constexpr std::size_t descriptorPage = 4096;
    EXPECT_EQ(reduceDesc.GetWorkspaceSize(handle, inDesc, outDesc),
              descriptorPage + c.partials * TypeSize(c.compType));
}

#define MIOPEN_REDUCE_MULTIBLOCK_NAME [](const auto& info) { return info.param.name; }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ReduceMultiBlock_FP16,
                         ::testing::ValuesIn(Fp16CompFp32Smoke()),
                         MIOPEN_REDUCE_MULTIBLOCK_NAME);

// The Standard tier is the one the per-PR CI (TheRock "standard" category) runs
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ReduceMultiBlock_FP16,
                         ::testing::ValuesIn([] {
                             auto cases     = Fp16CompFp32Standard();
                             const auto rnn = RnnBiasGradientStandard();
                             cases.insert(cases.end(), rnn.begin(), rnn.end());
                             const auto same = SameCompTypeStandard();
                             cases.insert(cases.end(), same.begin(), same.end());
                             const auto nan = NanPropagationStandard();
                             cases.insert(cases.end(), nan.begin(), nan.end());
                             const auto alpha = AlphaConversionFp16();
                             cases.insert(cases.end(), alpha.begin(), alpha.end());
                             const auto beta = BetaConversionFp16();
                             cases.insert(cases.end(), beta.begin(), beta.end());
                             return cases;
                         }()),
                         MIOPEN_REDUCE_MULTIBLOCK_NAME);
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ReduceMultiBlock_FP32,
                         ::testing::ValuesIn([] {
                             auto cases      = Fp32Standard();
                             const auto wide = BetaWiderOutput();
                             cases.insert(cases.end(), wide.begin(), wide.end());
                             return cases;
                         }()),
                         MIOPEN_REDUCE_MULTIBLOCK_NAME);
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ReduceMultiBlock_I8,
                         ::testing::ValuesIn([] {
                             auto cases       = Int8Standard();
                             const auto alpha = AlphaConversionInt8();
                             cases.insert(cases.end(), alpha.begin(), alpha.end());
                             const auto idx = AlphaConversionInt8Indices();
                             cases.insert(cases.end(), idx.begin(), idx.end());
                             const auto beta = BetaConversionInt8();
                             cases.insert(cases.end(), beta.begin(), beta.end());
                             const auto betaIdx = BetaConversionInt8Indices();
                             cases.insert(cases.end(), betaIdx.begin(), betaIdx.end());
                             return cases;
                         }()),
                         MIOPEN_REDUCE_MULTIBLOCK_NAME);
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ReduceMultiBlockWorkspace_FP16,
                         ::testing::ValuesIn(GetWorkspaceSizeCases()),
                         MIOPEN_REDUCE_MULTIBLOCK_NAME);
