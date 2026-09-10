// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace hipdnn_test_sdk::detail
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::fp4_e2m1;
using hipdnn_data_sdk::types::fp6_e2m3;
using hipdnn_data_sdk::types::fp6_e3m2;
using hipdnn_data_sdk::types::fp8_e4m3;
using hipdnn_data_sdk::types::fp8_e5m2;
using hipdnn_data_sdk::types::fp8_e8m0;
using hipdnn_data_sdk::types::half;

// Type trait to validate tensor types (arithmetic types + half + bfloat16 + fp8 types)
template <typename T>
constexpr bool IS_VALID_TENSOR_TYPE_V = std::disjunction_v<std::is_arithmetic<T>,
                                                           std::is_same<T, half>,
                                                           std::is_same<T, bfloat16>,
                                                           std::is_same<T, fp4_e2m1>,
                                                           std::is_same<T, fp6_e2m3>,
                                                           std::is_same<T, fp6_e3m2>,
                                                           std::is_same<T, fp8_e4m3>,
                                                           std::is_same<T, fp8_e5m2>,
                                                           std::is_same<T, fp8_e8m0>>;

/**
 * @brief Safely convert between types while avoiding implicit precision loss warnings
 *
 * This function handles type conversions that may trigger compiler warnings about
 * implicit precision loss, particularly when converting from double to bfloat16
 * or half. It makes the conversion path explicit to eliminate warnings.
 *
 * @tparam TargetType The type to convert to
 * @tparam SourceType The type to convert from
 * @param value The value to convert
 * @return The converted value
 */
template <typename TargetType, typename SourceType>
inline TargetType safeConvert(const SourceType& value)
{
    if constexpr(std::is_same_v<TargetType, bfloat16> || std::is_same_v<TargetType, half>)
    {
        // For bfloat16/half, explicitly convert through float to avoid precision warnings
        // These types lack direct constructors from double, only from float
        return static_cast<TargetType>(static_cast<float>(value));
    }
    else if constexpr(std::is_same_v<TargetType, fp4_e2m1> || std::is_same_v<TargetType, fp6_e2m3>
                      || std::is_same_v<TargetType, fp6_e3m2>
                      || std::is_same_v<TargetType, fp8_e4m3>
                      || std::is_same_v<TargetType, fp8_e5m2>
                      || std::is_same_v<TargetType, fp8_e8m0>)
    {
        // For FP8 types, convert through float
        return TargetType(static_cast<float>(value));
    }
    else
    {
        // For all other types, direct cast is fine
        return static_cast<TargetType>(value);
    }
}

/**
 * @brief Safely cast test values with range validation.
 *
 * This helper validates source values against the representable range of
 * TargetType and throws if out-of-range (or non-finite for floating-like sources).
 *
 * @tparam TargetType The type to cast to
 * @tparam SourceType The type to cast from
 * @param value The value to cast
 * @return The safely cast value
 */
template <typename TargetType, typename SourceType>
inline TargetType safeTestTypeCast(SourceType value)
{
    static_assert(std::numeric_limits<std::remove_cv_t<SourceType>>::is_specialized,
                  "safeTestTypeCast: SourceType must define numeric_limits");
    static_assert(std::numeric_limits<std::remove_cv_t<TargetType>>::is_specialized,
                  "safeTestTypeCast: TargetType must define numeric_limits");

    const auto src = safeConvert<double>(value);

    // If SourceType is not integral, treat it as floating-like and reject NaN/Inf.
    if constexpr(!std::is_integral_v<std::remove_cv_t<SourceType>>)
    {
        if(!std::isfinite(src))
        {
            throw std::out_of_range("safeTestTypeCast: non-finite source value");
        }
    }

    const auto lo = safeConvert<double>(std::numeric_limits<TargetType>::lowest());
    const auto hi = safeConvert<double>(std::numeric_limits<TargetType>::max());
    if(src < lo || src > hi)
    {
        throw std::out_of_range("safeTestTypeCast: value out of representable range");
    }

    return safeConvert<TargetType>(src);
}

struct JoinableThread : std::thread
{
    template <typename... Xs>
    JoinableThread(Xs&&... xs)
        : std::thread(std::forward<Xs>(xs)...)
    {
    }

    JoinableThread(JoinableThread&&) = default;
    JoinableThread& operator=(JoinableThread&&) = default;

    ~JoinableThread()
    {
        if(this->joinable())
        {
            this->join();
        }
    }
};

template <typename F, typename T, std::size_t... Is>
auto callFuncUnpackArgsImpl(F f, T args, [[maybe_unused]] std::index_sequence<Is...> sequence)
{
    return f(std::get<Is>(args)...);
}

template <typename F, typename T>
auto callFuncUnpackArgs(F f, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};
    return callFuncUnpackArgsImpl(f, args, std::make_index_sequence<N>{});
}

/**
 * @brief Row-major decomposition of a flat work index, plus the split of that work
 * across threads.
 *
 * Factored out so the parallel tensor functors below differ only in what they hand the
 * callee, not in how the work is divided.
 */
struct ParallelTensorRange
{
    std::vector<std::size_t> lengths;
    std::vector<std::size_t> strides;
    std::size_t totalElements{1};

    explicit ParallelTensorRange(const std::vector<int64_t>& dimensions)
        : lengths(dimensions.begin(), dimensions.end())
        , strides(dimensions.size())
    {
        if(lengths.empty())
        {
            totalElements = 0;
            return;
        }

        auto generatedStrides = hipdnn_data_sdk::utilities::generateStrides(dimensions);
        strides.assign(generatedStrides.begin(), generatedStrides.end());
        totalElements = strides[0] * lengths[0];
    }

    void fillNdIndices(std::size_t i, std::vector<int64_t>& indices) const
    {
        indices.resize(lengths.size());

        for(std::size_t idim = 0; idim < lengths.size(); ++idim)
        {
            indices[idim] = static_cast<int64_t>(i / strides[idim]);
            i -= static_cast<std::size_t>(indices[idim]) * strides[idim];
        }
    }

    std::vector<int64_t> getNdIndices(std::size_t i) const
    {
        std::vector<int64_t> indices;
        fillNdIndices(i, indices);

        return indices;
    }

    /**
     * @brief Runs `body(workBegin, workEnd)` once per thread over disjoint work ranges.
     *
     * Every thread is joined before this returns, so `body` may capture by reference.
     */
    template <typename Body>
    void runChunked(std::size_t numThreads, const Body& body) const
    {
        if(totalElements == 0)
        {
            return;
        }
        numThreads = std::min(totalElements, std::max<std::size_t>(1, numThreads));

        const std::size_t workPerThread = (totalElements + numThreads - 1) / numThreads;

        std::vector<JoinableThread> threads(numThreads);

        for(std::size_t threadIdx = 0; threadIdx < numThreads; ++threadIdx)
        {
            const std::size_t workBegin = threadIdx * workPerThread;
            const std::size_t workEnd = std::min((threadIdx + 1) * workPerThread, totalElements);

            threads[threadIdx]
                = JoinableThread([&body, workBegin, workEnd] { body(workBegin, workEnd); });
        }
    }
};

/**
 * @brief Runs a functor over every position of an index space, in parallel.
 *
 * The callee is invoked as `func(indices)`. A functor returning `bool` stops its own
 * thread early when it returns false.
 */
template <typename F>
struct ParallelTensorFunctorDynamic : ParallelTensorRange
{
    F func;

    ParallelTensorFunctorDynamic(F f, const std::vector<int64_t>& dimensions)
        : ParallelTensorRange(dimensions)
        , func(f)
    {
    }

    void operator()(std::size_t numThreads = 1) const
    {
        runChunked(numThreads, [this](std::size_t workBegin, std::size_t workEnd) {
            // One index buffer for the whole work range: the functor body only reads
            // it, so allocating a fresh vector per element is pure heap traffic.
            std::vector<int64_t> indices;

            for(std::size_t workIdx = workBegin; workIdx < workEnd; ++workIdx)
            {
                fillNdIndices(workIdx, indices);

                if constexpr(std::is_invocable_r_v<bool, F, std::vector<int64_t>>)
                {
                    if(!func(indices))
                    {
                        return;
                    }
                }
                else
                {
                    func(indices);
                }
            }
        });
    }
};

/**
 * @brief Runs a functor over every position of an index space, in parallel, handing each
 * worker thread its own `Scratch`.
 *
 * The scratch is constructed once per thread and reused for every work item that thread
 * handles. That is the lifetime a callee needs for a buffer it rebuilds per work item but
 * must not reallocate per work item - a ConvolutionWindow, for instance.
 *
 * ParallelTensorFunctorDynamic cannot express that: it invokes the callee with indices
 * only, so a callee needing state that outlives a single work item has nowhere to put it
 * short of a function-local `thread_local`. Prefer this, which makes the lifetime explicit
 * and ends it with the parallel region.
 *
 * The callee is invoked as `func(scratch, indices)`.
 */
template <typename Scratch, typename F>
struct ParallelTensorFunctorWithScratch : ParallelTensorRange
{
    static_assert(std::is_default_constructible_v<Scratch>,
                  "Scratch must be default constructible; one is created per worker thread");

    F func;

    ParallelTensorFunctorWithScratch(F f, const std::vector<int64_t>& dimensions)
        : ParallelTensorRange(dimensions)
        , func(f)
    {
    }

    void operator()(std::size_t numThreads = 1) const
    {
        runChunked(numThreads, [this](std::size_t workBegin, std::size_t workEnd) {
            Scratch scratch;
            std::vector<int64_t> indices;

            for(std::size_t workIdx = workBegin; workIdx < workEnd; ++workIdx)
            {
                fillNdIndices(workIdx, indices);

                if constexpr(std::is_invocable_r_v<bool, F, Scratch&, std::vector<int64_t>>)
                {
                    if(!func(scratch, indices))
                    {
                        return;
                    }
                }
                else
                {
                    func(scratch, indices);
                }
            }
        });
    }
};

/**
 * @brief The valid taps of a convolution window, flattened to offset pairs.
 *
 * Each spatial dimension of a convolution maps a window index to a source index
 * independently, and independently decides that the tap has no source element -
 * it lands in padding, or (for dgrad) is not stride-aligned. Validity and both
 * flat offsets therefore factor per dimension, so the whole window can be
 * resolved once per output element instead of being re-derived for every
 * (output element x channel x tap).
 *
 * Buffers are reused across rebuilds, so a `thread_local` instance stops
 * allocating after the first output element. That is the point: the index-vector
 * formulation this replaces allocated several `std::vector<int64_t>` per tap,
 * which dominated the reference's runtime on allocators without a per-thread
 * cache (the Windows heap, notably).
 */
class ConvolutionWindow
{
public:
    struct Tap
    {
        int64_t windowOffset; ///< flat offset into the tensor being walked
        int64_t sourceOffset; ///< flat offset into the tensor being sampled
    };

    /**
     * @brief Rebuilds the tap list for a single output position.
     *
     * @param nDims Number of spatial dimensions.
     * @param extents Window extent per spatial dimension.
     * @param windowStrides Strides of the walked tensor, spatial dimensions only.
     * @param sourceStrides Strides of the sampled tensor, spatial dimensions only.
     * @param mapIndex `(dim, windowIndex) -> sourceIndex`, negative when the tap
     *        has no source element.
     *
     * Taps come out in row-major window order, which is the order the
     * `iterateAlongDimensions` formulation accumulated in, so floating-point
     * results are unchanged.
     */
    template <typename MapIndex>
    void build(std::size_t nDims,
               const int64_t* extents,
               const int64_t* windowStrides,
               const int64_t* sourceStrides,
               MapIndex&& mapIndex)
    {
        _taps.assign(1, Tap{0, 0});

        for(std::size_t dim = 0; dim < nDims; ++dim)
        {
            _dimTaps.clear();
            for(int64_t windowIndex = 0; windowIndex < extents[dim]; ++windowIndex)
            {
                const int64_t sourceIndex = mapIndex(dim, windowIndex);
                if(sourceIndex >= 0)
                {
                    _dimTaps.push_back(
                        Tap{windowIndex * windowStrides[dim], sourceIndex * sourceStrides[dim]});
                }
            }

            // Prefixes outer, this dimension inner: keeps the product in row-major order.
            _scratch.clear();
            for(const auto& prefix : _taps)
            {
                for(const auto& tail : _dimTaps)
                {
                    _scratch.push_back(Tap{prefix.windowOffset + tail.windowOffset,
                                           prefix.sourceOffset + tail.sourceOffset});
                }
            }
            _taps.swap(_scratch);

            if(_taps.empty())
            {
                return;
            }
        }
    }

    const std::vector<Tap>& taps() const
    {
        return _taps;
    }

private:
    std::vector<Tap> _taps;
    std::vector<Tap> _dimTaps;
    std::vector<Tap> _scratch;
};

/**
 * @brief Row-major flat offsets for a dense walk of `extents` against `strides`.
 *
 * The normalization references walk a fixed sub-block of a tensor once per output
 * position, and the walk is identical for every output position. Building an index
 * vector per element - a heap allocation - and reducing it against the strides is
 * therefore repeated work: hoist it into a table built once per call and index that.
 *
 * `strides` must have `extents.size()` entries. A zero stride is meaningful, and is
 * how a broadcast axis contributes nothing to the address.
 */
inline std::vector<int64_t> buildDenseOffsets(const std::vector<int64_t>& extents,
                                              const int64_t* strides)
{
    std::vector<int64_t> offsets{0};
    std::vector<int64_t> scratch;

    for(std::size_t dim = 0; dim < extents.size(); ++dim)
    {
        scratch.clear();
        scratch.reserve(offsets.size() * static_cast<std::size_t>(extents[dim]));

        // Prefixes outer, this dimension inner: keeps the product in row-major order.
        for(const auto prefix : offsets)
        {
            for(int64_t index = 0; index < extents[dim]; ++index)
            {
                scratch.push_back(prefix + (index * strides[dim]));
            }
        }

        offsets.swap(scratch);
    }

    return offsets;
}

/// Flat offset of the first `count` entries of `indices` against `strides`.
inline int64_t flatOffset(const int64_t* indices, const int64_t* strides, std::size_t count)
{
    int64_t offset = 0;
    for(std::size_t dim = 0; dim < count; ++dim)
    {
        offset += indices[dim] * strides[dim];
    }

    return offset;
}

template <typename F>
auto makeParallelTensorFunctor(F f, const std::vector<int64_t>& dimensions)
{
    return ParallelTensorFunctorDynamic<F>(f, dimensions);
}

/// Companion to makeParallelTensorFunctor for callees that need per-thread scratch.
/// `Scratch` is explicit; `F` is deduced. The callee takes `(Scratch&, indices)`.
template <typename Scratch, typename F>
auto makeParallelTensorFunctorWithScratch(F f, const std::vector<int64_t>& dimensions)
{
    return ParallelTensorFunctorWithScratch<Scratch, F>(f, dimensions);
}

/**
 * @brief Reject a ragged tensor with a message identifying which argument it was.
 *
 * @param tensor The tensor to check.
 * @param errorPrefix Prefix identifying the calling CPU reference (e.g. "MyOp: ").
 * @param name The argument name to report if the tensor is ragged.
 */
inline void validateNoRaggedTensor(const hipdnn_data_sdk::utilities::ITensor& tensor,
                                   const std::string& errorPrefix,
                                   const char* name)
{
    if(tensor.raggedIterationInfo().has_value())
    {
        throw std::runtime_error(errorPrefix + "ragged " + name + " tensor is not supported");
    }
}

} // namespace hipdnn_test_sdk::detail
