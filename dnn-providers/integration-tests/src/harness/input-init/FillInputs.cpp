// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/input-init/FillInputs.hpp"
#include "harness/DeviceCallCheck.hpp"
#include "harness/input-init/ResampleBackwardInputs.hpp"
#include "harness/input-init/SdpaBackwardInputs.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/PackedFp4Tensor.hpp>
#include <hipdnn_data_sdk/utilities/PackedFp6Tensor.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#if defined(USE_ROCRAND)
#include <hip/hip_runtime.h>
#include <hipdnn-gpu-ref/GpuFillKinds.hpp>
#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>
#endif

// The sub-byte packed tensors are only tested for here (dynamic_cast), never built. Naming a
// class template that way instantiates its declarations but not its virtual members, nor those
// of the MigratableMemory<uint8_t> it holds, which portability-template-virtual-member-function
// rejects. Explicitly instantiating them defines every member. Valid at global scope only.
template class hipdnn_data_sdk::utilities::PackedSubByteTensor<hipdnn_data_sdk::types::fp4_e2m1, 4>;
template class hipdnn_data_sdk::utilities::PackedSubByteTensor<hipdnn_data_sdk::types::fp6_e2m3, 6>;
template class hipdnn_data_sdk::utilities::PackedSubByteTensor<hipdnn_data_sdk::types::fp6_e3m2, 6>;

namespace hipdnn_integration_tests
{
namespace
{

using TensorDataTypes = std::unordered_map<int64_t, hipdnn_flatbuffers_sdk::data_objects::DataType>;

TensorDataTypes collectDataTypes(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph)
{
    TensorDataTypes dataTypes;
    if(graph.tensors() == nullptr)
    {
        return dataTypes;
    }
    for(const auto* tensor : *graph.tensors())
    {
        dataTypes.emplace(tensor->uid(), tensor->data_type());
    }
    return dataTypes;
}

// Symmetric range over every finite value of T, so a uniform draw rounds onto
// each representable magnitude, including the extremes.
template <typename T>
FillRecipe symmetricFullRange()
{
    const auto max = static_cast<float>(std::numeric_limits<T>::max());
    return FillRecipe::free(-max, max);
}

// ── Host generators for the NONZERO, ZEROS, LARGE, SUBNORMAL and SPECIAL kinds ──────────────
//
// FREE and FIXED go through ITensor's own fill; the kinds below need to know the element
// type, which ITensor does not expose, so they dispatch on the tensor's concrete type
// (tryFillAs). Each generator is a callable returning the next element's value; the store
// calls it once per logical element, in index order, and writes the result where the
// element lives. A bit-packed sub-byte tensor and an unpacked one of the same type, given
// the same recipe and seed, therefore hold the same values, which the sub-byte graphs rely
// on (they are filled once in each layout).
//
// All draws come from one std::mt19937 seeded with the tensor's seed, so the same recipe,
// element type, shape and seed give identical bytes. The stream is the host's only: a tensor
// the DeviceInputFiller takes is generated on the device from rocRAND's draws instead
// (deviceFillArgs), and holds the same kind of values but not the same ones.

namespace types = hipdnn_data_sdk::types;
using ITensor = hipdnn_data_sdk::utilities::ITensor;

// Redraws before NONZERO falls back to the range's largest-magnitude endpoint.
constexpr int K_MAX_NONZERO_DRAWS = 64;

// The element types the generators handle. Draws are made in float, except for double.
template <typename T>
using DrawType = std::conditional_t<std::is_same_v<T, double>, double, float>;

template <typename T>
constexpr const char* dataTypeName()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return "float";
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        return "double";
    }
    else if constexpr(std::is_same_v<T, types::half>)
    {
        return "half";
    }
    else if constexpr(std::is_same_v<T, types::bfloat16>)
    {
        return "bfloat16";
    }
    else if constexpr(std::is_same_v<T, types::fp8_e4m3>)
    {
        return "fp8_e4m3";
    }
    else if constexpr(std::is_same_v<T, types::fp8_e5m2>)
    {
        return "fp8_e5m2";
    }
    else if constexpr(std::is_same_v<T, types::fp8_e4m3_fnuz>)
    {
        return "fp8_e4m3_fnuz";
    }
    else if constexpr(std::is_same_v<T, types::fp8_e5m2_fnuz>)
    {
        return "fp8_e5m2_fnuz";
    }
    else if constexpr(std::is_same_v<T, types::fp8_e8m0>)
    {
        return "fp8_e8m0";
    }
    else if constexpr(std::is_same_v<T, types::fp4_e2m1>)
    {
        return "fp4_e2m1";
    }
    else if constexpr(std::is_same_v<T, types::fp6_e2m3>)
    {
        return "fp6_e2m3";
    }
    else if constexpr(std::is_same_v<T, types::fp6_e3m2>)
    {
        return "fp6_e3m2";
    }
    else if constexpr(std::is_same_v<T, std::int8_t>)
    {
        return "int8";
    }
    else if constexpr(std::is_same_v<T, std::uint8_t>)
    {
        return "uint8";
    }
    else if constexpr(std::is_same_v<T, std::int32_t>)
    {
        return "int32";
    }
    else
    {
        return "int64";
    }
}

// The sub-byte floating types are stored either one per byte (Tensor<T>) or bit-packed
// (PackedSubByteTensor<T, bits>); every other type has no packed layout.
template <typename T>
struct PackedTensorFor
{
    using type = void;
};
template <>
struct PackedTensorFor<types::fp4_e2m1>
{
    using type = hipdnn_data_sdk::utilities::PackedFp4Tensor;
};
template <>
struct PackedTensorFor<types::fp6_e2m3>
{
    using type = hipdnn_data_sdk::utilities::PackedFp6Tensor<types::fp6_e2m3>;
};
template <>
struct PackedTensorFor<types::fp6_e3m2>
{
    using type = hipdnn_data_sdk::utilities::PackedFp6Tensor<types::fp6_e3m2>;
};

// Float-valued view of an element, exact for every type here: the narrow floats and the
// integers up to 2^53 fit a double, and double is itself.
template <typename T>
double toDouble(T v)
{
    if constexpr(std::is_same_v<T, double> || std::is_integral_v<T>)
    {
        return static_cast<double>(v);
    }
    else
    {
        return static_cast<double>(static_cast<float>(v));
    }
}

// `v` rounded to T. A floating T is reached through float, so `v` must be within float range,
// or infinite. An integer T is only given whole numbers within its range.
template <typename T>
T fromDouble(double v)
{
    if constexpr(std::is_same_v<T, double>)
    {
        return v;
    }
    else if constexpr(std::is_integral_v<T>)
    {
        return static_cast<T>(v);
    }
    else
    {
        return static_cast<T>(static_cast<float>(v));
    }
}

// Zero in T, which is also what an element that rounded to zero in T holds.
template <typename T>
bool isZeroElement(T v)
{
    return toDouble(v) == 0.0;
}

// `v`, a whole number held in a double, as an integer in T's range (saturating).
template <typename T>
std::int64_t clampToInteger(double v)
{
    static_assert(std::is_integral_v<T>);
    const auto lowest = std::numeric_limits<T>::lowest();
    const auto highest = std::numeric_limits<T>::max();
    if(v <= static_cast<double>(lowest))
    {
        return static_cast<std::int64_t>(lowest);
    }
    if(v >= static_cast<double>(highest))
    {
        return static_cast<std::int64_t>(highest);
    }
    return static_cast<std::int64_t>(v);
}

// A range uniform_real_distribution can draw from: finite, ordered, with a finite width.
bool isDrawableRange(float lo, float hi)
{
    return std::isfinite(lo) && std::isfinite(hi) && lo <= hi && std::isfinite(hi - lo);
}

FillResult unsupportedKind(const FillRecipe& recipe, const char* typeName, const std::string& why)
{
    return FillResult::unsupported(std::string("fill kind \"")
                                   + InputFillRecipes::kindToString(recipe.kind) + "\" on "
                                   + typeName + ": " + why);
}

// How many elements a SPECIAL fill plants: round(fraction * count), at least one when fraction
// is positive, at most all of them.
std::uint64_t specialQuota(float fraction, std::size_t count)
{
    const auto exact = std::llround(static_cast<double>(fraction) * static_cast<double>(count));
    return std::min(
        static_cast<std::uint64_t>(count),
        std::max(static_cast<std::uint64_t>(exact), std::uint64_t{fraction > 0.0f ? 1U : 0U}));
}

// The value a SPECIAL recipe plants in T, or nullopt when T cannot hold it.
template <typename T>
std::optional<T> nonFiniteValue(FillRecipe::SpecialValue special)
{
    if constexpr(std::is_integral_v<T>)
    {
        return std::nullopt;
    }
    else
    {
        switch(special)
        {
        case FillRecipe::SpecialValue::QUIET_NAN:
            if(std::numeric_limits<T>::has_quiet_NaN)
            {
                return std::numeric_limits<T>::quiet_NaN();
            }
            return std::nullopt;
        case FillRecipe::SpecialValue::POS_INF:
            if constexpr(std::numeric_limits<T>::has_infinity)
            {
                // Not converted from float: the fp8 constructors saturate an infinity to the
                // largest finite value.
                return std::numeric_limits<T>::infinity();
            }
            return std::nullopt;
        case FillRecipe::SpecialValue::NEG_INF:
            if constexpr(std::numeric_limits<T>::has_infinity)
            {
                return -std::numeric_limits<T>::infinity();
            }
            return std::nullopt;
        default:
            break;
        }
        return std::nullopt;
    }
}

// Generates the tensor's elements for a NONZERO, ZEROS, LARGE, SUBNORMAL or SPECIAL recipe
// and hands the generator to `store`; returns unsupported, before anything is written, when
// T cannot hold the kind or the recipe's parameters are unusable.
//
//   ZEROS      T(0). Unsupported where T has no zero (fp8_e8m0).
//   NONZERO    A FREE draw over [lo, hi], redrawn (up to K_MAX_NONZERO_DRAWS times) while it
//              rounds to zero in T, then replaced by the endpoint of larger magnitude, which
//              is the value of the range that rounds furthest from zero. Unsupported when
//              that endpoint is itself zero in T (the range is all zero), or lo > hi.
//              Integer types draw uniformly from the integers in [ceil(lo), floor(hi)]
//              except zero.
//   LARGE      top = min(scale, T's largest finite value); each magnitude is uniform in
//              [top / 2, top], rounded to T, and the sign is random where T is signed. A
//              value at most top never rounds to infinity or NaN, so every element is at
//              most T's largest finite value in magnitude. Integer types draw from the
//              integers in [ceil(top / 2), floor(top)] and need scale >= 1.
//   SUBNORMAL  Every element is k * denorm_min with k uniform in [1, min / denorm_min - 1],
//              a random sign where T is signed: exactly T's subnormal values. Unsupported
//              for integers and for types without subnormals (fp8_e8m0). fp4_e2m1 has one
//              subnormal magnitude (0.5); the fp6 and fp8 formats a few.
//   SPECIAL    round(fraction * count) elements, at least one when fraction > 0, are the
//              non-finite value; the rest are FREE over [lo, hi]. Which elements is chosen
//              by selection sampling over the logical index order. Unsupported for integers
//              and for a value T lacks: NaN for fp4/fp6, infinity for fp4, fp6, fp8_e4m3,
//              fp8_e4m3_fnuz, fp8_e5m2_fnuz and fp8_e8m0.
template <typename T, typename Store>
FillResult fillTyped(const FillRecipe& recipe,
                     unsigned int seed,
                     [[maybe_unused]] std::size_t count,
                     Store&& store)
{
    // Some of these are read only by the floating-point or only by the integer instantiation.
    using Real [[maybe_unused]] = DrawType<T>;
    const bool isInteger = std::is_integral_v<T>;
    [[maybe_unused]] const bool isSigned = std::numeric_limits<T>::is_signed;
    const char* const typeName = dataTypeName<T>();

    std::mt19937 rng(seed);

    switch(recipe.kind)
    {
    case FillRecipe::Kind::ZEROS:
    {
        if(!isZeroElement(fromDouble<T>(0.0)))
        {
            return unsupportedKind(recipe, typeName, "the data type cannot represent zero");
        }
        auto gen = []() -> T { return fromDouble<T>(0.0); };
        store(gen);
        return FillResult::ok();
    }
    case FillRecipe::Kind::NONZERO:
    {
        if(!isDrawableRange(recipe.lo, recipe.hi))
        {
            return unsupportedKind(recipe, typeName, "needs finite lo <= hi with a finite hi - lo");
        }
        if constexpr(isInteger)
        {
            // The integers of [lo, hi] are tested against the type before any clamping, so a
            // range the type cannot reach is refused rather than collapsed onto its edge.
            const double ceilLo = std::ceil(static_cast<double>(recipe.lo));
            const double floorHi = std::floor(static_cast<double>(recipe.hi));
            const bool reachable
                = ceilLo <= floorHi
                  && floorHi >= static_cast<double>(std::numeric_limits<T>::lowest())
                  && ceilLo <= static_cast<double>(std::numeric_limits<T>::max());
            const auto first = clampToInteger<T>(ceilLo);
            const auto last = clampToInteger<T>(floorHi);
            const bool spansZero = first <= 0 && last >= 0;
            if(!reachable || (first == 0 && last == 0))
            {
                return unsupportedKind(recipe,
                                       typeName,
                                       "the range [" + std::to_string(recipe.lo) + ", "
                                           + std::to_string(recipe.hi)
                                           + "] holds no nonzero integer the type can store");
            }
            // Zero is skipped by drawing from one value fewer and shifting the draws that
            // reach zero up by one.
            std::uniform_int_distribution<std::int64_t> dist(first, spansZero ? last - 1 : last);
            auto gen = [&]() -> T {
                auto drawn = dist(rng);
                if(spansZero && drawn >= 0)
                {
                    ++drawn;
                }
                return static_cast<T>(drawn);
            };
            store(gen);
            return FillResult::ok();
        }
        else
        {
            const float farthest
                = std::fabs(recipe.lo) >= std::fabs(recipe.hi) ? recipe.lo : recipe.hi;
            const auto fallback = fromDouble<T>(static_cast<double>(farthest));
            if(isZeroElement(fallback))
            {
                return unsupportedKind(
                    recipe, typeName, "every value in [lo, hi] rounds to zero in this data type");
            }
            std::uniform_real_distribution<Real> dist(static_cast<Real>(recipe.lo),
                                                      static_cast<Real>(recipe.hi));
            auto gen = [&]() -> T {
                for(int attempt = 0; attempt < K_MAX_NONZERO_DRAWS; ++attempt)
                {
                    const auto drawn = fromDouble<T>(static_cast<double>(dist(rng)));
                    if(!isZeroElement(drawn))
                    {
                        return drawn;
                    }
                }
                return fallback;
            };
            store(gen);
            return FillResult::ok();
        }
    }
    case FillRecipe::Kind::LARGE:
    {
        if(!std::isfinite(recipe.scale) || recipe.scale <= 0.0f)
        {
            return unsupportedKind(recipe, typeName, "needs a finite scale > 0");
        }
        if constexpr(isInteger)
        {
            const auto scale = static_cast<double>(recipe.scale);
            const auto first = clampToInteger<T>(std::ceil(
                std::min(scale, static_cast<double>(std::numeric_limits<T>::max())) / 2.0));
            const auto last = clampToInteger<T>(std::floor(scale));
            if(first > last)
            {
                return unsupportedKind(recipe, typeName, "needs scale >= 1 for an integer type");
            }
            std::uniform_int_distribution<std::int64_t> dist(first, last);
            auto gen = [&]() -> T {
                const auto magnitude = dist(rng);
                const bool negative = isSigned && (rng() & 1U) != 0U;
                return static_cast<T>(negative ? -magnitude : magnitude);
            };
            store(gen);
            return FillResult::ok();
        }
        else
        {
            const double maxFinite = toDouble(std::numeric_limits<T>::max());
            const double top = std::min(static_cast<double>(recipe.scale), maxFinite);
            std::uniform_real_distribution<Real> dist(static_cast<Real>(top / 2.0),
                                                      static_cast<Real>(top));
            auto gen = [&]() -> T {
                const auto magnitude = static_cast<double>(dist(rng));
                const bool negative = isSigned && (rng() & 1U) != 0U;
                return fromDouble<T>(negative ? -magnitude : magnitude);
            };
            store(gen);
            return FillResult::ok();
        }
    }
    case FillRecipe::Kind::SUBNORMAL:
    {
        if constexpr(isInteger)
        {
            return unsupportedKind(recipe, typeName, "an integer type has no subnormal values");
        }
        else
        {
            // Subnormals are the multiples k * denorm_min below the smallest normal, so
            // drawing k draws a subnormal exactly, with no rounding to push it across.
            const double smallest = toDouble(std::numeric_limits<T>::denorm_min());
            const double smallestNormal = toDouble(std::numeric_limits<T>::min());
            if(smallest <= 0.0 || smallest >= smallestNormal)
            {
                return unsupportedKind(recipe, typeName, "the data type has no subnormal values");
            }
            const auto codes = static_cast<std::uint64_t>(smallestNormal / smallest);
            std::uniform_int_distribution<std::uint64_t> pick(1, codes - 1);
            auto gen = [&]() -> T {
                const auto magnitude = static_cast<double>(pick(rng)) * smallest;
                const bool negative = isSigned && (rng() & 1U) != 0U;
                return fromDouble<T>(negative ? -magnitude : magnitude);
            };
            store(gen);
            return FillResult::ok();
        }
    }
    case FillRecipe::Kind::SPECIAL:
    {
        if(!std::isfinite(recipe.fraction) || recipe.fraction < 0.0f || recipe.fraction > 1.0f)
        {
            return unsupportedKind(recipe, typeName, "needs fraction in [0, 1]");
        }
        if(!isDrawableRange(recipe.lo, recipe.hi))
        {
            return unsupportedKind(
                recipe, typeName, "needs finite lo <= hi with a finite hi - lo for the free share");
        }
        if constexpr(isInteger)
        {
            return unsupportedKind(
                recipe, typeName, "an integer type has no NaN or infinity to plant");
        }
        else
        {
            const auto planted = nonFiniteValue<T>(recipe.special);
            if(!planted.has_value())
            {
                return unsupportedKind(
                    recipe, typeName, "the data type cannot represent the value");
            }
            const std::uint64_t quota = specialQuota(recipe.fraction, count);
            std::uint64_t remaining = count;
            std::uint64_t pending = quota;
            std::uniform_real_distribution<Real> freeDist(static_cast<Real>(recipe.lo),
                                                          static_cast<Real>(recipe.hi));
            auto gen = [&]() -> T {
                bool plant = false;
                if(pending > 0)
                {
                    std::uniform_int_distribution<std::uint64_t> pick(0, remaining - 1);
                    plant = pick(rng) < pending;
                }
                --remaining;
                if(plant)
                {
                    --pending;
                    return *planted;
                }
                return fromDouble<T>(static_cast<double>(freeDist(rng)));
            };
            store(gen);
            return FillResult::ok();
        }
    }
    case FillRecipe::Kind::FREE:
    case FillRecipe::Kind::FIXED:
    default:
        break;
    }
    return unsupportedKind(recipe, typeName, "not a kind with a typed host generator");
}

// Writes `gen()` into every element of an element-addressable tensor, in logical index
// order, like Tensor<T>::fillWithRandomValues.
template <typename T, typename Gen>
void storeElements(ITensor& tensor, Gen& gen)
{
    tensor.markHostModified();
    for(auto* valuePtr : tensor)
    {
        *static_cast<T*>(valuePtr) = gen();
    }
}

// Writes `gen()` into every element of a bit-packed sub-byte tensor, element i at the bit slot
// its coordinates reach through the strides (the tensor is dense), as
// PackedSubByteTensor::fillTensorWithRandomValues does. That class has no element writer, so
// its packing is repeated here: LSB-first codes, neighbours' bits left untouched.
template <typename T, typename Gen>
void storePacked(ITensor& tensor, Gen& gen)
{
    using Traits = hipdnn_data_sdk::utilities::PackedElementTraits<T>;
    const std::size_t bitsPerElement = Traits::BITS_PER_ELEMENT;
    const std::size_t bitsPerByte = 8;

    const auto& dims = tensor.dims();
    const auto& strides = tensor.strides();
    const std::size_t count = tensor.elementCount();
    tensor.markHostModified();
    auto* host = static_cast<std::uint8_t*>(tensor.rawHostData());

    for(std::size_t i = 0; i < count; ++i)
    {
        std::size_t remaining = i;
        std::int64_t slot = 0;
        for(std::size_t axis = dims.size(); axis-- > 0;)
        {
            const auto extent = static_cast<std::size_t>(dims[axis]);
            slot += static_cast<std::int64_t>(remaining % extent) * strides[axis];
            remaining /= extent;
        }

        const std::uint8_t code = Traits::encode(static_cast<float>(gen()));
        const std::size_t bitOffset = static_cast<std::size_t>(slot) * bitsPerElement;
        const std::size_t byteIndex = bitOffset / bitsPerByte;
        const auto bitIndex = static_cast<unsigned>(bitOffset % bitsPerByte);

        const auto lowMask = static_cast<std::uint8_t>(Traits::CODE_MASK << bitIndex);
        const auto lowBits = static_cast<std::uint8_t>(code << bitIndex);
        host[byteIndex] = static_cast<std::uint8_t>((host[byteIndex] & ~lowMask) | lowBits);

        if(bitIndex > bitsPerByte - bitsPerElement)
        {
            const std::size_t highShift = bitsPerByte - bitIndex;
            const auto highMask = static_cast<std::uint8_t>(Traits::CODE_MASK >> highShift);
            const auto highBits = static_cast<std::uint8_t>(code >> highShift);
            host[byteIndex + 1]
                = static_cast<std::uint8_t>((host[byteIndex + 1] & ~highMask) | highBits);
        }
    }
}

// Fills `tensor` through fillTyped<T> when it stores T (unpacked or, for the sub-byte types,
// packed) and returns true with the outcome in `result`; returns false when it stores
// something else.
template <typename T>
bool tryFillAs(ITensor& tensor, const FillRecipe& recipe, unsigned int seed, FillResult& result)
{
    if(dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<T>*>(&tensor) != nullptr)
    {
        result = fillTyped<T>(recipe, seed, tensor.elementCount(), [&tensor](auto& gen) {
            storeElements<T>(tensor, gen);
        });
        return true;
    }
    if constexpr(!std::is_void_v<typename PackedTensorFor<T>::type>)
    {
        if(dynamic_cast<typename PackedTensorFor<T>::type*>(&tensor) != nullptr)
        {
            result = fillTyped<T>(recipe, seed, tensor.elementCount(), [&tensor](auto& gen) {
                storePacked<T>(tensor, gen);
            });
            return true;
        }
    }
    return false;
}

FillResult fillOnHost(ITensor& tensor, const FillRecipe& recipe, unsigned int seed)
{
    FillResult result;
    if(tryFillAs<float>(tensor, recipe, seed, result)
       || tryFillAs<double>(tensor, recipe, seed, result)
       || tryFillAs<types::half>(tensor, recipe, seed, result)
       || tryFillAs<types::bfloat16>(tensor, recipe, seed, result)
       || tryFillAs<types::fp8_e4m3>(tensor, recipe, seed, result)
       || tryFillAs<types::fp8_e5m2>(tensor, recipe, seed, result)
       || tryFillAs<types::fp8_e4m3_fnuz>(tensor, recipe, seed, result)
       || tryFillAs<types::fp8_e5m2_fnuz>(tensor, recipe, seed, result)
       || tryFillAs<types::fp8_e8m0>(tensor, recipe, seed, result)
       || tryFillAs<types::fp4_e2m1>(tensor, recipe, seed, result)
       || tryFillAs<types::fp6_e2m3>(tensor, recipe, seed, result)
       || tryFillAs<types::fp6_e3m2>(tensor, recipe, seed, result)
       || tryFillAs<std::int8_t>(tensor, recipe, seed, result)
       || tryFillAs<std::uint8_t>(tensor, recipe, seed, result)
       || tryFillAs<std::int32_t>(tensor, recipe, seed, result)
       || tryFillAs<std::int64_t>(tensor, recipe, seed, result))
    {
        return result;
    }

    if(recipe.kind == FillRecipe::Kind::ZEROS)
    {
        // Whatever the element type, ITensor can write a zero.
        tensor.fillTensorWithValue(0.0f);
        return FillResult::ok();
    }
    return FillResult::unsupported(std::string("fill kind \"")
                                   + InputFillRecipes::kindToString(recipe.kind)
                                   + "\": the tensor's element type has no host generator for it");
}

// ── VALUES fill ─────────────────────────────────────────────────────────────

bool holdsSubByteElements(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    return dataType == DataType::FP4_E2M1 || dataType == DataType::FP6_E2M3
           || dataType == DataType::FP6_E3M2 || dataType == DataType::INT4;
}

// Integer element types take only exact integers in their range; BOOLEAN takes 0 or 1.
// Floating-point element types take any number, rounded to the element type.
template <typename T, typename V>
std::optional<std::string> toElement([[maybe_unused]] V number, [[maybe_unused]] T& element)
{
    if constexpr(std::is_integral_v<T> && std::is_floating_point_v<V>)
    {
        return std::string("the list holds a non-integer number, so it cannot fill an "
                           "integer tensor");
    }
    else if constexpr(std::is_integral_v<T>)
    {
        bool inRange = false;
        if constexpr(std::is_same_v<T, bool>)
        {
            inRange = number == 0 || number == 1;
        }
        else if constexpr(std::is_same_v<T, int64_t>)
        {
            inRange = true;
        }
        else
        {
            inRange = number >= static_cast<int64_t>(std::numeric_limits<T>::min())
                      && number <= static_cast<int64_t>(std::numeric_limits<T>::max());
        }
        if(!inRange)
        {
            return std::to_string(number) + " is out of range for the tensor's data type";
        }
        element = static_cast<T>(number);
        return std::nullopt;
    }
    else
    {
        element = static_cast<T>(number);
        return std::nullopt;
    }
}

template <typename T, typename V>
std::optional<std::string> writeValues(hipdnn_data_sdk::utilities::TensorBase<T>& tensor,
                                       const std::vector<V>& numbers,
                                       bool repeat)
{
    const std::size_t count = tensor.elementCount();
    if(!repeat && numbers.size() != count)
    {
        return "repeat false needs one value per element: " + std::to_string(numbers.size())
               + " values for " + std::to_string(count) + " elements";
    }
    if(numbers.size() > count)
    {
        return "the list is longer than the tensor: " + std::to_string(numbers.size())
               + " values for " + std::to_string(count) + " elements";
    }

    std::vector<T> elements(numbers.size());
    for(std::size_t i = 0; i < numbers.size(); ++i)
    {
        T element{};
        if(auto why = toElement(numbers[i], element))
        {
            return "values[" + std::to_string(i) + "]: " + *why;
        }
        elements[i] = element;
    }

    std::size_t i = 0;
    for(auto it = tensor.begin(); it != tensor.end(); ++it, ++i)
    {
        *static_cast<T*>(*it) = elements[i % elements.size()];
    }
    tensor.markHostModified();
    return std::nullopt;
}

// Fills `tensor` when its element type is T; nullopt when it is not.
template <typename T>
std::optional<FillResult> fillValuesAs(hipdnn_data_sdk::utilities::ITensor& tensor,
                                       const FillRecipe& recipe)
{
    auto* typed = dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<T>*>(&tensor);
    if(typed == nullptr)
    {
        return std::nullopt;
    }
    const auto why = std::visit(
        [&](const auto& numbers) { return writeValues(*typed, numbers, recipe.repeat); },
        *recipe.list);
    return why.has_value() ? FillResult::invalid(*why) : FillResult::ok();
}

template <typename... Ts>
FillResult fillValuesAsAnyOf(hipdnn_data_sdk::utilities::ITensor& tensor, const FillRecipe& recipe)
{
    std::optional<FillResult> result;
    static_cast<void>((... || (result = fillValuesAs<Ts>(tensor, recipe)).has_value()));
    return result.has_value() ? *result
                              : FillResult::invalid("kind values cannot fill this element type");
}

FillResult fillValues(hipdnn_data_sdk::utilities::ITensor& tensor,
                      const FillRecipe& recipe,
                      std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> dataType)
{
    if(recipe.list == nullptr)
    {
        return FillResult::invalid("kind values has no list");
    }
    // INT4 tensors hold uint8_t elements, so only the data type tells them apart.
    if(!dataType.has_value())
    {
        return FillResult::invalid("kind values needs the tensor's data type");
    }
    if(holdsSubByteElements(*dataType))
    {
        return FillResult::invalid(std::string("kind values cannot fill a ")
                                   + EnumNameDataType(*dataType) + " tensor");
    }

    return fillValuesAsAnyOf<float,
                             double,
                             types::half,
                             types::bfloat16,
                             types::fp8_e4m3,
                             types::fp8_e5m2,
                             types::fp8_e4m3_fnuz,
                             types::fp8_e5m2_fnuz,
                             types::fp8_e8m0,
                             int8_t,
                             uint8_t,
                             int32_t,
                             int64_t,
                             bool>(tensor, recipe);
}

// ── Fill dispatch ───────────────────────────────────────────────────────────

// The byte a host fill sets every byte of a non-packed input's allocation to before it writes the
// logical elements, so a gap holds a NaN (an integer's all-ones pattern for an integer type) and
// not whatever malloc returned. An engine that reads a gap then produces a visibly wrong output
// on every run instead of an intermittent one; the reference never reads a gap, so only the
// engine can show it.
//
// 0xFF is a NaN of float, double, half, bfloat16, the OCP fp8 formats and e8m0. The FNUZ fp8
// formats have the single NaN 0x80, and 0xFF is a finite value there, so they get 0x80. Integers
// get 0xFF. No byte is poisoned for bool, and none for the bit-packed sub-byte types, whose
// allocation is a bit stream rather than an array of elements.
std::optional<std::uint8_t>
    gapPoisonByteFor(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(dataType)
    {
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ:
        return std::uint8_t{0x80};
    case DataType::FLOAT:
    case DataType::HALF:
    case DataType::BFLOAT16:
    case DataType::DOUBLE:
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E8M0:
    case DataType::UINT8:
    case DataType::INT32:
    case DataType::INT8:
    case DataType::INT64:
        return std::uint8_t{0xFF};
    case DataType::FP4_E2M1:
    case DataType::FP6_E2M3:
    case DataType::FP6_E3M2:
    case DataType::INT4:
    case DataType::BOOLEAN:
    case DataType::UNSET:
    default:
        return std::nullopt;
    }
}

// Sets every byte of `tensor`'s host allocation to `poison` when the tensor is not packed, that
// is, when its allocation holds gap elements beside the logical ones. A packed tensor has no gap,
// and its elements are all written anyway.
void poisonGaps(ITensor& tensor, std::optional<std::uint8_t> poison)
{
    if(!poison.has_value() || tensor.elementSpace() == tensor.elementCount())
    {
        return;
    }
    std::memset(tensor.rawHostData(), *poison, tensor.elementSpace() * tensor.elementSize());
    tensor.markHostModified();
}

// A tensor a host fill writes is poisoned first (poisonGaps). The device fill is not: it draws
// over the whole allocation, so it leaves no gap to poison.
FillResult fill(hipdnn_data_sdk::utilities::ITensor& tensor,
                const FillRecipe& recipe,
                unsigned int seed,
                DeviceInputFiller* device,
                std::optional<std::uint8_t> gapPoison,
                std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> dataType)
{
    switch(recipe.kind)
    {
    case FillRecipe::Kind::FREE:
        if(device != nullptr && device->tryFill(tensor, recipe, seed))
        {
            return FillResult::ok(/*deviceFilled=*/1);
        }
        poisonGaps(tensor, gapPoison);
        tensor.fillTensorWithRandomValues(recipe.lo, recipe.hi, seed);
        return FillResult::ok();
    case FillRecipe::Kind::FIXED:
        poisonGaps(tensor, gapPoison);
        tensor.fillTensorWithValue(recipe.value);
        return FillResult::ok();
    // On the device when the filler takes the tensor, else from the host generators, which
    // also decide whether the tensor's data type can hold the kind at all.
    case FillRecipe::Kind::NONZERO:
    case FillRecipe::Kind::ZEROS:
    case FillRecipe::Kind::LARGE:
    case FillRecipe::Kind::SUBNORMAL:
    case FillRecipe::Kind::SPECIAL:
        if(device != nullptr && device->tryFill(tensor, recipe, seed))
        {
            return FillResult::ok(/*deviceFilled=*/1);
        }
        poisonGaps(tensor, gapPoison);
        return fillOnHost(tensor, recipe, seed);
    case FillRecipe::Kind::VALUES:
        poisonGaps(tensor, gapPoison);
        return fillValues(tensor, recipe, dataType);
    default:
        return FillResult::unsupported("unknown FillRecipe kind");
    }
}

// ── Per-op init defaults ────────────────────────────────────────────────────
// Each function sets defaults for one node type via recipes.setDefault().
// setDefault uses try_emplace — if the test already set() a uid, the default
// is silently skipped.

// ── Batchnorm ────────────────────────────────────────────────────────────────

void setBatchnormInferenceInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                       InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->mean_tensor_uid(), FillRecipe::free(-0.1f, 0.1f));
    recipes.setDefault(a->inv_variance_tensor_uid(), FillRecipe::free(0.5f, 1.5f));
}

void setBatchnormInferenceVarianceInitDefaults(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node, InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->mean_tensor_uid(), FillRecipe::free(-0.1f, 0.1f));
    recipes.setDefault(a->variance_tensor_uid(), FillRecipe::free(0.5f, 1.5f));
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
}

void setBatchnormTrainingInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BatchnormAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
    recipes.setDefault(a->prev_running_mean_tensor_uid(), FillRecipe::free(-0.1f, 0.1f));
    recipes.setDefault(a->prev_running_variance_tensor_uid(), FillRecipe::free(0.5f, 1.5f));
    recipes.setDefault(a->momentum_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
}

void setBatchnormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BatchnormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->mean_tensor_uid(), FillRecipe::free(-0.1f, 0.1f));
    recipes.setDefault(a->inv_variance_tensor_uid(), FillRecipe::free(0.5f, 1.5f));
}

// ── LayerNorm ────────────────────────────────────────────────────────────────

void setLayernormInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                              InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_LayernormAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
}

void setLayernormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_LayernormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->mean_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
    recipes.setDefault(a->inv_variance_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
}

// ── RMSNorm ──────────────────────────────────────────────────────────────────

void setRmsnormInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                            InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_RMSNormAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
}

void setRmsnormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                    InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_RMSNormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->inv_rms_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
}

// ── Block-scale quantization ─────────────────────────────────────────────────

// Largest magnitude a recipe can produce among finite values; a non-finite SPECIAL value is
// outside any scale, so only its free share counts.
float recipeMagnitude(const FillRecipe& recipe)
{
    switch(recipe.kind)
    {
    case FillRecipe::Kind::FIXED:
        return std::fabs(recipe.value);
    case FillRecipe::Kind::LARGE:
        return recipe.scale;
    case FillRecipe::Kind::VALUES:
        if(recipe.list == nullptr)
        {
            return 0.0f;
        }
        return std::visit(
            [](const auto& numbers) {
                float magnitude = 0.0f;
                for(const auto number : numbers)
                {
                    magnitude = std::max(magnitude, std::fabs(static_cast<float>(number)));
                }
                return magnitude;
            },
            *recipe.list);
    case FillRecipe::Kind::FREE:
    case FillRecipe::Kind::NONZERO:
    case FillRecipe::Kind::SPECIAL:
        return std::max(std::fabs(recipe.lo), std::fabs(recipe.hi));
    case FillRecipe::Kind::ZEROS:
    case FillRecipe::Kind::SUBNORMAL:
    default:
        // Zero, or below every normal value: smaller than any amax that matters to a scale.
        return 0.0f;
    }
    return 0.0f;
}

void setBlockScaleDequantizeInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BlockScaleDequantizeAttributes();
    if(a == nullptr)
    {
        return;
    }

    // Scale is normalized against the operand range in effect (its data-type
    // default or a test override), as OCP MX does: scale ~ 2^-floor(log2(amax)) for
    // amax >= 1. The dequantized block then peaks between 2 and 4 (FP4/FP6:
    // 3 to 3.75), so products stay within FP16 range; an operand with
    // amax < 1 keeps e = 0 and peaks at 2 * amax. UE8M0 has no mantissa bits,
    // so the [0.5, 2] * 2^-e draw discretizes to three powers of two. An
    // operand in [-1, 1] gives e = 0, i.e. [0.5, 2].
    const float amax = recipeMagnitude(recipes.fill(a->x_tensor_uid()));
    const int e = amax >= 1.0f ? std::ilogb(amax) : 0;
    recipes.setDefault(a->scale_tensor_uid(),
                       FillRecipe::free(std::ldexp(0.5f, -e), std::ldexp(2.0f, -e)));
}

// ── SDPA ─────────────────────────────────────────────────────────────────────

void setSdpaForwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_SdpaAttributes();
    if(a == nullptr)
    {
        return;
    }

    recipes.setDefault(a->scale_tensor_uid(), FillRecipe::free(0.1f, 1.0f));
}

void setSdpaBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                 InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_SdpaBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }

    recipes.setDefault(a->scale_tensor_uid(), FillRecipe::free(0.1f, 1.0f));
    recipes.setDefault(a->dropout_scale_tensor_uid(), FillRecipe::free(0.1f, 1.0f));
    recipes.setDefault(a->dropout_scale_inv_tensor_uid(), FillRecipe::free(0.1f, 1.0f));

    // Overwritten by fillInputs() with the forward of Q, K and V. The fill stands only
    // where the CPU forward rejects the shapes, and the backward reference fails those.
    recipes.setDefault(a->o_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
    recipes.setDefault(a->stats_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
}

// The SDPA backward attributes of a node whose O and stats inputs fillInputs() derives
// from its Q, K and V after the fill; null for every other node.
const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes*
    derivesInputsAfterFill(const hipdnn_flatbuffers_sdk::data_objects::Node& node)
{
    return node.attributes_as_SdpaBackwardAttributes();
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

bool applyDefaultFills(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                       InputFillRecipes& recipes)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    switch(node.attributes_type())
    {
    case NA::BatchnormInferenceAttributes:
        setBatchnormInferenceInitDefaults(node, recipes);
        return true;
    case NA::BatchnormInferenceAttributesVarianceExt:
        setBatchnormInferenceVarianceInitDefaults(node, recipes);
        return true;
    case NA::BatchnormAttributes:
        setBatchnormTrainingInitDefaults(node, recipes);
        return true;
    case NA::BatchnormBackwardAttributes:
        setBatchnormBackwardInitDefaults(node, recipes);
        return true;
    case NA::LayernormAttributes:
        setLayernormInitDefaults(node, recipes);
        return true;
    case NA::LayernormBackwardAttributes:
        setLayernormBackwardInitDefaults(node, recipes);
        return true;
    case NA::RMSNormAttributes:
        setRmsnormInitDefaults(node, recipes);
        return true;
    case NA::RMSNormBackwardAttributes:
        setRmsnormBackwardInitDefaults(node, recipes);
        return true;
    case NA::BlockScaleDequantizeAttributes:
        setBlockScaleDequantizeInitDefaults(node, recipes);
        return true;
    case NA::SdpaAttributes:
        setSdpaForwardInitDefaults(node, recipes);
        return true;
    case NA::SdpaBackwardAttributes:
        setSdpaBackwardInitDefaults(node, recipes);
        return true;
    // Ops whose inputs need no special init; they get defaultFillFor(dtype).
    case NA::PointwiseAttributes:
    case NA::ConvolutionFwdAttributes:
    case NA::ConvolutionBwdAttributes:
    case NA::ConvolutionWrwAttributes:
    case NA::MatmulAttributes:
    case NA::ReductionAttributes:
    case NA::ResampleFwdAttributes:
    case NA::ResampleBwdAttributes:
    case NA::BlockScaleQuantizeAttributes:
    case NA::CustomOpAttributes:
    case NA::MoeGroupedMatmulAttributes:
    case NA::MoeGroupedMatmulBwdAttributes:
    case NA::NONE:
        return true;
    default:
        return false;
    }
}

} // anonymous namespace

#if defined(USE_ROCRAND)
using RocRandGenerator = hipdnn_gpu_ref::common::RocRandGenerator;

namespace
{

// The kernel's parameters for a recipe fillTyped<T> has accepted. Each is derived from the
// recipe and T as the host generator derives it, so the two paths agree on what the kind makes.
template <typename T>
hipdnn_gpu_ref::common::FillKindArgs
    deviceFillArgs(const FillRecipe& recipe, unsigned int seed, std::size_t count)
{
    using hipdnn_gpu_ref::common::FillKind;

    hipdnn_gpu_ref::common::FillKindArgs args{};
    args.seed = seed;
    args.lo = static_cast<double>(recipe.lo);
    args.hi = static_cast<double>(recipe.hi);
    switch(recipe.kind)
    {
    case FillRecipe::Kind::ZEROS:
        args.kind = static_cast<int>(FillKind::ZEROS);
        break;
    case FillRecipe::Kind::NONZERO:
        args.kind = static_cast<int>(FillKind::NONZERO);
        args.fallback = static_cast<double>(
            std::fabs(recipe.lo) >= std::fabs(recipe.hi) ? recipe.lo : recipe.hi);
        break;
    case FillRecipe::Kind::LARGE:
    {
        args.kind = static_cast<int>(FillKind::LARGE);
        const double top
            = std::min(static_cast<double>(recipe.scale), toDouble(std::numeric_limits<T>::max()));
        args.lo = top / 2.0;
        args.hi = top;
        break;
    }
    case FillRecipe::Kind::SUBNORMAL:
    {
        args.kind = static_cast<int>(FillKind::SUBNORMAL);
        const double smallest = toDouble(std::numeric_limits<T>::denorm_min());
        const double smallestNormal = toDouble(std::numeric_limits<T>::min());
        args.maxSubnormalCode = static_cast<std::uint64_t>(smallestNormal / smallest) - 1;
        break;
    }
    case FillRecipe::Kind::SPECIAL:
    {
        args.kind = static_cast<int>(FillKind::SPECIAL);
        static_assert(sizeof(T) <= sizeof(args.plantedBits));
        const T planted = nonFiniteValue<T>(recipe.special).value();
        // The low bytes of the little-endian word the kernel stores from.
        std::memcpy(&args.plantedBits, &planted, sizeof(T));
        args.quota = specialQuota(recipe.fraction, count);
        break;
    }
    case FillRecipe::Kind::FREE:
    case FillRecipe::Kind::FIXED:
    default:
        break;
    }
    return args;
}

} // anonymous namespace
#endif

struct DeviceInputFiller::Impl
{
#if defined(USE_ROCRAND)
    std::unique_ptr<RocRandGenerator> generator;
    bool fillPending = false;

    // Starts the fill when `tensor` is a T tensor; creates the generator on first use,
    // so a tensor of a type rocRAND does not fill never costs one.
    template <class T>
    bool fillOnDevice(hipdnn_data_sdk::utilities::ITensor& tensor,
                      const FillRecipe& recipe,
                      unsigned int seed)
    {
        auto* typed = dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<T>*>(&tensor);
        if(typed == nullptr)
        {
            return false;
        }

        if(generator == nullptr)
        {
            generator = std::make_unique<RocRandGenerator>(ROCRAND_RNG_PSEUDO_DEFAULT);
        }

        // Set before the launch: a fill that throws partway may already have queued work.
        fillPending = true;
        hipdnn_gpu_ref::common::gpu_fp_reference_tensor::gpuFillWithRandomValues<T>(
            *typed,
            static_cast<T>(recipe.lo),
            static_cast<T>(recipe.hi),
            seed,
            *generator,
            /*synchronize=*/false);
        return true;
    }

    // Starts the fill of a NONZERO, ZEROS, LARGE, SUBNORMAL or SPECIAL recipe when `tensor` is
    // a T tensor the kernel can write and the host generator accepts the recipe for T; returns
    // false otherwise, leaving the tensor untouched, so the host fill decides and reports it.
    template <class T>
    bool fillKindOnDevice(hipdnn_data_sdk::utilities::ITensor& tensor,
                          const FillRecipe& recipe,
                          unsigned int seed)
    {
        auto* typed = dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<T>*>(&tensor);
        if(typed == nullptr)
        {
            return false;
        }

        // The kernel addresses elements through dims and strides only.
        constexpr auto MAX_RANK
            = hipdnn_gpu_ref::common::gpu_fp_reference_tensor::FILL_KIND_MAX_RANK;
        if(tensor.raggedIterationInfo().has_value() || tensor.dims().size() > MAX_RANK)
        {
            return false;
        }

        // The host generator's checks, run without drawing: a store that writes nothing.
        const auto count = tensor.elementCount();
        if(!fillTyped<T>(recipe, seed, count, [](auto& /*gen*/) {}).filled)
        {
            return false;
        }

        if(generator == nullptr)
        {
            generator = std::make_unique<RocRandGenerator>(ROCRAND_RNG_PSEUDO_DEFAULT);
        }

        // Set before the launch: a fill that throws partway may already have queued work.
        fillPending = true;
        hipdnn_gpu_ref::common::gpu_fp_reference_tensor::gpuFillKind<T>(
            *typed, deviceFillArgs<T>(recipe, seed, count), *generator);
        return true;
    }
#endif
};

DeviceInputFiller::DeviceInputFiller()
    : _impl(std::make_unique<Impl>())
{
}

DeviceInputFiller::~DeviceInputFiller()
{
    // The generator is about to be destroyed; a fill must not still be using it.
    try
    {
        waitForFills();
    }
    catch(...) // NOLINT(bugprone-empty-catch): nothing useful to do about it here
    {
    }
}

bool hostInputFillForced()
{
    const auto value = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_HOST_INPUT_FILL");
    return !value.empty() && value != "0";
}

bool DeviceInputFiller::isSupported()
{
#if defined(USE_ROCRAND)
    return true;
#else
    return false;
#endif
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static): uses _impl with rocRAND
bool DeviceInputFiller::tryFill([[maybe_unused]] hipdnn_data_sdk::utilities::ITensor& tensor,
                                [[maybe_unused]] const FillRecipe& recipe,
                                [[maybe_unused]] unsigned int seed)
{
#if defined(USE_ROCRAND)
    // The nonzero, zeros, large, subnormal and special kinds are drawn with rocRAND and then
    // shaped by a kernel; FIXED is a constant fill, cheaper on the host than a launch, and
    // VALUES is an explicit list, written on the host.
    if(recipe.kind == FillRecipe::Kind::FIXED || recipe.kind == FillRecipe::Kind::VALUES)
    {
        return false;
    }
    if(tensor.elementSpace() < minElements())
    {
        return false;
    }

    try
    {
        auto& impl = *_impl;
        if(recipe.kind == FillRecipe::Kind::FREE)
        {
            return impl.fillOnDevice<float>(tensor, recipe, seed)
                   || impl.fillOnDevice<types::half>(tensor, recipe, seed)
                   || impl.fillOnDevice<types::bfloat16>(tensor, recipe, seed)
                   || impl.fillOnDevice<double>(tensor, recipe, seed);
        }
        return impl.fillKindOnDevice<float>(tensor, recipe, seed)
               || impl.fillKindOnDevice<types::half>(tensor, recipe, seed)
               || impl.fillKindOnDevice<types::bfloat16>(tensor, recipe, seed)
               || impl.fillKindOnDevice<types::fp8_e4m3>(tensor, recipe, seed)
               || impl.fillKindOnDevice<types::fp8_e5m2>(tensor, recipe, seed)
               || impl.fillKindOnDevice<types::fp8_e4m3_fnuz>(tensor, recipe, seed)
               || impl.fillKindOnDevice<types::fp8_e5m2_fnuz>(tensor, recipe, seed);
    }
    catch(const std::exception& e)
    {
        // Generator creation, the kernels' compile, scratch allocation and the launches are
        // all device work; none of it is a fault in the graph under test.
        throw DeviceInputError(std::string("device input fill failed: ") + e.what());
    }
#else
    return false;
#endif
}

void DeviceInputFiller::waitForFills()
{
#if defined(USE_ROCRAND)
    if(!_impl->fillPending)
    {
        return;
    }
    _impl->fillPending = false;

    checkDeviceCall<DeviceInputError>(hipDeviceSynchronize(), "device input fill");
#endif
}

namespace
{

// Waits for the device fills on every way out of fillInputs(), including an exception
// from a fill that threw partway: the tensors are about to be dropped, and fills
// already queued must not still be writing into memory being freed. Errors are
// swallowed here, since the one already propagating is the one worth reporting; the
// normal path waits explicitly and sees them.
class WaitForFillsOnExit
{
public:
    explicit WaitForFillsOnExit(DeviceInputFiller* device)
        : _device(device)
    {
    }

    ~WaitForFillsOnExit()
    {
        if(_device == nullptr)
        {
            return;
        }
        try
        {
            _device->waitForFills();
        }
        catch(...) // NOLINT(bugprone-empty-catch): see above
        {
        }
    }

    WaitForFillsOnExit(const WaitForFillsOnExit&) = delete;
    WaitForFillsOnExit& operator=(const WaitForFillsOnExit&) = delete;
    WaitForFillsOnExit(WaitForFillsOnExit&&) = delete;
    WaitForFillsOnExit& operator=(WaitForFillsOnExit&&) = delete;

private:
    DeviceInputFiller* _device;
};

} // anonymous namespace

FillRecipe defaultFillFor(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    namespace types = hipdnn_data_sdk::types;

    switch(dataType)
    {
    case DataType::FLOAT:
    case DataType::HALF:
    case DataType::BFLOAT16:
    case DataType::DOUBLE:
        return FillRecipe{};
    // Few enough codes that a uniform draw over the full range reaches all of them;
    // [-1, 1] would only reach 3 of the 8 FP4 magnitudes.
    case DataType::FP4_E2M1:
        return symmetricFullRange<types::fp4_e2m1>();
    case DataType::FP6_E2M3:
        return symmetricFullRange<types::fp6_e2m3>();
    case DataType::FP6_E3M2:
        return symmetricFullRange<types::fp6_e3m2>();
    // FP8 codes are spread logarithmically, so no uniform range reaches most of
    // them: a wide range leaves the small exponents empty. Full coverage needs a
    // log-uniform or code-uniform distribution, which ITensor cannot draw yet.
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ:
        return FillRecipe{};
    // Power-of-two scale; [0.5, 2] discretizes to {0.5, 1, 2}.
    case DataType::FP8_E8M0:
        return FillRecipe::free(0.5f, 2.0f);
    // Integer draws from [-1, 1] collapse to a few values. A meaningful range is
    // op-specific (indices, offsets, masks), so it belongs in per-op defaults.
    case DataType::UINT8:
    case DataType::INT32:
    case DataType::INT8:
    case DataType::INT4:
    case DataType::INT64:
    case DataType::BOOLEAN:
        return FillRecipe{};
    case DataType::UNSET:
    default:
        throw std::invalid_argument("defaultFillFor: no fill default for data type "
                                    + std::string(EnumNameDataType(dataType)));
    }
}

bool fillDerivesInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph)
{
    if(graph.nodes() == nullptr)
    {
        return false;
    }
    for(const auto* node : *graph.nodes())
    {
        if(derivesInputsAfterFill(*node) != nullptr)
        {
            return true;
        }
    }
    return false;
}

FillResult fillInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                      InputTensorMap& inputs,
                      const std::vector<int64_t>& ownedUids,
                      InputFillRecipes& recipes,
                      DeviceInputFiller* device,
                      const std::vector<TensorAlias>& aliases)
{
    std::unordered_set<int64_t> viewUids;
    std::unordered_set<int64_t> ownerUids;
    if(!aliases.empty())
    {
        if(fillDerivesInputs(graph))
        {
            throw std::invalid_argument("fillInputs: aliases need every input of a graph whose "
                                        "fill derives inputs from other inputs");
        }
        const std::unordered_set<int64_t> owned(ownedUids.begin(), ownedUids.end());
        for(const auto& alias : aliases)
        {
            if(owned.count(alias.ofUid) == 0 || inputs.count(alias.ofUid) == 0)
            {
                throw std::invalid_argument("fillInputs: " + tensor_alias_detail::describe(alias)
                                            + ": the owner has no input tensor to fill");
            }
            if(recipes.isSet(alias.uid))
            {
                return FillResult::invalid(
                    "uid " + std::to_string(alias.uid) + ": a fill recipe is set on a view of uid "
                    + std::to_string(alias.ofUid) + ", and a view is never filled");
            }
            viewUids.insert(alias.uid);
            ownerUids.insert(alias.ofUid);
        }
    }

    // Dtype defaults sit below test and per-op recipes. Only those that differ
    // from the generic FREE[-1, 1] are recorded, so meta.inputs stays unchanged
    // for float graphs and names the range for narrow formats. Registered
    // before the per-op pass so a default derived from another input's range
    // (block-scale dequantize's scale) sees that input's final range.
    const auto dataTypes = collectDataTypes(graph);
    for(const int64_t uid : ownedUids)
    {
        const auto dataType = dataTypes.find(uid);
        if(dataType == dataTypes.end())
        {
            continue;
        }
        const auto recipe = defaultFillFor(dataType->second);
        if(recipe != FillRecipe{})
        {
            recipes.setTypeDefault(uid, recipe);
        }
    }

    for(flatbuffers::uoffset_t i = 0; i < graph.nodes()->size(); ++i)
    {
        const auto& node = *graph.nodes()->Get(i);
        if(!applyDefaultFills(node, recipes))
        {
            const auto* name = node.name();
            return FillResult::unsupported(
                "no input fill registered for op "
                + std::string(name != nullptr ? name->c_str() : "(unnamed)"));
        }
    }

    // Sort so the rng sequence is deterministic regardless of discovery order.
    auto sortedUids = ownedUids;
    std::sort(sortedUids.begin(), sortedUids.end());

    std::mt19937 rng(recipes.globalSeed());

    const WaitForFillsOnExit waitOnExit(device);

    std::size_t deviceFilled = 0;
    std::unordered_map<int64_t, unsigned int> seeds;
    for(const int64_t uid : sortedUids)
    {
        const unsigned int seed
            = recipes.resolveSeed(uid).value_or(static_cast<unsigned int>(rng()));
        seeds.emplace(uid, seed);
        if(viewUids.count(uid) != 0)
        {
            continue;
        }

        // Views can read the gaps between an owner's elements, so the gaps hold zeros.
        // The owner is filled on the host, where the zeros are written, and its gaps are not
        // poisoned.
        auto& tensor = *inputs.at(uid);
        const bool isOwner = ownerUids.count(uid) != 0;
        if(isOwner)
        {
            std::memset(tensor.rawHostData(), 0, tensor.elementSpace() * tensor.elementSize());
        }
        const auto dataType = dataTypes.find(uid);
        const auto gapPoison = (!isOwner && dataType != dataTypes.end())
                                   ? gapPoisonByteFor(dataType->second)
                                   : std::optional<std::uint8_t>{};
        auto fillResult
            = fill(tensor,
                   recipes.fill(uid),
                   seed,
                   isOwner ? nullptr : device,
                   gapPoison,
                   dataType != dataTypes.end() ? std::optional(dataType->second) : std::nullopt);
        if(isOwner)
        {
            tensor.markHostModified();
        }
        if(!fillResult.filled)
        {
            auto reason = "uid " + std::to_string(uid) + ": " + fillResult.reason;
            return fillResult.status == FillResult::Status::INVALID
                       ? FillResult::invalid(std::move(reason))
                       : FillResult::unsupported(std::move(reason));
        }
        deviceFilled += fillResult.deviceFilled;
    }

    // One wait for every device fill, instead of one per tensor.
    if(device != nullptr)
    {
        device->waitForFills();
    }

    // O and the LSE of an SDPA backward node are outputs of a forward, not free inputs:
    // once Q, K and V hold their values, recompute them. Done after the waits, since
    // the forward reads Q, K and V on the host.
    for(flatbuffers::uoffset_t i = 0; i < graph.nodes()->size(); ++i)
    {
        const auto& node = *graph.nodes()->Get(i);
        const auto* sdpaBackward = derivesInputsAfterFill(node);
        if(sdpaBackward == nullptr)
        {
            continue;
        }
        if(auto why = deriveSdpaBackwardForwardOutputs(graph, *sdpaBackward, inputs, ownedUids))
        {
            const auto* name = node.name();
            return FillResult::unsupported(
                "cannot derive O and stats of SDPA backward node "
                + std::string(name != nullptr ? name->c_str() : "(unnamed)")
                + " from its Q, K and V: " + *why);
        }
    }

    // The index of a max-pooling resample backward node is the argmax output of a
    // forward, not a free input: recompute it from a scratch x seeded like the index.
    for(flatbuffers::uoffset_t i = 0; i < graph.nodes()->size(); ++i)
    {
        const auto& node = *graph.nodes()->Get(i);
        const auto* resampleBackward = node.attributes_as_ResampleBwdAttributes();
        if(resampleBackward == nullptr || !resampleBackward->index_tensor_uid().has_value())
        {
            continue;
        }
        const auto seed = seeds.find(*resampleBackward->index_tensor_uid());
        if(seed == seeds.end())
        {
            continue;
        }
        if(auto why
           = deriveResampleBackwardIndex(graph, *resampleBackward, inputs, ownedUids, seed->second))
        {
            const auto* name = node.name();
            return FillResult::unsupported(
                "cannot derive the index of resample backward node "
                + std::string(name != nullptr ? name->c_str() : "(unnamed)")
                + " from a forward: " + *why);
        }
    }
    return FillResult::ok(deviceFilled);
}

} // namespace hipdnn_integration_tests
