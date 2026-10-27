// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstring>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace hipdnn_test_sdk::utilities
{

/// Opt-in non-finite handling for the allClose validator (CpuFpReferenceValidation) and the
/// RMS validator (CpuFpReferenceMiopenRmsValidation). Default constructed, the mask is off and
/// the validators behave exactly as they did before it existed: any NaN or Inf fails.
///
/// With the mask on, an element position is classified by its reference and implementation
/// values:
///   - NaN and NaN match, whatever the payloads and signs. The position is excluded from the
///     tolerance math.
///   - Inf and Inf of the same sign match. The position is excluded from the tolerance math.
///   - NaN against a finite value, a finite value against Inf, +Inf against -Inf, and NaN
///     against Inf all fail.
///   - Finite against finite is graded as usual: by atol + rtol * |reference| for allClose,
///     by the relative RMS error for the RMS validator.
///   - A reference with no finite element at all (every element NaN or Inf) FAILS. With the
///     finite part empty there is nothing to grade, and a vacuous pass would hide a run that
///     produced garbage on both sides.
///
/// For a ragged reference (RFC 0014) a NaN reference value already marks an undefined position
/// and is skipped before the mask is consulted, so NaN-NaN matching never applies there; Inf
/// matching and the all-non-finite failure do.
///
/// Unwritten-output sentinel. A harness that fills engine outputs with a NaN before execute
/// would otherwise see an element the engine never wrote pass as "NaN matches NaN". With the
/// mask on, a sentinel bit pattern may be supplied; an implementation element whose bits equal
/// it FAILS before any other classification. The pattern must be a NaN with a distinctive
/// payload, so that a NaN the engine really computed does not collide with it; see
/// payloadNanSentinelBits.
///
/// The sentinel is an element's raw storage bits, zero extended to 64 bits, so it also
/// supports a finite pattern, and a payload NaN of a type payloadNanSentinelBits does not
/// offer (fp8 e5m2: 0x7D, a NaN a float conversion never yields). That matters for element
/// types with no offered payload: fp8 e4m3fn has a single NaN encoding (0x7F and 0xFF), so no
/// distinctive payload exists at all. A caller grading such a type directly chooses one of:
///   (a) Pass no sentinel. NaN-NaN matching still works; an unwritten element simply cannot be
///       told from a legitimate NaN, so unwritten-output detection is unavailable for that
///       type and the harness should say so. This is the default and the safe choice.
///   (b) Fill with a finite bit pattern the correct output can never hold and pass that
///       pattern. A pattern that the reference may legitimately produce (every e4m3fn
///       finite value is representable output) turns correct elements into failures, so this
///       is only sound when the harness knows the output domain excludes the pattern.
class NonFiniteMask
{
public:
    /// Mask off.
    constexpr NonFiniteMask() noexcept = default;

    /// Mask on. `unwrittenSentinelBits` is the storage bit pattern the harness filled the
    /// output with, or std::nullopt when unwritten-output detection is unavailable.
    static constexpr NonFiniteMask matching(std::optional<uint64_t> unwrittenSentinelBits
                                            = std::nullopt) noexcept
    {
        NonFiniteMask mask;
        mask._enabled = true;
        mask._unwrittenSentinelBits = unwrittenSentinelBits;
        return mask;
    }

    constexpr bool enabled() const noexcept
    {
        return _enabled;
    }

    constexpr const std::optional<uint64_t>& unwrittenSentinelBits() const noexcept
    {
        return _unwrittenSentinelBits;
    }

private:
    bool _enabled = false;
    std::optional<uint64_t> _unwrittenSentinelBits;
};

/// Named apart from hipdnn_test_sdk::detail, which headers in this namespace reach unqualified.
namespace non_finite_detail
{

/// Unsigned integer with the same width as T, for reading an element's storage bits.
template <typename T>
using StorageBits = std::conditional_t<
    sizeof(T) == 1,
    uint8_t,
    std::conditional_t<sizeof(T) == 2,
                       uint16_t,
                       std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;

/// The element's storage bits, zero extended to 64 bits.
template <typename T>
uint64_t elementBits(const T& value) noexcept
{
    static_assert(std::is_trivially_copyable_v<T> && sizeof(T) == sizeof(StorageBits<T>),
                  "Element type must be trivially copyable with a 1, 2, 4 or 8 byte size");
    StorageBits<T> bits{};
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}

/// Rejects a sentinel wider than the element type. It could never match, which would silently
/// disable unwritten-output detection.
template <typename T>
void requireSentinelFits([[maybe_unused]] const NonFiniteMask& nonFiniteMask)
{
    // An 8 byte element holds any 64 bit pattern, so there is nothing to check.
    if constexpr(sizeof(T) < sizeof(uint64_t))
    {
        if(nonFiniteMask.enabled() && nonFiniteMask.unwrittenSentinelBits().has_value()
           && *nonFiniteMask.unwrittenSentinelBits()
                  > static_cast<uint64_t>(std::numeric_limits<StorageBits<T>>::max()))
        {
            throw std::invalid_argument("Unwritten-output sentinel does not fit the element type");
        }
    }
}

} // namespace non_finite_detail

/// Storage bits of the unwritten-output sentinel for element type T: a quiet NaN whose payload
/// differs from the type's default quiet NaN, so that a NaN an engine computes (which is the
/// default quiet NaN or an input's NaN) does not collide with it. std::nullopt for a type
/// that cannot carry one.
///
/// Exactly float, double, half and bfloat16 (both rounding modes) have one. Every other type
/// returns std::nullopt: the fp8 formats (e4m3 has the single NaN encoding 0x7F/0xFF; e5m2 has
/// payload encodings, but no validator factory grades fp8, so none is offered; see NonFiniteMask
/// for what a caller grading such a type directly does), the FNUZ fp8 formats (single NaN 0x80),
/// fp8 e8m0 (single NaN 0xFF), fp6, fp4 and the integer types.
///
/// The pattern survives only a bitwise store and load. Converting through another floating
/// point type or arithmetic may canonicalize the payload, so fill with the raw bits (or with
/// payloadNanSentinelValue, which returns them in T) and never by converting a float NaN.
template <typename T>
constexpr std::optional<uint64_t> payloadNanSentinelBits() noexcept
{
    namespace types = hipdnn_data_sdk::types;
    if constexpr(std::is_same_v<T, float>)
    {
        return static_cast<uint64_t>(0x7FC0DEADU);
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        return static_cast<uint64_t>(0x7FF80000DEADBEEFULL);
    }
    else if constexpr(std::is_same_v<T, types::half>)
    {
        return static_cast<uint64_t>(0x7EADU);
    }
    else if constexpr(std::is_same_v<T, types::bfloat16>
                      || std::is_same_v<T, types::bfloat16_truncate>)
    {
        return static_cast<uint64_t>(0x7FEDU);
    }
    else
    {
        return std::nullopt;
    }
}

/// payloadNanSentinelBits for a runtime data type. std::nullopt for every data type without
/// a payload NaN, including FP8_E4M3 and the integer types.
inline std::optional<uint64_t>
    payloadNanSentinelBits(hipdnn_flatbuffers_sdk::data_objects::DataType dataType) noexcept
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(dataType)
    {
    case DataType::FLOAT:
        return payloadNanSentinelBits<float>();
    case DataType::DOUBLE:
        return payloadNanSentinelBits<double>();
    case DataType::HALF:
        return payloadNanSentinelBits<hipdnn_data_sdk::types::half>();
    case DataType::BFLOAT16:
        return payloadNanSentinelBits<hipdnn_data_sdk::types::bfloat16>();
    default:
        return std::nullopt;
    }
}

/// The sentinel as a value of T, for filling an output element-wise. Does not compile for a
/// type that payloadNanSentinelBits reports no payload NaN for.
template <typename T>
T payloadNanSentinelValue() noexcept
{
    constexpr std::optional<uint64_t> BITS = payloadNanSentinelBits<T>();
    static_assert(BITS.has_value(), "T has no payload NaN sentinel");
    const auto storage = static_cast<non_finite_detail::StorageBits<T>>(*BITS);
    T value{};
    std::memcpy(&value, &storage, sizeof(T));
    return value;
}

} // namespace hipdnn_test_sdk::utilities
