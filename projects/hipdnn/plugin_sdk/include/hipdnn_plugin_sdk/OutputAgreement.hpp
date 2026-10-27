// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// OutputAgreement is the rule two kernel outputs are compared by, and the fill pattern that
// tells an output a kernel wrote from one it never touched. It is not gated on
// HIPDNN_ENABLE_KERNEL_INGESTOR and includes only the flatbuffers DataType and the data_sdk
// type decoders, so any consumer that links plugin_sdk can share it.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>

namespace hipdnn_plugin_sdk::output_agreement
{

using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;

/// How an element of a dtype is read as a number.
enum class NumericKind
{
    NONE, ///< Not decodable here; an output of this type is never value-compared.
    SIGNED_INTEGER,
    UNSIGNED_INTEGER,
    FLOAT16,
    BFLOAT16,
    FLOAT32,
    FLOAT64
};

/// Only types whose decode is exact are listed; a wrong decode would report disagreement for
/// correct kernels. Packed, sub-byte and FP8 types are NONE.
inline NumericKind numericKind(DataType dataType)
{
    switch(dataType)
    {
    case DataType::DOUBLE:
        return NumericKind::FLOAT64;
    case DataType::FLOAT:
        return NumericKind::FLOAT32;
    case DataType::HALF:
        return NumericKind::FLOAT16;
    case DataType::BFLOAT16:
        return NumericKind::BFLOAT16;
    case DataType::INT8:
    case DataType::INT32:
    case DataType::INT64:
        return NumericKind::SIGNED_INTEGER;
    case DataType::UINT8:
    case DataType::BOOLEAN:
        return NumericKind::UNSIGNED_INTEGER;
    default:
        return NumericKind::NONE;
    }
}

/// Bytes per element of a decodable dtype; 0 when numericKind() is NONE.
inline size_t elementBytes(DataType dataType)
{
    switch(dataType)
    {
    case DataType::DOUBLE:
    case DataType::INT64:
        return 8;
    case DataType::FLOAT:
    case DataType::INT32:
        return 4;
    case DataType::HALF:
    case DataType::BFLOAT16:
        return 2;
    case DataType::INT8:
    case DataType::UINT8:
    case DataType::BOOLEAN:
        return 1;
    default:
        return 0;
    }
}

/// Relative tolerance two correct kernels may differ by. Non-zero because differing
/// reduction orders change the last bits; integers are exact.
inline double agreementTolerance(NumericKind kind)
{
    switch(kind)
    {
    case NumericKind::FLOAT64:
        return 1e-12;
    case NumericKind::FLOAT32:
        return 1e-5;
    case NumericKind::FLOAT16:
        return 2e-2; // 10 mantissa bits
    case NumericKind::BFLOAT16:
        return 6e-2; // 7 mantissa bits
    default:
        return 0.0;
    }
}

/// The absolute term's coefficient: 2^-mantissa bits, so `agreementFloor(kind) * scale` is
/// one ulp at the output's own magnitude.
inline double agreementFloor(NumericKind kind)
{
    switch(kind)
    {
    case NumericKind::FLOAT64:
        return 0x1p-52;
    case NumericKind::FLOAT32:
        return 0x1p-23;
    case NumericKind::FLOAT16:
        return 0x1p-10;
    case NumericKind::BFLOAT16:
        return 0x1p-7;
    default:
        return 0.0; // integers are exact; their bar is equality
    }
}

namespace detail
{

// One element's bytes -> double, per storage type. decodeElement() and the comparison loops
// share these so the element a message names is the element the comparison flagged.
inline double decodeDoubleAt(const uint8_t* bytes)
{
    double value = 0.0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

inline double decodeFloatAt(const uint8_t* bytes)
{
    float value = 0.0F;
    std::memcpy(&value, bytes, sizeof(value));
    return static_cast<double>(value);
}

inline double decodeHalfAt(const uint8_t* bytes)
{
    uint16_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return static_cast<double>(hipdnn_data_sdk::types::detail::half_bits_to_float(value));
}

inline double decodeBfloat16At(const uint8_t* bytes)
{
    uint16_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return static_cast<double>(hipdnn_data_sdk::types::detail::bfloat16_bits_to_float(value));
}

inline double decodeInt64At(const uint8_t* bytes)
{
    int64_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return static_cast<double>(value);
}

inline double decodeInt32At(const uint8_t* bytes)
{
    int32_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return static_cast<double>(value);
}

inline double decodeInt8At(const uint8_t* bytes)
{
    return static_cast<double>(static_cast<int8_t>(*bytes));
}

inline double decodeByteAt(const uint8_t* bytes)
{
    return static_cast<double>(*bytes); // UINT8, BOOLEAN
}

/// Calls @p visit with the decoder of @p dataType, so the type is chosen once per image and
/// not once per element. @p dataType must have a numericKind() other than NONE.
template <typename Visit>
auto withDecoder(DataType dataType, Visit&& visit)
{
    switch(dataType)
    {
    case DataType::DOUBLE:
        return visit(decodeDoubleAt);
    case DataType::FLOAT:
        return visit(decodeFloatAt);
    case DataType::HALF:
        return visit(decodeHalfAt);
    case DataType::BFLOAT16:
        return visit(decodeBfloat16At);
    case DataType::INT64:
        return visit(decodeInt64At);
    case DataType::INT32:
        return visit(decodeInt32At);
    case DataType::INT8:
        return visit(decodeInt8At);
    default:
        return visit(decodeByteAt);
    }
}

} // namespace detail

/// Element @p index of @p image as a double. The caller has bounds-checked @p index and
/// passes a dtype whose numericKind() is not NONE.
inline double decodeElement(DataType dataType, const std::vector<uint8_t>& image, size_t index)
{
    const uint8_t* bytes = image.data() + (index * elementBytes(dataType));
    return detail::withDecoder(dataType, [bytes](auto decode) { return decode(bytes); });
}

/// True when the dtype has a NaN bit pattern that fillByte() reaches. Integer, boolean,
/// FP4, FP6 and INT4 outputs have none.
inline bool hasNaNEncoding(DataType dataType)
{
    switch(dataType)
    {
    case DataType::FLOAT:
    case DataType::DOUBLE:
    case DataType::HALF:
    case DataType::BFLOAT16:
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E8M0:
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ:
        return true;
    default:
        return false;
    }
}

/// The byte every output buffer is filled with before a candidate runs: a NaN in every dtype
/// that has one (0x80 for the two FNUZ types, 0xFF otherwise), and a value no kernel is
/// likely to write elsewhere, so an unwritten integer output still differs from a written one.
inline uint8_t fillByte(DataType dataType)
{
    // The FNUZ types encode NaN as the lone 0x80 pattern; 0xFF is a NaN in every other type
    // that has one (float, double, half, bfloat16, FP8 E4M3 / E5M2 / E8M0).
    const bool fnuz = dataType == DataType::FP8_E4M3_FNUZ || dataType == DataType::FP8_E5M2_FNUZ;
    return fnuz ? 0x80 : 0xFF;
}

/// True when @p image is non-empty, @p dataType has a NaN encoding, and every byte is still
/// fillByte(@p dataType): the kernel never wrote this output.
inline bool imageUntouched(DataType dataType, const std::vector<uint8_t>& image)
{
    if(!hasNaNEncoding(dataType) || image.empty())
    {
        return false;
    }
    const uint8_t fill = fillByte(dataType);
    return std::all_of(image.begin(), image.end(), [fill](uint8_t byte) { return byte == fill; });
}

/// One element two images disagree on: the first deviation firstDisagreementAt() found.
struct Disagreement
{
    size_t index = 0; ///< Element index within the image.
    double left = 0.0; ///< The element decoded from the left image.
    double right = 0.0; ///< The element decoded from the right image.
    double tolerance = 0.0; ///< rtol*max(|left|,|right|) + floor*scale at this element.
};

/// The first NaN or infinity of an image: firstNonFiniteAt().
struct NonFinite
{
    size_t index = 0; ///< Element index within the image.
    double value = 0.0; ///< The element decoded.
};

namespace detail
{

/// firstDisagreementAt() over one storage type's @p decode.
///
/// Per element, `|a - b| <= rtol * max(|a|, |b|) + atol`, with atol one ulp at the largest
/// finite magnitude over both images: the relative term judges each element against itself,
/// and atol tolerates near-zero elements that are cancellation residue.
template <typename Decode>
std::optional<Disagreement> firstDisagreementDecoded(NumericKind kind,
                                                     size_t width,
                                                     const std::vector<uint8_t>& left,
                                                     const std::vector<uint8_t>& right,
                                                     Decode decode)
{
    const size_t count = left.size() / width;
    const uint8_t* leftBytes = left.data();
    const uint8_t* rightBytes = right.data();

    double scale = 0.0;
    for(size_t index = 0; index < count; ++index)
    {
        const double a = decode(leftBytes + (index * width));
        const double b = decode(rightBytes + (index * width));
        if(std::isfinite(a))
        {
            scale = std::max(scale, std::abs(a));
        }
        if(std::isfinite(b))
        {
            scale = std::max(scale, std::abs(b));
        }
    }
    const double relative = agreementTolerance(kind);
    const double absolute = agreementFloor(kind) * scale;

    for(size_t index = 0; index < count; ++index)
    {
        const double a = decode(leftBytes + (index * width));
        const double b = decode(rightBytes + (index * width));
        // Decide non-finite cases before the magnitude test: `NaN > threshold` is false.
        const bool bothNonFinite = !std::isfinite(a) && !std::isfinite(b);
        const bool sameNonFinite
            = bothNonFinite && !(a < b) && !(b < a) && std::isnan(a) == std::isnan(b);
        if(sameNonFinite)
        {
            continue;
        }
        const double threshold = (relative * std::max(std::abs(a), std::abs(b))) + absolute;
        if(!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > threshold)
        {
            return Disagreement{index, a, b, threshold};
        }
    }
    return std::nullopt;
}

} // namespace detail

/// The first element where the images deviate, or nullopt when they agree, are
/// byte-identical, differ in size, or @p dataType is NONE. The scale is the largest finite
/// magnitude over BOTH images; an element agrees when |a-b| <= rtol*max(|a|,|b|) +
/// floor*scale. The same non-finite kind at the same position agrees; NaN against finite,
/// and +Inf against -Inf, disagree. `left` and `right` are the elements read from @p left
/// and @p right.
inline std::optional<Disagreement> firstDisagreementAt(DataType dataType,
                                                       const std::vector<uint8_t>& left,
                                                       const std::vector<uint8_t>& right)
{
    const NumericKind kind = numericKind(dataType);
    const size_t width = elementBytes(dataType);
    if(kind == NumericKind::NONE || width == 0 || left.size() != right.size())
    {
        return std::nullopt;
    }
    // Identical bytes decode to identical values, and every such element agrees with itself,
    // so the decode cannot disagree where this agrees. It is the common case.
    if(left == right)
    {
        return std::nullopt;
    }
    return detail::withDecoder(dataType, [&](auto decode) {
        return detail::firstDisagreementDecoded(kind, width, left, right, decode);
    });
}

/// "" when firstDisagreementAt() finds none; otherwise the first deviation as
/// "<name>[<index>]: <a> vs <b> (tolerance <t>)", a read from @p left and b from @p right.
inline std::string firstDisagreement(const std::string& name,
                                     DataType dataType,
                                     const std::vector<uint8_t>& left,
                                     const std::vector<uint8_t>& right)
{
    const auto found = firstDisagreementAt(dataType, left, right);
    if(!found)
    {
        return {};
    }
    std::ostringstream message;
    message.precision(9);
    message << name << '[' << found->index << "]: " << found->left << " vs " << found->right
            << " (tolerance " << found->tolerance << ')';
    return message.str();
}

/// The first NaN or infinity of @p image, or nullopt when every element is finite or
/// @p dataType is NONE.
inline std::optional<NonFinite> firstNonFiniteAt(DataType dataType,
                                                 const std::vector<uint8_t>& image)
{
    const size_t width = elementBytes(dataType);
    if(numericKind(dataType) == NumericKind::NONE || width == 0)
    {
        return std::nullopt;
    }
    const size_t count = image.size() / width;
    for(size_t index = 0; index < count; ++index)
    {
        const double value = decodeElement(dataType, image, index);
        if(!std::isfinite(value))
        {
            return NonFinite{index, value};
        }
    }
    return std::nullopt;
}

/// "" when every element of @p image is finite or @p dataType is NONE; otherwise
/// "<name>[<index>]: <value> is not finite" for the first NaN or infinity.
inline std::string
    firstNonFinite(const std::string& name, DataType dataType, const std::vector<uint8_t>& image)
{
    const auto found = firstNonFiniteAt(dataType, image);
    if(!found)
    {
        return {};
    }
    std::ostringstream message;
    message << name << '[' << found->index << "]: " << found->value << " is not finite";
    return message.str();
}

} // namespace hipdnn_plugin_sdk::output_agreement
