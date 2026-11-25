// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/NonFiniteMask.hpp>

namespace hipdnn_test_sdk::utilities
{

/// Whether a comparison treats same-signed infinities in the reference and the
/// implementation as equal.
///
/// REJECTED is the default: an unexpected infinity is normally a real defect, most often an
/// output element the operation never wrote. ACCEPTED is for outputs whose correct value is
/// infinite on both sides, such as an SDPA forward STATS (log-sum-exp) tensor holding -inf
/// for every fully masked causal row.
enum class MatchingInfinities
{
    REJECTED,
    ACCEPTED
};

/// What the element rule decides for one (reference, implementation) pair.
enum class ElementVerdict
{
    /// The pair agrees by the non-finite rules and takes no part in the tolerance math.
    MATCH,
    /// A NaN or Inf the rules do not accept: the validators fail it.
    NON_FINITE_REJECT,
    /// The implementation element still holds the unwritten-output sentinel: the validators
    /// fail it before any other classification.
    UNWRITTEN_REJECT,
    /// Both values are finite: grade the pair by the validator's tolerance.
    GRADED,
};

/// The one element rule shared by the validators (CpuFpReferenceValidation,
/// CpuFpReferenceMiopenRmsValidation) and the tensor diff (computeTensorDiff), so a report
/// cannot disagree with the verdict about which elements are non-finite failures.
///
/// Mask on (NonFiniteMask::enabled()):
///   - an implementation element whose storage bits equal the mask's unwritten-output
///     sentinel is UNWRITTEN_REJECT, before anything else;
///   - NaN and NaN are MATCH, whatever the payloads and signs;
///   - Inf and Inf of the same sign are MATCH;
///   - any other pair with a NaN or Inf is NON_FINITE_REJECT;
///   - finite against finite is GRADED.
/// Mask off:
///   - Inf and Inf of the same sign are MATCH only when `matchingInfinities` is ACCEPTED;
///   - any other pair with a NaN or Inf is NON_FINITE_REJECT;
///   - finite against finite is GRADED.
///
/// The ragged-reference rule (a NaN reference value marks an undefined position, RFC 0014) is
/// applied by the callers before this one.
template <typename T>
ElementVerdict classifyElement(const T& referenceValue,
                               const T& implementationValue,
                               const NonFiniteMask& nonFiniteMask,
                               MatchingInfinities matchingInfinities) noexcept
{
    using hipdnn_data_sdk::types::isinf;
    using hipdnn_data_sdk::types::isnan;
    using hipdnn_data_sdk::types::signbit;

    if(nonFiniteMask.enabled())
    {
        const auto& sentinelBits = nonFiniteMask.unwrittenSentinelBits();
        if(sentinelBits.has_value()
           && non_finite_detail::elementBits(implementationValue) == *sentinelBits)
        {
            return ElementVerdict::UNWRITTEN_REJECT;
        }
    }

    const bool refIsNan = isnan(referenceValue);
    const bool implIsNan = isnan(implementationValue);
    const bool refIsInf = isinf(referenceValue);
    const bool implIsInf = isinf(implementationValue);
    if(!refIsNan && !implIsNan && !refIsInf && !implIsInf)
    {
        return ElementVerdict::GRADED;
    }

    const bool sameSignInfinities
        = refIsInf && implIsInf && signbit(referenceValue) == signbit(implementationValue);
    if(sameSignInfinities
       && (nonFiniteMask.enabled() || matchingInfinities == MatchingInfinities::ACCEPTED))
    {
        return ElementVerdict::MATCH;
    }
    if(nonFiniteMask.enabled() && refIsNan && implIsNan)
    {
        return ElementVerdict::MATCH;
    }
    return ElementVerdict::NON_FINITE_REJECT;
}

} // namespace hipdnn_test_sdk::utilities
