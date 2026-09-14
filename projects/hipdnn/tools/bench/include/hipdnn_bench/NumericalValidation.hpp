// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_bench/VariantPackBuilder.hpp>

#include <hipdnn_frontend.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

/// @file NumericalValidation.hpp
/// @brief Deciding whether a timed candidate is allowed to become a training label
///        (RFC 0019 §13.2, Open Question 19).
///
/// §13.2: "A timing is only a training label once the candidate is known correct." The two
/// failure directions are not symmetric. A candidate that never matched is a coverage gap --
/// the model simply never learns it, and the ranking is intact but incomplete. A candidate
/// that runs fast and computes the wrong answer produces the *best* time in its group, so it
/// becomes the label the ranker is trained to prefer, and nothing downstream catches it:
/// §6.3's contract checks fingerprint the feature contract, not the kernel's output.
///
/// **The reference is the catalog itself** -- Open Question 19(a)'s "trusted in-catalog
/// kernel". It is the only reference available to this tool: it deserializes an opaque graph
/// and never learns which operation it is about to run, so there is no per-op reference
/// executor for it to call, and `hipdnn_gpu_ref` is an integration-test library outside this
/// build. Every candidate of one problem reads the same inputs and writes the same output
/// tensors, so candidates are evidence about each other: the ones that agree corroborate,
/// and the one that stands alone is the one to refuse.
///
/// **Three verdicts, not two.** "We did not check" must never be spelled the same way as "we
/// checked and it agreed" -- a missing check reading as valid is precisely the inverted
/// oracle §13.2 exists to prevent. UNKNOWN is the honest state while Open Question 19 is open
/// (one candidate, or a dtype this file cannot decode); it does not suppress a label, because
/// nothing was shown about it either way. DISAGREED did show something, and suppresses one.
///
/// In a header rather than the tool's main file for the same reason CsvOutput.hpp is: the
/// decision is testable without a device, and its failure mode is silence rather than a crash.
namespace hipdnn_bench
{

/// Why a row's measurement may or may not be trusted as a label.
enum class NumericalVerdict
{
    AGREED,    ///< Cross-checked against the catalog and consistent with it.
    DISAGREED, ///< Cross-checked and inconsistent; §13.2's invalid marker.
    UNKNOWN    ///< Not cross-checkable here. Never a synonym for AGREED.
};

/// A verdict and the short machine-readable reason recorded beside it on the row.
struct ValidationOutcome
{
    NumericalVerdict verdict = NumericalVerdict::UNKNOWN;

    /// `prefix: detail`. The prefix is the class of outcome (`output_mismatch`,
    /// `no_reference`, ...) so a corpus can be grouped on it; the detail names the tensor
    /// and the deviation, because "a candidate is invalid" is not actionable and "tensor Y
    /// element 12 differs by 1.5 against a tolerance of 3.4e-04" is.
    std::string reason;
};

/// One candidate's memory state after a single untimed execution.
struct CandidateOutput
{
    /// False when the candidate could not be built or executed. Such a candidate is not
    /// evidence for or against anything, so it neither joins nor splits a cohort.
    bool executed = false;

    /// Why it did not execute. Recorded rather than discarded: §13.2 keeps both failure
    /// directions in the dataset so the model learns the failure surface.
    std::string failure;

    /// Host image of every non-virtual tensor, keyed by uid.
    std::map<int64_t, std::vector<uint8_t>> images;
};

/// What a uid is, for decoding and for naming a mismatch.
struct TensorDescription
{
    std::string name;
    hipdnn_frontend::DataType dataType = hipdnn_frontend::DataType::NOT_SET;
};

/// CSV spelling of @p verdict. Three words rather than a boolean, so the column cannot be
/// read back as one and quietly collapse UNKNOWN into one of the other two.
inline const char* verdictText(NumericalVerdict verdict)
{
    switch(verdict)
    {
    case NumericalVerdict::AGREED: return "True";
    case NumericalVerdict::DISAGREED: return "False";
    case NumericalVerdict::UNKNOWN:
    default: return "Unknown";
    }
}

namespace detail
{

/// How an element of a dtype is read as a number.
enum class NumericKind
{
    NONE, ///< Not decodable here; a tensor of this type is skipped, never assumed equal.
    SIGNED_INTEGER,
    UNSIGNED_INTEGER,
    FLOAT16,
    BFLOAT16,
    FLOAT32,
    FLOAT64
};

/// Deliberately narrow. A type is listed only where the decode below is exact, because a
/// wrong decode produces a mismatch report about a kernel that was fine -- which costs an
/// author a pack investigation and teaches them to distrust the gate. The packed, sub-byte
/// and complex types are left out and reported as UNKNOWN, which Open Question 19 permits
/// and a guessed comparison does not.
inline NumericKind numericKind(hipdnn_frontend::DataType dataType)
{
    using hipdnn_frontend::DataType;
    switch(dataType)
    {
    case DataType::DOUBLE: return NumericKind::FLOAT64;
    case DataType::FLOAT: return NumericKind::FLOAT32;
    case DataType::HALF: return NumericKind::FLOAT16;
    case DataType::BFLOAT16: return NumericKind::BFLOAT16;
    case DataType::INT8:
    case DataType::INT32:
    case DataType::INT64: return NumericKind::SIGNED_INTEGER;
    case DataType::UINT8:
    case DataType::BOOLEAN: return NumericKind::UNSIGNED_INTEGER;
    default: return NumericKind::NONE;
    }
}

/// IEEE binary16 -> double. Written out rather than taken from a half-float type because
/// this file must decode a host byte image with no device headers in scope.
inline double decodeHalf(uint16_t bits)
{
    const auto exponent = static_cast<int>((bits >> 10) & 0x1FU);
    const auto mantissa = static_cast<double>(bits & 0x3FFU);
    double value = 0.0;
    if(exponent == 0)
    {
        value = std::ldexp(mantissa, -24); // subnormal: mantissa * 2^-10 * 2^-14
    }
    else if(exponent == 0x1F)
    {
        value = (bits & 0x3FFU) != 0 ? std::numeric_limits<double>::quiet_NaN()
                                     : std::numeric_limits<double>::infinity();
    }
    else
    {
        value = std::ldexp(mantissa + 1024.0, exponent - 25); // (1024+m) * 2^(e-15-10)
    }
    return (bits & 0x8000U) != 0 ? -value : value;
}

/// bfloat16 is the top half of a binary32, so widening the bits is the whole decode.
inline double decodeBfloat16(uint16_t bits)
{
    const uint32_t widened = static_cast<uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &widened, sizeof(value));
    return static_cast<double>(value);
}

/// Element @p index of @p image, which the caller has already bounds-checked.
inline double decodeElement(const std::vector<uint8_t>& image,
                            size_t index,
                            hipdnn_frontend::DataType dataType)
{
    using hipdnn_frontend::DataType;
    const auto* bytes = image.data() + index * static_cast<size_t>(elementBits(dataType) / 8);
    switch(dataType)
    {
    case DataType::DOUBLE:
    {
        double value = 0.0;
        std::memcpy(&value, bytes, sizeof(value));
        return value;
    }
    case DataType::FLOAT:
    {
        float value = 0.0F;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    case DataType::HALF:
    case DataType::BFLOAT16:
    {
        uint16_t value = 0;
        std::memcpy(&value, bytes, sizeof(value));
        return dataType == DataType::HALF ? decodeHalf(value) : decodeBfloat16(value);
    }
    case DataType::INT64:
    {
        int64_t value = 0;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    case DataType::INT32:
    {
        int32_t value = 0;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    case DataType::INT8: return static_cast<double>(static_cast<int8_t>(*bytes));
    default: return static_cast<double>(*bytes); // UINT8, BOOLEAN
    }
}

/// Relative tolerance two correct kernels may differ by.
///
/// Not zero, and this is the whole reason the comparison is numeric rather than a memcmp:
/// two kernels that tile a reduction differently accumulate in a different order, so their
/// last bits differ by construction. A bitwise gate would mark every honest candidate
/// invalid and block every promotion, which is a worse failure than no gate at all.
///
/// Scaled by mantissa width. Integers are exact -- there is no rounding to absorb, and a
/// tolerance on an index or a count would hide an off-by-one.
inline double agreementTolerance(NumericKind kind)
{
    switch(kind)
    {
    case NumericKind::FLOAT64: return 1e-12;
    case NumericKind::FLOAT32: return 1e-5;
    case NumericKind::FLOAT16: return 2e-2; // 10 mantissa bits
    case NumericKind::BFLOAT16: return 6e-2; // 7 mantissa bits
    default: return 0.0;
    }
}

/// Where two images of the same tensor first differ, or empty when they agree.
///
/// Scaled by the larger of the two images' magnitudes rather than per element: a relative
/// test against an element that is itself near zero divides by noise and reports a mismatch
/// for a difference of one ulp. Two all-zero images agree -- there is no scale to be
/// relative to, and they are identical.
inline std::string firstDisagreement(const std::string& name,
                                     hipdnn_frontend::DataType dataType,
                                     const std::vector<uint8_t>& reference,
                                     const std::vector<uint8_t>& candidate)
{
    const auto kind = numericKind(dataType);
    const auto width = static_cast<size_t>(elementBits(dataType) / 8);
    if(kind == NumericKind::NONE || width == 0 || reference.size() != candidate.size())
    {
        return {}; // Not comparable; the caller counts it as skipped rather than as agreement.
    }
    const size_t count = reference.size() / width;

    double scale = 0.0;
    for(size_t index = 0; index < count; ++index)
    {
        const double left = decodeElement(reference, index, dataType);
        const double right = decodeElement(candidate, index, dataType);
        if(std::isfinite(left))
        {
            scale = std::max(scale, std::abs(left));
        }
        if(std::isfinite(right))
        {
            scale = std::max(scale, std::abs(right));
        }
    }
    const double threshold = agreementTolerance(kind) * scale;

    for(size_t index = 0; index < count; ++index)
    {
        const double left = decodeElement(reference, index, dataType);
        const double right = decodeElement(candidate, index, dataType);
        // A candidate that produced NaN where the reference produced a number is the
        // clearest possible disagreement, and `NaN > threshold` is false, so the
        // non-finite cases are decided before the magnitude test rather than by it.
        const bool bothNonFinite = !std::isfinite(left) && !std::isfinite(right);
        const bool sameNonFinite = bothNonFinite && !(left < right) && !(right < left)
                                   && std::isnan(left) == std::isnan(right);
        if(sameNonFinite)
        {
            continue;
        }
        if(!std::isfinite(left) || !std::isfinite(right) || std::abs(left - right) > threshold)
        {
            std::ostringstream detail;
            detail.precision(3);
            detail << "tensor '" << name << "' element " << index << " is " << std::scientific
                   << right << " against the catalog's " << left << ", outside a tolerance of "
                   << threshold;
            return detail.str();
        }
    }
    return {};
}

/// True when every comparable tensor of @p candidate is still the zero fill it was
/// allocated with.
///
/// This tool allocates zero-filled buffers and does not write inputs, so for many
/// operations a correct kernel and a kernel that writes nothing both leave zeros behind.
/// Unanimous agreement on an all-zero output therefore proves nothing at all, and reporting
/// it as AGREED would be the silent "valid" RFC 0019 §13.2 forbids.
inline bool leftOutputUntouched(const CandidateOutput& candidate,
                                const std::map<int64_t, TensorDescription>& tensors)
{
    for(const auto& [uid, tensor] : tensors)
    {
        const auto image = candidate.images.find(uid);
        if(numericKind(tensor.dataType) == NumericKind::NONE || image == candidate.images.end())
        {
            continue;
        }
        if(std::any_of(image->second.begin(), image->second.end(), [](uint8_t byte) {
               return byte != 0;
           }))
        {
            return false;
        }
    }
    return true;
}

/// True when @p left and @p right left every comparable tensor in the same state.
inline bool sameOutput(const CandidateOutput& left,
                       const CandidateOutput& right,
                       const std::map<int64_t, TensorDescription>& tensors)
{
    for(const auto& [uid, tensor] : tensors)
    {
        const auto leftImage = left.images.find(uid);
        const auto rightImage = right.images.find(uid);
        if(leftImage == left.images.end() || rightImage == right.images.end())
        {
            continue;
        }
        if(!firstDisagreement(tensor.name, tensor.dataType, leftImage->second, rightImage->second)
                .empty())
        {
            return false;
        }
    }
    return true;
}

/// Tensors of @p candidate this file can decode.
inline size_t comparableTensors(const CandidateOutput& candidate,
                                const std::map<int64_t, TensorDescription>& tensors)
{
    size_t comparable = 0;
    for(const auto& [uid, tensor] : tensors)
    {
        if(numericKind(tensor.dataType) != NumericKind::NONE && candidate.images.count(uid) != 0)
        {
            ++comparable;
        }
    }
    return comparable;
}

} // namespace detail

/// @brief One verdict per candidate, in input order (RFC 0019 §13.2).
///
/// Candidates are partitioned into cohorts that left identical output. The rule is majority,
/// not first-one-wins: if the catalog's first candidate is the broken one, taking it as truth
/// would invert the verdicts and mark every correct kernel invalid. A strict majority cohort
/// is the reference and its members are AGREED; every candidate outside it is DISAGREED.
///
/// An even split is DISAGREED for everyone in the dispute, which is the deliberate choice
/// here. At least one of those candidates is computing the wrong answer, and §13.2 says the
/// timing of a candidate that is not known correct is not a label. Reporting it as UNKNOWN
/// would let the wrong-but-fast one through, which is the case the section is about. §13.2
/// also names the remedy: the marker is cleared in the matcher or the kernel, not here.
inline std::vector<ValidationOutcome>
    crossCheckCandidates(const std::vector<CandidateOutput>& candidates,
                         const std::map<int64_t, TensorDescription>& tensors)
{
    std::vector<ValidationOutcome> outcomes(candidates.size());

    std::vector<size_t> representatives; // index of each cohort's first member
    std::vector<std::vector<size_t>> cohorts;
    for(size_t index = 0; index < candidates.size(); ++index)
    {
        const auto& candidate = candidates[index];
        if(!candidate.executed)
        {
            outcomes[index] = {NumericalVerdict::UNKNOWN,
                               "not_executed: " + (candidate.failure.empty()
                                                       ? std::string("the candidate produced no "
                                                                     "output to cross-check")
                                                       : candidate.failure)};
            continue;
        }
        if(detail::comparableTensors(candidate, tensors) == 0)
        {
            outcomes[index]
                = {NumericalVerdict::UNKNOWN,
                   "no_comparable_output: this problem declares no tensor of a type the "
                   "cross-check can decode (RFC 0019 Open Question 19 leaves the per-op "
                   "reference open)"};
            continue;
        }
        size_t cohort = cohorts.size();
        for(size_t existing = 0; existing < cohorts.size(); ++existing)
        {
            if(detail::sameOutput(candidates[representatives[existing]], candidate, tensors))
            {
                cohort = existing;
                break;
            }
        }
        if(cohort == cohorts.size())
        {
            representatives.push_back(index);
            cohorts.emplace_back();
        }
        cohorts[cohort].push_back(index);
    }

    if(cohorts.empty())
    {
        return outcomes;
    }
    size_t crossChecked = 0;
    size_t largest = 0;
    for(size_t cohort = 0; cohort < cohorts.size(); ++cohort)
    {
        crossChecked += cohorts[cohort].size();
        if(cohorts[cohort].size() > cohorts[largest].size())
        {
            largest = cohort;
        }
    }
    // Counted after `largest` is final: a running tally would miss an earlier cohort that
    // the eventual winner only matched in size, and report a split as decided.
    const size_t tied
        = static_cast<size_t>(std::count_if(cohorts.begin(), cohorts.end(), [&](const auto& c) {
              return c.size() == cohorts[largest].size();
          }));

    if(crossChecked < 2)
    {
        // One candidate agreeing with itself is not evidence, so the row says so rather
        // than claiming a verdict the run did not earn.
        outcomes[cohorts[largest].front()]
            = {NumericalVerdict::UNKNOWN,
               "no_reference: one candidate ran for this problem, so there was nothing to "
               "cross-check it against"};
        return outcomes;
    }

    if(cohorts.size() == 1 && detail::leftOutputUntouched(candidates[representatives[0]], tensors))
    {
        for(const size_t index : cohorts[0])
        {
            outcomes[index] = {NumericalVerdict::UNKNOWN,
                               "degenerate_reference: every cross-checked candidate left the "
                               "zero-filled output untouched, so their agreement is not "
                               "evidence that any of them computed anything"};
        }
        return outcomes;
    }

    const bool decided = tied == 1;
    const size_t reference = representatives[decided ? largest : 0];
    for(size_t cohort = 0; cohort < cohorts.size(); ++cohort)
    {
        for(const size_t index : cohorts[cohort])
        {
            if(decided && cohort == largest)
            {
                outcomes[index] = {NumericalVerdict::AGREED,
                                   "agrees_with_catalog: " + std::to_string(cohorts[cohort].size())
                                       + " of " + std::to_string(crossChecked)
                                       + " cross-checked candidates produced this output"};
                continue;
            }
            std::string detail = "the candidate's output has no corroboration in the catalog";
            for(const auto& [uid, tensor] : tensors)
            {
                const auto mine = candidates[index].images.find(uid);
                const auto theirs = candidates[reference].images.find(uid);
                if(mine == candidates[index].images.end()
                   || theirs == candidates[reference].images.end())
                {
                    continue;
                }
                const auto difference = detail::firstDisagreement(
                    tensor.name, tensor.dataType, theirs->second, mine->second);
                if(!difference.empty())
                {
                    detail = difference;
                    break;
                }
            }
            outcomes[index] = {NumericalVerdict::DISAGREED,
                               (decided ? "output_mismatch: " : "disputed_output: ") + detail};
        }
    }
    return outcomes;
}

} // namespace hipdnn_bench
