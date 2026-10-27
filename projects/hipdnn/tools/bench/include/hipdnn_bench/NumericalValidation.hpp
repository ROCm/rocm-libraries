// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_bench/VariantPackBuilder.hpp>

#include <hipdnn_frontend.hpp>
#include <hipdnn_plugin_sdk/OutputAgreement.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/// @file NumericalValidation.hpp
/// @brief Deciding whether a timed candidate may become a training label
///        (RFC 0019 §13.2, Open Question 19).
///
/// The reference is the catalog itself: every candidate of one problem reads the same inputs,
/// so candidates that agree corroborate each other. The tool runs an opaque graph and has no
/// per-op reference executor. UNKNOWN ("not checked") must never read as AGREED.
namespace hipdnn_bench
{

/// Why a row's measurement may or may not be trusted as a label.
enum class NumericalVerdict
{
    AGREED, ///< Cross-checked against the catalog and consistent with it.
    DISAGREED, ///< Cross-checked and inconsistent; §13.2's invalid marker.
    UNKNOWN ///< Not cross-checkable here. Never a synonym for AGREED.
};

/// A verdict and the short machine-readable reason recorded beside it on the row.
struct ValidationOutcome
{
    NumericalVerdict verdict = NumericalVerdict::UNKNOWN;

    /// `prefix: detail`. The prefix classifies the outcome (`output_mismatch`,
    /// `no_reference`, ...); the detail names the tensor and the deviation.
    std::string reason;
};

/// One candidate's memory state after a single untimed execution.
struct CandidateOutput
{
    /// False when the candidate could not be built or executed; it then joins no cohort.
    bool executed = false;

    /// Why it did not execute; kept so the dataset records failures (§13.2).
    std::string failure;

    /// Host image of every tensor crossCheckedOutputs() selects, keyed by uid.
    std::map<int64_t, std::vector<uint8_t>> images;
};

/// What a uid is, for decoding and for naming a mismatch.
struct TensorDescription
{
    std::string name;
    hipdnn_frontend::DataType dataType = hipdnn_frontend::DataType::NOT_SET;
};

/// @brief The tensors a candidate's capture holds and the cross-check judges: the
///        non-virtual tensors some node of the graph writes.
///
/// Outputs only: inputs are identical across candidates, so they would always agree.
inline std::vector<TensorRequirement> crossCheckedOutputs(const VariantPackPlan& plan)
{
    std::vector<TensorRequirement> outputs;
    std::copy_if(plan.tensors.begin(),
                 plan.tensors.end(),
                 std::back_inserter(outputs),
                 [](const TensorRequirement& tensor) { return tensor.produced; });
    return outputs;
}

/// CSV spelling of @p verdict; three words, not a boolean, so UNKNOWN cannot collapse.
inline const char* verdictText(NumericalVerdict verdict)
{
    switch(verdict)
    {
    case NumericalVerdict::AGREED:
        return "True";
    case NumericalVerdict::DISAGREED:
        return "False";
    case NumericalVerdict::UNKNOWN:
    default:
        return "Unknown";
    }
}

namespace detail
{

namespace agreement = hipdnn_plugin_sdk::output_agreement;

/// The flatbuffers dtype the shared output_agreement rules read. A frontend type with no
/// flatbuffers counterpart maps to UNSET, which the rules never decode.
inline agreement::DataType toSdkDataType(hipdnn_frontend::DataType dataType)
{
    using hipdnn_frontend::DataType;
    using SdkDataType = agreement::DataType;
    switch(dataType)
    {
    case DataType::FLOAT:
        return SdkDataType::FLOAT;
    case DataType::DOUBLE:
        return SdkDataType::DOUBLE;
    case DataType::HALF:
        return SdkDataType::HALF;
    case DataType::BFLOAT16:
        return SdkDataType::BFLOAT16;
    case DataType::INT8:
        return SdkDataType::INT8;
    case DataType::INT32:
        return SdkDataType::INT32;
    case DataType::INT64:
        return SdkDataType::INT64;
    case DataType::UINT8:
        return SdkDataType::UINT8;
    case DataType::BOOLEAN:
        return SdkDataType::BOOLEAN;
    case DataType::FP8_E4M3:
        return SdkDataType::FP8_E4M3;
    case DataType::FP8_E5M2:
        return SdkDataType::FP8_E5M2;
    case DataType::FP8_E8M0:
        return SdkDataType::FP8_E8M0;
    case DataType::FP8_E4M3_FNUZ:
        return SdkDataType::FP8_E4M3_FNUZ;
    case DataType::FP8_E5M2_FNUZ:
        return SdkDataType::FP8_E5M2_FNUZ;
    case DataType::FP4_E2M1:
        return SdkDataType::FP4_E2M1;
    case DataType::FP6_E2M3:
        return SdkDataType::FP6_E2M3;
    case DataType::FP6_E3M2:
        return SdkDataType::FP6_E3M2;
    case DataType::INT4:
        return SdkDataType::INT4;
    default:
        return SdkDataType::UNSET;
    }
}

/// True when the cross-check can read @p dataType as numbers; a tensor of any other type is
/// skipped, never assumed equal.
inline bool decodable(hipdnn_frontend::DataType dataType)
{
    return agreement::numericKind(toSdkDataType(dataType)) != agreement::NumericKind::NONE;
}

/// @brief One value of the input fill: (-1)^negative * 2^exponent, exponent 0 or 1.
///
/// 1 and 2 are exact in every type encodeFillElement() writes, so encoding never rounds.
/// Signed so large reductions do not overflow fp16; positive where an operation is undefined
/// on negative values (`inputFillImage`'s @p positive). An exact result can still overflow its
/// output type (a product of 64 values of magnitude 2 in fp16), so the cross-check does not
/// take a NaN or an infinity as wrong by itself.
struct FillValue
{
    bool negative = false;
    int exponent = 0;
};

/// splitmix64's finalizer over (@p seed, @p uid, @p index).
///
/// Hashed rather than alternating, so reductions do not cancel to noise. The uid makes
/// distinct input tensors differ (A == B would hide a transposed operand).
inline uint64_t fillHash(uint64_t seed, int64_t uid, size_t index)
{
    uint64_t state = seed + (static_cast<uint64_t>(uid) * 0x9E3779B97F4A7C15ULL)
                     + (static_cast<uint64_t>(index) * 0xBF58476D1CE4E5B9ULL);
    state ^= state >> 30U;
    state *= 0xBF58476D1CE4E5B9ULL;
    state ^= state >> 27U;
    state *= 0x94D049BB133111EBULL;
    state ^= state >> 31U;
    return state;
}

/// Two bits of @p hash: the sign and the choice between 1 and 2.
inline FillValue fillValue(uint64_t hash)
{
    return {(hash & 1U) != 0, static_cast<int>((hash >> 1U) & 1U)};
}

/// IEEE-style code for a power of two: sign, biased exponent, mantissa left at zero.
inline uint64_t powerOfTwoCode(FillValue value, int bias, int mantissaBits, int width)
{
    const uint64_t code = static_cast<uint64_t>(bias + value.exponent)
                          << static_cast<unsigned>(mantissaBits);
    return value.negative ? (code | (1ULL << static_cast<unsigned>(width - 1))) : code;
}

/// @brief Writes one element of @p dataType at @p bytes; false for a type not encoded here.
///
/// Sub-byte and packed types are declined and keep the zero fill; a wrong encode would put
/// NaN or inf into an input.
inline bool encodeFillElement(hipdnn_frontend::DataType dataType, FillValue value, uint8_t* bytes)
{
    using hipdnn_frontend::DataType;
    // Integers get magnitude only: integer inputs are often indices or counts.
    const int64_t magnitude = int64_t{1} << value.exponent;
    switch(dataType)
    {
    case DataType::DOUBLE:
    {
        const uint64_t code = powerOfTwoCode(value, 1023, 52, 64);
        std::memcpy(bytes, &code, sizeof(code));
        return true;
    }
    case DataType::FLOAT:
    {
        const auto code = static_cast<uint32_t>(powerOfTwoCode(value, 127, 23, 32));
        std::memcpy(bytes, &code, sizeof(code));
        return true;
    }
    case DataType::HALF:
    {
        const auto code = static_cast<uint16_t>(powerOfTwoCode(value, 15, 10, 16));
        std::memcpy(bytes, &code, sizeof(code));
        return true;
    }
    case DataType::BFLOAT16:
    {
        // bfloat16 is the top half of a binary32: binary32's bias, 7 mantissa bits.
        const auto code = static_cast<uint16_t>(powerOfTwoCode(value, 127, 7, 16));
        std::memcpy(bytes, &code, sizeof(code));
        return true;
    }
    case DataType::FP8_E4M3:
        *bytes = static_cast<uint8_t>(powerOfTwoCode(value, 7, 3, 8));
        return true;
    case DataType::FP8_E5M2:
        *bytes = static_cast<uint8_t>(powerOfTwoCode(value, 15, 2, 8));
        return true;
    // FNUZ uses the all-ones exponent for NaN, not inf, so its bias is one higher.
    case DataType::FP8_E4M3_FNUZ:
        *bytes = static_cast<uint8_t>(powerOfTwoCode(value, 8, 3, 8));
        return true;
    case DataType::FP8_E5M2_FNUZ:
        *bytes = static_cast<uint8_t>(powerOfTwoCode(value, 16, 2, 8));
        return true;
    // Block scale: no sign or mantissa. Filled, since a zero scale would zero the output.
    case DataType::FP8_E8M0:
        *bytes = static_cast<uint8_t>(127 + value.exponent);
        return true;
    case DataType::INT64:
    {
        std::memcpy(bytes, &magnitude, sizeof(magnitude));
        return true;
    }
    case DataType::INT32:
    {
        const auto code = static_cast<int32_t>(magnitude);
        std::memcpy(bytes, &code, sizeof(code));
        return true;
    }
    case DataType::INT8:
    case DataType::UINT8:
        *bytes = static_cast<uint8_t>(magnitude);
        return true;
    // All-true mask; false would switch off whatever it gates.
    case DataType::BOOLEAN:
        *bytes = 1;
        return true;
    default:
        return false;
    }
}

/// @brief A @p bytes long host image for a tensor of @p dataType filled with the pattern,
///        or empty for a type encodeFillElement() declines. @p positive drops the sign, for an
///        input an operation reads that is undefined on negative values.
inline std::vector<uint8_t> inputFillImage(hipdnn_frontend::DataType dataType,
                                           size_t bytes,
                                           uint64_t seed,
                                           int64_t uid,
                                           bool positive = false)
{
    const auto width = static_cast<size_t>(elementBits(dataType) / 8);
    if(width == 0 || bytes < width)
    {
        return {};
    }
    std::vector<uint8_t> image(bytes, 0);
    for(size_t index = 0; index < bytes / width; ++index)
    {
        auto value = fillValue(fillHash(seed, uid, index));
        value.negative = value.negative && !positive;
        // Refusal depends only on the type, so a half-filled buffer is never returned.
        if(!encodeFillElement(dataType, value, image.data() + (index * width)))
        {
            return {};
        }
    }
    return image;
}

/// @brief The fill seed for one problem: a hash of the serialized graph.
///
/// Every candidate of a problem must read identical inputs for the cross-check to be valid;
/// hashing the bytes also makes the fill reproducible across hosts.
inline uint64_t graphFillSeed(const std::vector<uint8_t>& graphBytes)
{
    uint64_t hash = 0xCBF29CE484222325ULL; // FNV-1a, 64 bit
    for(const uint8_t byte : graphBytes)
    {
        hash ^= byte;
        hash *= 0x100000001B3ULL;
    }
    return hash;
}

/// Where two images of the same tensor first differ, named for the row, or empty when they
/// agree. The rule is output_agreement::firstDisagreementAt(); two all-zero images agree:
/// they are identical, and identical is not a difference.
inline std::string firstDisagreement(const std::string& name,
                                     hipdnn_frontend::DataType dataType,
                                     const std::vector<uint8_t>& reference,
                                     const std::vector<uint8_t>& candidate)
{
    // Not comparable: the caller counts it as skipped rather than as agreement.
    const auto found
        = agreement::firstDisagreementAt(toSdkDataType(dataType), reference, candidate);
    if(!found)
    {
        return {};
    }
    std::ostringstream detailText;
    detailText.precision(3);
    detailText << "tensor '" << name << "' element " << found->index << " is " << std::scientific
               << found->right << " against the catalog's " << found->left
               << ", outside a tolerance of " << found->tolerance;
    return detailText.str();
}

/// True when every comparable tensor of @p candidate is still the zero fill it was
/// allocated with. Unanimous agreement on such output proves nothing, so it is not AGREED.
inline bool leftOutputUntouched(const CandidateOutput& candidate,
                                const std::map<int64_t, TensorDescription>& tensors)
{
    for(const auto& [uid, tensor] : tensors)
    {
        const auto image = candidate.images.find(uid);
        if(!decodable(tensor.dataType) || image == candidate.images.end())
        {
            continue;
        }
        if(std::any_of(
               image->second.begin(), image->second.end(), [](uint8_t byte) { return byte != 0; }))
        {
            return false;
        }
    }
    return true;
}

/// Where @p candidate's output first holds a NaN or an infinity, or empty when every
/// comparable element is finite. Such an output is an error value: candidates that all
/// produce it have failed alike, not agreed.
inline std::string firstNonFinite(const CandidateOutput& candidate,
                                  const std::map<int64_t, TensorDescription>& tensors)
{
    for(const auto& [uid, tensor] : tensors)
    {
        const auto image = candidate.images.find(uid);
        if(!decodable(tensor.dataType) || image == candidate.images.end())
        {
            continue;
        }
        const auto found
            = agreement::firstNonFiniteAt(toSdkDataType(tensor.dataType), image->second);
        if(found)
        {
            std::ostringstream detailText;
            detailText << "tensor '" << tensor.name << "' element " << found->index << " is "
                       << found->value;
            return detailText.str();
        }
    }
    return {};
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
        if(decodable(tensor.dataType) && candidate.images.count(uid) != 0)
        {
            ++comparable;
        }
    }
    return comparable;
}

} // namespace detail

/// @brief One verdict per candidate, in add order (RFC 0019 §13.2), holding one host image
///        per distinct answer rather than one per candidate.
///
/// A candidate whose output holds a NaN or an infinity takes no part in the vote. It is
/// DISAGREED when the finite candidates reach a majority, and UNKNOWN otherwise: an exact
/// result can overflow the output type, so a non-finite output every candidate shares may be
/// correct. The rest are grouped into
/// cohorts with identical output. A strict-majority cohort is
/// AGREED and everything else DISAGREED; without a strict majority every cross-checked
/// candidate is DISAGREED, since some candidate is wrong and first-one-wins could invert the
/// verdicts. Comparison is streaming: only each cohort's founder keeps its images.
class CatalogCrossCheck
{
public:
    /// @param tensors What each uid is. Referenced, not copied: it may be empty at
    ///        construction but must be complete before the first add(), and outlive this.
    explicit CatalogCrossCheck(const std::map<int64_t, TensorDescription>& tensors)
        : _tensors(tensors)
    {
    }

    /// @brief Classifies @p candidate, keeping its images only if it founds a cohort.
    void add(CandidateOutput candidate)
    {
        const size_t index = _outcomes.size();
        _outcomes.emplace_back();
        if(!candidate.executed)
        {
            _outcomes[index]
                = {NumericalVerdict::UNKNOWN,
                   "not_executed: "
                       + (candidate.failure.empty() ? std::string("the candidate produced no "
                                                                  "output to cross-check")
                                                    : candidate.failure)};
            return;
        }
        if(detail::comparableTensors(candidate, _tensors) == 0)
        {
            _outcomes[index]
                = {NumericalVerdict::UNKNOWN,
                   "no_comparable_output: this problem declares no tensor of a type the "
                   "cross-check can decode (RFC 0019 Open Question 19 leaves the per-op "
                   "reference open)"};
            return;
        }
        const auto nonFinite = detail::firstNonFinite(candidate, _tensors);
        if(!nonFinite.empty())
        {
            // An error value: it neither votes nor serves as a reference; verdicts() judges
            // it against the finite majority, if there is one.
            _nonFinite.emplace_back(index, nonFinite);
            return;
        }
        ++_crossChecked;
        for(auto& cohort : _cohorts)
        {
            if(detail::sameOutput(cohort.founder, candidate, _tensors))
            {
                cohort.members.push_back(index);
                return; // `candidate` dies here, and its images with it.
            }
        }
        Cohort founded;
        founded.members.push_back(index);
        founded.founder = std::move(candidate);
        _cohorts.push_back(std::move(founded));
    }

    /// Host image sets held right now: one per distinct answer seen, never one per candidate.
    size_t retainedImages() const
    {
        return static_cast<size_t>(
            std::count_if(_cohorts.begin(), _cohorts.end(), [](const Cohort& cohort) {
                return !cohort.founder.images.empty();
            }));
    }

    /// One outcome per added candidate, in add order.
    std::vector<ValidationOutcome> verdicts() const
    {
        auto outcomes = _outcomes;
        const bool decided = decide(outcomes);
        // A NaN or an infinity is wrong only beside an agreed finite output: the exact result
        // can overflow the output type, or an operation undefined on negative values can read
        // a computed tensor, and then every correct kernel writes it.
        for(const auto& [index, detailText] : _nonFinite)
        {
            outcomes[index] = decided
                                  ? ValidationOutcome{NumericalVerdict::DISAGREED,
                                                      "output_mismatch: " + detailText
                                                          + ", where the agreed output is finite"}
                                  : ValidationOutcome{NumericalVerdict::UNKNOWN,
                                                      "non_finite_output: " + detailText
                                                          + "; no candidate wrote a finite "
                                                            "output to compare, so it may be "
                                                            "correct"};
        }
        return outcomes;
    }

private:
    /// Fills the verdicts of the finite candidates; true when a strict majority agreed.
    ///
    /// A cohort that left the zero-filled output untouched computed nothing, so it never
    /// votes and is never the reference, however many candidates it holds: the majority is
    /// decided among the candidates that wrote their output.
    bool decide(std::vector<ValidationOutcome>& outcomes) const
    {
        if(_cohorts.empty())
        {
            return false;
        }

        if(_crossChecked < 2)
        {
            // One candidate agreeing with itself is not evidence.
            outcomes[_cohorts.front().members.front()]
                = {NumericalVerdict::UNKNOWN,
                   "no_reference: one candidate ran for this problem, so there was nothing to "
                   "cross-check it against"};
            return false;
        }

        std::vector<bool> untouched(_cohorts.size());
        size_t computing = 0;
        std::optional<size_t> largest;
        for(size_t cohort = 0; cohort < _cohorts.size(); ++cohort)
        {
            untouched[cohort] = detail::leftOutputUntouched(_cohorts[cohort].founder, _tensors);
            if(untouched[cohort])
            {
                continue;
            }
            computing += _cohorts[cohort].members.size();
            if(!largest || _cohorts[cohort].members.size() > _cohorts[*largest].members.size())
            {
                largest = cohort;
            }
        }

        // Strict majority of the candidates that computed, written to avoid overflow;
        // necessarily unique. One computing candidate has no corroboration.
        const bool decided
            = largest && computing >= 2 && _cohorts[*largest].members.size() > computing / 2;
        for(size_t cohort = 0; cohort < _cohorts.size(); ++cohort)
        {
            if(untouched[cohort] && !decided)
            {
                for(const size_t index : _cohorts[cohort].members)
                {
                    outcomes[index] = {NumericalVerdict::UNKNOWN,
                                       "degenerate_reference: the candidate left the zero-filled "
                                       "output untouched and no majority of the candidates that "
                                       "wrote their output corroborates an answer"};
                }
                continue;
            }
            if(!untouched[cohort] && computing < 2)
            {
                for(const size_t index : _cohorts[cohort].members)
                {
                    outcomes[index]
                        = {NumericalVerdict::UNKNOWN,
                           "no_reference: no other cross-checked candidate wrote its output, "
                           "so there was nothing to cross-check it against"};
                }
                continue;
            }
            if(decided && cohort == *largest)
            {
                for(const size_t index : _cohorts[cohort].members)
                {
                    outcomes[index]
                        = {NumericalVerdict::AGREED,
                           "agrees_with_catalog: " + std::to_string(_cohorts[cohort].members.size())
                               + " of " + std::to_string(computing)
                               + " cross-checked candidates that wrote their output produced it"};
                }
                continue;
            }
            // Members agree with the founder within tolerance, so its detail stands for all.
            const std::string detailText
                = untouched[cohort]
                      ? "the candidate left the zero-filled output untouched"
                      : mismatchDetail(_cohorts[cohort].founder, _cohorts[*largest].founder);
            for(const size_t index : _cohorts[cohort].members)
            {
                outcomes[index]
                    = {NumericalVerdict::DISAGREED,
                       (decided ? "output_mismatch: " : "disputed_output: ") + detailText};
            }
        }
        return decided;
    }

    struct Cohort
    {
        /// The one candidate of this cohort whose images are kept.
        CandidateOutput founder;

        /// Add-order indices of every candidate that produced this answer, founder first.
        std::vector<size_t> members;
    };

    /// Add-order index and first non-finite element of each candidate kept out of the vote.
    std::vector<std::pair<size_t, std::string>> _nonFinite;

    /// Where @p candidate first parts company with @p reference, named for the row.
    std::string mismatchDetail(const CandidateOutput& candidate,
                               const CandidateOutput& reference) const
    {
        for(const auto& [uid, tensor] : _tensors)
        {
            const auto mine = candidate.images.find(uid);
            const auto theirs = reference.images.find(uid);
            if(mine == candidate.images.end() || theirs == reference.images.end())
            {
                continue;
            }
            const auto difference = detail::firstDisagreement(
                tensor.name, tensor.dataType, theirs->second, mine->second);
            if(!difference.empty())
            {
                return difference;
            }
        }
        return "the candidate's output has no corroboration in the catalog";
    }

    const std::map<int64_t, TensorDescription>& _tensors;

    /// Indexed by add order. Set in add() for unjudgeable candidates; completed by verdicts().
    std::vector<ValidationOutcome> _outcomes;
    std::vector<Cohort> _cohorts;
    size_t _crossChecked = 0;
};

} // namespace hipdnn_bench
