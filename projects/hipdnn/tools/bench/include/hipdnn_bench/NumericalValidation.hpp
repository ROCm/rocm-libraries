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

/// @brief One value of the input fill: (-1)^negative * 2^exponent.
///
/// Powers of two, and only the first two, because 1 and 2 are exactly representable in
/// every type encodeFillElement() writes -- double down to fp8 -- so the encode sets an
/// exponent field and never rounds. A rounding step here would have to be right for eight
/// formats to avoid writing a wrong input, and a wrong input is indistinguishable from a
/// wrong kernel once the outputs come back.
///
/// Symmetric, because a reduction over a few thousand same-signed elements of 2 passes
/// fp16's 65504 and the gate would be left comparing two infinities.
struct FillValue
{
    bool negative = false;
    int exponent = 0;
};

/// splitmix64's finalizer over (@p seed, @p uid, @p index).
///
/// Hashed rather than alternating. A strictly alternating +1, -1 cancels a reduction down
/// to zero or to a single element, and two correct kernels that cancel in a different order
/// then differ by the whole magnitude of their operands -- the gate would manufacture a
/// disagreement out of its own fill. Hashed signs leave a K-term reduction at the usual
/// sqrt(K), clear of the noise the comparison has to tolerate.
///
/// The uid takes part so two input tensors of one graph do not receive the same image: an
/// A == B matmul is symmetric, and a kernel that transposed one of them would still agree.
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
/// The refusal is the same argument numericKind() makes, in the other direction: a wrong
/// exponent bias writes a NaN or an infinity into an *input*, every candidate then computes
/// NaN, and the gate condemns a catalog that was fine. The sub-byte and packed types are
/// left out because they are not one element per byte; they keep the zero fill, which is
/// what every tensor had before.
inline bool encodeFillElement(hipdnn_frontend::DataType dataType, FillValue value, uint8_t* bytes)
{
    using hipdnn_frontend::DataType;
    // Integers get magnitude only. An integer input to one of these graphs is more often an
    // index, a count or a sequence length than a summand, and a negative one is not a value
    // the kernel can be asked to survive.
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
        // bfloat16 is the top half of a binary32, so it carries binary32's bias and 7 of
        // its mantissa bits.
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
    // The FNUZ pair spends the all-ones exponent on NaN rather than on infinity, which buys
    // it one more exponent and moves the bias by one.
    case DataType::FP8_E4M3_FNUZ:
        *bytes = static_cast<uint8_t>(powerOfTwoCode(value, 8, 3, 8));
        return true;
    case DataType::FP8_E5M2_FNUZ:
        *bytes = static_cast<uint8_t>(powerOfTwoCode(value, 16, 2, 8));
        return true;
    // A block scale: no sign and no mantissa, so the code is the biased exponent alone. It
    // is filled rather than skipped because a zero scale zeroes the tensor it scales, which
    // is the all-zero output this fill exists to stop.
    case DataType::FP8_E8M0: *bytes = static_cast<uint8_t>(127 + value.exponent); return true;
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
    case DataType::UINT8: *bytes = static_cast<uint8_t>(magnitude); return true;
    // A mask of every element true. The other choice zeroes whatever it gates, and a fill
    // that switches the graph off is the state this replaces.
    case DataType::BOOLEAN: *bytes = 1; return true;
    default: return false;
    }
}

/// @brief A @p bytes long host image for a tensor of @p dataType filled with the pattern,
///        or empty for a type encodeFillElement() declines.
inline std::vector<uint8_t>
    inputFillImage(hipdnn_frontend::DataType dataType, size_t bytes, uint64_t seed, int64_t uid)
{
    const size_t width = static_cast<size_t>(elementBits(dataType) / 8);
    if(width == 0 || bytes < width)
    {
        return {};
    }
    std::vector<uint8_t> image(bytes, 0);
    for(size_t index = 0; index < bytes / width; ++index)
    {
        // The refusal depends only on the type, so this returns on the first element or on
        // none of them; a half-filled buffer is never handed back.
        if(!encodeFillElement(
               dataType, fillValue(fillHash(seed, uid, index)), image.data() + (index * width)))
        {
            return {};
        }
    }
    return image;
}

/// @brief The fill seed for one problem: a hash of the serialized graph.
///
/// Derived from the graph, and this is the property the cross-check rests on rather than a
/// convenience: every candidate of one problem must read the *same* inputs, or two kernels
/// that compute the same function would leave different outputs and the gate would report a
/// disagreement it created itself. A clock, an address or a candidate index all break that.
/// Taking it from the bytes -- which carry the graph's own id -- also makes the fill
/// identical on every host, so a disagreement a fleet run reports can be reproduced on a
/// desk by rerunning the same graph file.
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

/// Relative tolerance two correct kernels may differ by -- the `rtol` of the mixed
/// comparison firstDisagreement() applies.
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

/// The absolute term's coefficient: 2^-mantissa bits, so `agreementFloor(kind) * scale` is
/// one ulp at the tensor's own magnitude.
///
/// That is the step between the two representable numbers either side of that magnitude --
/// the granularity at which the type stops distinguishing values at all. A difference below
/// it is the representation, not the kernel.
inline double agreementFloor(NumericKind kind)
{
    switch(kind)
    {
    case NumericKind::FLOAT64: return 0x1p-52; // 52 mantissa bits
    case NumericKind::FLOAT32: return 0x1p-23;
    case NumericKind::FLOAT16: return 0x1p-10;
    case NumericKind::BFLOAT16: return 0x1p-7;
    default: return 0.0; // integers are exact; their bar is equality
    }
}

/// Where two images of the same tensor first differ, or empty when they agree.
///
/// `|a - b| <= rtol * max(|a|, |b|) + atol`, per element. Both terms are load-bearing:
///
///  - A single absolute bar taken from the largest element -- what this did until the
///    RFC 0019 §13.2 review -- lets every small element through. An fp16 tensor whose
///    largest element is 40 gave *every* element a bar of 2e-2 * 40 = 0.8, which is 800x
///    the magnitude of an element of 0.001: a kernel that wrote garbage everywhere except
///    the peak agreed with the catalog. `rtol * max(|a|, |b|)` judges an element against
///    itself instead.
///  - Pure per-element relative error divides by noise, which is the objection that bar was
///    answering and it is a correct objection: an element that is itself near zero is the
///    residue of terms that cancelled, it carries their rounding rather than its own, and
///    its counterpart from a kernel that cancelled in a different order differs from it by
///    an unbounded ratio. `atol` answers that without giving up the relative term -- it is
///    one ulp at the tensor's magnitude, so an element down in the noise is judged against
///    the noise, and an element at full magnitude is judged against itself.
///
/// Two all-zero images agree: they are identical, and identical is not a difference.
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

    // The tensor's magnitude, which only the absolute term uses: the noise an element near
    // zero carries came from terms the size of the tensor, not the size of the element.
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
    const double relative = agreementTolerance(kind);
    const double absolute = agreementFloor(kind) * scale;

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
        const double threshold = (relative * std::max(std::abs(left), std::abs(right))) + absolute;
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
/// The tool writes the inputs before the untimed re-run, but the output buffers keep their
/// zero fill, so a kernel that writes nothing and a kernel whose result is genuinely zero
/// still leave the same bytes behind. Unanimous agreement on an all-zero output therefore
/// proves nothing at all, and reporting it as AGREED would be the silent "valid" RFC 0019
/// §13.2 forbids.
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

/// @brief One verdict per candidate, in add order (RFC 0019 §13.2), holding one host image
///        per distinct answer rather than one per candidate.
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
///
/// **Streaming, because the batch form did not fit in host memory.** Comparison happens as
/// each candidate arrives, and only a cohort's founder keeps its image; a candidate that
/// joins an existing cohort is released the moment add() returns. The old form held every
/// candidate's image until the last one had run: a 60-candidate sweep of a problem whose
/// non-virtual tensors total 96 MB held 60 x 96 MB = 5.6 GB of host images at once, and
/// `uhd_gen generate` now drives nothing but --sweep. The peak is one image per *distinct
/// answer* plus the candidate being classified: 2 x 96 MB = 192 MB for the ordinary case of
/// one answer, 3 x 96 MB when one candidate is broken. It grows with disagreements, which
/// are a handful, rather than with candidates, which are tens.
class CatalogCrossCheck
{
public:
    /// @param tensors What each uid is, referenced rather than copied: the tool learns the
    /// tensor set from the first capture, so the map is still empty at construction and
    /// complete before the first add(). It must outlive this object.
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
            _outcomes[index] = {NumericalVerdict::UNKNOWN,
                                "not_executed: " + (candidate.failure.empty()
                                                        ? std::string("the candidate produced no "
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
        if(_cohorts.empty())
        {
            return outcomes;
        }
        size_t largest = 0;
        for(size_t cohort = 0; cohort < _cohorts.size(); ++cohort)
        {
            if(_cohorts[cohort].members.size() > _cohorts[largest].members.size())
            {
                largest = cohort;
            }
        }
        // Counted after `largest` is final: a running tally would miss an earlier cohort that
        // the eventual winner only matched in size, and report a split as decided.
        const size_t tied = static_cast<size_t>(
            std::count_if(_cohorts.begin(), _cohorts.end(), [&](const Cohort& cohort) {
                return cohort.members.size() == _cohorts[largest].members.size();
            }));

        if(_crossChecked < 2)
        {
            // One candidate agreeing with itself is not evidence, so the row says so rather
            // than claiming a verdict the run did not earn.
            outcomes[_cohorts[largest].members.front()]
                = {NumericalVerdict::UNKNOWN,
                   "no_reference: one candidate ran for this problem, so there was nothing to "
                   "cross-check it against"};
            return outcomes;
        }

        if(_cohorts.size() == 1 && detail::leftOutputUntouched(_cohorts[0].founder, _tensors))
        {
            for(const size_t index : _cohorts[0].members)
            {
                outcomes[index] = {NumericalVerdict::UNKNOWN,
                                   "degenerate_reference: every cross-checked candidate left the "
                                   "zero-filled output untouched, so their agreement is not "
                                   "evidence that any of them computed anything"};
            }
            return outcomes;
        }

        const bool decided = tied == 1;
        const CandidateOutput& reference = _cohorts[decided ? largest : 0].founder;
        for(size_t cohort = 0; cohort < _cohorts.size(); ++cohort)
        {
            if(decided && cohort == largest)
            {
                for(const size_t index : _cohorts[cohort].members)
                {
                    outcomes[index]
                        = {NumericalVerdict::AGREED,
                           "agrees_with_catalog: "
                               + std::to_string(_cohorts[cohort].members.size()) + " of "
                               + std::to_string(_crossChecked)
                               + " cross-checked candidates produced this output"};
                }
                continue;
            }
            // One detail for the whole cohort, read off its founder. Every member agreed
            // with that founder to within this gate's own tolerance, so the founder's
            // element *is* the member's element to the only precision the gate can
            // distinguish -- which is what lets one image stand in for all of them.
            const std::string detailText = mismatchDetail(_cohorts[cohort].founder, reference);
            for(const size_t index : _cohorts[cohort].members)
            {
                outcomes[index] = {NumericalVerdict::DISAGREED,
                                   (decided ? "output_mismatch: " : "disputed_output: ")
                                       + detailText};
            }
        }
        return outcomes;
    }

private:
    struct Cohort
    {
        /// The one candidate of this cohort whose images are kept.
        CandidateOutput founder;

        /// Add-order indices of every candidate that produced this answer, founder first.
        std::vector<size_t> members;
    };

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

    /// Indexed by add order. Filled at add() for the candidates no cohort can judge, and
    /// completed by verdicts() once the whole catalog has been seen.
    std::vector<ValidationOutcome> _outcomes;
    std::vector<Cohort> _cohorts;
    size_t _crossChecked = 0;
};

} // namespace hipdnn_bench
