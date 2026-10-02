// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestNumericalValidation.cpp
 * @brief The correctness gate RFC 0019 §13.2 puts in front of a training label.
 *
 * Tested here rather than through the tool because the decision needs no device: it is a
 * function of the output images, and the images are the only thing a GPU contributes. What
 * a device run could not test any better, and what these cases exist for, is that the gate
 * fails in the right direction. Both directions are damaging and only one is visible:
 *
 *  - Too permissive, and the wrong-but-fast kernel keeps the best time in its group and
 *    becomes the label the ranker is trained to prefer -- §13.2's inverted oracle, which
 *    nothing downstream catches because §6.3 fingerprints the feature contract, not output.
 *  - Too strict, and every honest candidate is marked invalid, promotion is blocked on a
 *    defect that does not exist, and the next author to see it learns to ignore the gate.
 */

#include <hipdnn_bench/NumericalValidation.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{

constexpr int64_t OUTPUT_UID = 7;

std::map<int64_t, hipdnn_bench::TensorDescription> tensors(hipdnn_frontend::DataType dataType
                                                           = hipdnn_frontend::DataType::FLOAT)
{
    return {{OUTPUT_UID, {"Y", dataType}}};
}

/// A candidate that executed and left @p values in the single output tensor.
hipdnn_bench::CandidateOutput ran(const std::vector<float>& values)
{
    hipdnn_bench::CandidateOutput candidate;
    candidate.executed = true;
    std::vector<uint8_t> image(values.size() * sizeof(float));
    std::memcpy(image.data(), values.data(), image.size());
    candidate.images[OUTPUT_UID] = std::move(image);
    return candidate;
}

/// The same, for a tensor of raw 16-bit codes (half, bfloat16) or an opaque dtype.
hipdnn_bench::CandidateOutput ranRaw(const std::vector<uint8_t>& bytes)
{
    hipdnn_bench::CandidateOutput candidate;
    candidate.executed = true;
    candidate.images[OUTPUT_UID] = bytes;
    return candidate;
}

std::vector<uint8_t> halfCodes(const std::vector<uint16_t>& codes)
{
    std::vector<uint8_t> bytes(codes.size() * sizeof(uint16_t));
    std::memcpy(bytes.data(), codes.data(), bytes.size());
    return bytes;
}

using hipdnn_bench::NumericalVerdict;

/// Every candidate of one problem, cross-checked in one go.
///
/// The gate itself is streaming -- it keeps one image per distinct answer, not one per
/// candidate -- but the verdicts are a function of the whole catalog, so most cases here
/// read better as a batch. The case that is about the retention itself drives add()
/// directly.
std::vector<hipdnn_bench::ValidationOutcome>
    crossCheck(std::vector<hipdnn_bench::CandidateOutput> candidates,
               const std::map<int64_t, hipdnn_bench::TensorDescription>& tensors)
{
    hipdnn_bench::CatalogCrossCheck check(tensors);
    for(auto& candidate : candidates)
    {
        check.add(std::move(candidate));
    }
    return check.verdicts();
}

/// A graph with one input X (uid 1, FLOAT) and one result Y (uid 2, @p outputType): Y = X * X.
hipdnn_bench::VariantPackPlan inputAndOutputPlan(hipdnn_frontend::DataType outputType)
{
    using hipdnn_frontend::graph::TensorAttributes;
    hipdnn_frontend::graph::Graph graph;
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X").set_dim({2}).set_stride({1}).set_data_type(
        hipdnn_frontend::DataType::FLOAT);
    hipdnn_frontend::graph::PointwiseAttributes square;
    square.set_mode(hipdnn_frontend::PointwiseMode::MUL);
    auto y = graph.pointwise(x, x, square);
    y->set_uid(2).set_name("Y").set_dim({2}).set_stride({1}).set_data_type(outputType);
    y->set_output(true);
    return hipdnn_bench::planVariantPack(graph);
}

/// What the tool reads back after a candidate ran, given what device memory then holds for
/// every uid -- the selection is the tool's own (crossCheckedOutputs), not this test's.
hipdnn_bench::CandidateOutput captured(const hipdnn_bench::VariantPackPlan& plan,
                                       const std::map<int64_t, std::vector<uint8_t>>& memory,
                                       std::map<int64_t, hipdnn_bench::TensorDescription>& tensors)
{
    hipdnn_bench::CandidateOutput candidate;
    candidate.executed = true;
    for(const auto& tensor : hipdnn_bench::crossCheckedOutputs(plan))
    {
        tensors[tensor.uid] = {tensor.name, tensor.dataType};
        candidate.images[tensor.uid] = memory.at(tensor.uid);
    }
    return candidate;
}

std::vector<uint8_t> floatBytes(const std::vector<float>& values)
{
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

} // namespace

TEST(TestNumericalValidation, IdenticalInputsAreNotAgreementAboutUndecodableOutputs)
{
    // Every candidate is handed the same filled inputs, so an input compares equal across the
    // catalog whatever the kernels computed. Counted as output, it made two candidates with
    // different FP8 results -- nothing the gate can decode -- come back AGREED.
    const auto plan = inputAndOutputPlan(hipdnn_frontend::DataType::FP8_E4M3);
    ASSERT_TRUE(plan.error.empty()) << plan.error;
    const auto input = floatBytes({1.0F, 2.0F});

    std::map<int64_t, hipdnn_bench::TensorDescription> declared;
    hipdnn_bench::CatalogCrossCheck check(declared);
    check.add(captured(plan, {{1, input}, {2, {0x01, 0x02}}}, declared));
    check.add(captured(plan, {{1, input}, {2, {0x40, 0x50}}}, declared));
    const auto verdicts = check.verdicts();

    for(const auto& verdict : verdicts)
    {
        EXPECT_EQ(verdict.verdict, NumericalVerdict::UNKNOWN) << verdict.reason;
        EXPECT_NE(verdict.reason.find("no_comparable_output"), std::string::npos);
    }
}

TEST(TestNumericalValidation, NonZeroInputsDoNotMakeAnUntouchedOutputEvidence)
{
    // The inputs are filled, the output buffers are not. A catalog that wrote nothing leaves
    // every Y at zero; counting the filled X as output made that set look touched, and the
    // degenerate-reference guard never fired.
    const auto plan = inputAndOutputPlan(hipdnn_frontend::DataType::FLOAT);
    ASSERT_TRUE(plan.error.empty()) << plan.error;
    const std::map<int64_t, std::vector<uint8_t>> memory{{1, floatBytes({1.0F, 2.0F})},
                                                         {2, floatBytes({0.0F, 0.0F})}};

    std::map<int64_t, hipdnn_bench::TensorDescription> declared;
    hipdnn_bench::CatalogCrossCheck check(declared);
    check.add(captured(plan, memory, declared));
    check.add(captured(plan, memory, declared));
    check.add(captured(plan, memory, declared));

    for(const auto& verdict : check.verdicts())
    {
        EXPECT_EQ(verdict.verdict, NumericalVerdict::UNKNOWN) << verdict.reason;
        EXPECT_NE(verdict.reason.find("degenerate_reference"), std::string::npos);
    }
}

TEST(TestNumericalValidation, APluralityIsNotAgreement)
{
    // {1, 1, 2, 3}: the largest cohort is unique but holds two of four. Half the catalog
    // computed something else, so the cohort is not the catalog's answer and its timings are
    // not known-correct labels.
    const auto verdicts
        = crossCheck({ran({1.0F}), ran({1.0F}), ran({2.0F}), ran({3.0F})}, tensors());

    for(const auto& verdict : verdicts)
    {
        EXPECT_EQ(verdict.verdict, NumericalVerdict::DISAGREED) << verdict.reason;
        EXPECT_NE(verdict.reason.find("disputed_output"), std::string::npos);
    }

    // One more vote for the cohort makes it three of five: a strict majority, which decides.
    const auto decided
        = crossCheck({ran({1.0F}), ran({1.0F}), ran({2.0F}), ran({3.0F}), ran({1.0F})}, tensors());
    EXPECT_EQ(decided[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(decided[2].verdict, NumericalVerdict::DISAGREED);
    EXPECT_NE(decided[2].reason.find("output_mismatch"), std::string::npos);
}

TEST(TestNumericalValidation, WrongKernelIsMarkedInvalidAndNamedInTheReason)
{
    // The case the whole gate exists for: two candidates compute the problem and a third
    // returns something else. Without this the third keeps whatever time it measured and,
    // because not computing the answer is fast, that time is the group's best.
    const auto verdicts = crossCheck(
        {ran({1.0F, 2.0F, 3.0F}), ran({1.0F, 2.0F, 3.0F}), ran({1.0F, 2.0F, 99.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::DISAGREED);
    // Named, because "a candidate is invalid" does not tell an author where to look and
    // §13.2's remedy is a matcher or kernel change they have to make by hand.
    EXPECT_NE(verdicts[2].reason.find("output_mismatch"), std::string::npos);
    EXPECT_NE(verdicts[2].reason.find("tensor 'Y' element 2"), std::string::npos);
}

TEST(TestNumericalValidation, MajorityDecidesWhenTheCatalogsFirstCandidateIsTheBrokenOne)
{
    // Taking candidate 0 as the reference is the obvious implementation and it inverts the
    // verdicts exactly when the gate matters most: the broken kernel would be declared the
    // truth and every correct one marked invalid, blocking promotion on the wrong pack.
    const auto verdicts
        = crossCheck({ran({99.0F, 99.0F}), ran({1.0F, 2.0F}), ran({1.0F, 2.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::DISAGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::AGREED);
}

TEST(TestNumericalValidation, RoundingDifferencesBetweenCorrectKernelsDoNotFailTheGate)
{
    // Two kernels that tile a reduction differently accumulate in a different order, so
    // their last bits differ by construction. A bitwise gate marks both invalid, which
    // blocks every promotion and is a worse failure than having no gate at all.
    const auto verdicts
        = crossCheck({ran({1000.0F, 2000.0F}), ran({1000.0001F, 1999.9999F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
}

TEST(TestNumericalValidation, AnUncorroboratedCandidateIsUnknownRatherThanValid)
{
    // One candidate agreeing with itself is not evidence. Reporting it valid is the silent
    // "we did not check" that reads as "we checked", which §13.2 forbids by name.
    const auto verdicts = crossCheck({ran({1.0F, 2.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::UNKNOWN);
    EXPECT_NE(verdicts[0].reason.find("no_reference"), std::string::npos);
}

TEST(TestNumericalValidation, UnanimousUntouchedOutputIsNotEvidenceOfCorrectness)
{
    // The tool fills inputs now, but the output buffers still arrive zero-filled, so for
    // many operations a correct kernel whose result is zero and a kernel that writes
    // nothing leave the same bytes. Unanimity on zeros would otherwise certify a catalog in
    // which nothing ran.
    const auto verdicts
        = crossCheck({ran({0.0F, 0.0F}), ran({0.0F, 0.0F}), ran({0.0F, 0.0F})}, tensors());

    for(const auto& verdict : verdicts)
    {
        EXPECT_EQ(verdict.verdict, NumericalVerdict::UNKNOWN);
        EXPECT_NE(verdict.reason.find("degenerate_reference"), std::string::npos);
    }
}

TEST(TestNumericalValidation, AnEvenSplitLeavesNoCandidateTrusted)
{
    // Two candidates, two answers: one of them is wrong and nothing here can say which. The
    // timing of a candidate that is not known correct is not a label (§13.2), so both are
    // suppressed and emission blocks until the author resolves it in the pack.
    const auto verdicts = crossCheck({ran({1.0F}), ran({5.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::DISAGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::DISAGREED);
    EXPECT_NE(verdicts[0].reason.find("disputed_output"), std::string::npos);
}

TEST(TestNumericalValidation, NonFiniteOutputDisagreesWithAFiniteReference)
{
    // A NaN fails every magnitude comparison it takes part in, so a candidate that produced
    // one would slip through a gate written as `abs(a - b) > tolerance` alone.
    const auto verdicts
        = crossCheck({ran({1.0F, 2.0F}), ran({1.0F, 2.0F}), ran({1.0F, std::nanf("")})}, tensors());

    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::DISAGREED);
}

TEST(TestNumericalValidation, HalfPrecisionIsDecodedRatherThanComparedAsBytes)
{
    // The binary16 decode is written out by hand here, and a wrong one is silent: it would
    // either wave a broken kernel through or condemn a good one. 0x3C00 is 1.0, 0x4000 is
    // 2.0 and 0x3C01 is one ulp above 1.0 -- inside tolerance -- while 0x4400 is 4.0.
    const auto agreeing
        = crossCheck({ranRaw(halfCodes({0x3C00, 0x4000})), ranRaw(halfCodes({0x3C01, 0x4000}))},
                     tensors(hipdnn_frontend::DataType::HALF));
    EXPECT_EQ(agreeing[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(agreeing[1].verdict, NumericalVerdict::AGREED);

    const auto split = crossCheck({ranRaw(halfCodes({0x3C00, 0x4000})),
                                   ranRaw(halfCodes({0x3C00, 0x4000})),
                                   ranRaw(halfCodes({0x3C00, 0x4400}))},
                                  tensors(hipdnn_frontend::DataType::HALF));
    EXPECT_EQ(split[2].verdict, NumericalVerdict::DISAGREED);
}

TEST(TestNumericalValidation, AnUndecodableDtypeIsUnknownRatherThanAssumedEqual)
{
    // FP8 and the packed types are deliberately outside the decoder (Open Question 19 leaves
    // the per-op reference open). Skipping such a tensor silently would make every candidate
    // of an FP8 problem compare equal on nothing at all and come back valid.
    const auto verdicts = crossCheck({ranRaw({0x01, 0x02}), ranRaw({0x40, 0x50})},
                                     tensors(hipdnn_frontend::DataType::FP8_E4M3));

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::UNKNOWN);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::UNKNOWN);
    EXPECT_NE(verdicts[0].reason.find("no_comparable_output"), std::string::npos);
}

TEST(TestNumericalValidation, ACandidateThatNeverRanNeitherJoinsNorSplitsACohort)
{
    // A candidate that could not be built is a coverage gap, not a corruption (§13.2): it is
    // recorded with its reason, and it must not count as a dissenting answer -- one build
    // failure would otherwise turn a unanimous catalog into an unresolvable even split.
    hipdnn_bench::CandidateOutput failed;
    failed.failure = "engine declined to build this configuration";

    const auto verdicts = crossCheck({ran({1.0F, 2.0F}), failed, ran({1.0F, 2.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::UNKNOWN);
    EXPECT_NE(verdicts[1].reason.find("engine declined"), std::string::npos);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::AGREED);
}

TEST(TestNumericalValidation, TheVerdictColumnCannotBeReadBackAsABoolean)
{
    // Three states in a CSV column that a reader will try to coerce. "Unknown" is spelled as
    // a word precisely so `astype(bool)` fails loudly instead of folding it into True.
    EXPECT_STREQ(hipdnn_bench::verdictText(NumericalVerdict::AGREED), "True");
    EXPECT_STREQ(hipdnn_bench::verdictText(NumericalVerdict::DISAGREED), "False");
    EXPECT_STREQ(hipdnn_bench::verdictText(NumericalVerdict::UNKNOWN), "Unknown");
}

TEST(TestNumericalValidation, GarbageInASmallElementIsNotHiddenByTheTensorsLargest)
{
    // The bar used to be one absolute threshold for the whole tensor: tolerance times the
    // largest element, here 1e-5 * 40 = 4e-4. The third candidate's element 1 is wrong by
    // 2e-4 -- twice its own value, and well inside that shared bar -- so a kernel whose
    // small elements are garbage and whose peak is right agreed with the catalog. This is
    // the fp16 case from the §13.2 review scaled to fp32: nothing about it needs a large
    // tensor, only one element far below the largest.
    const auto verdicts = crossCheck(
        {ran({40.0F, 1.0e-4F}), ran({40.0F, 1.0e-4F}), ran({40.0F, 3.0e-4F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::DISAGREED);

    // The element, both values, and the threshold that element was judged against, because
    // a bar of 4e-4 and a bar of 5e-6 are the difference between a gate and a formality.
    EXPECT_NE(verdicts[2].reason.find("element 1"), std::string::npos);
    EXPECT_NE(verdicts[2].reason.find("3.000e-04"), std::string::npos);
    EXPECT_NE(verdicts[2].reason.find("1.000e-04"), std::string::npos);
    EXPECT_NE(verdicts[2].reason.find("outside a tolerance of"), std::string::npos);
}

TEST(TestNumericalValidation, EveryCandidateOfAProblemReadsTheSameNonZeroInputs)
{
    // Agreement on a graph whose inputs were all zero is agreement on a bias term: a wrong
    // reduction order, a wrong mask and a wrong tile boundary are all bit-identical on zero
    // input, so `agrees_with_catalog` would claim more than the run tested. The fill is what
    // the candidates read instead, and four of its properties are load-bearing.
    using hipdnn_frontend::DataType;
    constexpr size_t ELEMENTS = 16;
    const uint64_t seed = hipdnn_bench::detail::graphFillSeed({0x01, 0x02, 0x03});
    const auto image = hipdnn_bench::detail::inputFillImage(
        DataType::FLOAT, ELEMENTS * sizeof(float), seed, OUTPUT_UID);
    ASSERT_EQ(image.size(), ELEMENTS * sizeof(float));

    // Not zero, or the graph is still running on the allocator's fill.
    EXPECT_NE(std::count(image.begin(), image.end(), uint8_t{0}),
              static_cast<std::ptrdiff_t>(image.size()));

    // Identical for every candidate of one problem. This is the property the cross-check
    // rests on: two kernels computing the same function must be handed the same bytes, or
    // the gate reports a disagreement it manufactured itself.
    EXPECT_EQ(image,
              hipdnn_bench::detail::inputFillImage(
                  DataType::FLOAT, ELEMENTS * sizeof(float), seed, OUTPUT_UID));

    // Different per tensor and per graph. Two input tensors filled alike make an A == B
    // matmul symmetric, and a kernel that transposed one of them would still agree.
    EXPECT_NE(image,
              hipdnn_bench::detail::inputFillImage(
                  DataType::FLOAT, ELEMENTS * sizeof(float), seed, OUTPUT_UID + 1));
    EXPECT_NE(image,
              hipdnn_bench::detail::inputFillImage(
                  DataType::FLOAT,
                  ELEMENTS * sizeof(float),
                  hipdnn_bench::detail::graphFillSeed({0x01, 0x02, 0x04}),
                  OUTPUT_UID));

    // Every value is 1 or 2 in magnitude: exactly representable in every type the encoder
    // writes, and small enough that a reduction over a filled tensor does not reach fp16's
    // 65504 and leave the gate comparing two infinities.
    const auto halfImage = hipdnn_bench::detail::inputFillImage(
        DataType::HALF, ELEMENTS * sizeof(uint16_t), seed, OUTPUT_UID);
    ASSERT_EQ(halfImage.size(), ELEMENTS * sizeof(uint16_t));
    for(size_t index = 0; index < ELEMENTS; ++index)
    {
        const double single
            = std::abs(hipdnn_bench::detail::decodeElement(image, index, DataType::FLOAT));
        const double half
            = std::abs(hipdnn_bench::detail::decodeElement(halfImage, index, DataType::HALF));
        EXPECT_TRUE(single == 1.0 || single == 2.0) << "element " << index << " is " << single;
        EXPECT_TRUE(half == 1.0 || half == 2.0) << "element " << index << " is " << half;
    }

    // A type the encoder cannot write exactly keeps the zero fill rather than a guess: a
    // wrong code in an input makes every candidate compute NaN, and the gate would then
    // condemn a catalog that was fine.
    EXPECT_TRUE(hipdnn_bench::detail::inputFillImage(DataType::FP4_E2M1, ELEMENTS, seed, OUTPUT_UID)
                    .empty());
}

TEST(TestNumericalValidation, ACandidateThatJoinsACohortDoesNotKeepItsImage)
{
    // Holding one host image per candidate is tens of GB for a 60-candidate sweep, and
    // --sweep is the only mode `uhd_gen generate` drives. Only an answer nobody has seen
    // before has to be kept: every later candidate is judged against the founder that
    // already holds it.
    const auto declared = tensors();
    hipdnn_bench::CatalogCrossCheck check(declared);

    check.add(ran({1.0F, 2.0F}));
    EXPECT_EQ(check.retainedImages(), 1U);
    check.add(ran({1.0F, 2.0F}));
    EXPECT_EQ(check.retainedImages(), 1U);
    check.add(ran({1.0F, 2.0F}));
    EXPECT_EQ(check.retainedImages(), 1U);
    // A new answer is the one thing that does have to be kept -- it is the evidence the
    // minority verdict is written from.
    check.add(ran({1.0F, 99.0F}));
    EXPECT_EQ(check.retainedImages(), 2U);

    // Releasing the joiners changes nothing a row can see.
    const auto verdicts = check.verdicts();
    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[3].verdict, NumericalVerdict::DISAGREED);
    EXPECT_NE(verdicts[3].reason.find("tensor 'Y' element 1"), std::string::npos);
}

namespace
{

/// The comparison as it was written before it was specialized per type: every element
/// decoded through decodeElement(), in the same two passes. Kept here as the oracle the
/// specialized form is held to.
std::string perElementDisagreement(const std::string& name,
                                   hipdnn_frontend::DataType dataType,
                                   const std::vector<uint8_t>& reference,
                                   const std::vector<uint8_t>& candidate)
{
    namespace detail = hipdnn_bench::detail;
    const auto kind = detail::numericKind(dataType);
    const auto width = static_cast<size_t>(hipdnn_bench::elementBits(dataType) / 8);
    if(kind == detail::NumericKind::NONE || width == 0 || reference.size() != candidate.size())
    {
        return {};
    }
    const size_t count = reference.size() / width;
    double scale = 0.0;
    for(size_t index = 0; index < count; ++index)
    {
        const double left = detail::decodeElement(reference, index, dataType);
        const double right = detail::decodeElement(candidate, index, dataType);
        if(std::isfinite(left))
        {
            scale = std::max(scale, std::abs(left));
        }
        if(std::isfinite(right))
        {
            scale = std::max(scale, std::abs(right));
        }
    }
    const double relative = detail::agreementTolerance(kind);
    const double absolute = detail::agreementFloor(kind) * scale;
    for(size_t index = 0; index < count; ++index)
    {
        const double left = detail::decodeElement(reference, index, dataType);
        const double right = detail::decodeElement(candidate, index, dataType);
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
            std::ostringstream detailText;
            detailText.precision(3);
            detailText << "tensor '" << name << "' element " << index << " is " << std::scientific
                       << right << " against the catalog's " << left << ", outside a tolerance of "
                       << threshold;
            return detailText.str();
        }
    }
    return {};
}

} // namespace

TEST(TestNumericalValidation, TheTypedComparisonDecidesEveryPairAsThePerElementOneDid)
{
    // The comparison picks its decoder once per tensor and skips the decode for byte-identical
    // images, because decoding element by element dominated a large sweep's host time. Neither
    // may move a verdict or the element a mismatch names, so both are held to the per-element
    // form over every decodable type, on random bit patterns -- NaNs, infinities and
    // denormals included -- and on near-copies that differ in their last bits or in one
    // element by a lot.
    using hipdnn_frontend::DataType;
    std::mt19937_64 random(0x5EEDULL);
    for(const auto dataType : {DataType::DOUBLE,
                               DataType::FLOAT,
                               DataType::HALF,
                               DataType::BFLOAT16,
                               DataType::INT64,
                               DataType::INT32,
                               DataType::INT8,
                               DataType::UINT8,
                               DataType::BOOLEAN})
    {
        const auto width = static_cast<size_t>(hipdnn_bench::elementBits(dataType) / 8);
        for(int trial = 0; trial < 200; ++trial)
        {
            std::vector<uint8_t> reference(width * 64);
            for(auto& byte : reference)
            {
                byte = static_cast<uint8_t>(random());
            }
            auto candidate = reference;
            switch(trial % 4)
            {
            case 0: // byte-identical, the shortcut's case
                break;
            case 1: // the lowest byte of a few elements, rounding-sized for the float types
                for(int flip = 0; flip < 3; ++flip)
                {
                    candidate[(random() % 64) * width]
                        ^= static_cast<uint8_t>(1U << (random() % 3));
                }
                break;
            case 2: // one element replaced outright
            {
                const size_t element = random() % 64;
                for(size_t byte = 0; byte < width; ++byte)
                {
                    candidate[(element * width) + byte] = static_cast<uint8_t>(random());
                }
                break;
            }
            default: // unrelated images
                for(auto& byte : candidate)
                {
                    byte = static_cast<uint8_t>(random());
                }
            }
            EXPECT_EQ(hipdnn_bench::detail::firstDisagreement("Y", dataType, reference, candidate),
                      perElementDisagreement("Y", dataType, reference, candidate))
                << "dtype " << static_cast<int>(dataType) << " trial " << trial;
        }
    }
}
