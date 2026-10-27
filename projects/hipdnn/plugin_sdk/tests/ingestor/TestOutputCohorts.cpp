// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/OutputAgreement.hpp>
#include <hipdnn_plugin_sdk/ingestor/OutputCohorts.hpp>

/**
 * @file TestOutputCohorts.cpp
 * @brief Unit tests for OutputCohorts.hpp: how candidate output images are grouped into
 *        cohorts and which cohort passes.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_plugin_sdk::output_agreement::firstDisagreement;
using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
using Verdict = OutputCohorts::Verdict;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

std::vector<uint8_t> floatBytes(std::initializer_list<float> values)
{
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    size_t offset = 0;
    for(const float value : values)
    {
        std::memcpy(bytes.data() + offset, &value, sizeof(float));
        offset += sizeof(float);
    }
    return bytes;
}

OutputBufferSpec floatSpec(int64_t uid, size_t elements)
{
    return {uid, DataType::FLOAT, elements * sizeof(float)};
}

/// What a candidate that never wrote an output of @p dataType leaves behind.
std::vector<uint8_t> fillImage(DataType dataType, size_t bytes)
{
    return std::vector<uint8_t>(bytes, hipdnn_plugin_sdk::output_agreement::fillByte(dataType));
}

/// Submits one single-output candidate.
void submitOne(OutputCohorts& cohorts, size_t index, std::vector<uint8_t> image)
{
    OutputImages images;
    images.push_back(std::move(image));
    cohorts.submit(index, std::move(images));
}

/// 1.0 and 1.000007 agree (relative tolerance 1e-5), as do 1.000007 and 1.000014, but 1.0 and
/// 1.000014 do not: tolerance is not transitive.
std::vector<uint8_t> chainLink(int position)
{
    constexpr float STEP = 7e-6F;
    return floatBytes({1.0F + (static_cast<float>(position) * STEP)});
}

TEST(TestIngestorOutputCohorts, CohortsFormInCandidateIndexOrderUnderNonTransitiveTolerance)
{
    // Precondition: the chain really is non-transitive.
    ASSERT_TRUE(firstDisagreement("o", DataType::FLOAT, chainLink(0), chainLink(1)).empty());
    ASSERT_TRUE(firstDisagreement("o", DataType::FLOAT, chainLink(1), chainLink(2)).empty());
    ASSERT_FALSE(firstDisagreement("o", DataType::FLOAT, chainLink(0), chainLink(2)).empty());

    // A, B, C in index order: B joins A's cohort, C meets only the founder A and splits off.
    OutputCohorts ascending({floatSpec(1, 1)});
    submitOne(ascending, 0, chainLink(0));
    submitOne(ascending, 1, chainLink(1));
    submitOne(ascending, 2, chainLink(2));
    const auto splitDecision = ascending.decide();
    EXPECT_EQ(splitDecision.cohorts, 2U);
    EXPECT_EQ(splitDecision.verdict, Verdict::MAJORITY);
    EXPECT_THAT(splitDecision.passing, ElementsAre(0U, 1U));
    ASSERT_EQ(splitDecision.failed.size(), 1U);
    EXPECT_EQ(splitDecision.failed[0].index, 2U);

    // B, A, C: B is the founder and both others are within tolerance of it.
    OutputCohorts middleFirst({floatSpec(1, 1)});
    submitOne(middleFirst, 0, chainLink(1));
    submitOne(middleFirst, 1, chainLink(0));
    submitOne(middleFirst, 2, chainLink(2));
    const auto mergedDecision = middleFirst.decide();
    EXPECT_EQ(mergedDecision.cohorts, 1U);
    EXPECT_THAT(mergedDecision.passing, ElementsAre(0U, 1U, 2U));
    EXPECT_TRUE(mergedDecision.failed.empty());
}

TEST(TestIngestorOutputCohorts, OnlyCohortFoundersRetainTheirImages)
{
    constexpr size_t IMAGE_BYTES = 4 * sizeof(float);
    OutputCohorts cohorts({floatSpec(1, 4)});

    for(size_t index = 0; index < 5; ++index)
    {
        submitOne(cohorts, index, floatBytes({1.0F, 2.0F, 3.0F, 4.0F}));
    }
    EXPECT_EQ(cohorts.cohortCount(), 1U);
    EXPECT_EQ(cohorts.retainedBytes(), IMAGE_BYTES) << "joiners must not keep their images";

    submitOne(cohorts, 5, floatBytes({9.0F, 9.0F, 9.0F, 9.0F}));
    submitOne(cohorts, 6, floatBytes({9.0F, 9.0F, 9.0F, 9.0F}));
    EXPECT_EQ(cohorts.cohortCount(), 2U);
    EXPECT_EQ(cohorts.retainedBytes(), 2 * IMAGE_BYTES);
}

TEST(TestIngestorOutputCohorts, AStrictMajorityBeatsTheHeuristicFirstCohort)
{
    const auto wrong = floatBytes({9.0F, 9.0F});
    const auto right = floatBytes({1.0F, 2.0F});
    OutputCohorts cohorts({floatSpec(1, 2)});
    submitOne(cohorts, 0, wrong);
    submitOne(cohorts, 1, right);
    submitOne(cohorts, 2, right);

    const auto decision = cohorts.decide();
    EXPECT_EQ(decision.verdict, Verdict::MAJORITY);
    EXPECT_EQ(decision.computing, 3U);
    EXPECT_EQ(decision.cohorts, 2U);
    EXPECT_THAT(decision.passing, ElementsAre(1U, 2U));
    EXPECT_EQ(decision.passingFounder, 1U);
    ASSERT_EQ(decision.failed.size(), 1U);
    EXPECT_EQ(decision.failed[0].index, 0U);
    EXPECT_FALSE(decision.failed[0].untouched);
    EXPECT_EQ(decision.failed[0].cohortFounder, 0U);
    // Founder against founder: the failed cohort's image on the left.
    EXPECT_EQ(decision.failed[0].reason,
              firstDisagreement("output 1", DataType::FLOAT, wrong, right));
    EXPECT_THAT(decision.failed[0].reason, HasSubstr("output 1[0]"));
}

TEST(TestIngestorOutputCohorts, ATieBreakPicksTheLowestTouchedIndexWhenIndexZeroIsUntouched)
{
    OutputCohorts cohorts({floatSpec(1, 2)});
    submitOne(cohorts, 0, fillImage(DataType::FLOAT, 2 * sizeof(float)));
    submitOne(cohorts, 1, floatBytes({1.0F, 2.0F}));
    submitOne(cohorts, 2, floatBytes({5.0F, 6.0F}));

    const auto decision = cohorts.decide();
    EXPECT_EQ(decision.verdict, Verdict::HEURISTIC_TIEBREAK);
    EXPECT_EQ(decision.computing, 2U);
    EXPECT_THAT(decision.passing, ElementsAre(1U));
    ASSERT_EQ(decision.failed.size(), 2U);
    EXPECT_EQ(decision.failed[0].index, 0U);
    EXPECT_TRUE(decision.failed[0].untouched);
    EXPECT_EQ(decision.failed[1].index, 2U);
    EXPECT_FALSE(decision.failed[1].untouched);
}

TEST(TestIngestorOutputCohorts, EqualSizedCohortsKeepTheFirstOne)
{
    const auto first = floatBytes({1.0F});
    const auto second = floatBytes({2.0F});
    OutputCohorts cohorts({floatSpec(1, 1)});
    submitOne(cohorts, 0, first);
    submitOne(cohorts, 1, first);
    submitOne(cohorts, 2, second);
    submitOne(cohorts, 3, second);

    const auto decision = cohorts.decide();
    EXPECT_EQ(decision.verdict, Verdict::HEURISTIC_TIEBREAK) << "2 of 4 is not a strict majority";
    EXPECT_THAT(decision.passing, ElementsAre(0U, 1U));
    ASSERT_EQ(decision.failed.size(), 2U);
    EXPECT_EQ(decision.failed[0].index, 2U);
    EXPECT_EQ(decision.failed[1].index, 3U);
    EXPECT_EQ(decision.failed[1].cohortFounder, 2U);
}

TEST(TestIngestorOutputCohorts, WhenNoCandidateIsTouchedNoneAndEveryUntouchedOneFails)
{
    OutputCohorts cohorts({floatSpec(1, 2)});
    for(size_t index = 0; index < 3; ++index)
    {
        submitOne(cohorts, index, fillImage(DataType::FLOAT, 2 * sizeof(float)));
    }

    const auto decision = cohorts.decide();
    EXPECT_EQ(decision.verdict, Verdict::NONE_PASS);
    EXPECT_TRUE(decision.passing.empty());
    EXPECT_EQ(decision.computing, 0U);
    EXPECT_EQ(decision.cohorts, 0U) << "an untouched candidate belongs to no cohort";
    ASSERT_EQ(decision.failed.size(), 3U);
    for(const auto& failure : decision.failed)
    {
        EXPECT_TRUE(failure.untouched);
        EXPECT_THAT(failure.reason,
                    HasSubstr("output 1 left untouched (still holds the NaN fill)"));
    }
}

TEST(TestIngestorOutputCohorts, ALoneTouchedCandidatePassesUnverified)
{
    OutputCohorts cohorts({floatSpec(1, 1)});
    submitOne(cohorts, 0, fillImage(DataType::FLOAT, sizeof(float)));
    submitOne(cohorts, 1, floatBytes({3.0F}));
    submitOne(cohorts, 2, fillImage(DataType::FLOAT, sizeof(float)));

    const auto decision = cohorts.decide();
    EXPECT_EQ(decision.verdict, Verdict::UNVERIFIED);
    EXPECT_THAT(decision.passing, ElementsAre(1U));
    ASSERT_EQ(decision.failed.size(), 2U);
    EXPECT_EQ(decision.failed[0].index, 0U);
    EXPECT_EQ(decision.failed[1].index, 2U);
}

TEST(TestIngestorOutputCohorts,
     AnOutputNoDtypeCanCompareIsInconclusiveAndEveryTouchedCandidatePasses)
{
    for(const auto dataType : {DataType::FP4_E2M1, DataType::FP6_E2M3, DataType::INT4})
    {
        SCOPED_TRACE(static_cast<int>(dataType));
        OutputCohorts cohorts({{1, dataType, 4}});
        submitOne(cohorts, 0, {0x01, 0x02, 0x03, 0x04});
        submitOne(cohorts, 1, {0x09, 0x08, 0x07, 0x06});

        const auto decision = cohorts.decide();
        EXPECT_EQ(decision.verdict, Verdict::INCONCLUSIVE);
        EXPECT_THAT(decision.passing, ElementsAre(0U, 1U));
        EXPECT_TRUE(decision.failed.empty());
        EXPECT_EQ(cohorts.retainedBytes(), 0U) << "nothing is compared, so nothing is kept";
    }
}

TEST(TestIngestorOutputCohorts, AnFp8OutputIsOnlyCheckedForBeingUntouched)
{
    OutputCohorts cohorts({{1, DataType::FP8_E4M3, 4}});
    submitOne(cohorts, 0, fillImage(DataType::FP8_E4M3, 4));
    submitOne(cohorts, 1, {0x01, 0x02, 0x03, 0x04});
    submitOne(cohorts, 2, {0x09, 0x08, 0x07, 0x06});

    const auto decision = cohorts.decide();
    EXPECT_EQ(decision.verdict, Verdict::INCONCLUSIVE);
    EXPECT_THAT(decision.passing, ElementsAre(1U, 2U));
    ASSERT_EQ(decision.failed.size(), 1U);
    EXPECT_EQ(decision.failed[0].index, 0U);
    EXPECT_TRUE(decision.failed[0].untouched);
}

TEST(TestIngestorOutputCohorts, AnyUntouchedOutputExcludesTheCandidateAndNamesIt)
{
    OutputCohorts cohorts({floatSpec(11, 2), floatSpec(22, 2)});
    {
        OutputImages onlyFirstWritten;
        onlyFirstWritten.push_back(floatBytes({1.0F, 2.0F}));
        onlyFirstWritten.push_back(fillImage(DataType::FLOAT, 2 * sizeof(float)));
        cohorts.submit(0, std::move(onlyFirstWritten));
    }
    {
        OutputImages bothWritten;
        bothWritten.push_back(floatBytes({1.0F, 2.0F}));
        bothWritten.push_back(floatBytes({3.0F, 4.0F}));
        cohorts.submit(1, std::move(bothWritten));
    }

    const auto decision = cohorts.decide();
    EXPECT_THAT(decision.passing, ElementsAre(1U));
    ASSERT_EQ(decision.failed.size(), 1U);
    EXPECT_EQ(decision.failed[0].index, 0U);
    EXPECT_TRUE(decision.failed[0].untouched);
    EXPECT_THAT(decision.failed[0].reason, HasSubstr("output 22 left untouched"));
}

TEST(TestIngestorOutputCohorts, DisagreeingOnAnySingleOutputSeparatesTheCohorts)
{
    const auto shared = floatBytes({1.0F, 2.0F});
    const auto right = floatBytes({3.0F, 4.0F});
    const auto wrong = floatBytes({3.0F, 7.0F});
    OutputCohorts cohorts({floatSpec(11, 2), floatSpec(22, 2)});
    for(size_t index = 0; index < 3; ++index)
    {
        OutputImages images;
        images.push_back(shared);
        images.push_back(index == 0 ? wrong : right);
        cohorts.submit(index, std::move(images));
    }

    const auto decision = cohorts.decide();
    EXPECT_THAT(decision.passing, ElementsAre(1U, 2U));
    ASSERT_EQ(decision.failed.size(), 1U);
    EXPECT_EQ(decision.failed[0].reason,
              firstDisagreement("output 22", DataType::FLOAT, wrong, right));
    EXPECT_THAT(decision.failed[0].reason, HasSubstr("output 22[1]"));
}

TEST(TestIngestorOutputCohorts, SameKindNonFiniteAgreesAndNanVersusFiniteDisagrees)
{
    constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
    constexpr float INF_VALUE = std::numeric_limits<float>::infinity();

    OutputCohorts nanCohorts({floatSpec(1, 2)});
    submitOne(nanCohorts, 0, floatBytes({NAN_VALUE, 1.0F}));
    submitOne(nanCohorts, 1, floatBytes({NAN_VALUE, 1.0F}));
    submitOne(nanCohorts, 2, floatBytes({1.0F, 1.0F}));
    const auto nanDecision = nanCohorts.decide();
    EXPECT_THAT(nanDecision.passing, ElementsAre(0U, 1U));
    ASSERT_EQ(nanDecision.failed.size(), 1U);
    EXPECT_EQ(nanDecision.failed[0].index, 2U);
    EXPECT_THAT(nanDecision.failed[0].reason, HasSubstr("output 1[0]"));

    OutputCohorts infinityCohorts({floatSpec(1, 2)});
    submitOne(infinityCohorts, 0, floatBytes({INF_VALUE, 1.0F}));
    submitOne(infinityCohorts, 1, floatBytes({-INF_VALUE, 1.0F}));
    submitOne(infinityCohorts, 2, floatBytes({INF_VALUE, 1.0F}));
    const auto infinityDecision = infinityCohorts.decide();
    EXPECT_THAT(infinityDecision.passing, ElementsAre(0U, 2U));
    ASSERT_EQ(infinityDecision.failed.size(), 1U);
    EXPECT_EQ(infinityDecision.failed[0].index, 1U);
}

TEST(TestIngestorOutputCohorts, ANonFiniteCandidateStaysInTheVote)
{
    constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
    OutputCohorts cohorts({floatSpec(1, 2)});
    submitOne(cohorts, 0, floatBytes({2.0F, 1.0F}));
    submitOne(cohorts, 1, floatBytes({NAN_VALUE, 1.0F}));
    submitOne(cohorts, 2, floatBytes({NAN_VALUE, 1.0F}));

    const auto decision = cohorts.decide();
    EXPECT_EQ(decision.computing, 3U);
    EXPECT_EQ(decision.verdict, Verdict::MAJORITY);
    EXPECT_THAT(decision.passing, ElementsAre(1U, 2U));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
