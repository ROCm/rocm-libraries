// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ReferenceValidator.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    po::variables_map validationArgs()
    {
        po::variables_map args;
        args["num-elements-to-validate"].value() = -1;
        args["print-max"].value() = 0;
        for(auto name : {"print-valids", "print-tensor-a", "print-tensor-b", "print-tensor-c",
                         "print-tensor-d", "print-tensor-ref", "print-tensor-bias",
                         "print-tensor-gate", "print-tensor-scale-alpha-vec", "print-tensor-amaxd"})
            args[name].value() = false;
        return args;
    }

    class RecordingValidator : public ReferenceValidator
    {
    public:
        explicit RecordingValidator(po::variables_map const& args)
            : ReferenceValidator(args, {})
        {
        }

        std::vector<std::vector<size_t>> checkedSlots;

    protected:
        void validateSlots(std::vector<size_t> const& slots) override
        {
            checkedSlots.push_back(slots);
        }
    };
}

TEST(ReferenceValidatorLifecycle, EnqueuesAreUncheckedByDefault)
{
    RecordingValidator validator(validationArgs());
    TimingEvents events({}, 0, 0);
    validator.preSolution(nullptr);
    validator.validateWarmups({0}, events, events);
    validator.validateEnqueues({0, 1}, events, events);
    EXPECT_EQ(validator.checkedSlots.size(), 1u);
}

TEST(ReferenceValidatorLifecycle, OptInChecksEveryGroupAfterWarmup)
{
    auto args = validationArgs();
    args["validate-after-enqueues"].value() = true;
    RecordingValidator validator(args);
    TimingEvents events({}, 0, 0);
    std::vector<size_t> const warmup{0}, firstGroup{1, 2}, secondGroup{3, 0};
    validator.preSolution(nullptr);
    validator.validateWarmups(warmup, events, events);
    EXPECT_FALSE(validator.needMoreRunsInSolution());
    validator.validateEnqueues(firstGroup, events, events);
    validator.validateEnqueues(secondGroup, events, events);
    ASSERT_EQ(validator.checkedSlots.size(), 3u);
    EXPECT_EQ(validator.checkedSlots[1], firstGroup);
    EXPECT_EQ(validator.checkedSlots[2], secondGroup);

    validator.preSolution(nullptr);
    EXPECT_TRUE(validator.needMoreRunsInSolution());
    validator.validateWarmups(warmup, events, events);
    EXPECT_EQ(validator.checkedSlots.size(), 4u);
}

TEST(ReferenceValidatorLifecycle, OptInRespectsDisabledReferenceValidation)
{
    auto args = validationArgs();
    args["validate-after-enqueues"].value() = true;
    args["num-elements-to-validate"].value() = 0;
    RecordingValidator validator(args);
    TimingEvents events({}, 0, 0);
    validator.validateEnqueues({0}, events, events);
    EXPECT_TRUE(validator.checkedSlots.empty());
}

TEST(ReferenceValidatorLifecycle, OptInRejectsAliasedReferenceInputs)
{
    auto args = validationArgs();
    args["validate-after-enqueues"].value() = true;
    args["c-equal-d"].value() = true;
    EXPECT_THROW(RecordingValidator{args}, std::runtime_error);

    args["validate-after-enqueues"].value() = false;
    EXPECT_NO_THROW(RecordingValidator{args});
}
