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
            : ReferenceValidator(args, nullptr)
        {
        }

        std::vector<std::shared_ptr<ProblemInputs>> checkedInputs;

    protected:
        bool validateSolution(std::shared_ptr<ProblemInputs> inputs) override
        {
            checkedInputs.push_back(inputs);
            return false;
        }
    };
}

TEST(ReferenceValidatorLifecycle, EnqueuesAreUncheckedByDefault)
{
    RecordingValidator validator(validationArgs());
    TimingEvents events(0, 0);
    auto inputs = std::make_shared<ContractionInputs>();
    validator.preSolution(nullptr);
    validator.validateWarmups(inputs, events, events);
    validator.validateEnqueues(inputs, events, events);
    EXPECT_EQ(validator.checkedInputs.size(), 1u);
}

TEST(ReferenceValidatorLifecycle, OptInChecksEveryGroupAfterWarmup)
{
    auto args = validationArgs();
    args["validate-after-enqueues"].value() = true;
    RecordingValidator validator(args);
    TimingEvents events(0, 0);
    auto first = std::make_shared<ContractionInputs>();
    auto last = std::make_shared<ContractionInputs>();
    validator.preSolution(nullptr);
    validator.validateWarmups(first, events, events);
    EXPECT_FALSE(validator.needMoreRunsInSolution());
    validator.validateEnqueues(last, events, events);
    validator.validateEnqueues(first, events, events);
    ASSERT_EQ(validator.checkedInputs.size(), 3u);
    EXPECT_EQ(validator.checkedInputs[1], last);
    EXPECT_EQ(validator.checkedInputs[2], first);

    validator.preSolution(nullptr);
    EXPECT_TRUE(validator.needMoreRunsInSolution());
    validator.validateWarmups(last, events, events);
    EXPECT_EQ(validator.checkedInputs.size(), 4u);
}

TEST(ReferenceValidatorLifecycle, OptInRespectsDisabledReferenceValidation)
{
    auto args = validationArgs();
    args["validate-after-enqueues"].value() = true;
    args["num-elements-to-validate"].value() = 0;
    RecordingValidator validator(args);
    TimingEvents events(0, 0);
    validator.validateEnqueues(nullptr, events, events);
    EXPECT_TRUE(validator.checkedInputs.empty());
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
