// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/DataTypes.hpp>

#include "DataInitializationTestUtils.hpp"

namespace
{
    template <typename T>
    void expectArgEq(TensileLite::Client::po::variables_map const& args,
                     char const*                                   key,
                     T const&                                      expected)
    {
        ASSERT_EQ(args.count(key), 1u) << "missing option: " << key;
        EXPECT_EQ(args.at(key).as<T>(), expected) << "option: " << key;
    }

    void expectRingArgs(TensileLite::Client::po::variables_map const& args,
                        int                                           elementsToValidate)
    {
        expectArgEq(args, "num-benchmarks", 0);
        expectArgEq(args, "num-enqueues-per-sync", 0);
        expectArgEq(args, "num-syncs-per-benchmark", 0);
        expectArgEq(args, "num-warmups", 0);

        expectArgEq(args, "num-elements-to-validate", elementsToValidate);
        expectArgEq(args, "print-valids", false);
        expectArgEq(args, "print-max", -1);

        expectArgEq(args, "pristine-on-gpu", true);
        expectArgEq(args, "bounds-check", TensileLite::Client::BoundsCheckMode::Disable);
        expectArgEq(args, "rotating-buffer-size", int32_t(0));
        expectArgEq(args, "rotating-buffer-mode", int32_t(0));

        expectArgEq(args, "sparse", 0);
        expectArgEq(args,
                    "bias-type-args",
                    std::vector<rocisa::DataType>{rocisa::DataType::None});

        expectArgEq(args, "mx-a-block", 0);
        expectArgEq(args, "mx-b-block", 0);
        expectArgEq(args, "mx-a-type", rocisa::DataType::E8);
        expectArgEq(args, "mx-b-type", rocisa::DataType::E8);
        expectArgEq(args, "mx-scale-format", 0);

        expectArgEq(args, "init-a", TensileLite::Client::InitMode::Random);
        expectArgEq(args, "init-b", TensileLite::Client::InitMode::Random);
        expectArgEq(args, "init-c", TensileLite::Client::InitMode::Random);
        expectArgEq(args, "init-d", TensileLite::Client::InitMode::Zero);
        expectArgEq(args, "init-e", TensileLite::Client::InitMode::Zero);
        expectArgEq(args, "init-alpha", TensileLite::Client::InitMode::Two);
        expectArgEq(args, "init-beta", TensileLite::Client::InitMode::Two);
        expectArgEq(args, "init-bias", TensileLite::Client::InitMode::One);
        expectArgEq(args, "init-scaleA", TensileLite::Client::InitMode::Two);
        expectArgEq(args, "init-scaleB", TensileLite::Client::InitMode::Two);
        expectArgEq(args, "init-scaleC", TensileLite::Client::InitMode::Two);
        expectArgEq(args, "init-scaleD", TensileLite::Client::InitMode::Two);
        expectArgEq(args, "init-scaleAlphaVec", TensileLite::Client::InitMode::One);
        expectArgEq(args, "init-mx-a", TensileLite::Client::InitMode::One);
        expectArgEq(args, "init-mx-b", TensileLite::Client::InitMode::One);
    }
} // namespace

TEST(BuildRingArgs, ProblemSizesAndCustomBase)
{
    auto defaultArgs = TensileLite::testing::buildRingArgs({{64, 64, 64}});
    expectRingArgs(defaultArgs, 1);

    auto baseArgs = TensileLite::testing::buildBaseDataInitArgs({{64, 64, 64}});
    TensileLite::testing::detail::setDataInitArg(baseArgs, "num-benchmarks", std::any(int(3)));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "num-enqueues-per-sync",
                                                 std::any(int(4)));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "num-syncs-per-benchmark",
                                                 std::any(int(5)));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "num-warmups", std::any(int(6)));

    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "num-elements-to-validate",
                                                 std::any(int(7)));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "print-valids", std::any(true));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "print-max", std::any(int(11)));

    TensileLite::testing::detail::setDataInitArg(baseArgs, "pristine-on-gpu", std::any(false));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "bounds-check",
                                                 std::any(TensileLite::Client::BoundsCheckMode::NaN));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "rotating-buffer-size",
                                                 std::any(int32_t(64)));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "rotating-buffer-mode",
                                                 std::any(int32_t(2)));

    TensileLite::testing::detail::setDataInitArg(baseArgs, "sparse", std::any(int(9)));
    TensileLite::testing::detail::setDataInitArg(
        baseArgs,
        "bias-type-args",
        std::any(std::vector<rocisa::DataType>{rocisa::DataType::Float}));

    TensileLite::testing::detail::setDataInitArg(baseArgs, "mx-a-block", std::any(int(4)));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "mx-b-block", std::any(int(5)));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "mx-a-type",
                                                 std::any(rocisa::DataType::Float));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "mx-b-type",
                                                 std::any(rocisa::DataType::Float));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "mx-scale-format", std::any(int(2)));

    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-a",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-b",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-c",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-d",
                                                 std::any(TensileLite::Client::InitMode::Random));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-e",
                                                 std::any(TensileLite::Client::InitMode::Random));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-alpha",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-beta",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-bias",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-scaleA",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-scaleB",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-scaleC",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-scaleD",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-scaleAlphaVec",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-mx-a",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "init-mx-b",
                                                 std::any(TensileLite::Client::InitMode::Zero));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "problem-identifier",
                                                 std::any(std::string("custom-base")));

    auto overrideArgs = TensileLite::testing::buildRingArgs(baseArgs, 0);
    expectRingArgs(overrideArgs, 0);
    EXPECT_EQ(overrideArgs.at("problem-identifier").as<std::string>(),
              baseArgs.at("problem-identifier").as<std::string>());
}
