// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/DataTypes.hpp>

#include <type_traits>

#include "DataInitializationTestUtils.hpp"

namespace
{
    static_assert(std::is_same_v<TensileLite::testing::DataInitConfig,
                                 TensileLite::testing::BaseDataInitArgsOptions>);

    template <typename T>
    void expectArgEq(TensileLite::Client::po::variables_map const& args,
                     char const*                                   key,
                     T const&                                      expected)
    {
        ASSERT_EQ(args.count(key), 1u) << "missing option: " << key;
        auto const& actual = args.at(key).as<T>();
        if constexpr(std::is_enum_v<T>)
        {
            using Underlying = std::underlying_type_t<T>;
            EXPECT_EQ(static_cast<Underlying>(actual), static_cast<Underlying>(expected))
                << "option: " << key;
        }
        else
        {
            EXPECT_EQ(actual, expected) << "option: " << key;
        }
    }

    void expectArgEq(TensileLite::Client::po::variables_map const& args,
                     char const*                                   key,
                     std::vector<rocisa::DataType> const&          expected)
    {
        ASSERT_EQ(args.count(key), 1u) << "missing option: " << key;
        auto const& actual = args.at(key).as<std::vector<rocisa::DataType>>();
        ASSERT_EQ(actual.size(), expected.size()) << "option: " << key;
        for(size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_EQ(static_cast<int>(actual[i]), static_cast<int>(expected[i]))
                << "option: " << key << " index: " << i;
        }
    }

    void expectRingArgs(TensileLite::Client::po::variables_map const& args,
                        int                                           elementsToValidate)
    {
        expectArgEq(args, "num-benchmarks", 0);
        expectArgEq(args, "num-enqueues-per-sync", 0);
        expectArgEq(args, "max-enqueues-per-sync", -1);
        expectArgEq(args, "min-flops-per-sync", size_t(0));
        expectArgEq(args, "num-syncs-per-benchmark", 0);
        expectArgEq(args, "num-warmups", 0);

        expectArgEq(args, "num-elements-to-validate", elementsToValidate);
        expectArgEq(args, "print-valids", false);
        expectArgEq(args, "print-max", -1);
        expectArgEq(args, "print-tensor-a", false);
        expectArgEq(args, "print-tensor-b", false);
        expectArgEq(args, "print-tensor-c", false);
        expectArgEq(args, "print-tensor-d", false);
        expectArgEq(args, "print-tensor-ref", false);
        expectArgEq(args, "print-tensor-bias", false);
        expectArgEq(args, "print-tensor-amaxd", false);

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

TEST(BuildBaseDataInitArgs, PopulatesDataInitConfigDefaultsAndOverrides)
{
    TensileLite::testing::DataInitConfig config;
    config.problemSizes       = {{64, 64, 64}};
    config.numWarmups         = 6;
    config.maxEnqueuesPerSync = 12;
    config.minFlopsPerSync    = size_t(34);
    config.printValids        = true;
    config.printMax           = 11;
    config.printTensorD       = true;
    config.printTensorAmaxD   = true;

    auto args = TensileLite::testing::buildBaseDataInitArgs(config);

    expectArgEq(args, "problem-identifier", std::string("Contraction_l_Alik_Bjlk_Cijk_Dijk"));
    ASSERT_EQ(args.count("problem-size"), 1u);
    EXPECT_EQ(args.at("problem-size").as<std::vector<std::vector<size_t>>>(),
              (std::vector<std::vector<size_t>>{{64, 64, 64}}));

    expectArgEq(args, "type", rocisa::DataType::Float);
    expectArgEq(args, "a-type", rocisa::DataType::Float);
    expectArgEq(args, "b-type", rocisa::DataType::Float);
    expectArgEq(args, "c-type", rocisa::DataType::Float);
    expectArgEq(args, "d-type", rocisa::DataType::Float);
    expectArgEq(args, "e-type", rocisa::DataType::None);
    expectArgEq(args, "amaxD-type", rocisa::DataType::None);
    expectArgEq(args, "alpha-type", rocisa::DataType::Float);
    expectArgEq(args, "beta-type", rocisa::DataType::Float);

    expectArgEq(args, "num-benchmarks", 0);
    expectArgEq(args, "num-enqueues-per-sync", 0);
    expectArgEq(args, "max-enqueues-per-sync", 12);
    expectArgEq(args, "min-flops-per-sync", size_t(34));
    expectArgEq(args, "num-syncs-per-benchmark", 0);
    expectArgEq(args, "num-warmups", 6);

    expectArgEq(args, "num-elements-to-validate", 0);
    expectArgEq(args, "print-valids", true);
    expectArgEq(args, "print-max", 11);
    expectArgEq(args, "print-tensor-a", false);
    expectArgEq(args, "print-tensor-b", false);
    expectArgEq(args, "print-tensor-c", false);
    expectArgEq(args, "print-tensor-d", true);
    expectArgEq(args, "print-tensor-ref", false);
    expectArgEq(args, "print-tensor-bias", false);
    expectArgEq(args, "print-tensor-amaxd", true);

    expectArgEq(args, "pristine-on-gpu", true);
    expectArgEq(args, "bounds-check", TensileLite::Client::BoundsCheckMode::Disable);
    expectArgEq(args, "rotating-buffer-size", int32_t(0));
    expectArgEq(args, "rotating-buffer-mode", int32_t(0));
}

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
                                                 "max-enqueues-per-sync",
                                                 std::any(int(9)));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "min-flops-per-sync",
                                                 std::any(size_t(13)));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "num-syncs-per-benchmark",
                                                 std::any(int(5)));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "num-warmups", std::any(int(6)));

    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "num-elements-to-validate",
                                                 std::any(int(7)));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "print-valids", std::any(true));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "print-max", std::any(int(11)));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "print-tensor-a", std::any(true));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "print-tensor-b", std::any(true));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "print-tensor-c", std::any(true));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "print-tensor-d", std::any(true));
    TensileLite::testing::detail::setDataInitArg(baseArgs, "print-tensor-ref", std::any(true));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "print-tensor-bias",
                                                 std::any(true));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "print-tensor-amaxd",
                                                 std::any(true));

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
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "a-type",
                                                 std::any(rocisa::DataType::BFloat16));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "activation-type",
                                                 std::any(TensileLite::ActivationType::Clippedrelu));
    TensileLite::testing::detail::setDataInitArg(baseArgs,
                                                 "compute-input-type-A",
                                                 std::any(rocisa::DataType::Double));

    auto overrideArgs = TensileLite::testing::buildRingArgs(baseArgs, 0);
    expectRingArgs(overrideArgs, 0);
    EXPECT_EQ(overrideArgs.at("problem-size").as<std::vector<std::vector<size_t>>>(),
              baseArgs.at("problem-size").as<std::vector<std::vector<size_t>>>());
    EXPECT_EQ(overrideArgs.at("problem-identifier").as<std::string>(),
              baseArgs.at("problem-identifier").as<std::string>());
    expectArgEq(overrideArgs, "a-type", rocisa::DataType::BFloat16);
    expectArgEq(overrideArgs, "activation-type", TensileLite::ActivationType::Clippedrelu);
    expectArgEq(overrideArgs, "compute-input-type-A", rocisa::DataType::Double);
}

TEST(MakePlainProblem, ContractPreservesTransposeAndDescriptorGeometry)
{
    TensileLite::testing::PlainProblemSpec spec;
    spec.m                 = 17;
    spec.n                 = 23;
    spec.k                 = 31;
    spec.batch             = 5;
    spec.transA            = false;
    spec.transB            = true;
    spec.aType             = rocisa::DataType::Half;
    spec.bType             = rocisa::DataType::BFloat16;
    spec.cType             = rocisa::DataType::Double;
    spec.dType             = rocisa::DataType::Float;
    spec.computeInputTypeA = rocisa::DataType::Double;
    spec.computeInputTypeB = rocisa::DataType::Half;
    spec.alphaType         = rocisa::DataType::BFloat16;
    spec.betaType          = rocisa::DataType::Double;
    spec.beta              = 1.5;

    auto problem = TensileLite::testing::makePlainProblem(spec);

    EXPECT_EQ(problem.freeSizeA(0), spec.m);
    EXPECT_EQ(problem.freeSizeB(0), spec.n);
    EXPECT_EQ(problem.boundSize(0), spec.k);
    EXPECT_EQ(problem.batchSize(0), spec.batch);
    EXPECT_DOUBLE_EQ(problem.beta(), spec.beta);
    EXPECT_TRUE(problem.stridedBatched());

    EXPECT_EQ(problem.a().dataType(), spec.aType);
    EXPECT_EQ(problem.b().dataType(), spec.bType);
    EXPECT_EQ(problem.c().dataType(), spec.cType);
    EXPECT_EQ(problem.d().dataType(), spec.dType);
    EXPECT_EQ(problem.computeInputTypeA(), spec.computeInputTypeA);
    EXPECT_EQ(problem.computeInputTypeB(), spec.computeInputTypeB);
    EXPECT_EQ(problem.alphaType(), spec.alphaType);
    EXPECT_EQ(problem.betaType(), spec.betaType);

    EXPECT_EQ(problem.a().sizes(), (std::vector<size_t>{spec.m, spec.k, spec.batch}));
    EXPECT_EQ(problem.a().strides(), (std::vector<size_t>{1, spec.m, spec.m * spec.k}));
    EXPECT_EQ(problem.b().sizes(), (std::vector<size_t>{spec.n, spec.k, spec.batch}));
    EXPECT_EQ(problem.b().strides(), (std::vector<size_t>{1, spec.n, spec.n * spec.k}));
    EXPECT_EQ(problem.c().sizes(), (std::vector<size_t>{spec.m, spec.n, spec.batch}));
    EXPECT_EQ(problem.c().strides(), (std::vector<size_t>{1, spec.m, spec.m * spec.n}));
    EXPECT_EQ(problem.d().sizes(), (std::vector<size_t>{spec.m, spec.n, spec.batch}));
    EXPECT_EQ(problem.d().strides(), (std::vector<size_t>{1, spec.m, spec.m * spec.n}));
}

TEST(MakeBatchedProblem, ContractPinsBatchStridesAndFloatTypes)
{
    auto problem = TensileLite::testing::makeBatchedProblem(32, 48, 16, 4);

    EXPECT_EQ(problem.freeSizeA(0), size_t(32));
    EXPECT_EQ(problem.freeSizeB(0), size_t(48));
    EXPECT_EQ(problem.boundSize(0), size_t(16));
    EXPECT_EQ(problem.batchSize(0), size_t(4));
    EXPECT_DOUBLE_EQ(problem.beta(), 0.0);
    EXPECT_FALSE(problem.stridedBatched());

    EXPECT_EQ(problem.a().dataType(), rocisa::DataType::Float);
    EXPECT_EQ(problem.b().dataType(), rocisa::DataType::Float);
    EXPECT_EQ(problem.c().dataType(), rocisa::DataType::Float);
    EXPECT_EQ(problem.d().dataType(), rocisa::DataType::Float);

    EXPECT_EQ(problem.a().sizes(), (std::vector<size_t>{32, 16, 4}));
    EXPECT_EQ(problem.a().strides(), (std::vector<size_t>{1, 32, 32 * 16}));
    EXPECT_EQ(problem.b().sizes(), (std::vector<size_t>{16, 48, 4}));
    EXPECT_EQ(problem.b().strides(), (std::vector<size_t>{1, 16, 16 * 48}));
    EXPECT_EQ(problem.c().sizes(), (std::vector<size_t>{32, 48, 4}));
    EXPECT_EQ(problem.c().strides(), (std::vector<size_t>{1, 32, 32 * 48}));
    EXPECT_EQ(problem.d().sizes(), (std::vector<size_t>{32, 48, 4}));
    EXPECT_EQ(problem.d().strides(), (std::vector<size_t>{1, 32, 32 * 48}));

    auto largerProblem = TensileLite::testing::makeBatchedProblem(64, 48, 16, 4);
    EXPECT_EQ(largerProblem.a().strides()[2], size_t(64 * 16));
    EXPECT_NE(largerProblem.a().strides()[2], problem.a().strides()[2]);
}
