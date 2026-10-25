// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <roc/hostnumerics/freivalds.hpp>
#include <roc/hostnumerics/generation.hpp>

namespace {
using namespace roc::hostnumerics;

FreivaldsPlan requirePlan(FreivaldsPreparation preparation) {
    if (auto* failure = std::get_if<FreivaldsPreparationFailure>(&preparation))
        throw std::runtime_error(failure->reason);
    return std::get<FreivaldsPlan>(std::move(preparation));
}

TEST(Freivalds, PortableArithmeticAgainstPythonIntegers) {
    // Python arbitrary-precision (a*b) % (2**61-1), Random(10553).
    constexpr std::array<std::array<uint64_t, 3>, 13> cases{{
        {0ULL, 2305843009213693950ULL, 0ULL},
        {1ULL, 2305843009213693950ULL, 2305843009213693950ULL},
        {2305843009213693950ULL, 2305843009213693950ULL, 1ULL},
        {1073741823ULL, 2147483647ULL, 2305843005992468481ULL},
        {2305843009213693944ULL, 1152921504606846977ULL, 1152921504606846965ULL},
        {104216524951811245ULL, 298223097308506144ULL, 1086628811878190498ULL},
        {592387488533065461ULL, 1439522938617976405ULL, 488141800147647174ULL},
        {2267224452869148450ULL, 1746580757113766441ULL, 266036010684942379ULL},
        {22643363649869263ULL, 2238659301217645286ULL, 1394380869765904007ULL},
        {2302603675674544793ULL, 2007655769066124803ULL, 1042870232776945308ULL},
        {757340777883838368ULL, 705280989041101058ULL, 731033953114695433ULL},
        {1572943161541149861ULL, 829758084459182445ULL, 474383800946562170ULL},
        {767893914326189169ULL, 901514372969724752ULL, 678976497543470989ULL},
    }};
    for (const auto& c : cases) EXPECT_EQ(freivaldsMultiply(c[0], c[1]), c[2]);
    EXPECT_EQ(freivaldsAdd(freivaldsModulus - 1, 1), 0);
    EXPECT_EQ(freivaldsResidue(std::numeric_limits<int64_t>::min()), freivaldsModulus - 4);
    EXPECT_THROW(freivaldsMultiply(freivaldsModulus, 1), std::runtime_error);
    for (auto algorithm :
         {FreivaldsProbeAlgorithm::SplitMix64Legacy, FreivaldsProbeAlgorithm::SplitMix64Rejection})
        for (size_t i = 0; i < 100; ++i) {
            const auto value = freivaldsProbe(73, i, algorithm);
            EXPECT_GT(value, 0);
            EXPECT_LT(value, freivaldsModulus);
            EXPECT_EQ(value, freivaldsProbe(73, i, algorithm));
        }
}

TEST(Freivalds, DenseIntegerOracleWithScalesBiasRoundingAndSaturation) {
    constexpr size_t m = 13, n = 11, k = 129;
    const auto recipe = GenerationRecipe::realOnlyLastDimensionFastest(
        GenerationRecipe::uniformInteger({.lower = -2, .upper = 2}), 10553);
    auto a = generate(ScalarType::BFloat16, Shape{m, k}, recipe);
    auto b = generate(ScalarType::BFloat16, Shape{k, n}, recipe);
    auto c = generate(ScalarType::Float32, Shape{m, n}, recipe);
    Tensor row(ScalarType::Float32, Shape{m, 1}), col(ScalarType::Float32, Shape{n});
    Tensor bias(ScalarType::Float32, Shape{m, 1});
    for (size_t i = 0; i < m; ++i) {
        row.storeFrom({i, 0}, int(1 + i % 3));
        bias.storeFrom({i, 0}, int(i % 5) - 2);
    }
    for (size_t j = 0; j < n; ++j) col.storeFrom({j}, int(1 + j % 2));
    EpilogueOptions e(ScalarType::Float32);
    e.inputScale = Tensor(3.0f);
    e.inputScaleFactors = {row, col};
    e.addendScale = Tensor(-2.0f);
    e.addend = c;
    e.bias = bias;
    e.outputScale = Tensor(2.0f);
    for (const auto type : {ScalarType::Float32, ScalarType::Float16, ScalarType::BFloat16,
                            ScalarType::Int8, ScalarType::Float8E4M3Fnuz}) {
        SCOPED_TRACE(static_cast<int>(type));
        auto plan = requirePlan(
            prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32}, e, type, {}, {}));
        Tensor d(type, Layout(Shape{m, n}, {1, ptrdiff_t(m + 7)}, 3));
        const ScalarConversionOptions conversion{IntegerRounding::TowardZero,
                                                 IntegerOverflow::Saturate};
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) {
                int64_t sum = 0;
                for (size_t z = 0; z < k; ++z)
                    sum += a.loadAs<int64_t>({i, z}) * b.loadAs<int64_t>({z, j});
                const int64_t expected = 2 * (3 * int64_t(1 + i % 3) * int64_t(1 + j % 2) * sum -
                                              2 * c.loadAs<int64_t>({i, j}) + int64_t(i % 5) - 2);
                EXPECT_EQ(plan.exactValue(i, j), expected);
                d.storeFrom({i, j}, expected, conversion);
            }
        auto report = plan.check(d, std::numeric_limits<size_t>::max(), {});
        EXPECT_EQ(report.status, VerificationStatus::Passed) << report.reason;
        EXPECT_EQ(report.checkedElements, m * n);
        if (type == ScalarType::BFloat16 || type == ScalarType::Int8)
            EXPECT_GT(report.recomputedElements, 0);
        // Store a representable finite wrong value for every output format.
        const double correct = d.loadAs<double>({5, 7});
        d.storeFrom({5, 7}, correct == 1 ? 2 : 1);
        report = plan.check(d, std::numeric_limits<size_t>::max(), {});
        EXPECT_EQ(report.status, VerificationStatus::Mismatch);
    }
}

TEST(Freivalds, ImmutableInputsAndSingleAndCancellingErrors) {
    Tensor a(ScalarType::Float32, Shape{2, 3}), b(ScalarType::Float32, Shape{3, 2});
    for (size_t i = 0; i < 2; ++i)
        for (size_t k = 0; k < 3; ++k) {
            a.storeFrom({i, k}, int(i + k + 1));
            b.storeFrom({k, i}, int(k - i + 1));
        }
    auto plan = requirePlan(prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32},
                                             EpilogueOptions{ScalarType::Float32},
                                             ScalarType::Float32, {}, {}));
    Tensor d(ScalarType::Float32, Shape{2, 2});
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j) {
            int64_t sum = 0;
            for (size_t k = 0; k < 3; ++k)
                sum += a.loadAs<int64_t>({i, k}) * b.loadAs<int64_t>({k, j});
            d.storeFrom({i, j}, sum);
        }
    a.storeFrom({0, 0}, 99);
    b.storeFrom({0, 0}, -7);
    EXPECT_EQ(plan.check(d, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Passed);
    d.storeFrom({1, 0}, d.loadAs<int>({1, 0}) + 1);
    auto result = plan.check(d, std::numeric_limits<size_t>::max(), {});
    EXPECT_EQ(result.status, VerificationStatus::Mismatch);
    EXPECT_EQ(result.badRows, (std::vector<size_t>{1}));
    EXPECT_EQ(result.badColumns, (std::vector<size_t>{0}));
    d.storeFrom({1, 1}, d.loadAs<int>({1, 1}) - 1);
    EXPECT_EQ(plan.check(d, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Mismatch);
}

TEST(Freivalds, BoundedAccumulationRetainsExactDefaultAndDetectsCorruption) {
    // GFX11 DOT2 can return 2^-23 for -2 + 1*1 + 1*1. See rocm-systems#12056.
    Tensor a(ScalarType::Float32, Shape{1, 4}), b(ScalarType::Float32, Shape{4, 1});
    a.storeFrom({0, 0}, -2);
    a.storeFrom({0, 2}, 1);
    a.storeFrom({0, 3}, 1);
    for (size_t k = 0; k < 4; ++k) b.storeFrom({k, 0}, 1);
    constexpr size_t budget = 4;
    const FreivaldsCheckOptions options{.maximumAbsoluteComputationError = 0x1p-20};
    for (const auto type : {ScalarType::Float32, ScalarType::Float16, ScalarType::BFloat16}) {
        const auto plan =
            requirePlan(prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32},
                                         EpilogueOptions{ScalarType::Float32}, type, {}, {}));
        Tensor d(type, Shape{1, 1});
        d.storeFrom({0, 0}, 0x1p-23);
        EXPECT_EQ(plan.check(d, std::numeric_limits<size_t>::max(), {}).status,
                  VerificationStatus::Mismatch);
        const auto check = plan.check(d, budget, options);
        EXPECT_EQ(check.status, VerificationStatus::Passed);
        EXPECT_EQ(check.recomputedElements, 1);
        EXPECT_EQ(plan.check(d, 0, options).status, VerificationStatus::Incomplete);
        d.storeFrom({0, 0}, 1);
        EXPECT_EQ(plan.check(d, budget, options).status, VerificationStatus::Mismatch);
        d.storeFrom({0, 0}, 0x1p-8);
        EXPECT_EQ(plan.check(d, budget, options).status, VerificationStatus::Mismatch);
        d.storeFrom({0, 0}, std::numeric_limits<double>::quiet_NaN());
        EXPECT_EQ(plan.check(d, budget, options).status, VerificationStatus::Mismatch);
        d.storeFrom({0, 0}, std::numeric_limits<double>::infinity());
        EXPECT_EQ(plan.check(d, budget, options).status, VerificationStatus::Mismatch);
        for (const double invalid : {-1.0, 0.5, std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN()})
            EXPECT_EQ(plan.check(d, budget, {invalid}).status, VerificationStatus::Unsupported);
    }
}

TEST(Freivalds, ComputationBoundIncludesOutputRoundingTies) {
    Tensor a(ScalarType::Float32, Shape{2, 1}), b(ScalarType::Float32, Shape{1, 1});
    b.storeFrom({0, 0}, 1);
    // The perturbation survives the F32 conversion preceding F16/BF16 encoding.
    const FreivaldsCheckOptions options{.maximumAbsoluteComputationError = 0x1p-8};
    for (const auto type : {ScalarType::Float16, ScalarType::BFloat16}) {
        const int midpoint = type == ScalarType::Float16 ? 2049 : 257;
        a.storeFrom({0, 0}, midpoint);
        a.storeFrom({1, 0}, -midpoint);
        const auto plan =
            requirePlan(prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32},
                                         EpilogueOptions{ScalarType::Float32}, type, {}, {}));
        Tensor d(type, Shape{2, 1});
        d.storeFrom({0, 0}, midpoint + 1);
        d.storeFrom({1, 0}, -midpoint - 1);
        EXPECT_EQ(plan.check(d, std::numeric_limits<size_t>::max(), {}).status,
                  VerificationStatus::Mismatch);
        EXPECT_EQ(plan.check(d, 2, options).status, VerificationStatus::Passed);
        d.storeFrom({0, 0}, midpoint + 3);
        EXPECT_EQ(plan.check(d, 2, options).status, VerificationStatus::Mismatch);
    }
}

TEST(Freivalds, ComputationBoundCannotCrossConsecutiveIntegerBoundary) {
    const FreivaldsCheckOptions options{.maximumAbsoluteComputationError =
                                            std::nextafter(0.5, 0.0)};
    for (const auto type : {ScalarType::Float64, ScalarType::Float32, ScalarType::Float16,
                            ScalarType::BFloat16, ScalarType::Int32, ScalarType::Int8}) {
        SCOPED_TRACE(scalarTypeInfo(type).name);
        const double limit = type == ScalarType::Float16    ? 0x1p11
                             : type == ScalarType::BFloat16 ? 0x1p8
                             : type == ScalarType::Int8     ? 128
                                                            : 0x1p24;
        Tensor a(ScalarType::Float32, Shape{2, 1}), b(ScalarType::Float32, Shape{1, 1});
        a.storeFrom({0, 0}, limit - 1);
        a.storeFrom({1, 0}, -limit + 1);
        b.storeFrom({0, 0}, 1);
        const auto plan =
            requirePlan(prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32},
                                         EpilogueOptions{ScalarType::Float32}, type, {}, {}));
        Tensor d(type, Shape{2, 1});
        d.storeFrom({0, 0}, limit - 1);
        d.storeFrom({1, 0}, -limit + 1);
        EXPECT_EQ(plan.check(d, 2, options).status, VerificationStatus::Passed);
        d.storeFrom({1, 0}, -limit);
        EXPECT_EQ(plan.check(d, 2, options).status, VerificationStatus::Mismatch);
        if (type != ScalarType::Int8) {
            d.storeFrom({0, 0}, limit);
            EXPECT_EQ(plan.check(d, 2, options).status, VerificationStatus::Mismatch);
        }
    }
    Tensor a(ScalarType::Float64, Shape{1, 1}), b(ScalarType::Float64, Shape{1, 1});
    a.storeFrom({0, 0}, 0x1p50);
    b.storeFrom({0, 0}, 1);
    const auto plan = requirePlan(prepareFreivalds(a, b, MatmulOptions(ScalarType::Float64),
                                                   EpilogueOptions(ScalarType::Float64),
                                                   ScalarType::Float64, {}, {}));
    Tensor d(ScalarType::Float64, Shape{1, 1});
    d.storeFrom({0, 0}, 0x1p50);
    EXPECT_EQ(plan.check(d, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Passed);
    EXPECT_EQ(plan.check(d, 1, {0x1p-20}).status, VerificationStatus::Unsupported);
}

TEST(Freivalds, ExportedTensorsCannotMutatePlanState) {
    Tensor a(ScalarType::Float32, Shape{2, 1}), b(ScalarType::Float32, Shape{1, 2});
    a.storeFrom({0, 0}, 1);
    a.storeFrom({1, 0}, 2);
    b.storeFrom({0, 0}, 3);
    b.storeFrom({0, 1}, 4);
    const auto plan = requirePlan(prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32},
                                                   EpilogueOptions{ScalarType::Float32},
                                                   ScalarType::Float32, {}, {}));
    const auto request = plan.copyRequest();
    ASSERT_EQ(request.rightProbe.type(), ScalarType::UInt64);
    ASSERT_EQ(request.rightProbe.shape(), (Shape{2}));
    const auto right = request.rightProbe.loadAs<uint64_t>({0});
    const auto left = request.leftProbe.loadAs<uint64_t>({0});
    request.rightProbe.storeFrom({0}, 0);
    request.leftProbe.storeFrom({0}, 0);
    plan.copyExpectedRowSums().storeFrom({0}, 0);
    plan.copyExpectedColumnSums().storeFrom({0}, 0);
    EXPECT_EQ(plan.copyRequest().rightProbe.loadAs<uint64_t>({0}), right);
    EXPECT_EQ(plan.copyRequest().leftProbe.loadAs<uint64_t>({0}), left);
    auto d = matmul(a, b, ScalarType::Float32, MatmulOptions{ScalarType::Float32}, std::nullopt,
                    GemmBackend::Automatic);
    EXPECT_EQ(plan.check(d, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Passed);
    d.storeFrom({0, 0}, 4);
    EXPECT_EQ(plan.check(d, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Mismatch);
}

TEST(Freivalds, StridedObservationsValidateTypeResiduesAndCompleteBackingStorage) {
    Tensor a(ScalarType::Float32, Shape{2, 0}), b(ScalarType::Float32, Shape{0, 3});
    FreivaldsLimits limits;
    limits.maxObservationBytes = 256;
    const auto plan = requirePlan(prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32},
                                                   EpilogueOptions{ScalarType::Float32},
                                                   ScalarType::Float32, {}, limits));
    FreivaldsObservation observation;
    observation.requestIdentity = plan.copyRequest().identity;
    observation.checkedElements = 6;
    observation.rowSums = Tensor(ScalarType::UInt64, Layout(Shape{2}, {2}, 1));
    observation.columnSums = Tensor(ScalarType::UInt64, Layout(Shape{3}, {-1}, 2));
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Passed);
    observation.columnSums.storeFrom({2}, freivaldsModulus - 1);
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Mismatch);
    EXPECT_EQ(observation.columnSums.loadAs<uint64_t>({2}), freivaldsModulus - 1);
    observation.columnSums.storeFrom({2}, freivaldsModulus);
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Incomplete);
    observation.columnSums = Tensor(ScalarType::Int64, Shape{3});
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Incomplete);
    observation.columnSums = Tensor(ScalarType::UInt64, Shape{1, 3});
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Incomplete);
    observation.columnSums = Tensor::takeOwnershipOfEncodedBackingStorage(
        ScalarType::UInt64, Layout(Shape{3}, {1}, 0), std::vector<std::byte>(1024));
    const auto result = plan.check(observation, std::numeric_limits<size_t>::max(), {});
    EXPECT_EQ(result.status, VerificationStatus::Incomplete);
    EXPECT_NE(result.reason.find("storage budget"), std::string::npos);
}

TEST(Freivalds, ReducesModularPartialsWithArbitraryLayouts) {
    Tensor partials(ScalarType::UInt64, Layout(Shape{16, 2}, {1, 20}, 2));
    for (size_t chunk = 0; chunk < 16; ++chunk) {
        partials.storeFrom({chunk, 0}, freivaldsModulus - 1);
        partials.storeFrom({chunk, 1}, chunk % 2 ? freivaldsModulus - 7 : 7);
    }
    auto result = reduceFreivaldsProjections(partials);
    EXPECT_EQ(result.shape(), (Shape{2}));
    EXPECT_EQ(result.type(), ScalarType::UInt64);
    EXPECT_EQ(result.loadAs<uint64_t>({0}), freivaldsModulus - 16);
    EXPECT_EQ(result.loadAs<uint64_t>({1}), 0);
    result = reduceFreivaldsProjections(Tensor(ScalarType::UInt64, Shape{0, 2}));
    EXPECT_EQ(result.loadAs<uint64_t>({0}), 0);
    EXPECT_EQ(result.loadAs<uint64_t>({1}), 0);
    partials.storeFrom({13, 1}, freivaldsModulus);
    EXPECT_THROW(reduceFreivaldsProjections(partials), std::runtime_error);
    EXPECT_THROW(reduceFreivaldsProjections(Tensor(ScalarType::Float64, Shape{2, 2})),
                 std::runtime_error);
    EXPECT_THROW(reduceFreivaldsProjections(Tensor(ScalarType::UInt64, Shape{2})),
                 std::runtime_error);
}

TEST(Freivalds, ObservationCompletenessAndExceptionBudgets) {
    Tensor a(ScalarType::Float32, Shape{2, 0}), b(ScalarType::Float32, Shape{0, 3});
    FreivaldsLimits limits;
    limits.maxRecomputeMultiplyAdds = 1;
    auto plan = requirePlan(prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32},
                                             EpilogueOptions{ScalarType::Float32},
                                             ScalarType::Float32, {}, limits));
    FreivaldsObservation observation;
    observation.requestIdentity = plan.copyRequest().identity;
    observation.checkedElements = 6;
    observation.rowSums = Tensor(ScalarType::UInt64, Shape{2});
    observation.columnSums = Tensor(ScalarType::UInt64, Shape{3});
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Passed);
    observation.requestIdentity++;
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Incomplete);
    observation.requestIdentity--;
    observation.checkedElements--;
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Incomplete);
    observation.checkedElements++;
    observation.exceptions = {{1, 2, 0}};
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Passed);
    observation.exceptions.push_back({1, 2, 0});
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Incomplete);
    observation.exceptions = {{7, 2, 0}};
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Incomplete);
    observation.exceptions = {{1, 2, std::numeric_limits<double>::quiet_NaN()}};
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Mismatch);
    observation.exceptions.clear();
    observation.rowSums.storeFrom({0}, freivaldsModulus);
    EXPECT_EQ(plan.check(observation, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Incomplete);
}

TEST(Freivalds, RejectsInexactNonlinearAndOverBudgetProblems) {
    Tensor a(ScalarType::Float32, Shape{1, 2}), b(ScalarType::Float32, Shape{2, 1});
    a.storeFrom({0, 0}, 0x1p24f);
    a.storeFrom({0, 1}, -0x1p24f);
    b.storeFrom({0, 0}, 1.0f);
    b.storeFrom({1, 0}, 1.0f);
    auto rejected =
        prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32},
                         EpilogueOptions{ScalarType::Float32}, ScalarType::Float32, {}, {});
    EXPECT_EQ(std::get<FreivaldsPreparationFailure>(rejected).status,
              VerificationStatus::Unsupported);
    a.storeFrom({0, 0}, 0.5f);
    EXPECT_EQ(
        std::get<FreivaldsPreparationFailure>(
            prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32},
                             EpilogueOptions{ScalarType::Float32}, ScalarType::Float32, {}, {}))
            .status,
        VerificationStatus::Unsupported);
    EpilogueOptions e(ScalarType::Float32);
    e.activation = ReluActivation{};
    EXPECT_EQ(std::get<FreivaldsPreparationFailure>(
                  prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32}, e, ScalarType::Float32,
                                   {}, {}))
                  .status,
              VerificationStatus::Unsupported);
    // Zero-storage inputs can describe a huge output. Refuse its summaries
    // before attempting a shape-sized allocation.
    Tensor hugeA(ScalarType::Float32, Shape{size_t{1} << 32, 0});
    Tensor emptyB(ScalarType::Float32, Shape{0, 1});
    EXPECT_EQ(
        std::get<FreivaldsPreparationFailure>(
            prepareFreivalds(hugeA, emptyB, MatmulOptions{ScalarType::Float32},
                             EpilogueOptions{ScalarType::Float32}, ScalarType::Float32, {}, {}))
            .status,
        VerificationStatus::Incomplete);
    Tensor emptyA(ScalarType::Float32, Shape{0, 3}), normalB(ScalarType::Float32, Shape{3, 7});
    auto empty = requirePlan(prepareFreivalds(emptyA, normalB, MatmulOptions{ScalarType::Float32},
                                              EpilogueOptions{ScalarType::Float32},
                                              ScalarType::Float32, {}, {}));
    EXPECT_EQ(
        empty
            .check(Tensor(ScalarType::Float32, Shape{0, 7}), std::numeric_limits<size_t>::max(), {})
            .status,
        VerificationStatus::Passed);
}

TEST(Freivalds, UsesMatmulOperandScalesAndCertifiesComputeConversion) {
    Tensor a(ScalarType::Float32, Shape{2, 3}), b(ScalarType::Float32, Shape{3, 2});
    for (size_t i = 0; i < 2; ++i)
        for (size_t k = 0; k < 3; ++k) {
            a.storeFrom({i, k}, int(i + k) - 1);
            b.storeFrom({k, i}, int(k) - int(i));
        }
    MatmulOptions operation(ScalarType::Float32);
    operation.computeTypeA = ScalarType::Float16;
    Tensor alongK(ScalarType::Float32, Shape{3});
    for (size_t k = 0; k < 3; ++k) alongK.storeFrom({k}, int(k + 1));
    operation.preQuantizationScalesA = {Tensor(2.0f), alongK};
    operation.preQuantizationScalesB = {Tensor(-3.0f)};
    const auto plan = requirePlan(prepareFreivalds(
        a, b, operation, EpilogueOptions{ScalarType::Float32}, ScalarType::Float32, {}, {}));
    auto observed =
        matmul(a, b, ScalarType::Float32, operation, std::nullopt, GemmBackend::Automatic);
    EXPECT_EQ(plan.check(observed, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Passed);
    observed.storeFrom({1, 1}, observed.loadAs<float>({1, 1}) + 1);
    EXPECT_EQ(plan.check(observed, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Mismatch);

    // Integer input alone is insufficient: quantizing 2049 to F16 loses a unit.
    operation.preQuantizationScalesA.clear();
    a.storeFrom({0, 0}, 2049.0f);
    EXPECT_EQ(std::get<FreivaldsPreparationFailure>(
                  prepareFreivalds(a, b, operation, EpilogueOptions{ScalarType::Float32},
                                   ScalarType::Float32, {}, {}))
                  .status,
              VerificationStatus::Unsupported);
    // A subsequent zero must not hide an inexact scaling prefix.
    a.storeFrom({0, 0}, 1.0f);
    operation.preQuantizationScalesA = {Tensor(0x1p53), Tensor(0.0f)};
    EXPECT_EQ(std::get<FreivaldsPreparationFailure>(
                  prepareFreivalds(a, b, operation, EpilogueOptions{ScalarType::Float32},
                                   ScalarType::Float32, {}, {}))
                  .status,
              VerificationStatus::Unsupported);

    operation.preQuantizationScalesA = {Tensor(2.0f)};
    FreivaldsLimits limits;
    limits.maxPreparationMultiplyAdds = 1;
    EXPECT_EQ(std::get<FreivaldsPreparationFailure>(
                  prepareFreivalds(a, b, operation, EpilogueOptions{ScalarType::Float32},
                                   ScalarType::Float32, {}, limits))
                  .status,
              VerificationStatus::Incomplete);
}

TEST(Freivalds, ObservationStorageIsBoundedBeforeCopiesAndExceptionGrowth) {
    Tensor a(ScalarType::Float32, Shape{2, 0}), b(ScalarType::Float32, Shape{0, 3});
    FreivaldsLimits limits;
    limits.maxObservationBytes = 256;
    auto plan = requirePlan(prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32},
                                             EpilogueOptions{ScalarType::Float32},
                                             ScalarType::Float32, {}, limits));
    EXPECT_EQ(plan.copyRequest().maximumExceptions, 2);
    Tensor d(ScalarType::Float32, Shape{2, 3});
    EXPECT_EQ(plan.check(d, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Passed);
    for (size_t j = 0; j < 3; ++j) d.storeFrom({0, j}, 0x1p24f);
    EXPECT_EQ(plan.check(d, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Incomplete);

    FreivaldsObservation observation;
    observation.requestIdentity = plan.copyRequest().identity;
    observation.checkedElements = 6;
    observation.rowSums = Tensor(ScalarType::UInt64, Shape{2});
    observation.columnSums = Tensor(ScalarType::UInt64, Shape{3});
    observation.exceptions = {{1, 2, 0}, {1, 2, 0}};
    auto result = plan.check(observation, std::numeric_limits<size_t>::max(), {});
    EXPECT_EQ(result.status, VerificationStatus::Incomplete);
    EXPECT_NE(result.reason.find("duplicated"), std::string::npos);
    observation.exceptions.clear();
    observation.exceptions.reserve(3);
    result = plan.check(observation, std::numeric_limits<size_t>::max(), {});
    EXPECT_EQ(result.status, VerificationStatus::Incomplete);
    EXPECT_NE(result.reason.find("storage budget"), std::string::npos);
    limits.maxObservationBytes = 1;
    EXPECT_EQ(
        std::get<FreivaldsPreparationFailure>(
            prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32},
                             EpilogueOptions{ScalarType::Float32}, ScalarType::Float32, {}, limits))
            .status,
        VerificationStatus::Incomplete);
}

TEST(Freivalds, ZeroResultDoesNotHideInexactFactorPrefixes) {
    Tensor a(ScalarType::Float32, Shape{1, 1}), b(ScalarType::Float32, Shape{1, 1});
    EpilogueOptions epilogue(ScalarType::Float32);
    // (4097*4097) rounds in F32 before the later zero is encountered.
    epilogue.inputScaleFactors = {Tensor(4097.0f), Tensor(4097.0f), Tensor(0.0f)};
    auto result = prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32}, epilogue,
                                   ScalarType::Float32, {}, {});
    EXPECT_EQ(std::get<FreivaldsPreparationFailure>(result).status,
              VerificationStatus::Unsupported);
    // The same issue across a row and a column factor must be caught too.
    epilogue.inputScaleFactors[0] = Tensor(4097.0f).reshapeSharingStorage(Shape{1, 1});
    result = prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32}, epilogue,
                              ScalarType::Float32, {}, {});
    EXPECT_EQ(std::get<FreivaldsPreparationFailure>(result).status,
              VerificationStatus::Unsupported);
    epilogue.inputScaleFactors = {Tensor(0.0f), Tensor(4097.0f), Tensor(4097.0f)};
    auto plan = requirePlan(prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32}, epilogue,
                                             ScalarType::Float32, {}, {}));
    EXPECT_EQ(
        plan.check(Tensor(ScalarType::Float32, Shape{1, 1}), std::numeric_limits<size_t>::max(), {})
            .status,
        VerificationStatus::Passed);
}

TEST(Freivalds, WideIntegerComputationAndRoundedInfinity) {
    Tensor a(ScalarType::Float64, Shape{1, 3}), b(ScalarType::Float64, Shape{3, 1});
    a.storeFrom({0, 0}, 0x1p40);
    a.storeFrom({0, 1}, 3.0);
    a.storeFrom({0, 2}, -0x1p40);
    for (size_t k = 0; k < 3; ++k) b.storeFrom({k, 0}, 1.0);
    const MatmulOptions operation(ScalarType::Float64);
    const EpilogueOptions epilogue(ScalarType::Float64);
    auto plan =
        requirePlan(prepareFreivalds(a, b, operation, epilogue, ScalarType::Float64, {}, {}));
    Tensor d(ScalarType::Float64, Shape{1, 1});
    d.storeFrom({0, 0}, 3.0);
    EXPECT_EQ(plan.check(d, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Passed);
    d.storeFrom({0, 0}, 4.0);
    EXPECT_EQ(plan.check(d, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Mismatch);

    a.storeFrom({0, 2}, 0.0);
    auto narrow =
        requirePlan(prepareFreivalds(a, b, operation, epilogue, ScalarType::Float16, {}, {}));
    Tensor half(ScalarType::Float16, Shape{1, 1});
    half.storeFrom({0, 0}, 0x1p40 + 3);
    ASSERT_TRUE(std::isinf(half.loadAs<double>({0, 0})));
    EXPECT_EQ(narrow.check(half, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Passed);
    EXPECT_EQ(narrow.check(half, std::numeric_limits<size_t>::max(), {}).recomputedElements, 1U);
    EXPECT_EQ(narrow.check(half, 0, {}).status, VerificationStatus::Incomplete);
    half.storeFrom({0, 0}, -std::numeric_limits<double>::infinity());
    EXPECT_EQ(narrow.check(half, std::numeric_limits<size_t>::max(), {}).status,
              VerificationStatus::Mismatch);
}

TEST(Freivalds, RefusesModulusAliasesAndFractionalCoefficients) {
    Tensor a = Tensor(2.0f).reshapeSharingStorage(Shape{1, 1});
    Tensor b = Tensor(1.0f).reshapeSharingStorage(Shape{1, 1});
    EpilogueOptions epilogue(ScalarType::Float32);
    // An incorrect zero aliases +/- (2^61-1) under every modular probe.
    // Input certification must reject this before accepting any observation.
    for (int sign : {1, -1}) {
        epilogue.inputScale = Tensor(sign * 0x1p60);
        epilogue.bias = Tensor(float(-sign)).reshapeSharingStorage(Shape{1, 1});
        EXPECT_EQ(std::get<FreivaldsPreparationFailure>(
                      prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32}, epilogue,
                                       ScalarType::Float32, {}, {}))
                      .status,
                  VerificationStatus::Unsupported);
    }
    epilogue.inputScale = Tensor(1.0f);
    epilogue.bias = Tensor(-2.0f).reshapeSharingStorage(Shape{1, 1});
    EXPECT_EQ(
        requirePlan(prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32}, epilogue,
                                     ScalarType::Float32, {}, {}))
            .check(Tensor(ScalarType::Float32, Shape{1, 1}), std::numeric_limits<size_t>::max(), {})
            .status,
        VerificationStatus::Passed);
    for (int field = 0; field < 4; ++field) {
        EpilogueOptions fractional(ScalarType::Float32);
        switch (field) {
            case 0:
                fractional.inputScale = Tensor(0.5f);
                break;
            case 1:
                fractional.outputScale = Tensor(0.5f);
                break;
            case 2:
                fractional.inputScaleFactors = {Tensor(0.5f)};
                break;
            case 3:
                fractional.bias = Tensor(0.5f).reshapeSharingStorage(Shape{1, 1});
                break;
        }
        EXPECT_EQ(std::get<FreivaldsPreparationFailure>(
                      prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32}, fractional,
                                       ScalarType::Float32, {}, {}))
                      .status,
                  VerificationStatus::Unsupported);
    }
}

TEST(Freivalds, DetectsShiftedRowsAndDuplicatedColumns) {
    Tensor a(ScalarType::Float32, Shape{3, 1}), b(ScalarType::Float32, Shape{1, 4});
    for (size_t i = 0; i < 3; ++i) a.storeFrom({i, 0}, int(i + 1));
    for (size_t j = 0; j < 4; ++j) b.storeFrom({0, j}, int(j + 2));
    const auto plan = requirePlan(prepareFreivalds(a, b, MatmulOptions{ScalarType::Float32},
                                                   EpilogueOptions{ScalarType::Float32},
                                                   ScalarType::Float32, {}, {}));
    for (int fault = 0; fault < 3; ++fault) {
        Tensor d(ScalarType::Float32, Shape{3, 4});
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j) {
                const auto row = fault == 1 ? (i + 1) % 3 : i;
                const auto col = fault == 2 && j == 2 ? 1 : j;
                d.storeFrom({i, j}, int((row + 1) * (col + 2)));
            }
        EXPECT_EQ(plan.check(d, std::numeric_limits<size_t>::max(), {}).status,
                  fault ? VerificationStatus::Mismatch : VerificationStatus::Passed);
    }
}
}  // namespace
