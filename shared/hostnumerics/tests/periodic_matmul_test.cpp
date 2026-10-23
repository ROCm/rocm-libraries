// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <roc/hostnumerics/generation.hpp>
#include <roc/hostnumerics/periodic_matmul.hpp>

namespace {
using namespace roc::hostnumerics;

PeriodicMatmulSegment segment(size_t m, size_t n, size_t begin, size_t k, size_t p, size_t q,
                              uint64_t seed) {
    auto recipe = GenerationRecipe::realOnly(
        GenerationRecipe::uniformInteger({.lower = -2, .upper = 2}), {.seed = seed});
    auto a = generate(ScalarType::BFloat16, Shape{p, k}, recipe);
    auto b = generate(ScalarType::BFloat16, Shape{k, q}, recipe);
    return {begin, PeriodicInitialization(a, Shape{m, k}), PeriodicInitialization(b, Shape{k, n})};
}

PeriodicMatmulPlan requirePlan(PeriodicMatmulPreparation prepared) {
    if (const auto* fail = std::get_if<VerificationFailure>(&prepared))
        throw std::runtime_error(fail->reason);
    return std::get<PeriodicMatmulPlan>(std::move(prepared));
}

TEST(PeriodicMatmul, ExactDenseOracleChunksAndEpilogue) {
    for (size_t k : {0, 1, 5, 201}) {
        SCOPED_TRACE(k);
        constexpr size_t m = 17, n = 13;
        const std::array segments{segment(m, n, 0, k / 2, 3, 5, 17),
                                  segment(m, n, k / 2, k - k / 2, 5, 3, 91)};
        auto plan = requirePlan(preparePeriodicMatmul(Shape{m, n}, k, segments,
                                                      MatmulOptions{ScalarType::Float32},
                                                      PeriodicMatmulMode::ExactInteger, {}));
        Tensor a(ScalarType::BFloat16, Layout(Shape{m, k}, {1, m + 3}, 0));
        Tensor b(ScalarType::BFloat16, Layout(Shape{k, n}, {1, static_cast<ptrdiff_t>(k + 2)}, 0));
        for (const auto& s : segments) {
            s.a.fillInto(a.shareStorageWithLayout(Layout(
                s.a.shape(), {1, m + 3}, static_cast<ptrdiff_t>(s.reductionBegin * (m + 3)))));
            s.b.fillInto(
                b.shareStorageWithLayout(Layout(s.b.shape(), {1, static_cast<ptrdiff_t>(k + 2)},
                                                static_cast<ptrdiff_t>(s.reductionBegin))));
        }
        Tensor dense(ScalarType::Float32, Shape{m, n});
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) {
                int64_t sum = 0;
                for (size_t z = 0; z < k; ++z)
                    sum += a.loadAs<int64_t>({i, z}) * b.loadAs<int64_t>({z, j});
                dense.storeFrom({i, j}, 2 * sum - int64_t(i + 2 * j) + int64_t(3 * i));
            }
        for (size_t start = 0; start < n; start += 4) {
            const size_t width = std::min(size_t(4), n - start);
            const auto observed = dense.shareStorageWithLayout(
                Layout(Shape{m, width}, {n, 1}, static_cast<ptrdiff_t>(start)));
            Tensor c(ScalarType::Float32, Shape{m, width});
            Tensor bias(ScalarType::Float32, Shape{m, 1});
            for (size_t i = 0; i < m; ++i) {
                bias.storeFrom({i, 0}, int(3 * i));
                for (size_t j = 0; j < width; ++j) c.storeFrom({i, j}, int(i + 2 * (start + j)));
            }
            EpilogueOptions e(ScalarType::Float32);
            e.inputScale = Tensor(2.0f);
            e.addendScale = Tensor(-1.0f);
            e.addend = c;
            e.bias = bias;
            auto report = plan.checkRegion(observed, std::array<size_t, 2>{0, start}, e);
            EXPECT_EQ(report.status, VerificationStatus::Passed) << report.reason;
            EXPECT_EQ(report.comparison.compared, m * width);
            EXPECT_EQ(report.absoluteTolerance, 0);
        }
        EXPECT_EQ(plan.statistics().referenceMultiplyAdds, 15 * k);
    }
}

TEST(PeriodicMatmul, ReferenceUsesStoredBFloat16) {
    Tensor a(ScalarType::BFloat16, Shape{1, 2});
    a.storeFrom({0, 0}, 1.001f);
    a.storeFrom({0, 1}, 1.001f);
    Tensor b(ScalarType::BFloat16, Shape{2, 1});
    b.storeFrom({0, 0}, 1.0f);
    b.storeFrom({1, 0}, 1.0f);
    const std::array segments{PeriodicMatmulSegment{0, PeriodicInitialization(a, Shape{3, 2}),
                                                    PeriodicInitialization(b, Shape{2, 4})}};
    auto plan = requirePlan(preparePeriodicMatmul(Shape{3, 4}, 2, segments,
                                                  MatmulOptions{ScalarType::Float32},
                                                  PeriodicMatmulMode::FloatingPoint, {}));
    a.storeFrom({0, 0}, 100.0f);
    const auto reference = plan.productRegion(std::array<size_t, 2>{2, 3}, Shape{1, 1});
    EXPECT_EQ(reference.loadAs<double>({0, 0}), 2);
    Tensor observed(ScalarType::Float32, Shape{1, 1});
    observed.storeFrom({0, 0}, 2.0f);
    EXPECT_EQ(plan.checkRegion(observed, std::array<size_t, 2>{2, 3},
                               EpilogueOptions{ScalarType::Float32})
                  .status,
              VerificationStatus::Passed);
    observed.storeFrom({0, 0}, 2.002f);
    const auto bad = plan.checkRegion(observed, std::array<size_t, 2>{2, 3},
                                      EpilogueOptions{ScalarType::Float32});
    EXPECT_EQ(bad.status, VerificationStatus::Mismatch);
    ASSERT_EQ(bad.comparison.reportedMismatches.size(), 1);
    EXPECT_EQ(bad.comparison.reportedMismatches[0].coordinates, (std::vector<size_t>{2, 3}));
    EXPECT_EQ(bad.comparison.reportedMismatches[0].index, 11);
}

TEST(PeriodicMatmul, FloatingCancellationAndDifferentReductionOrders) {
    const std::array<float, 4> values{0x1p25f, 1.5f, -0x1p25f, 0.25f};
    std::vector<PeriodicMatmulSegment> segments;
    for (size_t s = 0; s < 2; ++s) {
        Tensor a(ScalarType::BFloat16, Shape{1, 2});
        Tensor b(ScalarType::BFloat16, Shape{2, 1});
        for (size_t z = 0; z < 2; ++z) {
            a.storeFrom({0, z}, values[2 * s + z]);
            b.storeFrom({z, 0}, 1.0f);
        }
        segments.push_back({2 * s, PeriodicInitialization(a, Shape{1, 2}),
                            PeriodicInitialization(b, Shape{2, 1})});
    }
    auto plan = requirePlan(preparePeriodicMatmul(Shape{1, 1}, 4, segments,
                                                  MatmulOptions{ScalarType::Float32},
                                                  PeriodicMatmulMode::FloatingPoint, {}));
    EXPECT_EQ(plan.productRegion(std::array<size_t, 2>{0, 0}, Shape{1, 1}).loadAs<double>({0, 0}),
              1.75);
    float sequential = 0;
    for (float v : values) sequential = std::fma(v, 1.0f, sequential);
    const float split = (values[0] + values[1]) + (values[2] + values[3]);
    EXPECT_NE(sequential, 1.75f);
    for (float result : {sequential, split}) {
        Tensor observed(ScalarType::Float32, Shape{1, 1});
        observed.storeFrom({0, 0}, result);
        const auto report = plan.checkRegion(observed, std::array<size_t, 2>{0, 0},
                                             EpilogueOptions{ScalarType::Float32});
        EXPECT_EQ(report.status, VerificationStatus::Passed) << report.reason;
        EXPECT_GT(report.absoluteTolerance, 0);
        observed.storeFrom({0, 0}, float(1.75 + 4 * report.absoluteTolerance));
        EXPECT_EQ(plan.checkRegion(observed, std::array<size_t, 2>{0, 0},
                                   EpilogueOptions{ScalarType::Float32})
                      .status,
                  VerificationStatus::Mismatch);
    }
    auto exact = preparePeriodicMatmul(Shape{1, 1}, 4, segments, MatmulOptions{ScalarType::Float32},
                                       PeriodicMatmulMode::ExactInteger, {});
    ASSERT_TRUE(std::holds_alternative<VerificationFailure>(exact));
    EXPECT_EQ(std::get<VerificationFailure>(exact).status, VerificationStatus::Unsupported);
}

TEST(PeriodicMatmul, UnsupportedIncompleteAndInvalidAreDistinct) {
    const std::array segments{segment(17, 13, 0, 5, 3, 5, 7)};
    MatmulOptions scaled(ScalarType::Float32);
    scaled.preQuantizationScalesA.push_back(Tensor(2.0f));
    auto unsupported = preparePeriodicMatmul(Shape{17, 13}, 5, segments, scaled,
                                             PeriodicMatmulMode::FloatingPoint, {});
    EXPECT_EQ(std::get<VerificationFailure>(unsupported).status, VerificationStatus::Unsupported);
    PeriodicMatmulLimits limits;
    limits.maxReferenceMultiplyAdds = 1;
    auto incomplete =
        preparePeriodicMatmul(Shape{17, 13}, 5, segments, MatmulOptions{ScalarType::Float32},
                              PeriodicMatmulMode::FloatingPoint, limits);
    EXPECT_EQ(std::get<VerificationFailure>(incomplete).status, VerificationStatus::Incomplete);
    EXPECT_THROW(
        preparePeriodicMatmul(Shape{17, 13}, 6, segments, MatmulOptions{ScalarType::Float32},
                              PeriodicMatmulMode::FloatingPoint, {}),
        std::runtime_error);
    limits.maxReferenceMultiplyAdds = 1000;
    limits.maxCheckBytes = 32;
    auto plan = requirePlan(preparePeriodicMatmul(Shape{17, 13}, 5, segments,
                                                  MatmulOptions{ScalarType::Float32},
                                                  PeriodicMatmulMode::FloatingPoint, limits));
    Tensor observed(ScalarType::Float32, Shape{2, 2});
    EXPECT_EQ(plan.checkRegion(observed, std::array<size_t, 2>{0, 0},
                               EpilogueOptions{ScalarType::Float32})
                  .status,
              VerificationStatus::Incomplete);
    EXPECT_THROW(plan.productRegion(std::array<size_t, 2>{0, 12}, Shape{2, 2}), std::runtime_error);
}

TEST(PeriodicMatmul, ExactBoundRejectsCancellationAndFractionalEpilogue) {
    Tensor a(ScalarType::Float32, Shape{1, 2});
    Tensor b(ScalarType::Float32, Shape{2, 1});
    a.storeFrom({0, 0}, 0x1p24f);
    a.storeFrom({0, 1}, -0x1p24f);
    b.storeFrom({0, 0}, 1.0f);
    b.storeFrom({1, 0}, 1.0f);
    const std::array segments{PeriodicMatmulSegment{0, PeriodicInitialization(a, Shape{1, 2}),
                                                    PeriodicInitialization(b, Shape{2, 1})}};
    EXPECT_TRUE(std::holds_alternative<VerificationFailure>(
        preparePeriodicMatmul(Shape{1, 1}, 2, segments, MatmulOptions{ScalarType::Float32},
                              PeriodicMatmulMode::ExactInteger, {})));
    const auto zero =
        requirePlan(preparePeriodicMatmul(Shape{1, 1}, 0, {}, MatmulOptions{ScalarType::Float32},
                                          PeriodicMatmulMode::ExactInteger, {}));
    EpilogueOptions e(ScalarType::Float32);
    e.addend = Tensor(0.5f);
    Tensor observed(ScalarType::Float32, Shape{1, 1});
    observed.storeFrom({0, 0}, 0.5f);
    EXPECT_EQ(zero.checkRegion(observed, std::array<size_t, 2>{0, 0}, e).status,
              VerificationStatus::Unsupported);
}

TEST(PeriodicMatmul, EmptyOutputAndUnderflowRisk) {
    auto empty =
        requirePlan(preparePeriodicMatmul(Shape{0, 7}, 0, {}, MatmulOptions{ScalarType::Float32},
                                          PeriodicMatmulMode::FloatingPoint, {}));
    EpilogueOptions e(ScalarType::Float32);
    e.addend = Tensor(std::numeric_limits<float>::quiet_NaN());
    Tensor observed(ScalarType::Float32, Shape{0, 7});
    EXPECT_EQ(empty.checkRegion(observed, std::array<size_t, 2>{0, 0}, e).status,
              VerificationStatus::Passed);

    for (float value : {0x1p-100f, std::numeric_limits<float>::denorm_min(),
                        std::numeric_limits<float>::infinity()}) {
        Tensor a(ScalarType::Float32, Shape{1, 1});
        a.storeFrom({0, 0}, value);
        const std::array segments{PeriodicMatmulSegment{0, PeriodicInitialization(a, Shape{1, 1}),
                                                        PeriodicInitialization(a, Shape{1, 1})}};
        auto rejected =
            preparePeriodicMatmul(Shape{1, 1}, 1, segments, MatmulOptions{ScalarType::Float32},
                                  PeriodicMatmulMode::FloatingPoint, {});
        EXPECT_EQ(std::get<VerificationFailure>(rejected).status, VerificationStatus::Unsupported);
    }
}
}  // namespace
