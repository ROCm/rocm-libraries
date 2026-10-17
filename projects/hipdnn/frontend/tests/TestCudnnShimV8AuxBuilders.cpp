// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x ReductionDesc, Resample, Rng and VariantPack builders
// against the shim's backend C API and the mock hipDNN backend.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_ReductionDesc.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Resample.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Rng.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_VariantPack.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

struct BuildFailure
{
    bool thrown = false;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    std::string what;
};

template <typename Builder>
BuildFailure buildFailure(Builder& builder)
{
    BuildFailure failure;
    try
    {
        auto built = builder.build();
        static_cast<void>(built);
    }
    catch(fe::cudnnException& e)
    {
        failure.thrown = true;
        failure.status = e.getCudnnStatus();
        failure.what = e.what();
    }
    return failure;
}

template <typename T>
T getScalar(cudnnBackendDescriptor_t descriptor,
            cudnnBackendAttributeName_t name,
            cudnnBackendAttributeType_t type)
{
    T value{};
    std::int64_t count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(descriptor, name, type, 1, &count, &value),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);
    return value;
}

std::vector<std::pair<std::int64_t, std::int64_t>> getFractions(cudnnBackendDescriptor_t descriptor,
                                                                cudnnBackendAttributeName_t name)
{
    std::array<cudnnFraction_t, CUDNN_DIM_MAX> values{};
    std::int64_t count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(descriptor,
                                       name,
                                       CUDNN_TYPE_FRACTION,
                                       static_cast<std::int64_t>(values.size()),
                                       &count,
                                       values.data()),
              CUDNN_STATUS_SUCCESS);
    std::vector<std::pair<std::int64_t, std::int64_t>> result;
    for(std::int64_t i = 0; i < count; ++i)
    {
        const cudnnFraction_t& value = values[static_cast<std::size_t>(i)];
        result.emplace_back(value.numerator, value.denominator);
    }
    return result;
}

class TestCudnnShimV8AuxBuilders : public hipdnn_shim_test::ShimV8MockBackendFixture
{
protected:
    static void configureMaxPool(fe::ResampleDescBuilder_v8& builder)
    {
        const std::array<std::int64_t, 2> window = {3, 3};
        builder.setResampleMode(fe::ResampleMode_t::MAXPOOL)
            .setComputeType(fe::DataType_t::FLOAT)
            .setPaddingMode(CUDNN_NEG_INF_PAD)
            .setSpatialDim(2, window.data());
        const std::array<std::int64_t, 2> pre = {1, 0};
        const std::array<cudnnFraction_t, 2> post = {{{0, 1}, {1, 1}}};
        const std::array<std::int64_t, 2> stride = {2, 2};
        builder.setPrePadding(2, pre.data())
            .setPostPadding(2, post.data())
            .setSpatialStride(2, stride.data());
    }

    bool variantPackCreatedOnHipdnn() const
    {
        return !created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR).empty();
    }
};

TEST_F(TestCudnnShimV8AuxBuilders, ReductionBuilderSetsShimAttributes)
{
    fe::ReductionDescBuilder_v8 builder;
    builder.setComputeType(fe::DataType_t::FLOAT)
        .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
        .setIsDeterministic(true);
    auto reduction = builder.build();

    ASSERT_EQ(reduction.get_status(), CUDNN_STATUS_SUCCESS);
    cudnnBackendDescriptor_t raw = reduction.get_raw_desc();
    EXPECT_EQ(getScalar<cudnnDataType_t>(raw, CUDNN_ATTR_REDUCTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE),
              CUDNN_DATA_FLOAT);
    EXPECT_EQ(getScalar<cudnnReduceTensorOp_t>(
                  raw, CUDNN_ATTR_REDUCTION_OPERATOR, CUDNN_TYPE_REDUCTION_OPERATOR_TYPE),
              CUDNN_REDUCE_TENSOR_MAX);
    EXPECT_TRUE(getScalar<bool>(raw, CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC, CUDNN_TYPE_BOOLEAN));

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    EXPECT_NE(reduction.describe().find(R"(Reduction mode ["MAX"] Is deterministic true)"),
              std::string::npos)
        << reduction.describe();
#endif
}

TEST_F(TestCudnnShimV8AuxBuilders, ResampleBuilderSetsShimAttributes)
{
    fe::ResampleDescBuilder_v8 builder;
    configureMaxPool(builder);
    auto resample = builder.build();

    ASSERT_EQ(resample.get_status(), CUDNN_STATUS_SUCCESS);
    cudnnBackendDescriptor_t raw = resample.get_raw_desc();
    EXPECT_EQ(
        getScalar<cudnnResampleMode_t>(raw, CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE),
        CUDNN_RESAMPLE_MAXPOOL);
    EXPECT_EQ(getScalar<cudnnPaddingMode_t>(
                  raw, CUDNN_ATTR_RESAMPLE_PADDING_MODE, CUDNN_TYPE_PADDING_MODE),
              CUDNN_NEG_INF_PAD);
    EXPECT_EQ(getScalar<std::int64_t>(raw, CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, CUDNN_TYPE_INT64), 2);

    using Fractions = std::vector<std::pair<std::int64_t, std::int64_t>>;
    EXPECT_EQ(getFractions(raw, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS), (Fractions{{3, 1}, {3, 1}}));
    EXPECT_EQ(getFractions(raw, CUDNN_ATTR_RESAMPLE_PRE_PADDINGS), (Fractions{{1, 1}, {0, 1}}));
    EXPECT_EQ(getFractions(raw, CUDNN_ATTR_RESAMPLE_POST_PADDINGS), (Fractions{{0, 1}, {1, 1}}));
    EXPECT_EQ(getFractions(raw, CUDNN_ATTR_RESAMPLE_STRIDES), (Fractions{{2, 1}, {2, 1}}));
}

TEST_F(TestCudnnShimV8AuxBuilders, ResampleCudnnAvgpoolModeBuilds)
{
    fe::ResampleDescBuilder_v8 builder;
    configureMaxPool(builder);
    builder.setResampleMode(CUDNN_RESAMPLE_AVGPOOL).setPaddingMode(fe::PaddingMode_t::ZERO_PAD);
    auto resample = builder.build();

    ASSERT_EQ(resample.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(resample.getMode(), fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING);
    EXPECT_EQ(getScalar<cudnnResampleMode_t>(
                  resample.get_raw_desc(), CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE),
              CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING);
}

TEST_F(TestCudnnShimV8AuxBuilders, ResampleNearestIsNotSupported)
{
    fe::ResampleDescBuilder_v8 builder;
    configureMaxPool(builder);
    builder.setResampleMode(fe::ResampleMode_t::NEAREST);

    const BuildFailure failure = buildFailure(builder);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(failure.what.find("CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: cudnnFinalize Failed"),
              std::string::npos)
        << failure.what;
    EXPECT_NE(failure.what.find("CUDNN_ATTR_RESAMPLE_MODE"), std::string::npos) << failure.what;
}

TEST_F(TestCudnnShimV8AuxBuilders, RngGettersReturnSetValues)
{
    fe::RngDescBuilder_v8 pairs;
    pairs.setRngDistribution(CUDNN_RNG_DISTRIBUTION_UNIFORM)
        .setNormalDistParams(1.5, 2.5)
        .setUniformDistParams(7.0, -3.0)
        .setBernoulliDistProbability(0.25);
    auto fromPairs = pairs.build();

    ASSERT_EQ(fromPairs.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(fromPairs.getDistribution(), fe::RngDistribution_t::UNIFORM);
    EXPECT_EQ(fromPairs.getNormalDistMean(), 1.5);
    EXPECT_EQ(fromPairs.getNormalDistStdDev(), 2.5);
    EXPECT_EQ(fromPairs.getUniformDistMax(), 7.0);
    EXPECT_EQ(fromPairs.getUniformDistMin(), -3.0);
    EXPECT_EQ(fromPairs.getBernoulliDistProbability(), 0.25);
    EXPECT_EQ(getScalar<double>(
                  fromPairs.get_raw_desc(), CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM, CUDNN_TYPE_DOUBLE),
              -3.0);

    fe::RngDescBuilder_v8 singles;
    singles.setRngDistribution(fe::RngDistribution_t::NORMAL)
        .setNormalDistMean(4.0)
        .setNormalDistStdDev(5.0)
        .setUniformDistMax(6.0)
        .setUniformDistMin(-6.0)
        .setBernoulliDistProbability(0.75);
    auto fromSingles = singles.build();

    ASSERT_EQ(fromSingles.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(fromSingles.getDistribution(), fe::RngDistribution_t::NORMAL);
    EXPECT_EQ(fromSingles.getNormalDistMean(), 4.0);
    EXPECT_EQ(fromSingles.getNormalDistStdDev(), 5.0);
    EXPECT_EQ(fromSingles.getUniformDistMax(), 6.0);
    EXPECT_EQ(fromSingles.getUniformDistMin(), -6.0);
    EXPECT_EQ(fromSingles.getBernoulliDistProbability(), 0.75);
}

TEST_F(TestCudnnShimV8AuxBuilders, VariantPackForwardsToHipdnn)
{
    int x = 0;
    int y = 0;
    int workspace = 0;
    std::array<void*, 2> pointers = {&x, &y};
    const std::array<std::int64_t, 2> uids = {11, 22};

    fe::VariantPackBuilder_v8 builder;
    builder.setDataPointers(2, pointers.data())
        .setUids(2, uids.data())
        .setWorkspacePointer(&workspace);
    auto pack = builder.build();
    ASSERT_EQ(pack.get_status(), CUDNN_STATUS_SUCCESS);

    const std::vector<hipdnnBackendDescriptor_t> packs
        = created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    ASSERT_EQ(packs.size(), 1U);
    const std::vector<hipdnn_shim_test::CapturedSetAttribute> sets = setAttributesOf(packs[0]);
    ASSERT_EQ(sets.size(), 3U);
    EXPECT_EQ(sets[0].name, HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS);
    EXPECT_EQ(sets[0].values<void*>(), (std::vector<void*>{&x, &y}));
    EXPECT_EQ(sets[1].name, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS);
    EXPECT_EQ(sets[1].values<std::int64_t>(), (std::vector<std::int64_t>{11, 22}));
    EXPECT_EQ(sets[2].name, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE);
    EXPECT_EQ(sets[2].value<void*>(), static_cast<void*>(&workspace));
    const std::vector<hipdnnBackendDescriptor_t> done = finalized();
    EXPECT_NE(std::find(done.begin(), done.end(), packs[0]), done.end());
}

TEST_F(TestCudnnShimV8AuxBuilders, VariantPackFromUidPointerSet)
{
    int a = 0;
    int b = 0;
    int c = 0;
    const std::set<std::pair<std::uint64_t, void*>> uidPointers = {{30, &c}, {10, &a}, {20, &b}};

    fe::VariantPackBuilder_v8 builder;
    builder.setDataPointers(uidPointers);
    auto pack = builder.build();
    ASSERT_EQ(pack.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_NE(pack.describe().find("has 3 data pointers"), std::string::npos) << pack.describe();

    const std::vector<hipdnnBackendDescriptor_t> packs
        = created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    ASSERT_EQ(packs.size(), 1U);
    const std::vector<hipdnn_shim_test::CapturedSetAttribute> sets = setAttributesOf(packs[0]);
    ASSERT_EQ(sets.size(), 3U);
    EXPECT_EQ(sets[0].values<void*>(), (std::vector<void*>{&a, &b, &c}));
    EXPECT_EQ(sets[1].values<std::int64_t>(), (std::vector<std::int64_t>{10, 20, 30}));
    EXPECT_EQ(sets[2].value<void*>(), nullptr);
}

TEST_F(TestCudnnShimV8AuxBuilders, VariantPackUidCountMismatchIsBadParam)
{
    int x = 0;
    int y = 0;
    std::array<void*, 2> pointers = {&x, &y};
    std::array<std::int64_t, 1> uids = {11};

    fe::VariantPackBuilder_v8 builder;
    builder.setDataPointers(2, pointers.data()).setUids(1, uids.data());

    const BuildFailure failure = buildFailure(builder);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(failure.what.find("CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS has 2 elements but "
                                "CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS has 1"),
              std::string::npos)
        << failure.what;
    EXPECT_FALSE(variantPackCreatedOnHipdnn());
}

TEST_F(TestCudnnShimV8AuxBuilders, EmptyVariantPackIsBadParam)
{
    fe::VariantPackBuilder_v8 builder;

    const BuildFailure failure = buildFailure(builder);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(failure.what.find("CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS"), std::string::npos)
        << failure.what;
    EXPECT_FALSE(variantPackCreatedOnHipdnn());
}

} // namespace
