// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "node_factory.h"
#include "plan.h"
#include "tree_node.h"

#include "../../../shared/device_properties.h"
#include "rocfft/rocfft.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

// ProcessNode is the whole planning pipeline: build the tree, collect leaves,
// assign buffers, fuse, pad and collapse.
static std::unique_ptr<ExecPlan> build_plan(const std::vector<size_t>& length,
                                            size_t                     batch,
                                            rocfft_transform_type      type,
                                            rocfft_result_placement    placement)
{
    static const bool setup_ok = []() { return rocfft_setup() == rocfft_status_success; }();
    EXPECT_TRUE(setup_ok);

    std::vector<size_t> stride(length.size(), 1);
    for(size_t i = 1; i < length.size(); ++i)
        stride[i] = stride[i - 1] * length[i - 1];
    const size_t dist = stride.back() * length.back();

    NodeMetaData rootPlanData(nullptr);
    rootPlanData.dimension         = length.size();
    rootPlanData.length            = length;
    rootPlanData.batch             = batch;
    rootPlanData.inStride          = stride;
    rootPlanData.outStride         = stride;
    rootPlanData.iDist             = dist;
    rootPlanData.oDist             = dist;
    rootPlanData.direction         = type == rocfft_transform_type_complex_forward ? -1 : 1;
    rootPlanData.placement         = placement;
    rootPlanData.precision         = rocfft_precision_single;
    rootPlanData.inArrayType       = rocfft_array_type_complex_interleaved;
    rootPlanData.outArrayType      = rocfft_array_type_complex_interleaved;
    rootPlanData.rootTransformType = type;
    rootPlanData.deviceProp        = get_curr_device_prop();

    auto plan        = std::make_unique<ExecPlan>(0, false, rocfft_location_t{});
    plan->deviceProp = rootPlanData.deviceProp;
    plan->rootPlan   = NodeFactory::CreateExplicitNode(rootPlanData, nullptr);
    plan->iLength    = rootPlanData.length;
    plan->oLength
        = rootPlanData.outputLength.empty() ? rootPlanData.length : rootPlanData.outputLength;

    plan->rootPlan->inStrideUnit  = BufferIsUnitStride(*plan, OB_USER_IN);
    plan->rootPlan->outStrideUnit = BufferIsUnitStride(*plan, OB_USER_OUT);

    ProcessNode(*plan);
    return plan;
}

static std::vector<ComputeScheme> schemes_of(const ExecPlan& plan)
{
    std::vector<ComputeScheme> schemes;
    for(const auto* node : plan.execSeq)
        schemes.push_back(node->scheme);
    return schemes;
}

static bool contains(const std::vector<ComputeScheme>& schemes, ComputeScheme scheme)
{
    return std::find(schemes.begin(), schemes.end(), scheme) != schemes.end();
}

static std::unique_ptr<ExecPlan> build_c2c(const std::vector<size_t>& length,
                                           rocfft_result_placement    placement
                                           = rocfft_placement_notinplace)
{
    return build_plan(length, 1, rocfft_transform_type_complex_forward, placement);
}

TEST(rocfft_internal, plan_small_1d_is_a_single_kernel)
{
    auto plan = build_c2c({64});

    ASSERT_EQ(plan->execSeq.size(), 1u);
    EXPECT_EQ(plan->execSeq.front()->scheme, CS_KERNEL_STOCKHAM);
}

TEST(rocfft_internal, plan_large_1d_splits_into_multiple_kernels)
{
    auto plan = build_c2c({8192});

    EXPECT_EQ(plan->rootPlan->scheme, CS_L1D_CC);
    EXPECT_GT(plan->execSeq.size(), 1u);
}

TEST(rocfft_internal, plan_prime_length_uses_bluestein)
{
    auto plan = build_c2c({1009});

    EXPECT_EQ(plan->rootPlan->scheme, CS_BLUESTEIN);
    EXPECT_TRUE(contains(schemes_of(*plan), CS_KERNEL_CHIRP));
}

TEST(rocfft_internal, plan_supported_length_avoids_bluestein)
{
    auto plan = build_c2c({64});

    EXPECT_NE(plan->rootPlan->scheme, CS_BLUESTEIN);
    EXPECT_FALSE(contains(schemes_of(*plan), CS_KERNEL_CHIRP));
}

TEST(rocfft_internal, plan_bluestein_decision_matches_radix_support)
{
    function_pool pool(get_curr_device_prop());

    EXPECT_TRUE(NodeFactory::SupportedLength(pool, rocfft_precision_single, 64));
    EXPECT_FALSE(NodeFactory::SupportedLength(pool, rocfft_precision_single, 1009));
}

TEST(rocfft_internal, plan_2d_and_3d_decompose_per_dimension)
{
    auto plan2d = build_c2c({64, 64});
    EXPECT_EQ(plan2d->rootPlan->scheme, CS_2D_RC);
    EXPECT_EQ(plan2d->execSeq.size(), 2u);

    auto plan3d = build_c2c({64, 64, 64});
    EXPECT_EQ(plan3d->rootPlan->scheme, CS_3D_RC);
    EXPECT_EQ(plan3d->execSeq.size(), 3u);
}

// If fusion stops firing these split into separate transpose nodes and the
// transform silently gets slower.
TEST(rocfft_internal, plan_large_3d_fuses_transpose_into_transform)
{
    auto       plan    = build_c2c({128, 128, 128});
    const auto schemes = schemes_of(*plan);

    EXPECT_EQ(plan->rootPlan->scheme, CS_3D_BLOCK_RC);
    EXPECT_TRUE(contains(schemes, CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z));
    EXPECT_FALSE(contains(schemes, CS_KERNEL_TRANSPOSE_XY_Z));
}

TEST(rocfft_internal, plan_every_node_has_buffers_assigned)
{
    for(const auto& length : std::vector<std::vector<size_t>>{{64}, {8192}, {64, 64}, {64, 64, 64}})
    {
        auto plan = build_c2c(length);
        ASSERT_FALSE(plan->execSeq.empty());
        for(const auto* node : plan->execSeq)
        {
            EXPECT_NE(node->obIn, OB_UNINIT);
            EXPECT_NE(node->obOut, OB_UNINIT);
        }
    }
}

TEST(rocfft_internal, plan_out_of_place_reads_input_and_writes_output)
{
    auto plan = build_c2c({64, 64}, rocfft_placement_notinplace);
    ASSERT_FALSE(plan->execSeq.empty());

    EXPECT_EQ(plan->execSeq.front()->obIn, OB_USER_IN);
    EXPECT_EQ(plan->execSeq.back()->obOut, OB_USER_OUT);
}

TEST(rocfft_internal, plan_in_place_never_reads_a_separate_input)
{
    auto plan = build_c2c({64, 64}, rocfft_placement_inplace);
    ASSERT_FALSE(plan->execSeq.empty());

    EXPECT_EQ(plan->execSeq.back()->obOut, OB_USER_OUT);
    for(const auto* node : plan->execSeq)
        EXPECT_NE(node->obIn, OB_USER_IN);
}

TEST(rocfft_internal, plan_single_kernel_uses_no_temp_buffer)
{
    auto plan = build_c2c({64});
    ASSERT_EQ(plan->execSeq.size(), 1u);

    for(const auto* node : plan->execSeq)
    {
        EXPECT_NE(node->obIn, OB_TEMP);
        EXPECT_NE(node->obOut, OB_TEMP);
    }
}
