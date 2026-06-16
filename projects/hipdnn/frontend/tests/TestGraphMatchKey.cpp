// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU-free unit test for the autotune config-file match-key selectors
// (hipdnn_frontend::detail::getCoreOperationName / getMatchKeyTensors). It
// builds real convolution graphs through the shared SelectorUnitGraph helper
// (which delegates structure to FrontendGraphFactory, no handle, no device) and
// asserts the op string plus the canonical tensor set/order the writer emits
// (Gap B).
//
// The helper assigns UIDs to the graph's physical tensors (the selector filters
// on has_uid() and pre-build graphs have none) and reports them back keyed by
// role/name, so each test DISCOVERS its tensors (x/w/dy) and forms its EXPECT
// values from the discovered UIDs rather than hand-rolling the graph.

#include <gtest/gtest.h>
#include <hipdnn_frontend/detail/GraphMatchKey.hpp>
#include <hipdnn_test_sdk/utilities/SelectorUnitGraph.hpp>

using namespace hipdnn_frontend;
using hipdnn_test_sdk::utilities::OperationType;
using hipdnn_test_sdk::utilities::SelectorUnitGraph;

class TestGraphMatchKey : public ::testing::Test
{
};

TEST_F(TestGraphMatchKey, ConvFpropOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::CONV_FORWARD);
    const auto& x = unitGraph.byName("x");
    const auto& w = unitGraph.byName("w");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "conv_fprop");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 2u);
    // Canonical order: (x, w); output (y) excluded (virtual by default).
    EXPECT_EQ(tensors[0]->get_uid(), x->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), w->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), x->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), x->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), w->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), w->get_stride());
}

TEST_F(TestGraphMatchKey, ConvDgradOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::CONV_BACKWARD_DATA);
    const auto& dy = unitGraph.byName("dy");
    const auto& w = unitGraph.byName("w");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "conv_dgrad");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 2u);
    // Canonical order: (dy, w); output (dx) excluded.
    EXPECT_EQ(tensors[0]->get_uid(), dy->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), w->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), dy->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), dy->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), w->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), w->get_stride());
}

TEST_F(TestGraphMatchKey, ConvWgradOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::CONV_BACKWARD_WEIGHTS);
    const auto& x = unitGraph.byName("x");
    const auto& dy = unitGraph.byName("dy");

    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "conv_wgrad");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    ASSERT_EQ(tensors.size(), 2u);
    // Canonical order: (x, dy) — note this is NOT the input enum order (dy, x);
    // output (dw) excluded.
    EXPECT_EQ(tensors[0]->get_uid(), x->get_uid());
    EXPECT_EQ(tensors[1]->get_uid(), dy->get_uid());
    EXPECT_EQ(tensors[0]->get_dim(), x->get_dim());
    EXPECT_EQ(tensors[0]->get_stride(), x->get_stride());
    EXPECT_EQ(tensors[1]->get_dim(), dy->get_dim());
    EXPECT_EQ(tensors[1]->get_stride(), dy->get_stride());
}

TEST_F(TestGraphMatchKey, UnsupportedOpYieldsEmptyMatchKeyAndGraphNameOpString)
{
    // REDUCTION has a factory builder (so it is GPU-free constructible) but NO
    // op-aware branch in getMatchKeyTensors and NO op-string case in
    // getCoreOperationName. It is intentionally unsupported for config
    // round-trip, so the selectors must report it as such: an EMPTY match key
    // and the graph-name op-string fallback (NOT a specific op string). This
    // pins the empty-key contract — a future regression (re-adding a generic
    // UID-sort fallback, or REDUCTION accidentally gaining a branch) is caught.
    SelectorUnitGraph unitGraph(OperationType::REDUCTION);

    // Graph-name fallback, not a specific op string (createReductionGraph sets
    // the graph name to "Test_Reduction").
    EXPECT_EQ(detail::getCoreOperationName(unitGraph.graph()), "Test_Reduction");

    auto tensors = detail::getMatchKeyTensors(unitGraph.graph());
    EXPECT_EQ(tensors.size(), 0u);
}
