// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>

TEST(TestBatchnormBackwardAttributes, CreateBatchnormBackwardAttributes)
{
    hipdnn_frontend::graph::BatchnormBackwardAttributes batchnormAttributes;

    batchnormAttributes.set_dy(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_x(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_inv_variance(
        std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_dx(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());

    auto dyTensor = batchnormAttributes.get_dy();
    dyTensor->set_uid(1)
        .set_name("DyTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto xTensor = batchnormAttributes.get_x();
    xTensor->set_uid(2)
        .set_name("XTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto meanTensor = batchnormAttributes.get_mean();
    meanTensor->set_uid(4)
        .set_name("MeanTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto invVarianceTensor = batchnormAttributes.get_inv_variance();
    invVarianceTensor->set_uid(5)
        .set_name("InvVarianceTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto dxTensor = batchnormAttributes.get_dx();
    dxTensor->set_uid(6)
        .set_name("DxTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto dscaleTensor = batchnormAttributes.get_dscale();
    dscaleTensor->set_uid(7)
        .set_name("DscaleTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto dbiasTensor = batchnormAttributes.get_dbias();
    dbiasTensor->set_uid(8)
        .set_name("DbiasTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto peerStat1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    peerStat1->set_uid(9)
        .set_name("PeerStat1")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2})
        .set_stride({3, 4});

    auto peerStat2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    peerStat2->set_uid(10)
        .set_name("PeerStat2")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({5, 6})
        .set_stride({7, 8});

    batchnormAttributes.set_peer_stats({peerStat1, peerStat2});

    EXPECT_EQ(dyTensor->get_uid(), 1);
    EXPECT_EQ(dyTensor->get_name(), "DyTensor");
    EXPECT_EQ(dyTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(dyTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(dyTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(xTensor->get_uid(), 2);
    EXPECT_EQ(xTensor->get_name(), "XTensor");
    EXPECT_EQ(xTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(xTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(xTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(scaleTensor->get_uid(), 3);
    EXPECT_EQ(scaleTensor->get_name(), "ScaleTensor");
    EXPECT_EQ(scaleTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(scaleTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(scaleTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(meanTensor->get_uid(), 4);
    EXPECT_EQ(meanTensor->get_name(), "MeanTensor");
    EXPECT_EQ(meanTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(meanTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(meanTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(invVarianceTensor->get_uid(), 5);
    EXPECT_EQ(invVarianceTensor->get_name(), "InvVarianceTensor");
    EXPECT_EQ(invVarianceTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(invVarianceTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(invVarianceTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(dxTensor->get_uid(), 6);
    EXPECT_EQ(dxTensor->get_name(), "DxTensor");
    EXPECT_EQ(dxTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(dxTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(dxTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(dscaleTensor->get_uid(), 7);
    EXPECT_EQ(dscaleTensor->get_name(), "DscaleTensor");
    EXPECT_EQ(dscaleTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(dscaleTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(dscaleTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(dbiasTensor->get_uid(), 8);
    EXPECT_EQ(dbiasTensor->get_name(), "DbiasTensor");
    EXPECT_EQ(dbiasTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(dbiasTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(dbiasTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    const auto& peerStats = batchnormAttributes.get_peer_stats();
    ASSERT_EQ(peerStats.size(), 2);

    EXPECT_EQ(peerStats[0]->get_uid(), 9);
    EXPECT_EQ(peerStats[0]->get_name(), "PeerStat1");
    EXPECT_EQ(peerStats[0]->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(peerStats[0]->get_dim(), (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(peerStats[0]->get_stride(), (std::vector<int64_t>{3, 4}));

    EXPECT_EQ(peerStats[1]->get_uid(), 10);
    EXPECT_EQ(peerStats[1]->get_name(), "PeerStat2");
    EXPECT_EQ(peerStats[1]->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(peerStats[1]->get_dim(), (std::vector<int64_t>{5, 6}));
    EXPECT_EQ(peerStats[1]->get_stride(), (std::vector<int64_t>{7, 8}));
}

TEST(TestBatchnormBackwardAttributes, SetDyWithMove)
{
    hipdnn_frontend::graph::BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dyTensor->set_uid(1).set_name("DyTensor");

    auto rawPtr = dyTensor.get();

    batchnormAttributes.set_dy(std::move(dyTensor));

    auto retrieved = batchnormAttributes.get_dy();
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "DyTensor");

    EXPECT_EQ(dyTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBatchnormBackwardAttributes, SetXWithMove)
{
    hipdnn_frontend::graph::BatchnormBackwardAttributes batchnormAttributes;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    xTensor->set_uid(2).set_name("XTensor");

    auto rawPtr = xTensor.get();

    batchnormAttributes.set_x(std::move(xTensor));

    auto retrieved = batchnormAttributes.get_x();
    EXPECT_EQ(retrieved->get_uid(), 2);
    EXPECT_EQ(retrieved->get_name(), "XTensor");

    EXPECT_EQ(xTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBatchnormBackwardAttributes, SetScaleWithMove)
{
    hipdnn_frontend::graph::BatchnormBackwardAttributes batchnormAttributes;

    auto scaleTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scaleTensor->set_uid(3).set_name("ScaleTensor");

    auto rawPtr = scaleTensor.get();

    batchnormAttributes.set_scale(std::move(scaleTensor));

    auto retrieved = batchnormAttributes.get_scale();
    EXPECT_EQ(retrieved->get_uid(), 3);
    EXPECT_EQ(retrieved->get_name(), "ScaleTensor");

    EXPECT_EQ(scaleTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBatchnormBackwardAttributes, SetDxWithMove)
{
    hipdnn_frontend::graph::BatchnormBackwardAttributes batchnormAttributes;

    auto dxTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dxTensor->set_uid(4).set_name("DxTensor");

    auto rawPtr = dxTensor.get();

    batchnormAttributes.set_dx(std::move(dxTensor));

    auto retrieved = batchnormAttributes.get_dx();
    EXPECT_EQ(retrieved->get_uid(), 4);
    EXPECT_EQ(retrieved->get_name(), "DxTensor");

    EXPECT_EQ(dxTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBatchnormBackwardAttributes, SetPeerStatsWithMove)
{
    hipdnn_frontend::graph::BatchnormBackwardAttributes batchnormAttributes;

    auto peerStat1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    peerStat1->set_uid(10).set_name("PeerStat1");

    auto peerStat2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    peerStat2->set_uid(11).set_name("PeerStat2");

    auto rawPtr1 = peerStat1.get();
    auto rawPtr2 = peerStat2.get();

    std::vector<std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>> peerStatsMove
        = {peerStat1, peerStat2};

    batchnormAttributes.set_peer_stats(std::move(peerStatsMove));

    const auto& peerStats = batchnormAttributes.get_peer_stats();
    ASSERT_EQ(peerStats.size(), 2);
    EXPECT_EQ(peerStats[0]->get_uid(), 10);
    EXPECT_EQ(peerStats[0]->get_name(), "PeerStat1");
    EXPECT_EQ(peerStats[1]->get_uid(), 11);
    EXPECT_EQ(peerStats[1]->get_name(), "PeerStat2");

    // Verify the raw pointers match (same objects were moved)
    EXPECT_EQ(peerStats[0].get(), rawPtr1);
    EXPECT_EQ(peerStats[1].get(), rawPtr2);
}

TEST(TestBatchnormBackwardAttributes, SetSavedMeanAndInvVarianceWithMove)
{
    hipdnn_frontend::graph::BatchnormBackwardAttributes batchnormAttributes;

    auto meanTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    meanTensor->set_uid(20).set_name("MeanTensor");

    auto invVarianceTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    invVarianceTensor->set_uid(21).set_name("InvVarianceTensor");

    auto rawMeanPtr = meanTensor.get();
    auto rawInvVariancePtr = invVarianceTensor.get();

    batchnormAttributes.set_saved_mean_and_inv_variance(std::move(meanTensor),
                                                        std::move(invVarianceTensor));

    auto retrievedMean = batchnormAttributes.get_mean();
    EXPECT_EQ(retrievedMean->get_uid(), 20);
    EXPECT_EQ(retrievedMean->get_name(), "MeanTensor");

    auto retrievedInvVariance = batchnormAttributes.get_inv_variance();
    EXPECT_EQ(retrievedInvVariance->get_uid(), 21);
    EXPECT_EQ(retrievedInvVariance->get_name(), "InvVarianceTensor");

    // Verify the objects were moved
    EXPECT_EQ(meanTensor, nullptr);
    EXPECT_EQ(invVarianceTensor, nullptr);

    // Verify the raw pointers match
    EXPECT_EQ(retrievedMean.get(), rawMeanPtr);
    EXPECT_EQ(retrievedInvVariance.get(), rawInvVariancePtr);
}

// Simplified move tests - testing move semantics without setting uid/name

TEST(TestBatchnormBackwardAttributes, SimplifiedSetDyWithMove)
{
    hipdnn_frontend::graph::BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    batchnormAttributes.set_dy(std::move(dyTensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnormAttributes.get_dy(), nullptr);
}

TEST(TestBatchnormBackwardAttributes, SimplifiedSetPeerStatsWithMove)
{
    hipdnn_frontend::graph::BatchnormBackwardAttributes batchnormAttributes;

    std::vector<std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>> peerStatsMove;
    peerStatsMove.push_back(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    peerStatsMove.push_back(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    peerStatsMove.push_back(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());

    size_t originalSize = peerStatsMove.size();
    batchnormAttributes.set_peer_stats(std::move(peerStatsMove));

    // Verify the vector was moved
    const auto& peerStats = batchnormAttributes.get_peer_stats();
    EXPECT_EQ(peerStats.size(), originalSize);
}

TEST(TestBatchnormBackwardAttributes, SimplifiedSetSavedMeanAndInvVarianceWithMove)
{
    hipdnn_frontend::graph::BatchnormBackwardAttributes batchnormAttributes;

    auto meanTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    auto invVarianceTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();

    batchnormAttributes.set_saved_mean_and_inv_variance(std::move(meanTensor),
                                                        std::move(invVarianceTensor));

    // Just verify the tensors were set
    EXPECT_NE(batchnormAttributes.get_mean(), nullptr);
    EXPECT_NE(batchnormAttributes.get_inv_variance(), nullptr);
}
