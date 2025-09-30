// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>

TEST(TestBatchnormAttributes, CreateBatchnormAttributes)
{
    hipdnn_frontend::graph::BatchnormAttributes batchnormAttributes;

    batchnormAttributes.set_x(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_prev_running_mean(
        std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_prev_running_variance(
        std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_momentum(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_inv_variance(
        std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_next_running_mean(
        std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_next_running_variance(
        std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_epsilon(
        std::make_shared<hipdnn_frontend::graph::TensorAttributes>()); // Set epsilon

    auto xTensor = batchnormAttributes.get_x();
    xTensor->set_uid(1)
        .set_name("XTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto yTensor = batchnormAttributes.get_y();
    yTensor->set_uid(2)
        .set_name("YTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto biasTensor = batchnormAttributes.get_bias();
    biasTensor->set_uid(4)
        .set_name("BiasTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto prevMeanTensor = batchnormAttributes.get_prev_running_mean();
    prevMeanTensor->set_uid(5)
        .set_name("PrevMeanTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto prevVarianceTensor = batchnormAttributes.get_prev_running_variance();
    prevVarianceTensor->set_uid(6)
        .set_name("PrevVarianceTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto momentumTensor = batchnormAttributes.get_momentum();
    momentumTensor->set_uid(7)
        .set_name("MomentumTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto meanTensor = batchnormAttributes.get_mean();
    meanTensor->set_uid(8)
        .set_name("MeanTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto invVarianceTensor = batchnormAttributes.get_inv_variance();
    invVarianceTensor->set_uid(9)
        .set_name("InvVarianceTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto nextMeanTensor = batchnormAttributes.get_next_running_mean();
    nextMeanTensor->set_uid(10)
        .set_name("NextMeanTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto nextVarianceTensor = batchnormAttributes.get_next_running_variance();
    nextVarianceTensor->set_uid(11)
        .set_name("NextVarianceTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto epsilonTensor = batchnormAttributes.get_epsilon();
    epsilonTensor->set_uid(14)
        .set_name("EpsilonTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1})
        .set_stride({1});

    auto peerStat1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    peerStat1->set_uid(12)
        .set_name("PeerStat1")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2})
        .set_stride({3, 4});

    auto peerStat2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    peerStat2->set_uid(13)
        .set_name("PeerStat2")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({5, 6})
        .set_stride({7, 8});

    batchnormAttributes.set_peer_stats({peerStat1, peerStat2});

    const auto& peerStats = batchnormAttributes.get_peer_stats();
    ASSERT_EQ(peerStats.size(), 2);

    EXPECT_EQ(peerStats[0]->get_uid(), 12);
    EXPECT_EQ(peerStats[0]->get_name(), "PeerStat1");
    EXPECT_EQ(peerStats[0]->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(peerStats[0]->get_dim(), (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(peerStats[0]->get_stride(), (std::vector<int64_t>{3, 4}));

    EXPECT_EQ(peerStats[1]->get_uid(), 13);
    EXPECT_EQ(peerStats[1]->get_name(), "PeerStat2");
    EXPECT_EQ(peerStats[1]->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(peerStats[1]->get_dim(), (std::vector<int64_t>{5, 6}));
    EXPECT_EQ(peerStats[1]->get_stride(), (std::vector<int64_t>{7, 8}));

    EXPECT_EQ(xTensor->get_uid(), 1);
    EXPECT_EQ(xTensor->get_name(), "XTensor");
    EXPECT_EQ(xTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(xTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(xTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(yTensor->get_uid(), 2);
    EXPECT_EQ(yTensor->get_name(), "YTensor");
    EXPECT_EQ(yTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(scaleTensor->get_uid(), 3);
    EXPECT_EQ(scaleTensor->get_name(), "ScaleTensor");
    EXPECT_EQ(scaleTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(scaleTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(scaleTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(biasTensor->get_uid(), 4);
    EXPECT_EQ(biasTensor->get_name(), "BiasTensor");
    EXPECT_EQ(biasTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(biasTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(biasTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(prevMeanTensor->get_uid(), 5);
    EXPECT_EQ(prevMeanTensor->get_name(), "PrevMeanTensor");
    EXPECT_EQ(prevMeanTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(prevMeanTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(prevMeanTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(prevVarianceTensor->get_uid(), 6);
    EXPECT_EQ(prevVarianceTensor->get_name(), "PrevVarianceTensor");
    EXPECT_EQ(prevVarianceTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(prevVarianceTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(prevVarianceTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(momentumTensor->get_uid(), 7);
    EXPECT_EQ(momentumTensor->get_name(), "MomentumTensor");
    EXPECT_EQ(momentumTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(momentumTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(momentumTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(meanTensor->get_uid(), 8);
    EXPECT_EQ(meanTensor->get_name(), "MeanTensor");
    EXPECT_EQ(meanTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(meanTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(meanTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(invVarianceTensor->get_uid(), 9);
    EXPECT_EQ(invVarianceTensor->get_name(), "InvVarianceTensor");
    EXPECT_EQ(invVarianceTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(invVarianceTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(invVarianceTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(nextMeanTensor->get_uid(), 10);
    EXPECT_EQ(nextMeanTensor->get_name(), "NextMeanTensor");
    EXPECT_EQ(nextMeanTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(nextMeanTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(nextMeanTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(nextVarianceTensor->get_uid(), 11);
    EXPECT_EQ(nextVarianceTensor->get_name(), "NextVarianceTensor");
    EXPECT_EQ(nextVarianceTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(nextVarianceTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(nextVarianceTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(epsilonTensor->get_uid(), 14);
    EXPECT_EQ(epsilonTensor->get_name(), "EpsilonTensor");
    EXPECT_EQ(epsilonTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(epsilonTensor->get_dim(), (std::vector<int64_t>{1}));
    EXPECT_EQ(epsilonTensor->get_stride(), (std::vector<int64_t>{1}));
}

TEST(TestBatchnormAttributes, PackAttributes)
{
    hipdnn_frontend::graph::BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    xTensor->set_uid(1);
    batchnormAttributes.set_x(xTensor);

    auto yTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    yTensor->set_uid(2);
    batchnormAttributes.set_y(yTensor);

    auto scaleTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scaleTensor->set_uid(3);
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    biasTensor->set_uid(4);
    batchnormAttributes.set_bias(biasTensor);

    auto prevMeanTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    prevMeanTensor->set_uid(5);
    batchnormAttributes.set_prev_running_mean(prevMeanTensor);

    auto prevVarianceTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    prevVarianceTensor->set_uid(6);
    batchnormAttributes.set_prev_running_variance(prevVarianceTensor);

    auto momentumTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    momentumTensor->set_uid(7);
    batchnormAttributes.set_momentum(momentumTensor);

    auto meanTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    meanTensor->set_uid(8);
    batchnormAttributes.set_mean(meanTensor);

    auto invVarianceTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    invVarianceTensor->set_uid(9);
    batchnormAttributes.set_inv_variance(invVarianceTensor);

    auto nextMeanTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    nextMeanTensor->set_uid(10);
    batchnormAttributes.set_next_running_mean(nextMeanTensor);

    auto nextVarianceTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    nextVarianceTensor->set_uid(11);
    batchnormAttributes.set_next_running_variance(nextVarianceTensor);

    auto epsilonTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilonTensor->set_uid(14);
    batchnormAttributes.set_epsilon(epsilonTensor);

    auto peerStat1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    peerStat1->set_uid(12);

    auto peerStat2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    peerStat2->set_uid(13);

    batchnormAttributes.set_peer_stats({peerStat1, peerStat2});

    flatbuffers::FlatBufferBuilder builder;
    auto packedAttributes = batchnormAttributes.pack_attributes(builder);
    builder.Finish(packedAttributes);

    auto buffer = builder.GetBufferPointer();
    auto batchnormAttributesFb
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormAttributes>(buffer);

    EXPECT_EQ(batchnormAttributesFb->x_tensor_uid(), 1);
    EXPECT_EQ(batchnormAttributesFb->y_tensor_uid(), 2);
    EXPECT_EQ(batchnormAttributesFb->scale_tensor_uid(), 3);
    EXPECT_EQ(batchnormAttributesFb->bias_tensor_uid(), 4);
    EXPECT_EQ(batchnormAttributesFb->prev_running_mean_tensor_uid(), 5);
    EXPECT_EQ(batchnormAttributesFb->prev_running_variance_tensor_uid(), 6);
    EXPECT_EQ(batchnormAttributesFb->momentum_tensor_uid(), 7);
    EXPECT_EQ(batchnormAttributesFb->mean_tensor_uid(), 8);
    EXPECT_EQ(batchnormAttributesFb->inv_variance_tensor_uid(), 9);
    EXPECT_EQ(batchnormAttributesFb->next_running_mean_tensor_uid(), 10);
    EXPECT_EQ(batchnormAttributesFb->next_running_variance_tensor_uid(), 11);
    EXPECT_EQ(batchnormAttributesFb->epsilon_tensor_uid(), 14);

    ASSERT_EQ(batchnormAttributesFb->peer_stats_tensor_uid()->size(), 2);
    EXPECT_EQ(batchnormAttributesFb->peer_stats_tensor_uid()->Get(0), 12);
    EXPECT_EQ(batchnormAttributesFb->peer_stats_tensor_uid()->Get(1), 13);
}

TEST(TestBatchnormAttributes, PackAttributesWithoutOptionalValues)
{
    hipdnn_frontend::graph::BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    xTensor->set_uid(1);
    batchnormAttributes.set_x(xTensor);

    auto yTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    yTensor->set_uid(2);
    batchnormAttributes.set_y(yTensor);

    auto scaleTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scaleTensor->set_uid(3);
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    biasTensor->set_uid(4);
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilonTensor->set_uid(5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    flatbuffers::FlatBufferBuilder builder;
    auto packedAttributes = batchnormAttributes.pack_attributes(builder);
    builder.Finish(packedAttributes);

    auto buffer = builder.GetBufferPointer();
    auto batchnormAttributesFb
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormAttributes>(buffer);

    EXPECT_EQ(batchnormAttributesFb->x_tensor_uid(), 1);
    EXPECT_EQ(batchnormAttributesFb->y_tensor_uid(), 2);
    EXPECT_EQ(batchnormAttributesFb->scale_tensor_uid(), 3);
    EXPECT_EQ(batchnormAttributesFb->bias_tensor_uid(), 4);
    EXPECT_EQ(batchnormAttributesFb->epsilon_tensor_uid(), 5);

    EXPECT_EQ(batchnormAttributesFb->prev_running_mean_tensor_uid(), flatbuffers::nullopt);
    EXPECT_EQ(batchnormAttributesFb->prev_running_variance_tensor_uid(), flatbuffers::nullopt);
    EXPECT_EQ(batchnormAttributesFb->momentum_tensor_uid(), flatbuffers::nullopt);
    EXPECT_EQ(batchnormAttributesFb->mean_tensor_uid(), flatbuffers::nullopt);
    EXPECT_EQ(batchnormAttributesFb->inv_variance_tensor_uid(), flatbuffers::nullopt);
    EXPECT_EQ(batchnormAttributesFb->next_running_mean_tensor_uid(), flatbuffers::nullopt);
    EXPECT_EQ(batchnormAttributesFb->next_running_variance_tensor_uid(), flatbuffers::nullopt);

    ASSERT_EQ(batchnormAttributesFb->peer_stats_tensor_uid()->size(), 0);
}

TEST(TestBatchnormAttributes, SetXWithMove)
{
    hipdnn_frontend::graph::BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    xTensor->set_uid(1).set_name("XTensor");

    auto rawPtr = xTensor.get();

    batchnormAttributes.set_x(std::move(xTensor));

    auto retrieved = batchnormAttributes.get_x();
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "XTensor");
    EXPECT_EQ(xTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBatchnormAttributes, SetScaleWithMove)
{
    hipdnn_frontend::graph::BatchnormAttributes batchnormAttributes;

    auto scaleTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scaleTensor->set_uid(2).set_name("ScaleTensor");

    auto rawPtr = scaleTensor.get();

    batchnormAttributes.set_scale(std::move(scaleTensor));

    auto retrieved = batchnormAttributes.get_scale();
    EXPECT_EQ(retrieved->get_uid(), 2);
    EXPECT_EQ(retrieved->get_name(), "ScaleTensor");

    EXPECT_EQ(scaleTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBatchnormAttributes, SetBiasWithMove)
{
    hipdnn_frontend::graph::BatchnormAttributes batchnormAttributes;

    auto biasTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    biasTensor->set_uid(3).set_name("BiasTensor");

    auto rawPtr = biasTensor.get();

    batchnormAttributes.set_bias(std::move(biasTensor));

    auto retrieved = batchnormAttributes.get_bias();
    EXPECT_EQ(retrieved->get_uid(), 3);
    EXPECT_EQ(retrieved->get_name(), "BiasTensor");

    EXPECT_EQ(biasTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBatchnormAttributes, SetPeerStatsWithMove)
{
    hipdnn_frontend::graph::BatchnormAttributes batchnormAttributes;

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

TEST(TestBatchnormAttributes, SetPreviousRunningStatsWithMove)
{
    hipdnn_frontend::graph::BatchnormAttributes batchnormAttributes;

    auto meanTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    meanTensor->set_uid(20).set_name("MeanTensor");

    auto varianceTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    varianceTensor->set_uid(21).set_name("VarianceTensor");

    auto momentumTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    momentumTensor->set_uid(22).set_name("MomentumTensor");

    auto rawMeanPtr = meanTensor.get();
    auto rawVariancePtr = varianceTensor.get();
    auto rawMomentumPtr = momentumTensor.get();

    batchnormAttributes.set_previous_running_stats(
        std::move(meanTensor), std::move(varianceTensor), std::move(momentumTensor));

    auto retrievedMean = batchnormAttributes.get_prev_running_mean();
    EXPECT_EQ(retrievedMean->get_uid(), 20);
    EXPECT_EQ(retrievedMean->get_name(), "MeanTensor");

    auto retrievedVariance = batchnormAttributes.get_prev_running_variance();
    EXPECT_EQ(retrievedVariance->get_uid(), 21);
    EXPECT_EQ(retrievedVariance->get_name(), "VarianceTensor");

    auto retrievedMomentum = batchnormAttributes.get_momentum();
    EXPECT_EQ(retrievedMomentum->get_uid(), 22);
    EXPECT_EQ(retrievedMomentum->get_name(), "MomentumTensor");

    // Verify the objects were moved
    EXPECT_EQ(meanTensor, nullptr);
    EXPECT_EQ(varianceTensor, nullptr);
    EXPECT_EQ(momentumTensor, nullptr);

    // Verify the raw pointers match
    EXPECT_EQ(retrievedMean.get(), rawMeanPtr);
    EXPECT_EQ(retrievedVariance.get(), rawVariancePtr);
    EXPECT_EQ(retrievedMomentum.get(), rawMomentumPtr);
}

// Simplified move tests - testing move semantics without setting uid/name

TEST(TestBatchnormAttributes, SimplifiedSetXWithMove)
{
    hipdnn_frontend::graph::BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    batchnormAttributes.set_x(std::move(xTensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnormAttributes.get_x(), nullptr);
}

TEST(TestBatchnormAttributes, SimplifiedSetPeerStatsWithMove)
{
    hipdnn_frontend::graph::BatchnormAttributes batchnormAttributes;

    std::vector<std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>> peerStatsMove;
    peerStatsMove.push_back(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    peerStatsMove.push_back(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());

    size_t originalSize = peerStatsMove.size();
    batchnormAttributes.set_peer_stats(std::move(peerStatsMove));

    // Verify the vector was moved
    const auto& peerStats = batchnormAttributes.get_peer_stats();
    EXPECT_EQ(peerStats.size(), originalSize);
}

TEST(TestBatchnormAttributes, SimplifiedSetPreviousRunningStatsWithMove)
{
    hipdnn_frontend::graph::BatchnormAttributes batchnormAttributes;

    auto meanTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    auto varianceTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    auto momentumTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();

    batchnormAttributes.set_previous_running_stats(
        std::move(meanTensor), std::move(varianceTensor), std::move(momentumTensor));

    // Just verify the tensors were set
    EXPECT_NE(batchnormAttributes.get_prev_running_mean(), nullptr);
    EXPECT_NE(batchnormAttributes.get_prev_running_variance(), nullptr);
    EXPECT_NE(batchnormAttributes.get_momentum(), nullptr);
}
