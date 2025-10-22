// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/FileUtilities.hpp>
#include <hipdnn_sdk/test_utilities/ScopedExecute.hpp>
#include <hipdnn_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

namespace hipdnn_sdk::utilities
{

TEST(TestFillTensorFromFile, InvalidPath)
{
    Tensor<float> tensor({1});
    std::filesystem::path filepath = "./ea0w399059.txt";
    EXPECT_FALSE(std::filesystem::exists(filepath));
    EXPECT_THROW(detail::fillTensorFromFile(tensor, filepath), std::runtime_error);
}

TEST(TestFillTensorFromFile, PathToDirectory)
{
    Tensor<float> tensor({1});
    hipdnn_sdk::test_utilities::ScopedDirectory dir("oijaweorij33");
    EXPECT_THROW(detail::fillTensorFromFile(tensor, dir.path()), std::runtime_error);
}

template <class T>
void writeVectorToFile(const std::filesystem::path& filename, const std::vector<T>& values)
{
    std::ofstream f(filename, std::ios_base::binary);
    ASSERT_TRUE(f.good());

    f.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(int)));
}

TEST(TestFillTensorFromFile, Valid)
{
    std::filesystem::path filename = "SimpleTensor0123.bin";
    test_utilities::ScopedExecute fileDeleter([filename]() { std::filesystem::remove(filename); });

    std::vector<int> values{0, 1, 2, 3};
    writeVectorToFile(filename, values);

    Tensor<int> tensor({static_cast<int64_t>(values.size())});
    ASSERT_NO_THROW(detail::fillTensorFromFile(tensor, filename));

    ASSERT_EQ(tensor.memory().count(), values.size());

    for(size_t i = 0; i < values.size(); i++)
    {
        EXPECT_EQ(values[i], tensor.memory().hostData()[i]);
    }
}

TEST(TestLoadGraphAndTensors, Valid)
{
    std::filesystem::path filepath
        = utilities::getCurrentExecutableDirectory()
          / "../lib/hipdnn_reference_data/BatchnormFwdInference/nchw/fp32/Small.json";

    // TODO: Temporary fix until reference data can be properly installed
    if(!std::filesystem::exists(filepath))
    {
        HIPDNN_LOG_WARN("Could not find {}", filepath.string());
        GTEST_SKIP();
    }

    auto res = loadGraphAndTensors(filepath);

    EXPECT_EQ(res.graph().compute_type(), data_objects::DataType::FLOAT);
    EXPECT_EQ(res.graph().io_type(), data_objects::DataType::FLOAT);
    EXPECT_EQ(res.graph().intermediate_type(), data_objects::DataType::FLOAT);
    EXPECT_EQ(res.graph().nodes()->size(), 1);
    EXPECT_EQ(res.graph().tensors()->size(), 6);

    std::unordered_map<int64_t, std::vector<int64_t>> expectedAttributes;
    expectedAttributes[0] = {2, 3, 4, 5}; // x
    expectedAttributes[1] = {1, 3, 1, 1}; // mean
    expectedAttributes[2] = {1, 3, 1, 1}; // inv_variance
    expectedAttributes[3] = {1, 3, 1, 1}; // scale
    expectedAttributes[4] = {1, 3, 1, 1}; // bias
    expectedAttributes[5] = {2, 3, 4, 5}; // y

    for(const auto& [uid, tensorPtr] : res.tensorMap)
    {
        EXPECT_EQ(expectedAttributes[uid], tensorPtr->dims());
    }

    auto deviceBuffers = res.deviceBuffers();
    EXPECT_EQ(deviceBuffers.size(), res.tensorMap.size());
    for(auto db : deviceBuffers)
    {
        auto& tensorPtr = res.tensorMap.at(db.uid);
        EXPECT_EQ(tensorPtr->rawDeviceData(), db.ptr);
    }
}

TEST(TestLoadGraphAndTensors, ExtractAndClearOutputTensorData)
{
    std::filesystem::path filepath
        = utilities::getCurrentExecutableDirectory()
          / "../lib/hipdnn_reference_data/BatchnormFwdInference/nchw/fp32/Small.json";

    // TODO: Temporary fix until reference data can be properly installed
    if(!std::filesystem::exists(filepath))
    {
        HIPDNN_LOG_WARN("Could not find {}", filepath.string());
        GTEST_SKIP();
    }

    auto res = loadGraphAndTensors(filepath);

    std::unordered_map<int64_t, std::unique_ptr<ITensor>> savedTensorOutputs;

    // Save tensor data
    for(auto id : res.outputTensorUids)
    {
        const auto& tensor = res.tensorMap.at(id);
        size_t bytesInTensor = tensor->elementSpace() * tensor->elementSize();
        auto& savedTensor = savedTensorOutputs[id]
            = std::unique_ptr<ITensor>(new Tensor<float>(tensor->dims(), tensor->strides()));
        savedTensor->fillWithData(tensor->rawHostData(), bytesInTensor);
    }

    auto outputMap = res.extractAndClearOutputTensorData();

    ASSERT_EQ(outputMap.size(), res.outputTensorUids.size());

    for(auto id : res.outputTensorUids)
    {
        EXPECT_EQ(outputMap.count(id), 1);
        TensorView<float> savedTensorView{*savedTensorOutputs[id]};
        TensorView<float> extractedTensorView{*outputMap.at(id)};

        auto savedIter = savedTensorView.cbegin();
        auto extractedIter = extractedTensorView.cbegin();

        for(; savedIter != savedTensorView.cend() && extractedIter != extractedTensorView.cend();
            savedIter++, extractedIter++)
        {
            EXPECT_EQ(*savedIter, *extractedIter);
        }

        for(auto value : TensorView<float>(*res.tensorMap[id]))
        {
            EXPECT_EQ(value, 0.0);
        }
    }
}
}
