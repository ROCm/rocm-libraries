// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/ExecuteOnLeavingScope.hpp>
#include <hipdnn_sdk/test_utilities/FileUtilities.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_sdk/utilities/json/LoadGraphAndTensors.hpp>

using namespace hipdnn_sdk::json;

namespace hipdnn_sdk::json
{

TEST(TestFillTensorFromFile, InvalidPath)
{
    hipdnn_sdk::utilities::Tensor<float> tensor({1});
    std::filesystem::path filepath = "./ea0w399059.txt";
    EXPECT_FALSE(std::filesystem::exists(filepath));
    EXPECT_THROW(detail::fillTensorFromFile(tensor, filepath), std::runtime_error);
}

TEST(TestFillTensorFromFile, PathToDirectory)
{
    hipdnn_sdk::utilities::Tensor<float> tensor({1});
    hipdnn_sdk::test_utilities::TempDirectory dir("oijaweorij33");
    EXPECT_THROW(detail::fillTensorFromFile(tensor, dir.path()), std::runtime_error);
}

template <class T>
void writeVectorToFile(std::filesystem::path const& filename, std::vector<T> const& values)
{
    std::ofstream f(filename, std::ios_base::binary);
    ASSERT_TRUE(f.good());

    f.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(int)));
}

TEST(TestFillTensorFromFile, Valid)
{
    std::filesystem::path filename = "SimpleTensor0123.bin";
    test_utilities::ExecuteOnLeavingScope fileDeleter(
        [filename]() { std::filesystem::remove(filename); });

    std::vector<int> values{0, 1, 2, 3};
    writeVectorToFile(filename, values);

    hipdnn_sdk::utilities::Tensor<int> tensor({static_cast<int64_t>(values.size())});
    ASSERT_NO_THROW(detail::fillTensorFromFile(tensor, filename));

    ASSERT_EQ(tensor.memory().count(), values.size());
    for(size_t i = 0; i < values.size(); i++)
    {
        EXPECT_EQ(values[i], tensor.memory().hostData()[i]);
    }
}

TEST(TestLoadGraphAndTensors, Valid)
{
    std::filesystem::path filepath = utilities::getCurrentExecutableDirectory()
                                     / "../lib/reference_data/BatchnormForwardInference.json";

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

    for(const auto& [uid, value] : res.tensorMap)
    {
        auto& tensor = std::get<std::unique_ptr<hipdnn_sdk::utilities::Tensor<float>>>(value);
        EXPECT_EQ(expectedAttributes[uid], tensor->dims());
    }

    auto deviceBuffers = res.deviceBuffers();
    EXPECT_EQ(deviceBuffers.size(), res.tensorMap.size());
    for(auto db : deviceBuffers)
    {
        auto& tensor = std::get<std::unique_ptr<hipdnn_sdk::utilities::Tensor<float>>>(
            res.tensorMap.at(db.uid));
        EXPECT_EQ(tensor->memory().deviceData(), db.ptr);
    }
}
}
