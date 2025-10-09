// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/ExecuteOnLeavingScope.hpp>
#include <hipdnn_sdk/test_utilities/TempDirectory.hpp>
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

struct TestTensorFile
{
    std::filesystem::path filename;
    TestTensorFile(std::filesystem::path inFilename)
        : filename(std::move(inFilename))
    {
        if(std::filesystem::exists(filename))
        {
            throw std::runtime_error("File already exists");
        }

        std::ofstream f(filename, std::ios_base::binary);
        if(!f.good())
        {
            throw std::runtime_error("Could not open file");
        }
        std::vector<int> values = {0, 1, 2, 3};
        f.write(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(int)));
    }

    ~TestTensorFile()
    {
        if(std::filesystem::exists(filename))
        {
            std::filesystem::remove(filename);
        }
    }
};

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
    std::filesystem::path filepath
        = utilities::getBinaryDir() / "../lib/reference_data/BatchnormForwardInference.json";

    auto res = loadGraphAndTensors(filepath);

    EXPECT_EQ(res.graph().compute_type(), data_objects::DataType::FLOAT);
    EXPECT_EQ(res.graph().io_type(), data_objects::DataType::FLOAT);
    EXPECT_EQ(res.graph().intermediate_type(), data_objects::DataType::FLOAT);
    EXPECT_EQ(res.graph().nodes()->size(), 1);
    EXPECT_EQ(res.graph().tensors()->size(), 6);

    using ExpectedType = std::unordered_map<int64_t, std::unique_ptr<utilities::Tensor<float>>>;

    ASSERT_TRUE(std::holds_alternative<ExpectedType>(res.tensorMap));
    const ExpectedType& tensorMap = std::get<ExpectedType>(res.tensorMap);

    std::unordered_map<int64_t, std::vector<int64_t>> expectedAttributes;
    expectedAttributes[0] = {2, 3, 4, 5}; // x
    expectedAttributes[1] = {1, 3, 1, 1}; // mean
    expectedAttributes[2] = {1, 3, 1, 1}; // inv_variance
    expectedAttributes[3] = {1, 3, 1, 1}; // scale
    expectedAttributes[4] = {1, 3, 1, 1}; // bias
    expectedAttributes[5] = {2, 3, 4, 5}; // y

    for(const auto& [uid, value] : tensorMap)
    {
        EXPECT_EQ(expectedAttributes[uid], value->dims());
    }

    auto deviceBuffers = res.deviceBuffers();
    EXPECT_EQ(deviceBuffers.size(), tensorMap.size());
    for(auto db : deviceBuffers)
    {
        EXPECT_EQ(tensorMap.at(db.uid)->memory().deviceData(), db.ptr);
    }
}
}
