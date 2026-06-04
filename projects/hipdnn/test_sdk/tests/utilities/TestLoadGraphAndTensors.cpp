// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/detail/ScopedExecute.hpp>
#include <hipdnn_test_sdk/utilities/detail/TensorFileUtils.hpp>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_test_sdk::detail;

namespace hipdnn_test_sdk::utilities
{

TEST(TestFillTensorFromFile, InvalidPath)
{
    Tensor<float> tensor({1});
    const std::filesystem::path filepath = "./ea0w399059.txt";
    EXPECT_FALSE(std::filesystem::exists(filepath));
    EXPECT_THROW(fillTensorFromFile(tensor, filepath), std::runtime_error);
}

TEST(TestFillTensorFromFile, PathToDirectory)
{
    Tensor<float> tensor({1});
    const ScopedDirectory dir("oijaweorij33");
    EXPECT_THROW(fillTensorFromFile(tensor, dir.path()), std::runtime_error);
}

namespace
{
template <class T>
void writeVectorToFile(const std::filesystem::path& filename, const std::vector<T>& values)
{
    std::ofstream f(filename, std::ios_base::binary);
    ASSERT_TRUE(f.good());

    f.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(int)));
}
} // namespace

TEST(TestFillTensorFromFile, Valid)
{
    const std::filesystem::path filename = "SimpleTensor0123.bin";
    const ScopedExecute fileDeleter([filename]() { std::filesystem::remove(filename); });

    std::vector<int> values{0, 1, 2, 3};
    writeVectorToFile(filename, values);

    Tensor<int> tensor({static_cast<int64_t>(values.size())});
    ASSERT_NO_THROW(fillTensorFromFile(tensor, filename));

    ASSERT_EQ(tensor.memory().count(), values.size());

    for(size_t i = 0; i < values.size(); i++)
    {
        EXPECT_EQ(values[i], tensor.memory().hostData()[i]);
    }
}

TEST(TestFillTensorFromFile, SizeMismatchSmaller)
{
    const std::filesystem::path filename = "SizeMismatchSmallerTensor.bin";
    const ScopedExecute fileDeleter([filename]() { std::filesystem::remove(filename); });

    // Write 3 ints to file but create a tensor expecting 4
    const std::vector<int> values{0, 1, 2};
    writeVectorToFile(filename, values);

    Tensor<int> tensor({4});
    EXPECT_THROW(fillTensorFromFile(tensor, filename), std::runtime_error);
}

TEST(TestFillTensorFromFile, SizeMismatchLarger)
{
    const std::filesystem::path filename = "SizeMismatchLargerTensor.bin";
    const ScopedExecute fileDeleter([filename]() { std::filesystem::remove(filename); });

    // Write 5 ints to file but create a tensor expecting 4
    const std::vector<int> values{0, 1, 2, 3, 4};
    writeVectorToFile(filename, values);

    Tensor<int> tensor({4});
    EXPECT_THROW(fillTensorFromFile(tensor, filename), std::runtime_error);
}

TEST(TestFillTensorFromFile, NonPackedTensor)
{
    const std::filesystem::path filename = "NonPackedTensor.bin";
    const ScopedExecute fileDeleter([filename]() { std::filesystem::remove(filename); });

    // dims={2,3}, strides={4,1} -> elementCount=6, elementSpace=7
    // File must have exactly 7 ints (elementSpace * elementSize)
    const std::vector<int> values{0, 1, 2, 0, 3, 4, 5};
    writeVectorToFile(filename, values);

    Tensor<int> tensor({2, 3}, {4, 1});
    ASSERT_NO_THROW(fillTensorFromFile(tensor, filename));

    // Verify the raw buffer matches what was written
    const int* data = tensor.memory().hostData();
    ASSERT_NE(data, nullptr);
    for(size_t i = 0; i < values.size(); i++)
    {
        EXPECT_EQ(data[i], values[i]) << "Mismatch at buffer index " << i;
    }
}

// --- Tests for scanBundleJsonFiles ---

namespace
{
void touchFile(const std::filesystem::path& p)
{
    const std::ofstream f(p);
    ASSERT_TRUE(f.good()) << "Failed to create " << p;
}
} // namespace

TEST(TestScanBundleJsonFiles, NonexistentDirectory)
{
    auto results = scanBundleJsonFiles("/tmp/nonexistent_dir_xyzzy_42");
    EXPECT_TRUE(results.empty());
}

TEST(TestScanBundleJsonFiles, EmptyDirectory)
{
    const ScopedDirectory dir(std::filesystem::temp_directory_path() / "test_scan_empty");

    auto results = scanBundleJsonFiles(dir.path());
    EXPECT_TRUE(results.empty());
}

TEST(TestScanBundleJsonFiles, DiscoversJsonRecursively)
{
    const ScopedDirectory dir(std::filesystem::temp_directory_path() / "test_scan_recursive");
    const auto nested = dir.path() / "sub1" / "sub2";
    std::filesystem::create_directories(nested);

    touchFile(dir.path() / "top.json");
    touchFile(nested / "deep.json");
    touchFile(nested / "data.bin"); // non-json, excluded

    auto results = scanBundleJsonFiles(dir.path());
    ASSERT_EQ(results.size(), 2u);

    std::vector<std::string> filenames;
    filenames.reserve(results.size());
    for(const auto& p : results)
    {
        filenames.push_back(p.filename().string());
    }
    EXPECT_NE(std::find(filenames.begin(), filenames.end(), "top.json"), filenames.end());
    EXPECT_NE(std::find(filenames.begin(), filenames.end(), "deep.json"), filenames.end());
}

TEST(TestScanBundleJsonFiles, ExcludesMetaJson)
{
    const ScopedDirectory dir(std::filesystem::temp_directory_path() / "test_scan_meta");

    touchFile(dir.path() / "bundle.json");
    touchFile(dir.path() / "meta.json");

    auto results = scanBundleJsonFiles(dir.path());
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].filename(), "bundle.json");
}

TEST(TestScanBundleJsonFiles, ReturnsSortedPaths)
{
    const ScopedDirectory dir(std::filesystem::temp_directory_path() / "test_scan_sorted");
    const auto subC = dir.path() / "c_dir";
    const auto subA = dir.path() / "a_dir";
    std::filesystem::create_directory(subC);
    std::filesystem::create_directory(subA);

    touchFile(subC / "zebra.json");
    touchFile(subA / "alpha.json");
    touchFile(dir.path() / "middle.json");

    auto results = scanBundleJsonFiles(dir.path());
    ASSERT_EQ(results.size(), 3u);
    EXPECT_TRUE(std::is_sorted(results.begin(), results.end()));
}

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

TEST(TestLoadGraphAndTensors, Valid)
{
    SKIP_IF_NO_DEVICES();

    const std::filesystem::path filepath
        = getCurrentExecutableDirectory()
          / "../lib/golden_reference_data/quick/BatchnormFwdInference/nchw/fp32/Small/Small.json";

    // TODO: Temporary fix until reference data can be properly installed
    if(!std::filesystem::exists(filepath))
    {
        HIPDNN_SDK_LOG_WARN("Could not find " << filepath.string());
        GTEST_SKIP();
    }

    auto res = loadGraphAndTensors(filepath);

    EXPECT_EQ(res.graph().compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(res.graph().io_data_type(), DataType::FLOAT);
    EXPECT_EQ(res.graph().intermediate_data_type(), DataType::FLOAT);
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
    const std::filesystem::path filepath
        = getCurrentExecutableDirectory()
          / "../lib/golden_reference_data/quick/BatchnormFwdInference/nchw/fp32/Small/Small.json";

    // TODO: Temporary fix until reference data can be properly installed
    if(!std::filesystem::exists(filepath))
    {
        HIPDNN_SDK_LOG_WARN("Could not find " << filepath.string());
        GTEST_SKIP();
    }

    auto res = loadGraphAndTensors(filepath);

    std::unordered_map<int64_t, std::unique_ptr<ITensor>> savedTensorOutputs;

    // Save tensor data
    for(auto id : res.outputTensorUids)
    {
        const auto& tensor = res.tensorMap.at(id);
        const size_t bytesInTensor = tensor->elementSpace() * tensor->elementSize();
        auto& savedTensor = savedTensorOutputs[id]
            = std::unique_ptr<ITensor>(new Tensor<float>(tensor->dims(), tensor->strides()));
        savedTensor->fillWithData(tensor->rawHostData(), bytesInTensor);
    }

    auto outputMap = res.extractAndClearOutputTensorData();

    ASSERT_EQ(outputMap.size(), res.outputTensorUids.size());

    for(auto id : res.outputTensorUids)
    {
        EXPECT_EQ(outputMap.count(id), 1);
        const TensorView<float> savedTensorView{*savedTensorOutputs[id]};
        const TensorView<float> extractedTensorView{*outputMap.at(id)};

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

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

}
