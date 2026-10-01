// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
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
            static_cast<std::streamsize>(values.size() * sizeof(T)));
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
    const ScopedDirectory dir = claimScratchDirectory("scan_empty");

    auto results = scanBundleJsonFiles(dir.path());
    EXPECT_TRUE(results.empty());
}

TEST(TestScanBundleJsonFiles, DiscoversJsonRecursively)
{
    const ScopedDirectory dir = claimScratchDirectory("scan_recursive");
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
    const ScopedDirectory dir = claimScratchDirectory("scan_meta");

    touchFile(dir.path() / "bundle.json");
    touchFile(dir.path() / "meta.json");
    touchFile(dir.path() / "bundle.meta.json"); // compound .meta.json extension

    auto results = scanBundleJsonFiles(dir.path());
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].filename(), "bundle.json");
}

TEST(TestScanBundleJsonFiles, ReturnsSortedPaths)
{
    const ScopedDirectory dir = claimScratchDirectory("scan_sorted");
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

    const std::filesystem::path filepath = getCurrentExecutableDirectory()
                                           / "../lib/integration-test-bundles/quick/"
                                             "BatchnormFwdInference/nchw/fp32/Small/Small.json";

    // TODO: Temporary fix until reference data can be properly installed
    if(!std::filesystem::exists(filepath))
    {
        HIPDNN_SDK_LOG_WARN("Could not find " << filepath.string());
        GTEST_SKIP();
    }

    auto basePath = filepath;
    basePath.replace_extension();
    const std::filesystem::path tensor0Path = basePath.string() + ".tensor0.bin";
    if(!std::filesystem::exists(tensor0Path))
    {
        HIPDNN_SDK_LOG_WARN("Could not find " << tensor0Path.string());
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
    const std::filesystem::path filepath = getCurrentExecutableDirectory()
                                           / "../lib/integration-test-bundles/quick/"
                                             "BatchnormFwdInference/nchw/fp32/Small/Small.json";

    // TODO: Temporary fix until reference data can be properly installed
    if(!std::filesystem::exists(filepath))
    {
        HIPDNN_SDK_LOG_WARN("Could not find " << filepath.string());
        GTEST_SKIP();
    }

    auto basePath = filepath;
    basePath.replace_extension();
    const std::filesystem::path tensor0Path = basePath.string() + ".tensor0.bin";
    if(!std::filesystem::exists(tensor0Path))
    {
        HIPDNN_SDK_LOG_WARN("Could not find " << tensor0Path.string());
        GTEST_SKIP();
    }

    auto res = loadGraphAndTensors(filepath);

    std::unordered_map<int64_t, std::shared_ptr<ITensor>> savedTensorOutputs;

    // Save tensor data
    for(auto id : res.outputTensorUids)
    {
        const auto& tensor = res.tensorMap.at(id);
        const size_t bytesInTensor = tensor->elementSpace() * tensor->elementSize();
        auto& savedTensor = savedTensorOutputs[id]
            = std::shared_ptr<ITensor>(new Tensor<float>(tensor->dims(), tensor->strides()));
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

TEST(TestLoadGraphAndTensors, LoadsRaggedBundle)
{
    const std::filesystem::path filepath
        = getCurrentExecutableDirectory()
          / "../lib/integration-test-bundles/quick/"
            "SdpaFwd/bshd/bf16/hd192_nomask_ragged/Small/Small.json";

    if(!std::filesystem::exists(filepath))
    {
        HIPDNN_SDK_LOG_WARN("Could not find " << filepath.string());
        GTEST_SKIP();
    }

    auto basePath = filepath;
    basePath.replace_extension();
    const std::filesystem::path tensor0Path = basePath.string() + ".tensor0.bin";
    if(!std::filesystem::exists(tensor0Path))
    {
        HIPDNN_SDK_LOG_WARN("Could not find " << tensor0Path.string());
        GTEST_SKIP();
    }

    constexpr int64_t Q_UID = 0;
    constexpr int64_t K_UID = 1;
    constexpr int64_t V_UID = 2;
    constexpr int64_t O_UID = 3;
    constexpr int64_t QO_RAGGED_OFFSET_UID = 10;
    constexpr int64_t KV_RAGGED_OFFSET_UID = 11;

    const std::unordered_map<int64_t, int64_t> expectedRaggedOffsetUids{
        {Q_UID, QO_RAGGED_OFFSET_UID},
        {K_UID, KV_RAGGED_OFFSET_UID},
        {V_UID, KV_RAGGED_OFFSET_UID},
        {O_UID, QO_RAGGED_OFFSET_UID},
    };

    auto res = loadGraphAndTensors(filepath);

    const auto graphWrapper = res.createGraphWrapper();
    const auto& tensorAttributeMap = graphWrapper.getTensorMap();

    ASSERT_EQ(res.tensorMap.size(), 6u);

    const auto batchSize = tensorAttributeMap.at(Q_UID)->dims()->Get(0);
    const std::vector<int64_t> expectedOffsetDims{batchSize + 1, 1, 1, 1};

    for(const auto offsetUid : {QO_RAGGED_OFFSET_UID, KV_RAGGED_OFFSET_UID})
    {
        ASSERT_EQ(res.tensorMap.count(offsetUid), 1u);
        const auto& offsetTensor = res.tensorMap.at(offsetUid);
        EXPECT_EQ(offsetTensor->dims(), expectedOffsetDims);
        EXPECT_FALSE(offsetTensor->raggedIterationInfo().has_value());
    }

    for(const auto& [uid, raggedOffsetUid] : expectedRaggedOffsetUids)
    {
        ASSERT_EQ(res.tensorMap.count(uid), 1u);
        ASSERT_EQ(tensorAttributeMap.count(uid), 1u);

        const auto* attributes = tensorAttributeMap.at(uid);
        ASSERT_TRUE(attributes->ragged_offset_tensor_uid().has_value());
        EXPECT_EQ(attributes->ragged_offset_tensor_uid().value(), raggedOffsetUid);

        const auto& tensor = res.tensorMap.at(uid);
        EXPECT_EQ(tensor->dims(),
                  hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                      attributes->dims()));
        EXPECT_TRUE(tensor->raggedIterationInfo().has_value());

        const auto* raggedTensor
            = dynamic_cast<const RaggedTensorBase<hipdnn_data_sdk::types::bfloat16>*>(tensor.get());
        ASSERT_NE(raggedTensor, nullptr);
        EXPECT_EQ(raggedTensor->raggedOffset(), res.tensorMap.at(raggedOffsetUid).get());
    }
}

namespace
{

constexpr int64_t RAGGED_X_UID = 0;
constexpr int64_t RAGGED_Y_UID = 5;
constexpr int64_t RAGGED_OFFSET_UID = 6;

std::string raggedTensorJson(int64_t uid, int64_t raggedOffsetUid)
{
    return R"({"name": "", "uid": )" + std::to_string(uid)
           + R"(, "strides": [8, 2, 2, 1], "dims": [2, 4, 1, 2], "data_type": "float", )"
             R"("virtual": false, "ragged_offset_tensor_uid": )"
           + std::to_string(raggedOffsetUid) + "}";
}

std::string denseStatsTensorJson(int64_t uid)
{
    return R"({"name": "", "uid": )" + std::to_string(uid)
           + R"(, "strides": [2, 2, 2, 1], "dims": [1, 1, 1, 2], "data_type": "float", )"
             R"("virtual": false})";
}

// The offset tensor is declared last so loading must defer the ragged tensors until it exists.
void writeRaggedBatchnormBundle(const std::filesystem::path& jsonPath,
                                int64_t referencedOffsetUid,
                                const std::vector<int32_t>& offsets,
                                const std::vector<float>& raggedValues)
{
    std::ofstream(jsonPath)
        << R"({"nodes": [{"inputs": {"x_tensor_uid": 0, "mean_tensor_uid": 1, )"
           R"("inv_variance_tensor_uid": 2, "scale_tensor_uid": 3, "bias_tensor_uid": 4}, )"
           R"("outputs": {"y_tensor_uid": 5}, "type": "BatchnormInferenceAttributes", )"
           R"("compute_data_type": "float", "name": ""}], "tensors": [)"
        << raggedTensorJson(RAGGED_X_UID, referencedOffsetUid) << ", " << denseStatsTensorJson(1)
        << ", " << denseStatsTensorJson(2) << ", " << denseStatsTensorJson(3) << ", "
        << denseStatsTensorJson(4) << ", " << raggedTensorJson(RAGGED_Y_UID, referencedOffsetUid)
        << ", "
        << R"({"name": "", "uid": 6, "strides": [1, 1, 1, 1], "dims": [3, 1, 1, 1], )"
           R"("data_type": "int32", "virtual": false}], "io_data_type": "float", )"
           R"("compute_data_type": "float", "intermediate_data_type": "float", "name": ""})";

    auto basePath = jsonPath;
    basePath.replace_extension();
    const auto blobPath = [&](int64_t uid) {
        return std::filesystem::path(basePath.string() + ".tensor" + std::to_string(uid) + ".bin");
    };

    const std::vector<float> stats{1.0f, 2.0f};
    for(const int64_t uid : {1, 2, 3, 4})
    {
        writeVectorToFile(blobPath(uid), stats);
    }
    writeVectorToFile(blobPath(RAGGED_X_UID), raggedValues);
    writeVectorToFile(blobPath(RAGGED_Y_UID), raggedValues);
    writeVectorToFile(blobPath(RAGGED_OFFSET_UID), offsets);
}

} // namespace

TEST(TestLoadGraphAndTensors, LoadsRaggedTensorsDeclaredBeforeTheirOffset)
{
    const ScopedDirectory dir = claimScratchDirectory("load_ragged");
    const auto jsonPath = dir.path() / "Ragged.json";
    const std::vector<float> raggedValues{0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    writeRaggedBatchnormBundle(jsonPath, RAGGED_OFFSET_UID, {0, 4, 6}, raggedValues);

    auto res = loadGraphAndTensors(jsonPath);

    ASSERT_EQ(res.tensorMap.size(), 7u);
    EXPECT_EQ(res.outputTensorUids, std::vector<int64_t>{RAGGED_Y_UID});
    const auto& offsetTensor = res.tensorMap.at(RAGGED_OFFSET_UID);
    EXPECT_FALSE(offsetTensor->raggedIterationInfo().has_value());

    for(const int64_t uid : {RAGGED_X_UID, RAGGED_Y_UID})
    {
        const auto* ragged
            = dynamic_cast<const RaggedTensorBase<float>*>(res.tensorMap.at(uid).get());
        ASSERT_NE(ragged, nullptr) << "uid " << uid;
        EXPECT_EQ(ragged->raggedOffset(), offsetTensor.get()) << "uid " << uid;
        EXPECT_EQ(ragged->dims(), (std::vector<int64_t>{2, 4, 1, 2})) << "uid " << uid;
        EXPECT_EQ(ragged->raggedIterationInfo()->rowOffsets, (std::vector<int64_t>{0, 4, 6}))
            << "uid " << uid;
        ASSERT_EQ(ragged->elementSpace(), raggedValues.size()) << "uid " << uid;

        const auto* data = static_cast<const float*>(res.tensorMap.at(uid)->rawHostData());
        EXPECT_EQ(std::vector<float>(data, data + raggedValues.size()), raggedValues)
            << "uid " << uid;
    }
}

TEST(TestLoadGraphAndTensors, ThrowsWhenRaggedOffsetTensorMissing)
{
    const ScopedDirectory dir = claimScratchDirectory("load_ragged_missing_offset");
    const auto jsonPath = dir.path() / "Ragged.json";
    constexpr int64_t UNDECLARED_UID = 42;
    writeRaggedBatchnormBundle(jsonPath, UNDECLARED_UID, {0, 4, 6}, std::vector<float>(6, 0.0f));

    try
    {
        loadGraphAndTensors(jsonPath);
        FAIL() << "Expected loadGraphAndTensors to throw";
    }
    catch(const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("references missing offset tensor 42"),
                  std::string::npos)
            << e.what();
    }
}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

}
