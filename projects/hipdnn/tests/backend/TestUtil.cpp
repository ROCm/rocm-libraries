// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <filesystem>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <span>
#include <stdexcept>

namespace fs = std::filesystem;

namespace test_util
{

void createTestHandle(hipdnnHandle_t* handle)
{
    ASSERT_EQ(hipdnnCreate(handle), HIPDNN_STATUS_SUCCESS);
}

void createTestGraph(hipdnnBackendDescriptor_t* descriptor, hipdnnHandle_t handle)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto graph
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "Test GRAPH!",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graph);
    flatbuffers::DetachedBuffer serializedGraph = builder.Release();

    ASSERT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(
                  descriptor, serializedGraph.data(), serializedGraph.size()),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(
                  *descriptor, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle),
              HIPDNN_STATUS_SUCCESS);
}

void populateTestEngine(hipdnnBackendDescriptor_t engine,
                        hipdnnBackendDescriptor_t* graph,
                        hipdnnHandle_t handle,
                        int64_t gidx,
                        bool finalize)
{
    if(*graph == nullptr)
    {
        createTestGraph(graph, handle);
    }

    ASSERT_EQ(hipdnnBackendFinalize(*graph), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            engine, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, graph),
        HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx),
              HIPDNN_STATUS_SUCCESS);

    if(finalize)
    {
        ASSERT_EQ(hipdnnBackendFinalize(engine), HIPDNN_STATUS_SUCCESS);
    }
}

void createTestEngine(hipdnnBackendDescriptor_t* engine,
                      hipdnnBackendDescriptor_t* graph,
                      hipdnnHandle_t handle,
                      int64_t gidx,
                      bool finalize)
{
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, engine),
              HIPDNN_STATUS_SUCCESS);
    populateTestEngine(*engine, graph, handle, gidx, finalize);
}

void populateTestEngineConfig(hipdnnBackendDescriptor_t* engineConfig,
                              hipdnnBackendDescriptor_t* engine,
                              hipdnnBackendDescriptor_t* graph,
                              hipdnnHandle_t handle,
                              int64_t gidx,
                              bool finalize)
{
    if(*engine == nullptr)
    {
        createTestEngine(engine, graph, handle, gidx, true);
    }

    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            *engineConfig, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, engine),
        HIPDNN_STATUS_SUCCESS);

    if(finalize)
    {
        ASSERT_EQ(hipdnnBackendFinalize(*engineConfig), HIPDNN_STATUS_SUCCESS);
    }
}

void createTestEngineConfig(hipdnnBackendDescriptor_t* engineConfig,
                            hipdnnBackendDescriptor_t* engine,
                            hipdnnBackendDescriptor_t* graph,
                            hipdnnHandle_t handle,
                            int64_t gidx,
                            bool finalize)
{
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, engineConfig),
              HIPDNN_STATUS_SUCCESS);
    populateTestEngineConfig(engineConfig, engine, graph, handle, gidx, finalize);
}

void populateTestExecutionPlan(hipdnnBackendDescriptor_t* executionPlan,
                               hipdnnBackendDescriptor_t* engineConfig,
                               hipdnnBackendDescriptor_t* engine,
                               hipdnnBackendDescriptor_t* graph,
                               hipdnnHandle_t handle,
                               int64_t gidx,
                               bool finalize)
{
    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            *executionPlan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle),
        HIPDNN_STATUS_SUCCESS);

    if(*engineConfig == nullptr)
    {
        createTestEngineConfig(engineConfig, engine, graph, handle, gidx, true);
    }

    ASSERT_EQ(hipdnnBackendSetAttribute(*executionPlan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        engineConfig),
              HIPDNN_STATUS_SUCCESS);

    if(finalize)
    {
        ASSERT_EQ(hipdnnBackendFinalize(*executionPlan), HIPDNN_STATUS_SUCCESS);
    }
}

void* allocateTensorMemory([[maybe_unused]] const int64_t* dims,
                           [[maybe_unused]] size_t dimsCount,
                           [[maybe_unused]] hipdnnBackendAttributeType_t dataType,
                           [[maybe_unused]] bool initialize)
{
    // TODO: Implement memory allocation logic based on the data type and dimensions
    // For now, just return a dummy pointer
    void* memory = malloc(0);
    return memory;
}

void freeTensorMemory(void* dataPtr)
{
    if(dataPtr != nullptr)
    {
        free(dataPtr);
    }
}

void setTensorMappingsInVariantPack(hipdnnBackendDescriptor_t variantPack,
                                    const std::vector<int64_t>& tensorIds,
                                    const std::vector<void*>& dataPtrs)
{
    ASSERT_EQ(hipdnnBackendSetAttribute(variantPack,
                                        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(tensorIds.size()),
                                        tensorIds.data()),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(variantPack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        static_cast<int64_t>(dataPtrs.size()),
                                        dataPtrs.data()),
              HIPDNN_STATUS_SUCCESS);
}

void setWorkspaceInVariantPack(hipdnnBackendDescriptor_t variantPack, void* workspace)
{
    if(workspace != nullptr)
    {
        ASSERT_EQ(hipdnnBackendSetAttribute(variantPack,
                                            HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                            HIPDNN_TYPE_VOID_PTR,
                                            1,
                                            &workspace),
                  HIPDNN_STATUS_SUCCESS);
    }
}

void finalizeVariantPack(hipdnnBackendDescriptor_t variantPack)
{
    ASSERT_EQ(hipdnnBackendFinalize(variantPack), HIPDNN_STATUS_SUCCESS);
}

void extractTensorMappings(const std::unordered_map<int64_t, void*>& dataPtrMappings,
                           std::vector<int64_t>& tensorIds,
                           std::vector<void*>& dataPtrs)
{
    for(const auto& [id, dataPtr] : dataPtrMappings)
    {
        ASSERT_NE(dataPtr, nullptr);
        tensorIds.push_back(id);
        dataPtrs.push_back(dataPtr);
    }
    ASSERT_EQ(tensorIds.size(), dataPtrs.size());
    ASSERT_FALSE(tensorIds.empty());
    ASSERT_FALSE(dataPtrs.empty());
}

void populateVariantPackWithMappings(hipdnnBackendDescriptor_t variantPack,
                                     const std::unordered_map<int64_t, void*>& dataPtrMappings,
                                     void* workspace)
{
    std::vector<int64_t> tensorIds;
    std::vector<void*> dataPtrs;

    extractTensorMappings(dataPtrMappings, tensorIds, dataPtrs);
    setTensorMappingsInVariantPack(variantPack, tensorIds, dataPtrs);
    setWorkspaceInVariantPack(variantPack, workspace);
    finalizeVariantPack(variantPack);
}

void createAndInitializeBackendDescriptor(hipdnnBackendDescriptor_t* backendDescriptor,
                                          const flatbuffers::DetachedBuffer& serializedGraph,
                                          hipdnnHandle_t handle)
{
    ASSERT_EQ(*backendDescriptor, nullptr);

    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        backendDescriptor, serializedGraph.data(), serializedGraph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            *backendDescriptor, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle),
        HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendFinalize(*backendDescriptor);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

flatbuffers::FlatBufferBuilder createAndPopulateBatchnormNode()
{
    return hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
}

void extractTensorInfoFromGraph(const flatbuffers::DetachedBuffer& serializedGraph,
                                std::unordered_map<int64_t, std::string>& uidToNameMap,
                                std::unordered_map<std::string, int64_t>& nameToUidMap,
                                std::unordered_map<int64_t, std::vector<int64_t>>& uidToDimsMap)
{
    uidToNameMap.clear();
    nameToUidMap.clear();
    uidToDimsMap.clear();

    auto deserializedGraph = hipdnn_sdk::data_objects::UnPackGraph(serializedGraph.data());
    ASSERT_NE(deserializedGraph, nullptr);

    // Extract all tensor information from the deserialized graph
    for(const auto& tensor : deserializedGraph->tensors)
    {
        int64_t uid = tensor->uid;
        std::string name = tensor->name;

        uidToNameMap[uid] = name;
        nameToUidMap[name] = uid;

        if(!tensor->dims.empty())
        {
            uidToDimsMap[uid] = tensor->dims;
        }
    }

    ASSERT_FALSE(uidToNameMap.empty());
}

std::vector<std::string> getLoadedPlugins(hipdnnHandle_t handle)
{
    size_t numPlugins = 0;
    size_t maxPathLength = 0;
    auto status
        = hipdnnGetLoadedEnginePluginPaths_ext(handle, &numPlugins, nullptr, &maxPathLength);

    if(status != HIPDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to get loaded plugin paths");
    }

    if(numPlugins == 0)
    {
        return {};
    }

    std::vector<std::vector<char>> pathBuffers(numPlugins, std::vector<char>(maxPathLength));
    std::vector<char*> pluginPathsC(numPlugins);
    for(size_t i = 0; i < numPlugins; ++i)
    {
        pluginPathsC[i] = pathBuffers[i].data();
    }

    status = hipdnnGetLoadedEnginePluginPaths_ext(
        handle, &numPlugins, pluginPathsC.data(), &maxPathLength);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to get loaded plugin paths");
    }

    std::vector<std::string> pluginPaths;
    pluginPaths.reserve(numPlugins);
    for(size_t i = 0; i < numPlugins; ++i)
    {
        pluginPaths.emplace_back(pluginPathsC[i]);
    }
    return pluginPaths;
}

static inline bool stemEq(const fs::path& pathFileName, const fs::path& suffixFileName)
{
    using hipdnn_sdk::utilities::pathCompEq;
    if(pathCompEq(pathFileName, suffixFileName))
    {
        return true;
    }

    // The full path should have an extension, but the suffix may not
    return pathFileName.has_extension() && !suffixFileName.has_extension()
           && pathCompEq(pathFileName.stem(), suffixFileName);
}

static bool isPluginLoadedByRelativePathInternal(const fs::path& fullPath, const fs::path& suffix)
{
    using hipdnn_sdk::utilities::pathCompEq;

    fs::path suffixNorm = suffix.lexically_normal();
    fs::path fullPathNorm = fullPath.lexically_normal();

    if(suffixNorm.empty())
    {
        return false;
    }

    for(const auto& c : suffixNorm)
    {
        if(c == "..")
        {
            return false; // Is unresolvable
        }
    }

    std::vector<fs::path> suffixComps;
    std::vector<fs::path> fullComps;
    for(const auto& c : suffixNorm)
    {
        suffixComps.push_back(c);
    }
    for(const auto& c : fullPathNorm)
    {
        fullComps.push_back(c);
    }
    if(suffixComps.size() > fullComps.size())
    {
        return false;
    }

    auto si = suffixComps.rbegin();
    auto fi = fullComps.rbegin();

    if(!stemEq(*si, *fi))
    {
        return false;
    }
    ++si;
    ++fi;

    for(; si != suffixComps.rend(); ++si, ++fi)
    {
        if(!pathCompEq(*si, *fi))
        {
            return false;
        }
    }
    return true;
}

bool isPluginLoaded(const std::vector<std::string>& loadedPlugins, const std::string& pluginName)
{
    return std::any_of(
        loadedPlugins.begin(), loadedPlugins.end(), [&](const std::string& loadedPathStr) {
            try
            {
                return fs::canonical(fs::path(loadedPathStr))
                       == fs::canonical(fs::path(pluginName));
            }
            catch(const fs::filesystem_error&)
            {
                return false;
            }
        });
}

bool isPluginLoadedByRelativePath(const std::vector<std::string>& loadedPlugins,
                                  const std::string& relativePath)
{
    // We cannot resolve a relative path to the hipdnn_backend module externally
    // of the backend. We can retrieve the absolute loaded path, but our
    // comparison to the originally specified path is limited. In particular, we
    // can check that the loaded path is suffixed with the resolved portion of
    // the relative path.
    const fs::path suffix{relativePath};
    for(const auto& s : loadedPlugins)
    {
        if(isPluginLoadedByRelativePathInternal(fs::path{s}, suffix))
        {
            return true;
        }
    }
    return false;
}

} // namespace test_util
