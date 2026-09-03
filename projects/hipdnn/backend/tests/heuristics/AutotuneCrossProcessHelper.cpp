// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Standalone helper process for TestAutotuneRankingStore's cross-process cache case.
//
// Invoked as: AutotuneCrossProcessHelper <write|read> <tensor-uid> <dim> <engine-ids-csv>
//
// A second OS process is required because the store is file-backed and cross-process
// persistence is not observable from inside a single process.
//
// Exit codes: 0 on success; 1 usage error; 2 the key could not be derived; 3 read mode
// found no entry (a miss, distinct from a failure to look).

#include "heuristics/config/AutotuneCacheKey.hpp"
#include "heuristics/config/AutotuneRankingStore.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <flatbuffers/flatbuffers.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{
namespace fb = hipdnn_flatbuffers_sdk::data_objects;

/// Mirrors TestAutotuneCacheKey.cpp's builder: only uid's ordinal folds into the key.
std::vector<uint8_t> buildSingleTensorGraphBuffer(int64_t uid, int64_t dim)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims{dim};
    const std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        fb::CreateTensorAttributesDirect(builder, uid, "t", fb::DataType::FLOAT, nullptr, &dims)};
    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             &tensors,
                                             nullptr,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

std::vector<int64_t> parseCsvIds(const std::string& csv)
{
    std::vector<int64_t> ids;
    size_t start = 0;
    while(start <= csv.size())
    {
        const size_t comma = csv.find(',', start);
        const std::string field
            = csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if(!field.empty())
        {
            ids.push_back(std::strtoll(field.c_str(), nullptr, 10));
        }
        if(comma == std::string::npos)
        {
            break;
        }
        start = comma + 1;
    }
    return ids;
}
} // namespace

int main(int argc, char** argv)
{
    if(argc != 5)
    {
        std::fprintf(
            stderr, "usage: %s <write|read> <tensor-uid> <dim> <engine-ids-csv>\n", argv[0]);
        return 1;
    }

    const std::string mode = argv[1];
    const int64_t uid = std::strtoll(argv[2], nullptr, 10);
    const int64_t dim = std::strtoll(argv[3], nullptr, 10);
    const std::vector<int64_t> engineIds = parseCsvIds(argv[4]);

    const std::vector<uint8_t> graphBuffer = buildSingleTensorGraphBuffer(uid, dim);
    const std::vector<uint8_t> deviceBytes{0xDE, 0xAD, 0xBE, 0xEF};

    const hipdnnPluginConstData_t graphView{graphBuffer.data(), graphBuffer.size()};
    const hipdnnPluginConstData_t deviceView{deviceBytes.data(), deviceBytes.size()};

    const auto key = hipdnn_backend::heuristics::config::deriveCacheKey(graphView, deviceView);
    if(!key.has_value())
    {
        std::fprintf(stderr, "helper: could not derive a cache key\n");
        return 2;
    }

    auto& store = hipdnn_backend::heuristics::config::exactCacheStore();
    const std::vector<uint8_t> emptyDeviceKey;

    if(mode == "write")
    {
        store.put(*key, emptyDeviceKey, engineIds, engineIds);
        return 0;
    }

    if(mode == "read")
    {
        const auto entry = store.get(*key, emptyDeviceKey);
        if(!entry.has_value())
        {
            return 3;
        }
        std::string out;
        for(size_t i = 0; i < entry->order.size(); ++i)
        {
            if(i > 0)
            {
                out += ",";
            }
            out += std::to_string(entry->order[i]);
        }
        std::printf("%s", out.c_str());
        return 0;
    }

    std::fprintf(stderr, "helper: unknown mode '%s'\n", mode.c_str());
    return 1;
}
