// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace hipdnn_integration_tests::bundle
{

/// Philox dropout is validated through RNG_DUMP (SdpaSemanticsRegister.md SEM-015): the engine
/// writes its keep mask to the rng_dump output, the reference reads that mask as its explicit
/// dropout mask, and rng_dump itself is not compared (it is the oracle's input, not a result).
struct SdpaRngDump
{
    int64_t uid;
    float dropoutProbability;
};

/// The rng_dump output of the graph's SDPA forward node when it uses Philox dropout, if any.
inline std::optional<SdpaRngDump>
    findSdpaRngDump(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
{
    for(uint32_t i = 0; i < graph.nodeCount(); ++i)
    {
        const auto* sdpa = graph.getNode(i).attributes_as_SdpaAttributes();
        if(sdpa != nullptr && sdpa->rng_dump_tensor_uid().has_value()
           && sdpa->dropout_probability().has_value())
        {
            return SdpaRngDump{sdpa->rng_dump_tensor_uid().value(),
                               sdpa->dropout_probability().value()};
        }
    }
    return std::nullopt;
}

/// Copies the engine's dumped keep mask into the reference's rng_dump buffer (host side; the
/// reference's variant pack is built afterwards, which uploads it if the reference needs it).
inline void copyRngDump(hipdnn_data_sdk::utilities::ITensor& engineDump,
                        hipdnn_data_sdk::utilities::ITensor& referenceDump)
{
    const auto bytes = engineDump.elementSpace() * engineDump.elementSize();
    if(bytes != referenceDump.elementSpace() * referenceDump.elementSize())
    {
        throw std::runtime_error("rng_dump hand-off: engine and reference buffers differ in size");
    }
    referenceDump.fillWithData(engineDump.rawHostData(), bytes);
}

/// Sanity checks on a dumped FLOAT keep mask: every element is 0 or 1, and the keep rate is
/// within a 6-sigma binomial bound of 1 - p (catches an engine that dumps all ones, all zeros or
/// garbage). Returns an empty string when the mask looks valid, else the reason.
inline std::string checkRngDump(hipdnn_data_sdk::utilities::ITensor& dump, float dropoutProbability)
{
    if(dump.elementSize() != sizeof(float))
    {
        return "rng_dump must be FLOAT";
    }
    const auto count = dump.elementSpace();
    const auto* values = static_cast<const float*>(dump.rawHostData());
    size_t kept = 0;
    for(size_t i = 0; i < count; ++i)
    {
        float value = 0.0F;
        std::memcpy(&value, values + i, sizeof(float));
        if(value == 1.0F)
        {
            ++kept;
        }
        else if(value != 0.0F)
        {
            return "rng_dump element " + std::to_string(i) + " is " + std::to_string(value)
                   + ", not 0 or 1";
        }
    }
    if(count == 0)
    {
        return {};
    }
    const double keep = 1.0 - static_cast<double>(dropoutProbability);
    const auto n = static_cast<double>(count);
    const double rate = static_cast<double>(kept) / n;
    const double bound = 6.0 * std::sqrt(keep * (1.0 - keep) / n) + 1.0 / n;
    if(std::abs(rate - keep) > bound)
    {
        return "rng_dump keep rate " + std::to_string(rate) + " is not within "
               + std::to_string(bound) + " of the expected " + std::to_string(keep);
    }
    return {};
}

/// The output uids to compare: every output except a Philox rng_dump.
inline std::vector<int64_t> comparedOutputUids(const std::vector<int64_t>& outputUids,
                                               const std::optional<SdpaRngDump>& dump)
{
    std::vector<int64_t> uids;
    for(const auto uid : outputUids)
    {
        if(!dump.has_value() || uid != dump->uid)
        {
            uids.push_back(uid);
        }
    }
    return uids;
}

} // namespace hipdnn_integration_tests::bundle
