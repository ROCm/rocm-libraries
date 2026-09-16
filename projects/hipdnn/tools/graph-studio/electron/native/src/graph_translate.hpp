// Studio graph JSON → hipDNN frontend graph.
//
// engine.cpp owns the N-API surface and plan lifecycle; this unit owns the
// operator-by-operator mapping onto the frontend builder and the string↔enum
// tables the Studio parameters use.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <nlohmann/json.hpp>

namespace studio
{

using Graph = hipdnn_frontend::graph::Graph;
using TensorPtr = std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>;

// Thrown for graph-translation problems (bad/missing connections, unsupported
// ops). Carries an EngineErrorCode string so the renderer shows a precise cause.
struct BuildInputError
{
    std::string code;
    std::string message;
};

// Graph-wide I/O data type, taken from the first Input node.
hipdnn_frontend::DataType pickIoDtype(const nlohmann::json& root);

std::size_t dtypeSize(hipdnn_frontend::DataType dt);

// Elements a tensor spans in memory: the highest addressable offset plus one.
// Equals the dim product when packed and stays correct when strides pad, so it
// is what device buffers must be sized by. Falls back to the dim product when
// no usable stride list is available.
int64_t storageElements(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides);

// Construct the frontend graph from Studio JSON. outputByPort maps
// "node:port" → the tensor that port produces. Throws BuildInputError.
void translateGraph(const nlohmann::json& root,
                    Graph& g,
                    hipdnn_frontend::DataType ioDtype,
                    std::unordered_map<std::string, TensorPtr>& outputByPort);

} // namespace studio
