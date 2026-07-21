// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_graph_matcher/GraphView.hpp>
#include <unordered_set>

namespace hipdnn::graph_matcher {

const std::vector<Endpoint> GraphView::kEmptyEndpoints{};

namespace {

// Appends the UIDs a role list contributes, tagging each with its endpoint. A
// UID absent from the tensor map is dropped: a required scalar field left unset
// defaults to 0, which is not a real edge unless the graph actually declares a
// tensor with that UID.
template <class Sink>
void collectRoles(const std::vector<EdgeRole>& roles, const void* attrs, const IGraph& graph,
                  uint32_t nodeIndex, Sink sink) {
    const auto& tensorMap = graph.getTensorMap();
    std::vector<int64_t> uids;
    for (uint32_t roleIndex = 0; roleIndex < roles.size(); ++roleIndex) {
        uids.clear();
        roles[roleIndex].read(attrs, uids);
        for (uint32_t slot = 0; slot < uids.size(); ++slot) {
            const int64_t uid = uids[slot];
            if (tensorMap.find(uid) == tensorMap.end()) {
                continue;
            }
            sink(uid, Endpoint{nodeIndex, roleIndex, slot});
        }
    }
}

}  // namespace

GraphView::GraphView(const IGraph& graph, const OpSchemaRegistry& registry)
    : _graph(graph), _registry(registry) {
    const uint32_t nodeCount = _graph.nodeCount();
    for (uint32_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        const auto& node = _graph.getNodeWrapper(nodeIndex);
        const OpSchema* schema = _registry.forNode(node);
        if (schema == nullptr) {
            continue;  // Unknown op: edges cannot be resolved generically.
        }

        const void* attrs = node.attributes();
        if (attrs == nullptr) {
            continue;
        }

        collectRoles(schema->operands, attrs, _graph, nodeIndex,
                     [&](int64_t uid, Endpoint ep) { _consumers[uid].push_back(ep); });
        collectRoles(schema->results, attrs, _graph, nodeIndex,
                     [&](int64_t uid, Endpoint ep) { _producers.emplace(uid, ep); });
    }
}

const Endpoint* GraphView::producerOf(int64_t uid) const noexcept {
    const auto it = _producers.find(uid);
    return it == _producers.end() ? nullptr : &it->second;
}

const std::vector<Endpoint>& GraphView::consumersOf(int64_t uid) const noexcept {
    const auto it = _consumers.find(uid);
    return it == _consumers.end() ? kEmptyEndpoints : it->second;
}

size_t GraphView::consumerNodeCount(int64_t uid) const noexcept {
    const auto& endpoints = consumersOf(uid);
    std::unordered_set<uint32_t> nodes;
    nodes.reserve(endpoints.size());
    for (const auto& ep : endpoints) {
        nodes.insert(ep.nodeIndex);
    }
    return nodes.size();
}

const TensorAttributes* GraphView::tensor(int64_t uid) const noexcept {
    const auto& tensorMap = _graph.getTensorMap();
    const auto it = tensorMap.find(uid);
    return it == tensorMap.end() ? nullptr : it->second;
}

std::vector<int64_t> GraphView::operandUids(uint32_t nodeIndex) const {
    std::vector<int64_t> uids;
    const auto& node = _graph.getNodeWrapper(nodeIndex);
    const OpSchema* schema = _registry.forNode(node);
    if (schema != nullptr && node.attributes() != nullptr) {
        for (const auto& role : schema->operands) {
            role.read(node.attributes(), uids);
        }
    }
    return uids;
}

std::vector<int64_t> GraphView::resultUids(uint32_t nodeIndex) const {
    std::vector<int64_t> uids;
    const auto& node = _graph.getNodeWrapper(nodeIndex);
    const OpSchema* schema = _registry.forNode(node);
    if (schema != nullptr && node.attributes() != nullptr) {
        for (const auto& role : schema->results) {
            role.read(node.attributes(), uids);
        }
    }
    return uids;
}

const OpSchema* GraphView::schemaOf(uint32_t nodeIndex) const noexcept {
    return _registry.forNode(_graph.getNodeWrapper(nodeIndex));
}

std::string_view GraphView::opcodeOf(uint32_t nodeIndex) const {
    const OpSchema* schema = schemaOf(nodeIndex);
    return schema == nullptr ? std::string_view{} : schema->opcode;
}

std::vector<int64_t> GraphView::roleUids(uint32_t nodeIndex, bool result,
                                         uint32_t roleIndex) const {
    std::vector<int64_t> uids;
    const auto& node = _graph.getNodeWrapper(nodeIndex);
    const OpSchema* schema = _registry.forNode(node);
    if (schema == nullptr || node.attributes() == nullptr) {
        return uids;
    }
    const auto& roles = result ? schema->results : schema->operands;
    if (roleIndex < roles.size()) {
        roles[roleIndex].read(node.attributes(), uids);
    }
    return uids;
}

std::optional<int64_t> GraphView::attrInt(uint32_t nodeIndex, std::string_view attrName) const {
    const auto& node = _graph.getNodeWrapper(nodeIndex);
    const OpSchema* schema = _registry.forNode(node);
    if (schema == nullptr || node.attributes() == nullptr) {
        return std::nullopt;
    }
    const AttrAccessor* attr = schema->findAttr(attrName);
    return attr == nullptr ? std::nullopt : attr->read(node.attributes());
}

}  // namespace hipdnn::graph_matcher
