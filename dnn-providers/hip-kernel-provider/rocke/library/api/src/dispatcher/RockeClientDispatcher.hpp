// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>

#include "dispatcher/AotCatalog.hpp"
#include "dispatcher/AotInstance.hpp"
#include "dispatcher/SdpaProblem.hpp"

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
{
class IGraph;
}

namespace rocke_client
{
struct RockeClientHandle;
}

namespace rocke_client::dispatcher
{

// Owns the AOT catalog and performs kernel selection: graph -> normalized form
// -> filter to AOT-available instances -> winning instance.
//
// PHASE 1: the catalog is empty, so selection always yields nothing and the
// engine declines. The plan-construction step (winning instance -> kernel_id ->
// launch metadata -> load pre-built kernel from the kpack) is a fast-follow and
// lives in the engine's initializeExecutionContext seam, not here.
class RockeClientDispatcher
{
public:
    explicit RockeClientDispatcher(AotCatalog catalog);

    // Pure selection over an already-normalized problem (no HIP): filters catalog
    // candidates for (problem.op, problem.arch) by satisfies() and returns the
    // winner. Returns an owned copy by design, so the result stays valid past the
    // borrowed candidate view (see AotInstance).
    //
    // Tie-break: deterministic stable catalog order, first match.
    // TODO(heuristics): when >1 instances match AND a trained per-arch FMHA
    // model exists, break ties with the model score instead of catalog order.
    std::optional<AotInstance> select(const SdpaProblem& problem) const;

    // Full path used by the engine: translate the graph, detect the device arch
    // from the handle's stream, then select. Never throws.
    std::optional<AotInstance> selectInstance(
        const RockeClientHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) const noexcept;

    // Test/explicit-arch seam: run translate + select for a caller-supplied arch,
    // bypassing HIP device detection (which returns "" on host CI without a GPU).
    // Lets the full graph -> problem -> select path be exercised deterministically.
    // Never throws.
    std::optional<AotInstance> selectForArch(
        const std::string& arch,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) const noexcept;

    // Graph accept/reject: whether any AOT instance can serve this graph.
    bool isApplicable(
        const RockeClientHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) const noexcept;

    const AotCatalog& catalog() const
    {
        return _catalog;
    }

private:
    AotCatalog _catalog;
};

} // namespace rocke_client::dispatcher
