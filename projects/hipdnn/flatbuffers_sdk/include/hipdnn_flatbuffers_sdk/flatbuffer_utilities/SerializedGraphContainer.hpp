// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include <hipdnn_flatbuffers_sdk/data_objects/serialized_graph_and_plan_generated.h>

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
{

/*
 * Single home for all "SerializedGraphAndPlan" (file identifier "HDGP") container
 * format knowledge. The helpers here accept EITHER a bare serialized Graph/plan
 * buffer OR an HDGP container that wraps a self-contained graph blob and an
 * optional self-contained plan blob.
 *
 * Container detection is by the FlatBuffers file identifier: the container
 * carries "HDGP" while a bare serialized Graph carries none, so the presence of
 * the identifier unambiguously distinguishes the two formats.
 *
 * The returned SerializedBlobView objects are non-owning and alias into the
 * caller-supplied buffer. They are only valid while that buffer outlives the
 * view; no inner bytes are copied.
 */

// Non-owning view over a serialized blob. Aliases the caller's buffer and is
// only valid while that buffer outlives the view.
struct SerializedBlobView
{
    const uint8_t* data = nullptr;
    size_t size = 0;
};

// Builds an "HDGP" SerializedGraphAndPlan container that frames an
// already-serialized graph blob and an optional already-serialized plan blob.
// This is the single write-side home for the container format and the mirror of
// the extract* read helpers below.
//
// Both embedded blobs are force-aligned to 8 bytes. graph_blob and plan_blob are
// themselves self-contained FlatBuffers buffers (the inner Graph, and the
// SerializedExecutionPlan wrapper) whose roots carry int64 fields and are parsed
// IN PLACE on deserialize (GetRoot<Graph> / GetSerializedExecutionPlan over the
// peeled view). A plain [ubyte] vector is only 4-byte aligned, so without this
// the int64 reads off the peeled offset would be misaligned-access UB (silent on
// x86, a trap under -fsanitize=alignment and on strict-alignment ISAs). We build
// the vectors here and pass their offsets to the low-level
// CreateSerializedGraphAndPlan(), which receives already-built vectors and so
// cannot apply the schema's force_align (only the generated *Direct/object-API
// builders do, and we do not use them); the alignment must therefore be forced
// explicitly, immediately before each CreateVector. (The opaque provider payload
// nested inside the plan is copied to a fresh heap buffer before hand-off, so it
// needs no container-level alignment.)
//
// graphData must be non-null with graphSize > 0 (graph_blob is mandatory).
// planData may be null / planSize 0 for a graph-only container.
inline flatbuffers::DetachedBuffer buildGraphAndPlanContainer(const void* graphData,
                                                              size_t graphSize,
                                                              const void* planData,
                                                              size_t planSize)
{
    flatbuffers::FlatBufferBuilder builder;

    builder.ForceVectorAlignment(graphSize, sizeof(uint8_t), 8);
    auto graphVec = builder.CreateVector(static_cast<const uint8_t*>(graphData), graphSize);

    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> planVec = 0;
    if(planData != nullptr && planSize > 0)
    {
        builder.ForceVectorAlignment(planSize, sizeof(uint8_t), 8);
        planVec = builder.CreateVector(static_cast<const uint8_t*>(planData), planSize);
    }

    auto root = data_objects::CreateSerializedGraphAndPlan(builder, graphVec, planVec);
    data_objects::FinishSerializedGraphAndPlanBuffer(builder, root);
    return builder.Release();
}

// Returns true when @p buffer is large enough to hold a container header and
// carries the "HDGP" file identifier. Does not run the full verifier; pair with
// verifyGraphAndPlanContainer() before reading fields. Never throws.
inline bool isGraphAndPlanContainer(const void* buffer, size_t size) noexcept
{
    // SerializedGraphAndPlanBufferHasIdentifier() reads the 4-byte file
    // identifier at bytes [4, 8) without bounds-checking, so the buffer must be
    // at least 8 bytes (sizeof(flatbuffers::uoffset_t) + the 4-byte identifier)
    // before we may inspect it.
    if(buffer == nullptr || size < 8)
    {
        return false;
    }

    return data_objects::SerializedGraphAndPlanBufferHasIdentifier(
        static_cast<const uint8_t*>(buffer));
}

// Runs the FlatBuffers verifier over @p buffer and returns the typed container
// root. The returned pointer aliases into @p buffer and is only valid for the
// lifetime of that buffer; no inner bytes are copied. Throws
// std::invalid_argument on verification failure.
inline const data_objects::SerializedGraphAndPlan* verifyGraphAndPlanContainer(const void* buffer,
                                                                               size_t size)
{
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
    if(!data_objects::VerifySerializedGraphAndPlanBuffer(verifier))
    {
        throw std::invalid_argument(
            "Invalid buffer: unable to verify the serialized graph-and-plan container.");
    }

    const auto* container = data_objects::GetSerializedGraphAndPlan(buffer);
    if(container == nullptr)
    {
        throw std::invalid_argument(
            "Invalid buffer: unable to read the serialized graph-and-plan container root.");
    }

    return container;
}

// Extracts a view over the graph blob. When @p buffer is an HDGP container, the
// container is verified and a view over its graph_blob is returned. The graph
// blob is mandatory per the schema, so this throws std::invalid_argument if the
// graph blob is missing or empty. Otherwise the buffer is treated as a bare
// serialized Graph and returned unchanged.
//
// Note the deliberate asymmetry with extractPlanBlob(): graph_blob is mandatory
// (throws on null/empty) while plan_blob is optional (yields an empty view).
inline SerializedBlobView extractGraphBlob(const void* buffer, size_t size)
{
    // The "HDGP" identifier is checked twice on purpose: isGraphAndPlanContainer()
    // is a cheap header-only detect, and verifyGraphAndPlanContainer() then runs
    // the full verifier. The redundancy is intentional defense-in-depth.
    if(isGraphAndPlanContainer(buffer, size))
    {
        const auto* container = verifyGraphAndPlanContainer(buffer, size);
        if(container->graph_blob() == nullptr || container->graph_blob()->empty())
        {
            throw std::invalid_argument(
                "Invalid buffer: the serialized graph-and-plan container has no graph blob.");
        }
        return {container->graph_blob()->data(), container->graph_blob()->size()};
    }

    return {static_cast<const uint8_t*>(buffer), size};
}

// Extracts a view over the plan blob. When @p buffer is an HDGP container, the
// container is verified and a view over its plan_blob is returned. The plan blob
// is OPTIONAL per the schema (absent/empty => graph-only), so a null or empty
// plan_blob yields an empty SerializedBlobView{} (data == nullptr, size == 0);
// callers should check .data/.size rather than relying on a throw. Only a
// malformed container that fails verification throws std::invalid_argument (via
// verifyGraphAndPlanContainer). Otherwise the buffer is treated as a bare
// serialized plan and returned unchanged.
//
// Note the deliberate asymmetry with extractGraphBlob(): graph_blob is mandatory
// (throws on null/empty) while plan_blob is optional (yields an empty view).
inline SerializedBlobView extractPlanBlob(const void* buffer, size_t size)
{
    // The "HDGP" identifier is checked twice on purpose: isGraphAndPlanContainer()
    // is a cheap header-only detect, and verifyGraphAndPlanContainer() then runs
    // the full verifier. The redundancy is intentional defense-in-depth.
    if(isGraphAndPlanContainer(buffer, size))
    {
        const auto* container = verifyGraphAndPlanContainer(buffer, size);
        if(container->plan_blob() == nullptr || container->plan_blob()->empty())
        {
            // Plan is optional per schema: graph-only containers are valid, so
            // return an empty view instead of throwing.
            return {};
        }
        return {container->plan_blob()->data(), container->plan_blob()->size()};
    }

    return {static_cast<const uint8_t*>(buffer), size};
}

} // namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
