// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// UniversalGraphMatcher.hpp - RFC 0018 (UMD) graph matcher.
//
// All names below live in namespace hip_kernel_provider_common::umd.
//
// A UniversalGraphMatcher owns exactly one CompiledUmd -- there is a 1:1
// relationship between a matcher and its UMD JSON descriptor -- and matches a
// live flatbuffer graph against it. Matching does double duty (RFC 0018 §4): it
// decides applicability and binds the descriptor's named variables into a
// queryable BindingContext.
//
// match():
//   1. structural -- locate a distinct graph node per pattern node (by opcode),
//      honor the `allow_override_shape` gate, and bind each operand/result name
//      to a graph tensor via the Phase 0 generated UID readers, enforcing that
//      a shared pattern variable resolves to one tensor across nodes (implicit
//      edges); decline on a missing required tensor or an edge conflict;
//   2. criteria -- construct the BindingContext and evaluate the compiled
//      criteria expression; a non-true result declines;
//   3. return a MatchResult carrying the queryable bindings and the descriptor id.
//
// Fail closed: any structural failure or a false criterion yields no match; the
// matcher never matches a graph by default (RFC 0018 §14).

#include "hip_kernel_provider_common/JsonLogic.hpp"
#include "hip_kernel_provider_common/umd/BindingContext.hpp"
#include "hip_kernel_provider_common/umd/UmdCompiler.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hip_kernel_provider_common::umd
{

// The device-property scalars a criterion may read via `$device.<field>`
// (RFC 0018 §4). Populated from the runtime device (hipDeviceProp_t
// sharedMemPerBlock / warpSize) and surfaced by name in the BindingContext.
struct DeviceProperties
{
    std::int64_t ldsSize = 0;
    std::int64_t warpSize = 0;
};

struct MatchResult
{
    bool matched = false;
    std::string umdId;
    BindingContext bindings; // valid when matched; queryable post-match
};

class UniversalGraphMatcher
{
public:
    using IGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;

    // Compile a descriptor into this matcher. There is a 1:1 relationship
    // between a matcher and its UMD; a matcher matches graphs against exactly
    // one descriptor. Throws UmdCompileError on an invalid descriptor
    // (RFC 0018 A.10).
    explicit UniversalGraphMatcher(const nlohmann::json& descriptor)
        : _umd(std::make_unique<CompiledUmd>(UmdCompiler::compile(descriptor)))
    {
    }

    // Adopt an already-compiled descriptor.
    explicit UniversalGraphMatcher(CompiledUmd umd)
        : _umd(std::make_unique<CompiledUmd>(std::move(umd)))
    {
    }

    // The id of the descriptor this matcher matches against.
    const std::string& umdId() const
    {
        return _umd->id;
    }

    const CompiledUmd& descriptor() const
    {
        return *_umd;
    }

    // True when this matcher's descriptor references `$kernel.<field>`. Such a
    // descriptor must be matched with the kernel-metadata overload of match();
    // the two-argument overload throws for it.
    bool referencesKernelMetadata() const
    {
        return _umd->referencesKernelMetadata;
    }

    // Match a graph against this matcher's single descriptor.
    //
    // Throws std::logic_error if the descriptor references `$kernel.<field>`:
    // resolving those needs kernel metadata, so the caller must use the
    // three-argument overload. Use referencesKernelMetadata() to route.
    MatchResult match(const DeviceProperties& device, const IGraph& graph) const
    {
        if(_umd->referencesKernelMetadata)
        {
            throw std::logic_error(
                "UMD '" + _umd->id
                + "' references $kernel metadata; call the kernel-metadata match overload");
        }
        return matchImpl(device, graph, nullptr);
    }

    // Match a graph against this matcher's single descriptor, resolving
    // `$kernel.<field>` references against the supplied kernel metadata (a UKD's
    // KMD values). An unresolved kernel field reads null (fail closed).
    MatchResult match(const DeviceProperties& device,
                      const IGraph& graph,
                      const nlohmann::json& kernelMetadata) const
    {
        return matchImpl(device, graph, &kernelMetadata);
    }

private:
    using INodeWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper;

    MatchResult matchImpl(const DeviceProperties& device,
                          const IGraph& graph,
                          const nlohmann::json* kernel) const
    {
        // Fail closed on an invalid or nodeless graph. nodeCount() tolerates a
        // null `nodes` field; nodeWrappers() would throw on it, so this guard
        // also keeps match() exception-free for a graph with no nodes.
        if(!graph.isValid() || graph.nodeCount() == 0)
        {
            return {};
        }
        std::unordered_map<std::string, jlogic::Value> deviceMap;
        deviceMap.emplace("lds_size", jlogic::Value(device.ldsSize));
        deviceMap.emplace("warp_size", jlogic::Value(device.warpSize));

        return tryMatch(*_umd, graph, graph.nodeWrappers(), deviceMap, kernel);
    }

    MatchResult tryMatch(const CompiledUmd& umd,
                         const IGraph& graph,
                         const std::vector<std::unique_ptr<INodeWrapper>>& wrappers,
                         const std::unordered_map<std::string, jlogic::Value>& deviceMap,
                         const nlohmann::json* kernel) const
    {
        // allow_override_shape gate (RFC 0018 §3): decline override-shape graphs
        // unless the descriptor opts in.
        if(graph.getGraph().is_override_shape_enabled() && !umd.allowOverrideShape)
        {
            return {};
        }

        // Structural: candidate graph nodes per pattern node (same opcode).
        std::vector<std::vector<const INodeWrapper*>> candidates(umd.nodes.size());
        for(std::size_t i = 0; i < umd.nodes.size(); ++i)
        {
            for(const auto& w : wrappers)
            {
                if(w->attributesType() == umd.nodes[i].opSchema->attributesType
                   && w->attributes() != nullptr)
                {
                    candidates[i].push_back(w.get());
                }
            }
            if(candidates[i].empty())
            {
                return {};
            }
        }

        // Assign each pattern node to a distinct graph node (injective), then
        // enforce edges and evaluate criteria. Patterns are tiny, but a hostile
        // graph could present many same-opcode candidates; a step budget caps
        // the search so an adversarial input declines rather than stalls
        // (RFC 0018 §14). Exhausting the budget fails closed.
        std::vector<const INodeWrapper*> assignment(umd.nodes.size(), nullptr);
        std::vector<const INodeWrapper*> used;
        std::size_t steps = 0;
        MatchResult result;
        if(assignNode(
               umd, graph, deviceMap, kernel, candidates, 0, assignment, used, steps, result))
        {
            return result;
        }
        return {};
    }

    // Upper bound on backtracking steps (candidate placements attempted). Real
    // fusion patterns need a handful; the cap only fires on pathological input.
    static constexpr std::size_t K_MAX_SEARCH_STEPS = 100000;

    bool assignNode(const CompiledUmd& umd,
                    const IGraph& graph,
                    const std::unordered_map<std::string, jlogic::Value>& deviceMap,
                    const nlohmann::json* kernel,
                    const std::vector<std::vector<const INodeWrapper*>>& candidates,
                    std::size_t index,
                    std::vector<const INodeWrapper*>& assignment,
                    std::vector<const INodeWrapper*>& used,
                    std::size_t& steps,
                    MatchResult& result) const
    {
        if(index == umd.nodes.size())
        {
            return finalize(umd, graph, deviceMap, kernel, assignment, result);
        }
        for(const INodeWrapper* cand : candidates[index])
        {
            if(++steps > K_MAX_SEARCH_STEPS)
            {
                return false; // search budget exhausted -> fail closed (RFC 0018 §14)
            }
            if(std::find(used.begin(), used.end(), cand) != used.end())
            {
                continue; // a graph node backs at most one pattern node
            }
            assignment[index] = cand;
            used.push_back(cand);
            if(assignNode(umd,
                          graph,
                          deviceMap,
                          kernel,
                          candidates,
                          index + 1,
                          assignment,
                          used,
                          steps,
                          result))
            {
                return true;
            }
            used.pop_back();
            assignment[index] = nullptr;
        }
        return false;
    }

    // Resolve every edge for a full node assignment, enforcing that a shared
    // pattern variable resolves to one tensor across all nodes (implicit edges,
    // RFC 0018 §3/A.3), then evaluate the criteria over the BindingContext.
    static bool finalize(const CompiledUmd& umd,
                         const IGraph& graph,
                         const std::unordered_map<std::string, jlogic::Value>& deviceMap,
                         const nlohmann::json* kernel,
                         const std::vector<const INodeWrapper*>& assignment,
                         MatchResult& result)
    {
        struct VarBinding
        {
            bool present = false;
            std::int64_t uid = 0;
        };
        std::unordered_map<std::string, VarBinding> binds;

        for(std::size_t i = 0; i < umd.nodes.size(); ++i)
        {
            const void* attrs = assignment[i]->attributes();
            for(const EdgeSlot& slot : umd.nodes[i].edges)
            {
                // NOLINTNEXTLINE(misc-const-correctness) - reader out-parameter
                std::int64_t uid = 0;
                const bool present = (*slot.reader)(attrs, uid);
                const auto it = binds.find(slot.tvar);
                if(!present)
                {
                    if(!slot.optional)
                    {
                        return false; // required tensor absent -> decline
                    }
                    if(it == binds.end())
                    {
                        binds.emplace(slot.tvar, VarBinding{false, 0});
                    }
                    else if(it->second.present)
                    {
                        return false; // absent here but present on another edge
                    }
                    continue;
                }
                if(it == binds.end())
                {
                    binds.emplace(slot.tvar, VarBinding{true, uid});
                }
                else if(!it->second.present || it->second.uid != uid)
                {
                    return false; // edge conflict: same variable, different tensor
                }
            }
        }

        BindingContext ctx(&graph, deviceMap);
        if(kernel != nullptr)
        {
            ctx.bindKernelMetadata(*kernel);
        }
        for(std::size_t i = 0; i < umd.nodes.size(); ++i)
        {
            ctx.bindNode(umd.nodes[i].id, umd.nodes[i].opSchema, assignment[i]->attributes());
        }

        const auto& tensorMap = graph.getTensorMap();
        for(const TensorVarSpec& spec : umd.tvars)
        {
            const auto it = binds.find(spec.tvar);
            if(it == binds.end() || !it->second.present)
            {
                ctx.bindTensor(spec.tvar, nullptr, spec.optional, &spec.dimNames);
                continue;
            }
            const auto t = tensorMap.find(it->second.uid);
            if(t == tensorMap.end())
            {
                return false; // dangling UID -> decline (malformed graph)
            }
            ctx.bindTensor(spec.tvar, t->second, spec.optional, &spec.dimNames);
        }

        if(!umd.criteria(ctx).truthy())
        {
            return false;
        }
        result.matched = true;
        result.umdId = umd.id;
        result.bindings = std::move(ctx);
        return true;
    }

    std::unique_ptr<CompiledUmd> _umd;
};

} // namespace hip_kernel_provider_common::umd
