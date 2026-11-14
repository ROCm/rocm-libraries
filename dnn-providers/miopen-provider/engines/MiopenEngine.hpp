// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "HipdnnMiopenHandle.hpp"
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_plugin_sdk/heuristics/uhd/EnginePredictor.hpp>
#endif

namespace miopen_plugin
{

/// An engine's declared L1 models (RFC 0019 Open Question 7): architecture key (`default`,
/// or a gcnArchName base id such as `gfx942`) to ranking metric to the UUID of the
/// `predict_engine` UHD bound for that pair. The longest matching key wins and `default`
/// answers only when none does (RFC 0019 §8.3).
using L1ModelIds = std::map<std::string, std::map<std::string, std::string>>;

/**
 * @brief MIOpen implementation of the IEngine interface.
 *
 * This class implements the templated IEngine interface using MIOpen-specific types.
 * It manages a collection of plan builders and delegates operations to them.
 */
class MiopenEngine : public hipdnn_plugin_sdk::
                         IEngine<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
{
public:
    /// @param name        This engine's hipDNN name, as its container declared it.
    /// @param l1ModelIds  The models this engine binds. A model whose `score.metric` differs
    ///                    from the metric its id was declared for is refused. Empty or
    ///                    undeployed means no estimate.
    MiopenEngine(int64_t id, std::string name, L1ModelIds l1ModelIds);

    int64_t id() const override;

    bool isApplicable(
        HipdnnMiopenHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(HipdnnMiopenHandle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    /// @brief The engine's calibrated L1 estimate in the requested ranking metric, when a
    /// model for that metric is deployed; no other metric's model answers instead. A
    /// description (@p evaluate false) still names the declared `uhd_id`, so collection
    /// knows what id to promote a first model under.
    hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT
        getPrediction(HipdnnMiopenHandle& handle,
                      const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& config,
                      hipdnnEnginePredictionKind_t kind,
                      bool evaluate) const override;

    size_t getMaxWorkspaceSize(const HipdnnMiopenHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                   engineConfig) const override;

    void initializeExecutionContext(
        const HipdnnMiopenHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        HipdnnMiopenContext& executionContext) const override;

    void addPlanBuilder(
        std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<HipdnnMiopenHandle,
                                                        HipdnnMiopenSettings,
                                                        HipdnnMiopenContext>> planBuilder);

private:
    int64_t _id;
    std::string _name;
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    /// `miopen-provider/<major.minor.patch>/<engine>-<policy revision>/miopen-<x.y.z>`;
    /// must match a model's `trained_against.selector_revision`. Excludes the build's
    /// commit so a model survives a rebuild.
    std::string _selectorRevision;
    L1ModelIds _l1ModelIds;
    /// Empty when no descriptor root is installed: reported as UNAVAILABLE, not an error.
    hipdnn_plugin_sdk::uhd::EngineModelBinding _l1Models;
#endif
    std::vector<std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<HipdnnMiopenHandle,
                                                                HipdnnMiopenSettings,
                                                                HipdnnMiopenContext>>>
        _planBuilders;
};

}
