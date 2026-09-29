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
    /// @param l1ModelIds  Registered ranking metric to the UUID of the `predict_engine`
    ///                    UHD this engine binds for it. Every id binds under `default`:
    ///                    one model may serve several architectures, and its artifact's
    ///                    `training_arches` decides which per query. A deployed model
    ///                    whose `score.metric` differs from the metric its id is declared
    ///                    for is refused. See MiopenContainer::getEngineDefinitions(),
    ///                    which is where the declaration lives: MIOpen serves two engines
    ///                    from this one class and they perform differently, so each
    ///                    declares its own ids. Empty, or an id nothing deploys, means no
    ///                    L1 estimate.
    MiopenEngine(int64_t id, std::string name, std::map<std::string, std::string> l1ModelIds);

    int64_t id() const override;

    bool isApplicable(
        HipdnnMiopenHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(HipdnnMiopenHandle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    /// @brief The engine's calibrated L1 estimate in the requested ranking metric, when a
    /// model for that metric is deployed. No other metric's model ever answers instead.
    ///
    /// RFC 0019 Open Question 7 (RESOLVED): MIOpen ships no UED, so it binds its L1 models
    /// by naming those UHDs' UUIDs in its provider's own engine definition rather than
    /// through a UED role map (§3.1). The document itself claims nothing -- §4.1 keeps a
    /// UHD free of `engine`, `role` and `arch` members -- so the binding identity is
    /// entirely in compiled provider code. A description (@p evaluate false) names the
    /// id declared for the requested metric as the binding's `uhd_id` even when nothing
    /// is deployed, so collection knows what to promote a first model under.
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
    /// `miopen-provider/<major.minor.patch>/<engine>-<policy revision>/miopen-<x.y.z>`,
    /// resolved once at construction: it queries the MIOpen library version, and it is
    /// both what a deployed model must have recorded (RFC 0019 §4.1
    /// `trained_against.selector_revision`) and what the prediction reports. Never the
    /// build's commit, so a model survives a rebuild at another commit.
    std::string _selectorRevision;
    /// Metric to declared UHD id, as the container passed it.
    std::map<std::string, std::string> _l1ModelIds;
    /// Resolved once at construction from the declared ids. Empty when no descriptor root
    /// is installed, which the engine reports as UNAVAILABLE rather than an error.
    hipdnn_plugin_sdk::uhd::EngineModelBinding _l1Models;
#endif
    std::vector<std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<HipdnnMiopenHandle,
                                                                HipdnnMiopenSettings,
                                                                HipdnnMiopenContext>>>
        _planBuilders;
};

}
