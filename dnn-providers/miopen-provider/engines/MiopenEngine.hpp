// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "HipdnnMiopenHandle.hpp"
#include <hipdnn_plugin_sdk/heuristics/uhd/EnginePredictor.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

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
    /// @param l1ModelIds  Architecture (`default`, or a gcnArchName prefix) to the UUID
    ///                    of the `predict_engine_tflops` UHD this engine binds. See
    ///                    MiopenContainer::getEngineDefinitions(), which is where the
    ///                    declaration lives: MIOpen serves two engines from this one
    ///                    class and they perform differently, so each declares its own
    ///                    id. Empty, or an id nothing deploys, means no L1 estimate.
    MiopenEngine(int64_t id, std::string name, std::map<std::string, std::string> l1ModelIds);

    int64_t id() const override;

    bool isApplicable(
        HipdnnMiopenHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(HipdnnMiopenHandle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    /// @brief The engine's calibrated L1 throughput estimate, when one is deployed.
    ///
    /// RFC 0019 Open Question 7 (RESOLVED): MIOpen ships no UED, so it binds its L1 model
    /// by naming that UHD's UUID in its provider's own engine definition rather than
    /// through a UED role map (§3.1). The document itself claims nothing -- §4.1 keeps a
    /// UHD free of `engine`, `role` and `arch` members -- so the binding identity is
    /// entirely in compiled provider code.
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
    /// `<provider>/<provider version>/<selector>/<library>`, resolved once at
    /// construction: it queries the MIOpen library version, and it is both what a
    /// deployed model must have recorded (RFC 0019 §4.1
    /// `trained_against.selector_revision`) and what the prediction reports.
    std::string _selectorRevision;
    /// Resolved once at construction from the declared ids. Empty when no descriptor root
    /// is installed, which the engine reports as UNAVAILABLE rather than an error.
    hipdnn_plugin_sdk::uhd::EngineModelBinding _l1Models;
    std::vector<std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<HipdnnMiopenHandle,
                                                                HipdnnMiopenSettings,
                                                                HipdnnMiopenContext>>>
        _planBuilders;
};

}
