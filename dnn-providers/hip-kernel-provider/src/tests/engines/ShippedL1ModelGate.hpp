// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <iterator>
#include <string>
#include <string_view>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include <gtest/gtest.h>

namespace hip_kernel_provider::shipped_model_gate
{

/// One shipped `.uhd.json`, as cmake/HkpShippedModels.cmake read it at configure time:
/// its path relative to the provider directory, its `id`, and the
/// `trained_against.selector_revision` it records ("" when it records none).
struct ShippedModel
{
    const char* path;
    const char* id;
    const char* revision;
};

/// Whether `path` ends with `suffix`.
inline bool endsWith(std::string_view path, std::string_view suffix)
{
    return path.size() >= suffix.size() && path.substr(path.size() - suffix.size()) == suffix;
}

/// Every L1 model an engine declares by id (`declaredIds`: (architecture, UUID) pairs) must
/// be in the list that stages it for its architecture (`stagedFiles`, a `.uhd.json` and a
/// `.bin` under `<architecture>/`), and the models its source tree ships (`shipped`,
/// embedded at configure time so the test needs no checkout at run time) must record the
/// selector revision this build computes and ship for the architecture that declares them.
///
/// A model whose recorded revision differs is refused at load (RFC 0019 §4.1) with a log
/// line, and engine selection then falls back to priority as if no model shipped; nothing
/// else fails. So a change to the engine's selector inputs, or a rebase onto one, must
/// fail the build here instead, at the model that no longer binds.
///
/// The loader binds a declared id under the declaring architecture wherever the model
/// ships, and the models record no architecture, so an id declared for the wrong one (two
/// swapped in the table, say) is bound under an architecture its model was not trained for.
/// The loader then refuses the estimate there ("UHD model has no coverage for this
/// architecture"), so that architecture goes unranked and nothing fails. The directory a
/// model ships in names its architecture; only this check compares the two.
///
/// An install carries only the architectures of its families, so a test that runs from one
/// cannot tell a model that was never staged from one that another job carries. The list
/// that stages the models is the same in every build, so a declared architecture missing
/// from it is caught here whichever install the test runs from.
///
/// @param engineName   The engine's registered name; it names this build's selector dump.
/// @param retrainGuide The file, relative to the repository root, whose header gives the
///                     steps to retrain the engine's models.
/// @param shipped      The embedded table of the engine's shipped models.
/// @param stagedFiles  The embedded list of files that stage and install the models, relative
///                     to the descriptor root (`.../<architecture>/<model>.uhd.json` and
///                     `.bin`).
/// @param dumpDir      Where the build writes `<engineName>.selector.json`.
/// @param declaredIds  The engine's declared (architecture, model id) table.
/// @param revision     The engine's `selectorRevision()`.
///
/// A shipped model no declared id names is not checked, but a table matching none of them
/// means the case would pass having checked nothing.
template <typename ShippedModels, typename StagedFiles, typename DeclaredIds>
::testing::AssertionResult shippedL1ModelsBindThisBuild(std::string_view engineName,
                                                        std::string_view retrainGuide,
                                                        const ShippedModels& shipped,
                                                        const StagedFiles& stagedFiles,
                                                        const std::filesystem::path& dumpDir,
                                                        const DeclaredIds& declaredIds,
                                                        std::string_view revision)
{
    const auto dump = (dumpDir / (std::string(engineName) + ".selector.json")).string();
    size_t checked = 0;
    std::string failures;
    for(const auto& [arch, declaredId] : declaredIds)
    {
        bool hasDescriptor = false;
        bool hasArtifact = false;
        for(const char* staged : stagedFiles)
        {
            if(std::filesystem::path(staged).parent_path().filename().string()
               != std::string_view(arch))
            {
                continue;
            }
            hasDescriptor = hasDescriptor || endsWith(staged, ".uhd.json");
            hasArtifact = hasArtifact || endsWith(staged, ".bin");
        }
        if(hasDescriptor && hasArtifact)
        {
            continue;
        }
        const char* missing = !hasDescriptor && !hasArtifact ? "/*.uhd.json and /*.bin"
                              : !hasDescriptor               ? "/*.uhd.json"
                                                             : "/*.bin";
        failures.append("\n")
            .append(engineName)
            .append(" declares the model ")
            .append(declaredId)
            .append(" for ")
            .append(arch)
            .append(", but the list that stages its models holds no ")
            .append(arch)
            .append(missing)
            .append(", so no model ships for it and the engine is unranked there.\n");
    }
    for(const ShippedModel& model : shipped)
    {
        const auto id = hipdnn_data_sdk::utilities::toLower(model.id);
        const auto shippedArch
            = std::filesystem::path(model.path).parent_path().filename().string();
        bool declared = false;
        for(const auto& [arch, declaredId] : declaredIds)
        {
            if(hipdnn_data_sdk::utilities::toLower(std::string(declaredId)) != id)
            {
                continue;
            }
            declared = true;
            if(std::string_view(arch) != shippedArch)
            {
                failures.append("\n")
                    .append(engineName)
                    .append(" declares the model ")
                    .append(id)
                    .append(" for ")
                    .append(arch)
                    .append(", but it ships as ")
                    .append(model.path)
                    .append(", for ")
                    .append(shippedArch)
                    .append(". The loader binds it to ")
                    .append(arch)
                    .append(" all the same, but it was not measured there, so the loader "
                            "refuses its estimate and that architecture is unranked.\n");
            }
        }
        if(!declared)
        {
            continue;
        }
        ++checked;

        const std::string recorded = model.revision[0] == '\0' ? "<none>" : model.revision;
        if(recorded == revision)
        {
            continue;
        }
        failures.append("\n")
            .append(engineName)
            .append(": the shipped model ")
            .append(model.path)
            .append(" (id ")
            .append(id)
            .append(") records selector_revision '")
            .append(recorded)
            .append("' but this build computes '")
            .append(revision)
            .append("'. The loader refuses it and engine selection falls back as if no model "
                    "shipped.\n  Retrain it against this build in the same change; the steps "
                    "are in the header of ")
            .append(retrainGuide)
            .append(". This build's selector is dumped at ")
            .append(dump)
            .append("\n");
    }
    if(checked == 0)
    {
        return ::testing::AssertionFailure()
               << engineName << ": none of the " << std::size(shipped)
               << " shipped models embedded at configure time carries a declared model id, so "
                  "no model was checked. The declared id table and the shipped models have "
                  "drifted apart."
               << failures;
    }
    if(failures.empty())
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << failures;
}

} // namespace hip_kernel_provider::shipped_model_gate
