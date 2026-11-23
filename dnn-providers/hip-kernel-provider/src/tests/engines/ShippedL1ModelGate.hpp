// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iterator>
#include <string>
#include <string_view>

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

namespace detail
{

inline std::string lowercase(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return text;
}

} // namespace detail

/// Every L1 model an engine declares by id (`declaredIds`: (architecture, UUID) pairs) that
/// its source tree ships (`shipped`, embedded at configure time so the test needs no
/// checkout at run time) must record the selector revision this build computes, and must
/// ship for the architecture that declares it.
///
/// A model whose recorded revision differs is refused at load (RFC 0019 §4.1) with a log
/// line, and engine selection then falls back to priority as if no model shipped; nothing
/// else fails. So a change to the engine's selector inputs, or a rebase onto one, must
/// fail the build here instead, at the model that no longer binds.
///
/// The loader binds a declared id under the declaring architecture wherever the model
/// ships, and the models record no architecture, so an id declared for the wrong one (two
/// swapped in the table, say) would answer for devices it was never measured on. The
/// directory a model ships in names its architecture; only this check compares the two.
///
/// @param engineName   The engine's registered name; it names this build's selector dump.
/// @param retrainGuide The file, relative to the repository root, whose header gives the
///                     steps to retrain the engine's models.
/// @param shipped      The embedded table of the engine's shipped models.
/// @param dumpDir      Where the build writes `<engineName>.selector.json`.
/// @param declaredIds  The engine's declared (architecture, model id) table.
/// @param revision     The engine's `selectorRevision()`.
///
/// A declared id no shipped model carries is fine (ids may ship ahead of their models), but
/// a table matching none of them means the case would pass having checked nothing.
template <typename ShippedModels, typename DeclaredIds>
::testing::AssertionResult shippedL1ModelsBindThisBuild(std::string_view engineName,
                                                        std::string_view retrainGuide,
                                                        const ShippedModels& shipped,
                                                        const std::filesystem::path& dumpDir,
                                                        const DeclaredIds& declaredIds,
                                                        std::string_view revision)
{
    const auto dump = (dumpDir / (std::string(engineName) + ".selector.json")).string();
    size_t checked = 0;
    std::string failures;
    for(const ShippedModel& model : shipped)
    {
        const auto id = detail::lowercase(model.id);
        const auto shippedArch
            = std::filesystem::path(model.path).parent_path().filename().string();
        bool declared = false;
        for(const auto& [arch, declaredId] : declaredIds)
        {
            if(detail::lowercase(std::string(declaredId)) != id)
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
                    .append(" all the same, so those devices would be answered by a model "
                            "measured on another architecture.\n");
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
                  "drifted apart.";
    }
    if(failures.empty())
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << failures;
}

} // namespace hip_kernel_provider::shipped_model_gate
