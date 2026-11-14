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
/// checkout at run time) must record the selector revision this build computes.
///
/// A model whose recorded revision differs is refused at load (RFC 0019 §4.1) with a log
/// line, and engine selection then falls back to priority as if no model shipped; nothing
/// else fails. So a change to the engine's selector inputs, or a rebase onto one, must
/// fail the build here instead, at the model that no longer binds.
///
/// @param engineName   The engine's registered name; it names this build's selector dump.
/// @param dumpEngine   The `HKP_ENGINE` value of HkpSelectorManifest.cmake for it.
/// @param shipped      The embedded table of the engine's shipped models.
/// @param dumpDir      Where the build writes `<engineName>.selector.json`.
/// @param declaredIds  The engine's declared (architecture, model id) table.
/// @param revision     The engine's `selectorRevision()`.
///
/// A declared id no shipped model carries is fine (ids may ship ahead of their models), but
/// a table matching none of them means the case would pass having checked nothing.
template <typename ShippedModels, typename DeclaredIds>
::testing::AssertionResult shippedL1ModelsBindThisBuild(std::string_view engineName,
                                                        std::string_view dumpEngine,
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
        const bool declared
            = std::any_of(declaredIds.begin(), declaredIds.end(), [&](const auto& declaration) {
                  return detail::lowercase(std::string(declaration.second)) == id;
              });
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
        failures += "\n" + std::string(engineName) + ": the shipped model " + model.path + " (id "
                    + id + ") records selector_revision '" + recorded
                    + "' but this build computes '" + std::string(revision)
                    + "'. The loader refuses it and engine selection falls back as if no model "
                      "shipped.\n  Restamp it only with proof:\n    python -m uhd_gen restamp "
                    + model.path + " --selector-before <dump at the training build>"
                    + " --selector-after " + dump
                    + " [--rows-before <L1 corpus at the old revision> --rows-after <L1 corpus "
                      "at the new revision>] --write\n";
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
    return ::testing::AssertionFailure()
           << failures
           << "\nMake the before-dump at the training commit with\n  cmake -DHKP_ENGINE="
           << dumpEngine
           << " -DHKP_PROVIDER_DIR=<checkout>/dnn-providers/hip-kernel-provider -DHKP_OUT=<file> "
              "-P <checkout>/dnn-providers/hip-kernel-provider/cmake/HkpSelectorManifest.cmake\n"
              "or retrain the model against this build.";
}

} // namespace hip_kernel_provider::shipped_model_gate
