// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

/**
 * @file SdpaProvenEngines.hpp
 * @brief The rows of the SDPA note-selection suites: the graphs they run
 *        `deselect_numeric_notes({NONDETERMINISTIC})` on, per device architecture.
 *
 * Shared by the native suite (IntegrationGpuSdpaNativeNoteFilter.cpp) and the cuDNN v9 shim suite
 * (IntegrationGpuCudnnShimSdpaDeterministic.cpp), so both list the same rows. No claim is listed
 * here: which engines survive is read from the engines' notes at run time, and a claim lives in
 * the engine's descriptor. Two fixed facts are per row: the engines named in
 * SelectionRow::unclaimedRanked rank the row's graph and report NONDETERMINISTIC; and whether a
 * claimed engine ranks the graph at all (SelectionRow::expectsSurvivor), so the filter leaves a
 * survivor.
 */
// CMakeLists.txt defines them for the integration binary: the comma-separated architectures an
// engine of the build serves the forward and the backward graph on.
#ifndef HKP_SDPA_SELECTION_FWD_ARCHS
#define HKP_SDPA_SELECTION_FWD_ARCHS ""
#endif
#ifndef HKP_SDPA_SELECTION_BWD_ARCHS
#define HKP_SDPA_SELECTION_BWD_ARCHS ""
#endif

namespace hip_kernel_provider::sdpa_selection
{

/// The graph a row builds. BACKWARD_ASM_ONLY is a backward graph that only ASM_SDPA_ENGINE ranks
/// (only its O and dQ are padded, to a row pitch of D + 1 that the FlyDSL backward matcher declines
/// and the ASM builder accepts), pinned to gfx942: ASM_SDPA_ENGINE backward also serves gfx950, but
/// whether FlyDSL's gfx950 backward declines the same padded pitch is unverified.
enum class GraphKind
{
    FORWARD,
    BACKWARD,
    BACKWARD_ASM_ONLY,
};

struct SelectionRow
{
    /// `<Case>_<arch>`: the last token is the one device architecture the row runs on.
    std::string name;
    GraphKind kind = GraphKind::FORWARD;
    std::string arch;
    /// The engines that rank the row's graph on this architecture and report NONDETERMINISTIC
    /// whatever the claims (ASM_SDPA_ENGINE backward is never claimed). Both suites assert
    /// each is in the unfiltered ranking and reports NONDETERMINISTIC, and the native pin test
    /// pins the first.
    std::vector<std::string> unclaimedRanked;
    /// Whether a claimed engine ranks the row's graph on this architecture, so the filter is
    /// expected to leave a survivor. False only where every ranked engine is unclaimed
    /// (BwdAsmOnly on gfx942, where ASM_SDPA_ENGINE ranks alone and backward never claims).
    /// A fact about the graph's ranking, not a list of claimed engines.
    bool expectsSurvivor = true;
};

/// The registered names of @p ids (the hexadecimal id of an unregistered one), for messages and
/// test properties.
inline std::string engineNamesText(const std::vector<int64_t>& ids)
{
    std::string text;
    for(const auto id : ids)
    {
        text += (text.empty() ? "" : ",") + hipdnn_data_sdk::utilities::engineNameOrHex(id);
    }
    return text.empty() ? "(none)" : text;
}

/// Every row of both suites, one per (case, architecture) the build can rank the graph on.
inline std::vector<SelectionRow> selectionRows()
{
    struct Case
    {
        const char* token;
        GraphKind kind;
        std::vector<std::string> archs;
        std::vector<std::string> served;
        /// The architectures on which `unclaimed` rank the graph.
        std::vector<std::string> unclaimedArchs;
        /// The engines that rank the graph on those architectures and report NONDETERMINISTIC.
        std::vector<std::string> unclaimed;
        /// The architectures on which every ranked engine is unclaimed, so no survivor is
        /// expected.
        std::vector<std::string> noSurvivorArchs;
    };
    // A row exists only for an architecture an engine of this build serves its graph on
    // (CMakeLists.txt defines the HKP_SDPA_SELECTION_* macros), so no row fails for want of any
    // engine to rank.
    const auto archsOf = [](const char* list) {
        std::vector<std::string> archs;
        const std::string text = list;
        size_t begin = 0;
        while(begin < text.size())
        {
            auto end = text.find(',', begin);
            if(end == std::string::npos)
            {
                end = text.size();
            }
            if(end > begin)
            {
                archs.push_back(text.substr(begin, end - begin));
            }
            begin = end + 1;
        }
        return archs;
    };
    const auto built = [](const std::vector<std::string>& served, const std::string& arch) {
        return std::find(served.begin(), served.end(), arch) != served.end();
    };
    const auto fwdServed = archsOf(HKP_SDPA_SELECTION_FWD_ARCHS);
    const auto bwdServed = archsOf(HKP_SDPA_SELECTION_BWD_ARCHS);

    std::vector<Case> cases;
    // Forward: ASM forward declines the graph -- its kernel table has no bf16 head-size-64 row
    // (key.empty(), SdpaFwdPlanBuilder.cpp:1041-1045) -- and Flash2 serves fp16 only
    // (HipFlash2FwdPlanBuilder.cpp:178), so every engine ranking it is a claimable one: no
    // unclaimed engine is expected to rank it.
    cases.push_back({"FwdBf16Hd64Causal",
                     GraphKind::FORWARD,
                     {"gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1201"},
                     fwdServed,
                     {},
                     {},
                     {}});
    // Backward: ASM_SDPA_ENGINE accepts the packed bf16 hd128 graph (equal heads at
    // SdpaBwdPlanBuilder.cpp:1185-1186, packed stats at :1196-1219) beside the FlyDSL split engine,
    // on gfx942 and gfx950 (isSupportedArch :636-638, isBatchShapeSupported :699-721), and is never
    // claimed, when this build has it.
    std::vector<std::string> bwdUnclaimedArchs;
    std::vector<std::string> bwdUnclaimed;
#ifdef HKP_SDPA_SELECTION_ASM_ONLY
    bwdUnclaimedArchs.emplace_back("gfx942");
    bwdUnclaimedArchs.emplace_back("gfx950");
    bwdUnclaimed.emplace_back("ASM_SDPA_ENGINE");
#endif
    cases.push_back({"BwdBf16Hd128",
                     GraphKind::BACKWARD,
                     {"gfx90a", "gfx942", "gfx950", "gfx1100"},
                     bwdServed,
                     bwdUnclaimedArchs,
                     bwdUnclaimed,
                     {}});
#ifdef HKP_SDPA_SELECTION_ASM_ONLY
    cases.push_back({"BwdAsmOnly",
                     GraphKind::BACKWARD_ASM_ONLY,
                     {"gfx942"},
                     {"gfx942"},
                     {"gfx942"},
                     {"ASM_SDPA_ENGINE"},
                     {"gfx942"}});
#endif
    std::vector<SelectionRow> rows;
    for(const auto& selectionCase : cases)
    {
        for(const auto& arch : selectionCase.archs)
        {
            if(!built(selectionCase.served, arch))
            {
                continue;
            }
            rows.push_back({std::string(selectionCase.token) + "_" + arch,
                            selectionCase.kind,
                            arch,
                            built(selectionCase.unclaimedArchs, arch) ? selectionCase.unclaimed
                                                                      : std::vector<std::string>{},
                            !built(selectionCase.noSurvivorArchs, arch)});
        }
    }
    return rows;
}

} // namespace hip_kernel_provider::sdpa_selection
