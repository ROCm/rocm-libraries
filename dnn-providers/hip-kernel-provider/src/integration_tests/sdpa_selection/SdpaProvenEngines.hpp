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
// engine of the build serves the forward and the backward graph on; the subset of each that
// ships a claimed engine (HKP_SDPA_SELECTION_FWD_CLAIMED_ARCHS, _BWD_CLAIMED_ARCHS); and, for the
// three FlyDSL forward engines that are never claimed (the head-pad copy they wrap is unproven),
// the architectures each ships forward content for.
#ifndef HKP_SDPA_SELECTION_FWD_ARCHS
#define HKP_SDPA_SELECTION_FWD_ARCHS ""
#endif
#ifndef HKP_SDPA_SELECTION_FWD_CLAIMED_ARCHS
#define HKP_SDPA_SELECTION_FWD_CLAIMED_ARCHS ""
#endif
#ifndef HKP_SDPA_SELECTION_BWD_ARCHS
#define HKP_SDPA_SELECTION_BWD_ARCHS ""
#endif
#ifndef HKP_SDPA_SELECTION_BWD_CLAIMED_ARCHS
#define HKP_SDPA_SELECTION_BWD_CLAIMED_ARCHS ""
#endif
#ifndef HKP_SDPA_SELECTION_FLYDSL_CDNA_ARCHS
#define HKP_SDPA_SELECTION_FLYDSL_CDNA_ARCHS ""
#endif
#ifndef HKP_SDPA_SELECTION_FLYDSL_RDNA_ARCHS
#define HKP_SDPA_SELECTION_FLYDSL_RDNA_ARCHS ""
#endif
#ifndef HKP_SDPA_SELECTION_FLYDSL_AIW_ARCHS
#define HKP_SDPA_SELECTION_FLYDSL_AIW_ARCHS ""
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
    /// expected to leave a survivor. False on an architecture where every engine that ranks the
    /// graph is unclaimed: the forward graph's architectures served only by
    /// hipkernel:flydsl_sdpa_cdna, _rdna or _aiw (the head-pad copy they wrap is unproven);
    /// BwdAsmOnly's one architecture (ASM_SDPA_ENGINE ranks alone and backward never claims);
    /// and BwdBf16Hd128's architectures that ASM serves but no claimed backward engine is built
    /// for.
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
    /// One engine that is never claimed on this row's graph, and the architectures it ranks the
    /// graph on.
    struct UnclaimedEngine
    {
        const char* name;
        std::vector<std::string> archs;
    };

    struct Case
    {
        const char* token;
        GraphKind kind;
        std::vector<std::string> archs;
        std::vector<std::string> served;
        /// The unclaimed engines that may rank the graph; a row's unclaimedRanked lists the
        /// subset whose archs include that row's architecture.
        std::vector<UnclaimedEngine> unclaimed;
        /// The architectures on which every engine that ranks the graph is unclaimed, so no
        /// survivor is expected.
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
    const auto fwdClaimedArchs = archsOf(HKP_SDPA_SELECTION_FWD_CLAIMED_ARCHS);
    const auto bwdServed = archsOf(HKP_SDPA_SELECTION_BWD_ARCHS);
    const auto bwdClaimedArchs = archsOf(HKP_SDPA_SELECTION_BWD_CLAIMED_ARCHS);
    const auto cdnaArchs = archsOf(HKP_SDPA_SELECTION_FLYDSL_CDNA_ARCHS);
    const auto rdnaArchs = archsOf(HKP_SDPA_SELECTION_FLYDSL_RDNA_ARCHS);
    const auto aiwArchs = archsOf(HKP_SDPA_SELECTION_FLYDSL_AIW_ARCHS);

    std::vector<Case> cases;
    // Forward: ASM forward declines the graph -- its kernel table has no bf16 head-size-64 row
    // (key.empty(), checkFwdApplicability, SdpaFwdPlanBuilder.cpp) -- and Flash2 serves fp16 only
    // (isApplicable, HipFlash2FwdPlanBuilder.cpp), so every claimed engine ranking it is a rocKE or
    // WMMA one. hipkernel:flydsl_sdpa_cdna, _rdna and _aiw also rank this dense bf16/hd64/causal
    // shape through their generic unsplit prefill kernel (each engine's own determinism table,
    // first class) but are never claimed, so an architecture only one of them serves (no rocKE/WMMA
    // engine of this build targets it, e.g. gfx90a) gets no survivor.
    std::vector<std::string> fwdNoSurvivorArchs;
    for(const auto& arch : fwdServed)
    {
        if(!built(fwdClaimedArchs, arch))
        {
            fwdNoSurvivorArchs.push_back(arch);
        }
    }
    cases.push_back({"FwdBf16Hd64Causal",
                     GraphKind::FORWARD,
                     {"gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1201"},
                     fwdServed,
                     {{"hipkernel:flydsl_sdpa_cdna", cdnaArchs},
                      {"hipkernel:flydsl_sdpa_rdna", rdnaArchs},
                      {"hipkernel:flydsl_sdpa_aiw", aiwArchs}},
                     fwdNoSurvivorArchs});
    // Backward: ASM_SDPA_ENGINE accepts the packed bf16 hd128 graph (equal heads and packed stats,
    // both in checkBwdApplicability, SdpaBwdPlanBuilder.cpp) beside the FlyDSL split engine, on
    // gfx942 and gfx950 (isSupportedArch, isBatchShapeSupported, both SdpaBwdPlanBuilder.cpp), and
    // is never claimed, when this build has it.
    std::vector<UnclaimedEngine> bwdUnclaimed;
#ifdef HKP_SDPA_SELECTION_ASM_ONLY
    bwdUnclaimed.push_back({"ASM_SDPA_ENGINE", {"gfx942", "gfx950"}});
#endif
    // Architectures ASM serves without a claimed backward engine built: the filter leaves no
    // survivor there, whatever FlyDSL's content for this build happens to cover.
    std::vector<std::string> bwdNoSurvivorArchs;
    for(const auto& arch : bwdServed)
    {
        if(!built(bwdClaimedArchs, arch))
        {
            bwdNoSurvivorArchs.push_back(arch);
        }
    }
    cases.push_back({"BwdBf16Hd128",
                     GraphKind::BACKWARD,
                     {"gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1201"},
                     bwdServed,
                     bwdUnclaimed,
                     bwdNoSurvivorArchs});
#ifdef HKP_SDPA_SELECTION_ASM_ONLY
    cases.push_back({"BwdAsmOnly",
                     GraphKind::BACKWARD_ASM_ONLY,
                     {"gfx942"},
                     {"gfx942"},
                     {{"ASM_SDPA_ENGINE", {"gfx942"}}},
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
            std::vector<std::string> unclaimedRanked;
            for(const auto& engine : selectionCase.unclaimed)
            {
                if(built(engine.archs, arch))
                {
                    unclaimedRanked.emplace_back(engine.name);
                }
            }
            rows.push_back({std::string(selectionCase.token) + "_" + arch,
                            selectionCase.kind,
                            arch,
                            std::move(unclaimedRanked),
                            !built(selectionCase.noSurvivorArchs, arch)});
        }
    }
    return rows;
}

} // namespace hip_kernel_provider::sdpa_selection
