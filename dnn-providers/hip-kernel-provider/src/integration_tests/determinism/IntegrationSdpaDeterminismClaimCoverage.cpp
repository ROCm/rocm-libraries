// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "SdpaDeterminismHarness.hpp"

/**
 * @file IntegrationSdpaDeterminismClaimCoverage.cpp
 * @brief Host-only check tying the 16 claimed SDPA engines to the determinism rows that are
 *        their evidence: without it, deleting an engine's row group, or shipping a claimed
 *        FlyDSL op on an architecture no row names, would leave every other test green.
 *        Needs no device: it reads registeredDeterminismRows(), which every determinism
 *        translation unit's static initializer has already populated by the time a test body
 *        runs (gtest runs no test until every translation unit's static initializers have).
 *
 * The claimed-engine table below is this file's own record of what each engine ships, cited
 * per entry from the catalog and generator sources, not from the row tables the other
 * determinism translation units carry: reading the row tables to build this one would make the
 * check compare the rows against themselves, which no deletion or omission could ever fail.
 * Each entry is active only while its HKP_DETERMINISM_* compile definition is set (the gate
 * ../determinism/CMakeLists.txt applies to the translation unit that would otherwise carry the
 * engine's rows), so a build that compiles none of an engine's rows checks none of its archs.
 *
 * ASM_SDPA_ENGINE's forward rows and HIP_FLASH2_ENGINE's rows are PROVEN (they pass the
 * bitwise repeat test; IntegrationGpuSdpaDeterminismAsmFlash2.cpp's RowMeta default), but
 * neither engine is in the claimed table: their descriptors keep the engine NONDETERMINISTIC
 * for a reason the repeat test does not cover (each accepts an output tensor whose strides map
 * two elements to one address; KNOWN_DIVERGENCES.md). Both are exempted, by name, from the
 * "every PROVEN row names a claimed engine" check below.
 */

namespace
{

using namespace hip_kernel_provider::sdpa_determinism;

/// One claimed engine and the architectures its descriptors ship.
struct ClaimedEngine
{
    const char* engine;
    std::vector<std::string> archs;
};

/// Engines whose PROVEN rows are expected despite the engine never claiming determinism (file
/// comment above).
const std::set<std::string>& provenButUnclaimedEngines()
{
    static const std::set<std::string> s_engines = {"ASM_SDPA_ENGINE", "HIP_FLASH2_ENGINE"};
    return s_engines;
}

/// The 16 claimed engines and the architectures their descriptors ship, read from the catalog
/// and generator sources.
std::vector<ClaimedEngine> claimedEngines()
{
    std::vector<ClaimedEngine> engines;
#ifdef HKP_DETERMINISM_ROCKE_GFX942_DENSE
    // projects/hipdnn/tools/IngestorGenerator/configs/gfx942_attention_dense.yaml:284-285
    // (packs[0].arch: [gfx942]).
    engines.push_back({"hipkernel:Gfx942AttentionDense", {"gfx942"}});
#endif
#ifdef HKP_DETERMINISM_ROCKE_GFX942_UNIFIED_2D
    // .../configs/gfx942_attention_unified_2d.yaml:175-176 (arch: [gfx942]).
    engines.push_back({"hipkernel:Gfx942AttentionUnified2D", {"gfx942"}});
#endif
#ifdef HKP_DETERMINISM_ROCKE_GFX942_UNIFIED_3D
    // .../configs/gfx942_attention_unified_3d.yaml:106-107 (arch: [gfx942]).
    engines.push_back({"hipkernel:Gfx942AttentionUnified3d", {"gfx942"}});
#endif
#ifdef HKP_DETERMINISM_ROCKE_GFX950_DENSE
    // .../configs/gfx950_attention_dense.yaml:191-192 (arch: [gfx950]).
    engines.push_back({"hipkernel:Gfx950AttentionDense", {"gfx950"}});
#endif
#ifdef HKP_DETERMINISM_ROCKE_GFX950_UNIFIED_2D
    // .../configs/gfx950_attention_unified_2d.yaml:166-167 (arch: [gfx950]).
    engines.push_back({"hipkernel:Gfx950AttentionUnified2D", {"gfx950"}});
#endif
#ifdef HKP_DETERMINISM_ROCKE_GFX950_UNIFIED_3D
    // .../configs/gfx950_attention_unified_3d.yaml:114-115 (arch: [gfx950]).
    engines.push_back({"hipkernel:Gfx950AttentionUnified3D", {"gfx950"}});
#endif
#ifdef HKP_DETERMINISM_WMMA_PREFILL
    // .../configs/gfx1151_wmma_attention.yaml:135 ships the gfx11-generic and gfx12-generic
    // families; a generic family is proven by running its code object on one named device,
    // the tokens IntegrationGpuSdpaDeterminismWmma.cpp's runArchs() returns.
    engines.push_back({"hipkernel:Gfx1151WmmaAttention", {"gfx1100", "gfx1201"}});
#endif
#ifdef HKP_DETERMINISM_WMMA_DECODE
    // .../configs/gfx1151_wmma_attention_decode.yaml:86, same families and the same rule.
    engines.push_back({"hipkernel:Gfx1151WmmaAttentionDecode", {"gfx1100", "gfx1201"}});
#endif
#ifdef HKP_DETERMINISM_FLYDSL_SDPA_CDNA
    // flydsl/generators/_instances.py:1364 ("for arch in (gfx950, gfx942, gfx90a)").
    engines.push_back({"hipkernel:flydsl_sdpa_cdna", {"gfx90a", "gfx942", "gfx950"}});
#endif
#ifdef HKP_DETERMINISM_FLYDSL_SDPA_CDNA_DECODE
    // Same CDNA family targets as sdpa_cdna (_instances.py:1364); IntegrationGpuSdpaDeterminism
    // FlydslFwd.cpp's cdna_decode table pins the same three archs.
    engines.push_back({"hipkernel:flydsl_sdpa_cdna_decode", {"gfx90a", "gfx942", "gfx950"}});
#endif
#ifdef HKP_DETERMINISM_FLYDSL_SDPA_CDNA_DUALWAVE
    // The dual-wave kernel is gfx950 only (IntegrationGpuSdpaDeterminismFlydslFwd.cpp's own
    // comment: "gfx950's dual-wave kernel").
    engines.push_back({"hipkernel:flydsl_sdpa_cdna_dualwave", {"gfx950"}});
#endif
#ifdef HKP_DETERMINISM_FLYDSL_SDPA_RDNA
    // _instances.py:799 ("on gfx11-generic and gfx12-generic"); run tokens gfx1100/gfx1201.
    engines.push_back({"hipkernel:flydsl_sdpa_rdna", {"gfx1100", "gfx1201"}});
#endif
#ifdef HKP_DETERMINISM_FLYDSL_SDPA_RDNA_DECODE
    // Same RDNA family targets as sdpa_rdna.
    engines.push_back({"hipkernel:flydsl_sdpa_rdna_decode", {"gfx1100", "gfx1201"}});
#endif
#ifdef HKP_DETERMINISM_FLYDSL_SDPA_AIW
    // _instances.py:2399 (SDPA_AIW_TARGETS = ("gfx12-generic",)); run token gfx1201.
    engines.push_back({"hipkernel:flydsl_sdpa_aiw", {"gfx1201"}});
#endif
#ifdef HKP_DETERMINISM_FLYDSL_BWD_MERGED
    // _instances.py:2635 (SDPA_BWD_TARGET = "gfx950").
    engines.push_back({"hipkernel:flydsl_sdpa_bwd_merged", {"gfx950"}});
#endif
#ifdef HKP_DETERMINISM_FLYDSL_BWD_SPLIT
    // _instances.py:2635 (SDPA_BWD_TARGET = "gfx950", reused for fp32 on gfx942/gfx90a),
    // :2695 (SDPA_BWD_GFX11_TARGET = "gfx11-generic", run token gfx1100), :2698
    // (SDPA_BWD_CDNA9_TARGETS = ("gfx942", "gfx90a")).
    engines.push_back(
        {"hipkernel:flydsl_sdpa_bwd_split", {"gfx950", "gfx942", "gfx90a", "gfx1100"}});
#endif
    return engines;
}

std::string joinWith(const std::vector<std::string>& parts, const std::string& separator)
{
    std::string joined;
    for(const auto& part : parts)
    {
        joined += (joined.empty() ? "" : separator) + part;
    }
    return joined;
}

/// Every (engine, arch) the table requires, with no PROVEN registered row covering it.
std::vector<std::string> missingRows(const std::vector<ClaimedEngine>& claimed,
                                     const std::vector<RegisteredRow>& rows)
{
    std::vector<std::string> missing;
    for(const auto& engine : claimed)
    {
        for(const auto& arch : engine.archs)
        {
            const bool covered
                = std::any_of(rows.begin(), rows.end(), [&](const RegisteredRow& row) {
                      return row.disposition == Disposition::PROVEN && row.engine == engine.engine
                             && std::find(row.archs.begin(), row.archs.end(), arch)
                                    != row.archs.end();
                  });
            if(!covered)
            {
                missing.push_back(std::string(engine.engine) + "@" + arch);
            }
        }
    }
    return missing;
}

/// Every PROVEN registered row whose engine names neither a claimed engine nor an exempted one.
std::vector<std::string> unexpectedProvenRows(const std::vector<ClaimedEngine>& claimed,
                                              const std::vector<RegisteredRow>& rows)
{
    std::set<std::string> known = provenButUnclaimedEngines();
    for(const auto& engine : claimed)
    {
        known.insert(engine.engine);
    }
    std::vector<std::string> unexpected;
    for(const auto& row : rows)
    {
        if(row.disposition == Disposition::PROVEN && known.count(row.engine) == 0)
        {
            unexpected.push_back(row.engine);
        }
    }
    return unexpected;
}

void checkEveryClaimedEngineArchHasAProvenRow()
{
    const auto missing = missingRows(claimedEngines(), registeredDeterminismRows());
    EXPECT_TRUE(missing.empty())
        << "No PROVEN determinism row covers: " << joinWith(missing, ", ")
        << " (a deleted row group, or an arch this file's table claims with no matching row)";
}

void checkNoProvenRowForAnUnclaimedEngine()
{
    const auto unexpected = unexpectedProvenRows(claimedEngines(), registeredDeterminismRows());
    EXPECT_TRUE(unexpected.empty())
        << "PROVEN determinism row(s) for an engine absent from the claimed-engine table: "
        << joinWith(unexpected, ", ")
        << " (add the engine to claimedEngines() if its claim is real, or fix the row's "
           "disposition)";
}

/// Runs one of the checks above. ::testing::RegisterTest, not TEST(), because the "Standard/"
/// tier prefix needs a literal '/' before the suite name, which TEST()'s identifier-token
/// arguments cannot carry.
class ClaimCoverageCase : public ::testing::Test
{
public:
    explicit ClaimCoverageCase(std::function<void()> check)
        : _check(std::move(check))
    {
    }

private:
    void TestBody() override
    {
        _check();
    }

    std::function<void()> _check;
};

bool registerClaimCoverageTests()
{
    static_cast<void>(::testing::RegisterTest("Standard/IntegrationSdpaDeterminismClaimCoverage",
                                              "EveryClaimedEngineArchHasAProvenRow",
                                              nullptr,
                                              nullptr,
                                              __FILE__,
                                              __LINE__,
                                              []() -> ::testing::Test* {
                                                  return new ClaimCoverageCase(
                                                      checkEveryClaimedEngineArchHasAProvenRow);
                                              }));
    static_cast<void>(::testing::RegisterTest("Standard/IntegrationSdpaDeterminismClaimCoverage",
                                              "NoProvenRowForAnUnclaimedEngine",
                                              nullptr,
                                              nullptr,
                                              __FILE__,
                                              __LINE__,
                                              []() -> ::testing::Test* {
                                                  return new ClaimCoverageCase(
                                                      checkNoProvenRowForAnUnclaimedEngine);
                                              }));
    return true;
}

[[maybe_unused]] const bool REGISTERED = registerClaimCoverageTests();

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENABLE_SDPA
