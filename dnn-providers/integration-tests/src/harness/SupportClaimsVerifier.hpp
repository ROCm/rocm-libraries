// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <vector>

#include "harness/SupportClaims.hpp"

namespace hipdnn_integration_tests
{

// Per-rule output anchor in docs/support-claims-failures.md. The anchors
// are stable — the failure message format embeds them so a CI log can be
// clicked straight to the remediation step (RFC 0012 §14 "Runtime
// linking — load-bearing"). If you rename an anchor, update both ends.
namespace support_claim_anchors
{
inline constexpr const char* RULE_A = "#rule-a-claim-broken";
inline constexpr const char* RULE_B = "#rule-b-issue-before-the-test-runs";
inline constexpr const char* RULE_C = "#rule-c-zero-coverage-matcher";
inline constexpr const char* RULE_D = "#rule-d-engine-over-claim";
inline constexpr const char* RULE_E = "#rule-e-unclaimed-gain";
} // namespace support_claim_anchors

// One finding produced by the verifier. The body is rendered ready for
// stderr and the artifact file — multi-line, already indented for its
// section heading.
struct SupportClaimFinding
{
    enum class Severity
    {
        FAIL,
        NOTE,
        WARNING,
    };

    enum class Rule
    {
        A_CLAIM_BROKEN,
        B_ISSUE_BEFORE_TEST,
        C_ZERO_COVERAGE_MATCHER,
        D_ENGINE_OVER_CLAIM,
        E_UNCLAIMED_GAIN,
    };

    Rule rule;
    Severity severity;
    std::string body;
};

class SupportClaimsVerifier
{
public:
    // Inputs from main.cpp:
    //   claims        — the loaded sidecar's parsed [[supported]] data
    //   engineName    — the active engine name (matches [meta].engine)
    //   archToken     — short arch (e.g. "gfx942"), pre-tokenized
    //   platform      — "windows" or "linux"
    //   fullCiMode    — true iff no --gtest_filter and no GTEST_SHARD_*
    //                   env vars are set. Rule C only fires hard in full
    //                   CI mode; locally it downgrades to a note.
    //   artifactPath  — where to write the long-form failure list (CI
    //                   uploads this); empty disables file output.
    SupportClaimsVerifier(const SupportClaims& claims,
                          std::string engineName,
                          std::string archToken,
                          std::string platform,
                          bool fullCiMode,
                          std::string artifactPath = "support_claim_failures.txt");

    // Run all five rules against the SupportMatrixCollector snapshot and
    // the gtest UnitTest state. Returns true iff zero FAIL-severity
    // findings were produced — caller maps this onto the process exit
    // code so a regression actually fails CI, not just logs noise.
    bool runAndReport();

    const std::vector<SupportClaimFinding>& findings() const
    {
        return _findings;
    }

private:
    const SupportClaims& _claims;
    std::string _engineName;
    std::string _archToken;
    std::string _platform;
    bool _fullCiMode;
    std::string _artifactPath;
    std::vector<SupportClaimFinding> _findings;
};

// Helper for main.cpp to decide whether we're in a full CI run for the
// purposes of Rule C. RFC 0012 §6.2: any --gtest_filter or shard env var
// downgrades the run to "partial".
bool detectFullCiMode();

} // namespace hipdnn_integration_tests
