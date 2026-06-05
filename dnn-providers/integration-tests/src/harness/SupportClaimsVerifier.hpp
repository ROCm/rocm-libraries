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
inline constexpr const char* RULE_C = "#rule-c-support-status-unknown";
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
        C_STATUS_UNKNOWN,
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
    //   artifactPath  — where to write the long-form failure list (CI
    //                   uploads this); empty disables file output.
    SupportClaimsVerifier(const SupportClaims& claims,
                          std::string engineName,
                          std::string archToken,
                          std::string platform,
                          std::string artifactPath = "support_claim_failures.txt");

    // Run all rules against the SupportMatrixCollector snapshot and the
    // gtest UnitTest state. Returns true iff zero FAIL-severity findings
    // were produced — caller maps this onto the process exit code so a
    // regression actually fails CI, not just logs noise.
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
    std::string _artifactPath;
    std::vector<SupportClaimFinding> _findings;
};

} // namespace hipdnn_integration_tests
