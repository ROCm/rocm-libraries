// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "harness/bundle/SupportObservationLog.hpp"

namespace hipdnn_integration_tests::bundle
{

struct WriteSummary
{
    size_t filesWritten = 0;
    size_t filesUnchanged = 0; // on-disk bytes already matched — no mtime bump
    size_t filesSkipped = 0; // left untouched: nothing to claim, or refused
    size_t observationsApplied = 0;
    std::vector<std::string> errors;
};

WriteSummary writeObservedSupportClaims(const std::vector<ObservedGraphSupport>& observations);

struct AuthoringResult
{
    WriteSummary writeSummary;
    bool shouldFail = false;
};

AuthoringResult authorSupportClaims(const std::vector<ObservedGraphSupport>& observations,
                                    std::size_t graphsObserved,
                                    std::size_t graphsUnobserved,
                                    std::size_t graphsRegistered,
                                    std::ostream& log);

} // namespace hipdnn_integration_tests::bundle
