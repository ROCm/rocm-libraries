// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
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
    std::vector<std::string> errors;
};

WriteSummary writeObservedSupportClaims(const std::vector<ObservedGraphSupport>& observations);

} // namespace hipdnn_integration_tests::bundle
