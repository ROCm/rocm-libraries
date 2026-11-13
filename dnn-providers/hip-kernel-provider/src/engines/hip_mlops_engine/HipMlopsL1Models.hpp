// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <string_view>
#include <utility>

namespace hip_kernel_provider
{

/// The L1 (`predict_engine`) models HIP_MLOPS_ENGINE binds: architecture to UHD UUID (RFC
/// 0019 Open Question 7). The engine ships no UED, so this compiled-in table is the binding
/// identity, never the document. One row per (architecture, ranking metric); each model
/// answers in its own `score.metric`. An id nothing deploys leaves the engine UNAVAILABLE,
/// so ids can ship ahead of their models; once a model ships, changing its id orphans it.
///
/// Kept out of HipMlopsSelectorRevision.cmake's inputs: adding a model for one
/// architecture must not expire the others.
inline constexpr std::array<std::pair<std::string_view, std::string_view>, 2>
    HIP_MLOPS_L1_MODEL_IDS{{
        {"gfx942", "1c541b68-06c6-4b6e-b2a5-18bcdcc03fea"}, // tflops
        {"gfx950", "163d8e83-807f-43cc-979e-8a1333b1beb4"}, // tflops
    }};

} // namespace hip_kernel_provider
