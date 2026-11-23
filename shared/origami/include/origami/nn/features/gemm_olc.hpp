// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "origami/hardware.hpp"
#include "origami/nn/olc/types.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

// Feature extractor of OLC: feature schema v5 (c6-5), one row of float64 columns
// per (problem, candidate) from Origami's own intermediates, the candidate's config and the
// caller's kernel fields (olc::kernel_fields_t).
namespace origami::nn::features::gemm_olc {

constexpr std::size_t num_columns = 159;

constexpr const char* catalog_id         = "gemm_olc";
constexpr const char* schema_id          = "origami-olc-gfx950-c6@5";
constexpr const char* feature_names_hash =
    "sha256:c2eb5fa5276cb5926ce9c8d7a76e8ab12e0b5a165bb446ec668ff7436653f0f3";

/// The compiled feature schema (c6-5, 159 columns: c6-3 plus the origami_launch group).
ORIGAMI_EXPORT const olc::feature_schema_t& schema();

/// Every column the extractor computes, in schema order.
ORIGAMI_EXPORT const std::vector<std::string>& slot_names();

/// Index of @p name in slot_names(), or -1 if the extractor does not compute it.
ORIGAMI_EXPORT int slot_index(const std::string& name);

/**
 * @brief Compute every schema feature for one candidate.
 *
 * @param out slot_names().size() doubles (NaN where a definition yields NaN).
 * @param available optional slot_names().size() flags; 0 marks a feature whose input
 *        kernel field the caller did not provide (its value is NaN and must not be used).
 * @return false if the candidate is unusable (Origami rejected it); @p why says why.
 */
ORIGAMI_EXPORT bool extract(const problem_t&          problem,
                            const hardware_t&         hardware,
                            const olc::candidate_t& candidate,
                            double*                   out,
                            std::string*              why       = nullptr,
                            std::uint8_t*             available = nullptr);

}  // namespace origami::nn::features::gemm_olc
