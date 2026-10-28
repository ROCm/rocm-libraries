// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "origami/nn/boost/types.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

// Library stamp of Origami Boost models.
//
// A model stores a canonical manifest of its training library's PredictionMatching rows (one
// dtype class). At load the caller builds the same manifest from the library in use and
// diff_library decides: removed kernels are fine, added kernels are fine when every gated
// structural value they carry was seen in training, and anything else needs retraining.
namespace origami::nn::boost {

class ensemble_t;

/// One kernel of a PredictionMatching row of the library under check.
struct library_kernel_t {
  std::string     name;        ///< solution name
  std::int64_t    index = -1;  ///< solution index (informational: it shifts between builds)
  config_t        config{};    ///< as the caller builds it for rank_configs
  kernel_fields_t fields{};
};

/// A structural field of the library stamp.
struct structural_field_t {
  const char* name;
  bool        gate;  ///< a value no training kernel had refuses the model
};

/// The structural fields in manifest order (the only table: library_stamp.cpp).
ORIGAMI_EXPORT const std::vector<structural_field_t>& structural_fields();

/// A kernel's canonical structural values, in structural_fields() order.
ORIGAMI_EXPORT std::vector<std::string> structural_values(const config_t&        config,
                                                         const kernel_fields_t& fields);

/// 64-bit digest of a solution name (first 16 hex digits of its sha256): a kernel's identity.
ORIGAMI_EXPORT std::string name_digest(const std::string& name);

/// One canonical manifest entry. Identity is (digest, values); index is informational and the
/// name is kept only when the caller has it (the library side), for reports.
struct manifest_entry_t {
  std::string              digest;
  std::int64_t             index = -1;
  std::vector<std::string> values;
  std::string              name;
};

/// Unique (digest, values) entries sorted by (digest, values), each with its smallest index.
ORIGAMI_EXPORT std::vector<manifest_entry_t> canonical_manifest(
    const std::vector<library_kernel_t>& kernels);

/// Per structural field (structural_fields() order), the values @p entries carry.
ORIGAMI_EXPORT std::vector<std::set<std::string>> seen_structural_values(
    const std::vector<manifest_entry_t>& entries);

/// Text of a model's MANI section (the exporter writes the same bytes).
ORIGAMI_EXPORT std::string manifest_section_text(const std::string&                        dtype,
                                                 const std::vector<manifest_entry_t>&      entries,
                                                 const std::vector<std::set<std::string>>& seen);

enum class library_decision_t { accept, accept_with_additions, retrain_required };

/// "ACCEPT", "ACCEPT-WITH-ADDITIONS" or "RETRAIN-REQUIRED".
ORIGAMI_EXPORT const char* to_string(library_decision_t decision);

/// Outcome of diff_library. The model exporter's library check prints the same summary and JSON.
struct library_diff_t {
  library_decision_t decision = library_decision_t::accept;
  std::string        dtype;
  std::size_t        model_kernels = 0, library_kernels = 0, kept = 0, removed = 0, added = 0;
  /// Gated fields: values an added kernel carries that no training kernel had.
  std::map<std::string, std::set<std::string>> new_values;
  /// Ungated fields: the same, informational only.
  std::map<std::string, std::set<std::string>> new_ungated_values;
  /// Added kernels carrying a new gated value, sorted by name.
  std::vector<std::string> affected;

  ORIGAMI_EXPORT std::string summary() const;
  ORIGAMI_EXPORT std::string json() const;
};

/// Diff @p library (the canonical manifest of the library in use) against @p model's training
/// manifest.
ORIGAMI_EXPORT library_diff_t diff_library(const ensemble_t&                    model,
                                           const std::vector<manifest_entry_t>& library);

namespace detail {

enum class manifest_parse_t { ok, malformed, unsupported };

/// Parse a MANI section. unsupported: a manifest version or structural-field table this
/// runtime does not have.
manifest_parse_t parse_manifest_section(const std::string&                  text,
                                        std::string&                        dtype,
                                        std::vector<manifest_entry_t>&      entries,
                                        std::vector<std::set<std::string>>& seen,
                                        std::string&                        error);

}  // namespace detail
}  // namespace origami::nn::boost
