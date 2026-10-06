// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "model_impl.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace tilewright {
namespace detail {

// Device values from the caller plus the arch constants the model was trained
// with.
struct HwView {
  double N_CU           = 0.0;
  double LDS            = 0.0;
  double L2             = 0.0;
  double parallel_mi_cu = 1.0;
  double c0             = 0.0;
  double c1             = 0.0;
  double c2             = 0.0;
};

HwView hw_view(const Model& model, const Hardware& hardware);

constexpr std::size_t kMaxViewKeySize = 16;

// The features a model was trained on, selected by the hash in its file.
// Feature math runs in double and rounds once to float, as the training
// pipeline does (float64 arithmetic, then a float32 cast).
struct FeatureCatalog {
  const char* hash;
  std::size_t query_dim;
  std::size_t item_dim;
  std::size_t inter_dim;
  // Configs with equal view keys are indistinguishable to the catalog and to
  // the ranking rules: they get identical features, feasibility, whitelist
  // membership and scores. At most kMaxViewKeySize values.
  std::size_t view_key_size;
  void (*view_key)(const Config& c, std::uint64_t* out);
  // The execution contexts the catalog's models were trained for.
  bool (*supports)(const ExecutionContext& context, const Hardware& hardware);
  // Config attributes the catalog reads; callers may leave out all others.
  const char* const* attribute_names;
  std::size_t attribute_count;
  void (*query)(const Problem& p, const HwView& hw, float* out);
  void (*item)(const Config& c, float* out);
  void (*inter)(const Model& m, const Problem& p, const Config& c, const HwView& hw, float* out);
};

// The catalog whose hash is `hash`, or nullptr.
const FeatureCatalog* find_catalog(const std::string& hash) noexcept;

// The catalog of the shipped models (MLREC_v2 files).
const FeatureCatalog& v2_catalog() noexcept;

// Raw MI cycles for (mi_m, mi_n, mi_k, dtype) from the model's table, with fnuz
// dtypes looked up as their non-fnuz counterparts.
double mi_cycles(const Model& model,
                 std::size_t mi_m,
                 std::size_t mi_n,
                 std::size_t mi_k,
                 DataType dt);

bool check_lds_capacity(const Hardware& hardware, const Dim3& mt, DataType a, DataType b);

// `nt_a_available` / `nt_b_available`: the candidate pool contains a kernel with
// cache_hints_a / cache_hints_b == 4.
bool is_kernel_feasible(const Problem& p,
                        const Config& c,
                        bool nt_a_available,
                        bool nt_b_available);

Signature signature_of(const Config& c);

}  // namespace detail
}  // namespace tilewright
