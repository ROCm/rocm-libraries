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
  // `inter` in parts, for callers that reuse values across problems:
  // inter_kernel writes inter_kernel_dim values that depend on the config, the
  // problem's data types, the device and the model; inter_problem writes
  // inter_problem_dim values that depend on the problem and the device;
  // inter_pair then writes what `inter` writes.
  std::size_t inter_kernel_dim;
  std::size_t inter_problem_dim;
  void (*inter_kernel)(const Model& m,
                       const Problem& p,
                       const Config& c,
                       const HwView& hw,
                       float* out);
  void (*inter_problem)(const Problem& p, const HwView& hw, float* out);
  void (*inter_pair)(const Problem& p,
                     const Config& c,
                     const HwView& hw,
                     const float* kernel,
                     const float* problem,
                     float* out);
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

// The feasibility rules for one problem and device, with the values that
// depend only on them computed once. `nt_a_available` / `nt_b_available`: the
// candidate pool contains a kernel with cache_hints_a / cache_hints_b == 4.
class Feasibility {
 public:
  Feasibility(const Problem& p, const Hardware& hardware, bool nt_a_available, bool nt_b_available);

  // The tile's A and B slices fit in LDS.
  bool lds_fits(const KernelShape& k) const {
    return lds_valid_ &&
           static_cast<double>(k.mk) * bytes_a_ + static_cast<double>(k.nk) * bytes_b_ <=
               lds_capacity_;
  }

  bool rules_pass(const KernelShape& k) const {
    // 1) A small batched problem must fit in one tile.
    if (small_batched_ && (k.mt_m < M_ || k.mt_n < N_)) return false;
    // 2) Dot2 is only correct for M < 3.
    if (k.dot2 && M_ > 2) return false;
    // 3) An operand whose free dimension is contiguous is read in global-read
    //    vectors, so that dimension must hold one (TensileLite's
    //    LeadingFree0/1SizesGreaterOrEqual; not emitted for Int8x4).
    if (leading_free_a_ && M_ < k.grvw_a) return false;
    if (leading_free_b_ && N_ < k.grvw_b) return false;
    // 4) Non-temporal hints: on 128-byte-aligned K a skinny problem requires the
    //    matching operand's non-temporal variant when the pool has one, and
    //    every other aligned problem rejects hinted kernels.
    const bool hinted = k.cache_hints_a != 0 || k.cache_hints_b != 0;
    if (!k_aligned_ || (k.mt_k % 1024) * a_bits_ % 1024 != 0) return !hinted;
    if (skinny_b_ && at_most_twice(M_, k.mt_m)) return !(nt_b_ && k.cache_hints_b != 4);
    if (skinny_a_ && at_most_twice(N_, k.mt_n)) return !(nt_a_ && k.cache_hints_a != 4);
    return !hinted;
  }

  bool operator()(const KernelShape& k) const { return lds_fits(k) && rules_pass(k); }

 private:
  // v <= 2 * t without overflow.
  static bool at_most_twice(std::uint64_t v, std::uint64_t t) { return (v >> 1) + (v & 1) <= t; }

  std::uint64_t M_ = 0, N_ = 0, a_bits_ = 0;
  double bytes_a_ = 0.0, bytes_b_ = 0.0, lds_capacity_ = 0.0;
  bool lds_valid_ = false, small_batched_ = false, leading_free_a_ = false, leading_free_b_ = false,
       k_aligned_ = false, skinny_a_ = false, skinny_b_ = false, nt_a_ = false, nt_b_ = false;
};

bool check_lds_capacity(const Hardware& hardware, const Dim3& mt, DataType a, DataType b);

bool is_kernel_feasible(const Problem& p,
                        const Config& c,
                        bool nt_a_available,
                        bool nt_b_available);

Signature signature_of(const Config& c);

}  // namespace detail
}  // namespace tilewright
