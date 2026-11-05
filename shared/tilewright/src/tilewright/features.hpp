// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "model_impl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace tilewright {
namespace detail {

// Device values from the caller and the execution context, plus the arch
// constants the model was trained with.
struct HwView {
  double N_CU = 0.0;
  // CUs the kernels may use: ExecutionContext::cu_budget when it is below
  // N_CU and nonzero, else N_CU. Equal to N_CU for MLREC_v2 models.
  double N_eff          = 0.0;
  double LDS            = 0.0;
  double L2             = 0.0;
  double parallel_mi_cu = 1.0;
  double c0             = 0.0;
  double c1             = 0.0;
  double c2             = 0.0;
  Schedule schedule     = Schedule::Default;
};

HwView hw_view(const Model& model, const Hardware& hardware, const ExecutionContext& context);

// The usable CUs of `context` on `hardware`, as HwView::N_eff.
std::size_t effective_cus(const ExecutionContext& context, const Hardware& hardware) noexcept;

// An MLREC_v3 view key: the signature (signature_of), then the attributes
// of the model's spec, at most 64.
constexpr std::size_t kV3ViewKeyFields   = 8;
constexpr std::size_t kMaxAttributeReads = 64;
constexpr std::size_t kMaxViewKeySize    = 12 + kMaxAttributeReads;

// The features a model was trained on, selected by the hash in its file.
// Feature math runs in double and rounds once to float, as the training
// pipeline does (float64 arithmetic, then a float32 cast). `attrs` holds the
// config's values of Model::attribute_reads; `plan` is the config's launch
// (catalogs with `reads_launch` only).
struct FeatureCatalog {
  const char* hash;
  int format_version;
  std::size_t query_dim;
  // Without the model's attribute features (Model::item_dim has them).
  std::size_t item_dim;
  std::size_t inter_dim;
  bool reads_launch;
  // Configs with equal view keys are indistinguishable to the catalog and to
  // the ranking rules: they get identical features, feasibility, whitelist
  // membership and scores. Model::view_key_size values.
  void (*view_key)(const Model& m, const Config& c, const std::int64_t* attrs, std::uint64_t* out);
  // The execution contexts the model declares (tilewright::supports).
  bool (*supports)(const Model& m, const ExecutionContext& context, const Hardware& hardware);
  // The context a ranking for `context` is scored at (tilewright::scoring_context):
  // `context` when `supports` holds; for a context outside it only by its CU
  // budget, the same schedule with the usable CUs the model supports nearest
  // to the context's or, on a device where it supports none, the most its
  // largest fraction allows; std::nullopt otherwise.
  std::optional<ExecutionContext> (*scoring_context)(const Model& m,
                                                     const ExecutionContext& context,
                                                     const Hardware& hardware);
  void (*query)(const Problem& p, const HwView& hw, float* out);
  void (*item)(const Model& m, const Config& c, const std::int64_t* attrs, float* out);
  void (*inter)(const Model& m,
                const Problem& p,
                const Config& c,
                const std::int64_t* attrs,
                const HwView& hw,
                const LaunchPlan& plan,
                float* out);
  // `inter` in parts, for callers that reuse values across problems:
  // inter_kernel writes inter_kernel_dim values that depend on the config, the
  // problem's data types, the device's LDS and the model; inter_problem writes
  // inter_problem_dim values that depend on the problem, the device and the
  // context; inter_pair then writes what `inter` writes.
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
                     const std::int64_t* attrs,
                     const Model& m,
                     const HwView& hw,
                     const float* kernel,
                     const float* problem,
                     const LaunchPlan& plan,
                     float* out);
};

// static_cast<float>(std::log2(std::max(x, 1.0))), from a table for the
// integers it holds.
float count_log2(double x);

// The catalog whose hash is `hash`, or nullptr.
const FeatureCatalog* find_catalog(const std::string& hash) noexcept;

// The catalog of MLREC_v2 files.
const FeatureCatalog& v2_catalog() noexcept;

// The base catalog of MLREC_v3 files (before a model's attribute features).
const FeatureCatalog& v3_catalog() noexcept;

// Number of item features `f` adds.
std::size_t attribute_feature_count(const AttributeFeature& f) noexcept;

// The feature_spec_hash of an MLREC_v3 file with item-attribute spec `spec`:
// FNV-1a 64 of feature_spec_text, which holds the v3 catalog's hash and
// version, the names of its features with the spec's, the spec, and the
// features of fixed golden inputs (features.cpp), so that the hash changes
// when any of them does.
std::string feature_spec_text(const std::vector<AttributeFeature>& spec);
std::uint64_t feature_spec_hash(const std::vector<AttributeFeature>& spec);

// FNV-1a 64 of a view key's values, each as 8 little-endian bytes: the hash
// MLREC_v3 whitelists list.
std::uint64_t view_key_hash(const std::uint64_t* key, std::size_t n) noexcept;

// Raw MI cycles for (mi_m, mi_n, mi_k, dtype) from the model's table, with fnuz
// dtypes looked up as their non-fnuz counterparts.
double mi_cycles(const Model& model,
                 std::size_t mi_m,
                 std::size_t mi_n,
                 std::size_t mi_k,
                 DataType dt);

// What the non-temporal rule reads of a kernel's cache hints: 0 for none,
// else 1, plus 2 when A's is the non-temporal one (4), plus 4 when B's is.
inline unsigned hint_class(int hints_a, int hints_b) {
  if (hints_a == 0 && hints_b == 0) return 0;
  return 1u | (hints_a == 4 ? 2u : 0u) | (hints_b == 4 ? 4u : 0u);
}

// The extremes of a set of kernels, for Feasibility::all_pass (kernels
// without a cache hint) and all_but_non_temporal_pass.
struct PlainBounds {
  std::uint64_t mk = 0, nk = 0, grvw_a = 0, grvw_b = 0;
  std::uint64_t mt_m = ~std::uint64_t{0}, mt_n = ~std::uint64_t{0};
  bool dot2 = false;

  void include(const KernelShape& k) {
    mk     = std::max(mk, k.mk);
    nk     = std::max(nk, k.nk);
    grvw_a = std::max(grvw_a, k.grvw_a);
    grvw_b = std::max(grvw_b, k.grvw_b);
    mt_m   = std::min(mt_m, k.mt_m);
    mt_n   = std::min(mt_n, k.mt_n);
    dot2   = dot2 || k.dot2;
  }
};

// Where the family-wise non-temporal rule applies: batched problems with
// K <= 512 that are thin along exactly one of M and N (M <= 32 < N, or
// 32 < M <= 128 with N <= 32). The bounds are the routing grid's tier edges.
inline bool in_family_region(std::uint64_t m,
                             std::uint64_t n,
                             std::uint64_t k,
                             std::uint64_t batch) {
  return batch > 1 && k <= 512 && ((m <= 32 && n > 32) || (m > 32 && m <= 128 && n <= 32));
}

// The feasibility rules for one problem and device, with the values that
// depend only on them computed once. `nt_a_available` / `nt_b_available`: the
// candidate pool contains a kernel with cache_hints_a / cache_hints_b == 4.
// The LDS fit always applies; `soft` selects the others, and `veto` is the
// model's alignment veto, a serving rule for rankings only.
class Feasibility {
 public:
  Feasibility(const Problem& p,
              const Hardware& hardware,
              bool nt_a_available,
              bool nt_b_available,
              const SoftRules& soft     = SoftRules{},
              const AlignmentVeto& veto = AlignmentVeto{});

  // Validity: the tile's A and B slices fit in LDS. A kernel whose LDS
  // allocation exceeds the device's cannot launch, and TensileLite rejects
  // kernels whose A and B LDS buffers exceed the device LDS when it builds
  // them.
  bool lds_fits(const KernelShape& k) const {
    return lds_valid_ &&
           static_cast<double>(k.mk) * bytes_a_ + static_cast<double>(k.nk) * bytes_b_ <=
               lds_capacity_;
  }

  bool rules_pass(const KernelShape& k) const {
    // Validity: an operand whose free dimension is contiguous is read in
    // global-read vectors, so that dimension must hold one (TensileLite's
    // LeadingFree0/1SizesGreaterOrEqual; not emitted for Int8x4).
    if (leading_free_a_ && M_ < k.grvw_a) return false;
    if (leading_free_b_ && N_ < k.grvw_b) return false;
    // Soft: a small batched problem uses a tile that covers it.
    if (small_batched_ && (k.mt_m < M_ || k.mt_n < N_)) return false;
    // Soft: Dot2 kernels serve only M < 3.
    if (dot2_rule_ && k.dot2 && M_ > 2) return false;
    const unsigned hints = hint_class(k.cache_hints_a, k.cache_hints_b);
    if (vetoed(hints)) return false;
    // Soft: on 128-byte-aligned K a skinny problem requires the matching
    // operand's non-temporal variant when the pool has one, and every other
    // aligned problem rejects hinted kernels.
    return non_temporal_rule(
        tile_aligned(k.mt_k % 1024), b_branch(k.mt_m), a_branch(k.mt_n), hints, k.family);
  }

  // Whether the alignment veto applies to this problem, and whether it
  // rejects a kernel of hint_class `hints`: one whose cache hint on the
  // operand the veto reads is the non-temporal one.
  bool vetoes() const { return veto_ != 0; }
  bool vetoed(unsigned hints) const { return (hints & veto_) != 0; }

  // The rules without the alignment veto, for a ranking it would leave
  // without a kernel to rank.
  void drop_veto() { veto_ = 0; }

  // The non-temporal rule from its tests of a kernel: whether its tile keeps
  // K aligned (tile_aligned), whether the B and A branches apply, its
  // hint_class and its family bits (kFamily*).
  bool non_temporal_rule(bool aligned,
                         bool by_b,
                         bool by_a,
                         unsigned hints,
                         unsigned family = 0) const {
    if (!non_temporal_rule_) return true;
    bool pass = hints == 0;
    if (aligned && by_b)
      pass = !(nt_b_ && (hints & 4) == 0);
    else if (aligned && by_a)
      pass = !(nt_a_ && (hints & 2) == 0);
    return pass || ((family & kFamilyFallback) != 0 && family_keeps(aligned, by_b, by_a, family));
  }

  // Whether the family-wise rule passes a family's kept member here: the
  // problem is in its region and the non-temporal rule passes no member of
  // the family. The tests are the family's, which its members share. Out of
  // line: the rule is inlined wherever a ranking tests a key, and only
  // problems in the region get this far.
  bool family_keeps(bool aligned, bool by_b, bool by_a, unsigned family) const;

  // family_keeps for every value of its tests and of a family's members, at
  // aligned | by_b << 1 | by_a << 2 | (family & 7) << 3.
  void family_keep_table(unsigned char keep[64]) const;

  // Whether the family-wise rule applies to this problem.
  bool family_region() const { return family_; }
  bool tile_aligned(std::uint64_t mt_k_low) const {
    return k_aligned_ && mt_k_low * a_bits_ % 1024 == 0;
  }
  bool b_branch(std::uint64_t mt_m) const { return skinny_b_ && at_most_twice(M_, mt_m); }
  bool a_branch(std::uint64_t mt_n) const { return skinny_a_ && at_most_twice(N_, mt_n); }

  bool operator()(const KernelShape& k) const { return lds_fits(k) && rules_pass(k); }

  // Whether rules_pass may accept a kernel with a cache hint.
  bool hinted_may_pass() const {
    return !non_temporal_rule_ || family_ || (k_aligned_ && (skinny_a_ || skinny_b_));
  }

  // Whether the only kernels with a cache hint rules_pass may accept are the
  // kept members of families without a hint-free member.
  bool only_kept_hinted_may_pass() const {
    return family_ && !(k_aligned_ && (skinny_a_ || skinny_b_));
  }

  // Whether every kernel without a cache hint within `b` is feasible. Each
  // rule's test is monotone in the kernel's values, so the extremes decide.
  bool all_pass(const PlainBounds& b) const {
    return all_but_non_temporal_pass(b) && (!non_temporal_rule_ || !k_aligned_ ||
                                            ((!skinny_b_ || !nt_b_) && (!skinny_a_ || !nt_a_)));
  }

  // Whether every kernel within `b`, with or without a cache hint, passes
  // the rules other than the non-temporal one (all_pass without it).
  bool all_but_non_temporal_pass(const PlainBounds& b) const {
    if (!lds_valid_ ||
        static_cast<double>(b.mk) * bytes_a_ + static_cast<double>(b.nk) * bytes_b_ > lds_capacity_)
      return false;
    if (leading_free_a_ && M_ < b.grvw_a) return false;
    if (leading_free_b_ && N_ < b.grvw_b) return false;
    if (small_batched_ && (b.mt_m < M_ || b.mt_n < N_)) return false;
    return !(dot2_rule_ && b.dot2 && M_ > 2);
  }

 private:
  // v <= 2 * t without overflow.
  static bool at_most_twice(std::uint64_t v, std::uint64_t t) { return (v >> 1) + (v & 1) <= t; }

  std::uint64_t M_ = 0, N_ = 0, a_bits_ = 0;
  // The hint_class bit of the kernels the alignment veto rejects: 2 (A's
  // cache hint), 4 (B's), or 0 where the veto does not apply.
  unsigned veto_  = 0;
  double bytes_a_ = 0.0, bytes_b_ = 0.0, lds_capacity_ = 0.0;
  bool lds_valid_ = false, small_batched_ = false, dot2_rule_ = true, leading_free_a_ = false,
       leading_free_b_ = false, non_temporal_rule_ = true, k_aligned_ = false, skinny_a_ = false,
       skinny_b_ = false, nt_a_ = false, nt_b_ = false, family_ = false;
};

// Whether Feasibility rejects `k` for every problem within `bounds` whatever
// its data types, layout and device, by the rules decided by M, N, K and the
// batch alone: the non-temporal (with KernelShape::family), Dot2 and small
// batched rules. False when `bounds` reaches no problem.
bool never_feasible(const CellBounds& bounds,
                    const KernelShape& k,
                    bool nt_a_available,
                    bool nt_b_available,
                    const SoftRules& soft);

bool check_lds_capacity(const Hardware& hardware, const Dim3& mt, DataType a, DataType b);

bool is_kernel_feasible(const Problem& p,
                        const Config& c,
                        bool nt_a_available,
                        bool nt_b_available,
                        bool non_temporal_rule = true);

bool is_kernel_feasible(const Problem& p,
                        const Config& c,
                        bool nt_a_available,
                        bool nt_b_available,
                        const SoftRules& soft);

Signature signature_of(const Config& c);

}  // namespace detail
}  // namespace tilewright
