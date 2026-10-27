// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "tilewright/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tilewright {

class Model;
using ModelPtr = std::shared_ptr<const Model>;

// Where an MLREC_v3 model came from, as its file records it; a field the file
// does not record is empty.
struct Provenance {
  std::string config_id;           // the training configuration
  std::string training_data_hash;  // the data the model was trained on
  std::string source_revision;     // the training pipeline's sources
  std::string date;                // when the model was trained (ISO 8601)
  std::string launch_model;        // LaunchModel::id() of the plans it trained with
};

struct ModelInfo {
  std::string arch;
  std::string feature_catalog_hash;
  WeightType weight_type = WeightType::Fp32;
  std::size_t n_cells    = 0;
  std::size_t n_splits   = 0;

  int format_version              = 2;  // MLREC_v<format_version>
  std::uint64_t feature_spec_hash = 0;  // of the item-attribute spec (MLREC_v3)
  std::size_t query_dim = 0, item_dim = 0, inter_dim = 0;
  // Supported contexts: usable CUs as a fraction of the device's, and a bit
  // per Schedule value.
  double min_cu_fraction = 1.0;
  double max_cu_fraction = 1.0;
  std::uint8_t schedules = 1;
  // MLREC_v3: the fewest usable CUs the model supports on any device (its
  // CU floor), 0 for a model without one.
  std::size_t min_effective_cus = 0;
  // Serving options.
  std::size_t cascade_keys = 0;
  bool whitelist_tier      = true;
  bool launch_keys         = false;  // the pool is grouped by launch key (MLREC_v3)
  bool small_batched_rule  = true;
  bool dot2_rule           = true;
  bool non_temporal_rule   = true;
  // MLREC_v3: batched problems with K <= 512 that are thin along exactly one
  // of M and N (M <= 32 < N, or 32 < M <= 128 with N <= 32) keep one signature
  // of each kernel family (equal mt and mi) the non-temporal rule rejects
  // whole: its hint-free one, else the one with the lowest (cache_hints_a,
  // cache_hints_b), with every kernel of it whatever its attributes or launch
  // key.
  bool non_temporal_family_rule = false;
  bool clip                     = false;
  // MLREC_v3: the model was trained on the integration's bank as well as its
  // table, so an integration with a bank serves both. Ranking does not read it.
  bool bank = false;
  // MLREC_v3: only kernels whose view key had training data (the file lists
  // `trained_keys` of them) are scored, unless none of them is feasible for
  // the problem; then every feasible kernel is (CandidateSet::untrained_keys).
  // What the non-temporal rule reads of the pool still counts every kernel.
  bool trained_key_gate    = false;
  std::size_t trained_keys = 0;
  // Base grid labels with no node in the split tree. An MLREC_v3 model scores
  // their problems through the nearest base label that has a trained cell.
  std::size_t uncovered_base_labels = 0;
  // Host memory of the weights: the packed form, held while the model lives,
  // and the fp32 copies of every cell, of which the engine holds those of the
  // cells in use within the cache limit (set_cache_limit). The matrices of the
  // query and item towers are held at the stored precision instead (int4 as
  // int8).
  std::size_t packed_bytes   = 0;
  std::size_t expanded_bytes = 0;
  Provenance provenance;
};

// Feature vectors the scorer computes for one (problem, config) pair, before
// whitening. compute_features returns empty vectors for invalid hardware.
struct Features {
  std::vector<float> query;
  std::vector<float> item;
  std::vector<float> interaction;
};

// Hash of the feature catalog of MLREC_v2 files.
const char* feature_catalog_hash() noexcept;

// Hashes of every feature catalog this engine computes (MLREC_v2, then
// MLREC_v3). Model files carrying another hash are rejected at load.
std::vector<std::string> feature_catalog_hashes();

// Loads a model file. Returns nullptr on any error and stores the reason in
// *error when error is non-null. Loading a path that resolves to an already
// loaded file returns the same model.
ModelPtr load_model(const std::string& path, std::string* error = nullptr) noexcept;

// Loads a model from an in-memory image of a model file. Not cached.
ModelPtr load_model_from_memory(const void* data,
                                std::size_t size,
                                std::string* error = nullptr) noexcept;

// Looks up `logic_stem` in `<dir>/tilewright_index` and loads the model of
// the stem's first row without a PCI chip ID. Returns nullptr with *error left
// empty when the index file or such a row is absent, and nullptr with *error
// set when the listed model fails to load or a row of the stem is malformed.
ModelPtr load_model_by_index(const std::string& logic_stem,
                             const std::string& dir,
                             std::string* error = nullptr) noexcept;

// As above for a device whose PCI chip ID is `pci_chip_id` (0: unknown): the
// stem's first row naming that chip ID is preferred to its first row without
// one. A plain literal 0 as the third argument is ambiguous with the overload
// above; pass a std::uint32_t (a variable, 0u).
ModelPtr load_model_by_index(const std::string& logic_stem,
                             const std::string& dir,
                             std::uint32_t pci_chip_id,
                             std::string* error = nullptr) noexcept;

ModelInfo describe(const Model& model);

// Index of the trained cell that scores `problem` (the split-tree leaf, or its
// nearest trained ancestor), or -1 when there is none. When the problem's base
// label has neither, an MLREC_v3 model routes through the nearest base label
// that has one (summed tier distance over M, N, K and batch), so it routes
// every problem.
int route(const Model& model, const Problem& problem) noexcept;

// Label of cell `cell` (as returned by route), or an empty string.
std::string cell_label(const Model& model, int cell);

// For exclusive use of the device (a default ExecutionContext).
Features compute_features(const Model& model,
                          const Problem& problem,
                          const Config& config,
                          const Hardware& hardware);

Features compute_features(const Model& model,
                          const Problem& problem,
                          const Config& config,
                          const Hardware& hardware,
                          const ExecutionContext& context);

// For a config launched as `plan` (MLREC_v2 features ignore it); the overload
// above uses LaunchPlan{}.
Features compute_features(const Model& model,
                          const Problem& problem,
                          const Config& config,
                          const Hardware& hardware,
                          const ExecutionContext& context,
                          const LaunchPlan& plan);

// True when the model was trained for `context` on `hardware`. v2 models cover
// only every CU (cu_budget 0 or at least hardware.N_CU) with Schedule::Default.
// v3 models cover the schedules they list and the CU budgets whose usable
// CUs, as a fraction of hardware.N_CU, lie in their range and, for a model
// with a CU floor, number at least that floor (ModelInfo).
bool supports(const Model& model,
              const ExecutionContext& context,
              const Hardware& hardware) noexcept;

// Names of the Config attributes the model reads, so callers can leave out
// the others (v2 models read none). A config without one of them is ranked
// as if it had the model's default value.
std::vector<std::string> attribute_names(const Model& model);

// The data rankings build on first use and can build again: the fp32 weights
// of a model's cells and a CandidateSet's per-cell whitelist membership and
// item embeddings. Over every model and candidate set of the process it stays
// in memory while its total is within the cache limit; past it, the least
// recently used parts are released and rebuilt when next needed, with the
// same results. The limit is TILEWRIGHT_CACHE_MB MiB when that is set, else
// 1 GiB. A part in use is kept even when it alone exceeds the limit.
void set_cache_limit(std::size_t bytes) noexcept;
std::size_t cache_limit() noexcept;
// Bytes the cache holds now.
std::size_t cache_usage() noexcept;

// Zeroed memory for data that a ranking reads on every call, such as what a
// LaunchModel's plan() reads of each config: at least `bytes`, 64-byte
// aligned. On Linux it is whole 2 MiB pages at a 2 MiB boundary, backed by
// transparent huge pages unless the system has them off. Throws
// std::bad_alloc when the memory cannot be had. free_huge_pages takes it back
// given the same `bytes`.
void* allocate_huge_pages(std::size_t bytes);
void free_huge_pages(void* p, std::size_t bytes) noexcept;

// A caller's fixed pool of candidate kernels, ranked against one model.
// Per-cell data that depends only on the pool is computed once and reused
// within the cache limit (set_cache_limit): whitelist membership and the item
// embeddings of whitelisted configs on the first rank call that reaches the
// cell, the item embeddings of the other configs on the first rank call that
// scores one of them. Configs the model's feature catalog cannot tell apart
// share that data and their score; for an MLREC_v3 model those are configs
// with equal signature and spec attributes, and equal PoolOptions::launch
// launch key when the model serves launch keys (ModelInfo::launch_keys),
// launched as its plan for the first of them in pool order. Safe to share
// between threads.
class CandidateSet {
 public:
  // Throws std::invalid_argument for a null model, for a config with an
  // empty or duplicate attribute name, or when the model's provenance names
  // a launch model (Provenance::launch_model) and options.launch is null or
  // has another id().
  CandidateSet(ModelPtr model, std::vector<Config> configs, PoolOptions options = {});
  ~CandidateSet();

  CandidateSet(const CandidateSet&)            = delete;
  CandidateSet& operator=(const CandidateSet&) = delete;

  const Model& model() const noexcept;
  const std::vector<Config>& configs() const noexcept;

  // The groups of configs a model with the trained-key gate
  // (ModelInfo::trained_key_gate) leaves out while a trained one is feasible,
  // and their configs; 0 for any other model.
  std::size_t untrained_keys() const noexcept;
  std::size_t untrained_configs() const noexcept;

  // Ranks for exclusive use of the device (a default ExecutionContext).
  // Returns one Result per config. Scored configs come first: survivors of the
  // cell's whitelist (every feasible config when the whitelist keeps none)
  // best-first, followed (only when min_scored exceeds their count) by the
  // remaining feasible configs best-first; ties keep input order. Unscored
  // configs, including those whose score is not finite, follow in input order.
  // Invalid hardware (a zero field) or a problem with no trained cell returns
  // every config unscored.
  std::vector<Result> rank(const Problem& problem,
                           const Hardware& hardware,
                           std::size_t min_scored = 0) const;

  // As above, for `context`: every config is unscored unless
  // supports(model(), context, hardware). With TieBreak::Prior, configs with
  // equal scores are ordered by ascending (*prior)[config position],
  // non-finite last, then by input order; with TieBreak::PoolOrder or a null
  // prior, ties keep input order. A non-null prior must hold one value per
  // config (std::invalid_argument otherwise).
  std::vector<Result> rank(const Problem& problem,
                           const Hardware& hardware,
                           const ExecutionContext& context,
                           std::size_t min_scored           = 0,
                           const std::vector<double>* prior = nullptr) const;

  // The best k configs: the scored results among the first k of
  // rank(problem, hardware, context, k, prior), so fewer than k when fewer
  // are scored. Only these results are ordered and written.
  std::vector<Result> rank_top(const Problem& problem,
                               const Hardware& hardware,
                               const ExecutionContext& context,
                               std::size_t k,
                               const std::vector<double>* prior = nullptr) const;

  // As above, replacing the contents of `out`, whose capacity is reused.
  void rank_top(const Problem& problem,
                const Hardware& hardware,
                const ExecutionContext& context,
                std::size_t k,
                std::vector<Result>& out,
                const std::vector<double>* prior = nullptr) const;

  // Builds now what rankings would build on first use, so that a caller can
  // pay for it at load time: for each cell of `cells` (indices as route()
  // returns them; other values are ignored), its fp32 weights, its whitelist
  // membership and the item embeddings of every config. What it builds is
  // held within the cache limit like the rest, so with a limit below the
  // total the cells prepared first may be released again.
  void prepare(const std::vector<int>& cells) const;

  // As prepare(cells) with every cell of the model, then a copy of what
  // rankings of a cell's whitelist read (every cell's whitelist and tensors,
  // the pool's arrays and, once built, the kernel part of the interaction
  // features) in memory of allocate_huge_pages, which later rankings read.
  // The copy is held as long as the CandidateSet, outside the cache limit;
  // without the memory for it, rankings read the data in place.
  void prepare() const;

  // As prepare(cells) with the cell `problem` routes to, plus the interaction
  // features that depend only on the kernel for problems with the data types
  // of `problem` on `hardware`.
  void prepare(const Problem& problem, const Hardware& hardware) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Same result as CandidateSet::rank for an ad-hoc config list, without caching.
// Throws std::invalid_argument for a config with an empty or duplicate
// attribute name.
std::vector<Result> rank_configs(const Model& model,
                                 const Problem& problem,
                                 const Hardware& hardware,
                                 const std::vector<Config>& configs,
                                 std::size_t min_scored = 0);

// Same result as a CandidateSet of `configs` with PoolOptions{tie_break}
// ranked with these arguments, without caching.
std::vector<Result> rank_configs(const Model& model,
                                 const Problem& problem,
                                 const Hardware& hardware,
                                 const ExecutionContext& context,
                                 const std::vector<Config>& configs,
                                 std::size_t min_scored           = 0,
                                 TieBreak tie_break               = TieBreak::PoolOrder,
                                 const std::vector<double>* prior = nullptr);

// Same result as a CandidateSet of `configs` with `options` ranked with these
// arguments, without caching.
std::vector<Result> rank_configs(const Model& model,
                                 const Problem& problem,
                                 const Hardware& hardware,
                                 const ExecutionContext& context,
                                 const std::vector<Config>& configs,
                                 const PoolOptions& options,
                                 std::size_t min_scored           = 0,
                                 const std::vector<double>* prior = nullptr);

}  // namespace tilewright
