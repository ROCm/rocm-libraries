// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// =============================================================================
// mosaic -- framework-agnostic neutral types
// =============================================================================
//
// These types are mosaic's OWN, deliberately self-contained: the mosaic
// kernel-recommender engine includes NO framework headers (no origami, no
// hipBLASLt). Each struct/enum below carries ONLY the fields the engine and
// its feature-extraction actually consume.
//
// The enum value ORDER intentionally mirrors origami's `data_type_t`,
// `transpose_t`, and `prediction_modes_t` so that a framework adapter can
// convert with a plain static_cast (and so the engine's per-enum switch logic
// stays byte-identical to the reconstructed origami runtime).
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace mosaic {

// Mirror of origami::data_type_t (same declaration order -> same integer
// values, so the adapter may static_cast across).
enum class DataType : int {
  Float,
  Double,
  ComplexFloat,
  ComplexDouble,
  Half,
  Int8x4,
  Int32,
  BFloat16,
  Int8,
  Int4,
  Int64,
  XFloat32,
  Float8_fnuz,
  BFloat8_fnuz,
  Float8BFloat8_fnuz,
  BFloat8Float8_fnuz,
  Float8,
  BFloat8,
  Float8BFloat8,
  BFloat8Float8,
  Float6,
  BFloat6,
  Float4,
  Count,
  None = Count
};

// Mirror of origami::transpose_t.
enum class Transpose { T, N, Count };

// Mirror of origami::prediction_modes_t (used only so a Config can carry the
// mode through; the mosaic ML ranker itself does not branch on it).
enum class PredictionMode : std::uint32_t {
  estimation     = 0,
  simulation     = 1,
  ml_recommender = 2,
  count,
  none = 0xFFFFFFFFu
};

// Compact (M, N, K) triple. Mirrors origami::dim3_t for the fields the engine
// reads, plus the mk()/nk() helpers used by the LDS-capacity gate.
struct Dim3 {
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;

  constexpr std::size_t mn() const noexcept { return m * n; }
  constexpr std::size_t mk() const noexcept { return m * k; }
  constexpr std::size_t nk() const noexcept { return n * k; }
};

// GEMM problem description. Carries exactly the fields the feature builders,
// routing and feasibility filter read: size, batch, the A/B transposes, and
// the A/B/C/D/compute data types.
struct Problem {
  Dim3 size{0, 0, 0};
  std::size_t batch = 1;

  Transpose a_transpose = Transpose::N;
  Transpose b_transpose = Transpose::N;

  DataType a_dtype  = DataType::None;
  DataType b_dtype  = DataType::None;
  DataType c_dtype  = DataType::None;
  DataType d_dtype  = DataType::None;
  DataType mi_dtype = DataType::None;
};

// Kernel configuration. Holds the tile/MI shape and the scalar knobs the item
// + interaction feature builders read (occupancy, cache hints, vector widths),
// plus a few Tensile-derived knobs and bookkeeping fields the adapter forwards
// for completeness. (The mosaic engine itself reads: mt, mi, occupancy,
// cache_hints_a/b, grvw_a/b, gwvw_d.)
struct Config {
  Dim3 mt{0, 0, 0};
  Dim3 mi{0, 0, 0};

  int occupancy = -1;

  int cache_hints_a = 0;
  int cache_hints_b = 0;

  std::size_t grvw_a = 1;
  std::size_t grvw_b = 1;
  std::size_t gwvw_d = 1;

  int vector_width_a = 1;
  int vector_width_b = 1;

  // Tensile-derived knobs (carried through; not read by the ML scorer).
  std::size_t depth_u       = 0;
  std::int16_t global_split_u = 1;
  int local_split_u         = 1;
  int prefetch_global_read  = 2;

  std::size_t index = 0;

  PredictionMode prediction_mode = PredictionMode::estimation;
};

// Optional per-config ML feature fields. 1:1 mirror of origami::config_ml_t.
struct ConfigML {
  int cache_hints_c         = 0;
  int cache_hints_d         = 0;
  int cache_hints_e         = 0;
  int prefetch_global_read  = 1;
  int prefetch_local_read   = 1;
  int lds_read_vector_width = 1;
  int local_split_u         = 1;
  int lds_pad_a             = 0;
  int lds_pad_b             = 0;
  int lds_buffer_pad_a      = 0;
  int lds_buffer_pad_b      = 0;
};

// Hardware characteristics the feature builders and the LDS gate read.
struct Hardware {
  std::size_t N_CU          = 0;
  std::size_t lds_capacity  = 0;
  std::size_t L2_capacity   = 0;
  std::size_t parallel_mi_cu = 1;
  // mem_bw_per_wg coefficients (c0, c1, c2).
  std::tuple<double, double, double> mem_bw_per_wg_coefficients{0.0, 0.0, 0.0};
};

// Ranking result for a single input config.
//   config_index : index into the input `configs` vector.
//   score        : the raw two-tower score (dot(qe,ie)/T + inter_mlp); higher
//                  is better. Meaningful only when `scored` is true.
//   scored       : true if the config survived the LDS gate, feasibility
//                  filter and (optional) smart-K signature filter and was
//                  scored; false if it was filtered out (caller should treat
//                  its latency as NaN).
struct Result {
  std::size_t config_index = 0;
  double score             = 0.0;
  bool scored              = false;
};

}  // namespace mosaic
