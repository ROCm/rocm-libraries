// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// =============================================================================
// tilewright -- framework-agnostic neutral types
// =============================================================================
//
// These types are tilewright's OWN, deliberately self-contained: the
// kernel-recommender engine includes NO GEMM-framework headers. Each
// struct/enum below carries ONLY the fields the engine and its
// feature-extraction actually consume.
//
// The enum value ORDER is fixed so a framework adapter (e.g. the hipBLASLt
// backend) can convert its own GEMM dtype/transpose/prediction-mode enums with
// a plain static_cast, keeping the engine's per-enum switch logic stable.
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace tilewright {

// GEMM element data type. Fixed declaration order -> fixed integer values, so a
// framework adapter may static_cast its own dtype enum across.
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

// Matrix transpose flag.
enum class Transpose { T, N, Count };

// Compact (M, N, K) triple, plus the mk()/nk() helpers used by the
// LDS-capacity gate.
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

// A candidate kernel configuration. The scorer reads mt, mi, occupancy,
// cache_hints_a/b, and grvw_a/b/gwvw_d; index identifies the kernel to the host.
struct Config {
  Dim3 mt{0, 0, 0};
  Dim3 mi{0, 0, 0};

  int occupancy = -1;

  int cache_hints_a = 0;
  int cache_hints_b = 0;

  std::size_t grvw_a = 1;
  std::size_t grvw_b = 1;
  std::size_t gwvw_d = 1;

  std::size_t index = 0;
};

// Hardware characteristics the feature builders and the LDS gate read.
struct Hardware {
  std::size_t N_CU           = 0;
  std::size_t lds_capacity   = 0;
  std::size_t L2_capacity    = 0;
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

}  // namespace tilewright
