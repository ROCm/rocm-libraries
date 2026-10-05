// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>

namespace tilewright {

// Element data type. The integer values are stored in model files (MI-latency
// table keys), so members must never be reordered or inserted.
enum class DataType : int {
  Float              = 0,
  Double             = 1,
  ComplexFloat       = 2,
  ComplexDouble      = 3,
  Half               = 4,
  Int8x4             = 5,
  Int32              = 6,
  BFloat16           = 7,
  Int8               = 8,
  Int4               = 9,
  Int64              = 10,
  XFloat32           = 11,
  Float8_fnuz        = 12,
  BFloat8_fnuz       = 13,
  Float8BFloat8_fnuz = 14,
  BFloat8Float8_fnuz = 15,
  Float8             = 16,
  BFloat8            = 17,
  Float8BFloat8      = 18,
  BFloat8Float8      = 19,
  Float6             = 20,
  BFloat6            = 21,
  Float4             = 22,
  Count              = 23,
  None               = Count
};

enum class Transpose : int { T = 0, N = 1, Count = 2 };

// Storage type of a model's weight matrices.
enum class WeightType : std::uint8_t { Fp32 = 0, Bf16 = 1, Int8 = 2, Int4 = 3 };

struct Dim3 {
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;

  constexpr std::size_t mn() const noexcept { return m * n; }
  constexpr std::size_t mk() const noexcept { return m * k; }
  constexpr std::size_t nk() const noexcept { return n * k; }
};

// A GEMM problem: D = op(A) * op(B) with A MxK, B KxN, batched `batch` times.
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

// A candidate kernel. `index` is the caller's identifier for the kernel and is
// not interpreted by the engine.
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

// Device properties queried at runtime. Constants that a model was trained
// with (matrix-instruction latencies, bandwidth model) live in the model file.
struct Hardware {
  std::size_t N_CU         = 0;
  std::size_t lds_capacity = 0;  // bytes
  std::size_t L2_capacity  = 0;  // bytes
};

// Ranking result for one input config, which is configs[config_index] of the
// ranked list. `score` is meaningful only when `scored` is true; higher is better.
struct Result {
  std::size_t config_index = 0;
  double score             = 0.0;
  bool scored              = false;
};

}  // namespace tilewright
