// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/types.hpp"

#include <cmath>
#include <cstdlib>
#include <string>

namespace origami {

runtime_options::runtime_options() { update_from_env(); }

runtime_options::runtime_options(bool debug, bool heuristics, double variance)
    : debug_enabled(debug), heuristics_enabled(heuristics), heuristics_variance(variance) {}

bool runtime_options::read_debug_from_env() {
  const char* env = std::getenv("ANALYTICAL_GEMM_DEBUG");
  return env && std::string(env) == "1";
}

bool runtime_options::read_heuristics_from_env() {
  const char* env = std::getenv("ANALYTICAL_GEMM_HEURISTICS");
  return !(env && std::string(env) == "0");
}

double runtime_options::read_heuristics_variance_from_env() {
  constexpr double default_variance = 0.01;  // 1%

  if (const char* env = std::getenv("ANALYTICAL_GEMM_HEURISTICS_VARIANCE")) {
    try {
      double val = std::stod(env);
      if (std::isfinite(val) && val >= 0.0) { return val; }
    } catch (...) {
      // fall through to default
    }
  }
  return default_variance;
}

void runtime_options::update_from_env() {
  debug_enabled       = read_debug_from_env();
  heuristics_enabled  = read_heuristics_from_env();
  heuristics_variance = read_heuristics_variance_from_env();
}

int datatype_to_bits(data_type_t type) {
  switch (type) {
    case data_type_t::Float: return 32;
    case data_type_t::Double: return 64;
    case data_type_t::ComplexFloat: return 64;
    case data_type_t::ComplexDouble: return 128;
    case data_type_t::Half: return 16;
    case data_type_t::Int8x4: return 32;
    case data_type_t::Int32: return 32;
    case data_type_t::BFloat16: return 16;
    case data_type_t::Int8: return 8;
    case data_type_t::Int4: return 4;
    case data_type_t::Int64: return 64;
    case data_type_t::XFloat32: return 32;
    case data_type_t::Float8_fnuz: return 8;
    case data_type_t::BFloat8_fnuz: return 8;
    case data_type_t::Float8BFloat8_fnuz: return 8;
    case data_type_t::BFloat8Float8_fnuz: return 8;
    case data_type_t::Float8: return 8;
    case data_type_t::BFloat8: return 8;
    case data_type_t::Float8BFloat8: return 8;
    case data_type_t::BFloat8Float8: return 8;
    case data_type_t::Float6: return 6;
    case data_type_t::BFloat6: return 6;
    case data_type_t::Float4: return 4;
    default: return -1;  // Invalid type
  }
}

std::string datatype_to_string(data_type_t type) {
  switch (type) {
    case data_type_t::Float: return "Float";
    case data_type_t::Double: return "Double";
    case data_type_t::ComplexFloat: return "ComplexFloat";
    case data_type_t::ComplexDouble: return "ComplexDouble";
    case data_type_t::Half: return "Half";
    case data_type_t::Int8x4: return "Int8x4";
    case data_type_t::Int32: return "Int32";
    case data_type_t::BFloat16: return "BFloat16";
    case data_type_t::Int8: return "Int8";
    case data_type_t::Int4: return "Int4";
    case data_type_t::Int64: return "Int64";
    case data_type_t::XFloat32: return "XFloat32";
    case data_type_t::Float8_fnuz: return "Float8_fnuz";
    case data_type_t::BFloat8_fnuz: return "BFloat8_fnuz";
    case data_type_t::Float8BFloat8_fnuz: return "Float8BFloat8_fnuz";
    case data_type_t::BFloat8Float8_fnuz: return "BFloat8Float8_fnuz";
    case data_type_t::Float8: return "Float8";
    case data_type_t::BFloat8: return "BFloat8";
    case data_type_t::Float8BFloat8: return "Float8BFloat8";
    case data_type_t::BFloat8Float8: return "BFloat8Float8";
    case data_type_t::Float6: return "Float6";
    case data_type_t::BFloat6: return "BFloat6";
    case data_type_t::Float4: return "Float4";
    default: return "Invalid";
  }
}

data_type_t string_to_datatype(std::string s) {
  if (s == "f32") return data_type_t::Float;
  if (s == "c32") return data_type_t::ComplexFloat;
  if (s == "c64") return data_type_t::ComplexDouble;
  if (s == "f64") return data_type_t::Double;
  if (s == "f16") return data_type_t::Half;
  if (s == "i32") return data_type_t::Int32;
  if (s == "bf16") return data_type_t::BFloat16;
  if (s == "i8") return data_type_t::Int8;
  if (s == "i4") return data_type_t::Int4;
  if (s == "xf32") return data_type_t::XFloat32;
  if (s == "f8") return data_type_t::Float8;
  if (s == "bf8") return data_type_t::BFloat8;
  if (s == "f6") return data_type_t::Float6;
  if (s == "bf6") return data_type_t::BFloat6;
  if (s == "f4") return data_type_t::Float4;
  return data_type_t::None;
}

}  // namespace origami
