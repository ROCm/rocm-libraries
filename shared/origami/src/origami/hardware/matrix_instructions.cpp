// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/hardware/matrix_instructions.hpp"

#include <stdexcept>

namespace origami {

namespace {

// One table per family.  Each row is {{MI_M, MI_N, MI_K, input type}, latency in clocks}.  The
// row order is part of the model's behaviour: it decides the iteration order of the map, which
// is the order hardware_t::get_valid_matrix_instructions returns instructions in.

// CDNA2 (gfx90a) MFMA instructions.
const matrix_instruction_latencies_t& cdna2_latencies() {
  // clang-format off
  static const matrix_instruction_latencies_t latencies = {
      // F32
      {{32, 32,  2, data_type_t::Float},    64},  // v_mfma_f32_32x32x2_f32
      {{32, 32,  1, data_type_t::Float},    64},  // v_mfma_f32_32x32x1_2b_f32
      {{16, 16,  4, data_type_t::Float},    32},  // v_mfma_f32_16x16x4_f32
      {{16, 16,  1, data_type_t::Float},    32},  // v_mfma_f32_16x16x1_4b_f32
      {{ 4,  4,  1, data_type_t::Float},     8},  // v_mfma_f32_4x4x1_16b_f32

      // F64
      {{16, 16,  4, data_type_t::Double},   32},  // v_mfma_f64_16x16x4_f64
      {{ 4,  4,  4, data_type_t::Double},   16},  // v_mfma_f64_4x4x4_4b_f64

      // TODO ComplexFloat
      // TODO ComplexDouble

      // F16
      {{32, 32,  4, data_type_t::Half},     64},  // v_mfma_f32_32x32x4_2b_f16
      {{32, 32,  8, data_type_t::Half},     64},  // v_mfma_f32_32x32x8_f16
      {{16, 16,  4, data_type_t::Half},     32},  // v_mfma_f32_16x16x4_4b_f16
      {{16, 16, 16, data_type_t::Half},     32},  // v_mfma_f32_16x16x16_f16
      {{ 4,  4,  4, data_type_t::Half},      8},  // v_mfma_f32_4x4x4_16b_f16

      // BF16
      {{32, 32,  4, data_type_t::BFloat16}, 64},  // v_mfma_f32_32x32x4_2b_bf16
      {{32, 32,  8, data_type_t::BFloat16}, 32},  // v_mfma_f32_32x32x8_bf16
      {{16, 16,  4, data_type_t::BFloat16}, 32},  // v_mfma_f32_16x16x4_4b_bf16
      {{16, 16, 16, data_type_t::BFloat16}, 16},  // v_mfma_f32_16x16x16_bf16
      {{ 4,  4,  4, data_type_t::BFloat16},  8},  // v_mfma_f32_4x4x4_16b_bf16

      // I8
      {{32, 32,  8, data_type_t::Int8},     64},  // v_mfma_f32_32x32x16_f8
      {{32, 32,  4, data_type_t::Int8},     64},  // v_mfma_i32_32x32x4_2b_i8
      {{16, 16, 16, data_type_t::Int8},     32},  // v_mfma_f32_16x16x32_i8
      {{16, 16,  4, data_type_t::Int8},     32},  // v_mfma_i32_16x16x4_4b_i8
      {{ 4,  4,  4, data_type_t::Int8},      8},  // v_mfma_i32_4x4x4_16b_i8

      // XF32
      {{32, 32,  8, data_type_t::XFloat32}, 96},  // v_mfma_f32_32x32x8_bf16 * 3
      {{32, 32, 16, data_type_t::XFloat32}, 96},  // v_mfma_f32_32x32x16_bf16 * 3
      {{16, 16, 16, data_type_t::XFloat32}, 48},  // v_mfma_f32_16x16x16_bf16 * 3
      {{16, 16, 32, data_type_t::XFloat32}, 48},  // v_mfma_f32_16x16x16_bf16 * 3
  };
  // clang-format on
  return latencies;
}

// CDNA3 (gfx942) MFMA instructions.
const matrix_instruction_latencies_t& cdna3_latencies() {
  // clang-format off
  static const matrix_instruction_latencies_t latencies = {
      // F32
      {{32, 32,  2, data_type_t::Float},              64},  // v_mfma_f32_32x32x2_f32
      {{32, 32,  1, data_type_t::Float},              64},  // v_mfma_f32_32x32x1_2b_f32
      {{16, 16,  4, data_type_t::Float},              32},  // v_mfma_f32_16x16x4_f32
      {{16, 16,  1, data_type_t::Float},              32},  // v_mfma_f32_16x16x1_4b_f32
      {{ 4,  4,  1, data_type_t::Float},               8},  // v_mfma_f32_4x4x1_16b_f32

      // F64
      {{16, 16,  4, data_type_t::Double},             32},  // v_mfma_f64_16x16x4_f64
      {{ 4,  4,  4, data_type_t::Double},             16},  // v_mfma_f64_4x4x4_4b_f64

      // TODO ComplexFloat
      // TODO ComplexDouble

      // F16
      {{32, 32,  4, data_type_t::Half},               64},  // v_mfma_f32_32x32x4_2b_f16
      {{32, 32,  8, data_type_t::Half},               32},  // v_mfma_f32_32x32x8_f16
      {{16, 16,  4, data_type_t::Half},               32},  // v_mfma_f32_16x16x4_4b_f16
      {{16, 16, 16, data_type_t::Half},               16},  // v_mfma_f32_16x16x16_f16
      {{ 4,  4,  4, data_type_t::Half},                8},  // v_mfma_f32_4x4x4_16b_f16

      // BF16
      {{32, 32,  4, data_type_t::BFloat16},           64},  // v_mfma_f32_32x32x4_2b_bf16
      {{32, 32,  8, data_type_t::BFloat16},           32},  // v_mfma_f32_32x32x8_bf16
      {{16, 16,  4, data_type_t::BFloat16},           32},  // v_mfma_f32_16x16x4_4b_bf16
      {{16, 16, 16, data_type_t::BFloat16},           16},  // v_mfma_f32_16x16x16_bf16
      {{ 4,  4,  4, data_type_t::BFloat16},            8},  // v_mfma_f32_4x4x4_16b_bf16

      // F8
      {{32, 32, 16, data_type_t::Float8_fnuz},        32},  // v_mfma_f32_32x32x16_f8
      {{16, 16, 32, data_type_t::Float8_fnuz},        16},  // v_mfma_f32_16x16x32_f8

      // BF8
      {{32, 32, 16, data_type_t::BFloat8_fnuz},       32},  // v_mfma_f32_32x32x16_bf8
      {{16, 16, 32, data_type_t::BFloat8_fnuz},       16},  // v_mfma_f32_16x16x32_bf8

      // F8B8
      {{32, 32, 16, data_type_t::Float8BFloat8_fnuz}, 32},  // v_mfma_f32_32x32x16_f8_bf8
      {{16, 16, 32, data_type_t::Float8BFloat8_fnuz}, 16},  // v_mfma_f32_16x16x32_f8_bf8

      // B8F8
      {{32, 32, 16, data_type_t::BFloat8Float8_fnuz}, 32},  // v_mfma_f32_32x32x16_bf8_f8
      {{16, 16, 32, data_type_t::BFloat8Float8_fnuz}, 16},  // v_mfma_f32_16x16x32_bf8_f8

      // I8
      {{32, 32, 16, data_type_t::Int8},               32},  // v_mfma_f32_32x32x16_f8
      {{32, 32,  4, data_type_t::Int8},               64},  // v_mfma_i32_32x32x4_2b_i8
      {{16, 16, 32, data_type_t::Int8},               16},  // v_mfma_f32_16x16x32_i8
      {{16, 16,  4, data_type_t::Int8},               32},  // v_mfma_i32_16x16x4_4b_i8
      {{ 4,  4,  4, data_type_t::Int8},                8},  // v_mfma_i32_4x4x4_16b_i8

      // XF32
      {{32, 32,  4, data_type_t::XFloat32},           32},  // v_mfma_f32_32x32x4_xf32
      {{16, 16, 32, data_type_t::XFloat32},           16},  // v_mfma_f32_16x16x8_xf32
  };
  // clang-format on
  return latencies;
}

// CDNA4 (gfx950) MFMA instructions.
const matrix_instruction_latencies_t& cdna4_latencies() {
  // clang-format off
  static const matrix_instruction_latencies_t latencies = {
      // F32
      {{32, 32,   2, data_type_t::Float},         64},  // v_mfma_f32_32x32x2_f32
      {{32, 32,   1, data_type_t::Float},         64},  // v_mfma_f32_32x32x1_2b_f32
      {{16, 16,   4, data_type_t::Float},         32},  // v_mfma_f32_16x16x4_f32
      {{16, 16,   1, data_type_t::Float},         32},  // v_mfma_f32_16x16x1_4b_f32
      {{ 4,  4,   1, data_type_t::Float},          8},  // v_mfma_f32_4x4x1_16b_f32

      // F64
      {{16, 16,   4, data_type_t::Double},        64},  // v_mfma_f64_16x16x4_f64
      {{ 4,  4,   4, data_type_t::Double},        16},  // v_mfma_f64_4x4x4_4b_f64

      // TODO ComplexFloat
      // TODO ComplexDouble

      // F16
      {{32, 32,   8, data_type_t::Half},          32},  // v_mfma_f32_32x32x8_f16
      {{32, 32,  16, data_type_t::Half},          32},  // v_mfma_f32_32x32x16_f16
      {{16, 16,  16, data_type_t::Half},          16},  // v_mfma_f32_16x16x16_f16
      {{16, 16,  32, data_type_t::Half},          16},  // v_mfma_f32_16x16x32_f16

      // BF16
      {{32, 32,   8, data_type_t::BFloat16},      32},  // v_mfma_f32_32x32x8_bf16
      {{32, 32,  16, data_type_t::BFloat16},      32},  // v_mfma_f32_32x32x16_bf16
      {{16, 16,  16, data_type_t::BFloat16},      16},  // v_mfma_f32_16x16x16_bf16
      {{16, 16,  32, data_type_t::BFloat16},      16},  // v_mfma_f32_16x16x16_bf16

      // F8
      {{32, 32,  64, data_type_t::Float8},        64},  // v_mfma_f32_32x32x64_f8
      {{32, 32,  16, data_type_t::Float8},        32},  // v_mfma_f32_32x32x16_f8
      {{16, 16, 128, data_type_t::Float8},        32},  // v_mfma_f32_16x16x128_f8
      {{16, 16,  32, data_type_t::Float8},        16},  // v_mfma_f32_16x16x32_f8

      // BF8
      {{32, 32,  64, data_type_t::BFloat8},       64},  // v_mfma_f32_32x32x64_bf8
      {{32, 32,  16, data_type_t::BFloat8},       32},  // v_mfma_f32_32x32x16_bf8
      {{16, 16, 128, data_type_t::BFloat8},       32},  // v_mfma_f32_16x16x128_bf8
      {{16, 16,  32, data_type_t::BFloat8},       16},  // v_mfma_f32_16x16x32_bf8

      // F8B8
      {{32, 32,  64, data_type_t::Float8BFloat8}, 64},  // v_mfma_f32_32x32x64_f8_bf8
      {{32, 32,  16, data_type_t::Float8BFloat8}, 32},  // v_mfma_f32_32x32x16_f8_bf8
      {{16, 16, 128, data_type_t::Float8BFloat8}, 32},  // v_mfma_f32_16x16x128_f8_bf8
      {{16, 16,  32, data_type_t::Float8BFloat8}, 16},  // v_mfma_f32_16x16x32_f8_bf8

      // B8F8
      {{32, 32,  64, data_type_t::BFloat8Float8}, 64},  // v_mfma_f32_32x32x64_bf8_f8
      {{32, 32,  16, data_type_t::BFloat8Float8}, 32},  // v_mfma_f32_32x32x16_bf8_f8
      {{16, 16, 128, data_type_t::BFloat8Float8}, 32},  // v_mfma_f32_16x16x128_bf8_f8
      {{16, 16,  32, data_type_t::BFloat8Float8}, 16},  // v_mfma_f32_16x16x32_bf8_f8

      // I8
      {{32, 32,  16, data_type_t::Int8},          32},  // v_mfma_i32_32x32x16_i8
      {{32, 32,  32, data_type_t::Int8},          32},  // v_mfma_i32_32x32x32_i8
      {{32, 32,   4, data_type_t::Int8},          64},  // v_mfma_i32_32x32x4_2b_i8
      {{16, 16,  32, data_type_t::Int8},          16},  // v_mfma_i32_16x16x32_i8
      {{16, 16,  64, data_type_t::Int8},          16},  // v_mfma_i32_16x16x64_i8
      {{16, 16,   4, data_type_t::Int8},          32},  // v_mfma_i32_16x16x4_4b_i8
      {{ 4,  4,   4, data_type_t::Int8},           8},  // v_mfma_i32_4x4x4_16b_i8

      // XF32
      {{32, 32,   8, data_type_t::XFloat32},      96},  // v_mfma_f32_32x32x8_bf16 * 3
      {{32, 32,  16, data_type_t::XFloat32},      96},  // v_mfma_f32_32x32x16_bf16 * 3
      {{16, 16,  16, data_type_t::XFloat32},      48},  // v_mfma_f32_16x16x16_bf16 * 3
      {{16, 16,  32, data_type_t::XFloat32},      48},  // v_mfma_f32_16x16x16_bf16 * 3

      // F6
      {{32, 32,  64, data_type_t::Float6},        32},  // v_mfma_f32_32x32x64_f6
      {{16, 16, 128, data_type_t::Float6},        16},  // v_mfma_f32_16x16x128_f6

      // BF6
      {{32, 32,  64, data_type_t::BFloat6},       32},  // v_mfma_f32_32x32x64_bf6
      {{16, 16, 128, data_type_t::BFloat6},       16},  // v_mfma_f32_16x16x128_bf6

      // F4
      {{32, 32,  64, data_type_t::Float4},        32},  // v_mfma_f32_32x32x64_f4
      {{16, 16, 128, data_type_t::Float4},        16},  // v_mfma_f32_16x16x128_f4

      // DOT2
      {{ 1,  1,  64, data_type_t::Half},          16},  // V_DOT2_F32_F16
      {{ 1,  1,  64, data_type_t::BFloat16},      16},  // V_DOT2_F32_BF16
  };
  // clang-format on
  return latencies;
}

// CDNA5 (gfx1250) WMMA instructions.
const matrix_instruction_latencies_t& cdna5_latencies() {
  // clang-format off
  static const matrix_instruction_latencies_t latencies = {
      // F64
      // DGEMM: V_WMMA_F64_16x16x4_F64, 16x4 F64 x 4x16 F64 = 16x16 F64
      {{16, 16,   4, data_type_t::Double},           16},

      // F32
      // SGEMM: V_WMMA_F32_16X16X4_F32, 16x4 F32 x 4x16 F32 = 16x16 F32
      {{16, 16,   4, data_type_t::Float},            16},

      // CGEMM: same tile shape as SGEMM, 4x real latency
      {{16, 16,   4, data_type_t::ComplexFloat},     64},

      // ZGEMM: same tile shape as DGEMM, 4x real latency
      {{16, 16,   4, data_type_t::ComplexDouble},    64},

      // F16
      // HHS: V_WMMA_F16_16X16X32_F16, 16x32 F16 x 32x16 F16 = 16x16 F16
      // HSS: V_WMMA_F32_16X16X32_F16, 16x32 F16 x 32x16 F16 = 16x16 F32
      {{16, 16,  32, data_type_t::Half},              8},

      // BF16
      // BBS:   V_WMMA_BF16_16X16X32_BF16,    16x32 BF16 x 32x16 BF16 = 16x16 BF16
      // BSS:   V_WMMA_F32_16X16X32_BF16,     16x32 BF16 x 32x16 BF16 = 16x16 F32
      // BBSB?: V_WMMA_BF16F32_16X16X32_BF16, 16x32 BF16 x 32x16 BF16 + 16x16 C F32 = 16x16 D BF16
      {{16, 16,  32, data_type_t::BFloat16},          8},

      // F8
      // F8SS: V_WMMA_F32_16X16X64_FP8_FP8,  16x64  FP8 x 64x16  FP8 = 16x16 F32
      //       V_WMMA_F32_16X16X128_FP8_FP8, 16x128 FP8 x 128x16 FP8 = 16x16 F32
      // F8HS: V_WMMA_F16_16X16X64_FP8_FP8,  16x64  FP8 x 64x16  FP8 = 16x16 F16
      //       V_WMMA_F16_16X16X128_FP8_FP8  16x128 FP8 x 128x16 FP8 = 16x16 F16
      {{16, 16,  64, data_type_t::Float8},            4},
      {{16, 16, 128, data_type_t::Float8},            8},

      // BF8
      // B8SS: V_WMMA_F32_16X16X64_BF8_BF8,  16x64  BF8 x 64x16  BF8 = 16x16 F32
      //       V_WMMA_F32_16X16X128_BF8_BF8, 16x128 BF8 x 128x16 BF8 = 16x16 F32
      // B8HS: V_WMMA_F16_16X16X64_BF8_BF8,  16x64  BF8 x 64x16  BF8 = 16x16 F16
      //       V_WMMA_F16_16X16X128_BF8_BF8, 16x128 BF8 x 128x16 BF8 = 16x16 F16
      {{16, 16,  64, data_type_t::BFloat8},           4},
      {{16, 16, 128, data_type_t::BFloat8},           8},

      // F8B8
      // F8B8SS: V_WMMA_F32_16X16X64_FP8_BF8,  16x64  FP8 x 64x16  BF8 = 16x16 F32
      //         V_WMMA_F32_16X16X128_FP8_BF8, 16x128 FP8 x 128x16 BF8 = 16x16 F32
      // F8B8HS: V_WMMA_F16_16X16X64_FP8_BF8,  16x64  FP8 x 64x16  BF8 = 16x16 F16
      //         V_WMMA_F16_16X16X128_FP8_BF8  16x128 FP8 x 128x16 BF8 = 16x16 F16
      {{16, 16,  64, data_type_t::Float8BFloat8},     4},
      {{16, 16, 128, data_type_t::Float8BFloat8},     8},

      // B8F8
      // B8F8SS: V_WMMA_F32_16X16X64_BF8_FP8, 16x64 BF8 x 64x16 FP8 = 16x16 F32
      //         V_WMMA_F32_16X16X128_BF8_FP8, 16x128 BF8 x 128x16 FP8 = 16x16 F32
      // B8F8HS: V_WMMA_F16_16X16X64_BF8_FP8, 16x64 BF8 x 64x16 FP8 = 16x16 F16
      //         V_WMMA_F16_16X16X128_BF8_FP8, 16x128 BF8 x 128x16 FP8 = 16x16 F16
      {{16, 16,  64, data_type_t::BFloat8Float8},     4},
      {{16, 16, 128, data_type_t::BFloat8Float8},     8},

      // MXF8
      // V_WMMA_F32_16x16x128_F8F6F4,  16x128 FP4/6/8 x 128x16 FP4/6/8 = 16x16 F32
      // TODO: confirm this is the 16x16x128 Float8 entry above.

      // F6
      // V_WMMA_F32_16x16x128_F8F6F4,  16x128 FP4/6/8 x 128x16 FP4/6/8 = 16x16 F32
      {{16, 16, 128, data_type_t::Float6},            8},

      // F4
      // V_WMMA_F32_16x16x128_F8F6F4,  16x128 FP4/6/8 x 128x16 FP4/6/8 = 16x16 F32
      // V_WMMA_F32_32x16x128_F4
      {{16, 16, 128, data_type_t::Float4},            4},
      {{32, 16, 128, data_type_t::Float4},            8},

      // I8
      // V_WMMA_I32_16X16X64_IU8, 16x64 IU8 x 64x16 IU8 = 16x16 I32
      {{16, 16,  64, data_type_t::Int8},              8},

      // XF32
      // x3 emulation: 3 x V_WMMA_BF16_16X16X32_BF16
      {{16, 16,  32, data_type_t::XFloat32},      8 * 3},

      // TODO: BF6, DOT2
  };
  // clang-format on
  return latencies;
}

// RDNA3 (gfx110x, gfx115x) WMMA instructions.
const matrix_instruction_latencies_t& rdna3_latencies() {
  // clang-format off
  static const matrix_instruction_latencies_t latencies = {
      // F16
      {{16, 16, 16, data_type_t::Half},     32},  // v_wmma_f32_16x16x16_f16/v_wmma_f16_16x16x16_f16

      // BF16
      {{16, 16, 16, data_type_t::BFloat16}, 32},  // v_wmma_f32_16x16x16_bf16/v_wmma_bf16_16x16x16_bf16

      // I8
      {{16, 16, 16, data_type_t::Int8},     32},  // v_wmma_i32_16x16x16_iu8

      // I4
      {{16, 16, 16, data_type_t::Int4},     16},  // v_wmma_i32_16x16x16_iu4
  };
  // clang-format on
  return latencies;
}

// RDNA4 (gfx117x, gfx120x) WMMA instructions.
const matrix_instruction_latencies_t& rdna4_latencies() {
  // clang-format off
  static const matrix_instruction_latencies_t latencies = {
      // F16
      {{16, 16, 16, data_type_t::Half},          16},  // v_wmma_f16_16x16x16_f16/v_wmma_f32_16x16x16_f16

      // BF16
      {{16, 16, 16, data_type_t::BFloat16},      16},  // v_wmma_bf16_16x16x16_bf16/v_wmma_f32_16x16x16_bf16

      // F8
      {{16, 16, 16, data_type_t::Float8},         8},  // v_wmma_f32_16x16x16_fp8_fp8

      // F8B8
      {{16, 16, 16, data_type_t::Float8BFloat8},  8},  // v_wmma_f32_16x16x16_fp8_bf8

      // B8F8
      {{16, 16, 16, data_type_t::BFloat8Float8},  8},  // v_wmma_f32_16x16x16_bf8_fp8

      // B8
      {{16, 16, 16, data_type_t::BFloat8},        8},  // v_wmma_f32_16x16x16_bf8_bf8

      // I8
      {{16, 16, 16, data_type_t::Int8},           8},  // v_wmma_i32_16x16x16_iu8

      // I4
      {{16, 16, 16, data_type_t::Int4},           8},  // v_wmma_i32_16x16x16_iu4
      {{16, 16, 32, data_type_t::Int4},           8},  // v_wmma_i32_16x16x32_iu4
  };
  // clang-format on
  return latencies;
}

}  // namespace

const matrix_instruction_latencies_t& matrix_instruction_latencies(family_t family) {
  switch (family) {
    case family_t::cdna2: return cdna2_latencies();
    case family_t::cdna3: return cdna3_latencies();
    case family_t::cdna4: return cdna4_latencies();
    case family_t::cdna5: return cdna5_latencies();
    case family_t::rdna3: return rdna3_latencies();
    case family_t::rdna4: return rdna4_latencies();
    default: throw std::out_of_range("No matrix instructions for an unknown architecture family");
  }
}

}  // namespace origami
