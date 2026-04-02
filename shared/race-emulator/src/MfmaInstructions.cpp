// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/FloatTypes.h"
#include "race-emulator/Instruction.h"
#include "race-emulator/Util.h"
#include "race-emulator/Wave.h"
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

namespace {

// C/D coordinate mapping (shared by MFMA 16x16xK variants).
// Lane L -> 4 output VGPRs -> row = 4*(L/16)+elemIdx, col = L%16.
static std::pair<int, int> mapLaneToCoordC(int lane, int elemIdx) {
  int col = lane % 16;
  int row = 4 * (lane / 16) + elemIdx;
  return {row, col};
}

// v_mfma_f32_16x16x8_xf32 (gfx942, wave-64)
class VMfmaF32_16168_XF32 : public Instruction {
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 5 && "Unexpected operand count");

    auto dst0 = wave.getFirstRegister(partitioned[1]);
    auto A = wave.getFirstRegister(partitioned[2]).index;
    auto B = wave.getFirstRegister(partitioned[3]).index;
    auto C = wave.parseOperand<uint32_t>(partitioned[4]);

    auto waveSize = wave.getWaveSize();
    assert(waveSize == 64);

    return [&wave, dst0, A, B, C, waveSize]() {
      std::array<float, 16 * 8> matA = {};
      std::array<float, 16 * 8> matB = {};
      std::array<float, 16 * 16> out = {};

      for (int l = 0; l < waveSize; ++l) {
        auto rA0 = std::bit_cast<float>(wave.getVgpr(A + 0, l));
        auto rA1 = std::bit_cast<float>(wave.getVgpr(A + 1, l));
        auto rB0 = std::bit_cast<float>(wave.getVgpr(B + 0, l));
        auto rB1 = std::bit_cast<float>(wave.getVgpr(B + 1, l));

        auto rowA = l % 16;
        auto colA = (l / 16) * 2;

        matA[rowA * 8 + colA] = rA0;
        matA[rowA * 8 + colA + 1] = rA1;
        matB[colA * 16 + rowA] = rB0;
        matB[(colA + 1) * 16 + rowA] = rB1;

        if (!C.isLiteral) {
          for (int i = 0; i < 4; ++i) {
            uint32_t rC = wave.getVgpr(C.reg.index + i, l);
            auto col = l % 16;
            auto row = 4 * (l / 16) + i;
            out[row * 16 + col] = std::bit_cast<float>(rC);
          }
        } else {
          float initVal =
              (C.literalValue == 0)
                  ? 0.0f
                  : std::bit_cast<float>(static_cast<uint32_t>(C.literalValue));
          for (int i = 0; i < 4; ++i) {
            auto col = l % 16;
            auto row = 4 * (l / 16) + i;
            out[row * 16 + col] = initVal;
          }
        }
      }

      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          float sum = 0.0f;
          for (int k = 0; k < 8; ++k) {
            sum += matA[row * 8 + k] * matB[col + k * 16];
          }
          out[row * 16 + col] += sum;
        }
      }

      for (int l = 0; l < waveSize; ++l) {
        for (int i = 0; i < 4; ++i) {
          auto col = l % 16;
          auto row = 4 * (l / 16) + i;
          wave.setVgpr(dst0.index + i, l,
                       std::bit_cast<uint32_t>(out[row * 16 + col]));
        }
      }

      return wave.getPc() + 1;
    };
  }
};

// v_mfma_f32_16x16xK_{bf16,f16} (CDNA, wave-64)
//
// D[16x16 f32] = A[16xK T] * B[Kx16 T] + C[16x16 f32]
//
// Template parameters:
//   ToFloat    -- f16->float or bf16->float conversion function.
//   NInputRegs -- VGPRs per A/B operand: K/8 (2 for K=16, 4 for K=32).
//                Compile-time so the compiler can size arrays and unroll loops.
//
// K=16 is gfx942 (v_mfma_f32_16x16x16), K=32 is gfx950 (v_mfma_f32_16x16x32).
//
// Lane mapping (wave-64, 4 groups of 16 lanes):
//   A: lane L holds row (L % 16). Each group of 16 lanes covers K/4 elements
//      of the K dimension. NInputRegs VGPRs per operand, each holding
//      2 packed f16/bf16 values -> 2*NInputRegs elements per lane.
//   B: same layout as A but interpreted as columns.
//   C/D: lane L -> 4 output VGPRs -> row = 4*(L/16)+i, col = L%16.
template <float (*ToFloat)(uint16_t), int NInputRegs>
class VMfmaF32_16x16xK : public Instruction {
  static constexpr int K = 8 * NInputRegs;
  static constexpr int ElemsPerLane = 2 * NInputRegs;

public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 5 && "Unexpected operand count");

    auto dst0 = wave.getFirstRegister(partitioned[1]);
    auto A = wave.getFirstRegister(partitioned[2]).index;
    auto B = wave.getFirstRegister(partitioned[3]).index;
    auto C = wave.parseOperand<uint32_t>(partitioned[4]);

    int waveSize = wave.getWaveSize();
    assert(waveSize == 64);

    return [&wave, dst0, A, B, C, waveSize]() {
      std::array<float, 16 * K> matA = {};
      std::array<float, 16 * K> matB = {};
      std::array<float, 16 * 16> out = {};

      for (int l = 0; l < waveSize; ++l) {
        uint16_t a_raw[ElemsPerLane], b_raw[ElemsPerLane];
        for (int r = 0; r < NInputRegs; ++r) {
          uint32_t rA = wave.getVgpr(A + r, l);
          a_raw[2 * r] = static_cast<uint16_t>(rA & 0xFFFF);
          a_raw[2 * r + 1] = static_cast<uint16_t>(rA >> 16);
          uint32_t rB = wave.getVgpr(B + r, l);
          b_raw[2 * r] = static_cast<uint16_t>(rB & 0xFFFF);
          b_raw[2 * r + 1] = static_cast<uint16_t>(rB >> 16);
        }

        int row = l % 16;
        int kBase = ElemsPerLane * (l / 16);
        for (int i = 0; i < ElemsPerLane; ++i) {
          matA[row * K + kBase + i] = ToFloat(a_raw[i]);
          matB[row * K + kBase + i] = ToFloat(b_raw[i]);
        }

        if (!C.isLiteral) {
          for (int i = 0; i < 4; ++i) {
            uint32_t rC = wave.getVgpr(C.reg.index + i, l);
            auto [cRow, col] = mapLaneToCoordC(l, i);
            out[cRow * 16 + col] = std::bit_cast<float>(rC);
          }
        } else {
          float initVal =
              (C.literalValue == 0)
                  ? 0.0f
                  : std::bit_cast<float>(static_cast<uint32_t>(C.literalValue));
          for (int i = 0; i < 4; ++i) {
            auto [cRow, col] = mapLaneToCoordC(l, i);
            out[cRow * 16 + col] = initVal;
          }
        }
      }

      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          float sum = 0.0f;
          for (int k = 0; k < K; ++k) {
            sum += matA[row * K + k] * matB[col * K + k];
          }
          out[row * 16 + col] += sum;
        }
      }

      for (int l = 0; l < waveSize; ++l) {
        for (int i = 0; i < 4; ++i) {
          auto [cRow, col] = mapLaneToCoordC(l, i);
          wave.setVgpr(dst0.index + i, l,
                       std::bit_cast<uint32_t>(out[cRow * 16 + col]));
        }
      }

      return wave.getPc() + 1;
    };
  }
};

// v_wmma_f32_16x16x16_{bf16,f16} (RDNA3/3.5, wave-32)
//
// D[16x16 f32] = A[16x16 T] * B[16x16 T] + C[16x16 f32]
//
// Lane mapping (wave-32, 8 VGPRs per A/B operand):
//   A: lane L (0-15) holds row L; VGPR v holds A[L][2*v] (lo) and A[L][2*v+1]
//   (hi).
//      Lanes 16-31 replicate lanes 0-15.
//   B: lane L (0-15) holds column L; VGPR v holds B[2*v][L] (lo) and
//   B[2*v+1][L] (hi).
//      Lanes 16-31 replicate lanes 0-15.
//   C/D: lane L in [0,15]  -> VGPR v -> row 2*v,   col L
//        lane L in [16,31] -> VGPR v -> row 2*v+1, col L-16
//
// Templated on ToFloat to share code between bf16 and f16 variants.
template <float (*ToFloat)(uint16_t)>
class VWmmaF32_161616 : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 5 && "Unexpected operand count");

    auto dst0 = wave.getFirstRegister(partitioned[1]);
    auto A = wave.getFirstRegister(partitioned[2]).index;
    auto B = wave.getFirstRegister(partitioned[3]).index;
    auto C = wave.parseOperand<uint32_t>(partitioned[4]);

    int waveSize = wave.getWaveSize();
    assert(waveSize == 32);

    return [&wave, dst0, A, B, C, waveSize]() {
      std::array<float, 16 * 16> matA = {};
      std::array<float, 16 * 16> matB = {};
      std::array<float, 16 * 16> out = {};

      // Read A and B: 8 registers each, all lanes. Race check is O(1)
      // per register via regEventCount.
      auto sw = wave.profileScope("wmma_readOperands");
      for (int v = 0; v < 8; ++v) {
        uint32_t raw[32];
        wave.getVgprs(A + v, raw);
        for (int l = 0; l < 16; ++l) {
          matA[l * 16 + 2 * v] = ToFloat(raw[l] & 0xFFFF);
          matA[l * 16 + 2 * v + 1] = ToFloat(raw[l] >> 16);
        }
      }

      for (int v = 0; v < 8; ++v) {
        uint32_t raw[32];
        wave.getVgprs(B + v, raw);
        for (int l = 0; l < 16; ++l) {
          matB[(2 * v) * 16 + l] = ToFloat(raw[l] & 0xFFFF);
          matB[(2 * v + 1) * 16 + l] = ToFloat(raw[l] >> 16);
        }
      }

      // C: 8 registers × 32 lanes (f32, no conversion needed).
      if (!C.isLiteral) {
        for (int v = 0; v < 8; ++v) {
          uint32_t raw[32];
          wave.getVgprs(C.reg.index + v, raw);
          for (int l = 0; l < waveSize; ++l) {
            int col = l % 16;
            int row = (l < 16) ? (2 * v) : (2 * v + 1);
            out[row * 16 + col] = std::bit_cast<float>(raw[l]);
          }
        }
      }

      sw = wave.profileScope("wmma_matmul");
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          float sum = 0.0f;
          for (int k = 0; k < 16; ++k) {
            sum += matA[row * 16 + k] * matB[k * 16 + col];
          }
          out[row * 16 + col] += sum;
        }
      }

      sw = wave.profileScope("wmma_writeResult");
      for (int l = 0; l < waveSize; ++l) {
        int col = l % 16;
        for (int v = 0; v < 8; ++v) {
          int row = (l < 16) ? (2 * v) : (2 * v + 1);
          wave.setVgpr(dst0.index + v, l,
                       std::bit_cast<uint32_t>(out[row * 16 + col]));
        }
      }

      return wave.getPc() + 1;
    };
  }
};

// v_mfma_i32_32x32x16_i8: Integer MFMA. Used for accumulator initialization
// (zeroing with literal 0). Stub: writes C (or literal) to dst accumulators.
// 16 output VGPRs per lane, 64 lanes.
class VMfmaI32_32x32x16_I8 : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 5);

    auto dst0 = wave.getFirstRegister(partitioned[1]);
    // A and B are 2 VGPRs each (32 i8 values packed into 64 bits)
    auto A = wave.getFirstRegister(partitioned[2]).index;
    auto B = wave.getFirstRegister(partitioned[3]).index;
    auto C = wave.parseOperand<uint32_t>(partitioned[4]);

    int waveSize = wave.getWaveSize();
    constexpr int nOutputRegs = 16;

    return [&wave, dst0, A, B, C, waveSize]() {
      // Read inputs to trigger race checks.
      for (int l = 0; l < waveSize; ++l) {
        (void)wave.getVgpr(A, l);
        (void)wave.getVgpr(A + 1, l);
        (void)wave.getVgpr(B, l);
        (void)wave.getVgpr(B + 1, l);
      }

      // Write C (or literal 0) to dst.
      for (int l = 0; l < waveSize; ++l) {
        for (int i = 0; i < nOutputRegs; ++i) {
          uint32_t val =
              C.isLiteral ? C.literalValue : wave.getVgpr(C.reg.index + i, l);
          wave.setVgpr(dst0.index + i, l, val);
        }
      }

      return wave.getPc() + 1;
    };
  }
};

// v_mfma_scale_f32_16x16x128_f8f6f4: Scaled MFMA for MXFP4/FP6/FP8.
//
// Syntax:
//   v_mfma_scale_f32_16x16x128_f8f6f4 dst, A[4], B[4], C, scaleA, scaleB
//       op_sel:[a,b] op_sel_hi:[c,d] cbsz:X blgp:Y
//
// cbsz/blgp select the A/B data format (4 = FP4 E2M1).
// op_sel/op_sel_hi select which byte of scaleA/scaleB to use:
//   scaleA byte index = (op_sel_hi[0] << 1) | op_sel[0]
//   scaleB byte index = (op_sel_hi[1] << 1) | op_sel[1]
//
// VGPR layout for FP4 (ISA Section 7.1.5.1):
//   64 lanes, 4 groups of 16. Each lane holds 4 VGPRs = 32 FP4 nibbles.
//   Lane l contributes row (l % 16), K range [32*(l/16) .. 32*(l/16)+31].
//
// Currently only cbsz=4/blgp=4 (FP4) is emulated.
// TODO(newling) support FP6 (cbsz/blgp=2,3) and FP8 (cbsz/blgp=0,1).
class VMfmaScaleF32_16x16x128 : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() >= 7);

    auto dst0 = wave.getFirstRegister(partitioned[1]);
    auto A = wave.getFirstRegister(partitioned[2]).index;
    auto B = wave.getFirstRegister(partitioned[3]).index;
    auto C = wave.parseOperand<uint32_t>(partitioned[4]);
    auto scaleAReg = wave.getFirstRegister(partitioned[5]).index;
    auto scaleBReg = wave.getFirstRegister(partitioned[6]).index;

    // Parse modifiers to determine scale byte selection.
    auto mods = parsePackedModifiers(line);
    int scaleAByteIdx = 0;
    int scaleBByteIdx = 0;
    if (mods.count("op_sel") && mods["op_sel"].size() >= 2) {
      scaleAByteIdx = mods["op_sel"][0];
      scaleBByteIdx = mods["op_sel"][1];
    }
    if (mods.count("op_sel_hi") && mods["op_sel_hi"].size() >= 2) {
      scaleAByteIdx |= mods["op_sel_hi"][0] << 1;
      scaleBByteIdx |= mods["op_sel_hi"][1] << 1;
    }

    int waveSize = wave.getWaveSize();
    assert(waveSize == 64);
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 128;
    constexpr int nInputRegs = 4;
    constexpr int nOutputRegs = 4;

    return [&wave, dst0, A, B, C, scaleAReg, scaleBReg, scaleAByteIdx,
            scaleBByteIdx, waveSize]() {
      if (!wave.isCompleteEmulation()) {
        // Race-check only: read inputs, pass C through to dst.
        for (int l = 0; l < waveSize; ++l) {
          for (int r = 0; r < nInputRegs; ++r) {
            (void)wave.getVgpr(A + r, l);
            (void)wave.getVgpr(B + r, l);
          }
          (void)wave.getVgpr(scaleAReg, l);
          (void)wave.getVgpr(scaleBReg, l);
        }
        for (int l = 0; l < waveSize; ++l) {
          for (int i = 0; i < nOutputRegs; ++i) {
            uint32_t val =
                C.isLiteral ? C.literalValue : wave.getVgpr(C.reg.index + i, l);
            wave.setVgpr(dst0.index + i, l, val);
          }
        }
        return wave.getPc() + 1;
      }

      // --- Full emulation (FP4 only) ---

      // 1. Extract scale bytes from lane 0 (uniform across lanes).
      uint32_t scaleAWord = wave.getVgpr(scaleAReg, 0);
      uint32_t scaleBWord = wave.getVgpr(scaleBReg, 0);
      uint8_t scaleAByte =
          static_cast<uint8_t>((scaleAWord >> (8 * scaleAByteIdx)) & 0xFF);
      uint8_t scaleBByte =
          static_cast<uint8_t>((scaleBWord >> (8 * scaleBByteIdx)) & 0xFF);
      float scaleAF = e8m0ToFloat(scaleAByte);
      float scaleBF = e8m0ToFloat(scaleBByte);

      // 2. Unpack A[16][128] from VGPRs into flat FP4 bytes.
      // Lane l contributes row (l%16), K range [32*(l/16) .. +31].
      // 4 VGPRs = 16 bytes = 32 nibbles.
      uint8_t flatA[M * (K / 2)] = {};
      uint8_t flatB[M * (K / 2)] = {};
      for (int l = 0; l < waveSize; ++l) {
        int row = l % M;
        int kGroup = l / M; // 0-3
        int byteOffset = row * (K / 2) + kGroup * 16;
        for (int r = 0; r < nInputRegs; ++r) {
          uint32_t aVal = wave.getVgpr(A + r, l);
          uint32_t bVal = wave.getVgpr(B + r, l);
          std::memcpy(&flatA[byteOffset + r * 4], &aVal, 4);
          std::memcpy(&flatB[byteOffset + r * 4], &bVal, 4);
        }
      }

      // 3. Convert FP4 → F32. Scale=1.0 here; instruction-level scale
      //    applied after matmul.
      // Convert FP4 nibbles to F32 without per-block scaling (scale=1.0,
      // E8M0 exponent 127). The instruction-level scale is applied after
      // the matmul.
      constexpr uint8_t kUnitScale = 127; // E8M0 for 1.0
      uint8_t unitScales[M * (K / 32)];
      std::memset(unitScales, kUnitScale, sizeof(unitScales));
      float matA[M * K];
      float matB[N * K];
      mxfp4MatrixToF32<M, K>(matA, flatA, unitScales);
      mxfp4MatrixToF32<N, K>(matB, flatB, unitScales);

      // 4. Load C into output matrix.
      std::array<float, M * N> out = {};
      for (int l = 0; l < waveSize; ++l) {
        for (int i = 0; i < nOutputRegs; ++i) {
          auto [cRow, col] = mapLaneToCoordC(l, i);
          if (!C.isLiteral) {
            out[cRow * N + col] =
                std::bit_cast<float>(wave.getVgpr(C.reg.index + i, l));
          } else {
            out[cRow * N + col] =
                C.literalValue == 0
                    ? 0.0f
                    : std::bit_cast<float>(
                          static_cast<uint32_t>(C.literalValue));
          }
        }
      }

      // 5. Matmul: D += (A * B) * scaleA * scaleB.
      float combinedScale = scaleAF * scaleBF;
      for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
          float sum = 0.0f;
          for (int k = 0; k < K; ++k) {
            sum += matA[row * K + k] * matB[col * K + k];
          }
          out[row * N + col] += sum * combinedScale;
        }
      }

      // 6. Write results back to VGPRs.
      for (int l = 0; l < waveSize; ++l) {
        for (int i = 0; i < nOutputRegs; ++i) {
          auto [cRow, col] = mapLaneToCoordC(l, i);
          wave.setVgpr(dst0.index + i, l,
                       std::bit_cast<uint32_t>(out[cRow * N + col]));
        }
      }

      return wave.getPc() + 1;
    };
  }
};

// Registration

template <typename InstT> struct Register {
  template <typename... Args>
  Register(const std::string &name, Args &&...args) {
    InstructionRegistry::instance().add(
        name, std::make_unique<InstT>(std::forward<Args>(args)...));
  }
};

static Register<VMfmaF32_16x16xK<bf16ToFloat, 2>>
    v_mfma_bf16("v_mfma_f32_16x16x16_bf16");
static Register<VMfmaF32_16x16xK<bf16ToFloat, 2>>
    v_mfma_bf16_1k("v_mfma_f32_16x16x16bf16_1k");
static Register<VMfmaF32_16x16xK<f16ToFloat, 2>>
    v_mfma_f16("v_mfma_f32_16x16x16_f16");
static Register<VMfmaF32_16168_XF32> v_mfma_8("v_mfma_f32_16x16x8_xf32");
static Register<VMfmaF32_16x16xK<bf16ToFloat, 4>>
    v_mfma_bf16_32("v_mfma_f32_16x16x32_bf16");
static Register<VMfmaF32_16x16xK<f16ToFloat, 4>>
    v_mfma_f16_32("v_mfma_f32_16x16x32_f16");
static Register<VWmmaF32_161616<bf16ToFloat>>
    v_wmma_bf16("v_wmma_f32_16x16x16_bf16");
static Register<VWmmaF32_161616<f16ToFloat>>
    v_wmma_f16("v_wmma_f32_16x16x16_f16");
static Register<VMfmaI32_32x32x16_I8> v_mfma_i8("v_mfma_i32_32x32x16_i8");
static Register<VMfmaScaleF32_16x16x128>
    v_mfma_scale("v_mfma_scale_f32_16x16x128_f8f6f4");

} // namespace
} // namespace raceemulator
