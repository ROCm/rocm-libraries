#include "race-emulator/FloatTypes.h"
#include "race-emulator/Instruction.h"
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

// Coordinate mapping helpers (gfx942 MFMA 16x16x16 layout)
static std::pair<int, int> mapLaneToCoordA(int lane, int elemIdx) {
  int row = lane % 16;
  int col = 4 * (lane / 16) + elemIdx;
  return {row, col};
}

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

// v_mfma_f32_16x16x16_{bf16,f16} (gfx942, wave-64)
//
// The only difference between the bf16 and f16 variants is the conversion
// function used to unpack 16-bit inputs to float.  Templated on ToFloat.
template <float (*ToFloat)(uint16_t)>
class VMfmaF32_161616 : public Instruction {
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
      std::array<float, 16 * 16> matA = {};
      std::array<float, 16 * 16> matB = {};
      std::array<float, 16 * 16> out = {};

      for (int l = 0; l < waveSize; ++l) {
        uint32_t rA0 = wave.getVgpr(A + 0, l);
        uint32_t rA1 = wave.getVgpr(A + 1, l);
        uint16_t a_raw[4] = {static_cast<uint16_t>(rA0 & 0xFFFF),
                             static_cast<uint16_t>(rA0 >> 16),
                             static_cast<uint16_t>(rA1 & 0xFFFF),
                             static_cast<uint16_t>(rA1 >> 16)};

        uint32_t rB0 = wave.getVgpr(B + 0, l);
        uint32_t rB1 = wave.getVgpr(B + 1, l);
        uint16_t b_raw[4] = {static_cast<uint16_t>(rB0 & 0xFFFF),
                             static_cast<uint16_t>(rB0 >> 16),
                             static_cast<uint16_t>(rB1 & 0xFFFF),
                             static_cast<uint16_t>(rB1 >> 16)};

        for (int i = 0; i < 4; ++i) {
          auto [row, col] = mapLaneToCoordA(l, i);
          matA[row * 16 + col] = ToFloat(a_raw[i]);
          matB[row * 16 + col] = ToFloat(b_raw[i]);
        }

        if (!C.isLiteral) {
          for (int i = 0; i < 4; ++i) {
            uint32_t rC = wave.getVgpr(C.reg.index + i, l);
            auto [row, col] = mapLaneToCoordC(l, i);
            out[row * 16 + col] = std::bit_cast<float>(rC);
          }
        } else {
          float initVal =
              (C.literalValue == 0)
                  ? 0.0f
                  : std::bit_cast<float>(static_cast<uint32_t>(C.literalValue));
          for (int i = 0; i < 4; ++i) {
            auto [row, col] = mapLaneToCoordC(l, i);
            out[row * 16 + col] = initVal;
          }
        }
      }

      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          float sum = 0.0f;
          for (int k = 0; k < 16; ++k) {
            sum += matA[row * 16 + k] * matB[col * 16 + k];
          }
          out[row * 16 + col] += sum;
        }
      }

      for (int l = 0; l < waveSize; ++l) {
        for (int i = 0; i < 4; ++i) {
          auto [row, col] = mapLaneToCoordC(l, i);
          wave.setVgpr(dst0.index + i, l,
                       std::bit_cast<uint32_t>(out[row * 16 + col]));
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

// Registration

template <typename InstT> struct Register {
  template <typename... Args>
  Register(const std::string &name, Args &&...args) {
    InstructionRegistry::instance().add(
        name, std::make_unique<InstT>(std::forward<Args>(args)...));
  }
};

static Register<VMfmaF32_161616<bf16ToFloat>>
    v_mfma_bf16("v_mfma_f32_16x16x16_bf16");
static Register<VMfmaF32_161616<bf16ToFloat>>
    v_mfma_bf16_1k("v_mfma_f32_16x16x16bf16_1k");
static Register<VMfmaF32_161616<f16ToFloat>>
    v_mfma_f16("v_mfma_f32_16x16x16_f16");
static Register<VMfmaF32_16168_XF32> v_mfma_8("v_mfma_f32_16x16x8_xf32");
static Register<VWmmaF32_161616<bf16ToFloat>>
    v_wmma_bf16("v_wmma_f32_16x16x16_bf16");
static Register<VWmmaF32_161616<f16ToFloat>>
    v_wmma_f16("v_wmma_f32_16x16x16_f16");

} // namespace
} // namespace raceemulator
