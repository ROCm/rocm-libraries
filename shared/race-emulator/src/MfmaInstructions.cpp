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

// TODO: Factorize this!
// The AMD ISA has a plethora of MFMA instructions (various shapes, types,
// and iterations). Creating a distinct class for every permutation is
// not sustainable. We should implement a generic `MFMA_Op` template that
// accepts dimensions (M, N, K) and data types as parameters.

// Helper: BF16 <-> Float conversion
float bf16_to_float(uint16_t b) {
  uint32_t val = static_cast<uint32_t>(b) << 16;
  return std::bit_cast<float>(val);
}

// Coordinate mapping helpers
static std::pair<int, int> mapLaneToCoordA(int lane, int elemIdx) {
  int row = lane % 16;
  int col = 4 * (lane / 16) + elemIdx;
  return {row, col};
}

static std::pair<int, int> mapLaneToCoordB(int lane, int elemIdx) {
  auto forA = mapLaneToCoordA(lane, elemIdx);
  return {forA.second, forA.first};
}

static std::pair<int, int> mapLaneToCoordC(int lane, int elemIdx) {
  return mapLaneToCoordB(lane, elemIdx);
}

// v_mfma_f32_16x16x8_xf32
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
          // If C is a literal, it serves as an initialization value (usually 0)
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

      // Matrix Multiply
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          float sum = 0.0f;
          for (int k = 0; k < 8; ++k) {
            sum += matA[row * 8 + k] * matB[col + k * 16];
          }
          out[row * 16 + col] += sum;
        }
      }

      // Writeback
      for (int l = 0; l < waveSize; ++l) {
        for (int i = 0; i < 4; ++i) {
          auto col = l % 16;
          auto row = 4 * (l / 16) + i;
          uint32_t outVal = std::bit_cast<uint32_t>(out[row * 16 + col]);
          wave.setVgpr(dst0.index + i, l, outVal);
        }
      }

      return wave.getPc() + 1;
    };
  }
};

// v_mfma_f32_16x16x16_bf16
class VMfmaF32_161616_BF16 : public Instruction {
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
          matA[row * 16 + col] = bf16_to_float(a_raw[i]);
          matB[row * 16 + col] = bf16_to_float(b_raw[i]);
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

      // Matrix Multiply
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          float sum = 0.0f;
          for (int k = 0; k < 16; ++k) {
            sum += matA[row * 16 + k] * matB[col * 16 + k];
          }
          out[row * 16 + col] += sum;
        }
      }

      // Writeback
      for (int l = 0; l < waveSize; ++l) {
        for (int i = 0; i < 4; ++i) {
          auto [row, col] = mapLaneToCoordC(l, i);
          float result = out[row * 16 + col];
          wave.setVgpr(dst0.index + i, l, std::bit_cast<uint32_t>(result));
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

static Register<VMfmaF32_161616_BF16> v_mfma_16("v_mfma_f32_16x16x16_bf16");
static Register<VMfmaF32_161616_BF16>
    v_mfma_16_1k("v_mfma_f32_16x16x16bf16_1k");
static Register<VMfmaF32_16168_XF32> v_mfma_8("v_mfma_f32_16x16x8_xf32");

} // namespace
} // namespace raceemulator
