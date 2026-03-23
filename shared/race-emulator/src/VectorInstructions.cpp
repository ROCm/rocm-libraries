#include "race-emulator/FloatTypes.h"
#include "race-emulator/Instruction.h"
#include "race-emulator/Util.h"
#include "race-emulator/Wave.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace {
// Compute a * b + c, returning the low 64 bits and whether the result
// overflows into the upper 64 bits.
std::pair<uint64_t, bool> mulAddU64(uint64_t a, uint64_t b, uint64_t c) {
#ifdef _MSC_VER
  uint64_t hi = 0;
  uint64_t lo = _umul128(a, b, &hi);
  lo += c;
  if (lo < c)
    hi++;
  return {lo, hi != 0};
#else
  __uint128_t full = static_cast<__uint128_t>(a) * b + c;
  return {static_cast<uint64_t>(full), (full >> 64) != 0};
#endif
}
} // namespace
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace raceemulator {

namespace {

enum class InstructionMode : uint8_t {
  Standard,
  SDWA,
  DPP // Placeholder
};

// SDWA control helper
struct SdwaControl {

  struct Selection {
    uint32_t mask;
    uint32_t shift;
  };

  struct SelectionOperand {
    uint32_t result;
    uint32_t previous;
  };

  enum class UnusedMode : uint8_t { PAD = 0, SEXT = 1, PRESERVE = 2 };

  bool enabled = false;
  Selection src0 = {0xFFFFFFFF, 0};
  Selection src1 = {0xFFFFFFFF, 0};
  Selection dst = {0xFFFFFFFF, 0};
  UnusedMode dstUnused = UnusedMode::PAD;

  static constexpr std::string_view kSrc0Sel = "src0_sel:";
  static constexpr std::string_view kSrc1Sel = "src1_sel:";
  static constexpr std::string_view kDstSel = "dst_sel:";
  static constexpr std::string_view kDstUnused = "dst_unused:";

  void parse(const std::vector<std::string_view> &tokens, size_t startIndex) {
    for (size_t i = startIndex; i < tokens.size(); ++i) {
      std::string_view token = tokens[i];
      if (token.starts_with(kSrc0Sel)) {
        src0 = getSelParams(token.substr(kSrc0Sel.size()));
        enabled = true;
      } else if (token.starts_with(kSrc1Sel)) {
        src1 = getSelParams(token.substr(kSrc1Sel.size()));
        enabled = true;
      } else if (token.starts_with(kDstSel)) {
        dst = getSelParams(token.substr(kDstSel.size()));
        enabled = true;
      } else if (token.starts_with(kDstUnused)) {
        dstUnused = getUnusedMode(token.substr(kDstUnused.size()));
        enabled = true;
      }
    }
  }

  static uint32_t applySrc(uint32_t val, Selection sel) {
    return (val >> sel.shift) & sel.mask;
  }

  uint32_t applyDst(SelectionOperand selectionOperand) const {
    // uint32_t result, uint32_t oldVal) const {
    uint32_t windowedResult = selectionOperand.result & dst.mask;
    uint32_t dataToWrite = windowedResult << dst.shift;
    uint32_t writeMask = dst.mask << dst.shift;

    if (dstUnused == UnusedMode::PRESERVE) {
      return (selectionOperand.previous & ~writeMask) | dataToWrite;
    }
    if (dstUnused == UnusedMode::PAD) {
      return dataToWrite;
    }
    throw std::runtime_error("SDWA SEXT mode not implemented");
    return dataToWrite;
  }

private:
  static Selection getSelParams(std::string_view sel) {
    if (sel == "WORD_0") {
      return {0xFFFF, 0};
    }
    if (sel == "WORD_1") {
      return {0xFFFF, 16};
    }
    if (sel == "BYTE_0") {
      return {0xFF, 0};
    }
    if (sel == "BYTE_1") {
      return {0xFF, 8};
    }
    if (sel == "BYTE_2") {
      return {0xFF, 16};
    }
    if (sel == "BYTE_3") {
      return {0xFF, 24};
    }
    return {0xFFFFFFFF, 0};
  }

  static UnusedMode getUnusedMode(std::string_view val) {
    if (val == "UNUSED_PAD") {
      return UnusedMode::PAD;
    }
    if (val == "UNUSED_SEXT") {
      return UnusedMode::SEXT;
    }
    if (val == "UNUSED_PRESERVE") {
      return UnusedMode::PRESERVE;
    }
    return UnusedMode::PAD;
  }
};

// VOP1 instructions
template <typename T_DST, typename T_SRC, typename OpFunc>
class VOP1_Op : public Instruction {
  OpFunc func;

public:
  VOP1_Op(OpFunc func) : func(std::move(func)) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    auto dst = wave.getFirstRegister(partitioned[1]);
    assert(dst.type == CommonRegister::Type::VGPR &&
           "VOP1 destination must be VGPR");
    auto src = wave.parseOperand<T_SRC>(partitioned[2]);

    auto operation = this->func;
    return [&wave, dst, src, operation]() {
      auto run = [&wave, dst, src, operation](int lane) {
        T_SRC inVal = wave.getValue(src, lane);
        T_DST outVal = operation(inVal);
        if constexpr (sizeof(T_DST) == 8) {
          wave.setVgpr64(dst.index, lane, std::bit_cast<uint64_t>(outVal));
        } else {
          wave.setVgpr(dst.index, lane, std::bit_cast<uint32_t>(outVal));
        }
      };
      wave.runExecConditionedForLanes(run);
      return wave.getPc() + 1;
    };
  }
};

template <typename T_DST, typename T_SRC>
const auto doStaticCast = [](T_SRC inVal) { return static_cast<T_DST>(inVal); };

template <typename T_DST, typename T_SRC>
class VOP1_VConvert
    : public VOP1_Op<T_DST, T_SRC, decltype(doStaticCast<T_DST, T_SRC>)> {
public:
  VOP1_VConvert()
      : VOP1_Op<T_DST, T_SRC, decltype(doStaticCast<T_DST, T_SRC>)>(
            doStaticCast<T_DST, T_SRC>) {}
};

// VOP2 32-bit math
template <typename T, typename OpFunc> class VOP2_Math32 : public Instruction {
  static_assert(sizeof(T) == 4, "VOP2_Math32 only supports 32-bit types");

  OpFunc op;
  InstructionMode mode;

public:
  VOP2_Math32(OpFunc op, InstructionMode mode = InstructionMode::Standard)
      : op(std::move(op)), mode(mode) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() >= 4);

    auto dst = wave.getFirstRegister(partitioned[1]);
    assert(dst.type == CommonRegister::Type::VGPR && "Dst must be VGPR");

    auto src0 = wave.parseOperand<T>(partitioned[2]);
    auto src1 = wave.parseOperand<T>(partitioned[3]);

    SdwaControl sdwa;
    if (mode == InstructionMode::SDWA) {
      sdwa.parse(partitioned, 4);
    } else if (mode == InstructionMode::DPP) {
      throw std::runtime_error("DPP mode not yet implemented");
    }

    auto operation = this->op;
    int dstIdx = dst.index;

    return [&wave, dstIdx, src0, src1, operation, sdwa]() {
      auto run = [&](int lane) {
        uint32_t raw0 = std::bit_cast<uint32_t>(wave.getValue(src0, lane));
        uint32_t raw1 = std::bit_cast<uint32_t>(wave.getValue(src1, lane));

        if (sdwa.enabled) {
          raw0 = SdwaControl::applySrc(raw0, sdwa.src0);
          raw1 = SdwaControl::applySrc(raw1, sdwa.src1);
        }

        T val0 = std::bit_cast<T>(raw0);
        T val1 = std::bit_cast<T>(raw1);
        T resultT = operation(val0, val1, wave);

        uint32_t resultRaw = std::bit_cast<uint32_t>(resultT);

        if (sdwa.enabled) {
          uint32_t oldDst = wave.getVgpr(dstIdx, lane);
          SdwaControl::SelectionOperand selectionOperand(resultRaw, oldDst);
          uint32_t finalVal = sdwa.applyDst(selectionOperand);
          wave.setVgpr(dstIdx, lane, finalVal);
        } else {
          wave.setVgpr(dstIdx, lane, resultRaw);
        }
      };

      wave.runExecConditionedForLanes(run);
      return wave.getPc() + 1;
    };
  }
};

// VOP2 64-bit math
template <typename T, typename OpFunc> class VOP2_Math64 : public Instruction {
  static_assert(sizeof(T) == sizeof(uint64_t),
                "VOP2_Math64 only supports 64-bit types");

  OpFunc op;

public:
  VOP2_Math64(OpFunc op) : op(std::move(op)) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() >= 4);

    auto dst = wave.getFirstRegister(partitioned[1]);
    assert(dst.type == CommonRegister::Type::VGPR && "Dst must be VGPR");

    auto src0 = wave.parseOperand<T>(partitioned[2]);
    auto src1 = wave.parseOperand<T>(partitioned[3]);

    auto operation = this->op;
    int dstIdx = dst.index;

    return [&wave, dstIdx, src0, src1, operation]() {
      auto run = [&](int lane) {
        T val0 = wave.getValue(src0, lane);
        T val1 = wave.getValue(src1, lane);
        T result = operation(val0, val1, wave);
        wave.setVgpr64(dstIdx, lane, std::bit_cast<uint64_t>(result));
      };

      wave.runExecConditionedForLanes(run);
      return wave.getPc() + 1;
    };
  }
};

class VOP2_FMAC_F32 : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() >= 4);

    auto dst = wave.getFirstRegister(partitioned[1]);
    assert(dst.type != CommonRegister::Type::SGPR && "Dst must be not be SGPR");

    auto src0 = wave.parseOperand<float>(partitioned[2]);
    auto src1 = wave.parseOperand<float>(partitioned[3]);

    int dstIdx = dst.index;

    return [&wave, dstIdx, src0, src1]() {
      auto run = [&](int lane) {
        float s0 = wave.getValue(src0, lane);
        float s1 = wave.getValue(src1, lane);
        uint32_t rawAcc = wave.getVgpr(dstIdx, lane);
        float acc = std::bit_cast<float>(rawAcc);
        float result = std::fma(s0, s1, acc);
        wave.setVgpr(dstIdx, lane, std::bit_cast<uint32_t>(result));
      };

      wave.runExecConditionedForLanes(run);
      return wave.getPc() + 1;
    };
  }
};

// VOP3 instructions
template <typename T, typename OpFunc> class VOP3_Op : public Instruction {
  OpFunc op;

public:
  VOP3_Op(OpFunc op) : op(std::move(op)) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() >= 5);

    auto dst = wave.getFirstRegister(partitioned[1]);
    assert(dst.type != CommonRegister::Type::SGPR &&
           "VOP3 destination must not be sgpr");

    auto src0 = wave.parseOperand<T>(partitioned[2]);
    auto src1 = wave.parseOperand<T>(partitioned[3]);
    auto src2 = wave.parseOperand<T>(partitioned[4]);
    auto operation = this->op;

    return [&wave, dst, src0, src1, src2, operation]() {
      auto run = [&wave, dst, src0, src1, src2, operation](int lane) {
        T v0 = wave.getValue(src0, lane);
        T v1 = wave.getValue(src1, lane);
        T v2 = wave.getValue(src2, lane);
        T result = operation(v0, v1, v2, wave);
        if constexpr (sizeof(T) == 8) {
          wave.setVgpr64(dst.index, lane, std::bit_cast<uint64_t>(result));
        } else {
          wave.setVgpr(dst.index, lane, std::bit_cast<uint32_t>(result));
        }
      };
      wave.runExecConditionedForLanes(run);
      return wave.getPc() + 1;
    };
  }
};

// VOP3B instructions (carry out)
template <typename T, typename Src2T, typename OpFunc>
class VOP3B_Op : public Instruction {
  OpFunc op;

public:
  VOP3B_Op(OpFunc op) : op(std::move(op)) {}
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() >= 5 && "VOP3B requires at least 5 tokens");

    auto vDst = wave.getFirstRegister(partitioned[1]);
    assert(vDst.type == CommonRegister::Type::VGPR);
    auto sDst = wave.getFirstRegister(partitioned[2]);

    auto src0 = wave.parseOperand<T>(partitioned[3]);
    auto src1 = wave.parseOperand<T>(partitioned[4]);

    Operand<Src2T> src2{true, {}, static_cast<Src2T>(0)};
    if (partitioned.size() > 5) {
      src2 = wave.parseOperand<Src2T>(partitioned[5]);
    }

    auto operation = this->op;

    return [&wave, vDst, sDst, src0, src1, src2, operation]() {
      uint64_t carryMask = 0;

      auto capturingRun = [&](int lane) {
        T v0 = wave.getValue(src0, lane);
        T v1 = wave.getValue(src1, lane);
        Src2T v2 = wave.getValue(src2, lane);

        auto [result, carry] = operation(v0, v1, v2, lane, wave);

        if constexpr (sizeof(T) == 8) {
          wave.setVgpr64(vDst.index, lane, std::bit_cast<uint64_t>(result));
        } else {
          wave.setVgpr(vDst.index, lane, std::bit_cast<uint32_t>(result));
        }

        if (carry) {
          carryMask |= (1ULL << lane);
        }
      };

      wave.runExecConditionedForLanes(capturingRun);

      // Carry-out is 1 bit per lane: 32-bit on wave-32, 64-bit on wave-64.
      if (wave.getWaveSize() <= 32) {
        uint32_t oldMask = wave.getSgpr(sDst.index);
        uint32_t execMask = wave.getExecU32();
        uint32_t cm = static_cast<uint32_t>(carryMask);
        wave.setSgpr(sDst.index, (oldMask & ~execMask) | (cm & execMask));
      } else {
        uint64_t oldMask = wave.getSgpr64(sDst.index);
        uint64_t execMask = wave.getExecU64();
        uint64_t finalMask = (oldMask & ~execMask) | (carryMask & execMask);
        wave.setSgpr64(sDst.index, finalMask);
      }

      return wave.getPc() + 1;
    };
  }
};

// VOPC vector comparisons
template <typename T, typename Op, bool WriteExec = false>
class VOPC_Compare : public Instruction {
  Op op;
  InstructionMode mode;

public:
  VOPC_Compare(Op op, InstructionMode mode = InstructionMode::Standard)
      : op(std::move(op)), mode(mode) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);

    // VOPC operand layout differs between CDNA and RDNA for v_cmpx:
    //   CDNA:  v_cmpx_* dst, src0, src1  (4 tokens, writes dst AND exec)
    //   RDNA:  v_cmpx_* src0, src1       (3 tokens, writes exec only)
    // Plain v_cmp always has 4 tokens on both architectures.
    bool implicitExecDest = (partitioned.size() == 3);
    assert(partitioned.size() >= 3 && "VOPC requires at least Src0, Src1");

    CommonRegister dst{CommonRegister::Type::SGPR, 0};
    Operand<T> src0{true, {}, static_cast<T>(0)};
    Operand<T> src1{true, {}, static_cast<T>(0)};
    int sdwaStart = 4;

    if (implicitExecDest) {
      src0 = wave.parseOperand<T>(partitioned[1]);
      src1 = wave.parseOperand<T>(partitioned[2]);
      sdwaStart = 3;
    } else {
      dst = wave.getFirstRegister(partitioned[1]);
      assert(dst.type == CommonRegister::Type::SGPR &&
             "VOPC Dest must be SGPR/VCC");
      src0 = wave.parseOperand<T>(partitioned[2]);
      src1 = wave.parseOperand<T>(partitioned[3]);
    }

    SdwaControl sdwa;
    if (mode == InstructionMode::SDWA) {
      sdwa.parse(partitioned, sdwaStart);
    }

    auto operation = this->op;
    auto waveSize = wave.getWaveSize();

    return [&wave, dst, src0, src1, operation, waveSize, sdwa,
            implicitExecDest]() {
      uint64_t resultMask = 0;

      for (int lane = 0; lane < waveSize; ++lane) {
        T val0 = wave.getValue(src0, lane);
        T val1 = wave.getValue(src1, lane);

        uint32_t u0{0};
        uint32_t u1{0};
        if constexpr (std::is_floating_point_v<T>) {
          u0 = std::bit_cast<uint32_t>(val0);
          u1 = std::bit_cast<uint32_t>(val1);
        } else {
          u0 = static_cast<uint32_t>(val0);
          u1 = static_cast<uint32_t>(val1);
        }

        if (sdwa.enabled) {
          u0 = SdwaControl::applySrc(u0, sdwa.src0);
          u1 = SdwaControl::applySrc(u1, sdwa.src1);
        }

        bool pass{false};
        if constexpr (std::is_floating_point_v<T>) {
          pass = operation(std::bit_cast<T>(u0), std::bit_cast<T>(u1));
        } else {
          pass = operation(static_cast<T>(u0), static_cast<T>(u1));
        }

        if (pass) {
          resultMask |= (1ULL << lane);
        }
      }

      uint64_t currentExec = wave.getExecU64();
      resultMask &= currentExec;

      // Result mask is 1 bit per lane: 32-bit on wave-32, 64-bit on wave-64.
      if (wave.getWaveSize() <= 32) {
        if (!implicitExecDest) {
          wave.setSgpr(dst.index, static_cast<uint32_t>(resultMask));
        }
        if constexpr (WriteExec) {
          wave.setExecU32(static_cast<uint32_t>(resultMask));
        }
      } else {
        if (!implicitExecDest) {
          wave.setSgpr64(dst.index, resultMask);
        }
        if constexpr (WriteExec) {
          wave.setExecU64(resultMask);
        }
      }

      return wave.getPc() + 1;
    };
  }
};

template <typename T_Storage, typename T_Math>
class VOP3P_PackedBinary : public Instruction {
  std::function<T_Math(T_Math, T_Math)> op;

  struct PackedModifiers {
    std::vector<int> opSel = {0, 0};
    std::vector<int> opSelHi = {1, 1};
    std::vector<int> negLo = {0, 0};
    std::vector<int> negHi = {0, 0};

    static PackedModifiers
    parse(const std::map<std::string, std::vector<int>> &mods) {
      PackedModifiers pm;
      if (mods.contains("op_sel")) {
        pm.opSel = mods.at("op_sel");
      }
      if (mods.contains("op_sel_hi")) {
        pm.opSelHi = mods.at("op_sel_hi");
      }
      if (mods.contains("neg_lo")) {
        pm.negLo = mods.at("neg_lo");
      }
      if (mods.contains("neg_hi")) {
        pm.negHi = mods.at("neg_hi");
      }
      return pm;
    }
  };

  struct ExtractCfg {
    int sel;  // 0 or 1
    bool neg; // true or false
  };

  static T_Math extract(T_Storage pack, ExtractCfg cfg) {
    T_Math val;
    if constexpr (sizeof(T_Storage) == 8) {
      uint32_t raw = (cfg.sel == 1) ? (pack >> 32) : (pack & 0xFFFFFFFF);
      val = std::bit_cast<float>(raw);
    } else {
      val = 0.0f; // Placeholder
    }
    return (cfg.neg) ? -val : val;
  }

  struct RuntimeConfig {
    ExtractCfg src0_lo;
    ExtractCfg src1_lo;
    ExtractCfg src0_hi;
    ExtractCfg src1_hi;

    static RuntimeConfig build(const PackedModifiers &mods) {
      return {// Src0 Lo
              {(mods.opSel.empty() ? 0 : mods.opSel[0]),
               (mods.negLo.empty() ? false : (bool)mods.negLo[0])},
              // Src1 Lo
              {(mods.opSel.size() < 2 ? 0 : mods.opSel[1]),
               (mods.negLo.size() < 2 ? false : (bool)mods.negLo[1])},
              // Src0 Hi
              {(mods.opSelHi.empty() ? 1 : mods.opSelHi[0]),
               (mods.negHi.empty() ? false : (bool)mods.negHi[0])},
              // Src1 Hi
              {(mods.opSelHi.size() < 2 ? 1 : mods.opSelHi[1]),
               (mods.negHi.size() < 2 ? false : (bool)mods.negHi[1])}};
    }
  };

public:
  VOP3P_PackedBinary(std::function<T_Math(T_Math, T_Math)> op)
      : op(std::move(op)) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() >= 4);

    auto dst = wave.getFirstRegister(partitioned[1]);
    auto src0 = wave.parseOperand<T_Storage>(partitioned[2]);
    auto src1 = wave.parseOperand<T_Storage>(partitioned[3]);

    auto rawModifiers = parsePackedModifiers(line);
    PackedModifiers modsVec = PackedModifiers::parse(rawModifiers);

    RuntimeConfig cfg = RuntimeConfig::build(modsVec);

    auto operation = this->op;
    int dstIdx = dst.index;

    return [&wave, dstIdx, src0, src1, operation, cfg]() {
      auto runLane = [&](int lane) {
        T_Storage raw0 = wave.getValue(src0, lane);
        T_Storage raw1 = wave.getValue(src1, lane);

        // The logic is now linear and cleaner (No branching!)
        T_Math lo0 = extract(raw0, cfg.src0_lo);
        T_Math lo1 = extract(raw1, cfg.src1_lo);
        T_Math resLo = operation(lo0, lo1);

        T_Math hi0 = extract(raw0, cfg.src0_hi);
        T_Math hi1 = extract(raw1, cfg.src1_hi);
        T_Math resHi = operation(hi0, hi1);

        if constexpr (sizeof(T_Storage) == 8) {
          uint64_t final =
              (static_cast<uint64_t>(std::bit_cast<uint32_t>(resHi)) << 32) |
              std::bit_cast<uint32_t>(resLo);
          wave.setVgpr64(dstIdx, lane, final);
        }
      };

      wave.runExecConditionedForLanes(runLane);
      return wave.getPc() + 1;
    };
  }
};

class VLshlAddU64 : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    auto dst = wave.getFirstRegister(partitioned[1]);
    auto src0 = wave.parseOperand<uint64_t>(partitioned[2]);
    auto shiftOp = wave.parseOperand<uint32_t>(partitioned[3]);
    auto src1 = wave.parseOperand<uint64_t>(partitioned[4]);

    return [&wave, dst, src0, shiftOp, src1]() {
      auto run = [src0, src1, dst, shiftOp, &wave](int lane) {
        uint64_t val0 = wave.getValue(src0, lane);
        uint32_t shift = wave.getValue(shiftOp, lane) & 0x3F;
        uint64_t base = wave.getValue(src1, lane);
        uint64_t result = (val0 << shift) + base;
        wave.setVgpr64(dst.index, lane, result);
      };
      wave.runExecConditionedForLanes(run);
      return wave.getPc() + 1;
    };
  }
};

class VReadFirstLane : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    auto dst = wave.getFirstRegister(partitioned[1]);
    auto src = wave.getFirstRegister(partitioned[2]);

    assert(dst.type == CommonRegister::Type::SGPR);
    assert(src.type == CommonRegister::Type::VGPR);

    return [&wave, dst, src]() {
      uint64_t exec = wave.getExecU64();
      int lane = (exec != 0) ? std::countr_zero(exec) : 0;
      uint32_t value = wave.getVgpr(src.index, lane);
      wave.setSgpr(dst.index, value);
      return wave.getPc() + 1;
    };
  }
};

class VCndMask : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 5);

    auto dst = wave.getFirstRegister(partitioned[1]);
    auto src0 = wave.parseOperand<uint32_t>(partitioned[2]);
    auto src1 = wave.parseOperand<uint32_t>(partitioned[3]);
    auto maskOp = wave.parseOperand<uint64_t>(partitioned[4]);

    return [maskOp, &wave, dst, src0, src1]() {
      auto run = [src0, src1, &wave, dst, maskOp](int lane) {
        uint64_t maskValue = wave.getSgprOrLiteralValue(maskOp);
        bool condition = (maskValue >> lane) & 1;
        uint32_t val0 = wave.getValue(src0, lane);
        uint32_t val1 = wave.getValue(src1, lane);
        uint32_t result = condition ? val1 : val0;
        wave.setVgpr(dst.index, lane, result);
      };
      wave.runExecConditionedForLanes(run);
      return wave.getPc() + 1;
    };
  }
};

// Registration helpers

template <typename InstT> struct Register {
  template <typename... Args>
  Register(const std::string &name, Args &&...args) {
    InstructionRegistry::instance().add(
        name, std::make_unique<InstT>(std::forward<Args>(args)...));
  }
};

template <typename T, typename T2 = T> struct RegisterVOP1 {
  template <typename Op> RegisterVOP1(const std::string &name, Op op) {
    // Forward declaration
    using InstType = VOP1_Op<T, T2, Op>;
    InstructionRegistry::instance().add(name, std::make_unique<InstType>(op));
  }
};

template <typename T> struct RegisterVOP2_32 {
  template <typename Op>
  RegisterVOP2_32(const std::string &name, Op op,
                  InstructionMode mode = InstructionMode::Standard) {
    using InstType = VOP2_Math32<T, Op>;
    InstructionRegistry::instance().add(name,
                                        std::make_unique<InstType>(op, mode));
  }
};

template <typename T> struct RegisterVOP2_64 {
  template <typename Op> RegisterVOP2_64(const std::string &name, Op op) {
    using InstType = VOP2_Math64<T, Op>;
    InstructionRegistry::instance().add(name, std::make_unique<InstType>(op));
  }
};

template <typename T> struct RegisterVOP3a {
  template <typename Op> RegisterVOP3a(const std::string &name, Op op) {
    using InstType = VOP3_Op<T, Op>;
    InstructionRegistry::instance().add(name, std::make_unique<InstType>(op));
  }
};

template <typename T> struct RegisterVComp {
  template <typename Op> RegisterVComp(const std::string &name, Op op) {
    using InstType = VOPC_Compare<T, Op>;
    InstructionRegistry::instance().add(name, std::make_unique<InstType>(op));
  }
};

template <typename T> struct RegisterVCompX {
  template <typename Op> RegisterVCompX(const std::string &name, Op op) {
    using InstType = VOPC_Compare<T, Op, true>;
    InstructionRegistry::instance().add(name, std::make_unique<InstType>(op));
  }
};

template <typename T, typename T2, typename Op> struct RegisterVOP3b {
  RegisterVOP3b(const std::string &name, Op op) {
    using InstType = VOP3B_Op<T, T2, Op>;
    InstructionRegistry::instance().add(name, std::make_unique<InstType>(op));
  }
};

// 3. v_mad_u64_u32: D.u64 = S0.u32 * S1.u32 + S2.u64; carry out to SGPR pair
class VMadU64U32 : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    // v_mad_u64_u32 v[dst:dst+1], s[sdst:sdst+1], src0, src1, src2
    assert(partitioned.size() >= 6);

    auto vDst = wave.getFirstRegister(partitioned[1]);
    assert(vDst.type == CommonRegister::Type::VGPR);
    auto sDst = wave.getFirstRegister(partitioned[2]);

    auto src0 = wave.parseOperand<uint32_t>(partitioned[3]);
    auto src1 = wave.parseOperand<uint32_t>(partitioned[4]);
    auto src2 = wave.parseOperand<uint64_t>(partitioned[5]);

    return [&wave, vDst, sDst, src0, src1, src2]() {
      uint64_t carryMask = 0;

      auto run = [&](int lane) {
        uint64_t a = wave.getValue(src0, lane);
        uint64_t b = wave.getValue(src1, lane);
        uint64_t c = wave.getValue(src2, lane);

        auto [result, carry] = mulAddU64(a, b, c);

        wave.setVgpr64(vDst.index, lane, result);
        if (carry) {
          carryMask |= (1ULL << lane);
        }
      };

      wave.runExecConditionedForLanes(run);

      if (wave.getWaveSize() <= 32) {
        uint32_t oldMask = wave.getSgpr(sDst.index);
        uint32_t execMask = wave.getExecU32();
        uint32_t cm = static_cast<uint32_t>(carryMask);
        wave.setSgpr(sDst.index, (oldMask & ~execMask) | (cm & execMask));
      } else {
        uint64_t oldMask = wave.getSgpr64(sDst.index);
        uint64_t execMask = wave.getExecU64();
        uint64_t finalMask = (oldMask & ~execMask) | (carryMask & execMask);
        wave.setSgpr64(sDst.index, finalMask);
      }

      return wave.getPc() + 1;
    };
  }
};
static Register<VMadU64U32> v_mad_u64("v_mad_u64_u32");

// VOP1 conversions and moves
static Register<VOP1_VConvert<double, float>> v_cvt_d_f("v_cvt_f64_f32");
static Register<VOP1_VConvert<float, double>> v_cvt_f_d("v_cvt_f32_f64");
static Register<VOP1_VConvert<double, uint32_t>> v_cvt_d_u("v_cvt_f64_u32");
static Register<VOP1_VConvert<uint32_t, double>> v_cvt_u_d("v_cvt_u32_f64");
static Register<VOP1_VConvert<float, uint32_t>> v_cvt_f_u("v_cvt_f32_u32");
static Register<VOP1_VConvert<float, int32_t>> v_cvt_f_i("v_cvt_f32_i32");
// v_cvt_u32_f32: clamp to [0, UINT_MAX]. Negative and NaN map to 0.
static RegisterVOP1<uint32_t, float>
    v_cvt_u_f("v_cvt_u32_f32", [](float f) -> uint32_t {
      if (std::isnan(f) || f <= 0.0f) {
        return 0;
      }
      if (f >= 4294967296.0f) {
        return 0xFFFFFFFF;
      }
      return static_cast<uint32_t>(f);
    });

// F16 <-> F32 conversions: source/dest are 32-bit VGPRs, f16 in low 16 bits.
static RegisterVOP1<uint32_t> v_cvt_f32_f16("v_cvt_f32_f16", [](uint32_t v) {
  return std::bit_cast<uint32_t>(f16ToFloat(static_cast<uint16_t>(v & 0xFFFF)));
});
static RegisterVOP1<uint32_t> v_cvt_f16_f32("v_cvt_f16_f32", [](uint32_t v) {
  return static_cast<uint32_t>(floatToF16(std::bit_cast<float>(v)));
});

static Register<VOP1_VConvert<uint32_t, uint32_t>> v_mov_32("v_mov_b32");
static Register<VOP1_VConvert<uint64_t, uint64_t>> v_mov_64("v_mov_b64");
static Register<VOP1_VConvert<uint32_t, uint32_t>> v_acc_wr("v_accvgpr_write");
static Register<VOP1_VConvert<uint32_t, uint32_t>>
    v_acc_rd("v_accvgpr_read_b32");

// GPU reciprocal: the SFU computes 1/x via lookup + Newton-Raphson, accurate
// to 0.5 ULP but not necessarily IEEE correctly-rounded. We use IEEE division
// and flush denormal results to zero. The result matches hardware for most
// inputs; edge cases at the 0.5 ULP boundary may differ by 1 ULP.
const auto gpuRcpF32 = [](float a) -> float {
  float f = 1.0f / a;
  // Flush denormal results to zero
  uint32_t bits = std::bit_cast<uint32_t>(f);
  if ((bits & 0x7F800000) == 0 && (bits & 0x007FFFFF) != 0) {
    return std::copysign(0.0f, f);
  }
  return f;
};
static RegisterVOP1<float> v_rcp_f32("v_rcp_f32", gpuRcpF32);
static RegisterVOP1<float> v_rcp_iflag_f32("v_rcp_iflag_f32", gpuRcpF32);
static RegisterVOP1<double> v_rcp_f64("v_rcp_f64",
                                      [](double a) { return 1.0 / a; });
static RegisterVOP1<float> v_exp_f32("v_exp_f32",
                                     [](float a) { return std::exp2(a); });

// VOP2 math
static RegisterVOP2_64<double>
    v_add_f64("v_add_f64", [](double a, double b, Wave &) { return a + b; });
static RegisterVOP2_32<float>
    v_add_f32("v_add_f32", [](float a, float b, Wave &) { return a + b; });
static RegisterVOP2_32<uint32_t> v_sub_u32("v_sub_u32",
                                           [](uint32_t a, uint32_t b, Wave &) {
                                             return a - b;
                                           });
static RegisterVOP2_32<uint32_t> v_subrev_u32("v_subrev_u32",
                                              [](uint32_t a, uint32_t b,
                                                 Wave &) { return b - a; });
static RegisterVOP2_32<float>
    v_mul_f32("v_mul_f32", [](float a, float b, Wave &) { return a * b; });
static RegisterVOP2_32<float> v_max_f32("v_max_f32",
                                        [](float a, float b, Wave &) {
                                          return std::fmax(a, b);
                                        });
static RegisterVOP2_32<float> v_min_f32("v_min_f32",
                                        [](float a, float b, Wave &) {
                                          return std::fmin(a, b);
                                        });
static RegisterVOP2_64<double>
    v_mul_f64("v_mul_f64", [](double a, double b, Wave &) { return a * b; });
// v_pack_b32_f16: pack two f16 halves into one b32.
// dst[15:0]  = src0[15:0]  (low  f16)
// dst[31:16] = src1[15:0]  (high f16)
static RegisterVOP2_32<uint32_t>
    v_pack_b32_f16("v_pack_b32_f16", [](uint32_t a, uint32_t b, Wave &) {
      return (a & 0xFFFF) | ((b & 0xFFFF) << 16);
    });
static RegisterVOP2_32<uint32_t>
    v_and_32("v_and_b32", [](uint32_t a, uint32_t b, Wave &) { return a & b; });
static RegisterVOP2_32<uint32_t>
    v_or_32("v_or_b32", [](uint32_t a, uint32_t b, Wave &) { return a | b; });
static auto or_sdwa_op = [](uint32_t a, uint32_t b, Wave &) { return a | b; };
static RegisterVOP2_32<uint32_t> v_or_sdwa("v_or_b32_sdwa", or_sdwa_op,
                                           InstructionMode::SDWA);

static RegisterVOP2_32<uint32_t> v_add_u32("v_add_u32",
                                           [](uint32_t a, uint32_t b, Wave &) {
                                             return a + b;
                                           });
static RegisterVOP2_64<uint64_t>
    v_lshlrev_64("v_lshlrev_b64", [](uint64_t s0, uint64_t s1, Wave &) {
      return s1 << (s0 & 0x3F);
    });
static RegisterVOP2_32<uint32_t>
    v_lshrrev_32("v_lshrrev_b32", [](uint32_t s0, uint32_t s1, Wave &) {
      return s1 >> (s0 & 0x1F);
    });
static RegisterVOP2_32<uint32_t>
    v_lshlrev_32("v_lshlrev_b32", [](uint32_t s0, uint32_t s1, Wave &) {
      return s1 << (s0 & 0x1F);
    });
static RegisterVOP2_32<uint32_t>
    v_xor_32("v_xor_b32", [](uint32_t a, uint32_t b, Wave &) { return a ^ b; });
static RegisterVOP2_32<int32_t> v_max("v_max_i32",
                                      [](int32_t a, int32_t b, Wave &) {
                                        return std::max(a, b);
                                      });
static RegisterVOP2_32<int32_t> v_min("v_min_i32",
                                      [](int32_t a, int32_t b, Wave &) {
                                        return std::min(a, b);
                                      });
static RegisterVOP2_32<uint32_t> v_mul_lo("v_mul_lo_u32",
                                          [](uint32_t a, uint32_t b, Wave &) {
                                            return a * b;
                                          });
// v_mul_hi_u32
// D0.u32 = 32'U((64'U(S0.u32) * 64'U(S1.u32)) >> 32U)
static RegisterVOP2_32<uint32_t>
    v_mul_hi("v_mul_hi_u32", [](uint32_t a, uint32_t b, Wave &) {
      uint64_t res = (uint64_t)a * (uint64_t)b;
      return static_cast<uint32_t>(res >> 32);
    });

static RegisterVOP2_32<uint32_t>
    v_ashrrev("v_ashrrev_i32", [](uint32_t s, uint32_t v, Wave &) {
      return std::bit_cast<uint32_t>(std::bit_cast<int32_t>(v) >> (s & 0x1F));
    });
static RegisterVOP2_32<uint32_t>
    v_mul_u24("v_mul_u32_u24", [](uint32_t a, uint32_t b, Wave &) {
      return (a & 0xFFFFFF) * (b & 0xFFFFFF);
    });
static RegisterVOP2_32<int32_t>
    v_mul_i24("v_mul_i32_i24", [](int32_t a, int32_t b, Wave &) {
      return ((a << 8) >> 8) * ((b << 8) >> 8);
    });

static Register<VOP2_FMAC_F32> v_fmac("v_fmac_f32");
static Register<VOP3P_PackedBinary<uint64_t, float>>
    v_pk_mul("v_pk_mul_f32", [](float a, float b) { return a * b; });
static Register<VCndMask> v_cndmask("v_cndmask_b32");
static Register<VReadFirstLane> v_readfirst("v_readfirstlane_b32");
static Register<VLshlAddU64> v_lshl_add64("v_lshl_add_u64");

// VOP3 math
static RegisterVOP3a<float> v_fma_f32("v_fma_f32",
                                      [](float a, float b, float c, Wave &) {
                                        return std::fma(a, b, c);
                                      });

// VOP3 and carry ops
static RegisterVOP3a<uint32_t>
    v_lshl_or("v_lshl_or_b32", [](uint32_t a, uint32_t b, uint32_t c, Wave &) {
      return (a << (b & 0x1F)) | c;
    });
static RegisterVOP3a<uint32_t>
    v_add_lshl("v_add_lshl_u32", [](uint32_t a, uint32_t b, uint32_t c,
                                    Wave &) { return (a + b) << (c & 0x1F); });
static RegisterVOP3a<uint32_t> v_add3("v_add3_u32",
                                      [](uint32_t a, uint32_t b, uint32_t c,
                                         Wave &) { return a + b + c; });
static RegisterVOP3a<uint32_t> v_or3("v_or3_b32",
                                     [](uint32_t a, uint32_t b, uint32_t c,
                                        Wave &) { return a | b | c; });
static RegisterVOP3a<uint32_t> v_and_or("v_and_or_b32",
                                        [](uint32_t a, uint32_t b, uint32_t c,
                                           Wave &) { return (a & b) | c; });
static RegisterVOP3a<uint32_t>
    v_lshl_add("v_lshl_add_u32", [](uint32_t a, uint32_t b, uint32_t c,
                                    Wave &) { return (a << (b & 0x1F)) + c; });
// v_mad_u32_u24: D.u32 = S0[23:0] * S1[23:0] + S2. Multiply-add using
// the low 24 bits of each source.
static RegisterVOP3a<uint32_t>
    v_mad_u32_u24("v_mad_u32_u24",
                  [](uint32_t a, uint32_t b, uint32_t c, Wave &) -> uint32_t {
                    return (a & 0xFFFFFF) * (b & 0xFFFFFF) + c;
                  });
static RegisterVOP3a<uint32_t>
    v_bfe("v_bfe_u32", [](uint32_t data, uint32_t off, uint32_t w, Wave &) {
      uint32_t width = w & 0x1F;
      return width == 0 ? 0 : (data >> (off & 0x1F)) & ((1u << width) - 1);
    });

// v_perm_b32: byte permute from the 64-bit value {src0, src1}.
// The 8 input bytes are: in[0..3] = src1 bytes, in[4..7] = src0 bytes.
// Each byte of the selector (src2) picks one output byte:
//   0-7:  select input byte in[sel]
//   8:    sign-extend in[1] (replicate bit 7 of byte 1)
//   9:    sign-extend in[3]
//   10:   sign-extend in[5]
//   11:   sign-extend in[7]
//   12:   0x00
//   >=13: 0xFF
static RegisterVOP3a<uint32_t>
    v_perm("v_perm_b32",
           [](uint32_t src0, uint32_t src1, uint32_t sel, Wave &) -> uint32_t {
             // Concatenate into 8-byte array: in[0..3]=src1, in[4..7]=src0
             uint8_t in[8];
             for (int j = 0; j < 4; ++j) {
               in[j] = (src1 >> (j * 8)) & 0xFF;
               in[j + 4] = (src0 >> (j * 8)) & 0xFF;
             }
             uint32_t result = 0;
             for (int i = 0; i < 4; ++i) {
               uint32_t s = (sel >> (i * 8)) & 0xFF;
               uint8_t byte;
               if (s <= 7) {
                 byte = in[s];
               } else if (s == 8) {
                 byte = (in[1] & 0x80) ? 0xFF : 0x00;
               } else if (s == 9) {
                 byte = (in[3] & 0x80) ? 0xFF : 0x00;
               } else if (s == 10) {
                 byte = (in[5] & 0x80) ? 0xFF : 0x00;
               } else if (s == 11) {
                 byte = (in[7] & 0x80) ? 0xFF : 0x00;
               } else if (s == 12) {
                 byte = 0x00;
               } else {
                 byte = 0xFF; // sel >= 13
               }
               result |= static_cast<uint32_t>(byte) << (i * 8);
             }
             return result;
           });

using AddcOp = std::function<std::pair<uint32_t, bool>(uint32_t, uint32_t,
                                                       uint64_t, int, Wave &)>;
static auto addc_func = [](uint32_t s0, uint32_t s1, uint64_t cMask, int lane,
                           Wave &) -> std::pair<uint32_t, bool> {
  uint64_t cin = (cMask >> lane) & 1;
  uint64_t t = (uint64_t)s0 + s1 + cin;
  return {(uint32_t)t, (t >> 32) != 0};
};
static RegisterVOP3b<uint32_t, uint64_t, AddcOp> v_addc_co("v_addc_co_u32",
                                                           addc_func);
// RDNA-style alias
static RegisterVOP3b<uint32_t, uint64_t, AddcOp> v_add_co_ci("v_add_co_ci_u32",
                                                             addc_func);

using AddOp = std::function<std::pair<uint32_t, bool>(uint32_t, uint32_t,
                                                      uint32_t, int, Wave &)>;
static auto add_co_func = [](uint32_t s0, uint32_t s1, uint32_t, int,
                             Wave &) -> std::pair<uint32_t, bool> {
  uint64_t t = (uint64_t)s0 + s1;
  return {(uint32_t)t, (t >> 32) != 0};
};
static RegisterVOP3b<uint32_t, uint32_t, AddOp> v_add_co("v_add_co_u32",
                                                         add_co_func);

// VOPC vector comparisons
static auto eq_u32 = [](uint32_t a, uint32_t b) { return a == b; };

static RegisterVComp<int> v_cmp_gt_i("v_cmp_gt_i32", std::greater<int>());
static RegisterVComp<uint32_t> v_cmp_gt_u("v_cmp_gt_u32",
                                          std::greater<uint32_t>());
static RegisterVComp<uint32_t> v_cmp_le_u("v_cmp_le_u32",
                                          std::less_equal<uint32_t>());
static RegisterVComp<uint32_t> v_cmp_lt_u("v_cmp_lt_u32",
                                          std::less<uint32_t>());
static RegisterVComp<int> v_cmp_lt_i("v_cmp_lt_i32", std::less<int>());
static RegisterVComp<int> v_cmp_ge_i("v_cmp_ge_i32", std::greater_equal<int>());
static RegisterVComp<uint32_t> v_cmp_ge_u("v_cmp_ge_u32",
                                          std::greater_equal<uint32_t>());
static RegisterVComp<int> v_cmp_le_i("v_cmp_le_i32", std::less_equal<int>());
static RegisterVComp<uint32_t> v_cmp_eq_u("v_cmp_eq_u32", eq_u32);
static RegisterVComp<int32_t> v_cmp_eq_i("v_cmp_eq_i32",
                                         std::equal_to<int32_t>());

// V_CMP_EQ_F32 66
// Set the per-lane condition code to 1 iff the first input is equal to the
// second input. Store the result into VCC or a scalar register. D0.u64[laneId]
// = S0.f32 == S1.f32;
// // D0 = VCC in VOPC encoding.
static RegisterVComp<float> v_cmp_eq_f("v_cmp_eq_f32", std::equal_to<float>());

// V_CMP_U_F32 72
// Set the per-lane condition code to 1 iff the first input is not orderable to
// the second input. Store the result into VCC or a scalar register.
// D0.u64[laneId] = (isNAN(64'F(S0.f32)) || isNAN(64'F(S1.f32)));
// // D0 = VCC in VOPC encoding.
static RegisterVComp<float> v_cmp_u_f("v_cmp_u_f32", [](float a, float b) {
  return std::isnan(a) || std::isnan(b);
});

// // v_cmp_u_f32 s[12:13], v16, v16

static RegisterVComp<uint32_t> v_cmp_ne_u("v_cmp_ne_u32",
                                          std::not_equal_to<uint32_t>());
static Register<VOPC_Compare<uint32_t, decltype(eq_u32)>>
    v_cmp_eq_sdwa("v_cmp_eq_u32_sdwa", eq_u32, InstructionMode::SDWA);

static RegisterVCompX<uint32_t> v_cmpx_eq("v_cmpx_eq_u32", eq_u32);
static RegisterVCompX<uint32_t> v_cmpx_gt("v_cmpx_gt_u32",
                                          std::greater<uint32_t>());
static RegisterVCompX<uint32_t> v_cmpx_ge("v_cmpx_ge_u32",
                                          std::greater_equal<uint32_t>());

// Signed comparisons (v_cmpx writes exec mask)
static RegisterVCompX<int32_t> v_cmpx_gt_i("v_cmpx_gt_i32",
                                           std::greater<int32_t>());
static RegisterVCompX<int32_t> v_cmpx_lt_i("v_cmpx_lt_i32",
                                           std::less<int32_t>());

// v_add_nc_u32: RDNA name for v_add_u32 (add without carry).
static RegisterVOP2_32<uint32_t> v_add_nc("v_add_nc_u32",
                                          [](uint32_t a, uint32_t b, Wave &) {
                                            return a + b;
                                          });
// v_add_nc_i32: signed variant, same bit operation as v_add_nc_u32.
static RegisterVOP2_32<uint32_t> v_add_nc_i("v_add_nc_i32",
                                            [](uint32_t a, uint32_t b, Wave &) {
                                              return a + b;
                                            });
// v_sub_nc_u32: RDNA name for v_sub_u32 (subtract without carry).
static RegisterVOP2_32<uint32_t> v_sub_nc("v_sub_nc_u32",
                                          [](uint32_t a, uint32_t b, Wave &) {
                                            return a - b;
                                          });

// v_alignbit_b32: D.u32 = ({S0, S1} >> S2[4:0])[31:0]
// Concatenates S0 (high) and S1 (low) into a 64-bit value, shifts right
// by S2[4:0] bits, and returns the low 32 bits.
static RegisterVOP3a<uint32_t>
    v_alignbit("v_alignbit_b32",
               [](uint32_t s0, uint32_t s1, uint32_t s2, Wave &) -> uint32_t {
                 uint64_t combined = (static_cast<uint64_t>(s0) << 32) | s1;
                 return static_cast<uint32_t>(combined >> (s2 & 0x1F));
               });

// RDNA VOPD dual-issue aliases. The VOPD format packs two operations
// into one instruction, separated by "::" in the assembly text. The
// emulator handles "::" splitting in Wave.cpp, so these are registered
// as their single-operation equivalents.
static Register<VOP1_VConvert<uint32_t, uint32_t>> v_dual_mov("v_dual_mov_b32");
static RegisterVOP2_32<float> v_dual_mul("v_dual_mul_f32",
                                         [](float a, float b, Wave &) {
                                           return a * b;
                                         });
static RegisterVOP2_32<uint32_t>
    v_dual_lshlrev("v_dual_lshlrev_b32", [](uint32_t a, uint32_t b, Wave &) {
      return b << (a & 0x1F);
    });
static RegisterVOP2_32<uint32_t> v_dual_and("v_dual_and_b32",
                                            [](uint32_t a, uint32_t b, Wave &) {
                                              return a & b;
                                            });

} // namespace
} // namespace raceemulator
