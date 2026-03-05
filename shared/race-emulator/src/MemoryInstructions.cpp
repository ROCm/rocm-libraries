#include "race-emulator/CommonRegister.h"
#include "race-emulator/Instruction.h"
#include "race-emulator/LDS.h"
#include "race-emulator/Util.h"
#include "race-emulator/Wave.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace raceemulator {

namespace {

template <typename T_Storage>
void executeLoadAndWrite(Wave &wave, int lane, uint64_t finalAddr,
                         CommonRegister dst, int numElements,
                         bool isCompleteEmulation) {
  auto ptr = reinterpret_cast<T_Storage *>(finalAddr);
  for (int i = 0; i < numElements; ++i) {
    T_Storage value = 0;
    if (isCompleteEmulation) {
      value = ptr[i];
    }
    wave.setVgpr(dst.index + i, lane, static_cast<uint32_t>(value));
  }
}

// T_Storage:
//   The primitive type to read (uint32_t for dwords, uint16_t for shorts)
//
// Examples:
//   global_load_dword v3, v[0:1]
//   global_load_dword v3, v[0:1], off offset:20
//   global_load_dwordx4 v[2:5], v[2:3], off
//   global_load_dword v2, v0, s[0:1] offset:68
template <typename T_Storage> class GlobalLoad : public Instruction {
  int numElements;

public:
  GlobalLoad(int n = 1) : numElements(n) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    auto dst = wave.getFirstRegister(partitioned[1]);
    auto src0 = wave.getFirstRegister(partitioned[2]);

    bool hasSaddr = false;
    CommonRegister src1 = {CommonRegister::Type::UNKNOWN, -1};
    int32_t instOffset = 0;

    constexpr const char *const offsetPrefix = "offset:";
    const auto lenOffsetPrefix = std::char_traits<char>::length(offsetPrefix);

    // Iterate over optional operands starting from index 3
    for (size_t i = 3; i < partitioned.size(); ++i) {
      std::string_view token = partitioned[i];
      if (token == "off") {
        continue; // offset is disabled
      } else if (token.starts_with(offsetPrefix)) {
        // Parse "offset:120" -> 120
        auto valStr = token.substr(lenOffsetPrefix);
        instOffset = getIntFromView<int32_t>(valStr);
      } else {
        auto potentialReg = wave.getFirstRegister(token);
        if (potentialReg.type == CommonRegister::Type::SGPR) {
          hasSaddr = true;
          src1 = potentialReg;
        } else {
          throw std::runtime_error(
              "Unexpected token in GlobalLoad modifiers: " +
              std::string(token));
        }
      }
    }

    std::vector<uint32_t> waveWritten;
    for (int i = 0; i < numElements; ++i) {
      waveWritten.push_back(dst.index + i);
    }

    int n = numElements;
    return [&wave, dst, src0, src1, hasSaddr, instOffset, n, waveWritten]() {
      auto run = [&](int lane) {
        uint64_t finalAddr = 0;

        if (hasSaddr) {
          // Mode: base (SGPR) + offset (VGPR)
          // Example: global_load_dword v2, v0, s[0:1]
          uint64_t base = wave.getSgpr64(src1.index);
          uint32_t offset = wave.getVgpr(src0.index, lane);
          finalAddr = base + offset;
        } else {
          // Mode: pointer (VGPR 64-bit)
          // Example: global_load_dword v3, v[0:1]
          finalAddr = wave.getVgpr64(src0.index, lane);
        }
        finalAddr += instOffset;
        executeLoadAndWrite<T_Storage>(wave, lane, finalAddr, dst, n,
                                       wave.isCompleteEmulation());
      };

      wave.runExecConditionedForLanes(run);
      auto pc = wave.getPc();
      wave.registerGlobalToVgprEvent(pc, waveWritten);
      return pc + 1;
    };
  }
};

template <typename T_Storage, int Shift = 0>
class GlobalStore : public Instruction {
  int numElements;

public:
  GlobalStore(int n = 1) : numElements(n) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    auto addrSrc = wave.getFirstRegister(partitioned[1]);
    auto dataSrc = wave.getFirstRegister(partitioned[2]);

    bool hasSaddr = false;
    CommonRegister baseSrc = {CommonRegister::Type::UNKNOWN, -1};
    int32_t instOffset = 0;

    // Iterate over optional operands starting from index 3
    for (size_t i = 3; i < partitioned.size(); ++i) {
      std::string_view token = partitioned[i];

      if (token == "off") {
        continue; // explicit offset disabled
      } else if (token.starts_with("offset:")) {
        // Parse "offset:8" -> 8
        auto valStr = token.substr(7);
        instOffset = getIntFromView<int32_t>(valStr);
      } else {
        // Assume it's an SADDR register if it's not a keyword
        auto reg = wave.getFirstRegister(token);
        if (reg.type == CommonRegister::Type::SGPR) {
          hasSaddr = true;
          baseSrc = reg;
        } else {
          throw std::runtime_error(
              "Unexpected token in GlobalStore modifiers: " +
              std::string(token));
        }
      }
    }

    std::vector<uint32_t> waveRead;
    for (int i = 0; i < numElements; ++i) {
      waveRead.push_back(dataSrc.index + i);
    }

    int n = numElements;
    return [&wave, addrSrc, dataSrc, baseSrc, hasSaddr, instOffset, n,
            waveRead]() {
      auto run = [&](int lane) {
        uint64_t finalAddr = 0;
        if (hasSaddr) {
          uint64_t base = wave.getSgpr64(baseSrc.index);
          uint32_t offset = wave.getVgpr(addrSrc.index, lane);
          finalAddr = base + offset;
        } else {
          finalAddr = wave.getVgpr64(addrSrc.index, lane);
        }

        finalAddr += instOffset;
        auto ptr = reinterpret_cast<T_Storage *>(finalAddr);
        for (int i = 0; i < n; ++i) {
          uint32_t regVal = wave.getVgpr(dataSrc.index + i, lane);
          uint32_t shiftedVal = regVal >> Shift;

          if (wave.isCompleteEmulation()) {
            ptr[i] = static_cast<T_Storage>(shiftedVal);
          }
        }
      };

      wave.runExecConditionedForLanes(run);
      auto pc = wave.getPc();
      wave.registerVgprToGlobalEvent(pc, waveRead);
      return pc + 1;
    };
  }
};

// 1. Static configuration
struct BufferConfig {
  CommonRegister srsrc;      // T# Descriptor
  CommonRegister vIndexReg;  // VGPR for Index (if idxen)
  CommonRegister vOffsetReg; // VGPR for Offset (if offen)

  int sOffsetReg = -1;    // SGPR for offset (-1 if immediate)
  int32_t sOffsetImm = 0; // Immediate for SOffset
  int32_t instOffset = 0; // Immediate "offset:" modifier

  bool useIndex = false;  // idxen
  bool useOffset = false; // offen

  // Factory: Parses the raw string tokens into a clean config
  static BufferConfig parse(const std::vector<std::string_view> &parts,
                            Wave &wave) {
    BufferConfig cfg;

    // Parse Modifiers first to determine register layout
    for (const auto &token : parts) {
      if (token == "offen")
        cfg.useOffset = true;
      if (token == "idxen")
        cfg.useIndex = true;
      if (token.starts_with("offset:"))
        cfg.instOffset = std::stoi(std::string(token.substr(7)), nullptr, 0);
    }

    // Map Operands
    // parts[0]=Op, [1]=Data, [2]=VAddr, [3]=SRSRC, [4]=SOffset
    auto vAddrBase = wave.getFirstRegister(parts[2]);
    cfg.srsrc = wave.getFirstRegister(parts[3]);

    // Handle VADDR Logic: [Index?, Offset?]
    if (cfg.useIndex) {
      cfg.vIndexReg = vAddrBase;
      if (cfg.useOffset) {
        cfg.vOffsetReg = CommonRegister::getVgpr(vAddrBase.index + 1);
      }
    } else if (cfg.useOffset) {
      cfg.vOffsetReg = vAddrBase;
    }

    // Handle SOFFSET (SGPR or Imm)
    std::string_view sOff = parts[4];
    if (sOff.find('s') == 0) {
      cfg.sOffsetReg = wave.getFirstRegister(sOff).index;
    } else {
      cfg.sOffsetImm = std::stoi(std::string(sOff), nullptr, 0);
    }

    return cfg;
  }
};

struct BufferState {
  uint64_t baseAddress;
  uint32_t size;
  int64_t baseOffset;
  uint32_t index;
  bool isStructured;

  static BufferState compute(const Wave &wave, int lane,
                             const BufferConfig &cfg) {
    uint32_t w0 = wave.getSgpr(cfg.srsrc.index);
    uint32_t w1 = wave.getSgpr(cfg.srsrc.index + 1);
    uint32_t w2 = wave.getSgpr(cfg.srsrc.index + 2);

    uint64_t descBase = w0 | (static_cast<uint64_t>(w1 & 0xFFFF) << 32);
    uint32_t stride = (w1 >> 16) & 0x3FFF;
    uint32_t descSize = w2;
    uint32_t rawVIdx =
        cfg.useIndex ? wave.getVgpr(cfg.vIndexReg.index, lane) : 0;
    int64_t vOff =
        cfg.useOffset
            ? static_cast<int32_t>(wave.getVgpr(cfg.vOffsetReg.index, lane))
            : 0;

    int64_t sOff = 0;
    if (cfg.sOffsetReg >= 0) {
      sOff = static_cast<int32_t>(wave.getSgpr(cfg.sOffsetReg));
    } else {
      sOff = cfg.sOffsetImm;
    }

    BufferState state;
    state.isStructured = cfg.useIndex;
    state.index = rawVIdx;
    state.size = descSize;
    state.baseOffset = sOff + vOff + cfg.instOffset;

    state.baseAddress =
        descBase + (cfg.useIndex ? (uint64_t(rawVIdx) * stride) : 0);

    return state;
  }

  // Inside MemoryInstructions.h -> BufferState

  bool isInBounds(int64_t elementOffset, int elementSize) const {
    if (isStructured) {
      return index < size;
    } else {
      // Raw Buffer Bounds Check
      int64_t totalOffset = baseOffset + elementOffset;

      // 1. Hardware Check: Offsets cannot be negative relative to Base
      if (totalOffset < 0) {
        return false;
      }

      // 2. Hardware Check: The end of the read must fit within Size
      // We can safely cast to uint64_t now because we checked < 0 above.
      return static_cast<uint64_t>(totalOffset + elementSize) <= size;
    }
  }
};

template <typename T> class BufferLoad : public Instruction {
  int numElements;

public:
  BufferLoad(int n = 1) : numElements(n) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto parts = getPartitioned(line);
    auto config = BufferConfig::parse(parts, wave);
    auto dstReg = wave.getFirstRegister(parts[1]);
    int n = numElements;

    std::vector<uint32_t> waveWritten;
    for (int i = 0; i < numElements; ++i) {
      waveWritten.push_back(dstReg.index + i);
    }

    return [&wave, config, dstReg, n, waveWritten]() {
      auto run = [&](int lane) {
        // 1. Resolve State
        auto state = BufferState::compute(wave, lane, config);

        // 2. Element Loop
        for (int i = 0; i < n; ++i) {
          int64_t elemOffset = i * sizeof(T);
          if (state.isInBounds(elemOffset, sizeof(T))) {
            uint64_t addr = state.baseAddress + state.baseOffset + elemOffset;
            T val{0};
            if (wave.isCompleteEmulation()) {
              std::memcpy(&val, reinterpret_cast<const void *>(addr),
                          sizeof(T));
            }
            wave.setVgpr(dstReg.index + i, lane, val);
          } else {
            wave.setVgpr(dstReg.index + i, lane, 0);
          }
        }
      };
      wave.runExecConditionedForLanes(run);
      wave.registerGlobalToVgprEvent(wave.getPc(), waveWritten);
      return wave.getPc() + 1;
    };
  }
};

template <typename T> class BufferStore : public Instruction {
  int numElements;

public:
  BufferStore(int n = 1) : numElements(n) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto parts = getPartitioned(line);
    auto config = BufferConfig::parse(parts, wave);
    auto srcReg = wave.getFirstRegister(parts[1]);
    int n = numElements;

    std::vector<uint32_t> waveRead;
    for (int i = 0; i < numElements; ++i) {
      waveRead.push_back(srcReg.index + i);
    }

    return [&wave, config, srcReg, n, waveRead]() {
      auto run = [&](int lane) {
        auto state = BufferState::compute(wave, lane, config);

        for (int i = 0; i < n; ++i) {
          int64_t elemOffset = i * sizeof(T);

          if (state.isInBounds(elemOffset, sizeof(T))) {
            uint64_t addr = state.baseAddress + state.baseOffset + elemOffset;

            // Cast safely to T (handling 32-bit registers)
            uint32_t raw = wave.getVgpr(srcReg.index + i, lane);

            T val = static_cast<T>(raw);
            if (wave.isCompleteEmulation()) {
              std::memcpy(reinterpret_cast<void *>(addr), &val, sizeof(T));
            }
          }
          // OOB -> Drop silently
        }
      };
      wave.runExecConditionedForLanes(run);
      wave.registerVgprToGlobalEvent(wave.getPc(), waveRead);
      return wave.getPc() + 1;
    };
  }
};

class SLoadDword : public Instruction {
  int numDwords;

public:
  SLoadDword(int n) : numDwords(n) {}
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {

    auto partitioned = getPartitioned(line);
    auto dst = wave.getFirstRegister(partitioned[1]);
    auto src = wave.getFirstRegister(partitioned[2]);
    auto offset = wave.parseOperand<uint64_t>(partitioned[3]);
    int n = numDwords;

    return [&wave, dst, src, offset, n]() {
      uint64_t base = wave.getSgpr64(src.index);
      uint32_t *ptr;
      auto offsetVal = wave.getSgprOrLiteralValue(offset);
      ptr = reinterpret_cast<uint32_t *>(base + offsetVal);
      for (int i = 0; i < n; ++i) {
        wave.setSgpr(dst.index + i, ptr[i]);
      }
      return wave.getPc() + 1;
    };
  }
};

template <typename T_Storage> class DsWrite : public Instruction {
  int numElements;  // Number of T_Storage elements to write
  bool useHighBits; // Flag to extract high 16 bits (for d16_hi variants)

public:
  // n = number of elements (e.g., 1 for b32, 4 for b128)
  // high = true to read bits [31:16] of the source VGPR
  DsWrite(int n, bool high) : numElements(n), useHighBits(high) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {

    auto tokens = getPartitioned(line);

    // 1. Strict Format Check
    if (tokens.size() < 3) {
      throw std::runtime_error(
          "Invalid DS_WRITE format: " + std::string(tokens[0]) +
          " requires at least 2 operands");
    }

    auto addrReg = wave.getFirstRegister(tokens[1]);
    if (addrReg.type != CommonRegister::Type::VGPR) {
      throw std::runtime_error(
          "Invalid LDS Address Operand: must be a VGPR, found: " +
          std::string(tokens[1]));
    }

    auto dataReg = wave.getFirstRegister(tokens[2]);

    // 4. Parse Modifiers (offset, gds)
    int32_t instOffset = 0;

    for (size_t i = 3; i < tokens.size(); ++i) {
      std::string_view token = tokens[i];

      if (token.starts_with("offset:")) {
        auto valStr = token.substr(7);
        instOffset = getIntFromView<int32_t>(valStr);
      } else if (token == "gds") {
        throw std::runtime_error("Unsupported GDS Modifier: GDS (Global Data "
                                 "Share) emulation not supported");
      } else {
        throw std::runtime_error("Unknown token in DS_WRITE instruction: " +
                                 std::string(token));
      }
    }

    std::vector<uint32_t> waveRead;
    for (int i = 0; i < numElements; ++i) {
      waveRead.push_back(dataReg.index + i);
    }

    int n = numElements;
    int waveId = wave.getWaveId();
    bool high = useHighBits; // Capture the flag for the lambda

    return [&wave, addrReg, dataReg, instOffset, n, waveRead, waveId, high]() {
      LaneAndLDSBytes bytesWritten;
      bytesWritten.reserve(wave.getWaveSize() * n * sizeof(T_Storage));
      LDS &lds = wave.getLds();

      auto run = [&](int lane) {
        uint32_t vOffset = wave.getVgpr(addrReg.index, lane);
        uint32_t effectiveAddr = vOffset + static_cast<uint32_t>(instOffset);

        for (int i = 0; i < n; ++i) {

          uint32_t val = wave.getVgpr(dataReg.index + i, lane);

          if (high) {
            val >>= 16;
          }

          T_Storage valToStore = static_cast<T_Storage>(val);
          int64_t addr = effectiveAddr + i * sizeof(T_Storage);

          lds.write<T_Storage>(addr, waveId, lane, valToStore);

          for (uint32_t b = 0; b < sizeof(T_Storage); ++b) {
            bytesWritten.push_back({lane, static_cast<int>(addr + b)});
          }
        }
      };

      auto pc = wave.getPc();
      wave.runExecConditionedForLanes(run);
      wave.registerVgprToLdsEvent(pc, waveRead, bytesWritten);
      return pc + 1;
    };
  }
};

template <typename T_Mem, int N_Regs> class DsRead : public Instruction {
  bool isD16;  // If true, uses 16-bit packing (preserves other half of VGPR)
  bool isHigh; // If true, writes to bits [31:16]; otherwise [15:0]

public:
  // Default constructor: Standard reads (b32, b128, u8, i8).
  // Defaults to isD16=false (overwrite full register).
  DsRead() : isD16(false), isHigh(false) {}

  // Constructor for D16 variants
  // d16: enable packing behavior
  // high: target high bits [31:16]
  DsRead(bool d16, bool high) : isD16(d16), isHigh(high) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {

    auto tokens = getPartitioned(line);

    if (tokens.size() < 3) {
      throw std::runtime_error(
          "Invalid DS_READ format: " + std::string(tokens[0]) +
          " requires at least 2 operands");
    }

    auto dstReg = wave.getFirstRegister(tokens[1]);
    auto addrReg = wave.getFirstRegister(tokens[2]);
    if (addrReg.type != CommonRegister::Type::VGPR) {
      throw std::runtime_error("DS_READ Address must be a VGPR");
    }

    int32_t instOffset = 0;
    for (size_t i = 3; i < tokens.size(); ++i) {
      if (tokens[i].starts_with("offset:")) {
        instOffset = getIntFromView<int32_t>(tokens[i].substr(7));
      } else if (tokens[i] == "gds") {
        throw std::runtime_error("GDS not supported");
      }
    }

    std::vector<uint32_t> waveWritten;
    for (int i = 0; i < N_Regs; ++i) {
      waveWritten.push_back(dstReg.index + i);
    }

    // Capture flags for lambda
    bool d16 = isD16;
    bool high = isHigh;

    return [&wave, dstReg, addrReg, instOffset, waveWritten, d16, high]() {
      const LDS &lds = wave.getLds();
      LaneAndLDSBytes bytesRead;
      bytesRead.reserve(wave.getWaveSize() * N_Regs * sizeof(T_Mem));

      auto run = [&](int lane) {
        uint32_t vOffset = wave.getVgpr(addrReg.index, lane);
        uint32_t baseAddr = vOffset + static_cast<uint32_t>(instOffset);

        for (int i = 0; i < N_Regs; ++i) {
          uint32_t finalAddr = baseAddr + (i * sizeof(T_Mem));

          for (uint32_t b = 0; b < sizeof(T_Mem); ++b) {
            bytesRead.push_back({lane, static_cast<int>(finalAddr + b)});
          }

          int waveId = wave.getWaveId();
          T_Mem rawValue =
              lds.read<T_Mem>(static_cast<int>(finalAddr), waveId, lane);

          // --- Logic Branch: Standard vs D16 ---
          if (!d16) {
            // Standard Behavior: Zero/Sign extend to 32-bits and OVERWRITE
            // register (Used by ds_read_u8, ds_read_i8, ds_read_b32, etc.)
            uint32_t extended;
            if constexpr (std::is_signed_v<T_Mem>) {
              extended = static_cast<uint32_t>(static_cast<int32_t>(rawValue));
            } else {
              extended = static_cast<uint32_t>(rawValue);
            }
            wave.setVgpr(dstReg.index + i, lane, extended);
          } else {
            // D16 Behavior: Read-Modify-Write (Preserve other 16 bits)
            uint32_t currentDest = wave.getVgpr(dstReg.index + i, lane);
            uint16_t valToPack = 0;

            // 1. Handle Extension (8-bit to 16-bit)
            if constexpr (sizeof(T_Mem) == 1) {
              // Cast to target 16-bit type (handling sign extension if T_Mem is
              // signed)
              if constexpr (std::is_signed_v<T_Mem>) {
                valToPack =
                    static_cast<uint16_t>(static_cast<int16_t>(rawValue));
              } else {
                valToPack = static_cast<uint16_t>(rawValue);
              }
            } else {
              valToPack = static_cast<uint16_t>(rawValue);
            }

            // 2. Pack into 32-bit container
            uint32_t result = 0;
            if (high) {
              // Preserve Low [15:0], Write High [31:16]
              uint32_t preserved = currentDest & 0x0000FFFF;
              uint32_t inserted = static_cast<uint32_t>(valToPack) << 16;
              if (wave.getDsPreserve()) {
                result = preserved | inserted;
              } else {
                result = inserted;
              } //
            } else {
              // Preserve High [31:16], Write Low [15:0]
              uint32_t preserved = currentDest & 0xFFFF0000;
              uint32_t inserted = static_cast<uint32_t>(valToPack);
              if (wave.getDsPreserve()) {
                result = preserved | inserted;
              } else {
                result = inserted; // preserved | inserted;
              }
            }

            wave.setVgpr(dstReg.index + i, lane, result);
          }
        }
      };

      wave.runExecConditionedForLanes(run);
      auto pc = wave.getPc();
      wave.registerLdsToVgprEvent(pc, waveWritten, bytesRead);
      return pc + 1;
    };
  }
};

static Register<GlobalLoad<uint32_t>> gl_ld_1("global_load_dword", 1);
static Register<BufferLoad<uint32_t>> bf_ld_1("buffer_load_dword", 1);
static Register<BufferStore<uint32_t>> bf_st_1("buffer_store_dword", 1);
static Register<GlobalStore<uint32_t>> gl_st_1("global_store_dword", 1);

static Register<GlobalLoad<uint32_t>> gl_ld_2("global_load_dwordx2", 2);
static Register<BufferLoad<uint32_t>> bf_ld_2("buffer_load_dwordx2", 2);
static Register<BufferStore<uint32_t>> bf_st_2("buffer_store_dwordx2", 2);
static Register<GlobalStore<uint32_t>> gl_st_2("global_store_dwordx2", 2);

static Register<GlobalLoad<uint32_t>> gl_ld_3("global_load_dwordx3", 3);
static Register<BufferLoad<uint32_t>> bf_ld_3("buffer_load_dwordx3", 3);
static Register<BufferStore<uint32_t>> bf_st_3("buffer_store_dwordx3", 3);
static Register<GlobalStore<uint32_t>> gl_st_3("global_store_dwordx3", 3);

static Register<GlobalLoad<uint32_t>> gl_ld_4("global_load_dwordx4", 4);
static Register<BufferLoad<uint32_t>> bf_ld_4("buffer_load_dwordx4", 4);
static Register<BufferStore<uint32_t>> bf_st_4("buffer_store_dwordx4", 4);
static Register<GlobalStore<uint32_t>> gl_st_4("global_store_dwordx4", 4);

static Register<SLoadDword> s_ld_1("s_load_dword", 1);
static Register<SLoadDword> s_ld_2("s_load_dwordx2", 2);
static Register<SLoadDword> s_ld_3("s_load_dwordx3", 3);
static Register<SLoadDword> s_ld_4("s_load_dwordx4", 4);
static Register<SLoadDword> s_ld_8("s_load_dwordx8", 8);
static Register<SLoadDword> s_ld_16("s_load_dwordx16", 16);

// Sub-dword Memory

static Register<GlobalLoad<uint16_t>> gl_ld_u16("global_load_ushort", 1);
static Register<GlobalLoad<uint8_t>> gl_ld_u8("global_load_ubyte", 1);
static Register<GlobalStore<uint16_t>> gl_st_u16("global_store_short");
static Register<GlobalStore<uint8_t>> gl_st_u8("global_store_byte");
static Register<GlobalStore<uint16_t, 16>>
    gl_st_u16hi("global_store_short_d16_hi", 1);

// DS Memory
static Register<DsRead<uint8_t, 1>> ds_rd_u8("ds_read_u8");
static Register<DsRead<uint16_t, 1>> ds_rd_u16("ds_read_u16");
static Register<DsRead<int8_t, 1>> ds_rd_i8("ds_read_i8");
static Register<DsRead<int16_t, 1>> ds_rd_i16("ds_read_i16");
static Register<DsRead<uint32_t, 1>> ds_rd_32("ds_read_b32");
static Register<DsRead<uint32_t, 2>> ds_rd_64("ds_read_b64");
static Register<DsRead<uint32_t, 4>> ds_rd_128("ds_read_b128");

static Register<DsRead<uint8_t, 1>> ds_rd_u8_d16("ds_read_u8_d16", true, false);
static Register<DsRead<uint8_t, 1>> ds_rd_u8_d16_hi("ds_read_u8_d16_hi", true,
                                                    true);
static Register<DsRead<int8_t, 1>> ds_rd_i8_d16("ds_read_i8_d16", true, false);
static Register<DsRead<int8_t, 1>> ds_rd_i8_d16_hi("ds_read_i8_d16_hi", true,
                                                   true);
static Register<DsRead<uint16_t, 1>> ds_rd_u16_d16("ds_read_u16_d16", true,
                                                   false);
static Register<DsRead<uint16_t, 1>> ds_rd_u16_d16_hi("ds_read_u16_d16_hi",
                                                      true, true);

static Register<DsWrite<uint8_t>> ds_wr_8("ds_write_b8", 1, false);
static Register<DsWrite<uint16_t>> ds_wr_16("ds_write_b16", 1, false);
static Register<DsWrite<uint32_t>> ds_wr_32("ds_write_b32", 1, false);
static Register<DsWrite<uint32_t>> ds_wr_64("ds_write_b64", 2, false);
static Register<DsWrite<uint32_t>> ds_wr_96("ds_write_b96", 3, false);
static Register<DsWrite<uint32_t>> ds_wr_128("ds_write_b128", 4, false);

static Register<DsWrite<uint16_t>> ds_wr_16_hi("ds_write_b16_d16_hi", 1, true);
static Register<DsWrite<uint8_t>> ds_wr_8_hi("ds_write_b8_d16_hi", 1, true);
} // namespace
} // namespace raceemulator
