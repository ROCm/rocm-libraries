// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/CommonRegister.h"
#include "race-emulator/Instruction.h"
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
          T_Storage valToStore;
          if constexpr (sizeof(T_Storage) <= 2) {
            valToStore = static_cast<T_Storage>(
                wave.getHalfVgpr(dataSrc.index + i, lane, Shift >= 16));
          } else {
            uint32_t regVal = wave.getVgpr(dataSrc.index + i, lane);
            valToStore = static_cast<T_Storage>(regVal >> Shift);
          }

          if (wave.isCompleteEmulation()) {
            ptr[i] = valToStore;
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
  bool useLds = false;    // lds (direct-to-LDS)

  // Factory: Parses the raw string tokens into a clean config
  static BufferConfig parse(const std::vector<std::string_view> &parts,
                            Wave &wave) {
    BufferConfig cfg;

    // Parse Modifiers first to determine register layout
    for (const auto &token : parts) {
      if (token == "offen") {
        cfg.useOffset = true;
      }
      if (token == "idxen") {
        cfg.useIndex = true;
      }
      if (token == "lds") {
        cfg.useLds = true;
      }
      if (token.starts_with("offset:")) {
        cfg.instOffset = std::stoi(std::string(token.substr(7)), nullptr, 0);
      }
    }

    // Map Operands.
    // Standard: parts[1]=VDATA, [2]=VADDR, [3]=SRSRC, [4]=SOFFSET
    // DTL:      parts[1]=VDATA,            [2]=SRSRC, [3]=SOFFSET
    //   No separate VADDR -- VDATA doubles as the offen offset.
    // The LDS case has one fewer operand, so SRSRC/SOFFSET shift down by 1.
    int srsrcIdx = cfg.useLds ? 2 : 3;
    cfg.srsrc = wave.getFirstRegister(parts[srsrcIdx]);

    // VADDR / offen / idxen handling.
    auto vAddrBase = wave.getFirstRegister(parts[cfg.useLds ? 1 : 2]);
    if (cfg.useLds) {
      // DTL: VDATA provides the offen offset for the global address.
      if (cfg.useOffset) {
        cfg.vOffsetReg = vAddrBase;
      }
    } else {
      if (cfg.useIndex) {
        cfg.vIndexReg = vAddrBase;
        if (cfg.useOffset) {
          cfg.vOffsetReg = CommonRegister::getVgpr(vAddrBase.index + 1);
        }
      } else if (cfg.useOffset) {
        cfg.vOffsetReg = vAddrBase;
      }
    }

    // SOFFSET (SGPR or immediate).
    std::string_view sOff = parts[srsrcIdx + 1];
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
    int n = numElements;

    // ================================================================
    // Direct-to-LDS (DTL): buffer_load ... lds
    // ================================================================
    //
    // Overview
    // --------
    // When the `lds` modifier is present on a MUBUF buffer_load instruction,
    // the hardware bypasses VGPRs entirely and writes the loaded data straight
    // into LDS (Local Data Share / shared memory). This is used by high-
    // performance GEMM kernels (e.g. TensileLite on gfx950) to overlap global
    // memory fetches with compute -- the data arrives in LDS ready for the
    // math units without an intermediate ds_write step.
    //
    // ISA reference: CDNA4 ISA §9.1.9 "Memory Buffer Load to LDS".
    // Supported instructions (CDNA4): BUFFER_LOAD_{ubyte, sbyte, ushort,
    // sshort, dword, dwordX3, dwordX4, format_x}.
    // Note: CDNA3 (gfx942) only supports up to dword (no dwordX3/X4).
    //
    // Memory (source) address
    // -----------------------
    // Computed exactly like a normal buffer_load via the SRD (Shader Resource
    // Descriptor) + optional VGPR offset (offen/idxen) + SGPR/imm SOffset:
    //
    //   global_addr = SRD.base
    //                 + (idxen ? vgpr_index * SRD.stride : 0)
    //                 + (offen ? vgpr_offset : 0)
    //                 + sgpr_soffset
    //                 + inst_offset
    //
    // LDS (destination) address
    // -------------------------
    // The LDS write address is determined by M0 and the lane index:
    //
    //   lds_addr = M0[17:0] + TIDinWave * bytes_per_lane
    //
    // where bytes_per_lane is the number of bytes loaded per lane (e.g. 16
    // for dwordx4). For dwordx3 the hardware still uses bytes_per_lane=16,
    // writing 3 dwords and skipping the 4th.
    //
    // Operand encoding difference
    // ---------------------------
    // In a normal buffer_load, the VDATA field names the destination VGPR(s).
    // With LDS=1, VDATA is reinterpreted: the hardware reads M0 implicitly
    // for the LDS base address. In assembly syntax this changes the operand
    // layout:
    //
    //   Standard: buffer_load_dwordx4 v[dst], v[addr], s[srd], soff offen
    //             parts: [0]=op [1]=VDATA [2]=VADDR [3]=SRSRC [4]=SOFFSET
    //
    //   DTL:      buffer_load_dwordx4 v[addr], s[srd], soff offen lds
    //             parts: [0]=op [1]=VDATA(=VADDR) [2]=SRSRC [3]=SOFFSET
    //
    // The VDATA VGPR provides the offen/idxen offset for the global memory
    // address, NOT the LDS address. BufferConfig::parse handles this
    // 3-operand layout when useLds is true.
    //
    // ================================================================

    // Shared iteration: for each active lane, compute BufferState, check
    // bounds per element, and call onLoad/onOob. This is the common logic
    // between standard buffer loads (to VGPR) and DTL (to LDS).
    //
    // OOB behavior (ISA reference: CDNA3 §9.1, CDNA4 §9.1):
    // "When an address is out of range, reads return zero, and writes and
    // atomics are dropped." For standard loads the caller zeros the VGPR;
    // for DTL the caller skips the LDS write.
    auto forEachElement = [&wave, config, n](auto onLoad, auto onOob) {
      wave.runExecConditionedForLanes([&](int lane) {
        auto state = BufferState::compute(wave, lane, config);
        for (int i = 0; i < n; ++i) {
          int64_t elemOffset = i * sizeof(T);
          if (state.isInBounds(elemOffset, sizeof(T))) {
            uint64_t addr = state.baseAddress + state.baseOffset + elemOffset;
            T val{0};
            if (wave.isCompleteEmulation()) {
              std::memcpy(&val, reinterpret_cast<const void *>(addr),
                          sizeof(T));
            }
            onLoad(lane, i, val);
          } else {
            onOob(lane, i);
          }
        }
      });
    };

    if (config.useLds) {
      int bytesPerLane = n * static_cast<int>(sizeof(T));
      return [&wave, forEachElement, bytesPerLane]() {
        IntervalSet intervals;
        uint32_t m0 = wave.getM0();
        forEachElement(
            [&](int lane, int i, T val) {
              int ldsAddr =
                  static_cast<int>(m0 + lane * bytesPerLane + i * sizeof(T));
              wave.getWorkgroup().getLds().write<T>(ldsAddr, val);
              intervals.append(ldsAddr, ldsAddr + static_cast<int>(sizeof(T)));
            },
            [](int, int) {});
        intervals.finalize();
        wave.registerGlobalToLdsEvent(wave.getPc(), intervals);
        return wave.getPc() + 1;
      };
    }

    // Standard buffer load: data goes to VGPRs.
    auto dstReg = wave.getFirstRegister(parts[1]);

    std::vector<uint32_t> waveWritten;
    for (int i = 0; i < numElements; ++i) {
      waveWritten.push_back(dstReg.index + i);
    }

    return [&wave, forEachElement, dstReg, waveWritten]() {
      forEachElement(
          [&](int lane, int i, T val) {
            wave.setVgpr(dstReg.index + i, lane, val);
          },
          [&](int lane, int i) { wave.setVgpr(dstReg.index + i, lane, 0); });
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

            T val;
            if constexpr (sizeof(T) <= 2) {
              val = static_cast<T>(
                  wave.getHalfVgpr(srcReg.index + i, lane, false));
            } else {
              val = static_cast<T>(wave.getVgpr(srcReg.index + i, lane));
            }
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
    bool high = useHighBits; // Capture the flag for the lambda

    return [&wave, addrReg, dataReg, instOffset, n, waveRead, high]() {
      (void)high;
      IntervalSet intervals;
      intervals.reserve(wave.getWaveSize() * n);
      auto run = [&](int lane) {
        uint32_t vOffset = wave.getVgpr(addrReg.index, lane);
        uint32_t effectiveAddr = vOffset + static_cast<uint32_t>(instOffset);

        for (int i = 0; i < n; ++i) {

          T_Storage valToStore;
          if constexpr (sizeof(T_Storage) <= 2) {
            valToStore = static_cast<T_Storage>(
                wave.getHalfVgpr(dataReg.index + i, lane, high));
          } else {
            valToStore =
                static_cast<T_Storage>(wave.getVgpr(dataReg.index + i, lane));
          }
          int64_t addr = effectiveAddr + i * sizeof(T_Storage);

          wave.writeLds<T_Storage>(addr, lane, valToStore);

          intervals.append(static_cast<int>(addr),
                           static_cast<int>(addr + sizeof(T_Storage)));
        }
      };

      auto pc = wave.getPc();
      wave.runExecConditionedForLanes(run);
      intervals.finalize();
      wave.registerVgprToLdsEvent(pc, waveRead, intervals);
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
      auto sw = wave.profileScope("DsRead_intervals");
      std::vector<uint32_t> addrBuffer(wave.getWaveSize());
      wave.getVgprs(addrReg.index, addrBuffer.data());
      IntervalSet intervals;
      intervals.reserve(wave.getWaveSize() * N_Regs);
      wave.runExecConditionedForLanes([&](int lane) {
        uint32_t baseAddr =
            addrBuffer[lane] + static_cast<uint32_t>(instOffset);
        intervals.append(static_cast<int>(baseAddr),
                         static_cast<int>(baseAddr + sizeof(T_Mem) * N_Regs));
      });
      intervals.finalize();

      sw = wave.profileScope("DsRead_readLds");
      wave.runExecConditionedForLanes([&](int lane) {
        uint32_t baseAddr =
            addrBuffer[lane] + static_cast<uint32_t>(instOffset);
        T_Mem ldsValues[N_Regs];
        wave.readLds<T_Mem>(static_cast<int>(baseAddr), lane, ldsValues,
                            N_Regs);
        for (int i = 0; i < N_Regs; ++i) {
          T_Mem rawValue = ldsValues[i];
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
            // D16 Behavior: write only 16 bits of the destination register.
            uint16_t valToPack = 0;

            // 1. Handle Extension (8-bit to 16-bit)
            if constexpr (sizeof(T_Mem) == 1) {
              if constexpr (std::is_signed_v<T_Mem>) {
                valToPack =
                    static_cast<uint16_t>(static_cast<int16_t>(rawValue));
              } else {
                valToPack = static_cast<uint16_t>(rawValue);
              }
            } else {
              valToPack = static_cast<uint16_t>(rawValue);
            }

            // On RDNA (wave-32), d16 operations preserve the other half.
            // On CDNA (wave-64), they do not (hardware-verified).
            if (wave.getDsPreserve()) {
              wave.setVgprHalf(dstReg.index + i, lane, high, valToPack);
            } else {
              // CDNA: zero the other half.
              uint32_t result =
                  high ? (uint32_t(valToPack) << 16) : uint32_t(valToPack);
              wave.setVgpr(dstReg.index + i, lane, result);
            }
          }
        }
      });

      auto pc = wave.getPc();
      uint8_t byteMask = d16 ? (high ? 0xC : 0x3) : 0xF;
      wave.registerLdsToVgprEvent(pc, waveWritten, intervals, byteMask);
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

// buffer_load_d16_b16:     load short into VGPR[15:0], preserving VGPR[31:16].
// buffer_load_d16_hi_b16:  load short into VGPR[31:16], preserving VGPR[15:0].
class BufferLoadD16 : public Instruction {
  bool high; // true -> write bits [31:16], false -> write bits [15:0]

public:
  BufferLoadD16(bool high) : high(high) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto parts = getPartitioned(line);
    auto config = BufferConfig::parse(parts, wave);
    auto dstReg = wave.getFirstRegister(parts[1]);
    bool hi = high;

    std::vector<uint32_t> waveWritten = {static_cast<uint32_t>(dstReg.index)};

    return [&wave, config, dstReg, hi, waveWritten]() {
      auto run = [&](int lane) {
        auto state = BufferState::compute(wave, lane, config);
        if (state.isInBounds(0, sizeof(uint16_t))) {
          uint64_t addr = state.baseAddress + state.baseOffset;
          uint16_t val16{0};
          if (wave.isCompleteEmulation()) {
            std::memcpy(&val16, reinterpret_cast<const void *>(addr),
                        sizeof(uint16_t));
          }
          wave.setVgprHalf(dstReg.index, lane, hi, val16);
        }
      };
      wave.runExecConditionedForLanes(run);
      uint8_t byteMask = hi ? 0xC : 0x3;
      wave.registerGlobalToVgprEvent(wave.getPc(), waveWritten, byteMask);
      return wave.getPc() + 1;
    };
  }
};

// RDNA-style aliases (buffer_load_b<bits> = buffer_load_dword[xN])
static Register<BufferLoad<uint32_t>> bf_ld_b32("buffer_load_b32", 1);
static Register<BufferLoad<uint32_t>> bf_ld_b64("buffer_load_b64", 2);
static Register<BufferLoad<uint32_t>> bf_ld_b96("buffer_load_b96", 3);
static Register<BufferLoad<uint32_t>> bf_ld_b128("buffer_load_b128", 4);

static Register<BufferStore<uint16_t>> bf_st_b16("buffer_store_b16", 1);
static Register<BufferStore<uint16_t>> bf_st_short("buffer_store_short", 1);
static Register<BufferLoadD16> bf_ld_d16_lo("buffer_load_d16_b16", false);
static Register<BufferLoadD16> bf_ld_d16_hi("buffer_load_d16_hi_b16", true);
static Register<BufferStore<uint32_t>> bf_st_b32("buffer_store_b32", 1);
static Register<BufferStore<uint32_t>> bf_st_b64("buffer_store_b64", 2);
static Register<BufferStore<uint32_t>> bf_st_b96("buffer_store_b96", 3);
static Register<BufferStore<uint32_t>> bf_st_b128("buffer_store_b128", 4);

// RDNA-style aliases (global_load_b<bits> = global_load_dword[xN])
static Register<GlobalLoad<uint32_t>> gl_ld_b32("global_load_b32", 1);
static Register<GlobalLoad<uint32_t>> gl_ld_b64("global_load_b64", 2);
static Register<GlobalLoad<uint32_t>> gl_ld_b96("global_load_b96", 3);
static Register<GlobalLoad<uint32_t>> gl_ld_b128("global_load_b128", 4);

static Register<GlobalStore<uint32_t>> gl_st_b32("global_store_b32", 1);
static Register<GlobalStore<uint32_t>> gl_st_b64("global_store_b64", 2);
static Register<GlobalStore<uint32_t>> gl_st_b96("global_store_b96", 3);
static Register<GlobalStore<uint32_t>> gl_st_b128("global_store_b128", 4);

static Register<SLoadDword> s_ld_1("s_load_dword", 1);
static Register<SLoadDword> s_ld_2("s_load_dwordx2", 2);
static Register<SLoadDword> s_ld_3("s_load_dwordx3", 3);
static Register<SLoadDword> s_ld_4("s_load_dwordx4", 4);
static Register<SLoadDword> s_ld_8("s_load_dwordx8", 8);
static Register<SLoadDword> s_ld_16("s_load_dwordx16", 16);

// RDNA-style aliases (s_load_b<bits> = s_load_dword[xN], same semantics)
static Register<SLoadDword> s_ld_b32("s_load_b32", 1);
static Register<SLoadDword> s_ld_b64("s_load_b64", 2);
static Register<SLoadDword> s_ld_b96("s_load_b96", 3);
static Register<SLoadDword> s_ld_b128("s_load_b128", 4);
static Register<SLoadDword> s_ld_b256("s_load_b256", 8);
static Register<SLoadDword> s_ld_b512("s_load_b512", 16);

// Sub-dword Memory

static Register<GlobalLoad<uint16_t>> gl_ld_u16("global_load_ushort", 1);
// RDNA name for global_load_ushort
static Register<GlobalLoad<uint16_t>> gl_ld_u16_rdna("global_load_u16", 1);
static Register<GlobalLoad<uint8_t>> gl_ld_u8("global_load_ubyte", 1);
static Register<GlobalStore<uint16_t>> gl_st_u16("global_store_short");
static Register<GlobalStore<uint16_t>> gl_st_u16_rdna("global_store_b16");
static Register<GlobalStore<uint8_t>> gl_st_u8("global_store_byte");
static Register<GlobalStore<uint16_t, 16>>
    gl_st_u16hi("global_store_short_d16_hi", 1);
// RDNA name for global_store_short_d16_hi
static Register<GlobalStore<uint16_t, 16>>
    gl_st_d16hi_rdna("global_store_d16_hi_b16", 1);

// DS Memory
static Register<DsRead<uint8_t, 1>> ds_rd_u8("ds_read_u8");
static Register<DsRead<uint16_t, 1>> ds_rd_u16("ds_read_u16");
static Register<DsRead<int8_t, 1>> ds_rd_i8("ds_read_i8");
static Register<DsRead<int16_t, 1>> ds_rd_i16("ds_read_i16");
static Register<DsRead<uint32_t, 1>> ds_rd_32("ds_read_b32");
// RDNA name for ds_read_b32
// RDNA aliases: ds_load = ds_read, ds_store = ds_write
static Register<DsRead<uint32_t, 1>> ds_ld_32("ds_load_b32");
static Register<DsRead<uint32_t, 2>> ds_ld_64("ds_load_b64");
static Register<DsRead<uint32_t, 4>> ds_ld_128("ds_load_b128");
static Register<DsRead<uint16_t, 1>> ds_ld_u16("ds_load_u16");
static Register<DsRead<uint8_t, 1>> ds_ld_u8("ds_load_u8");
static Register<DsRead<uint16_t, 1>> ds_ld_u16_d16("ds_load_u16_d16", true,
                                                   false);
static Register<DsRead<uint16_t, 1>> ds_ld_u16_d16_hi("ds_load_u16_d16_hi",
                                                      true, true);
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
// RDNA name for ds_write_b32
static Register<DsWrite<uint32_t>> ds_st_32("ds_store_b32", 1, false);
static Register<DsWrite<uint32_t>> ds_st_64("ds_store_b64", 2, false);
static Register<DsWrite<uint32_t>> ds_st_96("ds_store_b96", 3, false);
static Register<DsWrite<uint32_t>> ds_st_128("ds_store_b128", 4, false);
static Register<DsWrite<uint32_t>> ds_wr_64("ds_write_b64", 2, false);
static Register<DsWrite<uint32_t>> ds_wr_96("ds_write_b96", 3, false);
static Register<DsWrite<uint32_t>> ds_wr_128("ds_write_b128", 4, false);

static Register<DsWrite<uint16_t>> ds_wr_16_hi("ds_write_b16_d16_hi", 1, true);
static Register<DsWrite<uint8_t>> ds_wr_8_hi("ds_write_b8_d16_hi", 1, true);
// RDNA3+ renames
static Register<DsWrite<uint16_t>> ds_st_16("ds_store_b16", 1, false);
static Register<DsWrite<uint16_t>> ds_st_16_hi("ds_store_b16_d16_hi", 1, true);

// ds_write2_b64: dual-offset write of 2x 8 bytes to LDS.
// Syntax: ds_write2_b64 vaddr, v[data0:data0+1], v[data1:data1+1] offset0:N
// offset1:M Writes 8 bytes from data0 to LDS[vaddr + offset0*8] and 8 bytes
// from data1 to LDS[vaddr + offset1*8].
class DsWrite2B64 : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto tokens = getPartitioned(line);
    // tokens: [op, vaddr, data0, data1, offset0:N, offset1:M]
    auto addrReg = wave.getFirstRegister(tokens[1]);
    auto data0Reg = wave.getFirstRegister(tokens[2]);
    auto data1Reg = wave.getFirstRegister(tokens[3]);

    int32_t offset0 = 0;
    int32_t offset1 = 0;
    for (size_t i = 4; i < tokens.size(); ++i) {
      if (tokens[i].starts_with("offset0:")) {
        offset0 = getIntFromView<int32_t>(tokens[i].substr(8));
      } else if (tokens[i].starts_with("offset1:")) {
        offset1 = getIntFromView<int32_t>(tokens[i].substr(8));
      }
    }

    std::vector<uint32_t> waveRead;
    for (int i = 0; i < 2; ++i) {
      waveRead.push_back(data0Reg.index + i);
    }
    for (int i = 0; i < 2; ++i) {
      waveRead.push_back(data1Reg.index + i);
    }

    return [&wave, addrReg, data0Reg, data1Reg, offset0, offset1, waveRead]() {
      IntervalSet intervals;
      intervals.reserve(wave.getWaveSize() * 4);
      auto run = [&](int lane) {
        uint32_t vAddr = wave.getVgpr(addrReg.index, lane);

        // Write first 8 bytes (2 dwords) at vAddr + offset0*8
        uint32_t addr0 = vAddr + static_cast<uint32_t>(offset0) * 8;
        for (int i = 0; i < 2; ++i) {
          uint32_t val = wave.getVgpr(data0Reg.index + i, lane);
          int64_t addr = addr0 + i * 4;
          wave.writeLds<uint32_t>(addr, lane, val);
          intervals.append(static_cast<int>(addr), static_cast<int>(addr + 4));
        }

        // Write second 8 bytes (2 dwords) at vAddr + offset1*8
        uint32_t addr1 = vAddr + static_cast<uint32_t>(offset1) * 8;
        for (int i = 0; i < 2; ++i) {
          uint32_t val = wave.getVgpr(data1Reg.index + i, lane);
          int64_t addr = addr1 + i * 4;
          wave.writeLds<uint32_t>(addr, lane, val);
          intervals.append(static_cast<int>(addr), static_cast<int>(addr + 4));
        }
      };

      auto pc = wave.getPc();
      wave.runExecConditionedForLanes(run);
      intervals.finalize();
      wave.registerVgprToLdsEvent(pc, waveRead, intervals);
      return pc + 1;
    };
  }
};
static Register<DsWrite2B64> ds_wr2_64("ds_write2_b64");
// RDNA alias
static Register<DsWrite2B64> ds_st2_64("ds_store_2addr_b64");

// ds_read2_b64: dual-offset read of 2x 8 bytes from LDS.
// Syntax: ds_read2_b64 v[dst:dst+3], vaddr offset0:N offset1:M
// Reads 8 bytes from LDS[vaddr + offset0*8] into v[dst:dst+1]
// and 8 bytes from LDS[vaddr + offset1*8] into v[dst+2:dst+3].
class DsRead2B64 : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto tokens = getPartitioned(line);
    auto dstReg = wave.getFirstRegister(tokens[1]);
    auto addrReg = wave.getFirstRegister(tokens[2]);

    int32_t offset0 = 0;
    int32_t offset1 = 0;
    for (size_t i = 3; i < tokens.size(); ++i) {
      if (tokens[i].starts_with("offset0:")) {
        offset0 = getIntFromView<int32_t>(tokens[i].substr(8));
      } else if (tokens[i].starts_with("offset1:")) {
        offset1 = getIntFromView<int32_t>(tokens[i].substr(8));
      }
    }

    std::vector<uint32_t> waveWritten;
    for (int i = 0; i < 4; ++i) {
      waveWritten.push_back(dstReg.index + i);
    }

    return [&wave, dstReg, addrReg, offset0, offset1, waveWritten]() {
      IntervalSet intervals;
      intervals.reserve(wave.getWaveSize() * 4);

      auto run = [&](int lane) {
        uint32_t vAddr = wave.getVgpr(addrReg.index, lane);

        // Read first 8 bytes (2 dwords) from vAddr + offset0*8
        uint32_t addr0 = vAddr + static_cast<uint32_t>(offset0) * 8;
        for (int i = 0; i < 2; ++i) {
          uint32_t finalAddr = addr0 + i * 4;
          intervals.append(static_cast<int>(finalAddr),
                           static_cast<int>(finalAddr + 4));
          uint32_t val =
              wave.readLds<uint32_t>(static_cast<int>(finalAddr), lane);
          wave.setVgpr(dstReg.index + i, lane, val);
        }

        // Read second 8 bytes (2 dwords) from vAddr + offset1*8
        uint32_t addr1 = vAddr + static_cast<uint32_t>(offset1) * 8;
        for (int i = 0; i < 2; ++i) {
          uint32_t finalAddr = addr1 + i * 4;
          intervals.append(static_cast<int>(finalAddr),
                           static_cast<int>(finalAddr + 4));
          uint32_t val =
              wave.readLds<uint32_t>(static_cast<int>(finalAddr), lane);
          wave.setVgpr(dstReg.index + 2 + i, lane, val);
        }
      };

      wave.runExecConditionedForLanes(run);
      intervals.finalize();
      auto pc = wave.getPc();
      wave.registerLdsToVgprEvent(pc, waveWritten, intervals);
      return pc + 1;
    };
  }
};
static Register<DsRead2B64> ds_rd2_64("ds_read2_b64");
// RDNA alias
static Register<DsRead2B64> ds_ld2_64("ds_load_2addr_b64");
} // namespace
} // namespace raceemulator
