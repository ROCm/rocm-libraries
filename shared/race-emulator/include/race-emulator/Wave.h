// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "CommonRegister.h"
#include "IntervalSet.h"
#include "Profiler.h"
#include "Types.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

class Workgroup;

/// An assembly macro (.macro / .endm) with its line range and argument names.
class Macro {
public:
  Macro() = default;
  Macro(int startLine, int endLine, const std::vector<std::string> &argNames)
      : startLine(startLine), endLine(endLine), argNames(argNames) {}

  int getStartLine() const { return startLine; }
  int getEndLine() const { return endLine; }
  const std::vector<std::string> &getArgNames() const { return argNames; }

  std::string str() const;

private:
  int startLine;
  int endLine;
  std::vector<std::string> argNames;
};

/// A parsed instruction operand: either a literal value or a register reference.
template <typename T> struct Operand {
  bool isLiteral;
  CommonRegister reg;
  T literalValue;

  void appendStr(std::ostream &os) const {
    if (isLiteral) {
      os << "Literal(" << literalValue << ")";
    } else {
      os << "Register";
    }
  }
};

class Macro;

/// A single SIMD wave. Manages VGPRs, SGPRs, exec mask, program counter,
/// memory event tracking, and race checking. Delegates LDS storage and
/// cross-wave event coordination to the Workgroup.
class Wave {

public:
  /// Full constructor with all parameters.
  Wave(int vgprCount, int agprOffset, int sgprCount, WaveSize,
       WaveId, Workgroup &workgroup,
       const std::map<std::string, int> *labels,
       const std::map<std::string, Macro> *macros);

  /// Convenience constructor for tests (no accumulators, labels, or macros;
  /// waveId defaults to zero).
  Wave(int vgprCount, int sgprCount, WaveSize, Workgroup &workgroup);

  /// Parse a register string (e.g., "s17", "s[16:17]", "v[4:7]", "vcc",
  /// "exec") and return the first register in the range.
  CommonRegister getFirstRegister(std::string_view regStr) const;

  /// Number of outstanding events of a given type on a register (across all
  /// lanes). Used by getVgprs() to skip per-lane checks when zero.
  int getRegEventCount(MemoryEventType type, int reg) const {
    return regEventCount[static_cast<int>(type)][reg];
  }

  /// Return the value in the given VGPR. Throws RaceConditionException if
  /// race checks are enabled and an outstanding load targets this register.
  uint32_t getVgpr(int reg, int lane) const;

  /// Read all lanes of a single VGPR into out[0..waveSize-1]. Race checking
  /// is O(1) per register via regEventCount; only falls back to per-lane
  /// checking when outstanding events exist.
  void getVgprs(int reg, uint32_t *out) const;

  /// Return a single byte from a VGPR. Race-checks only that byte.
  /// byteIdx 0 = bits [7:0], 3 = bits [31:24].
  uint8_t getVgprByte(int id, int lane, int byteIdx) const;

  /// Return 16 bits from a VGPR. Race-checks only the relevant 2 bytes.
  /// hi=false: bits [15:0], hi=true: bits [31:16].
  uint16_t getHalfVgpr(int id, int lane, bool hi) const;

  /// Write 16 bits directly to VGPR storage without reading the other half.
  /// Matches ISA d16 semantics.
  void setVgprHalf(int id, int lane, bool hi, uint16_t value);

  // TODO(newling) implement race checks for scalar registers.
  uint32_t getSgpr(int id) const;

  void setVgpr(int reg, int lane, uint32_t value) {
    assert((getExecU64() & (1ULL << lane)) != 0 &&
           "Writing to VGPR of inactive lane");
    auto index = reg * waveSize + lane;
    assert(index < static_cast<int64_t>(vgprs.size()));
    vgprs[index] = value;
  }
  void setSgpr(int id, uint32_t value);

  /// Get/set a pair of consecutively numbered 32-bit registers as uint64_t.
  uint64_t getSgpr64(int id) const;
  uint64_t getVgpr64(int id, int lane) const;
  void setSgpr64(int id, uint64_t value);
  void setVgpr64(int id, int lane, uint64_t value);

  WaveSize getWaveSize() const { return waveSize; }

  int getSgprCount() const { return sgprCount; }

  void tryExecute(const std::string &line, bool enableLineCaching);

  void setScc(bool value);
  bool getScc() const;

  void setM0(uint32_t value);
  uint32_t getM0() const;

  void setVccU32(uint32_t value);
  uint32_t getVccU32() const;
  void setVccU64(uint64_t value);
  uint64_t getVccU64() const;

  void setExecU32(uint32_t value);
  uint32_t getExecU32() const;

  /// Exec mask as uint64_t. Safe for both wave-32 and wave-64: on wave-32
  /// the upper 32 bits are always zero.
  void setExecU64(uint64_t value);
  uint64_t getExecU64() const;

  Workgroup &getWorkgroup();

  /// Validated LDS read: auto-supplies waveId and delegates to Workgroup.
  template <typename T> T readLds(int addr, int lane) const;

  /// Validated bulk LDS read: validates the full range once, then copies
  /// count elements of type T.
  template <typename T>
  void readLds(int addr, int lane, T *out, int count) const;

  /// Validated LDS write: auto-supplies waveId and delegates to Workgroup.
  template <typename T> void writeLds(int addr, int lane, T value);

  /// Register an in-flight memory event. The event remains outstanding until
  /// retired by s_waitcnt. byteMask selects which bytes of each register are
  /// covered (0xF = all 4 bytes).
  void registerGlobalToVgprEvent(int pc, const std::vector<uint32_t> &registers,
                                 uint8_t byteMask = 0xF);
  void registerVgprToGlobalEvent(int pc,
                                 const std::vector<uint32_t> &registers);
  void registerLdsToVgprEvent(int pc, const std::vector<uint32_t> &registers,
                              const IntervalSet &ldsIntervals,
                              uint8_t byteMask = 0xF);
  void registerVgprToLdsEvent(int pc, const std::vector<uint32_t> &registers,
                              const IntervalSet &ldsIntervals);

  /// Direct-to-LDS: no VGPR registers, only LDS intervals. Counted by vmcnt.
  void registerGlobalToLdsEvent(int pc, const IntervalSet &ldsIntervals);

  /// True if any outstanding store reads from the given VGPR lane.
  bool isOutstandingFromVgpr(int lane, int reg) const;

  bool isRaceChecks() const;
  bool isCompleteEmulation() const;

  /// Retire global memory events until at most vmcnt remain outstanding.
  void sWaitCntVmcnt(int vmcnt);

  /// Retire LDS events until at most lgkmcnt remain outstanding.
  void sWaitCntLgkmcnt(int lgkmcnt);

  std::vector<EventId> &getVgprMemoryEvents(int reg) {
    assert(reg < static_cast<int>(vgprMemoryEvents.size()));
    return vgprMemoryEvents[reg];
  }

  const std::vector<EventId> &getWaveMemoryEvents() const {
    return waveMemoryEvents;
  }

  const std::vector<EventId> &getWaveCompleteMemoryEvents() const {
    return waveCompleteMemoryEvents;
  }

  WaveId getWaveId() const { return waveId; }

  Profiler::ScopedStopwatch profileScope(std::string_view key);

  /// Discard all wave-complete events. Called when all waves have reached
  /// s_barrier (not when individual waves arrive).
  void flushWaveCompleteMemoryEvents();

  template <typename T> T getValue(const Operand<T> &operand, int lane) const;
  template <typename T>
  T getSgprOrLiteralValue(const Operand<T> &operand) const;
  template <typename T> Operand<T> parseOperand(std::string_view token) const;

  void setPc(int newPc) { pc = newPc; }
  int getPc() const { return pc; }

  const std::map<std::string, int> &getLabels() const {
    assert(labels != nullptr && "Labels map is null");
    return *labels;
  }

  void setDsPreserve(bool preserve) { dsPreserve = preserve; }
  bool getDsPreserve() const { return dsPreserve; }

  void deactivate() { active = false; }
  void setWaiting(bool w) { waiting = w; }
  bool isActive() const { return active; }
  bool isWaiting() const { return waiting; }

private:
  // When a macro is called, its arguments are stored here. E.g., for
  // `.macro FOO arg0:req, arg1:req` called as `FOO 14, 17`, the map
  // holds {"arg0": 14, "arg1": 17}.
  std::map<std::string, uint32_t> macroArguments;

  // When inside a macro, this is the program counter (PC) to return to.
  int macroReturnPc;

  // The current program counter (the current line in the assembly).
  int pc{0};

  // accumulator and vector general purpose registers.
  std::vector<uint32_t> vgprs;

  // scalar general purpose registers.
  std::vector<uint32_t> sgprs;

  // Per register: outstanding event IDs involving that VGPR.
  // Indexed as [reg]. Each event's exec mask (in the workgroup registry)
  // records which lanes it applies to. Race checks test the lane bit
  // when needed. This avoids 32x per-lane push/pop during event
  // registration and retirement.
  std::vector<std::vector<EventId>> vgprMemoryEvents;

  // Per-register event counts, partitioned by event type, summed across lanes.
  // regEventCount[eventType][reg] == 0 means no lane of that register has
  // any outstanding events of that type. Enables O(1) per-register checks:
  // getVgprs() only inspects lanes when GLOBAL_TO_VGPR or LDS_TO_VGPR
  // counts are nonzero.
  static constexpr int kNumEventTypes = static_cast<int>(MemoryEventType::N);
  std::array<std::vector<int>, kNumEventTypes> regEventCount;

  void regEventCountInc(MemoryEventType type, int reg) {
    regEventCount[static_cast<int>(type)][reg]++;
  }
  void regEventCountDec(MemoryEventType type, int reg) {
    regEventCount[static_cast<int>(type)][reg]--;
  }

  // All the outstanding event IDs for this wave.
  std::vector<EventId> waveMemoryEvents;

  // Event IDs that have completed (due to s_waitcnt) for this wave, but are
  // not complete for the entire workgroup because s_barrier has not yet
  // occurred.
  std::vector<EventId> waveCompleteMemoryEvents;

  int agprOffset;
  int sgprCount;
  WaveSize waveSize;

  /// Whether this wave is still executing (false after s_endpgm).
  bool active{true};

  /// Whether this wave is waiting at an s_barrier.
  bool waiting{false};

  /// D16 LDS reads: whether to preserve the non-targeted half of the VGPR.
  /// RDNA (wave-32) preserves, CDNA (wave-64) does not (hardware-verified).
  bool dsPreserve;

  // The id of this wave within the workgroup.
  WaveId waveId{0};

  // Pointer to the workgroup state (LDS memory, event registry).
  Workgroup *workgroup;

  // Pointer to the label map for the assembly.
  const std::map<std::string, int> *labels;

  // Point to the macro map for the assembly.
  const std::map<std::string, Macro> *macros;

  std::vector<std::function<int()>> instructionCache;

  void retireEventRegisters(EventId);

  void resolveWaitCnt(int limit,
                      std::function<bool(MemoryEventType)> isTargetType);

  std::function<int()> compileLine(const std::string &line,
                                   const std::map<std::string, Macro> &macros);

public:
  template <typename F> void runExecConditionedForLanes(F func) {

    int waveSize = getWaveSize();

    uint64_t execMask = getExecU64();

    // 1. Calculate the 'all active'
    uint64_t fullMask = (waveSize == 64) ? ~0ULL : ((1ULL << waveSize) - 1);

    // 2. Fast path: All lanes enabled
    if ((execMask & fullMask) == fullMask) {
      for (int lane = 0; lane < waveSize; ++lane) {
        func(lane);
      }
    }
    // 3. Slow path: Check bits
    else {
      for (int lane = 0; lane < waveSize; ++lane) {
        if ((execMask >> lane) & 1) {
          func(lane);
        }
      }
    }
  }
};

} // namespace raceemulator
