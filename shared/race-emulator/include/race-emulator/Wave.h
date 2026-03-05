#pragma once
#include "CommonRegister.h"
#include "Util.h"
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

class RegisterMemoryEvent {
public:
  RegisterMemoryEvent(uint32_t eventId, MemoryEventType type)
      : eventId(eventId), type(type) {}

  std::string str() const;
  uint32_t eventId;
  MemoryEventType type;
};

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

// An operand that knows if it is a Literal or a Register
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

class LDS;
class Macro;

class WaveMemoryEvent {
public:
  uint32_t pc;
  uint32_t eventId;
  MemoryEventType type;
  std::vector<uint32_t> registers;
  LaneAndLDSBytes ldsBytes;
  uint64_t mask;

  std::string str() const;

  static WaveMemoryEvent
  createGlobalToVgprEvent(uint32_t pc, uint32_t eventId,
                          const std::vector<uint32_t> &registers,
                          const uint64_t mask) {

    WaveMemoryEvent event;
    event.pc = pc;
    event.eventId = eventId;
    event.type = MemoryEventType::GLOBAL_TO_VGPR;
    event.registers = registers;
    event.mask = mask;
    return event;
  }

  static WaveMemoryEvent
  createVgprToGlobalEvent(uint32_t pc, uint32_t eventId,
                          const std::vector<uint32_t> &registers,
                          const uint64_t mask) {

    WaveMemoryEvent event;
    event.pc = pc;
    event.eventId = eventId;
    event.type = MemoryEventType::VGPR_TO_GLOBAL;
    event.registers = registers;
    event.mask = mask;
    return event;
  }

  static WaveMemoryEvent
  createLdsToVgprEvent(uint32_t pc, uint32_t eventId,
                       const std::vector<uint32_t> &registers,
                       const LaneAndLDSBytes &ldsBytes, const uint64_t mask) {
    WaveMemoryEvent event;
    event.pc = pc;
    event.eventId = eventId;
    event.type = MemoryEventType::LDS_TO_VGPR;
    event.registers = registers;
    event.ldsBytes = ldsBytes;
    event.mask = mask;
    return event;
  }

  static WaveMemoryEvent
  createVgprToLdsEvent(uint32_t pc, uint32_t eventId,
                       const std::vector<uint32_t> &registers,
                       const LaneAndLDSBytes &ldsBytes, const uint64_t mask) {
    WaveMemoryEvent event;
    event.pc = pc;
    event.eventId = eventId;
    event.type = MemoryEventType::VGPR_TO_LDS;
    event.registers = registers;
    event.ldsBytes = ldsBytes;
    event.mask = mask;
    return event;
  }
};

class Wave {

public:
  static Wave createGfx942(int waveId = 0);

  // vgprCount:  total number of vector and accumulator registers.
  //
  // agprOffset: starting index of accumulator registers.
  //
  // sgprCount:  total number of scalar registers.
  //
  // waveSize:   number of lanes in the wave (32 or 64).
  //
  // waveId:     the id of this wave within the workgroup.
  //
  // lds:        optional pointer to the LDS memory of the workgroup.
  //             If null, no LDS operations are allowed.
  //
  // labels:     optional pointer to the label map for the assembly.
  //             If null, no branching to labels is allowed.
  Wave(int vgprCount, int agprOffset, int sgprCount, int waveSize, int waveId,
       LDS *lds, const std::map<std::string, int> *labels,
       const std::map<std::string, Macro> *macros);

  // Construct a wave without accumulator registers, and without LDS or
  // labels. The waveId is set to zero.
  Wave(int vgprCount, int sgprCount, int waveSize);

  // Example s17      -> SGPR 17
  //         s[16:17] -> SGPR 16
  //         v5       -> VGPR 5
  //         v[4:7]   -> VGPR 4
  //
  // Also supports 'acc', 'm0', 'vcc', 'exec', mapping them to specific
  // scalar register assigned for emulation.
  CommonRegister getFirstRegister(std::string_view regStr) const;

  // Return the value in the given register. If race checks are enabled, will
  // raise an exception if the value in the register is currently in a race.
  uint32_t getVgpr(int id, int lane) const;

  // TODO(newling) implement race checks for scalar registers.

  // Return the value in the given register.
  uint32_t getSgpr(int id) const;

  void setVgpr(int id, int lane, uint32_t value);
  void setSgpr(int id, uint32_t value);

  // Get a pair of consecutively numbered 32-bit registers.
  uint64_t getSgpr64(int id) const;
  uint64_t getVgpr64(int id, int lane) const;

  // Set a pair of consecutively numbered 32-bit registers.
  void setSgpr64(int id, uint64_t value);
  void setVgpr64(int id, int lane, uint64_t value);

  int getWaveSize() const { return waveSize; }

  int getSgprCount() const { return sgprCount; }

  void tryExecute(const std::string &line, bool enableLineCaching);

  void setScc(bool value);
  bool getScc() const;

  void setM0(uint32_t value);
  uint32_t getM0() const;

  // For wave-32 we'll need a getVccU32, but for now wave-64 only.
  void setVccU64(uint64_t value);
  uint64_t getVccU64() const;

  void setExecU64(uint64_t value);
  uint64_t getExecU64() const;

  LDS &getLds();

  // Called by instructions like global load.
  void registerGlobalToVgprEvent(int pc,
                                 const std::vector<uint32_t> &registers);

  // Called by instructions like global store.
  void registerVgprToGlobalEvent(int pc,
                                 const std::vector<uint32_t> &registers);

  // Called by instructions like lds read.
  void registerLdsToVgprEvent(int pc, const std::vector<uint32_t> &registers,
                              const std::vector<LaneAndLDSByte> &ldsBytes);

  // Called by instructions like lds write.
  void registerVgprToLdsEvent(int pc, const std::vector<uint32_t> &registers,
                              const std::vector<LaneAndLDSByte> &ldsBytes);

  // Check if there are any outstanding memory events TO a specific vgpr, and
  // in a specific lane.
  bool isOutstandingToVgpr(int lane, int reg) const;

  // Check if there are any outstanding memory events FROM a specific vgpr, and
  // in a specific lane.
  bool isOutstandingFromVgpr(int lane, int reg) const;

  void enableRaceChecks(bool enable) { raceChecks = enable; }
  void enableCompleteEmulation(bool enable) { completeEmulation = enable; }

  bool isCompleteEmulation() const { return completeEmulation; }

  // Reduce the number of outstanding memory events down to `vmcnt` for global
  // memory operations.
  void sWaitCntVmcnt(int vmcnt);

  // Reduce the number of outstanding memory events down to `lgkmcnt` for
  // LDS memory operations.
  void sWaitCntLgkmcnt(int lgkmcnt);

  std::vector<RegisterMemoryEvent> &getVgprMemoryEvents(int reg, int lane) {
    auto index = reg * waveSize + lane;
    assert(index < static_cast<int64_t>(vgprMemoryEvents.size()));
    return vgprMemoryEvents[index];
  }

  const std::vector<WaveMemoryEvent> &getWaveMemoryEvents() const {
    return waveMemoryEvents;
  }

  const std::vector<WaveMemoryEvent> &getWaveCompleteMemoryEvents() const {
    return waveCompleteMemoryEvents;
  }

  int getWaveId() const { return waveId; }

  // s_barrier operation
  void flushWaveCompleteMemoryEvents();

  template <typename T> T getValue(const Operand<T> &operand, int lane) const;
  template <typename T>
  T getSgprOrLiteralValue(const Operand<T> &operand) const;
  template <typename T> Operand<T> parseOperand(std::string_view token) const;

  void setPc(int newPc) { pc = newPc; }
  int getPc() const { return pc; }

  // Get the label map:
  const std::map<std::string, int> &getLabels() const {
    assert(labels != nullptr && "Labels map is null");
    return *labels;
  }

  void setDsPreserve(bool preserve) { dsPreserve = preserve; }
  bool getDsPreserve() const { return dsPreserve; }

private:
  // The ISA says this should be true, but hardware testing suggests false.
  bool dsPreserve{false};

  // When a macro is called, the macro arguments are stored here.
  // For example, the assembly file might have:
  //
  // ```asm
  // [...]
  // .macro FOO arg0:req, arg1:req
  //    [...]
  // .endm
  // [...]
  // GLOBAL_OFFSET_A 14, 17
  // [...]
  // ```
  //
  // in which case, when executing the macro body, the map will have:
  //   "arg0" -> 14
  //   "arg1" -> 17.
  std::map<std::string, uint32_t> macroArguments;

  // When inside a macro, this is the program counter (PC) to return to. Note
  // that the PC in the emulator is just the line number in the assembly.
  int macroReturnPc;

  // The current program counter (the current line in the assembly) that is
  // being executed by this wave.
  int pc{0};

  // accumulator and vector general purpose registers.
  std::vector<uint32_t> vgprs;

  // scalar general purpose registers.
  std::vector<uint32_t> sgprs;

  // For every vgpr, a list of outstanding memory events involving the register.
  // These include the four types: [reads from, writes to] X [global, lds].
  std::vector<std::vector<RegisterMemoryEvent>> vgprMemoryEvents;

  // All the outstanding memory events for this wave.
  std::vector<WaveMemoryEvent> waveMemoryEvents;

  // All the memory events for this wave that have completed (due to an
  // s_waitcnt) for the wave, but are not complete for the entire workgroup
  // because an s_barrier has not yet occurred.
  std::vector<WaveMemoryEvent> waveCompleteMemoryEvents;

  // The number of vector and accumulator registers.
  int vgprCount;

  // The starting index of accumulator registers. This is effectively the number
  // of vector general purpose registers.
  int agprOffset;

  // The number of scalar registers, including special registers like scc, m0,
  // vcc, exec.
  int sgprCount;

  // The number of lanes in the wave (32 or 64 for simulating real GPUS, but can
  // be lower for unit testing).
  int waveSize;

  // The id of this wave within the workgroup.
  int waveId{0};

  // Pointer to the LDS memory of the workgroup.
  LDS *lds;

  // Pointer to the label map for the assembly.
  const std::map<std::string, int> *labels;

  // Point to the macro map for the assembly.
  const std::map<std::string, Macro> *macros;

  // Each line can get cached. This (in theory) accelerates the emulation, I
  // need to confirm that it does. Note that the cache is based on the program
  // count, not a string match for the contents of the line of assembly. Using
  // the cache means that the raw string of assembly does not need to be parsed.
  std::vector<std::function<int()>> instructionCache;

  // Race event count tracker for diagnostics.
  uint32_t eventCount{0};

  // This controls whether to perform race checks on memory operations, or not.
  bool raceChecks{false};

  bool completeEmulation{true};

  // Each register records the memory events that it is involved in. This
  // function removes `event` from the registers that it involves.
  void retireEventRegisters(const WaveMemoryEvent &event);

  void
  resolveWaitCnt(int limit, std::function<bool(MemoryEventType)> isTargetType,
                 std::function<void(const WaveMemoryEvent &)> extraCleanup);

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
