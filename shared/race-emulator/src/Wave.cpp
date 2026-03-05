#include "race-emulator/Wave.h"
#include "race-emulator/Instruction.h"
#include "race-emulator/LDS.h"
#include "race-emulator/Util.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace raceemulator {

// Underlying assumptions / guiding principles:
//
// We do not care about the time cost of visiting a line of assembly once. We
// can sink as much time as you like into making any subsequent visits of the
// line fast. i.e. we're only going to optimize lines that are in loops.

namespace {

// We set the number of registers in the emulation based on the number used in
// the kernel, rather than based on the HW. The motivation is (1) use less
// memory and (2) maybe improve forward compatibility.

// The extra scalar registers:
const constexpr int vccIndex = -2; // 2 SGPRs for VCC (u64)
// 1 SGPR for SCC (in reality it is 1 bit, we use a full SGPR)
const constexpr int sccIndex = -3;
// 2 SGPRs for EXEC (u64)
const constexpr int execIndex = -5;
// 1 SGPR for M0.
const constexpr int m0Index = -6;

const constexpr int nExtraSgrs = -m0Index + 1;

using LabelMap = std::map<std::string, int>;

} // namespace

Wave::Wave(int vgprCount, int agprOffset, int sgprCount, int waveSize,
           int waveId, LDS *lds, const std::map<std::string, int> *labels,
           const std::map<std::string, Macro> *macros)
    : vgprCount(vgprCount), agprOffset(agprOffset), sgprCount(sgprCount),
      waveSize(waveSize), waveId(waveId), lds(lds), labels(labels),
      macros(macros) {

  assert(labels != nullptr && "Labels map cannot be null");
  assert(macros != nullptr && "Macros map cannot be null");

  int avgprCount = vgprCount;

  // It seems like some code assumes that the vector registers are initialized
  // to zero, so we'll do that.
  // TODO(newling) is there something in a spec that says this is the case?

  vgprs.resize(avgprCount * waveSize, 0);  // 0x12345678);
  sgprs.resize(sgprCount + nExtraSgrs, 0); // 0x12345678);

  vgprMemoryEvents.resize(avgprCount * waveSize);

  static_assert(sizeof(0ULL) * 8 >= 64, "uint64_t must be at least 64 bits");

  // Initialize EXEC as all active.
  setExecU64(~0ULL);
}

std::string WaveMemoryEvent::str() const {
  std::string result;
  result += "Memory Event (ID: " + std::to_string(eventId) + ")\n";
  result += "  PC: " + std::to_string(pc) + "\n";
  result += "  Type: ";
  switch (type) {
  case MemoryEventType::GLOBAL_TO_VGPR:
    result += "GLOBAL_TO_VGPR\n";
    break;
  case MemoryEventType::VGPR_TO_GLOBAL:
    result += "VGPR_TO_GLOBAL\n";
    break;
  default:
    result += "UNKNOWN\n";
    break;
  }
  return result;
}

std::string RegisterMemoryEvent::str() const {
  std::string result;
  result +=
      "  Register Memory Event (Event ID: " + std::to_string(eventId) + ")\n";
  result += "    Type: ";
  switch (type) {
  case MemoryEventType::GLOBAL_TO_VGPR:
    result += "GLOBAL_TO_VGPR\n";
    break;
  case MemoryEventType::VGPR_TO_GLOBAL:
    result += "VGPR_TO_GLOBAL\n";
    break;
  default:
    result += "UNKNOWN\n";
    break;
  }
  return result;
}

void Wave::registerGlobalToVgprEvent(int pc,
                                     const std::vector<uint32_t> &regIds) {

  auto eventId = eventCount++;

  auto run = [&](int lane) {
    for (auto reg : regIds) {
      getVgprMemoryEvents(reg, lane).push_back(
          RegisterMemoryEvent(eventId, MemoryEventType::GLOBAL_TO_VGPR));
    }
  };
  runExecConditionedForLanes(run);

  auto currentMaskValue = getExecU64();
  WaveMemoryEvent event = WaveMemoryEvent::createGlobalToVgprEvent(
      pc, eventId, regIds, currentMaskValue);
  waveMemoryEvents.push_back(event);
}

void Wave::registerVgprToGlobalEvent(int pc,
                                     const std::vector<uint32_t> &wave) {

  auto eventId = eventCount++;

  auto run = [&](int lane) {
    for (auto reg : wave) {
      int32_t index = reg * waveSize + lane;
      vgprMemoryEvents[index].push_back(
          RegisterMemoryEvent(eventId, MemoryEventType::VGPR_TO_GLOBAL));
    }
  };

  runExecConditionedForLanes(run);

  auto currentMaskValue = getExecU64();

  WaveMemoryEvent event = WaveMemoryEvent::createVgprToGlobalEvent(
      pc, eventId, wave, currentMaskValue);

  waveMemoryEvents.push_back(event);
}

void Wave::registerLdsToVgprEvent(int pc, const std::vector<uint32_t> &regIds,
                                  const LaneAndLDSBytes &ldsBytes) {
  auto eventId = eventCount++;
  auto run = [&](int lane) {
    for (auto reg : regIds) {
      getVgprMemoryEvents(reg, lane).push_back(
          RegisterMemoryEvent(eventId, MemoryEventType::LDS_TO_VGPR));
    }
  };
  runExecConditionedForLanes(run);

  LDS &lds = getLds();

  // Note: ldsBytes contains all bytes touched by all active lanes.
  // It may contain duplicates if multiple lanes read the same address.
  // This is generally fine, but ensure LDS::addEvent handles it
  // (or just adds multiple entries with the same ID, which is safe).
  for (auto b : ldsBytes) {
    lds.addEvent(b.byte, LDSMemoryEvent(eventId, waveId, b.lane,
                                        MemoryEventType::LDS_TO_VGPR,
                                        LDSMemoryEvent::EventStatus::ACTIVE));
  }

  // 4. Record Wave History (for s_waitcnt lgkmcnt resolution)
  auto currentMaskValue = getExecU64();

  // You will need to add this factory method to WaveMemoryEvent
  WaveMemoryEvent event = WaveMemoryEvent::createLdsToVgprEvent(
      pc, eventId, regIds, ldsBytes, currentMaskValue);

  waveMemoryEvents.push_back(event);
}

void Wave::registerVgprToLdsEvent(int pc, const std::vector<uint32_t> &wave,
                                  const LaneAndLDSBytes &ldsBytes) {

  auto eventId = eventCount++;

  // 2. Mark Source VGPRs as 'Pending Read'
  //    (Used to detect WAR hazards on wave)
  auto run = [&](int lane) {
    for (auto reg : wave) {
      int32_t index = reg * waveSize + lane;
      vgprMemoryEvents[index].push_back(
          RegisterMemoryEvent(eventId, MemoryEventType::VGPR_TO_LDS));
    }
  };
  runExecConditionedForLanes(run);

  // 3. Mark Destination LDS Bytes as 'Active Write'
  LDS &lds = getLds();
  for (auto b : ldsBytes) {
    lds.addEvent(b.byte, LDSMemoryEvent(eventId, waveId, b.lane,
                                        MemoryEventType::VGPR_TO_LDS,
                                        LDSMemoryEvent::EventStatus::ACTIVE));
  }

  // 4. Record Wave History
  auto currentMaskValue = getExecU64();
  WaveMemoryEvent event = WaveMemoryEvent::createVgprToLdsEvent(
      pc, eventId, wave, ldsBytes, currentMaskValue);

  waveMemoryEvents.push_back(event);
}

uint32_t Wave::getVgpr(int reg, int lane) const {

  int32_t index = reg * waveSize + lane;
  assert(index < static_cast<int64_t>(vgprs.size()));

  if (raceChecks && isOutstandingToVgpr(lane, reg)) {
    throw RaceConditionException::Vgpr(reg, waveId, lane, false /* isWrite */);
  }

  return vgprs[index];
}

void Wave::setVgpr(int reg, int lane, uint32_t value) {
  // Assert that exec is active for this lane:
  assert((getExecU64() & (1ULL << lane)) != 0 &&
         "Writing to VGPR of inactive lane");

  auto index = reg * waveSize + lane;

  assert(index < static_cast<int64_t>(vgprs.size()));
  vgprs[index] = value;
}

uint32_t Wave::getSgpr(int id) const {

  id = id < 0 ? id + sgprs.size() : id;

  if (id >= static_cast<int64_t>(sgprs.size())) {
    throw std::runtime_error("SGPR index out of range: " + std::to_string(id) +
                             ". Max SGPRs: " + std::to_string(sgprs.size()));
  }

  return sgprs[id];
}

void Wave::setSgpr(int id, uint32_t value) {
  id = id < 0 ? id + sgprs.size() : id;
  assert(id < static_cast<int64_t>(sgprs.size()));
  sgprs[id] = value;
}

void Wave::setSgpr64(int id, uint64_t value) {
  id = id < 0 ? id + sgprs.size() : id;
  assert(id + 1 < static_cast<int64_t>(sgprs.size()));
  setSgpr(id, static_cast<uint32_t>(value));
  setSgpr(id + 1, static_cast<uint32_t>(value >> 32));
}

uint64_t Wave::getSgpr64(int id) const {
  id = id < 0 ? id + sgprs.size() : id;
  uint64_t low = getSgpr(id);
  uint64_t high = getSgpr(id + 1);
  return (high << 32) | low;
}

uint64_t Wave::getVgpr64(int id, int lane) const {
  assert(id % 2 == 0);
  uint64_t low = getVgpr(id, lane);
  uint64_t high = getVgpr(id + 1, lane);
  return (high << 32) | low;
}

void Wave::setVgpr64(int id, int lane, uint64_t value) {
  assert(id % 2 == 0);
  setVgpr(id, lane, static_cast<uint32_t>(value));
  setVgpr(id + 1, lane, static_cast<uint32_t>(value >> 32));
}

std::function<int()>
Wave::compileLine(const std::string &line,
                  // const std::map<std::string, int> &labelMap,
                  const std::map<std::string, Macro> &macros) {

  auto currentPc = pc;
  auto nullOpt = [currentPc]() -> int { return currentPc + 1; };

  if (line.find_first_not_of(" \t\r\n'") == std::string::npos) {
    return nullOpt;
  }

  // Check if the line is a label.
  // If it is, just increment currentPc.
  if (labels) {
    const auto &labelMap = *labels;
    auto firstNonSpace = line.find_first_not_of(" \t");
    auto firstColon = line.find(':', firstNonSpace);
    if (firstColon != std::string::npos && firstColon > firstNonSpace) {
      auto labelName = line.substr(firstNonSpace, firstColon);
      auto foundLabel = labelMap.find(labelName);
      if (foundLabel != labelMap.end()) {
        // instructionCache[currentPc] = nullOpt;
        return nullOpt;
        // currentPc + 1;
      }
    }
  }

  auto partitioned = getPartitioned(line);
  assert(!partitioned.empty() && "Empty partitioned line");

  if (partitioned[0] == ".macro") {
    auto found = macros.find(std::string(partitioned[1]));
    if (found == macros.end()) {
      throw std::runtime_error("Macro not found: " +
                               std::string(partitioned[1]));
    }
    // We jump to 1 after .mend
    int mendLine = found->second.getEndLine();
    return [mendLine]() -> int { return mendLine + 1; };
  }

  if (partitioned[0] == ".endm") {
    // Clear the symbol table:
    return [this]() -> int {
      macroArguments.clear();
      auto currentPc = macroReturnPc;
      macroReturnPc = -1;
      return currentPc;
    };
  }

  // If the line starts with .set , it's a null opt:
  if (partitioned[0] == ".set") {
    return nullOpt;
  }

  if (partitioned[0] == ".align") {
    return nullOpt;
  }

  const auto &instructions = getInstructions();

  // strip _e32 or _e64 suffixes for matching:
  std::string instructionWithoutEncodingSize = std::string(partitioned[0]);
  auto e32Pos = instructionWithoutEncodingSize.find("_e32");
  ;
  if (e32Pos != std::string::npos) {
    instructionWithoutEncodingSize =
        instructionWithoutEncodingSize.substr(0, e32Pos);
  }

  auto e64Pos = instructionWithoutEncodingSize.find("_e64");
  if (e64Pos != std::string::npos) {
    instructionWithoutEncodingSize =
        instructionWithoutEncodingSize.substr(0, e64Pos);
  }

  // auto instructionWithoutEncodingSize
  auto found = instructions.find(instructionWithoutEncodingSize);

  if (found != instructions.end()) {
    // JIT Compile: Create the executable lambda
    auto exec = found->second->getExecutor(*this, line); // currentPc, *labels);

    // Store in Cache
    // instructionCache[currentPc] = exec;

    // Execute immediately
    return exec;
  }

  return nullptr;
}

// macros is [where is .macro, where is .mend]
void Wave::tryExecute(const std::string &line_,
                      // const std::map<std::string, int> &labelMap,
                      // const std::map<std::string, Macro> &macros,
                      bool enableLineCaching) {

  std::string line = line_;
  if (!macroArguments.empty()) {
    line = getSymbolReducedLine(line, macroArguments);
    // assert(false && "can use symbol table here!");
  }

  auto nxt = [&]() {
    // Fast path (second+ time line is visited).
    if (pc < static_cast<int>(instructionCache.size()) &&
        instructionCache[pc] != nullptr) {
      return instructionCache[pc]();
    }

    // Slow path (first time line is visited).
    auto func = compileLine(line, *macros);
    // labelMap, macros);

    if (pc >= static_cast<int>(instructionCache.size())) {
      instructionCache.resize(pc + 16, nullptr);
    }

    bool isCacheable = enableLineCaching && macroArguments.empty();

    if (func != nullptr) {
      if (isCacheable) {
        instructionCache[pc] = func;
      }
      auto nxt = func();
      // instructionCache[currentPc]();
      return nxt;
    } else {

      auto partitioned = getPartitioned(line);

      // Check if partitioned[0] is in macros:
      auto iter = macros->find(std::string(partitioned[0]));
      if (iter != macros->end()) {
        auto macroRange = iter->second;
        auto macroStart = macroRange.getStartLine();
        const auto &argNames = iter->second.getArgNames();
        for (size_t i = 0; i < argNames.size(); ++i) {
          // get an integer value from partitioned[i + 1]:
          uint32_t value = 0;
          auto parsed = parseNumber(partitioned[i + 1], value);
          if (!parsed) {
            throw std::runtime_error("Error parsing macro argument: " +
                                     std::string(partitioned[i + 1]));
          }
          macroArguments.insert({"\\" + argNames[i], value});
          macroReturnPc = pc + 1;
        }
        return macroStart + 1;
      }

      else {
        throw std::runtime_error("Unimplemented instruction: " + line);
      }
    }
  }();

  // Get the instruction from the line.
  // Specifically, get the first string in the line (split on space).
  auto firstSpace = line.find(' ');
  std::string instructionName;
  if (firstSpace != std::string::npos) {
    instructionName = line.substr(0, firstSpace);
  } else {
    instructionName = line;
  }
  setPc(nxt);
}

void Wave::setScc(bool value) { setSgpr(sccIndex, value); }
bool Wave::getScc() const { return getSgpr(sccIndex) != 0; }

uint64_t Wave::getVccU64() const { return getSgpr64(vccIndex); }
uint64_t Wave::getExecU64() const {

  auto m = getSgpr64(execIndex);

  return m;
}

const static std::map<std::string, int> emptyLabels = {};
const static std::map<std::string, Macro> emptyMacros = {};

// Construct a wave without accumulator registers, and without LDS or
// labels. The waveId is set to zero.
Wave::Wave(int vgprCount, int sgprCount, int waveSize)
    : Wave(vgprCount, /* agprOffset= */ vgprCount, sgprCount, waveSize,
           /* waveId= */ 0, /* lds= */ nullptr, /* labels= */ &emptyLabels,
           /* macros= */ &emptyMacros) {}

void Wave::setVccU64(uint64_t value) { setSgpr64(vccIndex, value); }
void Wave::setExecU64(uint64_t value) { setSgpr64(execIndex, value); }

void Wave::setM0(uint32_t value) { setSgpr(m0Index, value); }
uint32_t Wave::getM0() const { return getSgpr(m0Index); }

// Helper 1: Removes the event lock from all affected VGPRs
void Wave::retireEventRegisters(const WaveMemoryEvent &event) {
  for (uint32_t regId : event.registers) {
    for (int lane = 0; lane < waveSize; ++lane) {
      // Only clear lanes that were active during the instruction
      if ((event.mask >> lane) & 1) {

        // Assuming you have this helper, or use vgprMemoryEvents[regId *
        // waveSize + lane]
        auto &eventsForReg = getVgprMemoryEvents(regId, lane);

        auto it = std::find_if(eventsForReg.begin(), eventsForReg.end(),
                               [&](const RegisterMemoryEvent &entry) {
                                 return entry.eventId == event.eventId;
                               });

        if (it == eventsForReg.end()) {
          throw std::runtime_error(
              "Memory Tracker Inconsistency: Event ID " +
              std::to_string(event.eventId) + ", Register v" +
              std::to_string(regId) + ", Lane " + std::to_string(lane) +
              ": Event retired by s_waitcnt but not found on register");
        }

        eventsForReg.erase(it);
      }
    }
  }
}

// Helper 2: The Generic 'Reverse Scan & Retire' Logic
void Wave::resolveWaitCnt(
    int limit, std::function<bool(MemoryEventType)> isTargetType,
    std::function<void(const WaveMemoryEvent &)> extraCleanup) {
  int seen = 0;
  std::vector<int> indicesToRemove;

  // 1. Identify events to remove (Reverse Scan)
  for (int i = waveMemoryEvents.size() - 1; i >= 0; --i) {
    const auto &event = waveMemoryEvents[i];
    if (isTargetType(event.type)) {
      seen++;
      if (seen > limit) {
        indicesToRemove.push_back(i);
      }
    }
  }

  // 2. Process removals
  for (int idx : indicesToRemove) {
    const auto &event = waveMemoryEvents[idx];

    // A. Common Cleanup (Unlock Registers)
    retireEventRegisters(event);

    // B. Specific Cleanup (Unlock LDS, SGPRs, etc.)
    if (extraCleanup) {
      extraCleanup(event);
    }

    // Will completely empty bucket at s_barrier.
    if (event.type == MemoryEventType::VGPR_TO_LDS ||
        event.type == MemoryEventType::LDS_TO_VGPR) {
      waveCompleteMemoryEvents.push_back(event);
    }

    // C. Remove from Wave History
    waveMemoryEvents.erase(waveMemoryEvents.begin() + idx);
  }
}

void Wave::sWaitCntVmcnt(int vmcnt) {
  resolveWaitCnt(
      vmcnt,
      // Predicate: Count Global Memory events
      [](MemoryEventType type) {
        return type == MemoryEventType::GLOBAL_TO_VGPR ||
               type == MemoryEventType::VGPR_TO_GLOBAL;
      },
      // Cleanup: No extra resource (like LDS) needs clearing for VM... yet.
      nullptr);
}

void Wave::sWaitCntLgkmcnt(int lgkmcnt) {
  resolveWaitCnt(
      lgkmcnt,
      // Predicate: Count LDS/GDS/Constant/Export events
      [](MemoryEventType type) {
        return type == MemoryEventType::LDS_TO_VGPR ||
               type == MemoryEventType::VGPR_TO_LDS;
      },
      // Cleanup: Unlock the specific LDS bytes. BUT ONLY FOR THIS WAVE UNTIL
      // THERE IS AN S_BARRIER!!!!
      [&](const WaveMemoryEvent &event) {
        if (!event.ldsBytes.empty()) {
          LDS &lds = getLds();
          for (auto b : event.ldsBytes) {
            lds.markEventWaveComplete(b.byte, waveId, event.eventId);
          }
        }
      });
}

void Wave::flushWaveCompleteMemoryEvents() {
  LDS &lds = getLds();
  for (const auto &event : waveCompleteMemoryEvents) {
    for (auto b : event.ldsBytes) {
      lds.removeEvent(b.byte, waveId, event.eventId);
    }
  }
  waveCompleteMemoryEvents.clear();
}

// Check the MemoryEventType for each event registered to this register.
bool Wave::isOutstandingToVgpr(int lane, int reg) const {

  // return false;

  // TODO(newling) const function better here.
  auto index = reg * waveSize + lane;

  assert(index >= 0 && index < static_cast<int>(vgprMemoryEvents.size()) &&
         "VGPR index out of bounds");

  const auto &eventIds = vgprMemoryEvents[index];

  for (const auto &eventId : eventIds) {
    if (eventId.type == MemoryEventType::GLOBAL_TO_VGPR ||
        eventId.type == MemoryEventType::LDS_TO_VGPR) {
      return true;
    }
  }
  return false;
}

bool Wave::isOutstandingFromVgpr(int lane, int reg) const {

  // TODO(newling) this change seems important!!!!
  //  auto index = lane * vgprCount + reg;
  auto index = reg * waveSize + lane;

  const auto &eventIds = vgprMemoryEvents[index];

  for (auto &eventId : eventIds) {
    if (eventId.type == MemoryEventType::VGPR_TO_GLOBAL ||
        eventId.type == MemoryEventType::VGPR_TO_LDS) {
      return true;
    }
  }
  return false;
}

LDS &Wave::getLds() {
  assert(lds && "LDS not initialized");
  return *lds;
}

void CommonRegister::appendStr(std::ostream &os) const {
  char prefix = (type == Type::SGPR) ? 's' : (type == Type::VGPR) ? 'v' : '?';
  os << prefix << index;
}

std::string CommonRegister::str() const {
  std::ostringstream oss;
  appendStr(oss);
  return oss.str();
}

std::ostream &operator<<(std::ostream &os, const CommonRegister &reg) {
  reg.appendStr(os);
  return os;
}

// ---------------------------------- Parsing Helpers

namespace {} // namespace

CommonRegister Wave::getFirstRegister(std::string_view regStr) const {

  auto isDigitOrBracket = [](char c) {
    return (c >= '0' && c <= '9') || c == '[';
  };

  bool isAcc = false;

  CommonRegister::Type regType = CommonRegister::Type::UNKNOWN;
  assert(regStr.size() >= 2 && "Register string too short");
  if (regStr == "exec") {
    return {CommonRegister::Type::SGPR, execIndex};
  } else if (regStr == "vcc") {
    return {CommonRegister::Type::SGPR, vccIndex};
  } else if (regStr == "m0") {
    return {CommonRegister::Type::SGPR, m0Index};
  } else if (regStr[0] == 's' && isDigitOrBracket(regStr[1])) {
    regType = CommonRegister::Type::SGPR;
  } else if (regStr[0] == 'v' && isDigitOrBracket(regStr[1])) {
    regType = CommonRegister::Type::VGPR;
    // if it starts with acc, it is an accumulator register
  } else if (regStr.size() >= 3 && regStr.substr(0, 3) == "acc") {
    isAcc = true;
    regType = CommonRegister::Type::VGPR;
  }

  else {
    throw std::runtime_error("Unknown register type from: " +
                             std::string(regStr));
  }

  const char *numStart = nullptr;
  const char *numEnd = nullptr;
  auto openBracket = regStr.find('[');

  if (openBracket != std::string_view::npos) {
    auto colon = regStr.find(':', openBracket);
    assert(colon != std::string_view::npos);
    numStart = regStr.data() + openBracket + 1;
    numEnd = regStr.data() + colon;
  } else {
    auto firstDigit = regStr.find_first_of("0123456789");
    assert(firstDigit != std::string_view::npos && "flipit");
    numStart = regStr.data() + firstDigit;
    numEnd = regStr.data() + regStr.size();
  }

  int index = -1;
  std::from_chars(numStart, numEnd, index);

  if (isAcc) {
    index += this->agprOffset;
  }

  CommonRegister cr = {regType, index};
  return cr;
}

std::string Macro::str() const {
  std::ostringstream oss;
  oss << "Macro(startLine=" << startLine << ", endLine=" << endLine
      << ", argNames=[";
  for (size_t i = 0; i < argNames.size(); ++i) {
    oss << argNames[i];
    if (i + 1 < argNames.size()) {
      oss << ", ";
    }
  }
  oss << "])";
  return oss.str();
}

// VGPR or SGPR or Literal
template <typename T>
T Wave::getValue(const Operand<T> &operand, int lane) const {
  if (operand.isLiteral) {
    return operand.literalValue;
  }
  if constexpr (sizeof(T) == 8) {
    if (operand.reg.type == CommonRegister::Type::VGPR) {
      return std::bit_cast<T>(getVgpr64(operand.reg.index, lane));
    } else if (operand.reg.type == CommonRegister::Type::SGPR) {
      return std::bit_cast<T>(getSgpr64(operand.reg.index));
    } else {
      throw std::runtime_error("Unsupported register type for 64-bit operand");
    }
  } else if constexpr (sizeof(T) == 4) {
    if (operand.reg.type == CommonRegister::Type::VGPR) {
      return std::bit_cast<T>(getVgpr(operand.reg.index, lane));
    } else if (operand.reg.type == CommonRegister::Type::SGPR) {
      return std::bit_cast<T>(getSgpr(operand.reg.index));
    } else {
      throw std::runtime_error("Unsupported register type for 32-bit operand");
    }
  } else {
    throw std::runtime_error("Unsupported operand size");
  }
}

//  SGPR or literal
template <typename T>
T Wave::getSgprOrLiteralValue(const Operand<T> &operand) const {
  assert(operand.reg.type != CommonRegister::Type::VGPR && "expected lane");
  if (operand.isLiteral) {
    return operand.literalValue;
  }
  if constexpr (sizeof(T) == 8) {
    return std::bit_cast<T>(getSgpr64(operand.reg.index));
  } else if constexpr (sizeof(T) == 4) {
    return std::bit_cast<T>(getSgpr(operand.reg.index));
  } else {
    throw std::runtime_error("Unsupported operand size");
  }
}

// Parse a string, returning either a literal or a register operand.
template <typename T>
Operand<T> Wave::parseOperand(std::string_view token) const {
  Operand<T> op;
  assert(!token.empty() && "Empty operand token");
  bool looksLikeLiteral =
      token[0] == '-' || (token[0] >= '0' && token[0] <= '9');
  bool looksLikeLabel = (token.size() > 2 && token[0] == 'l' &&
                         token[1] >= 'a' && token[2] == 'b');
  if (looksLikeLiteral) {
    op.isLiteral = true;
    op.reg = {CommonRegister::Type::UNKNOWN, -1};
    if constexpr (std::is_floating_point_v<T>) {
      op.literalValue = getFloatFromView<T>(token);
    } else if constexpr (std::is_integral_v<T>) {
      op.literalValue = getIntFromView<T>(token);
    } else {
      throw std::runtime_error(
          "Unsupported literal operand type in parseOperand");
    }
  } else if (looksLikeLabel) {
    if (labels) {
      auto it = labels->find(std::string(token));
      if (it != labels->end()) {
        int programCount = it->second;
        op.isLiteral = true;
        op.literalValue = static_cast<T>(4 * (programCount - pc));
      } else {
        throw std::runtime_error("Unknown label operand: " +
                                 std::string(token));
      }
    } else {
      throw std::runtime_error("Label operand without label map: " +
                               std::string(token));
    }
  }

  else {
    op.isLiteral = false;
    op.reg = getFirstRegister(token);
  }
  return op;
}

template int32_t Wave::getValue<int32_t>(const Operand<int32_t> &, int) const;
template uint32_t Wave::getValue<uint32_t>(const Operand<uint32_t> &,
                                           int) const;
template float Wave::getValue<float>(const Operand<float> &, int) const;
template int64_t Wave::getValue<int64_t>(const Operand<int64_t> &, int) const;
template uint64_t Wave::getValue<uint64_t>(const Operand<uint64_t> &,
                                           int) const;
template double Wave::getValue<double>(const Operand<double> &, int) const;

// Explicit instantiations — parseOperand
template Operand<float> Wave::parseOperand<float>(std::string_view) const;
template Operand<double> Wave::parseOperand<double>(std::string_view) const;
template Operand<int32_t> Wave::parseOperand<int32_t>(std::string_view) const;
template Operand<uint32_t> Wave::parseOperand<uint32_t>(std::string_view) const;
template Operand<short> Wave::parseOperand<short>(std::string_view) const;
template Operand<uint64_t> Wave::parseOperand<uint64_t>(std::string_view) const;

// Explicit instantiations — getSgprOrLiteralValue
template short Wave::getSgprOrLiteralValue<short>(const Operand<short> &) const;
template int32_t
Wave::getSgprOrLiteralValue<int32_t>(const Operand<int32_t> &) const;
template uint32_t
Wave::getSgprOrLiteralValue<uint32_t>(const Operand<uint32_t> &) const;
template uint64_t
Wave::getSgprOrLiteralValue<uint64_t>(const Operand<uint64_t> &) const;

} // namespace raceemulator
