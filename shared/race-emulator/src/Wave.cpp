// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Wave.h"
#include "race-emulator/Instruction.h"
#include "race-emulator/RaceDetector.h"
#include "race-emulator/Util.h"
#include "race-emulator/Workgroup.h"
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

namespace {

const constexpr int vccIndex = -2;
const constexpr int sccIndex = -3;
const constexpr int execIndex = -5;
const constexpr int m0Index = -6;
const constexpr int nullIndex = -7;

const constexpr int nExtraSgrs = -nullIndex + 1;

using LabelMap = std::map<std::string, int>;

} // namespace

Wave::Wave(int vgprCount, int agprOffset, int sgprCount, WaveSize waveSize,
           WaveId waveId, Workgroup &workgroup,
           const std::map<std::string, int> *labels,
           const std::map<std::string, Macro> *macros)
    : agprOffset(agprOffset), sgprCount(sgprCount), waveSize(waveSize),
      dsPreserve(waveSize == 32), waveId(waveId), workgroup(&workgroup),
      labels(labels), macros(macros) {

  assert(labels != nullptr && "Labels map cannot be null");
  assert(macros != nullptr && "Macros map cannot be null");

  int avgprCount = vgprCount;

  vgprs.resize(avgprCount * waveSize, 0);
  sgprs.resize(sgprCount + nExtraSgrs, 0);

  // Race state pointer is set by Workgroup after construction via
  // waves.back().raceState = ... (Workgroup is a friend).

  static_assert(sizeof(0ULL) * 8 >= 64, "uint64_t must be at least 64 bits");

  uint64_t allActive = (waveSize == 64) ? ~0ULL : ((1ULL << waveSize) - 1);
  setExecU64(allActive);
}

Wave::~Wave() = default;
Wave::Wave(Wave &&other) noexcept = default;
Wave &Wave::operator=(Wave &&other) noexcept = default;

void Wave::flushWaveCompleteMemoryEvents() {
  if (raceState) {
    raceState->flushWaveCompleteMemoryEvents();
  }
}

// --- VGPR access with race checking ---

uint8_t Wave::getVgprByte(int reg, int lane, int byteIdx) const {
  assert(byteIdx >= 0 && byteIdx < 4);
  int32_t index = reg * waveSize + lane;
  assert(index < static_cast<int64_t>(vgprs.size()));
  if (raceState) {
    uint8_t byteMask = static_cast<uint8_t>(1 << byteIdx);
    raceState->checkVgprRead(reg, lane, byteMask);
  }
  return static_cast<uint8_t>((vgprs[index] >> (byteIdx * 8)) & 0xFF);
}

uint16_t Wave::getHalfVgpr(int reg, int lane, bool hi) const {
  int32_t index = reg * waveSize + lane;
  assert(index < static_cast<int64_t>(vgprs.size()));
  if (raceState) {
    uint8_t byteMask = hi ? 0xC : 0x3;
    raceState->checkVgprRead(reg, lane, byteMask);
  }
  return hi ? static_cast<uint16_t>(vgprs[index] >> 16)
            : static_cast<uint16_t>(vgprs[index]);
}

void Wave::setVgprHalf(int reg, int lane, bool hi, uint16_t value) {
  assert((getExecU64() & (1ULL << lane)) != 0 &&
         "Writing to VGPR of inactive lane");
  auto index = reg * waveSize + lane;
  assert(index < static_cast<int64_t>(vgprs.size()));
  if (hi) {
    vgprs[index] = (vgprs[index] & 0x0000FFFFu) | (uint32_t(value) << 16);
  } else {
    vgprs[index] = (vgprs[index] & 0xFFFF0000u) | uint32_t(value);
  }
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
  if (id == nullIndex) {
    return;
  }
  id = id < 0 ? id + sgprs.size() : id;
  assert(id < static_cast<int64_t>(sgprs.size()));
  sgprs[id] = value;
}

void Wave::setSgpr64(int id, uint64_t value) {
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
  uint64_t low = getVgpr(id, lane);
  uint64_t high = getVgpr(id + 1, lane);
  return (high << 32) | low;
}

void Wave::setVgpr64(int id, int lane, uint64_t value) {
  setVgpr(id, lane, static_cast<uint32_t>(value));
  setVgpr(id + 1, lane, static_cast<uint32_t>(value >> 32));
}

std::function<int()>
Wave::compileLine(const std::string &line,
                  const std::map<std::string, Macro> &macros) {

  auto currentPc = pc;
  auto nullOpt = [currentPc]() -> int { return currentPc + 1; };

  if (line.find_first_not_of(" \t\r\n'") == std::string::npos) {
    return nullOpt;
  }

  if (labels) {
    const auto &labelMap = *labels;
    auto firstNonSpace = line.find_first_not_of(" \t");
    auto firstColon = line.find(':', firstNonSpace);
    if (firstColon != std::string::npos && firstColon > firstNonSpace) {
      auto labelName = line.substr(firstNonSpace, firstColon);
      auto foundLabel = labelMap.find(labelName);
      if (foundLabel != labelMap.end()) {
        return nullOpt;
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
    int mendLine = found->second.getEndLine();
    return [mendLine]() -> int { return mendLine + 1; };
  }

  if (partitioned[0] == ".endm") {
    return [this]() -> int {
      macroArguments.clear();
      auto currentPc = macroReturnPc;
      macroReturnPc = -1;
      return currentPc;
    };
  }

  if (partitioned[0] == ".set") {
    return nullOpt;
  }

  if (partitioned[0] == ".align") {
    return nullOpt;
  }

  const auto &instructions = getInstructions();

  std::string name = std::string(partitioned[0]);
  auto e32Pos = name.find("_e32");
  if (e32Pos != std::string::npos) {
    name = name.substr(0, e32Pos);
  }
  auto e64Pos = name.find("_e64");
  if (e64Pos != std::string::npos) {
    name = name.substr(0, e64Pos);
  }
  auto found = instructions.find(name);

  if (found != instructions.end()) {
    auto exec = found->second->getExecutor(*this, line);
    return exec;
  }

  return nullptr;
}

void Wave::tryExecute(const std::string &line_, bool enableLineCaching) {

  std::string macroReducedLine;
  if (!macroArguments.empty()) {
    macroReducedLine = getSymbolReducedLine(line_, macroArguments);
  }
  const std::string &line = macroArguments.empty() ? line_ : macroReducedLine;

  auto nxt = [&]() {
    if (pc < static_cast<int>(instructionCache.size()) &&
        instructionCache[pc] != nullptr) {
      return instructionCache[pc]();
    }

    std::function<int()> func;

    auto dualPos = line.find("::");
    if (dualPos != std::string::npos) {
      auto compileOrThrow = [&](const std::string &s) {
        auto f = compileLine(s, *macros);
        if (!f) {
          throw std::runtime_error("Unimplemented instruction: " + s);
        }
        return f;
      };
      auto func1 = compileOrThrow(line.substr(0, dualPos));
      auto func2 = compileOrThrow(line.substr(dualPos + 2));
      int currentPc = pc;
      func = [func1, func2, currentPc]() -> int {
        func1();
        func2();
        return currentPc + 1;
      };
    }
    if (!func) {
      func = compileLine(line, *macros);
    }
    if (!func) {
      auto partitioned = getPartitioned(line);
      auto iter = macros->find(std::string(partitioned[0]));
      if (iter != macros->end()) {
        auto macroStart = iter->second.getStartLine();
        const auto &argNames = iter->second.getArgNames();
        macroReturnPc = pc + 1;
        for (size_t i = 0; i < argNames.size(); ++i) {
          uint32_t value = 0;
          if (!parseNumber(partitioned[i + 1], value)) {
            throw std::runtime_error("Error parsing macro argument: " +
                                     std::string(partitioned[i + 1]));
          }
          macroArguments.insert({"\\" + argNames[i], value});
        }
        return macroStart + 1;
      }
    }
    if (!func) {
      throw std::runtime_error("Unimplemented instruction: " + line);
    }
    if (pc >= static_cast<int>(instructionCache.size())) {
      instructionCache.resize(pc + 16, nullptr);
    }
    if (enableLineCaching && macroArguments.empty()) {
      instructionCache[pc] = func;
    }
    return func();
  }();

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

uint32_t Wave::getVccU32() const { return getSgpr(vccIndex); }
uint64_t Wave::getVccU64() const { return getSgpr64(vccIndex); }
uint32_t Wave::getExecU32() const { return getSgpr(execIndex); }
uint64_t Wave::getExecU64() const { return getSgpr64(execIndex); }

void Wave::setVccU32(uint32_t value) { setSgpr(vccIndex, value); }
void Wave::setVccU64(uint64_t value) { setSgpr64(vccIndex, value); }
void Wave::setExecU32(uint32_t value) { setSgpr(execIndex, value); }
void Wave::setExecU64(uint64_t value) { setSgpr64(execIndex, value); }

void Wave::setM0(uint32_t value) { setSgpr(m0Index, value); }
uint32_t Wave::getM0() const { return getSgpr(m0Index); }

Workgroup &Wave::getWorkgroup() { return *workgroup; }

uint32_t Wave::getVgpr(int reg, int lane) const {
  int32_t index = reg * waveSize + lane;
  assert(index < static_cast<int64_t>(vgprs.size()));
  if (raceState) {
    raceState->checkVgprRead(reg, lane, 0xF);
  }
  return vgprs[index];
}

void Wave::getVgprs(int reg, uint32_t *out) const {
  assert(reg >= 0);
  assert((reg + 1) * waveSize <= static_cast<int>(vgprs.size()));
  if (raceState) {
    raceState->checkVgprReadAllLanes(reg);
  }
  std::memcpy(out, vgprs.data() + reg * waveSize, waveSize * sizeof(uint32_t));
}

template <typename T> T Wave::readLds(int addr, int lane) const {
  return workgroup->readLds<T>(addr, waveId, lane);
}

template <typename T>
void Wave::readLds(int addr, int lane, T *out, int count) const {
  workgroup->readLds<T>(addr, waveId, lane, out, count);
}

template <typename T> void Wave::writeLds(int addr, int lane, T value) {
  workgroup->writeLds<T>(addr, waveId, lane, value);
}

// Explicit instantiations for all LDS element types used by instructions.
template uint8_t Wave::readLds<uint8_t>(int, int) const;
template int8_t Wave::readLds<int8_t>(int, int) const;
template uint16_t Wave::readLds<uint16_t>(int, int) const;
template int16_t Wave::readLds<int16_t>(int, int) const;
template uint32_t Wave::readLds<uint32_t>(int, int) const;

template void Wave::readLds<uint8_t>(int, int, uint8_t *, int) const;
template void Wave::readLds<int8_t>(int, int, int8_t *, int) const;
template void Wave::readLds<uint16_t>(int, int, uint16_t *, int) const;
template void Wave::readLds<int16_t>(int, int, int16_t *, int) const;
template void Wave::readLds<uint32_t>(int, int, uint32_t *, int) const;

template void Wave::writeLds<uint8_t>(int, int, uint8_t);
template void Wave::writeLds<uint16_t>(int, int, uint16_t);
template void Wave::writeLds<uint32_t>(int, int, uint32_t);

bool Wave::isCompleteEmulation() const {
  return workgroup->isCompleteEmulation();
}

Profiler::ScopedStopwatch Wave::profileScope(std::string_view key) {
  auto *p = workgroup->getProfiler();
  return p ? p->scopedStopwatch(key) : Profiler::ScopedStopwatch{};
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
  if (regStr == "exec" || regStr == "exec_lo") {
    return {CommonRegister::Type::SGPR, execIndex};
  } else if (regStr == "exec_hi") {
    return {CommonRegister::Type::SGPR, execIndex + 1};
  } else if (regStr == "vcc" || regStr == "vcc_lo") {
    return {CommonRegister::Type::SGPR, vccIndex};
  } else if (regStr == "vcc_hi") {
    return {CommonRegister::Type::SGPR, vccIndex + 1};
  } else if (regStr == "null") {
    return {CommonRegister::Type::SGPR, nullIndex};
  } else if (regStr == "m0") {
    return {CommonRegister::Type::SGPR, m0Index};
  } else if (regStr[0] == 's' && isDigitOrBracket(regStr[1])) {
    regType = CommonRegister::Type::SGPR;
  } else if (regStr[0] == 'v' && isDigitOrBracket(regStr[1])) {
    regType = CommonRegister::Type::VGPR;
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

// Explicit instantiations -- parseOperand
template Operand<float> Wave::parseOperand<float>(std::string_view) const;
template Operand<double> Wave::parseOperand<double>(std::string_view) const;
template Operand<int32_t> Wave::parseOperand<int32_t>(std::string_view) const;
template Operand<uint32_t> Wave::parseOperand<uint32_t>(std::string_view) const;
template Operand<short> Wave::parseOperand<short>(std::string_view) const;
template Operand<uint64_t> Wave::parseOperand<uint64_t>(std::string_view) const;

// Explicit instantiations -- getSgprOrLiteralValue
template short Wave::getSgprOrLiteralValue<short>(const Operand<short> &) const;
template int32_t
Wave::getSgprOrLiteralValue<int32_t>(const Operand<int32_t> &) const;
template uint32_t
Wave::getSgprOrLiteralValue<uint32_t>(const Operand<uint32_t> &) const;
template uint64_t
Wave::getSgprOrLiteralValue<uint64_t>(const Operand<uint64_t> &) const;

} // namespace raceemulator
