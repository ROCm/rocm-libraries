// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Wave.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

/// Tracks which section of the assembly file the parser is currently in.
enum class ParserState { Root, Amdhsa, Kernels, Args, Macro };

/// A single line of assembly after parsing: comments stripped, symbols
/// resolved, labels and key-value pairs identified.
struct ParsedLine {
  ParsedLine(std::string_view line, int lineNumber,
             ParserState precedingParserState,
             const std::map<std::string, uint32_t> &symbolTable);

  std::string originalLine;
  /// Comments removed and symbols replaced.
  std::string commentFreeLine;
  bool isEmptyLine{false};
  /// 0-based line number in the original assembly.
  int lineNumber;
  ParserState precedingParserState;
  /// Column index of the first non-whitespace character.
  int indent{-1};
  bool isListItem{false};
  bool isLabel{false};
  bool isKeyValue{false};
  std::string key{"none"};
  std::string value{"none"};

  std::string str() const;
};

/// A kernel argument descriptor from the AMDGPU metadata YAML section.
struct KernelArg {
  int size = 0;
  int offset = 0;
  std::string valueKind;
  std::string valueType;
  std::string addressSpace;
  std::string name;
};

/// Pre-allocated scalar register ranges (e.g., kernarg pointer, workgroup ID),
/// determined by parsing the assembly metadata.
struct RegisterMapping {
  int start_register;
  int count;
};
struct AllocationResult {
  std::map<std::string, RegisterMapping> registers;
};

/// Top-level result of parsing an AMD GPU assembly file: kernel metadata,
/// parsed instruction lines, labels, and macros.
struct ParsedAsm {
  ParsedAsm(std::string_view assembly);

  std::string name;
  std::string assembly;
  std::string target;
  int kernargSegmentSize = 0;
  int wavefrontSize = 0;
  std::vector<KernelArg> args;
  std::vector<ParsedLine> tokens;
  AllocationResult initialRegisterAllocation;
  std::vector<std::pair<std::string, int>> amdhsa;

  /// Kernarg preload: hardware preloads this many dwords from the kernarg
  /// segment into SGPRs starting right after the kernarg segment pointer.
  int kernargPreloadLength = 0;
  int kernargPreloadOffset = 0;

  /// Label name -> line index.
  std::map<std::string, int> labels;

  /// Macro name -> line range and argument names.
  std::map<std::string, Macro> macros;

  void appendStr(std::ostream &os) const;
  std::string str() const;
  void appendTokensStr(std::ostream &os) const;
};

} // namespace raceemulator
