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

enum class ParserState { Root, Amdhsa, Kernels, Args, Macro };

// ParsedLine represents a single line of assembly code after parsing.
struct ParsedLine {
  ParsedLine(std::string_view line, int lineNumber,
             ParserState precedingParserState,
             const std::map<std::string, uint32_t> &symbolTable);

  // The original line of assembly code.
  std::string originalLine;

  // The line with comments removed, and symbols replaced.
  std::string commentFreeLine;

  // True if the comment-free line contains only whitespace.
  bool isEmptyLine{false};

  // The line number in the original assembly code.
  // Assembly lines start at index 0.
  int lineNumber;

  // The state of the preceding token.
  ParserState precedingParserState;

  // The indentation level of the line. More specifically, the index of the line
  // of the first non-whitespace character.
  int indent{-1};

  // True if the line starts with '- ', ignoring leading whitespace.
  bool isListItem{false};

  // True if the parser has determined that this line is a label in the
  // assembly.
  bool isLabel{false};

  // True if the parser has determined that this line is a key-value pair.
  bool isKeyValue{false};
  std::string key{"none"};
  std::string value{"none"};

  // String representation of the ParsedLine for debugging.
  std::string str() const;
};

// Kernel corresponds to the yaml entry in the amdgpu metadata of an assembly
// file, such as
//     .args:
//      - .name:          My important argument
//        .size:          4
//        .offset:        0
//        .value_kind:    by_value
//        .value_type:    u32
struct KernelArg {
  int size = 0;
  int offset = 0;
  std::string valueKind;
  std::string valueType;
  std::string addressSpace;
  std::string name;
};

// Scalar registers are initialized with certain values, such as a pointer to
// the kernel argument segment in global memory, and workgroup ID. These structs
// represent how the registers are allocated, as determined by parsing the .s
// file.
struct RegisterMapping {
  int start_register;
  int count;
};
struct AllocationResult {
  std::map<std::string, RegisterMapping> registers;
};

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

  // Kernarg preload: hardware preloads this many dwords from the kernarg
  // segment into SGPRs starting right after the kernarg segment pointer.
  int kernargPreloadLength = 0;
  int kernargPreloadOffset = 0;

  // labels to the lines they are on
  std::map<std::string, int> labels;

  // macro names, to the lines with '.macro' and '.endm'
  std::map<std::string, Macro> macros;

  void appendStr(std::ostream &os) const;
  std::string str() const;
  void appendTokensStr(std::ostream &os) const;
};

} // namespace raceemulator
