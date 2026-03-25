// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Parsing.h"
#include "race-emulator/Util.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

namespace {

void processInRoot(ParsedLine token, std::map<std::string, int> &labels,
                   std::map<std::string, uint32_t> &symbolTable) {
  // 1. handle labels
  if (token.isLabel) {
    labels.insert({token.key, token.lineNumber});
    return;
  }

  // 2. handle .set directives
  if (token.originalLine.find(".set") == 0) {
    std::string_view lineView = token.originalLine;
    std::string_view content = lineView.substr(4); // skip ".set"

    size_t commaPos = content.find(',');
    if (commaPos == std::string::npos) {
      throw std::runtime_error("parser error: .set directive missing comma");
    }

    // extract key
    std::string key(trim(content.substr(0, commaPos)));

    // extract and evaluate value (supports chained math "A + B + 4")
    std::string_view expr = trim(content.substr(commaPos + 1));
    std::string resultStr = maybeEvaluateExpression(expr, symbolTable);

    // store as integer in map
    uint32_t finalVal;
    if (!parseNumber(resultStr, finalVal)) {
      throw std::runtime_error(
          "internal error: evaluator returned non-number string");
    }
    symbolTable[key] = finalVal;
  }
}

std::string parserStateStr(ParserState state) {
  switch (state) {
  case ParserState::Root:
    return "Root";
  case ParserState::Amdhsa:
    return "Amdhsa";
  case ParserState::Kernels:
    return "Kernels";
  case ParserState::Args:
    return "Args";
  case ParserState::Macro:
    return "Macro";
  default:
    return "Unknown";
  }
}

// determine how sgprs are initialized.
class KernelStateParser {
public:
  static AllocationResult
  Parse(const std::vector<std::pair<std::string, int>> &metadata) {
    std::map<std::string, int> meta_map;
    for (const auto &item : metadata) {
      meta_map[item.first] = item.second;
    }

    AllocationResult result;
    int current_sgpr = 0;

    // 1. identify user sgpr count
    // this is the explicit boundary between user and system sgprs.
    int user_sgpr_count = 0;
    if (meta_map.find(".amdhsa_user_sgpr_count") != meta_map.end()) {
      user_sgpr_count = meta_map[".amdhsa_user_sgpr_count"];
    }

    // 2. allocate named user sgprs
    // these start at s0 and grow upwards.

    // kernarg segment ptr (s0, s1)
    if (meta_map[".amdhsa_user_sgpr_kernarg_segment_ptr"] != 0) {
      result.registers[".amdhsa_user_sgpr_kernarg_segment_ptr"] = {current_sgpr,
                                                                   2};
      current_sgpr += 2;
    }

    // ... (add other user sgprs here if you support them later) ...

    // 3. enforce the user sgpr boundary
    if (current_sgpr < user_sgpr_count) {
      current_sgpr = user_sgpr_count;
    }

    // 4. allocate system sgprs (workgroup ids)
    const std::vector<std::string> workgroup_ids = {
        ".amdhsa_system_sgpr_workgroup_id_x",
        ".amdhsa_system_sgpr_workgroup_id_y",
        ".amdhsa_system_sgpr_workgroup_id_z"};

    for (const auto &key : workgroup_ids) {
      if (meta_map.count(key) && meta_map[key] != 0) {
        result.registers[key] = {current_sgpr, 1};
        current_sgpr += 1;
      }
    }

    // enforce currently unsupported sgprs are 0
    const std::vector<std::string> unsupported_sgprs = {
        ".amdhsa_user_sgpr_dispatch_ptr",
        ".amdhsa_user_sgpr_queue_ptr",
        ".amdhsa_user_sgpr_dispatch_id",
        ".amdhsa_user_sgpr_flat_scratch_init",
        ".amdhsa_user_sgpr_private_segment_size",
        ".amdhsa_system_sgpr_workgroup_info",
        ".amdhsa_system_sgpr_private_segment_wave_byte_offset"};

    for (const auto &key : unsupported_sgprs) {
      auto found = meta_map.find(key);
      if (found != meta_map.end() && found->second != 0) {
        throw std::runtime_error(
            "parser error: currently unsupported metadata field " + key +
            " is set to " + std::to_string(meta_map[key]) + " (expected 0)");
      }
    }

    return result;
  }
};

} // namespace

std::string ParsedLine::str() const {
  std::ostringstream oss;
  oss << "ParsedLine(indent=";
  oss << indent;
  oss << ", isListItem=";
  oss << isListItem;
  oss << ", isLabel=";
  oss << isLabel;
  oss << ", key='";
  oss << key;
  oss << "'";
  oss << ", value='";
  oss << value << "')";
  return oss.str();
}

ParsedLine::ParsedLine(std::string_view inputLine, int n,
                       ParserState precedingParserState,
                       const std::map<std::string, uint32_t> &symbolTable)
    : originalLine(inputLine), lineNumber(n),
      precedingParserState(precedingParserState) {

  // strip comments
  std::string line = std::string(inputLine);
  size_t comment_pos = line.find(';');
  if (comment_pos != std::string::npos) {
    line = line.substr(0, comment_pos);
  }
  comment_pos = line.find("//");
  if (comment_pos != std::string::npos) {
    line = line.substr(0, comment_pos);
  }
  comment_pos = line.find("/*");
  if (comment_pos != std::string::npos) {
    line = line.substr(0, comment_pos);
  }

  commentFreeLine = line;
  if (precedingParserState == ParserState::Root ||
      precedingParserState == ParserState::Macro) {
    // replace symbols in the comment free line
    commentFreeLine = getSymbolReducedLine(commentFreeLine, symbolTable);
  }

  // calculate and remove indentation
  size_t first_char = line.find_first_not_of(" \t");
  if (first_char == std::string::npos) {
    isEmptyLine = true;
    return;
  }
  indent = static_cast<int>(first_char);

  // check if it is a list item
  line = trim(line);
  if (line.find("- ", 0) == 0) {
    isListItem = true;
    line = line.substr(2);
  } else {
    isListItem = false;
  }

  // check if it is a label
  size_t colon_pos = line.find(':');
  size_t space_pos = line.find_first_of(" \t\n");
  bool isSpace = (space_pos != std::string::npos);
  bool isColon = (colon_pos != std::string::npos);

  if (indent == 0 && isColon && !isSpace) {
    isLabel = true;
    key = trim(line.substr(0, colon_pos));
  }

  // check if key-value pair
  if (!isLabel && isColon && (!isSpace || space_pos == colon_pos + 1)) {
    isKeyValue = true;
    key = trim(line.substr(0, colon_pos));
    value = trim(line.substr(colon_pos + 1));
  }

  // amdhsa specific parsing
  if (precedingParserState == ParserState::Amdhsa) {
    size_t sp = line.find_first_of(" \t");
    if (sp == std::string::npos) {
      return;
    }
    isKeyValue = true;
    key = trim(line.substr(0, sp));
    value = trim(line.substr(sp + 1));
  }
}

ParsedAsm::ParsedAsm(std::string_view a) : assembly(a) {
  std::vector<std::string> assemblyLines;
  {
    std::istringstream stream(assembly);
    std::string currentLine;
    while (std::getline(stream, currentLine)) {
      assemblyLines.push_back(currentLine);
    }
  }

  int argsItemDepth = -1;
  ParserState state = ParserState::Root;

  auto updateParserState = [&](ParsedLine token) {
    if (state == ParserState::Root &&
        token.originalLine.find(".amdhsa_kernel") != std::string::npos) {
      state = ParserState::Amdhsa;
    }

    if (state == ParserState::Amdhsa &&
        token.originalLine.find(".end_amdhsa_kernel") != std::string::npos) {
      state = ParserState::Root;
    }

    if (state == ParserState::Root && token.key == "amdhsa.kernels") {
      state = ParserState::Kernels;
    }

    if (state == ParserState::Kernels && token.key == ".args") {
      state = ParserState::Args;
      argsItemDepth = -1; // will be set on first actual arg list item
    }

    if (state == ParserState::Args && !token.isEmptyLine && argsItemDepth > 0 &&
        token.indent < argsItemDepth) {
      state = ParserState::Kernels;
    }

    if (state == ParserState::Kernels &&
        token.originalLine.find(".end_amdgpu_metadata") != std::string::npos) {
      state = ParserState::Root;
    }

    if (state == ParserState::Root &&
        token.originalLine.find(".macro", 0) == 0) {
      state = ParserState::Macro;
    }

    if (state == ParserState::Macro) {
      auto trimmedLine = trim(token.commentFreeLine);
      if (trimmedLine == ".endm") {
        state = ParserState::Root;
      }
    }
  };

  auto processInKernel = [&](ParsedLine token) {
    if (token.key == ".name") {
      assert(name.empty() && "only support one kernel currently");
      name = token.value;
    } else if (token.key == ".kernarg_segment_size") {
      kernargSegmentSize = getIntFromView<int>(token.value);
    } else if (token.key == ".wavefront_size") {
      wavefrontSize = getIntFromView<int>(token.value);
    }
  };

  auto processInArgs = [&](ParsedLine token) {
    // Skip the ".args:" line itself — it enters Args state but isn't an arg.
    if (token.key == ".args") {
      return;
    }
    if (token.isListItem) {
      args.push_back(KernelArg{});
      if (argsItemDepth < 0) {
        argsItemDepth = token.indent;
      }
    }
    if (args.empty()) {
      return;
    }

    if (token.key == ".size") {
      args.back().size = getIntFromView<int>(token.value);
    } else if (token.key == ".offset") {
      args.back().offset = getIntFromView<int>(token.value);
    } else if (token.key == ".value_kind") {
      args.back().valueKind = token.value;
    } else if (token.key == ".value_type") {
      args.back().valueType = token.value;
    } else if (token.key == ".address_space") {
      args.back().addressSpace = token.value;
    } else if (token.key == ".name") {
      args.back().name = token.value;
    }
  };

  auto processInAmdhsa = [&](ParsedLine token) {
    if (token.originalLine.find(".amdhsa_kernel") != std::string::npos) {
      return;
    }
    if (token.isEmptyLine) {
      return;
    }
    amdhsa.push_back({token.key, std::stoi(token.value)});
  };

  auto processInMacro = [&](ParsedLine token) {
    if (token.originalLine.find(".macro") == 0) {
      std::istringstream iss(token.commentFreeLine);
      std::string directive, macroName;
      iss >> directive >> macroName;

      int startLine = token.lineNumber;
      int endLine = -1;
      for (size_t i = token.lineNumber + 1; i < assemblyLines.size(); ++i) {
        if (assemblyLines[i].find(".endm") == 0) {
          endLine = i;
          break;
        }
      }
      assert(endLine != -1 &&
             "parser error: missing .endm for macro definition");

      std::vector<std::string> argumentNames;
      size_t currentPos = token.commentFreeLine.find(macroName);
      assert(currentPos != std::string::npos);
      currentPos += macroName.length();

      while (currentPos != std::string::npos) {
        size_t start =
            token.commentFreeLine.find_first_not_of(" \t,", currentPos);
        if (start == std::string::npos) {
          break;
        }

        size_t colon = token.commentFreeLine.find(':', start);
        assert(colon != std::string::npos &&
               "malformed macro argument, missing ':'");

        argumentNames.push_back(
            token.commentFreeLine.substr(start, colon - start));

        size_t comma = token.commentFreeLine.find(',', colon);
        currentPos =
            (comma == std::string::npos) ? std::string::npos : comma + 1;
      }

      Macro macro(startLine, endLine, argumentNames);
      macros.insert({macroName, macro});
    }
  };

  auto process = [&](ParsedLine token,
                     std::map<std::string, uint32_t> &symbolTable) {
    if (state == ParserState::Root) {
      processInRoot(token, labels, symbolTable);
    }
    if (state == ParserState::Amdhsa) {
      processInAmdhsa(token);
    }
    if (state == ParserState::Kernels) {
      processInKernel(token);
    } else if (state == ParserState::Args) {
      processInArgs(token);
    } else if (state == ParserState::Macro) {
      processInMacro(token);
    }
  };

  tokens.clear();
  std::map<std::string, uint32_t> symbolTable;
  symbolTable["UNDEF"] = 0xFFFFFFFF;
  for (unsigned i = 0; i < assemblyLines.size(); ++i) {
    const auto &line = assemblyLines[i];
    ParsedLine token(line, i, state, symbolTable);
    tokens.push_back(token);
    updateParserState(token);
    process(token, symbolTable);

    // Extract .amdgcn_target value (outside processInRoot since it needs
    // access to ParsedAsm fields).
    if (state == ParserState::Root) {
      auto pos = line.find(".amdgcn_target");
      if (pos != std::string::npos) {
        auto q1 = line.find('"', pos);
        auto q2 = line.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos) {
          target = line.substr(q1 + 1, q2 - q1 - 1);
        }
      }
    }
  }

  initialRegisterAllocation = KernelStateParser::Parse(amdhsa);

  // Extract kernarg preload settings from amdhsa metadata.
  for (const auto &[key, val] : amdhsa) {
    if (key == ".amdhsa_user_sgpr_kernarg_preload_length") {
      kernargPreloadLength = val;
    } else if (key == ".amdhsa_user_sgpr_kernarg_preload_offset") {
      kernargPreloadOffset = val;
    }
  }
}

void ParsedAsm::appendStr(std::ostream &os) const {
  os << "Kernel Name: " << name << "\n";
  os << "Kernarg Segment Size: " << kernargSegmentSize << "\n";
  os << "Wavefront Size: " << wavefrontSize << "\n";
  os << "Kernel Arguments:\n";
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    os << "  Arg " << i << ": name='" << arg.name << "', size=" << arg.size
       << ", offset=" << arg.offset << ", valueKind='" << arg.valueKind
       << "', valueType='" << arg.valueType << "', addressSpace='"
       << arg.addressSpace << "'\n";
  }
  os << "Labels:\n";
  std::vector<std::pair<int, std::string>> sortedLabels;
  for (const auto &label : labels) {
    sortedLabels.push_back({label.second, label.first});
  }
  std::sort(sortedLabels.begin(), sortedLabels.end());
  for (const auto &label : sortedLabels) {
    os << "  " << label.second << ": line " << label.first << "\n";
  }

  // TODO(newling) print macros.

  os << "AMDHSA Metadata:\n";
  for (const auto &amdhsa : amdhsa) {
    os << "  " << amdhsa.first << " = " << amdhsa.second << "\n";
  }
}

void ParsedAsm::appendTokensStr(std::ostream &os) const {
  for (const auto &token : tokens) {
    os << "[KV=]" << token.isKeyValue << " [LI]=" << token.isListItem
       << " [LB]=" << token.isLabel << " "
       << parserStateStr(token.precedingParserState) << " "
       << token.commentFreeLine << "\n";
  }
}

std::string ParsedAsm::str() const {
  std::ostringstream oss;
  appendStr(oss);
  return oss.str();
}

} // namespace raceemulator
