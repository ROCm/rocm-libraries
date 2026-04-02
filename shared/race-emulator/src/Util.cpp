// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Util.h"
#include <cassert>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

// split line into tokens based on whitespace and commas.
std::vector<std::string_view> getPartitioned(std::string_view line) {
  std::vector<std::string_view> partitioned;
  uint64_t start = 0;
  uint64_t end = 0;
  while (true) {
    start = line.find_first_not_of(" \t,\n", end);
    if (start == std::string::npos) {
      return partitioned;
    }
    end = line.find_first_of(" \t,\n", start);
    partitioned.push_back(line.substr(start, end - start));
  }
}

std::string_view trim(std::string_view sv) {
  const auto strBegin = sv.find_first_not_of(" \t");
  if (strBegin == std::string::npos) {
    return std::string_view{};
  }
  const auto strEnd = sv.find_last_not_of(" \t");
  return sv.substr(strBegin, strEnd - strBegin + 1);
}

// Parse number from string_view (supports hex 0x prefix).
bool parseNumber(std::string_view sv, uint32_t &outVal) {
  if (sv.empty()) {
    return false;
  }

  int base = 10;
  const char *start = sv.data();
  const char *end = sv.data() + sv.size();

  // Detect "0x" or "0X" prefix
  if (sv.size() >= 2 && sv[0] == '0' && (sv[1] == 'x' || sv[1] == 'X')) {
    base = 16;
    start += 2;
  }

  auto [ptr, ec] = std::from_chars(start, end, outVal, base);
  return (ec == std::errc() && ptr == end);
}

// Resolve a single token as a number or symbol table lookup.
static std::optional<uint32_t>
resolveAtom(std::string_view rawToken,
            const std::map<std::string, uint32_t> &table) {
  std::string_view token = trim(rawToken);
  if (token.empty()) {
    return std::nullopt;
  }

  uint32_t val = 0;
  // 1. Try Number
  if (parseNumber(token, val)) {
    return val;
  }
  // 2. Try Symbol
  std::string key(token); // Map requires std::string
  auto it = table.find(key);
  if (it != table.end()) {
    return it->second;
  }

  return std::nullopt;
}

// Evaluate a product expression ("A * B * C").
static std::optional<uint32_t>
evaluateProduct(std::string_view productExpr,
                const std::map<std::string, uint32_t> &table) {
  uint32_t product = 1;
  size_t start = 0, end = 0;
  bool empty = true;

  // Split by '*'
  // We loop effectively: for each factor in "Factor * Factor * ..."
  do {
    end = productExpr.find('*', start);
    std::string_view factor = productExpr.substr(start, end - start);

    // Resolve the individual factor
    auto valOpt = resolveAtom(factor, table);
    if (!valOpt.has_value()) {
      return std::nullopt; // Propagate failure
    }

    product *= *valOpt;
    empty = false;

    start = end + 1;
  } while (end != std::string::npos);

  if (empty) {
    return std::nullopt;
  }
  return product;
}

// Evaluate a sum-of-products expression ("A + B*C + D").
std::optional<uint32_t>
evaluateExpression(std::string_view inputExpr,
                   const std::map<std::string, uint32_t> &table) {
  if (inputExpr.empty()) {
    return std::nullopt;
  }

  uint32_t totalSum = 0;
  size_t start = 0, end = 0;

  // Split by '+'
  do {
    end = inputExpr.find('+', start);
    std::string_view term = inputExpr.substr(start, end - start);

    // Evaluate the term (which might contain '*')
    auto termValOpt = evaluateProduct(term, table);
    if (!termValOpt.has_value()) {
      return std::nullopt; // Propagate failure
    }

    totalSum += *termValOpt;

    start = end + 1;
  } while (end != std::string::npos);

  return totalSum;
}

// Stringify result if evaluation succeeds, otherwise return original string.
std::string
maybeEvaluateExpression(std::string_view inputExpr,
                        const std::map<std::string, uint32_t> &table) {
  auto result = evaluateExpression(inputExpr, table);
  if (!result.has_value()) {
    return std::string(inputExpr);
  }
  return std::to_string(*result);
}

// Substitute symbols in an assembly line using the provided table.
std::string getSymbolReducedLine(const std::string &line,
                                 const std::map<std::string, uint32_t> &table) {
  std::string result;
  result.reserve(line.size());
  size_t currentPos = 0;

  // Pass 1: Handle Bracketed Registers (v[...], s[...])
  while (true) {
    size_t openBracket = line.find('[', currentPos);
    if (openBracket == std::string::npos) {
      result += line.substr(currentPos);
      break;
    }

    // 1. Check Register Prefix (v, s, acc)
    bool hasRegisterPrefix = false;
    static const std::vector<std::string> registerPrefixes = {"v", "s", "acc"};

    // Look backwards from '['
    for (const auto &prefix : registerPrefixes) {
      if (openBracket >= prefix.size()) {
        if (line.compare(openBracket - prefix.size(), prefix.size(), prefix) ==
            0) {
          hasRegisterPrefix = true;
          break;
        }
      }
    }

    if (!hasRegisterPrefix) {
      result += line.substr(currentPos, openBracket - currentPos + 1);
      currentPos = openBracket + 1;
      continue;
    }

    // Append text before the bracket
    result += line.substr(currentPos, openBracket - currentPos);

    size_t closeBracket = line.find(']', openBracket);
    if (closeBracket == std::string::npos) {
      throw std::runtime_error("Mismatched brackets in line");
    }

    // Extract content
    std::string_view content(line.data() + openBracket + 1,
                             closeBracket - openBracket - 1);

    size_t colonPos = content.find(':');
    if (colonPos != std::string::npos) {
      // Range: [Start : End]
      result += "[";
      result += maybeEvaluateExpression(content.substr(0, colonPos), table);
      result += ":";
      result += maybeEvaluateExpression(content.substr(colonPos + 1), table);
      result += "]";
    } else {
      // Single Value: Try to evaluate. If success, remove brackets.
      // If fail (unknown symbol), keep brackets (e.g. v[UNKNOWN]).
      auto maybeVal = evaluateExpression(content, table);
      if (maybeVal.has_value()) {
        result += std::to_string(*maybeVal);
      } else {
        result += "[";
        result += std::string(content);
        result += "]";
      }
    }
    currentPos = closeBracket + 1;
  }

  // Pass 2: Handle Standalone Operands (e.g. "v14, MT0, MT0*2")
  std::string finalResult;
  finalResult.reserve(result.size());

  size_t i = 0;
  const size_t len = result.size();

  while (i < len) {
    char c = result[i];

    // Skip Delimiters
    if (c == ' ' || c == '\t' || c == ',') {
      finalResult += c;
      i++;
      continue;
    }

    // Extract Token
    size_t start = i;
    while (i < len && result[i] != ' ' && result[i] != '\t' &&
           result[i] != ',') {
      i++;
    }
    std::string_view token(result.data() + start, i - start);

    // Try Resolve
    auto valOpt = evaluateExpression(token, table);
    if (valOpt.has_value()) {
      finalResult += std::to_string(*valOpt);
    } else {
      finalResult += token;
    }
  }

  return finalResult;
}

// Parse packed modifiers like "op_sel:[1,0] neg_lo:[1,0]" into a map.
// Key is the token before the colon, value list is inside the brackets.
std::map<std::string, std::vector<int>>
parsePackedModifiers(std::string_view line) {
  std::map<std::string, std::vector<int>> modifiers;

  size_t currentPos = 0;
  while (true) {
    // 1. Find Colon
    size_t colonPos = line.find(':', currentPos);
    if (colonPos == std::string_view::npos) {
      break;
    }

    // 2. Strict Check: Next char MUST be '['
    if (colonPos + 1 >= line.size() || line[colonPos + 1] != '[') {
      currentPos = colonPos + 1;
      continue;
    }

    // 3. Identify Key
    // Search backwards from colon for delimiters (space, tab, comma)
    size_t lastSpace = line.find_last_of(" \t,", colonPos);

    size_t keyStart;
    if (lastSpace == std::string_view::npos) {
      keyStart = 0;
    } else {
      keyStart = lastSpace + 1;
    }

    std::string key(trim(line.substr(keyStart, colonPos - keyStart)));

    // 4. Find Closing Bracket
    size_t bracketOpen = colonPos + 1; // Known to be '['
    size_t bracketClose = line.find(']', bracketOpen);

    if (bracketClose == std::string_view::npos) {
      currentPos = colonPos + 1;
      continue;
    }

    // 5. Parse Values
    std::string_view content =
        line.substr(bracketOpen + 1, bracketClose - bracketOpen - 1);
    std::vector<int> values;

    size_t valStart = 0;
    while (valStart < content.size()) {
      size_t valEnd = content.find(',', valStart);
      if (valEnd == std::string_view::npos) {
        valEnd = content.size();
      }

      std::string_view segment =
          trim(content.substr(valStart, valEnd - valStart));

      uint32_t num = 0;
      if (parseNumber(segment, num)) {
        values.push_back(static_cast<int>(num));
      }

      valStart = valEnd + 1;
    }

    if (!key.empty()) {
      modifiers[key] = values;
    }

    currentPos = bracketClose + 1;
  }

  return modifiers;
}

std::string stripComments(const std::string &line, bool &inBlockComment) {
  std::string result = line;
  if (inBlockComment) {
    auto closePos = result.find("*/");
    if (closePos != std::string::npos) {
      inBlockComment = false;
      result = result.substr(closePos + 2);
    } else {
      return "";
    }
  }
  // Remove block comments, possibly multiple per line.
  for (;;) {
    auto firstLineComment = std::min(result.find("//"), result.find(';'));
    auto openPos = result.find("/*");
    if (openPos == std::string::npos ||
        (firstLineComment != std::string::npos &&
         openPos >= firstLineComment)) {
      break;
    }
    auto closePos = result.find("*/", openPos + 2);
    if (closePos != std::string::npos) {
      result = result.substr(0, openPos) + result.substr(closePos + 2);
    } else {
      result = result.substr(0, openPos);
      inBlockComment = true;
      break;
    }
  }
  // Remove line comments (; and //).
  auto semi = result.find(';');
  if (semi != std::string::npos) {
    result = result.substr(0, semi);
  }
  auto dslash = result.find("//");
  if (dslash != std::string::npos) {
    result = result.substr(0, dslash);
  }
  return result;
}

} // namespace raceemulator
