// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <cassert>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Emulator & GPU independent functionality.

namespace raceemulator {

class Dim3d {
public:
  Dim3d() = delete;
  Dim3d(int x) : x(x), y(0), z(0) {}
  Dim3d(int x, int y) : x(x), y(y), z(0) {}
  Dim3d(int x, int y, int z) : x(x), y(y), z(z) {}
  int x = 0;
  int y = 0;
  int z = 0;
};

std::vector<std::string_view> getPartitioned(std::string_view line);

// Helper to trim whitespace, returning a view to avoid copies
std::string_view trim(std::string_view sv);

// Helper to parse number (Hex 0x... or Decimal) using std::from_chars
bool parseNumber(std::string_view sv, uint32_t &outVal);

// Evaluates "Symbol + 0 + AnotherSymbol + 4"
// Returns input if evaluation fails.
std::string
maybeEvaluateExpression(std::string_view inputExpr,
                        const std::map<std::string, uint32_t> &table);

std::string getSymbolReducedLine(const std::string &line,
                                 const std::map<std::string, uint32_t> &table);

std::optional<uint32_t>
evaluateExpression(std::string_view inputExpr,
                   const std::map<std::string, uint32_t> &table);

std::map<std::string, std::vector<int>>
parsePackedModifiers(std::string_view line);

template <typename T> T getIntFromView(std::string_view nStr) {
  if (nStr.empty()) {
    return 0;
  }

  // Strip optional leading '-'.
  bool negative = (nStr[0] == '-');
  auto digits = negative ? nStr.substr(1) : nStr;

  // Hex (0x/0X) or binary (0b/0B): parse as unsigned bit pattern.
  if (digits.size() > 2 && digits[0] == '0' &&
      (digits[1] == 'x' || digits[1] == 'X' || digits[1] == 'b' ||
       digits[1] == 'B')) {
    int base = (digits[1] == 'x' || digits[1] == 'X') ? 16 : 2;
    digits.remove_prefix(2);
    using U = std::make_unsigned_t<T>;
    U uValue = 0;
    auto [ptr, ec] = std::from_chars(
        digits.data(), digits.data() + digits.size(), uValue, base);
    if (ec != std::errc()) {
      throw std::runtime_error("Hex/Bin parsing failed for: " +
                               std::string(nStr));
    }
    int64_t signed_val = static_cast<int64_t>(uValue);
    return static_cast<T>(negative ? -signed_val : signed_val);
  }

  // Decimal: parse as signed to handle negative values, then cast to T.
  // (e.g. "-1" as uint32_t → 0xFFFFFFFF)
  if (negative) {
    int64_t sValue = 0;
    auto [ptr, ec] =
        std::from_chars(nStr.data(), nStr.data() + nStr.size(), sValue, 10);
    if (ec != std::errc()) {
      throw std::runtime_error("Decimal parsing failed for: " +
                               std::string(nStr));
    }
    return static_cast<T>(sValue);
  }

  // Positive decimal: parse as unsigned to handle large values.
  uint64_t uValue = 0;
  auto [ptr, ec] =
      std::from_chars(nStr.data(), nStr.data() + nStr.size(), uValue, 10);
  if (ec != std::errc()) {
    throw std::runtime_error("Decimal parsing failed for: " +
                             std::string(nStr));
  }
  return static_cast<T>(uValue);
}

template <typename T> T getFloatFromView(std::string_view nStr) {
  T value;
#if defined(__cpp_lib_to_chars)
  // C++17/20 high-performance standard parsing
  auto [ptr, ec] =
      std::from_chars(nStr.data(), nStr.data() + nStr.size(), value);
  if (ec != std::errc()) {
    throw std::runtime_error("Floating point parsing failed");
  }
#else
  static_assert(false &&
                "Requires C++20 std::from_chars for floating point parsing");
#endif
  return value;
}

} // namespace raceemulator
