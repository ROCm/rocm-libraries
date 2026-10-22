// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_TESTS_CODEGEN_EXPECTLINES_HPP
#define ROCM_TENSILELITE_TESTS_CODEGEN_EXPECTLINES_HPP

#include <gtest/gtest.h>

#include <cstddef>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace rocm::tensilelite::codegen::test {

// Checks each needle against the lines of text in order, so tests do not
// depend on rocisa's padding and comment formatting.
inline void expectLinesInOrder(const std::string &text,
                               std::vector<std::string_view> needles) {
  std::vector<std::string> lines;
  std::istringstream stream(text);
  for (std::string line; std::getline(stream, line);)
    lines.push_back(line);

  std::ptrdiff_t next = 0;
  for (std::string_view needle : needles) {
    while (next < std::ssize(lines) &&
           lines[next].find(needle) == std::string::npos)
      ++next;
    if (next == std::ssize(lines)) {
      ADD_FAILURE() << "missing or out of order: '" << needle << "' in:\n"
                    << text;
      return;
    }
    ++next;
  }
}

} // namespace rocm::tensilelite::codegen::test

#endif // ROCM_TENSILELITE_TESTS_CODEGEN_EXPECTLINES_HPP
