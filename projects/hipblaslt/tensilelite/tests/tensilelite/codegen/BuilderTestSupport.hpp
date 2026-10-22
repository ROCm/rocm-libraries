// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_TESTS_CODEGEN_BUILDERTESTSUPPORT_HPP
#define ROCM_TENSILELITE_TESTS_CODEGEN_BUILDERTESTSUPPORT_HPP

#include <tensilelite/codegen/IRBuilder.hpp>

#include "code.hpp"
#include "container.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <string_view>

namespace rocm::tensilelite::codegen::test {

inline std::string text(RegisterRef ref) { return ref.container()->toString(); }

inline std::string text(const RegisterRange &range) {
  return text(range.ref());
}

/// Builds "mnemonic dst, src" as rocisa prints it.
inline std::string line(std::string_view mnemonic, const std::string &dst,
                        const std::string &src) {
  return std::string(mnemonic) + " " + dst + ", " + src;
}

/// Builds "mnemonic dst, src0, src1" as rocisa prints it.
inline std::string line(std::string_view mnemonic, const std::string &dst,
                        const std::string &src0, const std::string &src1) {
  return line(mnemonic, dst, src0) + ", " + src1;
}

inline int countLinesContaining(const std::string &text,
                                std::string_view needle) {
  int count = 0;
  std::istringstream stream(text);
  for (std::string current; std::getline(stream, current);)
    if (current.find(needle) != std::string::npos)
      ++count;
  return count;
}

/// A target, allocator and builder over a body module, for host text tests.
class BuilderTest : public testing::Test {
protected:
  // Ranges a test creates are destroyed before the builder and the allocator.
  Target target = Target::get("gfx950");
  RegisterAllocator allocator{target};
  rocisa::Module body{"body"};
  IRBuilder builder{target, allocator, body};

  /// Runs emit on a builder over a fresh module, so each call is checked in
  /// isolation, and returns the module's text.
  template <typename Emit>
  std::string emitted(Emit emit) {
    rocisa::Module module{"module"};
    IRBuilder local{target, allocator, module};
    emit(local);
    return module.toString();
  }
};

} // namespace rocm::tensilelite::codegen::test

#endif // ROCM_TENSILELITE_TESTS_CODEGEN_BUILDERTESTSUPPORT_HPP
