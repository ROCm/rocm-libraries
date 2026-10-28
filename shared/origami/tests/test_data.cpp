// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_data.hpp"

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

#include <string>
#include <system_error>

namespace fs = std::filesystem;

fs::path test_executable() {
#if defined(_WIN32)
  std::wstring buf(32768, L'\0');
  const DWORD  n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
  if (n == 0 || n >= buf.size()) return {};
  buf.resize(n);
  return fs::path(buf);
#elif defined(__linux__)
  std::error_code ec;
  const fs::path  exe = fs::read_symlink("/proc/self/exe", ec);
  return ec ? fs::path() : exe;
#else
  return {};
#endif
}

fs::path origami_test_data(const fs::path& name, const char* source_dir) {
  if (const fs::path exe = test_executable(); !exe.empty()) {
    std::error_code ec;
    const fs::path  installed = exe.parent_path() / "origami" / "test-data" / name;
    if (fs::is_directory(installed, ec)) return installed;
  }
  return source_dir;
}

fs::path olc_test_data() { return origami_test_data("olc", ORIGAMI_OLC_TEST_DATA); }
