// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "EmulatorException.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace raceemulator {

/// Pure LDS (Local Data Share) memory storage. Provides typed read/write
/// access with bounds checking. Race detection and event tracking are
/// handled by the Workgroup class that owns this LDS instance.
class LDS {
public:
  LDS() = default;

  template <typename T> T read(int addr) const {
    if (addr < 0 || addr + sizeof(T) > memory.size()) {
      throw EmulatorException("LDS Read Out of Bounds: Address " +
                              std::to_string(addr) +
                              " (Size: " + std::to_string(sizeof(T)) + ")");
    }
    T value;
    std::memcpy(&value, memory.data() + addr, sizeof(T));
    return value;
  }

  /// Bulk read: copies nBytes from addr into out. Bounds are asserted
  /// (callers validate addresses before reaching this point).
  void read(int addr, void *out, int nBytes) const {
    assert(addr >= 0 && addr + nBytes <= static_cast<int>(memory.size()) &&
           "LDS bulk read out of bounds");
    std::memcpy(out, memory.data() + addr, nBytes);
  }

  template <typename T> void write(int addr, T value) {
    if (addr < 0 || addr + sizeof(T) > memory.size()) {
      throw EmulatorException("LDS Write Out of Bounds: Address " +
                              std::to_string(addr) +
                              " (Size: " + std::to_string(sizeof(T)) + ")");
    }
    std::memcpy(memory.data() + addr, &value, sizeof(T));
  }

  int getSize() const { return static_cast<int>(memory.size()); }

  void resize(int size);
  void clear();

  // Sentinel values used to initialize LDS memory. Reading these back
  // indicates uninitialized memory (helpful for debugging).
  static constexpr const uint8_t unset1 = 0x77;
  static constexpr const uint32_t unset4 = 0x77777777;

private:
  std::vector<char> memory;
};

} // namespace raceemulator
