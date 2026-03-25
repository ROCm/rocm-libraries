// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <cassert>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

namespace raceemulator {

enum class MemoryEventType {
  GLOBAL_TO_VGPR = 0,
  VGPR_TO_GLOBAL,
  LDS_TO_VGPR,
  VGPR_TO_LDS,

  // Direct-to-LDS (DTL): `buffer_load ... lds` bypasses VGPRs and writes
  // global memory data directly to LDS. Unlike VGPR_TO_LDS (which has VGPR
  // registers), GLOBAL_TO_LDS events have no VGPR registers — only LDS
  // intervals.
  //
  // Wait counter: DTL is a MUBUF instruction, so it uses vmcnt (not lgkmcnt).
  // s_waitcnt vmcnt(0) must complete before the owning wave can read the LDS
  // bytes. Cross-wave access requires s_barrier after vmcnt, same as ds_write
  // followed by s_barrier.
  GLOBAL_TO_LDS,

  N
};

// Event direction helpers: "to VGPR" means a load writing into a VGPR,
// "from VGPR" means a store reading out of a VGPR.
inline bool isToVgpr(MemoryEventType t) {
  return t == MemoryEventType::GLOBAL_TO_VGPR ||
         t == MemoryEventType::LDS_TO_VGPR;
}

inline bool isToLds(MemoryEventType t) {
  return t == MemoryEventType::VGPR_TO_LDS ||
         t == MemoryEventType::GLOBAL_TO_LDS;
}

inline bool isFromVgpr(MemoryEventType t) {
  return t == MemoryEventType::VGPR_TO_GLOBAL ||
         t == MemoryEventType::VGPR_TO_LDS;
}

class CommonRegister {
public:
  enum class Type { SGPR, VGPR, UNKNOWN };
  Type type;
  int index;

  static CommonRegister getVgpr(int idx) {
    return CommonRegister{Type::VGPR, idx};
  }

  // Append to an ostream
  void appendStr(std::ostream &os) const;
  std::string str() const;
};
} // namespace raceemulator
