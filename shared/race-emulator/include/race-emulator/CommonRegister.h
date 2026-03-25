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
