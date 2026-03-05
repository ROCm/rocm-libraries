#pragma once
#include <cassert>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

namespace raceemulator {

class LaneAndLDSByte {
public:
  LaneAndLDSByte(int lane, int byte) : lane(lane), byte(byte) {}
  int lane;
  int byte;
};

using LaneAndLDSBytes = std::vector<LaneAndLDSByte>;

enum class MemoryEventType {
  GLOBAL_TO_VGPR = 0,
  VGPR_TO_GLOBAL,
  LDS_TO_VGPR,
  VGPR_TO_LDS,
  N
};

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
