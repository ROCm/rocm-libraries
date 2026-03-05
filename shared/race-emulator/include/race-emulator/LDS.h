#pragma once
#include "CommonRegister.h"
#include "EmulatorException.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace raceemulator {

class LDSMemoryEvent {
public:
  // Status of the memory operation
  enum class EventStatus {
    ACTIVE,       // Pending. Unsafe for everyone.
    WAVE_COMPLETE // 's_waitcnt' passed. Safe for OWNING wave, Unsafe for
                  // OTHERS.
  };

  LDSMemoryEvent(uint32_t eventId, uint32_t waveId, uint32_t lane,
                 MemoryEventType type, EventStatus status)
      : eventId(eventId), waveId(waveId), lane(lane), type(type),
        status(status) {}

  std::string str() const;

  uint32_t eventId;
  uint32_t waveId;
  uint32_t lane;
  MemoryEventType type;
  EventStatus status;
};

class LDS {

private:
  // Confirm there are no outstanding writes to bytes being read.
  void validateRead(int addr, int wave, int lane, int nBytes) const;

  // Confirm outstanding reads or writes to bytes being written.
  void validateWrite(int addr, int wave, int lane, int nBytes) const;

public:
  LDS(std::vector<char> &&lds_) : memory(std::move(lds_)) {}
  LDS() = default;

  template <typename T> T read(int addr, int wave, int lane) const {
    if (addr < 0 || addr + sizeof(T) > memory.size()) {
      throw EmulatorException("LDS Read Out of Bounds: Address " +
                              std::to_string(addr) +
                              " (Size: " + std::to_string(sizeof(T)) + ")");
    }

    if (raceChecks) {
      validateRead(addr, wave, lane, sizeof(T));
    }

    T value;
    std::memcpy(&value, memory.data() + addr, sizeof(T));
    return value;
  }

  template <typename T> T readWithoutChecks(int addr) {
    T value;
    std::memcpy(&value, memory.data() + addr, sizeof(T));
    return value;
  }

  template <typename T> void write(int addr, int wave, int lane, T value) {
    if (addr < 0 || addr + sizeof(T) > memory.size()) {
      throw EmulatorException("LDS Write Out of Bounds: Address " +
                              std::to_string(addr) +
                              " (Size: " + std::to_string(sizeof(T)) + ")");
    }

    if (raceChecks) {
      validateWrite(addr, wave, lane, sizeof(T));
    }

    std::memcpy(memory.data() + addr, &value, sizeof(T));
  }

  template <typename T> void writeWithoutChecks(int addr, T value) {
    std::memcpy(memory.data() + addr, &value, sizeof(T));
  }

  int getSize() const { return static_cast<int>(memory.size()); }

  void clear();
  void resize(int size);

  static constexpr const uint8_t unset1 = 0x77;
  static constexpr const uint32_t unset4 = 0x77777777;

  void addEvent(int byteAddr, const LDSMemoryEvent &event) {
    assert(byteAddr >= 0);
    assert(byteAddr < static_cast<int64_t>(byteMemoryEvents.size()));
    byteMemoryEvents[byteAddr].push_back(event);
  }

  void removeEvent(int byteAddr, uint32_t waveId, uint32_t eventId);
  void markEventWaveComplete(int byteAddr, uint32_t waveId, uint32_t eventId);

  const std::vector<LDSMemoryEvent> &getByteMemoryEvents(int byteAddr) const {
    assert(byteAddr >= 0);
    assert(byteAddr < static_cast<int64_t>(byteMemoryEvents.size()));
    return byteMemoryEvents[byteAddr];
  }

  void enableRaceChecks(bool enable) { raceChecks = enable; }

private:
  std::vector<char> memory;
  std::vector<std::vector<LDSMemoryEvent>> byteMemoryEvents;

  bool raceChecks{false};
};

} // namespace raceemulator
