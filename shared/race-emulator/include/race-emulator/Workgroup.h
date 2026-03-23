#pragma once
#include "CommonRegister.h"
#include "EmulatorException.h"
#include "IntervalSet.h"
#include "LDS.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace raceemulator {

/// Identifies a wave (SIMD execution unit) within a workgroup. Wave 0 runs
/// lanes [0, waveSize), wave 1 runs [waveSize, 2*waveSize), and so on.
/// Strongly typed to prevent accidental mixing with lane indices, register
/// indices, or event IDs.
struct WaveId {
  int value;
  bool operator==(WaveId o) const { return value == o.value; }
  bool operator!=(WaveId o) const { return value != o.value; }
  bool operator<(WaveId o) const { return value < o.value; }
};

/// Identifies a memory event within a workgroup. Each memory instruction
/// (LDS read/write, global load/store) creates an event via
/// Workgroup::allocateEventId(). The event ID is used to track the event
/// through its lifecycle: registration, s_waitcnt completion, and barrier
/// retirement. Strongly typed to prevent accidental mixing with wave IDs,
/// register indices, or byte addresses.
struct EventId {
  int value;
  bool operator==(EventId o) const { return value == o.value; }
  bool operator!=(EventId o) const { return value != o.value; }
  bool operator<(EventId o) const { return value < o.value; }
};

/// Status of a memory event in the race detection lifecycle.
enum class EventStatus {
  ACTIVE,       // Pending. Unsafe for everyone.
  WAVE_COMPLETE // s_waitcnt passed. Safe for owning wave, unsafe for others.
};

/// Workgroup-scoped state: owns LDS memory, the event registry, and race
/// detection logic. Waves hold a reference to a Workgroup and use it for
/// all memory event tracking and validated LDS access.
///
/// Event lifecycle:
///   1. Wave issues a memory instruction → allocateEventId() allocates an
///      event. LDS events (VGPR_TO_LDS, LDS_TO_VGPR) are automatically
///      added to the appropriate live event list (writes or reads).
///   2. s_waitcnt → markEventWaveComplete() transitions the event to
///      WAVE_COMPLETE. The owning wave may now safely access the same
///      LDS bytes, but other waves may not.
///   3. s_barrier (all waves arrived) → retireLdsEvent() removes the event
///      from the live list and decrements per-byte counts, making the LDS
///      ranges available for cross-wave access.
///
/// Race validation uses a two-level approach:
///   - Fast path: per-byte counters (byteWriteCounts / byteReadCounts)
///     provide O(1) checks. If all bytes in a read/write range have zero
///     count, no race is possible and validation returns immediately.
///   - Slow path: when counts are non-zero, scans the live event list
///     (ldsWriteEvents or ldsReadEvents) and checks interval overlap
///     using binary search (IntervalSet::overlapsRange).
///
/// The event lists are split by direction so that validateRead only scans
/// write events (RAW check) and validateWrite only scans read events
/// (WAR check). This will extend naturally to GLOBAL_TO_LDS when
/// direct-to-LDS copies are supported.
class Workgroup {

  /// Per-event metadata, stored once in the event registry (not per byte).
  struct EventInfo {
    WaveId waveId;
    int pc;
    MemoryEventType type;
    EventStatus status;
    // Bitmask of which bytes within the 4-byte VGPR this event covers.
    // Bit 0 = byte 0 (bits [7:0]), bit 3 = byte 3 (bits [31:24]).
    // Important cases:
    //   0xF (0b1111) = all 4 bytes, full register (default, non-d16)
    //   0x3 (0b0011) = lower 2 bytes [15:0]  (buffer_load_d16_b16, etc.)
    //   0xC (0b1100) = upper 2 bytes [31:16]  (buffer_load_d16_hi_b16, etc.)
    uint8_t byteMask;
    uint64_t execMask;               // Exec mask at the time of the event.
    std::vector<uint32_t> registers; // VGPR indices involved in this event.
    // LDS byte ranges touched by this event. Sorted by start, non-overlapping,
    // merged where adjacent. Empty for non-LDS events.
    IntervalSet ldsIntervals;
  };

  /// Check for RAW (read-after-write) hazards: confirms no outstanding
  /// VGPR_TO_LDS write events overlap [addr, addr+nBytes). Uses per-byte
  /// counts for fast-path, then interval scanning if counts are non-zero.
  void validateRead(int addr, WaveId wave, int lane, int nBytes) const;

  /// Check for WAR (write-after-read) hazards: confirms no outstanding
  /// LDS_TO_VGPR read events overlap [addr, addr+nBytes).
  void validateWrite(int addr, WaveId wave, int lane, int nBytes) const;

public:
  Workgroup() = default;

  // --- LDS access with race checking ---

  template <typename T> T readLds(int addr, WaveId wave, int lane) const {
    if (raceChecks) {
      validateRead(addr, wave, lane, sizeof(T));
    }
    return lds.read<T>(addr);
  }

  template <typename T>
  void writeLds(int addr, WaveId wave, int lane, T value) {
    if (raceChecks) {
      validateWrite(addr, wave, lane, sizeof(T));
    }
    lds.write<T>(addr, value);
  }

  // Bulk LDS read: validates the full range once, then copies count elements.
  template <typename T>
  void readLds(int addr, WaveId wave, int lane, T *out, int count) const {
    int nBytes = count * sizeof(T);
    if (raceChecks) {
      validateRead(addr, wave, lane, nBytes);
    }
    lds.read(addr, out, nBytes);
  }

  // --- Direct LDS access (for test scaffolding) ---
  LDS &getLds() { return lds; }
  const LDS &getLds() const { return lds; }

  int getLdsSize() const { return lds.getSize(); }
  void resizeLds(int size);
  void clear();

  // --- LDS event tracking ---

  /// Transition an event from ACTIVE to WAVE_COMPLETE.
  void markEventWaveComplete(EventId eventId);

  /// Retire an LDS event at s_barrier. Updates two data structures:
  ///   1. Removes eventId from ldsWriteEvents or ldsReadEvents (by type).
  ///   2. Decrements per-byte counts (byteWriteCounts or byteReadCounts)
  ///      for each byte in the event's ldsIntervals.
  /// After retirement, the LDS bytes covered by this event are no longer
  /// considered in race validation.
  void retireLdsEvent(EventId eventId);

  /// Return the live LDS event IDs (for error reporting).
  const std::vector<EventId> &getLdsWriteEvents() const {
    return ldsWriteEvents;
  }
  const std::vector<EventId> &getLdsReadEvents() const { return ldsReadEvents; }

  void setRaceChecks(bool enable) { raceChecks = enable; }

  // --- Event registry ---

  /// Allocate a workgroup-global event ID and record its metadata.
  EventId allocateEventId(WaveId waveId, int pc, MemoryEventType type,
                          const std::vector<uint32_t> &registers,
                          uint64_t execMask, uint8_t byteMask = 0xF,
                          IntervalSet ldsIntervals = {}) {
    int id = nextEventId++;
    if (id >= static_cast<int64_t>(eventRegistry.size())) {
      eventRegistry.resize(id + 1);
    }
    bool hasLds = !ldsIntervals.empty();
    eventRegistry[id] = {
        waveId,   pc,       type,      EventStatus::ACTIVE,
        byteMask, execMask, registers, std::move(ldsIntervals)};
    EventId eid{id};
    if (hasLds) {
      const auto &ivs = eventRegistry[id].ldsIntervals;
      if (type == MemoryEventType::VGPR_TO_LDS) {
        ldsWriteEvents.push_back(eid);
        adjustByteCounts(ivs, byteWriteCounts, +1);
      } else if (type == MemoryEventType::LDS_TO_VGPR) {
        ldsReadEvents.push_back(eid);
        adjustByteCounts(ivs, byteReadCounts, +1);
      }
    }
    return eid;
  }

  /// Get the LDS byte ranges touched by an event.
  const IntervalSet &getEventIntervals(EventId eventId) const {
    return eventRegistry[eventId.value].ldsIntervals;
  }

  /// Look up the wave that owns a given event ID.
  WaveId getEventWaveId(EventId eventId) const {
    return eventRegistry[eventId.value].waveId;
  }

  /// Look up the type of a given event ID.
  MemoryEventType getEventType(EventId eventId) const {
    return eventRegistry[eventId.value].type;
  }

  /// Look up the status of a given event ID.
  EventStatus getEventStatus(EventId eventId) const {
    return eventRegistry[eventId.value].status;
  }

  /// Look up the VGPR byte mask of a given event ID.
  uint8_t getEventByteMask(EventId eventId) const {
    return eventRegistry[eventId.value].byteMask;
  }

  /// Look up the PC (assembly line number) of a given event ID.
  int getEventPc(EventId eventId) const {
    return eventRegistry[eventId.value].pc;
  }

  /// Look up the exec mask at the time of a given event ID.
  uint64_t getEventExecMask(EventId eventId) const {
    return eventRegistry[eventId.value].execMask;
  }

  /// Check whether a given lane was active when the event was issued.
  static inline bool isLaneActive(uint64_t execMask, int lane) {
    return (execMask >> lane) & 1;
  }

  bool isEventActiveForLane(EventId eventId, int lane) const {
    return isLaneActive(eventRegistry[eventId.value].execMask, lane);
  }

  /// Look up the VGPR indices involved in a given event ID.
  const std::vector<uint32_t> &getEventRegisters(EventId eventId) const {
    return eventRegistry[eventId.value].registers;
  }

private:
  static void adjustByteCounts(const IntervalSet &ivs, std::vector<int> &counts,
                               int delta) {
    for (const auto &iv : ivs) {
      for (int b = iv.start; b < iv.end; ++b) {
        counts[b] += delta;
      }
    }
  }

  LDS lds;

  // Live LDS events, split by direction. validateRead scans only writes,
  // validateWrite scans only reads. Retired at s_barrier.
  std::vector<EventId> ldsWriteEvents; // VGPR_TO_LDS
  std::vector<EventId> ldsReadEvents;  // LDS_TO_VGPR

  // Per-byte event counts for fast-path validation. If count is zero for all
  // bytes in a read/write range, no race is possible — skip interval scanning.
  std::vector<int> byteWriteCounts; // VGPR_TO_LDS events per byte
  std::vector<int> byteReadCounts;  // LDS_TO_VGPR events per byte

  // Per-event metadata, indexed by eventId (sequential from 0).
  std::vector<EventInfo> eventRegistry;
  int nextEventId{0};

  bool raceChecks{false};
};

} // namespace raceemulator
