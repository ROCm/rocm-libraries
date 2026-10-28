// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Codegen/Registers.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace rocm::tensilelite::codegen;

// accum_offset is rounded to rocisa's granule of 8; LLVM rounds to 4, which the
// hardware also accepts.
static constexpr int32_t accumOffsetGranule = 8;

static std::string_view kindName(RegisterKind kind) {
  switch (kind) {
  case RegisterKind::sgpr:
    return "sgpr";
  case RegisterKind::vgpr:
    return "vgpr";
  case RegisterKind::agpr:
    return "agpr";
  }
  throw std::invalid_argument("invalid register kind");
}

// Matches the prefix rocisa emits in assembly.
static std::string_view registerPrefix(RegisterKind kind) {
  switch (kind) {
  case RegisterKind::sgpr:
    return "s";
  case RegisterKind::vgpr:
    return "v";
  case RegisterKind::agpr:
    return "acc";
  }
  throw std::invalid_argument("invalid register kind");
}

// Formats a register range for diagnostics without touching rocisa's
// per-thread state.
static std::string registerText(RegisterKind kind, int32_t start,
                                int32_t count) {
  std::string prefix(registerPrefix(kind));
  if (count == 1)
    return prefix + std::to_string(start);
  return prefix + "[" + std::to_string(start) + ":" +
         std::to_string(start + count - 1) + "]";
}

static std::shared_ptr<rocisa::RegisterContainer>
makeContainer(RegisterKind kind, int32_t start, int32_t count) {
  float regNum = static_cast<float>(count);
  switch (kind) {
  case RegisterKind::sgpr:
    return rocisa::sgpr(start, regNum);
  case RegisterKind::vgpr:
    return rocisa::vgpr(start, regNum);
  case RegisterKind::agpr:
    return rocisa::accvgpr(start, regNum);
  }
  throw std::invalid_argument("invalid register kind");
}

// Occupies every free register below start one at a time, so that first-fit
// is forced to land on start. Returns the occupied indices.
static std::vector<int32_t> occupyBelow(rocisa::RegisterPool &regPool,
                                        int32_t start) {
  std::vector<int32_t> padding;
  while (true) {
    int32_t index = static_cast<int32_t>(
        regPool.checkOut(1, "reserve-padding", /*preventOverflow=*/1));
    if (index >= start) {
      regPool.checkIn(index);
      return padding;
    }
    padding.push_back(index);
  }
}

static void requirePositiveCount(int32_t count) {
  if (count <= 0)
    throw std::invalid_argument("register count must be positive, got " +
                                std::to_string(count));
}

RegisterRange::RegisterRange(RegisterAllocator &allocator, RegisterKind kind,
                             int32_t start, int32_t count)
    : allocator(&allocator), rangeKind(kind), rangeStart(start),
      rangeCount(count) {}

RegisterRange::RegisterRange(RegisterRange &&other) noexcept
    : allocator(std::exchange(other.allocator, nullptr)),
      rangeKind(other.rangeKind), rangeStart(other.rangeStart),
      rangeCount(other.rangeCount) {}

RegisterRange &RegisterRange::operator=(RegisterRange &&other) noexcept {
  if (this == &other)
    return *this;
  release();
  allocator = std::exchange(other.allocator, nullptr);
  rangeKind = other.rangeKind;
  rangeStart = other.rangeStart;
  rangeCount = other.rangeCount;
  return *this;
}

RegisterRange::~RegisterRange() { release(); }

RegisterKind RegisterRange::kind() const {
  assert(allocator && "register range owns no registers");
  return rangeKind;
}

int32_t RegisterRange::start() const {
  assert(allocator && "register range owns no registers");
  return rangeStart;
}

int32_t RegisterRange::count() const {
  assert(allocator && "register range owns no registers");
  return rangeCount;
}

std::shared_ptr<rocisa::RegisterContainer> RegisterRef::container() const {
  return makeContainer(kind, start, count);
}

RegisterRef RegisterRef::slice(int32_t offset, int32_t subCount) const {
  if (offset < 0 || subCount <= 0 || offset + subCount > count)
    throw std::out_of_range("register sub-range at offset " +
                            std::to_string(offset) + " with count " +
                            std::to_string(subCount) + " is outside " +
                            registerText(kind, start, count));
  return RegisterRef{kind, start + offset, subCount};
}

std::shared_ptr<rocisa::RegisterContainer> RegisterRange::container() const {
  return ref().container();
}

std::shared_ptr<rocisa::RegisterContainer>
RegisterRange::sub(int32_t offset, int32_t subCount) const {
  return ref(offset, subCount).container();
}

RegisterRef RegisterRange::ref() const {
  assert(allocator && "register range owns no registers");
  return RegisterRef{rangeKind, rangeStart, rangeCount};
}

RegisterRef RegisterRange::ref(int32_t offset, int32_t subCount) const {
  assert(allocator && "register range owns no registers");
  return ref().slice(offset, subCount);
}

void RegisterRange::release() {
  if (!allocator)
    return;
  allocator->release(rangeKind, rangeStart);
  allocator = nullptr;
}

RegisterAllocator::RegisterAllocator(const Target &target)
    : target(target),
      sgprPool(0, rocisa::RegisterType::Sgpr, /*defaultPreventOverflow=*/false),
      vgprPool(0, rocisa::RegisterType::Vgpr,
               /*defaultPreventOverflow=*/false),
      agprPool(0, rocisa::RegisterType::Accvgpr,
               /*defaultPreventOverflow=*/false) {}

RegisterAllocator::~RegisterAllocator() {
  assert(liveRanges == 0 &&
         "register allocator destroyed while register ranges are alive");
}

rocisa::RegisterPool &RegisterAllocator::pool(RegisterKind kind) {
  return const_cast<rocisa::RegisterPool &>(std::as_const(*this).pool(kind));
}

const rocisa::RegisterPool &RegisterAllocator::pool(RegisterKind kind) const {
  switch (kind) {
  case RegisterKind::sgpr:
    return sgprPool;
  case RegisterKind::vgpr:
    return vgprPool;
  case RegisterKind::agpr:
    return agprPool;
  }
  throw std::invalid_argument("invalid register kind");
}

RegisterRange RegisterAllocator::allocate(RegisterKind kind, int32_t count,
                                          int32_t alignment,
                                          std::string_view tag) {
  requirePositiveCount(count);
  if (alignment <= 0)
    throw std::invalid_argument("register alignment must be positive, got " +
                                std::to_string(alignment));
  std::string poolTag = tag.empty() ? "codegen" : std::string(tag);
  int32_t start = static_cast<int32_t>(
      pool(kind).checkOutAligned(count, alignment, poolTag));
  ++liveRanges;
  return RegisterRange(*this, kind, start, count);
}

void RegisterAllocator::requireFree(RegisterKind kind, int32_t start,
                                    int32_t count) const {
  std::vector<rocisa::RegisterPool::Register> slots = pool(kind).getPool();
  int32_t end = std::min(start + count, static_cast<int32_t>(slots.size()));
  for (int32_t index = start; index < end; ++index) {
    if (slots[index].status != rocisa::RegisterPool::Status::InUse)
      continue;
    throw std::invalid_argument(
        "cannot reserve " + registerText(kind, start, count) + ": " +
        registerText(kind, index, 1) + " is already in use");
  }
}

RegisterRange RegisterAllocator::reserve(RegisterKind kind, int32_t start,
                                         int32_t count, std::string_view tag) {
  requirePositiveCount(count);
  if (start < 0)
    throw std::invalid_argument("register start must not be negative, got " +
                                std::to_string(start));
  // Checked before growing the pool so a failure keeps the high-water mark.
  requireFree(kind, start, count);
  rocisa::RegisterPool &regPool = pool(kind);
  regPool.appendPool(start + count);
  std::vector<int32_t> padding = occupyBelow(regPool, start);
  std::string poolTag = tag.empty() ? "codegen" : std::string(tag);
  int32_t got = static_cast<int32_t>(regPool.checkOut(count, poolTag, 1));
  for (int32_t index : padding)
    regPool.checkIn(index);
  if (got != start) {
    regPool.checkIn(got);
    throw std::logic_error("reservation of " +
                           registerText(kind, start, count) +
                           " landed at index " + std::to_string(got));
  }
  ++liveRanges;
  return RegisterRange(*this, kind, start, count);
}

int32_t RegisterAllocator::highWaterMark(RegisterKind kind) const {
  return static_cast<int32_t>(pool(kind).size());
}

int32_t RegisterAllocator::accumOffset() const {
  int32_t vgprs = std::max(1, highWaterMark(RegisterKind::vgpr));
  return (vgprs + accumOffsetGranule - 1) / accumOffsetGranule *
         accumOffsetGranule;
}

void RegisterAllocator::checkLimit(RegisterKind kind, int32_t limit) const {
  int32_t used = highWaterMark(kind);
  if (used <= limit)
    return;
  throw std::runtime_error("kernel uses " + std::to_string(used) + " " +
                           std::string(kindName(kind)) + "s but " +
                           std::string(target.archName()) + " allows at most " +
                           std::to_string(limit));
}

void RegisterAllocator::checkUnifiedLimit() const {
  int32_t agprs = highWaterMark(RegisterKind::agpr);
  int32_t used = accumOffset() + agprs;
  if (used <= target.maxUnifiedVgpr())
    return;
  throw std::runtime_error(
      "kernel uses " + std::to_string(used) + " unified vgprs (accum offset " +
      std::to_string(accumOffset()) + " + " + std::to_string(agprs) +
      " agprs) but " + std::string(target.archName()) + " allows at most " +
      std::to_string(target.maxUnifiedVgpr()));
}

void RegisterAllocator::checkLimits() const {
  checkLimit(RegisterKind::sgpr, target.maxSgpr());
  checkLimit(RegisterKind::vgpr, target.maxVgpr());
  // Checked before the AGPR limit so an overflowing register file is reported
  // with its accum offset.
  checkUnifiedLimit();
  checkLimit(RegisterKind::agpr, target.maxAgpr());
}

void RegisterAllocator::release(RegisterKind kind, int32_t start) {
  pool(kind).checkIn(start);
  --liveRanges;
}
