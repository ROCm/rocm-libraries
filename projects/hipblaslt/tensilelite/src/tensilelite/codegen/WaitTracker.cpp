// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/WaitTracker.hpp>

#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace rocm::tensilelite::codegen;

namespace {

using Window = std::vector<std::vector<int>>;

bool contains(const std::vector<int> &ids, int id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

// The newest position holding id is the smallest distance, which is the safe
// one when several paths merged.
std::optional<int> distanceOf(const Window &window, int id) {
  for (int index = static_cast<int>(std::ssize(window)) - 1; index >= 0;
       --index)
    if (contains(window[index], id))
      return static_cast<int>(std::ssize(window)) - 1 - index;
  return std::nullopt;
}

// Waiting down to count completes every position at distance >= count.
void completeFrom(Window &window, int count) {
  int size = static_cast<int>(std::ssize(window));
  if (size <= count)
    return;
  window.erase(window.begin(), window.begin() + (size - count));
}

void appendUnique(std::vector<int> &ids, const std::vector<int> &more) {
  for (int id : more)
    if (!contains(ids, id))
      ids.push_back(id);
}

// Aligns the windows at their newest end and unions the ids at each distance.
Window joinWindows(const Window &a, const Window &b) {
  int sizeA = static_cast<int>(std::ssize(a));
  int sizeB = static_cast<int>(std::ssize(b));
  int size = std::max(sizeA, sizeB);
  Window joined(size);
  for (int distance = 0; distance < size; ++distance) {
    std::vector<int> &position = joined[size - 1 - distance];
    if (distance < sizeA)
      appendUnique(position, a[sizeA - 1 - distance]);
    if (distance < sizeB)
      appendUnique(position, b[sizeB - 1 - distance]);
  }
  return joined;
}

std::optional<WaitKind> classifyScalarOrFlat(const rocisa::Instruction &inst) {
  if (dynamic_cast<const rocisa::SMemLoadInstruction *>(&inst) ||
      dynamic_cast<const rocisa::SMemStoreInstruction *>(&inst))
    return WaitKind::scalarMemory;
  if (dynamic_cast<const rocisa::FLATReadInstruction *>(&inst) ||
      dynamic_cast<const rocisa::FLATStoreInstruction *>(&inst))
    throw std::invalid_argument("flat memory instructions count on both vmcnt "
                                "and lgkmcnt and are not tracked");
  return std::nullopt;
}

WaitKind classify(const rocisa::Instruction &inst) {
  // Scalar memory classes derive from the global ones, so they go first.
  if (std::optional<WaitKind> kind = classifyScalarOrFlat(inst))
    return *kind;
  if (dynamic_cast<const rocisa::MUBUFReadInstruction *>(&inst) ||
      dynamic_cast<const rocisa::MUBUFStoreInstruction *>(&inst) ||
      dynamic_cast<const rocisa::GLOBALLoadInstruction *>(&inst) ||
      dynamic_cast<const rocisa::GLOBALStoreInstruction *>(&inst))
    return WaitKind::vectorMemory;
  if (dynamic_cast<const rocisa::LocalReadInstruction *>(&inst) ||
      dynamic_cast<const rocisa::LocalWriteInstruction *>(&inst))
    return WaitKind::lds;
  throw std::invalid_argument("not a memory instruction the wait tracker "
                              "orders");
}

void lowerTo(std::optional<int> &count, int value) {
  count = count ? std::min(*count, value) : value;
}

void requireSameSize(const char *name, const Window &entry,
                     const Window &backEdge) {
  if (entry.size() == backEdge.size())
    return;
  throw std::logic_error(std::string("unbalanced loop: ") + name + " has " +
                         std::to_string(backEdge.size()) +
                         " outstanding operations at the back edge and " +
                         std::to_string(entry.size()) + " at entry");
}

} // namespace

WaitTracker::WaitTracker(IRBuilder &builder)
    : builder(&builder), maxVmcnt(builder.target().maxVmcnt()),
      maxLgkmcnt(builder.target().maxLgkmcnt()) {
  if (builder.target().isaVersion()[0] != 9)
    throw std::invalid_argument("wait tracker supports only gfx9 targets");
}

WaitToken WaitTracker::issue(const std::shared_ptr<rocisa::Instruction> &inst) {
  if (!inst)
    throw std::invalid_argument("null instruction");
  WaitKind kind = classify(*inst);
  int id = nextId++;
  switch (kind) {
  case WaitKind::vectorMemory:
    current.vectorMemory.push_back({id});
    break;
  case WaitKind::lds:
    current.lds.push_back({id});
    break;
  case WaitKind::scalarMemory:
    current.scalarMemory.push_back(id);
    break;
  }
  return {kind, id};
}

void WaitTracker::waitFor(WaitToken token) {
  waitFor(std::span<const WaitToken>(&token, 1));
}

void WaitTracker::waitFor(std::span<const WaitToken> tokens) {
  std::optional<int> vmcnt;
  std::optional<int> lgkmcnt;
  for (WaitToken token : tokens) {
    if (token.id < 0 || token.id >= nextId)
      throw std::invalid_argument("wait token was not issued by this tracker");
    if (token.kind == WaitKind::vectorMemory) {
      if (std::optional<int> d = distanceOf(current.vectorMemory, token.id))
        lowerTo(vmcnt, *d);
    } else if (token.kind == WaitKind::lds) {
      if (std::optional<int> d = distanceOf(current.lds, token.id))
        lowerTo(lgkmcnt, *d);
    } else if (contains(current.scalarMemory, token.id)) {
      lowerTo(lgkmcnt, 0);
    }
  }
  if (!vmcnt && !lgkmcnt)
    return;
  emitWait(vmcnt, lgkmcnt);
}

void WaitTracker::waitAll(WaitKind kind) {
  switch (kind) {
  case WaitKind::vectorMemory:
    if (!current.vectorMemory.empty())
      emitWait(0, std::nullopt);
    return;
  case WaitKind::lds:
    if (!current.lds.empty())
      emitWait(std::nullopt, 0);
    return;
  case WaitKind::scalarMemory:
    if (!current.scalarMemory.empty())
      emitWait(std::nullopt, 0);
    return;
  }
}

void WaitTracker::emitWait(std::optional<int> vmcnt,
                           std::optional<int> lgkmcnt) {
  // A capped count waits for more operations, so it still guarantees the token.
  if (vmcnt)
    vmcnt = std::min(*vmcnt, maxVmcnt);
  if (lgkmcnt)
    lgkmcnt = std::min(*lgkmcnt, maxLgkmcnt);
  builder->createWaitCnt({.vectorLoads = vmcnt, .lds = lgkmcnt});
  if (vmcnt)
    completeFrom(current.vectorMemory, *vmcnt);
  if (!lgkmcnt)
    return;
  completeFrom(current.lds, *lgkmcnt);
  if (*lgkmcnt == 0)
    current.scalarMemory.clear();
}

void WaitTracker::join(const State &other) {
  current.vectorMemory = joinWindows(current.vectorMemory, other.vectorMemory);
  current.lds = joinWindows(current.lds, other.lds);
  appendUnique(current.scalarMemory, other.scalarMemory);
}

void WaitTracker::beginLoop() { loopEntries.push_back(current); }

void WaitTracker::endLoop() {
  if (loopEntries.empty())
    throw std::logic_error("end of a loop that was not begun");
  // Pop first so the tracker stays usable if the check throws.
  State entry = std::move(loopEntries.back());
  loopEntries.pop_back();
  requireSameSize("vector memory", entry.vectorMemory, current.vectorMemory);
  requireSameSize("lds", entry.lds, current.lds);
  for (int id : current.scalarMemory)
    if (!contains(entry.scalarMemory, id))
      throw std::logic_error("unbalanced loop: scalar memory loads issued in "
                             "the body are outstanding at the back edge");
}
