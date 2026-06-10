// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Pure C++ port of the subtile InstructionScheduler slot-placement algorithm
// (Tensile/Components/Subtile/InstructionScheduler.py).
//
// This header is intentionally free of any nanobind / Python / rocisa
// dependency: it operates on a *data-only* typed model of the emitted modules
// (instruction "kinds" plus the waitcnt fields the vmcnt post-pass needs) and
// returns the final emission *order* as a list of (moduleIndex, instIdx)
// references. A Python shim maps those references back onto the live rocisa
// instruction objects and rebuilds the rocisa Module, so no rocisa object ever
// crosses into C++.
//
// SCOPE: only the instruction-scheduling slot-placement algorithm itself
// (dependency-path extraction, slot placement with the four validators / the
// buffer-load spreading adjuster, assembly order, and the vmcnt post-pass) is
// ported. No LogicalScheduler pass logic, InstructionEmitter dispatch, or
// Kernel.mainLoop behaviour lives here.
//
// The names below mirror the Python module so the two implementations can be
// reasoned about side by side:
//   extractPathsFromBeforeDeps -> extractPathsFromBeforeDeps
//   _classifyPaths             -> classifyPaths
//   _SlotPlacer / _SchedulingRules -> Scheduler (merged)
//   instructionSchedule        -> schedule

#pragma once

#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tw::subtile::insched {

// Instruction classification. Mirrors the isinstance() predicates the Python
// scheduler keys on. Anything the Python code does not special-case (a plain
// CommonInstruction, an MFMA living in a non-MFMA module, etc.) is `Other`.
enum class InstKind {
  Mfma,        // MFMAInstruction or MXMFMAInstruction
  LocalRead,   // LocalReadInstruction      (_isDsRead)
  GlobalRead,  // GlobalReadInstruction     (_isBufferLoad)
  WaitCnt,     // SWaitCnt                  (_isWaitCnt)
  M0Update,    // CommonInstruction with dst.regType == 'm' (_isM0Update)
  Other,
};

// Data-only view of one rocisa instruction. `vlcnt` / `adjustVmcnt` are only
// meaningful for WaitCnt instructions (they drive the vmcnt post-pass); for
// everything else they take the harmless defaults below.
struct InstRef {
  InstKind kind = InstKind::Other;
  long vlcnt = -1;
  bool adjustVmcnt = true;

  InstRef() = default;
  InstRef(InstKind k, long v = -1, bool a = true)
      : kind(k), vlcnt(v), adjustVmcnt(a) {}
};

// Data-only view of one LogicalScheduler.EmittedModule.
//   moduleId   — stable id used to resolve `before` links (NOT the list index)
//   opType     — module op-type string ("mfma", "wait_gr", ...)
//   before     — moduleId this module depends on (std::nullopt == None)
struct ModuleRef {
  int moduleId = -1;
  std::string opType;
  std::optional<int> before;
  std::vector<InstRef> instructions;

  ModuleRef() = default;
  ModuleRef(int id, std::string op, std::optional<int> bef,
            std::vector<InstRef> insts)
      : moduleId(id),
        opType(std::move(op)),
        before(bef),
        instructions(std::move(insts)) {}
};

// A reference into the input model: (index into the modules vector, index into
// that module's instructions vector). Uniquely identifies one instruction.
using InstId = std::pair<int, int>;

struct ScheduleResult {
  // Final emission order, as (moduleIndex, instIdx) references.
  std::vector<InstId> order;
  // Parallel to `order`: the kind of each emitted instruction (test convenience).
  std::vector<InstKind> kinds;
  // Parallel to `order`: the (post-pass adjusted) vlcnt of each emitted
  // instruction. For non-waitcnt instructions this is the input vlcnt (-1 by
  // default); for waitcnt instructions it reflects the vmcnt post-pass.
  std::vector<long> vlcnt;
  // (orderIndex, delta) for every waitcnt the post-pass actually bumped. The
  // shim applies `liveInst.vlcnt += delta`, exactly mirroring the Python
  // `inst.vlcnt += bufLoadCount`.
  std::vector<std::pair<int, long>> vmcntAdjustments;
};

// Result of dependency-path extraction. Paths are lists of *module indices*.
struct ExtractedPaths {
  int mfmaIdx = -1;
  std::vector<std::vector<int>> paths;         // regular (interleaved) paths
  std::vector<std::vector<int>> preMfmaPaths;  // emitted before the first MFMA
};

namespace detail {

// Python floor-division (rounds toward -inf), used by the buffer-load spreader.
inline long floordiv(long a, long b) {
  long q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

inline bool isDsRead(InstKind k) { return k == InstKind::LocalRead; }
inline bool isBufferLoad(InstKind k) { return k == InstKind::GlobalRead; }
inline bool isWaitCnt(InstKind k) { return k == InstKind::WaitCnt; }
inline bool isM0Update(InstKind k) { return k == InstKind::M0Update; }
inline bool isMFMA(InstKind k) { return k == InstKind::Mfma; }

// One placed item: the dependency module index `mid`, the source instruction
// reference, plus its kind / vmcnt fields (so validators need not chase the
// model during placement).
struct Item {
  int mid;        // module index (dependency node)
  int instIdx;    // instruction index within that module
  InstKind kind;
  long vlcnt;
  bool adjustVmcnt;
};

// Hardcoded gap to hide ds_read latency (mirrors Python's
// _MIN_MFMA_GAP_DS_READ_TO_WAIT). TODO: compute this more accurately.
constexpr int kMinMfmaGapDsReadToWait = 4;

// Merged port of _SlotPlacer + _SchedulingRules. The Python code injects the
// rules into the placer via callback lists; here the (fixed) rule set used by
// instructionSchedule is inlined, preserving the exact validator order.
class Scheduler {
 public:
  Scheduler(int intervals, int numModules,
            const std::vector<std::vector<int>>& pathOrders)
      : totalSlots_(intervals * 2),
        n_(numModules),
        prevInPath_(numModules, -1),
        nextInPath_(numModules, -1),
        placed_(static_cast<size_t>(intervals * 2)),
        firstPos_(numModules, std::nullopt),
        lastPos_(numModules, std::nullopt),
        earliestWaitCntPos_(intervals * 2) {
    for (const auto& order : pathOrders) {
      for (size_t i = 0; i + 1 < order.size(); ++i) {
        int a = order[i], b = order[i + 1];
        prevInPath_[b] = a;
        nextInPath_[a] = b;
      }
    }
  }

  int totalSlots() const { return totalSlots_; }

  // ── Per-path rule state (mirrors _SchedulingRules.resetPath) ──
  void resetPath() {
    firstBufLoadPos_ = std::nullopt;
    bufLoadIdx_ = 0;
    bufLoadMaxSlot_ = 0;
    numBufLoads_ = 0;
  }

  // ── _SchedulingRules.setupBufLoadSpreading ──
  void setupBufLoadSpreading(const std::vector<Item>& pathInsts,
                             const std::vector<int>& order) {
    numBufLoads_ = 0;
    for (const auto& it : pathInsts)
      if (isBufferLoad(it.kind)) ++numBufLoads_;
    if (numBufLoads_ > 1) {
      auto [lo, rawMax] = bounds(pathInsts.back().mid);
      (void)lo;
      std::set<int> grModuleIds;
      for (const auto& it : pathInsts)
        if (isBufferLoad(it.kind)) grModuleIds.insert(it.mid);
      // lastGrIdx = max(order.index(m) for m in grModuleIds if m in order)
      int lastGrIdx = -1;
      for (int m : grModuleIds) {
        for (size_t idx = 0; idx < order.size(); ++idx) {
          if (order[idx] == m) {
            if (static_cast<int>(idx) > lastGrIdx) lastGrIdx = static_cast<int>(idx);
            break;  // first occurrence, like list.index
          }
        }
      }
      std::set<int> tailModuleIds;
      for (size_t idx = static_cast<size_t>(lastGrIdx + 1); idx < order.size(); ++idx)
        tailModuleIds.insert(order[idx]);
      long numTailInsts = 0;
      for (const auto& it : pathInsts)
        if (tailModuleIds.count(it.mid)) ++numTailInsts;
      long v = rawMax - numTailInsts;
      bufLoadMaxSlot_ = v > 0 ? v : 0;
    }
  }

  // ── _SlotPlacer.placePath ──
  void placePath(const std::vector<Item>& pathInsts, bool reverse) {
    long limit = reverse ? (totalSlots_ - 1) : 0;
    for (const auto& item : pathInsts) {
      if (!reverse) limit = adjustLimit(limit, item);
      std::optional<int> pos = findSlot(item.mid, item, limit, reverse);
      int chosen = pos.has_value() ? *pos : forceSlot(item.mid, limit, reverse);
      place(chosen, item, reverse);
      limit = reverse ? (chosen - 1) : (chosen + 1);
    }
  }

  // ── _SlotPlacer.assemble (order only) ──
  // Appends the interleaved MFMA + placed-slot order into `out`.
  // `mfmas` holds (moduleIndex, instIdx) references for the MFMA instructions.
  void assembleInto(const std::vector<InstId>& mfmas, std::vector<InstId>& out,
                    std::vector<InstKind>& kinds, std::vector<long>& vlcnts,
                    const std::vector<ModuleRef>& modules) const {
    int intervals = static_cast<int>(mfmas.size()) - 1;
    auto pushRef = [&](const InstId& ref) {
      out.push_back(ref);
      const InstRef& ir = modules[ref.first].instructions[ref.second];
      kinds.push_back(ir.kind);
      vlcnts.push_back(ir.vlcnt);
    };
    pushRef(mfmas[0]);
    for (int i = 0; i < intervals; ++i) {
      for (int slot : {2 * i, 2 * i + 1}) {
        for (const auto& item : placed_[slot]) pushRef({item.mid, item.instIdx});
      }
      pushRef(mfmas[i + 1]);
    }
    // leftovers_ is always empty in this algorithm (placePath force-places),
    // but mirror the Python tail loop for fidelity.
    for (const auto& item : leftovers_) pushRef({item.mid, item.instIdx});
  }

 private:
  // ── _SlotPlacer.bounds ──
  std::pair<long, long> bounds(int mid) const {
    long lo = 0;
    int pred = prevInPath_[mid];
    if (pred >= 0 && pred < n_ && lastPos_[pred].has_value())
      lo = *lastPos_[pred] + 1;
    long hi = totalSlots_ - 1;
    int succ = nextInPath_[mid];
    if (succ >= 0 && succ < n_ && firstPos_[succ].has_value())
      hi = *firstPos_[succ] - 1;
    return {lo, hi};
  }

  // ── _SlotPlacer._canPlace ──
  bool canPlace(long pos, const Item& inst) const {
    if (pos < 0 || pos >= totalSlots_ ||
        placed_[static_cast<size_t>(pos)].size() >= 2)
      return false;
    // validators, in the exact order instructionSchedule registers them.
    return oneDsReadPerInterval(pos, inst) && minGapDsReadBeforeWait(pos, inst) &&
           minGapDsReadToWait(pos, inst) && noM0WithBufferLoad(pos, inst);
  }

  // ── _SlotPlacer.findSlot ──
  std::optional<int> findSlot(int mid, const Item& inst, long limit,
                              bool reverse) const {
    auto [lo, hi] = bounds(mid);
    if (reverse)
      hi = hi < limit ? hi : limit;
    else
      lo = lo > limit ? lo : limit;
    if (hi < lo) return std::nullopt;
    if (reverse) {
      for (long pos = hi; pos >= lo; --pos)
        if (canPlace(pos, inst)) return static_cast<int>(pos);
    } else {
      for (long pos = lo; pos <= hi; ++pos)
        if (canPlace(pos, inst)) return static_cast<int>(pos);
    }
    return std::nullopt;
  }

  // ── _SlotPlacer._forceSlot ──
  int forceSlot(int mid, long limit, bool reverse) const {
    auto [lo, hi] = bounds(mid);
    if (reverse) {
      hi = hi < limit ? hi : limit;
      lo = lo > 0 ? lo : 0;
      if (hi < lo) hi = lo;
      return static_cast<int>(hi);
    }
    lo = lo > limit ? lo : limit;
    hi = hi < (totalSlots_ - 1) ? hi : (totalSlots_ - 1);
    if (lo > hi) lo = hi;
    return static_cast<int>(lo);
  }

  // ── _SlotPlacer.place + _SchedulingRules.trackPlacement (onPlace) ──
  void place(int pos, const Item& item, bool reverse) {
    auto& slot = placed_[static_cast<size_t>(pos)];
    if (reverse)
      slot.insert(slot.begin(), item);
    else
      slot.push_back(item);
    int mid = item.mid;
    if (!firstPos_[mid].has_value() || pos < *firstPos_[mid]) firstPos_[mid] = pos;
    if (!lastPos_[mid].has_value() || pos > *lastPos_[mid]) lastPos_[mid] = pos;
    trackPlacement(pos, item);
  }

  // ── _SlotPlacer.adjustLimit -> spreadBufferLoads (the only adjuster) ──
  long adjustLimit(long limit, const Item& inst) {
    if (!isBufferLoad(inst.kind) || bufLoadMaxSlot_ <= 0) return limit;
    if (firstBufLoadPos_.has_value()) {
      long span = bufLoadMaxSlot_ - *firstBufLoadPos_;
      long stride = floordiv(span, numBufLoads_);
      if (stride < 1) stride = 1;
      long cand = *firstBufLoadPos_ + static_cast<long>(bufLoadIdx_) * stride;
      if (cand > limit) limit = cand;
    }
    ++bufLoadIdx_;
    return limit;
  }

  // ── Validators (return true == slot acceptable) ──
  bool oneDsReadPerInterval(long pos, const Item& inst) const {
    if (!isDsRead(inst.kind)) return true;
    long peer = pos ^ 1;
    if (peer < 0 || peer >= totalSlots_) return true;
    for (const auto& it : placed_[static_cast<size_t>(peer)])
      if (isDsRead(it.kind)) return false;
    return true;
  }

  bool minGapDsReadBeforeWait(long pos, const Item& inst) const {
    if (!isDsRead(inst.kind)) return true;
    long gap = kMinMfmaGapDsReadToWait * 2;
    return earliestWaitCntPos_ - pos >= gap;
  }

  bool minGapDsReadToWait(long pos, const Item& inst) const {
    if (!isWaitCnt(inst.kind) || lastDsReadPos_ < 0) return true;
    long gap = kMinMfmaGapDsReadToWait * 2;
    return pos - lastDsReadPos_ >= gap;
  }

  bool noM0WithBufferLoad(long pos, const Item& inst) const {
    bool m0 = isM0Update(inst.kind);
    bool buf = isBufferLoad(inst.kind);
    if (!m0 && !buf) return true;
    long peer = pos ^ 1;
    std::vector<long> slots{pos};
    if (peer >= 0 && peer < totalSlots_) slots.push_back(peer);
    if (m0) {
      for (long s : slots)
        for (const auto& it : placed_[static_cast<size_t>(s)])
          if (isBufferLoad(it.kind)) return false;
      return true;
    }
    for (long s : slots)
      for (const auto& it : placed_[static_cast<size_t>(s)])
        if (isM0Update(it.kind)) return false;
    return true;
  }

  // ── _SchedulingRules.trackPlacement ──
  void trackPlacement(int pos, const Item& inst) {
    if (isDsRead(inst.kind) && pos > lastDsReadPos_) lastDsReadPos_ = pos;
    if (isWaitCnt(inst.kind) && pos < earliestWaitCntPos_) earliestWaitCntPos_ = pos;
    if (isBufferLoad(inst.kind) && !firstBufLoadPos_.has_value())
      firstBufLoadPos_ = pos;
  }

  int totalSlots_;
  int n_;
  std::vector<int> prevInPath_;
  std::vector<int> nextInPath_;
  std::vector<std::vector<Item>> placed_;
  std::vector<std::optional<int>> firstPos_;
  std::vector<std::optional<int>> lastPos_;
  std::vector<Item> leftovers_;

  // Cross-path rule state.
  long lastDsReadPos_ = -1;
  long earliestWaitCntPos_;
  // Per-path rule state.
  std::optional<long> firstBufLoadPos_;
  long bufLoadIdx_ = 0;
  long bufLoadMaxSlot_ = 0;
  long numBufLoads_ = 0;
};

}  // namespace detail

// ── extractPathsFromBeforeDeps ──
inline ExtractedPaths extractPathsFromBeforeDeps(
    const std::vector<ModuleRef>& modules) {
  const int n = static_cast<int>(modules.size());
  std::unordered_map<int, int> idToIdx;
  for (int i = 0; i < n; ++i) idToIdx[modules[i].moduleId] = i;

  std::vector<int> mfmaModuleIds;
  for (int i = 0; i < n; ++i)
    if (modules[i].opType == "mfma") mfmaModuleIds.push_back(i);
  if (mfmaModuleIds.size() != 1)
    throw std::invalid_argument(
        "extractPathsFromBeforeDeps expects exactly one MFMA emitted module");
  const int mfmaIdx = mfmaModuleIds[0];

  std::vector<int> nonMfmaIds;
  nonMfmaIds.reserve(n);
  for (int i = 0; i < n; ++i)
    if (i != mfmaIdx) nonMfmaIds.push_back(i);
  std::vector<char> nonMfmaSet(n, 0);
  for (int i : nonMfmaIds) nonMfmaSet[i] = 1;

  // Identify the non-MFMA module the MFMA depends on (if any).
  int preMfmaTarget = -1;
  if (modules[mfmaIdx].before.has_value()) {
    auto it = idToIdx.find(*modules[mfmaIdx].before);
    if (it != idToIdx.end() && nonMfmaSet[it->second]) preMfmaTarget = it->second;
  }

  std::vector<int> pred(n, -1), child(n, -1);
  for (int i : nonMfmaIds) {
    int parent = -1;
    if (modules[i].before.has_value()) {
      auto it = idToIdx.find(*modules[i].before);
      if (it != idToIdx.end() && it->second != i && nonMfmaSet[it->second])
        parent = it->second;
    }
    pred[i] = parent;
    if (parent != -1) {
      if (child[parent] != -1)
        throw std::invalid_argument(
            "extractPathsFromBeforeDeps expects unique child per predecessor");
      child[parent] = i;
    }
  }

  auto findHead = [&](int mid) {
    int cur = mid;
    std::vector<char> seen(n, 0);
    while (pred[cur] != -1 && !seen[cur]) {
      seen[cur] = 1;
      cur = pred[cur];
    }
    return cur;
  };

  auto walkFromHead = [&](int head, const std::vector<char>& used) {
    std::vector<int> order;
    std::vector<char> localSeen(n, 0);
    int cur = head;
    while (cur != -1 && !used[cur] && !localSeen[cur]) {
      order.push_back(cur);
      localSeen[cur] = 1;
      cur = child[cur];
    }
    return order;
  };

  std::vector<char> used(n, 0);
  std::vector<std::vector<int>> paths;
  for (int mid : nonMfmaIds) {
    if (used[mid]) continue;
    int head = findHead(mid);
    std::vector<int> order = walkFromHead(head, used);
    if (order.empty())
      throw std::invalid_argument(
          "extractPathsFromBeforeDeps produced empty path");
    for (int i : order) used[i] = 1;
    paths.push_back(std::move(order));
  }

  ExtractedPaths result;
  result.mfmaIdx = mfmaIdx;
  for (auto& path : paths) {
    bool inPre = false;
    if (preMfmaTarget != -1) {
      for (int m : path)
        if (m == preMfmaTarget) {
          inPre = true;
          break;
        }
    }
    if (inPre)
      result.preMfmaPaths.push_back(std::move(path));
    else
      result.paths.push_back(std::move(path));
  }
  return result;
}

// ── _classifyPaths ──
// Returns (path, hasWaitGR) pairs sorted: wait_gr paths first, then by the
// path's first module index. Uses a stable sort to match Python's list.sort.
inline std::vector<std::pair<std::vector<int>, bool>> classifyPaths(
    const std::vector<std::vector<int>>& pathOrders,
    const std::vector<ModuleRef>& modules) {
  std::vector<std::pair<std::vector<int>, bool>> paths;
  paths.reserve(pathOrders.size());
  for (const auto& order : pathOrders) {
    bool hasWaitGR = false;
    for (int i : order)
      if (modules[i].opType == "wait_gr") {
        hasWaitGR = true;
        break;
      }
    paths.emplace_back(order, hasWaitGR);
  }
  std::stable_sort(paths.begin(), paths.end(), [](const auto& a, const auto& b) {
    long ka0 = a.second ? 0 : 1;
    long kb0 = b.second ? 0 : 1;
    if (ka0 != kb0) return ka0 < kb0;
    long ka1 = a.first.empty() ? 1000000000L : a.first.front();
    long kb1 = b.first.empty() ? 1000000000L : b.first.front();
    return ka1 < kb1;
  });
  return paths;
}

namespace detail {

// _flattenPath: walk a path of module indices into Items in instruction order.
inline std::vector<Item> flattenPath(const std::vector<int>& order,
                                     const std::vector<ModuleRef>& modules,
                                     bool reverse) {
  std::vector<Item> out;
  for (int mid : order) {
    const auto& insts = modules[mid].instructions;
    for (int j = 0; j < static_cast<int>(insts.size()); ++j) {
      const InstRef& ir = insts[j];
      out.push_back(Item{mid, j, ir.kind, ir.vlcnt, ir.adjustVmcnt});
    }
  }
  if (reverse) std::reverse(out.begin(), out.end());
  return out;
}

// Apply the vmcnt post-pass over an already-built emission order, recording
// the adjusted vlcnt values and the per-waitcnt deltas.
inline void applyVmcntPostPass(const std::vector<ModuleRef>& modules,
                               ScheduleResult& result) {
  long bufLoadCount = 0;
  for (size_t i = 0; i < result.order.size(); ++i) {
    const InstId& ref = result.order[i];
    const InstRef& ir = modules[ref.first].instructions[ref.second];
    if (isBufferLoad(ir.kind)) {
      ++bufLoadCount;
    } else if (isWaitCnt(ir.kind) && ir.vlcnt >= 0) {
      if (ir.adjustVmcnt) {
        result.vlcnt[i] = ir.vlcnt + bufLoadCount;
        result.vmcntAdjustments.emplace_back(static_cast<int>(i), bufLoadCount);
      }
    }
  }
}

}  // namespace detail

// ── instructionSchedule ──
// Returns the final emission order (and the vmcnt post-pass result) for an
// emitted-module chain. Throws std::invalid_argument when the chain violates a
// structural precondition (e.g. not exactly one MFMA module).
inline ScheduleResult schedule(const std::vector<ModuleRef>& modules) {
  ScheduleResult result;
  if (modules.empty()) return result;

  const int n = static_cast<int>(modules.size());
  ExtractedPaths ex = extractPathsFromBeforeDeps(modules);

  // mfmas = [x for x in modules[mfmaIdx].instructions if isMFMA(x)]
  std::vector<InstId> mfmas;
  {
    const auto& insts = modules[ex.mfmaIdx].instructions;
    for (int j = 0; j < static_cast<int>(insts.size()); ++j)
      if (detail::isMFMA(insts[j].kind)) mfmas.emplace_back(ex.mfmaIdx, j);
  }

  auto pushRef = [&](const InstId& ref) {
    result.order.push_back(ref);
    const InstRef& ir = modules[ref.first].instructions[ref.second];
    result.kinds.push_back(ir.kind);
    result.vlcnt.push_back(ir.vlcnt);
  };

  auto emitPreMfma = [&]() {
    for (const auto& order : ex.preMfmaPaths)
      for (int mid : order)
        for (int j = 0; j < static_cast<int>(modules[mid].instructions.size()); ++j)
          pushRef({mid, j});
  };

  // Single MFMA: no slots to interleave into — preMfma, MFMA(s), then paths
  // (in pathOrders order, NOT the classified order).
  if (mfmas.size() < 2) {
    emitPreMfma();
    for (const auto& ref : mfmas) pushRef(ref);
    for (const auto& order : ex.paths)
      for (int mid : order)
        for (int j = 0; j < static_cast<int>(modules[mid].instructions.size()); ++j)
          pushRef({mid, j});
    detail::applyVmcntPostPass(modules, result);
    return result;
  }

  auto classified = classifyPaths(ex.paths, modules);
  detail::Scheduler placer(static_cast<int>(mfmas.size()) - 1, n, ex.paths);

  for (const auto& [order, hasWaitGR] : classified) {
    if (order.empty()) continue;
    std::vector<detail::Item> pathInsts =
        detail::flattenPath(order, modules, /*reverse=*/hasWaitGR);
    placer.resetPath();
    if (!hasWaitGR) placer.setupBufLoadSpreading(pathInsts, order);
    placer.placePath(pathInsts, /*reverse=*/hasWaitGR);
  }

  emitPreMfma();
  placer.assembleInto(mfmas, result.order, result.kinds, result.vlcnt, modules);

  detail::applyVmcntPostPass(modules, result);
  return result;
}

}  // namespace tw::subtile::insched
