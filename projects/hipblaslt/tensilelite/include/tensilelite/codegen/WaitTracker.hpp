// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_CODEGEN_WAITTRACKER_HPP
#define ROCM_TENSILELITE_CODEGEN_WAITTRACKER_HPP

#include <tensilelite/codegen/IRBuilder.hpp>

#include "instruction/common.hpp"

#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocm::tensilelite::codegen {

/// The kinds of memory operation a WaitTracker orders.
enum class WaitKind {
  /// Buffer and global loads and stores, counted by vmcnt in issue order.
  vectorMemory,
  /// LDS operations, counted by lgkmcnt in issue order.
  lds,
  /// Scalar memory loads, counted by lgkmcnt out of order.
  scalarMemory,
};

/// Handle to one issued operation, returned by WaitTracker::issue. A
/// default-constructed token is invalid.
struct WaitToken {
  WaitKind kind = WaitKind::vectorMemory;
  int id = -1;
};

/// Emits s_waitcnt instructions in program order, one wait per use, the way
/// the MLIR reference does with its wait tokens. The emitter issues an
/// operation, keeps its token and waits for the token right before the use;
/// the tracker derives the counts, so they are right by construction and
/// waits that earlier ones already imply are elided.
///
/// Per counter the tracker keeps a window of positions, oldest first, each
/// holding the id of the operation there (several ids after a join). The
/// distance of a position is the number of later operations of that counter,
/// and waiting for count N completes every position at distance >= N, because
/// operations of one counter complete in order. A token waits for its newest
/// position. Scalar memory loads complete out of order and are ignored when
/// counting LDS distances; only lgkmcnt(0) guarantees one.
///
/// A loop body is emitted once, between beginLoop and endLoop, with waits
/// computed from the entry state. Tokens are positions: at the back edge the
/// operation at distance d takes the role of the one at distance d at entry.
/// So a ring that waits for slots[c] and then re-issues slots[c] gets counts
/// that hold in every iteration, and endLoop requires equal window sizes at
/// entry and at the back edge to guarantee that. An operation issued before
/// the loop and not re-issued is waited for conservatively. Scalar memory
/// loads issued in a body must be waited for in it.
///
/// After endLoop the state is the back-edge state. If the loop can run zero
/// times, the emitter must join the pre-loop snapshot at the exit label.
/// Waits emitted outside the tracker are not seen, so the tracker then only
/// over-waits. Only gfx9 targets are supported.
class WaitTracker {
public:
  /// Possibly outstanding operations at a program point.
  class State {
    friend class WaitTracker;
    // Oldest first; a position holds the ids of every op that may be there,
    // more than one after a join.
    std::vector<std::vector<int>> vectorMemory;
    std::vector<std::vector<int>> lds;
    // Unordered, so only their ids matter.
    std::vector<int> scalarMemory;
  };

  /// Emits waits through builder, which must outlive the tracker. Throws
  /// std::invalid_argument unless the target is gfx9.
  explicit WaitTracker(IRBuilder &builder);

  /// Records inst, just created at the insertion point, and returns its token.
  /// Throws std::invalid_argument if inst is null or is not a buffer, global,
  /// LDS or scalar memory instruction (FLAT is rejected, it counts on both
  /// counters).
  WaitToken issue(const std::shared_ptr<rocisa::Instruction> &inst);

  /// Emits one s_waitcnt with the largest counts that guarantee every token
  /// completed, or nothing if earlier waits already guarantee them. Throws
  /// std::invalid_argument for a token this tracker did not issue.
  void waitFor(WaitToken token);
  void waitFor(std::span<const WaitToken> tokens);
  /// Waits for every outstanding operation of kind (lds and scalarMemory both
  /// emit lgkmcnt(0)).
  void waitAll(WaitKind kind);

  State snapshot() const { return current; }
  void restore(State state) { current = std::move(state); }
  /// Merges other, the state at the end of another path into this point,
  /// into the current state conservatively.
  void join(const State &other);

  /// Marks the start of a loop body (records the entry state).
  void beginLoop();
  /// Marks the back edge; throws std::logic_error unless the body is balanced
  /// (see the class comment) or no loop is open.
  void endLoop();

private:
  void emitWait(std::optional<int> vmcnt, std::optional<int> lgkmcnt);

  IRBuilder *builder;
  int maxVmcnt;
  int maxLgkmcnt;
  int nextId = 0;
  State current;
  std::vector<State> loopEntries;
};

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_WAITTRACKER_HPP
