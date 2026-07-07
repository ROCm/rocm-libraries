/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// origami::comm — analytical communication cost model
//
// Communication primitives — the verbs a collective is built from
// (load/store/pull/push/reduce/signal/wait).
//
// First principle: a byte's cost is determined by *which stages of the memory
// hierarchy it must physically pass through*. Every primitive's resolve()
// walks that path and bills one cache line to each stage it crosses, for one
// software-pipelined iteration (cl_per_iter lines). The hierarchy, outermost
// to the registers, is:
//
//     HBM ── MALL(Infinity Cache) ── L2/TCC ── TCP/vL1D ── registers
//                                                   xGMI ┘ (to/from a peer GPU)
//
// A local load touches every level on the way in; a remote pull arrives over
// xGMI into L2 and never touches local MALL/HBM; a push reads locally then
// pays an xGMI egress write. Because latency.hpp later takes the max over
// stages, what matters is not the byte count but the *set* of stages each
// primitive lights up — that set is exactly what these structs encode.
#pragma once

#include "origami/comm/types.hpp"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace origami::comm {

/**
 * @brief The sizing of one software-pipelined iteration.
 *
 * The sizing of one software-pipelined iteration: how much data it moves and
 * the instruction density to move it. Each primitive's resolve() multiplies
 * these against its functional-unit pattern to produce the iteration's work.
 * Bundled so the dimensions travel as one named value, not 3 positional args.
 */
struct iter_dims_t {
  int cl_per_iter;        ///< cache lines moved per iteration
  int instrs_per_cl;      ///< VMEM instructions to move one cache line (load-width
                          ///< dependent; note: per cache line, not per iteration)
  int elements_per_iter;  ///< elements reduced per iteration (VALU term only)
};

// ─── Primitives ──────────────────────────────────────────────────
/**
 * @brief Read from local HBM into registers.
 *
 * Read from local HBM into registers. A cold load misses at every level, so
 * the line is charged at HBM, MALL, L2 and TCP (the read ripples up the whole
 * hierarchy), plus the VMEM issue slots to actually move it
 * (cl_per_iter × instrs_per_cl, where instrs_per_cl depends on load width).
 */
struct load_t {
  /**
   * @brief Resolve this load into functional-unit work for one iteration.
   *
   * @param iter Sizing of one software-pipelined iteration.
   * @return functional_unit_work_t Work charged across the local memory hierarchy.
   */
  constexpr functional_unit_work_t resolve(const iter_dims_t& iter) const noexcept {
    functional_unit_work_t w{};
    w.vmem_read_instrs = static_cast<std::int64_t>(iter.cl_per_iter) * iter.instrs_per_cl;
    w.tcp_read_cl      = iter.cl_per_iter;
    w.l2_read_cl       = iter.cl_per_iter;
    w.mall_read_cl     = iter.cl_per_iter;
    w.hbm_read_cl      = iter.cl_per_iter;
    return w;
  }
};

/**
 * @brief Write from registers to local HBM.
 *
 * Write from registers to local HBM. The write drains down the hierarchy
 * (TCP → L2 → MALL → HBM). `write_through=true` models the CDNA write-through
 * + atomic-release path used for inter-rank visibility: the line is pushed
 * straight past L2 (no allocate/writeback there), so the L2 line is not
 * charged — the data must reach a coherence point, not linger cached.
 */
struct store_t {
  bool write_through = false;  ///< Use the write-through + atomic-release path (skip L2 caching).

  /**
   * @brief Resolve this store into functional-unit work for one iteration.
   *
   * @param iter Sizing of one software-pipelined iteration.
   * @return functional_unit_work_t Work charged down the local memory hierarchy.
   */
  constexpr functional_unit_work_t resolve(const iter_dims_t& iter) const noexcept {
    functional_unit_work_t w{};
    w.vmem_write_instrs = static_cast<std::int64_t>(iter.cl_per_iter) * iter.instrs_per_cl;
    w.tcp_write_cl      = iter.cl_per_iter;
    w.l2_write_cl       = write_through ? 0 : static_cast<std::int64_t>(iter.cl_per_iter);
    w.mall_write_cl     = iter.cl_per_iter;
    w.hbm_write_cl      = iter.cl_per_iter;
    return w;
  }
};

/**
 * @brief Read from a remote GPU's HBM via xGMI (ingress).
 *
 * Read from a remote GPU's HBM via xGMI (ingress). The bytes originate on the
 * peer, so locally they never pass through *our* MALL or HBM — they enter over
 * the fabric and land in L2/TCP. Hence the charge is xGMI + L2 + TCP only; the
 * xGMI line is what makes this primitive latency-bound on the 660 ns RTT (see
 * the MSHR cap in latency.hpp) rather than on local memory bandwidth.
 */
struct pull_t {
  int peer = 0;  ///< Remote GPU rank the data is read from.

  /**
   * @brief Resolve this remote read into functional-unit work for one iteration.
   *
   * @param iter Sizing of one software-pipelined iteration.
   * @return functional_unit_work_t Work charged on xGMI, L2, and TCP.
   */
  constexpr functional_unit_work_t resolve(const iter_dims_t& iter) const noexcept {
    functional_unit_work_t w{};
    w.vmem_read_instrs = static_cast<std::int64_t>(iter.cl_per_iter) * iter.instrs_per_cl;
    w.tcp_read_cl      = iter.cl_per_iter;
    w.l2_read_cl       = iter.cl_per_iter;
    w.xgmi_read_cl     = iter.cl_per_iter;
    return w;
  }
};

/**
 * @brief Write to a remote GPU's HBM via xGMI (egress).
 *
 * Write to a remote GPU's HBM via xGMI (egress). Unlike pull, push first has
 * to *source* the data locally — a full local read (HBM→MALL→L2→TCP) — and
 * then emit it onto the fabric (xGMI write). It therefore touches both the
 * local read path and the egress link, which is why push-heavy collectives
 * (e.g. all-gather) are bound by the xGMI write-concentration curve.
 */
struct push_t {
  int peer = 0;  ///< Remote GPU rank the data is written to.

  /**
   * @brief Resolve this remote write into functional-unit work for one iteration.
   *
   * @param iter Sizing of one software-pipelined iteration.
   * @return functional_unit_work_t Work charged on the local read path plus xGMI egress.
   */
  constexpr functional_unit_work_t resolve(const iter_dims_t& iter) const noexcept {
    functional_unit_work_t w{};
    w.vmem_read_instrs = static_cast<std::int64_t>(iter.cl_per_iter) * iter.instrs_per_cl;
    w.tcp_read_cl      = iter.cl_per_iter;
    w.l2_read_cl       = iter.cl_per_iter;
    w.mall_read_cl     = iter.cl_per_iter;
    w.hbm_read_cl      = iter.cl_per_iter;
    w.xgmi_write_cl    = iter.cl_per_iter;
    return w;
  }
};

/**
 * @brief Element-wise reduction on data already resident in registers.
 *
 * Element-wise reduction on data already in registers (the arithmetic in a
 * reduce-scatter / all-reduce). It moves no lines — the data is resident — so
 * the only cost is VALU lane-ops, one per element reduced this iteration. This
 * is the sole primitive that can make a collective compute-bound rather than
 * bandwidth-bound.
 */
struct reduce_t {
  reduce_op_t op = reduce_op_t::SUM;  ///< Reduction operator applied element-wise.

  /**
   * @brief Resolve this reduction into functional-unit work for one iteration.
   *
   * @param iter Sizing of one software-pipelined iteration.
   * @return functional_unit_work_t VALU lane-ops, one per element reduced.
   */
  constexpr functional_unit_work_t resolve(const iter_dims_t& iter) const noexcept {
    functional_unit_work_t w{};
    w.valu_ops = iter.elements_per_iter;
    return w;
  }
};

/**
 * @brief Notify a peer that this rank's data is ready (producer side).
 *
 * Notify a peer that this rank's data is ready (producer side of the
 * handshake). Modeled as one fabric atomic (counted for T_sync, charged at
 * atomic_latency_cycles) plus the single line that carries the flag across
 * xGMI. It is iteration-independent — one signal per tile, not per element —
 * so resolve_work_graph routes it to sync_work, not the inner loop.
 */
struct signal_t {
  int peer = 0;  ///< Remote GPU rank that is signaled.

  /**
   * @brief Resolve this signal into per-tile sync work.
   *
   * The iteration dimensions are ignored: a signal is one per-tile cost, not
   * per-element inner-loop work.
   *
   * @return functional_unit_work_t One fabric atomic plus one xGMI write line.
   */
  constexpr functional_unit_work_t resolve(const iter_dims_t&) const noexcept {
    functional_unit_work_t w{};
    w.atomic_count  = 1;
    w.xgmi_write_cl = 1;
    return w;
  }
};

/**
 * @brief Spin-wait until a peer's signal arrives (consumer side).
 *
 * Spin-wait until a peer's signal arrives (consumer side). The polling read
 * resolves locally in L2 once the released flag has propagated, so it is one
 * atomic + one L2 line. Like signal, it is a per-tile sync cost, not inner-
 * loop work. Together signal/wait encode the producer→consumer dependency
 * whose count drives the per-timestep handshake latency.
 */
struct wait_t {
  int peer = 0;  ///< Remote GPU rank whose signal is awaited.

  /**
   * @brief Resolve this wait into per-tile sync work.
   *
   * The iteration dimensions are ignored: a wait is one per-tile cost, not
   * per-element inner-loop work.
   *
   * @return functional_unit_work_t One fabric atomic plus one L2 read line.
   */
  constexpr functional_unit_work_t resolve(const iter_dims_t&) const noexcept {
    functional_unit_work_t w{};
    w.atomic_count = 1;
    w.l2_read_cl   = 1;
    return w;
  }
};

/// Sum type for any primitive in a work graph.
using op_t = std::variant<load_t, store_t, pull_t, push_t, reduce_t, signal_t, wait_t>;

/**
 * @brief The two work buckets a resolved collective step splits into.
 *
 * A collective step is a small program of primitives; this composes that
 * program into two buckets that the latency model treats very differently:
 *   iter_work — the data-movement body that *repeats* once per pipelined
 *               iteration and therefore scales with tile size.
 *   sync_work — the signal/wait handshakes that happen *once per tile*
 *               regardless of size, charged as a fixed per-step latency.
 * Summing within each bucket (operator+) reflects that primitives in the same
 * step contend for the FUs together. Separating the buckets is what lets a
 * large transfer be bandwidth-bound while a tiny one is handshake-bound.
 */
struct resolved_work_t {
  functional_unit_work_t iter_work;  ///< Data-movement body repeated once per pipelined iteration.
  functional_unit_work_t sync_work;  ///< Signal/wait handshakes charged once per tile.
};

/**
 * @brief Resolve a work graph of primitives into iteration and sync work buckets.
 *
 * Visits each primitive, resolves it for the given iteration sizing, and routes
 * signal/wait into sync_work and everything else into iter_work, summing within
 * each bucket.
 *
 * @param ops The primitives composing one collective step.
 * @param iter Sizing of one software-pipelined iteration.
 * @return resolved_work_t The summed iter_work and sync_work for the step.
 */
resolved_work_t resolve_work_graph(const std::vector<op_t>& ops, const iter_dims_t& iter) noexcept;

}  // namespace origami::comm
