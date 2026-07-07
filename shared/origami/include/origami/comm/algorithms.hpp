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

// origami::comm — collective algorithms (umbrella header)
//
// Each collective algorithm emits the *schedule* of a collective as a pure
// function of (pid, timestep); the cost model prices the resulting per-step work
// graphs rather than hard-coding any per-collective cost. See base.hpp for the
// full design notes and the three quantities (num_timesteps, chunks_per_timestep,
// wgs_per_active_link) that drive the cost.
//
// The algorithms are grouped by family; include this umbrella to pull them all,
// or a single family header for a narrower dependency:
//   • algorithms/base.hpp    — schedule_entry_t, work_graph_fn_t, floor_mod, and
//                              the abstract collective_algorithm_t base.
//   • algorithms/direct.hpp  — direct/staggered algorithms (all-to-same,
//                              pid-staggered, pid-partitioned, two-shot AR).
//   • algorithms/ring.hpp    — ring algorithms + the ring_distribute helper
//                              they share.
//   • algorithms/resolve.hpp — public factories and resolve_algorithm.
#pragma once

#include "origami/comm/algorithms/base.hpp"
#include "origami/comm/algorithms/direct.hpp"
#include "origami/comm/algorithms/resolve.hpp"
#include "origami/comm/algorithms/ring.hpp"
