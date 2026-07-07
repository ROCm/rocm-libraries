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

// origami::comm — device bridge (origami::hardware_t → comm system_t).
//
// Unlike the rest of origami::comm, this header is intentionally HIP-dependent:
// it includes origami/hardware.hpp (which pulls in <hip/hip_runtime.h>) so a
// comm system_t can be built from the *live* device topology origami already
// knows how to query. The pure cost model (hardware.hpp, collective.hpp,
// tensor.hpp, and the origami_comm.hpp umbrella) stays HIP-free; only consumers
// that want device introspection include this file and link
// roc::origami-comm-device.
//
// Why this exists: the GEMM model (origami::hardware_t) and the comm model
// (origami::comm::system_t) describe the same physical GPU. Rather than hardcode
// a second nominal topology in comm, these factories source CU/XCD counts, L2
// capacity and clock from origami::hardware_t and fuse them with the calibrated,
// architecture-keyed comm ceilings (get_arch_ceilings). One device, one source
// of topological truth — and CPX partitioning is picked up automatically.
#pragma once

#include "origami/comm/hardware.hpp"
#include "origami/hardware.hpp"

namespace origami::comm {

/**
 * @brief Build a comm system_t from an already-constructed origami::hardware_t.
 *
 * Bridges the GEMM hardware model to the comm one: takes the GPU topology
 * origami resolved for kernel selection (architecture, CU/XCD counts, L2
 * capacity, clock — including any CPX-partition adjustments the caller applied)
 * and fuses it with the calibrated comm ceilings for that architecture via
 * make_system. Prefer this overload when a caller already holds an
 * origami::hardware_t (e.g. the GEMM selector) so both models describe the exact
 * same device with no duplicated topology constants.
 *
 * The communicator size (number of GPUs) is *not* taken from here — it is a
 * problem property (comm_problem_t::num_gpus); this describes a single rank's
 * machine.
 *
 * @param hw GEMM hardware description for the target device.
 * @return system_t Comm machine description for the same device.
 * @throws std::invalid_argument If no comm ceilings are calibrated for hw.arch.
 */
system_t system_from_hardware(const origami::hardware_t& hw);

/**
 * @brief Build a comm system_t by querying a live HIP device.
 *
 * Convenience wrapper that resolves origami::hardware_t for @p deviceId via
 * origami::hardware_t::get_hardware_for_device (runtime hipDeviceProp_t plus the
 * XCC-count query) and forwards to system_from_hardware. This is the path that
 * picks up CPX partitioning for free: the device reports its actual CU/XCD
 * counts. Requires a ROCm runtime and a visible device.
 *
 * @param deviceId HIP device ordinal.
 * @return system_t Comm machine description for that device.
 * @throws std::invalid_argument If the device's architecture has no comm ceilings.
 */
system_t system_from_device(int deviceId);

}  // namespace origami::comm
