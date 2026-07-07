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

// Test fixture: the nominal MI300X machine, built explicitly.
//
// The library no longer ships a hardcoded MI300X system_t — production callers
// build one from a live device (system_from_device / system_from_hardware) or
// from make_system with an explicit topology. The byte-identity suite, however,
// must pin an exact reference machine so the frozen golden numbers stay stable,
// so it fabricates the nominal MI300X here from the calibrated gfx942 ceilings
// and the part's known full-die topology at the nominal 2.0 GHz clock.
//
// make_system reproduces the previous inline MI300X tables bit-for-bit (same
// "GB/s / clock" and "ns * clock" arithmetic), so MI300X / MI300X_COMM /
// MI300X_SYSTEM below are drop-in references for every existing assertion. They
// are constexpr, so compile-time static_asserts keep working.
#pragma once

#include "origami/comm/hardware.hpp"

namespace origami::comm {

/// Full-die MI300X topology (gfx942): 304 CUs across 8 XCDs (38 CUs/XCD), 4 MiB
/// L2 per XCD. Live devices report this via origami::hardware_t; the test pins
/// it so the reference machine is independent of any present GPU.
inline constexpr gpu_topology_t kMI300XNominalTopology = {
    architecture_t::gfx942,
    304,
    8,
    38,
    4ULL * 1024ULL * 1024ULL,
};

/// The nominal MI300X machine at the 2.0 GHz reference clock. Byte-identical to
/// the constants the model used to ship inline.
inline constexpr system_t MI300X_SYSTEM =
    make_system(get_arch_ceilings(architecture_t::gfx942), kMI300XNominalTopology, 2.0);

/// Convenience references into the reference system, matching the names the
/// assertions were written against.
inline constexpr const hardware_t& MI300X           = MI300X_SYSTEM.gpu;
inline constexpr const comm_hardware_t& MI300X_COMM = MI300X_SYSTEM.fabric;

}  // namespace origami::comm
