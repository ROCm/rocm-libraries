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

#include "origami/comm/hardware_device.hpp"

#include <cstddef>

namespace origami::comm {

system_t system_from_hardware(const origami::hardware_t& hw) {
  // Guard against a degenerate XCD count so the per-XCD CU count stays well
  // defined; origami reports NUM_XCD>=1 for real devices.
  const std::size_t num_xcd = (hw.NUM_XCD > 0) ? hw.NUM_XCD : 1;

  // Topology comes entirely from the live device: under CPX partitioning the
  // same gfx942 part reports fewer CUs/XCDs, and this captures that. cu_per_xcd
  // is derived as N_CU/NUM_XCD (== CU_per_L2 on a per-XCD-L2 part like MI300X).
  //
  // TODO: the calibrated aggregate bandwidth ceilings (HBM/MALL/link) are still
  // whole-part figures. A CPX partition should scale those down with its share
  // of the device; until that lands, partition predictions overstate aggregate
  // bandwidth even though the CU/XCD-derived structure is correct.
  const gpu_topology_t topo{
      hw.arch,
      hw.N_CU,
      num_xcd,
      hw.N_CU / num_xcd,
      hw.L2_capacity,
  };

  return make_system(get_arch_ceilings(hw.arch), topo, hw.compute_clock_ghz);
}

system_t system_from_device(int deviceId) {
  return system_from_hardware(origami::hardware_t::get_hardware_for_device(deviceId));
}

}  // namespace origami::comm
