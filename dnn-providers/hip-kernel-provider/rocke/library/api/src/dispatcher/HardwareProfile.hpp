// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

#include <hip/hip_runtime.h>

#include "dispatcher/HardwareProfileSupplements.hpp"

namespace rocke_client::dispatcher
{

// The per-arch hardware constants the FMHA feature vector's group-C features
// (hw_num_cus .. hw_num_xcd, feature_spec.json indices 60-67) are built from.
//
// Company policy: no chip-configuration counts (CU count etc.) may be stored in
// source. So the chip-config fields default to 0 and are ONLY ever populated
// from the live device via fromDevice(); a zero profile yields obviously-
// degenerate features rather than a plausible-but-wrong hardcoded chip. The
// remaining fields are microarchitectural constants (not chip-config counts) and
// keep sensible defaults, still refined by fromDevice() when a device is up.
//
// This mirrors the Python FmhaFeatureEngine's hardware inputs field-for-field so
// the C++ featurizer (when it lands, #8866) and the training engine agree.
struct HardwareProfile
{
    // Chip-config counts: 0 until device-populated (no literals in source).
    int num_cus = 0;
    int shader_engines = 0;
    int num_xcd = 0;
    int max_clock_mhz = 0;
    int lds_capacity = 0;
    // Microarch constants (not chip-config counts).
    int simds_per_cu = 4;
    int wavefront_size = 64;
    // Cache sizes (not queryable from HIP, supplemented from generated data)
    int max_waves_per_cu = 0;
    int l1_cache_kb = 0;
    int l2_cache_kb = 0;
    int l3_cache_kb = 0;

    int total_simds() const
    {
        return num_cus * simds_per_cu;
    }

    // Populate from the live device instead of storing chip constants in source.
    // On query failure the chip-config fields stay 0 (degenerate, not wrong).
    // shader_engines and num_xcd are not exposed as clean hipDeviceProp_t scalars
    // and are left 0 (TODO: hipDeviceGetAttribute / per-uarch table if a model
    // ever needs them -- in the last FMHA result all hw_* features had zero
    // importance, so this is not on the critical path).
    static HardwareProfile fromDevice(int device = 0)
    {
        HardwareProfile hw;
        hipDeviceProp_t props{};
        if(hipGetDeviceProperties(&props, device) != hipSuccess)
        {
            return hw; // chip-config fields remain 0
        }
        hw.num_cus = props.multiProcessorCount;
        hw.max_clock_mhz = props.clockRate / 1000; // clockRate is in kHz
        hw.wavefront_size = props.warpSize;
        hw.lds_capacity = static_cast<int>(props.sharedMemPerBlock);
        return hw;
    }

    // Hybrid approach: query HIP for authoritative values, supplement with
    // generated data for fields HIP doesn't expose. This is the preferred
    // method for featurization as it provides complete hardware profiles.
    static HardwareProfile fromDeviceWithSupplement(int device, const std::string& arch)
    {
        HardwareProfile hw;

        // Query HIP for authoritative values
        hipDeviceProp_t props{};
        if(hipGetDeviceProperties(&props, device) == hipSuccess)
        {
            hw.num_cus = props.multiProcessorCount;
            hw.max_clock_mhz = props.clockRate / 1000;
            hw.wavefront_size = props.warpSize;
            hw.lds_capacity = static_cast<int>(props.sharedMemPerBlock);
        }

        // Supplement with fields HIP doesn't provide (from generated data)
        const auto* supplement = getSupplement(arch);
        if(supplement)
        {
            hw.shader_engines = supplement->shader_engines;
            hw.num_xcd = supplement->num_xcd;
            hw.simds_per_cu = supplement->simds_per_cu;
            hw.max_waves_per_cu = supplement->max_waves_per_cu;
            hw.l1_cache_kb = supplement->l1_cache_kb;
            hw.l2_cache_kb = supplement->l2_cache_kb;
            hw.l3_cache_kb = supplement->l3_cache_kb;
        }

        return hw;
    }
};

} // namespace rocke_client::dispatcher
