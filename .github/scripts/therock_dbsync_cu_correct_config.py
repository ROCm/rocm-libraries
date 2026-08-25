# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Emit a CU-count-corrected copy of a rocjitsu KMD config for the GPU-free dbsync gate.
#
# Why: CK's grouped-wrw auto-split-k path (device_grouped_conv_bwd_weight_explicit.hpp) derives
# k_batch from `max_capacity = GetMaxOccupancy() * DeviceProperties.num_cu_`, where
# `num_cu_ = hipGetDeviceProperties().multiProcessorCount` -- read straight from the device,
# bypassing MIOpen's Handle CU-override. rocjitsu's stock cdna configs declare a MINIMAL device
# (~4 CUs) for fast functional emulation, so under rocjitsu that auto k_batch is wrong and
# StaticFDBSync false-flags small grouped-wrw perf configs (the "+-1" split-k entries) as invalid
# -- entries that are valid on real hardware (validated: prototype run 32404021201, and #10043's
# real-MI300X run left them). Setting the device block's CU fields to the arch's real CU count
# makes multiProcessorCount report correctly and clears the false-positives, while leaving the
# `topology` untouched so functional-emulation cost is unchanged.
#
# multiProcessorCount derives from the device block (confirmed: bumping it alone, topology
# untouched, cleared the false-positives), so we set simd_count = cu * simd_per_cu and
# num_shader_engines = num_shader_arrays_per_engine = 1, num_cu_per_sh = cu.
#
# Usage: therock_dbsync_cu_correct_config.py <input_config.json> <cu_count> <output_config.json>

import json
import sys


def main():
    if len(sys.argv) != 4:
        sys.exit(
            f"usage: {sys.argv[0]} <input_config.json> <cu_count> <output_config.json>"
        )
    in_path, cu_count, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]

    with open(in_path) as f:
        config = json.load(f)

    dev = config["vm"]["gpu"]["device"]
    simd_per_cu = dev.get("simd_per_cu", 4)
    dev["simd_count"] = cu_count * simd_per_cu
    dev["num_shader_engines"] = 1
    dev["num_shader_arrays_per_engine"] = 1
    dev["num_cu_per_sh"] = cu_count

    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"CU-corrected {in_path} -> {out_path}: "
        f"multiProcessorCount target={cu_count} "
        f"(simd_count={dev['simd_count']}, simd_per_cu={simd_per_cu}, num_cu_per_sh={cu_count})"
    )


if __name__ == "__main__":
    main()
