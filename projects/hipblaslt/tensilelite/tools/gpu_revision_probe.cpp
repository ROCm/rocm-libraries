// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Minimal standalone probe that prints a device's compiler arch name and its
// ASIC revision, one per line:
//
//     <gcnArchName>
//     <asicRevision>
//
// gfx1250 ships as two silicon revisions that share the same ISA/arch name, so
// gcnArchName alone cannot tell them apart. hipDeviceProp_t::asicRevision is the
// only in-process signal that distinguishes them (empirically v0 -> 0, v1 -> 1).
// This mirrors the exact read in rocblaslt's handle.cpp (asic_rev =
// properties.asicRevision, guarded by HIP_VERSION >= 307).
//
// Build (done on demand by tasks.py):  hipcc -O0 gpu_revision_probe.cpp -o <out>
// Usage:                               <out> [deviceId]   (deviceId defaults to 0)
//
// Exit codes: 0 on success; nonzero if the device properties could not be read.
// A revision of -1 is printed when the running HIP is too old to expose the
// field; callers must treat -1 (and any read failure) as "unknown" and fall
// back to the shipping default (v1).

#include <hip/hip_runtime.h>

#include <climits>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    int deviceId = 0;
    if(argc > 1)
    {
        // strtol (not atoi) so non-numeric/garbage input errors out instead of
        // silently probing device 0.
        char*     end = nullptr;
        long      val = std::strtol(argv[1], &end, 10);
        if(end == argv[1] || *end != '\0' || val < 0 || val > INT_MAX)
        {
            std::fprintf(stderr, "gpu_revision_probe: invalid deviceId '%s'\n", argv[1]);
            return 2;
        }
        deviceId = static_cast<int>(val);
    }

    // Value-initialize so gcnArchName is null-terminated even if a
    // partially-populated (ABI-skewed) hipSuccess is returned.
    hipDeviceProp_t properties{};
    hipError_t      err = hipGetDeviceProperties(&properties, deviceId);
    if(err != hipSuccess)
    {
        std::fprintf(stderr,
                     "gpu_revision_probe: hipGetDeviceProperties(device=%d) failed: %s\n",
                     deviceId,
                     hipGetErrorString(err));
        return 1;
    }

#if HIP_VERSION >= 307
    int asicRevision = properties.asicRevision;
#else
    int asicRevision = -1;
#endif

    // gcnArchName first, asicRevision second; one value per line so the caller
    // can parse without splitting on characters that may appear in arch names.
    // The caller must cross-check gcnArchName before trusting the revision:
    // asicRevision 0 means v0 *only* for gfx1250, and any doubt falls back to v1.
    if(std::printf("%s\n%d\n", properties.gcnArchName, asicRevision) < 0
       || std::fflush(stdout) != 0)
    {
        std::fprintf(stderr, "gpu_revision_probe: failed to write output\n");
        return 3;
    }
    return 0;
}
