#pragma once

// Free of HIP headers, so the kernel headers can include it.

namespace hipconv
{

struct ArchEntry;

// Opaque, copyable handle to a library-managed architecture.
//
// ArchHandle may become an object in the future that allocates resources
// in its constructor and deallocates them in its destructor (RAII). Now,
// there are no per-architecture resources to manage. Either way, callers
// treat it as a copyable value.
using ArchHandle = const ArchEntry*;

// Target device for selecting and launching a layer.
//
// Every function that takes a Device reads the device's properties from it, so a caller can
// plan for a device other than the current one. Pass the same Device to every call for a layer:
// launch sizes its grid from cu_count, and get_workspace_size sized the workspace from it.
struct Device
{
    ArchHandle arch;
    // Compute units of the device, or of its partition: MIOpen's GetMaxComputeUnits(),
    // hipDeviceProp_t::multiProcessorCount.
    int cu_count;
};

} // namespace hipconv
