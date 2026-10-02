// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Placement of a device buffer across a 4 GiB virtual-address boundary, for testing address
// arithmetic that drops the carry out of the low 32 bits of an address (ROCM-32046). Such a
// defect goes wrong only when a buffer crosses a 4 GiB boundary, and the wrong access lands
// exactly 4 GiB away, outside every allocation.
//
// PlacedRegion reserves address space with HIP virtual memory management and maps three windows
// of the same size: the buffer, starting a chosen number of bytes before a 4 GiB boundary, and
// two poison windows exactly 4 GiB below and above it, filled with kFastCheckPoisonValue. A read
// whose carry is dropped lands in the low window and returns the poison value instead of
// faulting, so the GEMM result is wrong rather than the GPU stopping. A write whose carry is
// dropped changes the poison, which verify_poison() reports.

#pragma once

#include "fast_check.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

class PlacedRegion
{
public:
    static constexpr uint64_t k4GiB = uint64_t(1) << 32;

    // Maps `bytes` so that the 4 GiB boundary falls `offset_below` bytes after the start.
    // offset_below = 0 picks the middle of the buffer, rounded down to the mapping granularity.
    // Returns nullptr and sets *why on failure; *unsupported is true when the GPU or platform
    // cannot do this at all, as opposed to an invalid request.
    static std::unique_ptr<PlacedRegion> create(
        size_t bytes, size_t offset_below, hipDataType type, std::string* why, bool* unsupported)
    {
        auto fail = [&](const std::string& text, bool platform) {
            if(why)
                *why = text;
            if(unsupported)
                *unsupported = platform;
            return std::unique_ptr<PlacedRegion>();
        };
#ifdef _WIN32
        return fail("buffer placement needs HIP virtual memory management, which is not "
                    "supported on Windows",
                    true);
#else
        int device = 0, vmm = 0;
        if(hipGetDevice(&device) != hipSuccess
           || hipDeviceGetAttribute(
                  &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, device)
                  != hipSuccess
           || !vmm)
            return fail("this GPU does not support HIP virtual memory management", true);

        const size_t es = fast_check_element_size(type);
        if(!es)
            return fail("buffer placement does not support data type " + std::to_string(int(type)),
                        false);

        std::unique_ptr<PlacedRegion> r(new PlacedRegion);
        r->m_type               = type;
        r->m_prop.type          = hipMemAllocationTypePinned;
        r->m_prop.location.type = hipMemLocationTypeDevice;
        r->m_prop.location.id   = device;
        size_t granularity      = 0;
        if(hipMemGetAllocationGranularity(
               &granularity, &r->m_prop, hipMemAllocationGranularityMinimum)
               != hipSuccess
           || !granularity || granularity % es)
            return fail("could not get a usable mapping granularity", true);

        if(offset_below == 0)
            offset_below = std::max<size_t>(granularity, bytes / 2 / granularity * granularity);
        if(offset_below % granularity)
            return fail("the placement offset must be a multiple of the mapping granularity ("
                            + std::to_string(granularity) + " bytes)",
                        false);
        if(offset_below >= bytes)
            return fail("a " + std::to_string(bytes) + "-byte buffer cannot cross a 4 GiB boundary "
                            + std::to_string(offset_below) + " bytes after its start",
                        false);

        r->m_bytes = bytes;
        r->m_span  = (bytes + granularity - 1) / granularity * granularity;
        if(r->m_span >= k4GiB)
            return fail("a placed buffer must be smaller than 4 GiB", false);

        // Reserve enough to put both poison windows around a boundary inside the reservation,
        // and compute the boundary from the address actually returned; the address hint is not
        // guaranteed.
        r->m_reserved_bytes = 4 * k4GiB + 2 * r->m_span;
        if(hipMemAddressReserve(&r->m_reserved, r->m_reserved_bytes, 0, nullptr, 0) != hipSuccess)
        {
            r->m_reserved = nullptr;
            return fail("could not reserve address space for placement", false);
        }
        const uint64_t base     = reinterpret_cast<uint64_t>(r->m_reserved);
        const uint64_t boundary = (base + k4GiB + r->m_span + k4GiB - 1) / k4GiB * k4GiB;
        r->m_windows[0]         = boundary - offset_below; // the buffer
        r->m_windows[1]         = r->m_windows[0] - k4GiB; // poison below
        r->m_windows[2]         = r->m_windows[0] + k4GiB; // poison above

        hipMemAccessDesc access{};
        access.location = r->m_prop.location;
        access.flags    = hipMemAccessFlagsProtReadWrite;
        for(int w = 0; w < 3; w++)
        {
            // Each window has its own handle: a mapping must cover a whole handle.
            void* at = reinterpret_cast<void*>(r->m_windows[w]);
            if(hipMemCreate(&r->m_handles[w], r->m_span, &r->m_prop, 0) != hipSuccess)
                return fail("could not allocate " + std::to_string(r->m_span)
                                + " bytes of device memory for placement",
                            false);
            r->m_created[w] = true;
            if(hipMemMap(at, r->m_span, 0, r->m_handles[w], 0) != hipSuccess)
                return fail("could not map placed memory", false);
            r->m_mapped[w] = true;
            if(hipMemSetAccess(at, r->m_span, &access, 1) != hipSuccess)
                return fail("could not enable access to placed memory", false);
        }
        if(r->fill_poison(nullptr) != hipSuccess || hipDeviceSynchronize() != hipSuccess)
            return fail("could not fill the placement poison windows", false);
        return r;
#endif
    }

    ~PlacedRegion()
    {
        (void)hipDeviceSynchronize();
        for(int w = 0; w < 3; w++)
        {
            if(m_mapped[w])
                (void)hipMemUnmap(reinterpret_cast<void*>(m_windows[w]), m_span);
            if(m_created[w])
                (void)hipMemRelease(m_handles[w]);
        }
        if(m_reserved)
            (void)hipMemAddressFree(m_reserved, m_reserved_bytes);
    }

    PlacedRegion(const PlacedRegion&)            = delete;
    PlacedRegion& operator=(const PlacedRegion&) = delete;

    void* ptr() const
    {
        return reinterpret_cast<void*>(m_windows[0]);
    }

    // The first byte at or above the 4 GiB boundary the buffer crosses.
    uint64_t boundary() const
    {
        return (m_windows[0] / k4GiB + 1) * k4GiB;
    }

    // Refills both poison windows with kFastCheckPoisonValue.
    hipError_t fill_poison(hipStream_t stream) const
    {
        const size_t elements = m_span / fast_check_element_size(m_type);
        for(int w = 1; w < 3; w++)
            fast_check_poison_padding_device(
                {reinterpret_cast<void*>(m_windows[w]), m_type, 0, 0, 1, 0}, 1, elements, stream);
        return hipGetLastError();
    }

    // Fails when any element of either poison window no longer holds the poison value: a write
    // reached an address exactly 4 GiB away from where it belonged.
    FastCheckResult verify_poison(const std::string& operand, hipStream_t stream) const
    {
        FastCheckResult    result;
        std::ostringstream msg;
        const size_t       es       = fast_check_element_size(m_type);
        const char*        where[3] = {"", "below", "above"};
        for(int w = 1; w < 3; w++)
        {
            FastCheckChanged c = fast_check_count_changed_device(
                reinterpret_cast<const void*>(m_windows[w]), m_type, m_span / es, stream);
            if(!c.ok)
                return {false, "could not scan the placement poison windows"};
            if(c.count)
            {
                result.passed = false;
                msg << c.count << " elements of the poison window 4 GiB " << where[w] << " "
                    << operand << " were written. The first is " << operand << " element "
                    << c.first << " (byte offset " << c.first * es << ") moved 4 GiB " << where[w]
                    << ", the effect of an address that "
                    << (w == 1 ? "lost the carry out of its low 32 bits"
                               : "gained a carry into its high 32 bits")
                    << ".\n";
            }
        }
        result.message = msg.str();
        return result;
    }

private:
    PlacedRegion() = default;

    hipDataType                     m_type{};
    hipMemAllocationProp            m_prop{};
    size_t                          m_bytes          = 0;
    size_t                          m_span           = 0; // bytes mapped per window
    void*                           m_reserved       = nullptr;
    size_t                          m_reserved_bytes = 0;
    uint64_t                        m_windows[3]     = {}; // buffer, poison below, poison above
    hipMemGenericAllocationHandle_t m_handles[3]{};
    bool                            m_created[3] = {};
    bool                            m_mapped[3]  = {};
};
