// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// HIP observation and memory guards for the ordinary GEMM clients.
// Numerical certification and comparison are provided by HostNumerics.

#pragma once

#include <hipblaslt/hipblaslt.h>
#include <roc/hostnumerics/freivalds.hpp>

#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

// The bit pattern written over every element of D before each launch, chosen per type, in the
// low bytes. Floating types get a NaN: all-ones bytes, or 0x80 for the FNUZ fp8 types, whose only
// NaN is 0x80. An element that is never written then stands out as non-finite. Integer types have
// no NaN and get their minimum value. freivalds recomputes every integer element that holds the
// type's minimum or maximum exactly, so an unwritten element is still caught, unless its correct
// value saturates to the minimum.
uint64_t device_sentinel_bits(hipDataType type);

// Value written into the padding of A, B and C. integer_exact data lies in [-2, 2]; 40 is far
// outside that range and is exactly representable in every supported input type,
// including fp8 e4m3 and e5m2. It is finite so that kernels which load padding and multiply it by
// zero, which is legitimate, still produce 0 rather than NaN.
constexpr float kPaddingPoisonValue = 40.0f;

// One column-major device matrix as stored, before any transpose is applied.
struct DeviceMatrix
{
    const void* data   = nullptr; // first element of batch 0
    hipDataType type   = HIP_R_32F;
    int64_t     rows   = 0; // stored rows
    int64_t     cols   = 0; // stored columns
    int64_t     ld     = 0; // leading dimension, in elements
    int64_t     stride = 0; // distance between batches, in elements
};

struct DeviceVerificationStatistics
{
    size_t checked_elements     = 0;
    size_t recomputed_elements  = 0;
    size_t diagnostic_elements  = 0;
    size_t readback_bytes       = 0;
    size_t device_scratch_bytes = 0;
    double projection_seconds   = 0; // GPU event interval
    double readback_seconds     = 0; // GPU event intervals for projection/exception copies
    double host_check_seconds   = 0; // wall time, including bounded diagnostics
    bool   used_host_fallback   = false;
};

struct ValidationResult
{
    bool                         passed = true;
    std::string                  message; // empty when passed
    DeviceVerificationStatistics statistics;
};

// Returns true when the device validation helpers can load values of this type, and the reason when not.
bool supported_validation_type(hipDataType type, std::string* why = nullptr);

// Project on the GPU with the shared client observer. Unsupported output codecs
// or overflowing exception lists use an explicit full-output readback fallback.
ValidationResult verify_freivalds_device(const DeviceMatrix&                               output,
                                         std::span<const roc::hostnumerics::FreivaldsPlan> plans,
                                         hipStream_t                                       stream,
                                         size_t maximumExceptions
                                         = std::numeric_limits<size_t>::max());

// Returns why a case that needs device_bytes of device memory and host_bytes of host memory
// cannot run here, or an empty string when both fit. Device memory is what hipMemGetInfo reports
// free; host memory is MemAvailable from /proc/meminfo, which counts reclaimable page cache,
// unlike sysinfo's freeram. Before reporting a shortfall, releases idle client-pool buffers and
// queries again, since those buffers otherwise hide memory available to the next case.
std::string validation_memory_shortfall(size_t device_bytes, size_t host_bytes);

// Reads element i of a host buffer of the given type as a double.
double load_validation_scalar(const void* data, hipDataType type, size_t i);

// With an activation, D is not linear in the inputs, so it is checked through E, the
// pre-activation result the kernel also writes: E is verified like D (with scale_e in place of
// scale_d), and then every element of D must equal scale_d * act(E / scale_e), rounded to D's
// type. Copies the M x N x batch regions of D and E to the host. Sets *amax, when given, to the
// largest |act(E / scale_e)|, the value amaxD must hold. Fails, and sets *amax to NaN, when an
// element of E is outside the range its type stores exactly.
ValidationResult verify_activation_device(const DeviceMatrix&                          d,
                                          const DeviceMatrix&                          e,
                                          int64_t                                      batch_count,
                                          double                                       scale_d,
                                          double                                       scale_e,
                                          const roc::hostnumerics::ActivationFunction& activation,
                                          hipStream_t                                  stream,
                                          double*                                      amax);

// The largest |element| of a verified device D divided by scale_d, the value amaxD must hold
// without an activation, or NaN when D holds values outside the range its type stores exactly
// or cannot be copied.
// Copies D's M x N x batch region to the host.
double recover_amax_device(const DeviceMatrix& d,
                           int64_t             batch_count,
                           double              scale_d,
                           hipStream_t         stream);

// Checks the bias gradient a GEMM writes for one batch, exactly from the inputs: source 'a'
// (BGRADA) sums each row of op(A) over K, and 'b' (BGRADB) each column of op(B). bias holds the
// kernel's owned output Tensor on the host. Refuses non-integer inputs or a reduction whose
// absolute-sum bound reaches the compute type's exact integer limit.
ValidationResult verify_bias_gradient(const roc::hostnumerics::Tensor& input,
                                      char                             source,
                                      roc::hostnumerics::ScalarType    computeType,
                                      const roc::hostnumerics::Tensor& bias);

// Changes element (row, col) of the given batch of a device matrix to a different value: the
// finite zero or one, whichever differs from the stored value. Used by
// --validation-inject option to demonstrate a failing client invocation.
hipError_t corrupt_element_device(
    const DeviceMatrix& m, int64_t batch, int64_t row, int64_t col, hipStream_t stream);

// One device buffer a GEMM reads or writes, for the placement record in failure messages.
struct DeviceBufferRange
{
    const char* name;
    const void* base;
    size_t      bytes;
};

// Lists each buffer's address range, one per line, and flags any range that crosses a 4 GiB
// boundary. A kernel that drops the carry out of the low 32 bits of an address goes wrong only
// past such a boundary, so the record shows whether a failure lines up with one.
std::string describe_device_buffers(const std::vector<DeviceBufferRange>& buffers);

// Device helpers. A device matrix is described by DeviceMatrix with data pointing to device
// memory; total_elements is the size of its allocation in elements, which may extend past the
// last batch.

// Bytes per element, or 0 for an unsupported type.
size_t device_element_size(hipDataType type);

// The bit pattern of kPaddingPoisonValue in the given type, in the low bytes.
uint64_t device_poison_bits(hipDataType type);

// Counts the elements of a device range that no longer hold kPaddingPoisonValue.
struct ChangedElements
{
    uint64_t count = 0;
    uint64_t first = 0; // index of the first changed element, when count > 0
    bool     ok    = true; // false when the scan itself failed
};
ChangedElements
    count_changed_device(const void* data, hipDataType type, size_t elements, hipStream_t stream);

// Fills every element of a device buffer with device_sentinel_bits(type). Returns the launch
// error, or hipErrorInvalidValue for a unsupported type.
hipError_t
    fill_sentinel_device(void* buffer, hipDataType type, size_t elements, hipStream_t stream);

// Writes kPaddingPoisonValue into every element outside the rows x cols x batch region. Returns
// as fill_sentinel_device does.
hipError_t poison_padding_device(const DeviceMatrix& m,
                                 int64_t             batch_count,
                                 size_t              total_elements,
                                 hipStream_t         stream);

// Checks every element outside the rows x cols x batch region. With expect_poison, each must hold
// kPaddingPoisonValue (used when D shares C's poisoned memory); otherwise each must hold the
// sentinel pattern. Without expect_poison, and when the sentinel decodes to NaN in the matrix
// type, an element inside the region that still holds the sentinel is reported as never written.
ValidationResult scan_padding_device(const DeviceMatrix& m,
                                     int64_t             batch_count,
                                     size_t              total_elements,
                                     bool                expect_poison,
                                     hipStream_t         stream);

// Copies the rows x cols x batch region of a device matrix into contiguous host memory
// (leading dimension rows, batch stride rows * cols). host must hold that many elements.
hipError_t copy_matrix_to_host(void*               host,
                               const DeviceMatrix& device_matrix,
                               int64_t             batch_count,
                               hipStream_t         stream);
