// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_validation.hpp"
#include "d_vector.hpp"
#include <hipblaslt/client/FreivaldsDevice.hpp>
#include <hipblaslt/hostnumerics/Types.hpp>
#include <roc/hostnumerics/exact_verification.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace hn = roc::hostnumerics;

namespace
{
    // ------------------------------------------------------------------------------------------
    // Element types
    // ------------------------------------------------------------------------------------------

    size_t element_size(hipDataType t)
    {
        switch(t)
        {
        case HIP_R_64F:
            return 8;
        case HIP_R_32F:
        case HIP_R_32I:
            return 4;
        case HIP_R_16F:
        case HIP_R_16BF:
            return 2;
        case HIP_R_8F_E4M3_FNUZ:
        case HIP_R_8F_E5M2_FNUZ:
        case HIP_R_8F_E4M3:
        case HIP_R_8F_E5M2:
        case HIP_R_8I:
            return 1;
        default:
            return 0;
        }
    }

    // A scalar copy at the product byte boundary. Numerical loops consume
    // owned Tensors directly; only sentinel checks and bounded diagnostics use this.
    double load(const void* data, hipDataType type, size_t index)
    {
        const size_t bytes = element_size(type);
        if(!bytes || !data)
            return std::numeric_limits<double>::quiet_NaN();
        return hn::Tensor::copyEncodedBackingStorage(
                   hipblaslt::hostnumerics::scalarType(type),
                   hn::Layout(hn::Shape{}, {}),
                   std::span<const std::byte>(static_cast<const std::byte*>(data) + index * bytes,
                                              bytes))
            .item<double>();
    }

    // ------------------------------------------------------------------------------------------
    // Comparison and reporting, shared by the host and device passes over D
    // ------------------------------------------------------------------------------------------

    struct BadElement
    {
        int64_t batch;
        int64_t row;
        int64_t col;
        double  expected; // value a correct kernel stores
        double  got;
    };

    constexpr size_t kMaxReported = 8;

    std::string format_offset(const DeviceMatrix& d, const BadElement& e)
    {
        int64_t            ld     = d.ld;
        int64_t            stride = d.stride;
        int64_t            offset = e.batch * stride + e.col * ld + e.row;
        std::ostringstream s;
        s << "batch " << e.batch << ", row " << e.row << ", col " << e.col << ": expected "
          << e.expected << ", got " << e.got << " (element offset " << offset << " = 0x" << std::hex
          << offset << std::dec;
        if(offset >= (int64_t(1) << 32))
            s << ", at or above 2^32";
        else if(offset >= (int64_t(1) << 31))
            s << ", at or above 2^31";
        s << ")";
        return s.str();
    }

    // Numerical decisions come from HostNumerics; offsets and launch diagnostics
    // remain product-specific. A shared exception budget covers all batches.
    struct Report
    {
        const DeviceMatrix&          d;
        const size_t                 slow_budget;
        bool                         over_budget = false;
        DeviceVerificationStatistics statistics;
        std::string                  incomplete_reason;
        std::vector<BadElement>      bad_elements, located;
        size_t                       bad_element_count = 0, bad_row_count = 0, bad_col_count = 0;

        Report(const DeviceMatrix& output, size_t maximumExceptions)
            : d(output)
            , slow_budget(maximumExceptions)
        {
        }

        size_t remaining_exceptions() const
        {
            return size_t(slow_budget)
                   - std::min(size_t(slow_budget),
                              statistics.recomputed_elements + statistics.diagnostic_elements);
        }

        template <typename StoredAt>
        void compare(const hn::FreivaldsPlan&  plan,
                     int64_t                   batch,
                     const hn::FreivaldsCheck& check,
                     StoredAt&&                stored_at)
        {
            if(check.status != hn::VerificationStatus::Passed
               && check.status != hn::VerificationStatus::Mismatch)
            {
                over_budget       = true;
                incomplete_reason = check.reason;
                return;
            }
            statistics.checked_elements += check.checkedElements;
            statistics.recomputed_elements += check.recomputedElements;
            bad_element_count += check.badElementCount;
            bad_row_count += check.badRowCount;
            bad_col_count += check.badColumnCount;
            for(const auto& element : check.badElements)
                if(bad_elements.size() < kMaxReported && remaining_exceptions())
                {
                    ++statistics.diagnostic_elements;
                    bad_elements.push_back({batch,
                                            int64_t(element.row),
                                            int64_t(element.column),
                                            plan.roundedValue(element.row, element.column),
                                            element.storedValue});
                }
            for(size_t i : check.badRows)
                for(size_t j : check.badColumns)
                {
                    if(located.size() >= kMaxReported || !remaining_exceptions())
                        return;
                    ++statistics.diagnostic_elements;
                    const double expected = plan.roundedValue(i, j);
                    const double stored   = stored_at(i, j);
                    if(expected != stored)
                        located.push_back({batch, int64_t(i), int64_t(j), expected, stored});
                }
        }

        ValidationResult finish() const
        {
            ValidationResult result;
            result.statistics = statistics;
            std::ostringstream msg;
            auto               fail = [&](const std::string& text) {
                result.passed = false;
                msg << text << "\n";
            };
            if(over_budget)
                fail("freivalds stopped: " + incomplete_reason
                     + " Use a 32-bit output type, reduce K, alpha or the scale factors, "
                       "or use unit_check for this shape.");
            if(bad_element_count > 0)
            {
                std::ostringstream s;
                s << bad_element_count << " elements of D are wrong or not finite:";
                for(const auto& element : bad_elements)
                    s << "\n  " << format_offset(d, element);
                fail(s.str());
            }
            if(bad_row_count > 0 || bad_col_count > 0)
            {
                std::ostringstream s;
                s << "Probe sums disagree in " << bad_row_count << " rows and " << bad_col_count
                  << " columns of D.";
                if(!located.empty())
                {
                    s << " Mismatched elements found where they cross:";
                    for(const auto& element : located)
                        s << "\n  " << format_offset(d, element);
                }
                fail(s.str());
            }
            result.message = msg.str();
            return result;
        }
    };
} // namespace

bool supported_validation_type(hipDataType type, std::string* why)
{
    if(element_size(type) != 0)
        return true;
    if(why)
        *why = "device validation does not support data type " + std::to_string(int(type));
    return false;
}

std::string describe_device_buffers(const std::vector<DeviceBufferRange>& buffers)
{
    constexpr uint64_t four_gib = uint64_t(1) << 32;
    std::ostringstream s;
    s << "Buffer placement:";
    for(const auto& b : buffers)
    {
        if(!b.base || !b.bytes)
            continue;
        uint64_t begin = uint64_t(reinterpret_cast<uintptr_t>(b.base));
        uint64_t end   = begin + b.bytes; // one past the last byte
        s << "\n  " << b.name << ": 0x" << std::hex << begin << " to 0x" << end << std::dec << " ("
          << b.bytes << " bytes)";
        uint64_t boundary = (begin / four_gib + 1) * four_gib;
        if(boundary < end)
            s << ", crosses a 4 GiB boundary at 0x" << std::hex << boundary << std::dec;
    }
    return s.str();
}

// ----------------------------------------------------------------------------------------------
// Device helpers
// ----------------------------------------------------------------------------------------------

namespace
{
    __device__ inline bool in_region(
        size_t idx, int64_t rows, int64_t cols, int64_t ld, int64_t stride, int64_t batch_count)
    {
        size_t b = 0, rem = idx;
        if(batch_count > 1)
        {
            b   = idx / size_t(stride);
            rem = idx - b * size_t(stride);
        }
        if(b >= size_t(batch_count))
            return false;
        size_t j = rem / size_t(ld);
        size_t i = rem - j * size_t(ld);
        return j < size_t(cols) && i < size_t(rows);
    }

    template <typename U>
    __global__ void poison_kernel(U*      p,
                                  size_t  total,
                                  int64_t rows,
                                  int64_t cols,
                                  int64_t ld,
                                  int64_t stride,
                                  int64_t batch,
                                  U       value)
    {
        for(size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < total;
            idx += size_t(gridDim.x) * blockDim.x)
            if(!in_region(idx, rows, cols, ld, stride, batch))
                p[idx] = value;
    }

    // counters: [0] padding mismatches, [1] first padding mismatch offset,
    //           [2] unwritten elements,  [3] first unwritten offset
    template <typename U>
    __global__ void scan_kernel(const U*            p,
                                size_t              total,
                                int64_t             rows,
                                int64_t             cols,
                                int64_t             ld,
                                int64_t             stride,
                                int64_t             batch,
                                U                   padding_value,
                                bool                check_unwritten,
                                U                   sentinel,
                                unsigned long long* counters)
    {
        for(size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < total;
            idx += size_t(gridDim.x) * blockDim.x)
        {
            U v = p[idx];
            if(in_region(idx, rows, cols, ld, stride, batch))
            {
                if(check_unwritten && v == sentinel)
                {
                    atomicAdd(&counters[2], 1ull);
                    atomicMin(&counters[3], (unsigned long long)idx);
                }
            }
            else if(v != padding_value)
            {
                atomicAdd(&counters[0], 1ull);
                atomicMin(&counters[1], (unsigned long long)idx);
            }
        }
    }

    uint64_t poison_bits(hipDataType t)
    {
        uint64_t bits = 0;
        auto     put  = [&](auto value) { std::memcpy(&bits, &value, sizeof(value)); };
        switch(t)
        {
        case HIP_R_32F:
            put(float(kPaddingPoisonValue));
            break;
        case HIP_R_64F:
            put(double(kPaddingPoisonValue));
            break;
        case HIP_R_16F:
            put(hipblasLtHalf(kPaddingPoisonValue));
            break;
        case HIP_R_16BF:
            put(hip_bfloat16(kPaddingPoisonValue));
            break;
        case HIP_R_8F_E4M3_FNUZ:
            put(hipblaslt_f8_fnuz(kPaddingPoisonValue));
            break;
        case HIP_R_8F_E5M2_FNUZ:
            put(hipblaslt_bf8_fnuz(kPaddingPoisonValue));
            break;
        case HIP_R_8F_E4M3:
            put(hipblaslt_f8(kPaddingPoisonValue));
            break;
        case HIP_R_8F_E5M2:
            put(hipblaslt_bf8(kPaddingPoisonValue));
            break;
        case HIP_R_32I:
            put(int32_t(kPaddingPoisonValue));
            break;
        case HIP_R_8I:
            put(int8_t(kPaddingPoisonValue));
            break;
        default:
            break;
        }
        return bits;
    }

    uint64_t sentinel_bits(hipDataType t)
    {
        switch(t)
        {
        case HIP_R_8F_E4M3_FNUZ:
        case HIP_R_8F_E5M2_FNUZ:
        case HIP_R_8I:
            return 0x80;
        case HIP_R_32I:
            return 0x80000000;
        default:
        {
            const size_t es = element_size(t);
            return es >= 8 ? ~uint64_t(0) : (uint64_t(1) << (8 * es)) - 1;
        }
        }
    }

    bool sentinel_is_nan(hipDataType t)
    {
        uint64_t bits = sentinel_bits(t);
        return std::isnan(load(&bits, t, 0));
    }

    unsigned grid_for(size_t total)
    {
        return unsigned(std::min<size_t>((total + 255) / 256, 16384));
    }

    // Writes `value` into every element outside the rows x cols x batch region.
    template <typename U>
    hipError_t launch_poison(
        const DeviceMatrix& m, int64_t batch, size_t total, uint64_t value, hipStream_t stream)
    {
        // HIP rejects a launch with an empty grid.
        if(total == 0)
            return hipSuccess;
        hipLaunchKernelGGL(poison_kernel<U>,
                           dim3(grid_for(total)),
                           dim3(256),
                           0,
                           stream,
                           static_cast<U*>(const_cast<void*>(m.data)),
                           total,
                           m.rows,
                           m.cols,
                           m.ld,
                           m.stride,
                           batch,
                           U(value));
        return hipGetLastError();
    }

    // Leaves a buffer of an unsupported type alone and says so.
    hipError_t fill_outside_region(
        const DeviceMatrix& m, int64_t batch, size_t total, uint64_t value, hipStream_t stream)
    {
        switch(element_size(m.type))
        {
        case 1:
            return launch_poison<uint8_t>(m, batch, total, value, stream);
        case 2:
            return launch_poison<uint16_t>(m, batch, total, value, stream);
        case 4:
            return launch_poison<uint32_t>(m, batch, total, value, stream);
        case 8:
            return launch_poison<uint64_t>(m, batch, total, value, stream);
        default:
            return hipErrorInvalidValue;
        }
    }

    template <typename U>
    hipError_t launch_scan(const DeviceMatrix& m,
                           int64_t             batch,
                           size_t              total,
                           bool                expect_poison,
                           unsigned long long* d_counters,
                           hipStream_t         stream)
    {
        U sentinel      = U(sentinel_bits(m.type));
        U padding_value = expect_poison ? U(poison_bits(m.type)) : sentinel;
        if(total == 0)
            return hipSuccess;
        hipLaunchKernelGGL(scan_kernel<U>,
                           dim3(grid_for(total)),
                           dim3(256),
                           0,
                           stream,
                           static_cast<const U*>(m.data),
                           total,
                           m.rows,
                           m.cols,
                           m.ld,
                           m.stride,
                           batch,
                           padding_value,
                           !expect_poison && sentinel_is_nan(m.type),
                           sentinel,
                           d_counters);
        return hipGetLastError();
    }

    std::string describe_offset(const DeviceMatrix& m, int64_t batch, uint64_t offset)
    {
        std::ostringstream s;
        s << "element offset " << offset << " = 0x" << std::hex << offset << std::dec;
        uint64_t b = 0, rem = offset;
        if(batch > 1)
        {
            b   = offset / uint64_t(m.stride);
            rem = offset - b * uint64_t(m.stride);
        }
        s << " (batch " << b << ", row " << rem % uint64_t(m.ld) << ", col " << rem / uint64_t(m.ld)
          << ")";
        return s.str();
    }

    // Device memory released when it goes out of scope.
    template <typename T>
    struct DeviceArray
    {
        T* ptr = nullptr;
        explicit DeviceArray(size_t n)
        {
            if(n && hipMalloc(&ptr, n * sizeof(T)) != hipSuccess)
                ptr = nullptr;
        }
        ~DeviceArray()
        {
            (void)hipFree(ptr);
        }
        DeviceArray(const DeviceArray&) = delete;
        DeviceArray& operator=(const DeviceArray&) = delete;
    };

} // namespace

ValidationResult verify_freivalds_device(const DeviceMatrix&                output,
                                         std::span<const hn::FreivaldsPlan> plans,
                                         hipStream_t                        stream,
                                         size_t                             maximumExceptions)
{
    if(plans.empty() || output.rows == 0 || output.cols == 0)
        return {};
    Report report(output, maximumExceptions);
    try
    {
        for(size_t batch = 0; batch < plans.size() && !report.over_budget; ++batch)
        {
            const auto&      plan = plans[batch];
            const hn::Layout layout(hn::Shape{size_t(output.rows), size_t(output.cols)},
                                    {1, output.ld},
                                    ptrdiff_t(batch) * output.stride);
            auto             observed = hipblaslt::client::observeFreivalds(
                plan.copyRequest(), output.data, layout, stream);
            auto& statistics = report.statistics;
            statistics.readback_bytes += observed.readbackBytes;
            statistics.device_scratch_bytes
                = std::max(statistics.device_scratch_bytes, observed.deviceScratchBytes);
            statistics.projection_seconds += observed.projectionSeconds;
            statistics.readback_seconds += observed.readbackSeconds;
            statistics.host_check_seconds += observed.reductionSeconds;
            const auto* host = std::get_if<hn::Tensor>(&observed.output);
            statistics.used_host_fallback |= host != nullptr;
            const auto begin = std::chrono::steady_clock::now();
            const auto check = std::visit(
                [&](const auto& output) {
                    return plan.check(output, report.remaining_exceptions());
                },
                observed.output);
            report.compare(plan, batch, check, [&](size_t i, size_t j) {
                if(host)
                    return host->loadAs<double>({i, j});
                std::array<std::byte, 8> one{};
                const auto               bytes = element_size(output.type);
                const auto               offset
                    = ptrdiff_t(batch) * output.stride + ptrdiff_t(i) + ptrdiff_t(j) * output.ld;
                const auto status = hipMemcpy(one.data(),
                                              static_cast<const std::byte*>(output.data)
                                                  + offset * ptrdiff_t(bytes),
                                              bytes,
                                              hipMemcpyDeviceToHost);
                statistics.readback_bytes += bytes;
                return status == hipSuccess ? load(one.data(), output.type, 0)
                                            : std::numeric_limits<double>::quiet_NaN();
            });
            statistics.host_check_seconds
                += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        }
    }
    catch(const std::exception& error)
    {
        return {false, error.what(), report.statistics};
    }
    return report.finish();
}

uint64_t device_sentinel_bits(hipDataType type)
{
    return sentinel_bits(type);
}

hipError_t fill_sentinel_device(void* buffer, hipDataType type, size_t elements, hipStream_t stream)
{
    // An empty region makes every element padding.
    return fill_outside_region(
        {buffer, type, 0, 0, 1, 0}, 1, elements, sentinel_bits(type), stream);
}

hipError_t poison_padding_device(const DeviceMatrix& m,
                                 int64_t             batch_count,
                                 size_t              total_elements,
                                 hipStream_t         stream)
{
    return fill_outside_region(m, batch_count, total_elements, poison_bits(m.type), stream);
}

namespace
{
    // Runs scan_kernel and returns its counters: [0] padding mismatches, [1] first padding
    // mismatch, [2] unwritten elements, [3] first unwritten element.
    hipError_t run_scan(const DeviceMatrix& m,
                        int64_t             batch_count,
                        size_t              total_elements,
                        bool                expect_poison,
                        hipStream_t         stream,
                        unsigned long long (&counters)[4])
    {
        unsigned long long              init[4] = {0, ~0ull, 0, ~0ull};
        DeviceArray<unsigned long long> d_counters(4);
        if(!d_counters.ptr)
            return hipErrorOutOfMemory;
        hipError_t err
            = hipMemcpyAsync(d_counters.ptr, init, sizeof(init), hipMemcpyHostToDevice, stream);
        if(err != hipSuccess)
            return err;
        // A failed launch, or an unsupported type, would otherwise leave the counters clean.
        err = hipErrorInvalidValue;
        switch(element_size(m.type))
        {
        case 1:
            err = launch_scan<uint8_t>(
                m, batch_count, total_elements, expect_poison, d_counters.ptr, stream);
            break;
        case 2:
            err = launch_scan<uint16_t>(
                m, batch_count, total_elements, expect_poison, d_counters.ptr, stream);
            break;
        case 4:
            err = launch_scan<uint32_t>(
                m, batch_count, total_elements, expect_poison, d_counters.ptr, stream);
            break;
        case 8:
            err = launch_scan<uint64_t>(
                m, batch_count, total_elements, expect_poison, d_counters.ptr, stream);
            break;
        default:
            // An unsupported type would otherwise scan nothing and look clean.
            return hipErrorInvalidValue;
        }
        if(err != hipSuccess)
            return err;
        err = hipMemcpyAsync(
            counters, d_counters.ptr, sizeof(counters), hipMemcpyDeviceToHost, stream);
        if(err == hipSuccess)
            err = hipStreamSynchronize(stream);
        return err;
    }
} // namespace

size_t device_element_size(hipDataType type)
{
    return element_size(type);
}

uint64_t device_poison_bits(hipDataType type)
{
    return poison_bits(type);
}

ChangedElements
    count_changed_device(const void* data, hipDataType type, size_t elements, hipStream_t stream)
{
    // An empty region makes every element padding, so the scan compares all of them with the
    // poison value.
    DeviceMatrix       m{data, type, 0, 0, 1, 0};
    unsigned long long counters[4];
    ChangedElements    changed;
    if(run_scan(m, 1, elements, true, stream, counters) != hipSuccess)
    {
        changed.ok = false;
        return changed;
    }
    changed.count = counters[0];
    changed.first = counters[1];
    return changed;
}

ValidationResult scan_padding_device(const DeviceMatrix& m,
                                     int64_t             batch_count,
                                     size_t              total_elements,
                                     bool                expect_poison,
                                     hipStream_t         stream)
{
    ValidationResult   result;
    unsigned long long counters[4];
    hipError_t err = run_scan(m, batch_count, total_elements, expect_poison, stream, counters);
    if(err != hipSuccess)
        return {false, std::string("bounds check scan failed: ") + hipGetErrorString(err)};

    std::ostringstream msg;
    if(counters[0] > 0)
    {
        result.passed = false;
        msg << counters[0] << " elements outside the " << m.rows << " x " << m.cols << " x "
            << batch_count << " region were changed (out-of-bounds write); first at "
            << describe_offset(m, batch_count, counters[1]) << "\n";
    }
    if(counters[2] > 0)
    {
        result.passed = false;
        msg << counters[2] << " elements inside the region were never written; first at "
            << describe_offset(m, batch_count, counters[3]) << "\n";
    }
    result.message = msg.str();
    return result;
}

hipError_t
    copy_matrix_to_host(void* host, const DeviceMatrix& m, int64_t batch_count, hipStream_t stream)
{
    const size_t es    = element_size(m.type);
    const size_t width = size_t(m.rows) * es;
    for(int64_t b = 0; b < batch_count; b++)
    {
        char*       dst = static_cast<char*>(host) + size_t(b) * width * size_t(m.cols);
        const char* src = static_cast<const char*>(m.data) + size_t(b) * size_t(m.stride) * es;
        hipError_t  err = hipMemcpy2DAsync(dst,
                                          width,
                                          src,
                                          size_t(m.ld) * es,
                                          width,
                                          size_t(m.cols),
                                          hipMemcpyDeviceToHost,
                                          stream);
        if(err != hipSuccess)
        {
            // Some pitches exceed what the 2D copy accepts; fall back to one copy per column.
            (void)hipGetLastError();
            for(int64_t j = 0; j < m.cols; j++)
            {
                err = hipMemcpyAsync(dst + size_t(j) * width,
                                     src + size_t(j) * size_t(m.ld) * es,
                                     width,
                                     hipMemcpyDeviceToHost,
                                     stream);
                if(err != hipSuccess)
                    return err;
            }
        }
    }
    return hipStreamSynchronize(stream);
}

hipError_t corrupt_element_device(
    const DeviceMatrix& m, int64_t batch, int64_t row, int64_t col, hipStream_t stream)
{
    const size_t es = element_size(m.type);
    if(es == 0)
        return hipErrorInvalidValue;
    char* p = static_cast<char*>(const_cast<void*>(m.data))
              + (size_t(batch) * size_t(m.stride) + size_t(col) * size_t(m.ld) + size_t(row)) * es;
    uint64_t   current = 0;
    hipError_t err     = hipMemcpyAsync(&current, p, es, hipMemcpyDeviceToHost, stream);
    if(err == hipSuccess)
        err = hipStreamSynchronize(stream);
    if(err != hipSuccess)
        return err;
    const auto type        = hipblaslt::hostnumerics::scalarType(m.type);
    auto       replacement = hn::Tensor::allocateUninitialized(type, hn::Shape{});
    replacement.storeFrom({}, load(&current, m.type, 0) == 0 ? 1 : 0);
    err = hipMemcpyAsync(
        p, replacement.rawEncodedBackingStorage().data(), es, hipMemcpyHostToDevice, stream);
    if(err == hipSuccess)
        err = hipStreamSynchronize(stream);
    return err;
}

double load_validation_scalar(const void* data, hipDataType type, size_t i)
{
    return load(data, type, i);
}

namespace
{
    // The device region is compact on the host, with its allocation owned by Tensor.
    std::optional<hn::Tensor>
        host_region(const DeviceMatrix& m, int64_t batch_count, hipStream_t stream)
    {
        if(!element_size(m.type) || m.rows < 0 || m.cols < 0 || batch_count < 0)
            return std::nullopt;
        auto host = hn::Tensor::allocateUninitialized(
            hipblaslt::hostnumerics::scalarType(m.type),
            hn::Layout(hn::Shape{size_t(batch_count), size_t(m.rows), size_t(m.cols)},
                       {m.rows * m.cols, 1, m.rows}));
        if(copy_matrix_to_host(host.rawEncodedBackingStorage().data(), m, batch_count, stream)
           != hipSuccess)
            return std::nullopt;
        return host;
    }
}

ValidationResult verify_activation_device(const DeviceMatrix&           d,
                                          const DeviceMatrix&           e,
                                          int64_t                       batch_count,
                                          double                        scale_d,
                                          double                        scale_e,
                                          const hn::ActivationFunction& activation,
                                          hipStream_t                   stream,
                                          double*                       amax)
{
    if(amax)
        *amax = std::numeric_limits<double>::quiet_NaN();
    const auto hd = host_region(d, batch_count, stream);
    const auto he = host_region(e, batch_count, stream);
    if(!hd || !he)
        return {false, "could not copy D and E to the host\n"};
    const auto check = hn::verifyExactActivation(
        *hd, *he, hn::Tensor(scale_e), hn::Tensor(scale_d), activation, kMaxReported);
    if(amax && check.maximumAbsolute)
        *amax = check.maximumAbsolute->item<double>();
    if(check.status == hn::VerificationStatus::Passed)
        return {};
    if(check.uncertifiedElements)
        return {false,
                std::to_string(check.uncertifiedElements)
                    + " elements of E are outside the range its type stores exactly, so the "
                      "activation cannot be checked from them. Use a wider aux_type, or "
                      "smaller inputs or scales.\n"};
    if(check.status != hn::VerificationStatus::Mismatch)
        return {false, check.reason + "\n"};
    std::ostringstream message;
    message << check.comparison.mismatches
            << " elements of D do not match the activation applied to the verified E:";
    for(const auto& mismatch : check.comparison.reportedMismatches)
        message << "\n  batch " << mismatch.coordinates[0] << ", row " << mismatch.coordinates[1]
                << ", col " << mismatch.coordinates[2] << ": expected D " << mismatch.expected
                << ", got " << mismatch.observed;
    return {false, message.str() + "\n"};
}

double recover_amax_device(const DeviceMatrix& d,
                           int64_t             batch_count,
                           double              scale_d,
                           hipStream_t         stream)
{
    const auto host = host_region(d, batch_count, stream);
    if(!host)
        return std::numeric_limits<double>::quiet_NaN();
    const auto result = hn::recoverExactMaximumAbsolute(*host, hn::Tensor(scale_d));
    if(const auto* maximum = std::get_if<hn::Tensor>(&result))
        return maximum->item<double>();
    return std::numeric_limits<double>::quiet_NaN();
}

ValidationResult verify_bias_gradient(const hn::Tensor& input,
                                      char              source,
                                      hn::ScalarType    computeType,
                                      const hn::Tensor& bias)
{
    if(source != 'a' && source != 'b')
        return {false, "bias gradient source must be a or b\n"};
    const auto check = hn::verifyExactSum(
        bias, input, {source == 'a' ? size_t(1) : size_t(0)}, computeType, kMaxReported);
    if(check.status == hn::VerificationStatus::Passed)
        return {};
    if(check.status != hn::VerificationStatus::Mismatch)
        return {false, "the bias gradient cannot be checked exactly: " + check.reason + "\n"};
    std::ostringstream message;
    message << check.comparison.mismatches << " elements of the bias gradient (source " << source
            << ") are wrong:";
    for(const auto& mismatch : check.comparison.reportedMismatches)
        message << "\n  element " << mismatch.index << ": expected " << mismatch.expected
                << ", got " << mismatch.observed;
    return {false, message.str() + "\n"};
}

namespace
{
    // Host memory the operating system reports as available, or 0 when it cannot tell.
    size_t available_host_bytes()
    {
#ifdef _WIN32
        MEMORYSTATUSEX status = {};
        status.dwLength       = sizeof(status);
        return GlobalMemoryStatusEx(&status) ? size_t(status.ullAvailPhys) : 0;
#else
        std::ifstream meminfo("/proc/meminfo");
        std::string   key;
        size_t        kib = 0;
        while(meminfo >> key >> kib)
        {
            if(key == "MemAvailable:")
                return kib * 1024;
            meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        return 0;
#endif
    }
}

std::string validation_memory_shortfall(size_t device_bytes, size_t host_bytes)
{
    auto gib = [](size_t bytes) {
        std::ostringstream s;
        s.precision(3);
        s << double(bytes) / double(size_t(1) << 30) << " GiB";
        return s.str();
    };
    auto shortfall = [&]() -> std::string {
        size_t free_bytes = 0, total_bytes = 0;
        if(hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess && device_bytes > free_bytes)
            return "this case needs " + gib(device_bytes) + " of device memory and "
                   + gib(free_bytes) + " is free";

        const size_t available = available_host_bytes();
        if(available > 0 && host_bytes > available)
            return "this case needs " + gib(host_bytes) + " of host memory and " + gib(available)
                   + " is available";
        return {};
    };
    std::string why = shortfall();
    if(!why.empty())
    {
        // A previous large case can retain most of the free memory in these pools. The
        // allocator could reuse or release it, but a preflight skip never reaches allocation.
        memory_pool<d_memory>::ReleaseCached();
        memory_pool<h_memory>::ReleaseCached();
        why = shortfall();
    }
    return why;
}
