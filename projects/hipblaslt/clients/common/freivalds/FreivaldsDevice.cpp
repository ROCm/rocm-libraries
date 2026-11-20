// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/client/FreivaldsDevice.hpp>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace hn = roc::hostnumerics;
namespace hipblaslt::client
{
    namespace
    {
        constexpr uint64_t         P = hn::freivaldsModulus;
        __device__ inline uint64_t mod_add(uint64_t a, uint64_t b)
        {
            uint64_t s = a + b;
            return s >= P ? s - P : s;
        }

        // Residue of v. Values strictly between -P and P, which covers every value the probe sums
        // see, need no division.
        __device__ inline uint64_t to_mod(int64_t v)
        {
            constexpr int64_t p = int64_t(P);
            if(v > -p && v < p)
                return v < 0 ? uint64_t(v + p) : uint64_t(v);
            int64_t r = v % p;
            return r < 0 ? uint64_t(r + p) : uint64_t(r);
        }

        __device__ inline uint64_t device_mod_mul(uint64_t a, uint64_t b)
        {
            // a * b = hi * 2^64 + lo with hi below 2^58, and 2^64 == 8 (mod P).
            uint64_t lo = a * b;
            uint64_t hi = __umul64hi(a, b);
            uint64_t x  = (lo & P) + (lo >> 61) + (hi << 3);
            x           = (x & P) + (x >> 61);
            return x >= P ? x - P : x;
        }

        // Reads one element of D as an integer when it is an exact integer below limit in magnitude.
        template <typename T>
        __device__ inline bool device_exact_value(T x, double limit, int64_t& v, double& stored)
        {
            if constexpr(std::is_same_v<T, __half> || std::is_same_v<T, hip_bfloat16>)
                stored = float(x);
            else
                stored = double(x);
            if(!(fabs(stored) < limit))
                return false;
            v = int64_t(stored);
            return double(v) == stored;
        }

        // partial[chunk * M + i] = sum over the chunk's columns j of D(i, j) * r[j]. Elements that
        // are not exact integers below limit add nothing and are listed for the host instead.
        template <typename T>
        __global__ void row_probe_kernel(const T*            D,
                                         int64_t             M,
                                         int64_t             N,
                                         int64_t             row_stride,
                                         int64_t             column_stride,
                                         int64_t             cols_per_chunk,
                                         double              limit,
                                         const uint64_t*     r,
                                         uint64_t*           partial,
                                         unsigned long long* special_count,
                                         uint64_t*           special_offset,
                                         double*             special_value,
                                         uint64_t            special_capacity)
        {
            int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if(i >= M)
                return;
            int64_t  j0  = int64_t(blockIdx.y) * cols_per_chunk;
            int64_t  j1  = j0 + cols_per_chunk < N ? j0 + cols_per_chunk : N;
            uint64_t acc = 0;
            for(int64_t j = j0; j < j1; j++)
            {
                int64_t v;
                double  stored;
                if(device_exact_value(D[i * row_stride + j * column_stride], limit, v, stored))
                    acc = mod_add(acc, device_mod_mul(to_mod(v), r[j]));
                else
                {
                    unsigned long long slot = atomicAdd(special_count, 1ull);
                    if(slot < special_capacity)
                    {
                        special_offset[slot] = uint64_t(j) * uint64_t(M) + uint64_t(i);
                        special_value[slot]  = stored;
                    }
                }
            }
            partial[size_t(blockIdx.y) * size_t(M) + size_t(i)] = acc;
        }

        // partial[chunk * N + j] = sum over the chunk's rows i of t[i] * D(i, j), skipping the same
        // elements row_probe_kernel lists.
        template <typename T>
        __global__ void col_probe_kernel(const T*        D,
                                         int64_t         M,
                                         int64_t         N,
                                         int64_t         row_stride,
                                         int64_t         column_stride,
                                         int64_t         rows_per_chunk,
                                         double          limit,
                                         const uint64_t* t,
                                         uint64_t*       partial)
        {
            __shared__ uint64_t sums[256];
            int64_t             j   = blockIdx.x;
            int64_t             i0  = int64_t(blockIdx.y) * rows_per_chunk;
            int64_t             i1  = i0 + rows_per_chunk < M ? i0 + rows_per_chunk : M;
            uint64_t            acc = 0;
            const T*            col = D + j * column_stride;
            for(int64_t i = i0 + threadIdx.x; i < i1; i += blockDim.x)
            {
                int64_t v;
                double  stored;
                if(device_exact_value(col[i * row_stride], limit, v, stored))
                    acc = mod_add(acc, device_mod_mul(to_mod(v), t[i]));
            }
            sums[threadIdx.x] = acc;
            __syncthreads();
            for(unsigned s = blockDim.x / 2; s > 0; s /= 2)
            {
                if(threadIdx.x < s)
                    sums[threadIdx.x] = mod_add(sums[threadIdx.x], sums[threadIdx.x + s]);
                __syncthreads();
            }
            if(threadIdx.x == 0)
                partial[size_t(blockIdx.y) * size_t(N) + size_t(j)] = sums[0];
        }

        // Device memory released when it goes out of scope.
        template <typename T>
        struct DeviceArray
        {
            T*     ptr   = nullptr;
            size_t bytes = 0;
            explicit DeviceArray(size_t n)
            {
                if(n > std::numeric_limits<size_t>::max() / sizeof(T))
                    throw std::overflow_error("Freivalds device allocation size overflow.");
                if(n && hipMalloc(&ptr, n * sizeof(T)) == hipSuccess)
                    bytes = n * sizeof(T);
            }
            ~DeviceArray()
            {
                (void)hipFree(ptr);
            }
            DeviceArray(const DeviceArray&) = delete;
            DeviceArray& operator=(const DeviceArray&) = delete;
        };

        struct ProbeTiming
        {
            std::array<hipEvent_t, 3> events{};
            bool                      available = true;
            ProbeTiming()
            {
                for(auto& event : events)
                    available &= hipEventCreate(&event) == hipSuccess;
            }
            ~ProbeTiming()
            {
                for(auto event : events)
                    if(event)
                        (void)hipEventDestroy(event);
            }
            void record(size_t stage, hipStream_t stream)
            {
                if(available)
                    available &= hipEventRecord(events[stage], stream) == hipSuccess;
            }
            double seconds(size_t first, size_t last) const
            {
                float milliseconds = 0;
                if(available
                   && hipEventElapsedTime(&milliseconds, events[first], events[last]) == hipSuccess)
                    return milliseconds / 1000.0;
                return 0;
            }
        };

        void checked(hipError_t status)
        {
            if(status != hipSuccess)
                throw std::runtime_error(std::string("Freivalds device observation: ")
                                         + hipGetErrorString(status));
        }

        __global__ void gather(const unsigned char* source,
                               unsigned char*       output,
                               size_t               first,
                               size_t               count,
                               size_t               columns,
                               size_t               bytes,
                               ptrdiff_t            rowStride,
                               ptrdiff_t            columnStride)
        {
            for(size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
                index += size_t(gridDim.x) * blockDim.x)
            {
                const size_t logical = first + index;
                const auto   offset  = ptrdiff_t(logical / columns) * rowStride
                                    + ptrdiff_t(logical % columns) * columnStride;
                for(size_t byte = 0; byte < bytes; ++byte)
                    output[index * bytes + byte]
                        = source[offset * ptrdiff_t(bytes) + ptrdiff_t(byte)];
            }
        }

        hn::Tensor readback(const hn::FreivaldsRequest& request,
                            const void*                 data,
                            const hn::Layout&           layout,
                            hipStream_t                 stream,
                            FreivaldsDeviceObservation& result)
        {
            const size_t bytes = hn::scalarTypeInfo(request.outputType).storageBits / 8;
            const size_t m = request.shape[0], n = request.shape[1];
            const bool   columnMajor = layout.stride(0) == 1 && layout.stride(1) >= ptrdiff_t(m);
            auto         output      = hn::Tensor::allocateUninitialized(
                request.outputType,
                columnMajor ? hn::Layout::contiguousFirstDimensionFastest(request.shape)
                                         : hn::Layout::contiguousLastDimensionFastest(request.shape));
            auto        host = output.rawEncodedBackingStorage();
            ProbeTiming timing;
            timing.record(0, stream);
            if(columnMajor || (layout.stride(1) == 1 && layout.stride(0) >= ptrdiff_t(n)))
            {
                const size_t width = (columnMajor ? m : n) * bytes;
                checked(hipMemcpy2DAsync(host.data(),
                                         width,
                                         data,
                                         size_t(layout.stride(columnMajor ? 1 : 0)) * bytes,
                                         width,
                                         columnMajor ? n : m,
                                         hipMemcpyDeviceToHost,
                                         stream));
            }
            else
            {
                // General strided/reversed layouts use bounded device staging.
                const size_t chunk = std::min(request.shape.elementCount(), size_t(1) << 20);
                DeviceArray<unsigned char> staging(chunk * bytes);
                if(!staging.ptr)
                    throw std::runtime_error("Freivalds readback staging allocation failed.");
                result.deviceScratchBytes += staging.bytes;
                for(size_t first = 0; first < request.shape.elementCount(); first += chunk)
                {
                    const size_t count = std::min(chunk, request.shape.elementCount() - first);
                    hipLaunchKernelGGL(gather,
                                       dim3(unsigned((count + 255) / 256)),
                                       dim3(256),
                                       0,
                                       stream,
                                       static_cast<const unsigned char*>(data),
                                       staging.ptr,
                                       first,
                                       count,
                                       n,
                                       bytes,
                                       layout.stride(0),
                                       layout.stride(1));
                    checked(hipGetLastError());
                    checked(hipMemcpyAsync(host.data() + first * bytes,
                                           staging.ptr,
                                           count * bytes,
                                           hipMemcpyDeviceToHost,
                                           stream));
                    checked(hipStreamSynchronize(stream));
                }
            }
            timing.record(1, stream);
            checked(hipStreamSynchronize(stream));
            result.readbackBytes += host.size();
            result.readbackSeconds += timing.seconds(0, 1);
            return output;
        }
    }

    FreivaldsDeviceObservation observeFreivalds(const hn::FreivaldsRequest& request,
                                                const void*                 data,
                                                const hn::Layout&           layout,
                                                hipStream_t                 stream)
    {
        if(request.shape.rank() != 2 || layout.shape() != request.shape
           || hn::scalarTypeInfo(request.outputType).storageBits % 8 != 0)
            throw std::invalid_argument(
                "Freivalds requires a matching byte-addressable matrix layout.");
        if(hn::storageBytesForLayout(request.outputType, layout)
           > size_t(std::numeric_limits<ptrdiff_t>::max()))
            throw std::overflow_error("Freivalds device layout exceeds pointer arithmetic range.");
        FreivaldsDeviceObservation result;
        auto&                      empty = std::get<hn::FreivaldsObservation>(result.output);
        empty.requestIdentity            = request.identity;
        if(request.shape.elementCount() == 0)
            return result;
        if(!data)
            throw std::invalid_argument("Freivalds device output is null.");
        const size_t bytes = hn::scalarTypeInfo(request.outputType).storageBits / 8;
        data = static_cast<const unsigned char*>(data) + layout.offset() * ptrdiff_t(bytes);
        auto fallback = [&]() {
            result.output = readback(request, data, layout, stream, result);
            return std::move(result);
        };
        using hn::ScalarType;
        if(request.outputType != ScalarType::Float32 && request.outputType != ScalarType::Float64
           && request.outputType != ScalarType::Float16
           && request.outputType != ScalarType::BFloat16 && request.outputType != ScalarType::Int32
           && request.outputType != ScalarType::Int8)
            return fallback();

        // Both extents fit the HIP grid and signed chunk arithmetic. Prepared plans
        // normally reach their host work/storage budgets far below these limits.
        if(request.shape[0] > size_t(std::numeric_limits<int32_t>::max())
           || request.shape[1] > size_t(std::numeric_limits<int32_t>::max()))
            throw std::overflow_error(
                "Freivalds device projection dimensions exceed the grid limit.");
        if(!(request.fastMagnitudeLimit > 0 && request.fastMagnitudeLimit <= 0x1p61))
            throw std::invalid_argument("Freivalds device magnitude limit is invalid.");
        const int64_t m = int64_t(request.shape[0]), n = int64_t(request.shape[1]);
        for(const auto* probe : {&request.rightProbe, &request.leftProbe})
            if(probe->type() != ScalarType::UInt64
               || probe->layout() != hn::Layout::contiguousLastDimensionFastest(probe->shape()))
                throw std::invalid_argument(
                    "Freivalds device probes must be contiguous UInt64 tensors.");
        if(request.rightProbe.shape() != hn::Shape{size_t(n)}
           || request.leftProbe.shape() != hn::Shape{size_t(m)})
            throw std::invalid_argument(
                "Freivalds device probe dimensions differ from the matrix.");
        constexpr int64_t partialCap = int64_t(1) << 25;
        int64_t           colChunks  = std::min(
            int64_t(65535),
            std::clamp<int64_t>((n + 255) / 256, 1, std::max<int64_t>(1, partialCap / m)));
        const int64_t colsPerChunk = (n + colChunks - 1) / colChunks;
        colChunks                  = (n + colsPerChunk - 1) / colsPerChunk;
        int64_t rowChunks          = std::min(
            int64_t(65535),
            std::clamp<int64_t>((m + 4095) / 4096, 1, std::max<int64_t>(1, partialCap / n)));
        const int64_t rowsPerChunk = (m + rowChunks - 1) / rowChunks;
        rowChunks                  = (m + rowsPerChunk - 1) / rowsPerChunk;
        const size_t capacity
            = std::min({request.maximumExceptions, size_t(1) << 22, request.shape.elementCount()});
        DeviceArray<uint64_t>           right{size_t(n)}, left{size_t(m)};
        DeviceArray<uint64_t>           rowPartial(size_t(colChunks) * size_t(m));
        DeviceArray<uint64_t>           colPartial(size_t(rowChunks) * size_t(n));
        DeviceArray<unsigned long long> specialCount(1);
        DeviceArray<uint64_t>           offsets(std::max(size_t(1), capacity));
        DeviceArray<double>             values(std::max(size_t(1), capacity));
        result.deviceScratchBytes = right.bytes + left.bytes + rowPartial.bytes + colPartial.bytes
                                    + specialCount.bytes + offsets.bytes + values.bytes;
        if(!right.ptr || !left.ptr || !rowPartial.ptr || !colPartial.ptr || !specialCount.ptr
           || !offsets.ptr || !values.ptr)
            return fallback();
        checked(hipMemcpyAsync(right.ptr,
                               request.rightProbe.rawEncodedBackingStorage().data(),
                               size_t(n) * sizeof(uint64_t),
                               hipMemcpyHostToDevice,
                               stream));
        checked(hipMemcpyAsync(left.ptr,
                               request.leftProbe.rawEncodedBackingStorage().data(),
                               size_t(m) * sizeof(uint64_t),
                               hipMemcpyHostToDevice,
                               stream));
        checked(hipMemsetAsync(specialCount.ptr, 0, sizeof(unsigned long long), stream));
        auto               rows  = hn::Tensor::allocateUninitialized(ScalarType::UInt64,
                                                      hn::Shape{size_t(colChunks), size_t(m)});
        auto               cols  = hn::Tensor::allocateUninitialized(ScalarType::UInt64,
                                                      hn::Shape{size_t(rowChunks), size_t(n)});
        unsigned long long count = 0;
        ProbeTiming        timing;
        timing.record(0, stream);
        auto launch = [&]<class T>() {
            hipLaunchKernelGGL(row_probe_kernel<T>,
                               dim3(unsigned((m + 255) / 256), unsigned(colChunks)),
                               dim3(256),
                               0,
                               stream,
                               static_cast<const T*>(data),
                               m,
                               n,
                               layout.stride(0),
                               layout.stride(1),
                               colsPerChunk,
                               request.fastMagnitudeLimit,
                               right.ptr,
                               rowPartial.ptr,
                               specialCount.ptr,
                               offsets.ptr,
                               values.ptr,
                               uint64_t(capacity));
            checked(hipGetLastError());
            hipLaunchKernelGGL(col_probe_kernel<T>,
                               dim3(unsigned(n), unsigned(rowChunks)),
                               dim3(256),
                               0,
                               stream,
                               static_cast<const T*>(data),
                               m,
                               n,
                               layout.stride(0),
                               layout.stride(1),
                               rowsPerChunk,
                               request.fastMagnitudeLimit,
                               left.ptr,
                               colPartial.ptr);
            checked(hipGetLastError());
        };
        switch(request.outputType)
        {
        case ScalarType::Float32:
            launch.template operator()<float>();
            break;
        case ScalarType::Float64:
            launch.template operator()<double>();
            break;
        case ScalarType::Float16:
            launch.template operator()<__half>();
            break;
        case ScalarType::BFloat16:
            launch.template operator()<hip_bfloat16>();
            break;
        case ScalarType::Int32:
            launch.template operator()<int32_t>();
            break;
        case ScalarType::Int8:
            launch.template operator()<int8_t>();
            break;
        default:
            throw std::logic_error("Unexpected Freivalds device dispatch.");
        }
        timing.record(1, stream);
        for(auto pair : {std::pair{&rows, rowPartial.ptr}, std::pair{&cols, colPartial.ptr}})
            checked(hipMemcpyAsync(pair.first->rawEncodedBackingStorage().data(),
                                   pair.second,
                                   pair.first->rawEncodedBackingStorage().size(),
                                   hipMemcpyDeviceToHost,
                                   stream));
        checked(
            hipMemcpyAsync(&count, specialCount.ptr, sizeof(count), hipMemcpyDeviceToHost, stream));
        timing.record(2, stream);
        checked(hipStreamSynchronize(stream));
        result.readbackBytes += rows.rawEncodedBackingStorage().size()
                                + cols.rawEncodedBackingStorage().size() + sizeof(count);
        result.projectionSeconds += timing.seconds(0, 1);
        result.readbackSeconds += timing.seconds(1, 2);
        if(count > capacity)
            return fallback();
        std::vector<uint64_t> hostOffsets(count);
        std::vector<double>   hostValues(count);
        if(count)
        {
            timing.record(1, stream);
            checked(hipMemcpyAsync(hostOffsets.data(),
                                   offsets.ptr,
                                   count * sizeof(uint64_t),
                                   hipMemcpyDeviceToHost,
                                   stream));
            checked(hipMemcpyAsync(hostValues.data(),
                                   values.ptr,
                                   count * sizeof(double),
                                   hipMemcpyDeviceToHost,
                                   stream));
            timing.record(2, stream);
            checked(hipStreamSynchronize(stream));
            result.readbackBytes += count * (sizeof(uint64_t) + sizeof(double));
            result.readbackSeconds += timing.seconds(1, 2);
        }
        const auto               begin = std::chrono::steady_clock::now();
        hn::FreivaldsObservation observation;
        observation.requestIdentity = request.identity;
        observation.checkedElements = request.shape.elementCount();
        observation.rowSums         = hn::reduceFreivaldsProjections(rows);
        observation.columnSums      = hn::reduceFreivaldsProjections(cols);
        observation.exceptions.reserve(count);
        for(size_t i = 0; i < count; ++i)
            observation.exceptions.push_back({size_t(hostOffsets[i] % uint64_t(m)),
                                              size_t(hostOffsets[i] / uint64_t(m)),
                                              hostValues[i]});
        result.reductionSeconds
            = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        result.output = std::move(observation);
        return result;
    }
}
