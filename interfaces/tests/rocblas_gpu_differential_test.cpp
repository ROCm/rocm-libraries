// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void hip_require(hipError_t status, const char* operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

template <typename Function>
Function load_symbol(void* module, const char* name) {
    dlerror();
    void* address = dlsym(module, name);
    const char* error = dlerror();
    if (!address || error) throw std::runtime_error(std::string("missing symbol: ") + name);
    return reinterpret_cast<Function>(address);
}

struct Api {
    std::string name;
    void* module = nullptr;
    decltype(&rocblas_create_handle) create_handle = nullptr;
    decltype(&rocblas_destroy_handle) destroy_handle = nullptr;
    decltype(&rocblas_set_stream) set_stream = nullptr;
    decltype(&rocblas_get_stream) get_stream = nullptr;
    decltype(&rocblas_set_pointer_mode) set_pointer_mode = nullptr;
    decltype(&rocblas_get_pointer_mode) get_pointer_mode = nullptr;
    decltype(&rocblas_saxpy) saxpy = nullptr;
    decltype(&rocblas_saxpy_64) saxpy_64 = nullptr;
    decltype(&rocblas_scopy) scopy = nullptr;
    decltype(&rocblas_scopy_64) scopy_64 = nullptr;
    decltype(&rocblas_sscal) sscal = nullptr;
    decltype(&rocblas_sscal_64) sscal_64 = nullptr;
    decltype(&rocblas_sswap) sswap = nullptr;
    decltype(&rocblas_sswap_64) sswap_64 = nullptr;
};

Api load_api(const char* name, const char* path) {
    int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
    flags |= RTLD_DEEPBIND;
#endif
    void* module = dlopen(path, flags);
    if (!module) throw std::runtime_error(std::string("cannot load ") + name + ": " + dlerror());
    Api api;
    api.name = name;
    api.module = module;
    api.create_handle = load_symbol<decltype(api.create_handle)>(module, "rocblas_create_handle");
    api.destroy_handle =
        load_symbol<decltype(api.destroy_handle)>(module, "rocblas_destroy_handle");
    api.set_stream = load_symbol<decltype(api.set_stream)>(module, "rocblas_set_stream");
    api.get_stream = load_symbol<decltype(api.get_stream)>(module, "rocblas_get_stream");
    api.set_pointer_mode =
        load_symbol<decltype(api.set_pointer_mode)>(module, "rocblas_set_pointer_mode");
    api.get_pointer_mode =
        load_symbol<decltype(api.get_pointer_mode)>(module, "rocblas_get_pointer_mode");
    api.saxpy = load_symbol<decltype(api.saxpy)>(module, "rocblas_saxpy");
    api.saxpy_64 = load_symbol<decltype(api.saxpy_64)>(module, "rocblas_saxpy_64");
    api.scopy = load_symbol<decltype(api.scopy)>(module, "rocblas_scopy");
    api.scopy_64 = load_symbol<decltype(api.scopy_64)>(module, "rocblas_scopy_64");
    api.sscal = load_symbol<decltype(api.sscal)>(module, "rocblas_sscal");
    api.sscal_64 = load_symbol<decltype(api.sscal_64)>(module, "rocblas_sscal_64");
    api.sswap = load_symbol<decltype(api.sswap)>(module, "rocblas_sswap");
    api.sswap_64 = load_symbol<decltype(api.sswap_64)>(module, "rocblas_sswap_64");
    return api;
}

template <typename T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        hip_require(hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "hipMalloc");
    }
    ~DeviceBuffer() {
        if (data_) (void)hipFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const {
        return data_;
    }
    size_t size() const {
        return count_;
    }

   private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

struct CaseResult {
    std::array<rocblas_status, 4> statuses{};
    std::vector<float> axpy_y;
    std::vector<float> scal_x;
    std::vector<float> copy_y;
    std::vector<float> swap_x;
    std::vector<float> swap_y;
};

size_t storage_size(int64_t n, int64_t increment) {
    return n > 0 ? 1 + static_cast<size_t>(n - 1) * static_cast<size_t>(std::abs(increment)) : 1;
}

void reset(DeviceBuffer<float>& x, DeviceBuffer<float>& y, const std::vector<float>& host_x,
           const std::vector<float>& host_y, hipStream_t stream) {
    hip_require(hipMemcpyAsync(x.get(), host_x.data(), x.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream),
                "copy x to device");
    hip_require(hipMemcpyAsync(y.get(), host_y.data(), y.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream),
                "copy y to device");
}

std::pair<std::vector<float>, std::vector<float>> capture(DeviceBuffer<float>& x,
                                                          DeviceBuffer<float>& y,
                                                          hipStream_t stream,
                                                          hipEvent_t completion) {
    hip_require(hipEventRecord(completion, stream), "record completion event");
    hip_require(hipEventSynchronize(completion), "wait for completion event");
    hip_require(hipDeviceSynchronize(), "synchronize device after completion event");
    std::vector<float> host_x(x.size());
    std::vector<float> host_y(y.size());
    hip_require(hipMemcpyAsync(host_x.data(), x.get(), x.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream),
                "copy x from device");
    hip_require(hipMemcpyAsync(host_y.data(), y.get(), y.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream),
                "copy y from device");
    hip_require(hipStreamSynchronize(stream), "synchronize result copies");
    return {std::move(host_x), std::move(host_y)};
}

CaseResult run_case(const Api& api, bool index64, rocblas_pointer_mode pointer_mode,
                    hipStream_t stream, int64_t n, int64_t incx, int64_t incy) {
    std::cerr << "  " << api.name << ": setup" << std::endl;
    rocblas_handle handle = nullptr;
    require(api.create_handle(&handle) == rocblas_status_success && handle,
            api.name + ": create_handle failed");
    require(api.set_stream(handle, stream) == rocblas_status_success,
            api.name + ": set_stream failed");
    hipStream_t observed_stream = nullptr;
    require(api.get_stream(handle, &observed_stream) == rocblas_status_success &&
                observed_stream == stream,
            api.name + ": stream state mismatch");
    require(api.set_pointer_mode(handle, pointer_mode) == rocblas_status_success,
            api.name + ": set_pointer_mode failed");
    rocblas_pointer_mode observed_mode = rocblas_pointer_mode_host;
    require(api.get_pointer_mode(handle, &observed_mode) == rocblas_status_success &&
                observed_mode == pointer_mode,
            api.name + ": pointer-mode state mismatch");

    const size_t x_size = storage_size(n, incx);
    const size_t y_size = storage_size(n, incy);
    std::vector<float> host_x(x_size);
    std::vector<float> host_y(y_size);
    for (size_t i = 0; i < x_size; ++i) host_x[i] = static_cast<float>((i % 13) + 1);
    for (size_t i = 0; i < y_size; ++i) host_y[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
    DeviceBuffer<float> x(x_size);
    DeviceBuffer<float> y(y_size);
    DeviceBuffer<float> device_alpha(1);
    const float host_alpha = 2.0f;
    hip_require(hipMemcpyAsync(device_alpha.get(), &host_alpha, sizeof(host_alpha),
                               hipMemcpyHostToDevice, stream),
                "copy alpha to device");
    const float* alpha =
        pointer_mode == rocblas_pointer_mode_device ? device_alpha.get() : &host_alpha;
    std::cerr << "  " << api.name << ": pointers alpha=" << static_cast<const void*>(alpha)
              << " x=" << static_cast<void*>(x.get()) << " y=" << static_cast<void*>(y.get())
              << " mode=" << static_cast<unsigned>(pointer_mode) << std::endl;
    hipEvent_t completion = nullptr;
    hip_require(hipEventCreateWithFlags(&completion, hipEventDisableTiming), "create event");

    CaseResult result;
    reset(x, y, host_x, host_y, stream);
    std::cerr << "  " << api.name << ": axpy" << std::endl;
    result.statuses[0] = index64 ? api.saxpy_64(handle, n, alpha, x.get(), incx, y.get(), incy)
                                 : api.saxpy(handle, static_cast<rocblas_int>(n), alpha, x.get(),
                                             static_cast<rocblas_int>(incx), y.get(),
                                             static_cast<rocblas_int>(incy));
    std::cerr << "  " << api.name << ": axpy status=" << result.statuses[0] << std::endl;
    result.axpy_y = capture(x, y, stream, completion).second;

    reset(x, y, host_x, host_y, stream);
    std::cerr << "  " << api.name << ": scal" << std::endl;
    result.statuses[1] = index64 ? api.sscal_64(handle, n, alpha, x.get(), incx)
                                 : api.sscal(handle, static_cast<rocblas_int>(n), alpha, x.get(),
                                             static_cast<rocblas_int>(incx));
    result.scal_x = capture(x, y, stream, completion).first;

    reset(x, y, host_x, host_y, stream);
    std::cerr << "  " << api.name << ": copy" << std::endl;
    result.statuses[2] = index64 ? api.scopy_64(handle, n, x.get(), incx, y.get(), incy)
                                 : api.scopy(handle, static_cast<rocblas_int>(n), x.get(),
                                             static_cast<rocblas_int>(incx), y.get(),
                                             static_cast<rocblas_int>(incy));
    result.copy_y = capture(x, y, stream, completion).second;

    reset(x, y, host_x, host_y, stream);
    std::cerr << "  " << api.name << ": swap" << std::endl;
    result.statuses[3] = index64 ? api.sswap_64(handle, n, x.get(), incx, y.get(), incy)
                                 : api.sswap(handle, static_cast<rocblas_int>(n), x.get(),
                                             static_cast<rocblas_int>(incx), y.get(),
                                             static_cast<rocblas_int>(incy));
    auto swapped = capture(x, y, stream, completion);
    result.swap_x = std::move(swapped.first);
    result.swap_y = std::move(swapped.second);

    hip_require(hipEventDestroy(completion), "destroy completion event");
    require(api.destroy_handle(handle) == rocblas_status_success,
            api.name + ": destroy_handle failed");
    return result;
}

void compare(const CaseResult& expected, const CaseResult& actual, const std::string& label) {
    require(expected.statuses == actual.statuses, label + ": operation statuses differ");
    require(expected.axpy_y == actual.axpy_y, label + ": AXPY output differs");
    require(expected.scal_x == actual.scal_x, label + ": SCAL output differs");
    require(expected.copy_y == actual.copy_y, label + ": COPY output differs");
    require(expected.swap_x == actual.swap_x, label + ": SWAP x output differs");
    require(expected.swap_y == actual.swap_y, label + ": SWAP y output differs");
    require(std::all_of(actual.statuses.begin(), actual.statuses.end(),
                        [](rocblas_status status) { return status == rocblas_status_success; }),
            label + ": a valid operation failed");
}

std::vector<rocblas_status> edge_statuses(const Api& api, bool index64) {
    std::cerr << "running " << api.name << "/" << (index64 ? "i64" : "i32") << " edge cases"
              << std::endl;
    rocblas_handle handle = nullptr;
    require(api.create_handle(&handle) == rocblas_status_success && handle,
            api.name + ": edge-test handle creation failed");
    float alpha = 1.0f;
    DeviceBuffer<float> x(1);
    DeviceBuffer<float> y(1);
    hip_require(hipMemset(x.get(), 0, sizeof(float)), "initialize edge-test x");
    hip_require(hipMemset(y.get(), 0, sizeof(float)), "initialize edge-test y");
    std::vector<rocblas_status> result;
    if (index64) {
        result.push_back(api.saxpy_64(handle, 0, nullptr, nullptr, 1, nullptr, 1));
        result.push_back(api.sscal_64(handle, 0, nullptr, nullptr, 1));
        result.push_back(api.scopy_64(handle, 0, nullptr, 1, nullptr, 1));
        result.push_back(api.sswap_64(handle, 0, nullptr, 1, nullptr, 1));
        result.push_back(api.saxpy_64(nullptr, 1, &alpha, x.get(), 1, y.get(), 1));
        result.push_back(api.saxpy_64(handle, 1, nullptr, x.get(), 1, y.get(), 1));
        result.push_back(api.saxpy_64(handle, 1, &alpha, x.get(), 0, y.get(), 1));
    } else {
        result.push_back(api.saxpy(handle, 0, nullptr, nullptr, 1, nullptr, 1));
        result.push_back(api.sscal(handle, 0, nullptr, nullptr, 1));
        result.push_back(api.scopy(handle, 0, nullptr, 1, nullptr, 1));
        result.push_back(api.sswap(handle, 0, nullptr, 1, nullptr, 1));
        result.push_back(api.saxpy(nullptr, 1, &alpha, x.get(), 1, y.get(), 1));
        result.push_back(api.saxpy(handle, 1, nullptr, x.get(), 1, y.get(), 1));
        result.push_back(api.saxpy(handle, 1, &alpha, x.get(), 0, y.get(), 1));
    }
    hip_require(hipDeviceSynchronize(), "synchronize edge-case calls");
    require(api.destroy_handle(handle) == rocblas_status_success,
            api.name + ": edge-test handle destruction failed");
    return result;
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
            std::cout << "SKIP: no usable AMD GPU\n";
            return 77;
        }
        hip_require(hipSetDevice(0), "hipSetDevice");
        setenv("ROCM_INTERFACES_REAL_ROCBLAS_LIBRARY", REAL_ROCBLAS_PATH, 1);
        setenv("ROCM_INTERFACES_ROCBLAS_BRIDGE_PROVIDER", REAL_PROVIDER_PATH, 1);
        setenv("ROCM_INTERFACES_BLAS_V2_PROVIDER", REAL_NARROW_PROVIDER_PATH, 1);

        Api direct = load_api("canonical", REAL_ROCBLAS_PATH);
        Api exhaustive = load_api("exhaustive", EXHAUSTIVE_LOADER_PATH);
        Api narrow = load_api("narrow-v2", NARROW_LOADER_PATH);

        hipStream_t first = nullptr;
        hipStream_t second = nullptr;
        hip_require(hipStreamCreate(&first), "create first stream");
        hip_require(hipStreamCreate(&second), "create second stream");
        struct TestCase {
            bool index64;
            rocblas_pointer_mode pointer_mode;
            hipStream_t stream;
            int64_t n;
            int64_t incx;
            int64_t incy;
            const char* name;
        };
        const std::array<TestCase, 6> cases{{
            {false, rocblas_pointer_mode_host, first, 257, 2, 3, "i32-host-strided"},
            {true, rocblas_pointer_mode_host, second, 257, 2, 3, "i64-host-strided"},
            {false, rocblas_pointer_mode_device, second, 129, 1, 1, "i32-device-scalar"},
            {true, rocblas_pointer_mode_device, first, 129, 1, 1, "i64-device-scalar"},
            {false, rocblas_pointer_mode_host, first, 31, -2, -3, "i32-negative-inc"},
            {true, rocblas_pointer_mode_host, second, 31, -2, -3, "i64-negative-inc"},
        }};
        for (const TestCase& test : cases) {
            std::cerr << "running canonical/" << test.name << std::endl;
            const CaseResult expected = run_case(direct, test.index64, test.pointer_mode,
                                                 test.stream, test.n, test.incx, test.incy);
            std::cerr << "running exhaustive/" << test.name << std::endl;
            compare(expected,
                    run_case(exhaustive, test.index64, test.pointer_mode, test.stream, test.n,
                             test.incx, test.incy),
                    std::string("exhaustive/") + test.name);
            std::cerr << "running narrow-v2/" << test.name << std::endl;
            compare(expected,
                    run_case(narrow, test.index64, test.pointer_mode, test.stream, test.n,
                             test.incx, test.incy),
                    std::string("narrow-v2/") + test.name);
        }

        for (bool index64 : {false, true}) {
            const auto expected = edge_statuses(direct, index64);
            require(expected == edge_statuses(exhaustive, index64),
                    "exhaustive quick-return/invalid-argument behavior differs");
            require(expected == edge_statuses(narrow, index64),
                    "narrow-v2 quick-return/invalid-argument behavior differs");
        }

        hip_require(hipStreamDestroy(first), "destroy first stream");
        hip_require(hipStreamDestroy(second), "destroy second stream");
        dlclose(narrow.module);
        dlclose(exhaustive.module);
        dlclose(direct.module);
        std::cout << "GPU differential checks passed for canonical, exhaustive, and narrow-v2\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
