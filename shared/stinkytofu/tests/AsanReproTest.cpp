// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Standalone (non-gtest) executable that launches the InsertAsanCheckPass
// repro kernel (examples/asan-repro/) against a deliberately undersized
// host buffer and checks that the shadow-memory check catches the
// out-of-bounds access.
//
// This binary MUST be compiled with -fsanitize=address (see
// tests/CMakeLists.txt): real ASan shadow-memory poisoning only exists
// around allocations made by a process that is itself ASan-instrumented --
// see the AsanInstrument doc comment in bindings/python/Module.hpp. The GPU
// gets access to this host allocation (and therefore its ASan redzone) via
// hipHostRegister/hipHostGetDevicePointer, the same mechanism used by
// test_storeD_cload_pagefault in TensileLite's own test suite.
//
// Not a gtest case: gtest-based tests in this repo live in the ASan-free
// `unit_tests` binary, and forcing ASan onto that whole binary (and
// everything it links) for the sake of one test would be wrong. Instead
// this follows the plain-executable + SKIP_RETURN_CODE pattern already used
// for GPU/environment-dependent tests elsewhere in this repo (e.g.
// shared/origami/python/CMakeLists.txt's pytest-based add_test entries).
//
// Requires real gfx1250 hardware to mean anything; see main() for the
// runtime skip path when none is present.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

// This test cares about heap-buffer-overflow detection (the shadow check),
// not leak detection. Without this, LeakSanitizer's exit-time check flags
// unrelated small allocations made by libhsa-runtime64.so's own internal
// initialization (observed even when no GPU is present), which would abort
// the process before it reaches the skip/report-check logic below and print
// a leak report totally unrelated to this test's actual purpose.
extern "C" int __lsan_is_turned_off() {
    return 1;
}

namespace {

// Matches SKIP_RETURN_CODE set on this test in tests/CMakeLists.txt.
constexpr int kSkipExitCode = 77;

// AsanReportBuf wire format: InsertAsanCheckPass writes the failing PC here
// (see emitAsanCheck's global_store_b64 in InsertAsanCheckPass.cpp).
constexpr size_t kReportBufSize = 8;

// Byte size of the host buffer the kernel deliberately reads one byte past
// (kernel_body.s's `v_mov_b32 v1, 64` / `buffer_load_b32 ... offen`).
constexpr size_t kBufferSize = 64;

// Kernel ABI -- must match kernel_descriptor.s's .args: list.
struct KernArgs {
    void* aPtr;
    void* reportBufPtr;
};

#define HIP_CHECK(expr)                                                                      \
    do {                                                                                     \
        hipError_t _err = (expr);                                                            \
        if (_err != hipSuccess) {                                                            \
            std::fprintf(stderr, "%s:%d: HIP error %d (%s) from `%s`\n", __FILE__, __LINE__, \
                         static_cast<int>(_err), hipGetErrorString(_err), #expr);            \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

}  // namespace

int main() {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount <= 0) {
        std::printf(
            "No HIP device available -- skipping (this test requires real gfx1250 "
            "hardware).\n");
        return kSkipExitCode;
    }

    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    // gcnArchName looks like "gfx1250" or "gfx1250:sramecc+:xnack-" -- the
    // repro kernel is gfx1250-only (matches InsertAsanCheckPass's scope).
    if (std::strncmp(props.gcnArchName, "gfx1250", 7) != 0) {
        std::printf("Device 0 is %s, not gfx1250 -- skipping.\n", props.gcnArchName);
        return kSkipExitCode;
    }

    // Deliberately undersized host allocation. Real ASan places a poisoned
    // redzone immediately after it; the kernel reads byte 64 of this
    // 64-byte buffer (offset 64 == first out-of-bounds byte).
    char* buf = new char[kBufferSize];
    std::memset(buf, 0, kBufferSize);

    HIP_CHECK(hipHostRegister(buf, kBufferSize, hipHostRegisterDefault));
    void* devBufPtr = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(&devBufPtr, buf, 0));

    void* reportBuf = nullptr;
    HIP_CHECK(hipMalloc(&reportBuf, kReportBufSize));
    HIP_CHECK(hipMemset(reportBuf, 0, kReportBufSize));

    hipModule_t module = nullptr;
    HIP_CHECK(hipModuleLoad(&module, ASAN_REPRO_KERNEL_CO));
    hipFunction_t kernel = nullptr;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "asan_repro_kernel"));

    KernArgs args{devBufPtr, reportBuf};
    size_t argsSize = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize, HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr, nullptr, config));

    // Do NOT HIP_CHECK the synchronize: s_trap (no custom trap handler
    // installed) is expected to surface as a HIP error here on a real
    // violation, and we want to observe that rather than abort on it. This
    // is also why the report-buffer read below is best-effort -- whether it
    // survives a poisoned context is one of the things only real hardware
    // can answer (see the plan doc's "never run on real hardware" caveat).
    hipError_t syncResult = hipDeviceSynchronize();

    uint64_t reportedPc = 0;
    hipError_t copyResult =
        hipMemcpy(&reportedPc, reportBuf, kReportBufSize, hipMemcpyDeviceToHost);

    int exitCode = 1;
    if (copyResult == hipSuccess && reportedPc != 0) {
        std::printf(
            "PASS: ASan shadow check caught the out-of-bounds access. Failing PC = "
            "0x%llx\n",
            static_cast<unsigned long long>(reportedPc));
        exitCode = 0;
    } else if (syncResult != hipSuccess) {
        std::printf(
            "Kernel launch reported a HIP error (%d: %s) but the report buffer was not "
            "readable/nonzero (copyResult=%d, pc=0x%llx). The wave likely trapped, but the "
            "report write may not have been visible to the host -- see this test's header "
            "comment.\n",
            static_cast<int>(syncResult), hipGetErrorString(syncResult),
            static_cast<int>(copyResult), static_cast<unsigned long long>(reportedPc));
        exitCode = 1;
    } else {
        std::printf(
            "FAIL: no violation detected -- report buffer is still zero and the "
            "kernel returned success.\n");
        exitCode = 1;
    }

    // Skip cleanup on a poisoned context (matches TensileLite's own
    // test_storeD_cload_pagefault precedent for the same situation).
    if (syncResult == hipSuccess) {
        static_cast<void>(hipFree(reportBuf));
        static_cast<void>(hipModuleUnload(module));
        static_cast<void>(hipHostUnregister(buf));
    }
    delete[] buf;

    return exitCode;
}
