#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define HIP_CHECK(expr)                                                                         \
    do {                                                                                        \
        const hipError_t error_ = (expr);                                                       \
        if (error_ != hipSuccess) {                                                             \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(error_)); \
            std::exit(EXIT_FAILURE);                                                            \
        }                                                                                       \
    } while (false)

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr size_t kBytesPerLane = 32;
constexpr size_t kInputBytesPerLane = 64;

void launch(hipFunction_t function, int blocks, void* output, const void* input) {
    struct alignas(8) Args {
        void* output;
        const void* input;
    } args{output, input};
    size_t argsSize = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize, HIP_LAUNCH_PARAM_END};
    HIP_CHECK(hipModuleLaunchKernel(function, blocks, 1, 1, kThreadsPerBlock, 1, 1, 0, nullptr,
                                    nullptr, config));
    HIP_CHECK(hipDeviceSynchronize());
}

size_t countDifferentRecords(const std::vector<uint8_t>& lhs, const std::vector<uint8_t>& rhs) {
    size_t different = 0;
    for (size_t offset = 0; offset < lhs.size(); offset += kBytesPerLane)
        different += std::memcmp(lhs.data() + offset, rhs.data() + offset, kBytesPerLane) != 0;
    return different;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s CODE_OBJECT RUNS BLOCKS\n", argv[0]);
        return 2;
    }
    const int runs = std::atoi(argv[2]);
    const int blocks = std::atoi(argv[3]);
    if (runs <= 0 || blocks <= 0) {
        std::fprintf(stderr, "RUNS and BLOCKS must be positive\n");
        return 2;
    }

    hipModule_t module{};
    hipFunction_t noWait{};
    hipFunction_t wait1{};
    hipFunction_t wait0{};
    HIP_CHECK(hipModuleLoad(&module, argv[1]));
    HIP_CHECK(hipModuleGetFunction(&noWait, module, "va_vdst_cross_pipe_nowait"));
    HIP_CHECK(hipModuleGetFunction(&wait1, module, "va_vdst_cross_pipe_wait1"));
    HIP_CHECK(hipModuleGetFunction(&wait0, module, "va_vdst_cross_pipe_wait0"));

    const size_t lanes = static_cast<size_t>(blocks) * kThreadsPerBlock;
    const size_t outputBytes = lanes * kBytesPerLane;
    const size_t inputBytes = kThreadsPerBlock * kInputBytesPerLane;
    std::vector<uint8_t> hostInput(inputBytes);
    std::mt19937 random(0x5eed1250U);
    for (uint8_t& byte : hostInput) byte = static_cast<uint8_t>(random());

    void* deviceInput = nullptr;
    void* deviceNoWait = nullptr;
    void* deviceWait1 = nullptr;
    void* deviceWait0 = nullptr;
    HIP_CHECK(hipMalloc(&deviceInput, inputBytes));
    HIP_CHECK(hipMalloc(&deviceNoWait, outputBytes));
    HIP_CHECK(hipMalloc(&deviceWait1, outputBytes));
    HIP_CHECK(hipMalloc(&deviceWait0, outputBytes));
    HIP_CHECK(hipMemcpy(deviceInput, hostInput.data(), inputBytes, hipMemcpyHostToDevice));

    std::vector<uint8_t> hostNoWait(outputBytes);
    std::vector<uint8_t> hostWait1(outputBytes);
    std::vector<uint8_t> hostWait0(outputBytes);
    size_t noWaitDifferentRecords = 0;
    size_t wait1DifferentRecords = 0;
    int noWaitDifferentRuns = 0;
    int wait1DifferentRuns = 0;

    for (int run = 0; run < runs; ++run) {
        const bool reverse = (run & 1) != 0;
        launch(wait0, blocks, deviceWait0, deviceInput);
        if (reverse) {
            launch(wait1, blocks, deviceWait1, deviceInput);
            launch(noWait, blocks, deviceNoWait, deviceInput);
        } else {
            launch(noWait, blocks, deviceNoWait, deviceInput);
            launch(wait1, blocks, deviceWait1, deviceInput);
        }
        HIP_CHECK(hipMemcpy(hostWait0.data(), deviceWait0, outputBytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hostWait1.data(), deviceWait1, outputBytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hostNoWait.data(), deviceNoWait, outputBytes, hipMemcpyDeviceToHost));

        const size_t noWaitDifferent = countDifferentRecords(hostNoWait, hostWait0);
        const size_t wait1Different = countDifferentRecords(hostWait1, hostWait0);
        noWaitDifferentRecords += noWaitDifferent;
        wait1DifferentRecords += wait1Different;
        noWaitDifferentRuns += noWaitDifferent != 0;
        wait1DifferentRuns += wait1Different != 0;
    }

    const size_t laneRuns = lanes * static_cast<size_t>(runs);
    std::printf(
        "VA_VDST_CROSS_PIPE runs=%d runs blocks=%d blocks lanes=%zu lanes "
        "lane_runs=%zu lane-runs no_wait_bad=%zu lane-runs "
        "no_wait_bad_runs=%d runs wait1_bad=%zu lane-runs "
        "wait1_bad_runs=%d runs\n",
        runs, blocks, lanes, laneRuns, noWaitDifferentRecords, noWaitDifferentRuns,
        wait1DifferentRecords, wait1DifferentRuns);

    HIP_CHECK(hipFree(deviceInput));
    HIP_CHECK(hipFree(deviceNoWait));
    HIP_CHECK(hipFree(deviceWait1));
    HIP_CHECK(hipFree(deviceWait0));
    HIP_CHECK(hipModuleUnload(module));

    if (noWaitDifferentRecords == 0) {
        std::fprintf(stderr, "INCONCLUSIVE: the no-wait sensitivity control did not fail\n");
        return 3;
    }
    if (wait1DifferentRecords != 0) {
        std::fprintf(stderr, "PER_PIPE: va_vdst(1) did not retire the older XDL write\n");
        return 1;
    }
    std::puts("IN_ORDER_CROSS_PIPE: va_vdst(1) retired the older XDL write");
    return 0;
}
