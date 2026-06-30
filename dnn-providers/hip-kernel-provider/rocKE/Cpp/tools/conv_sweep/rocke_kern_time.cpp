// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// rocke_kern_time — standalone kernel timing helper.
//
// Loads an HSACO from disk, allocates device buffers, launches the kernel
// with warmup + timed iterations, and prints the average latency to stdout.
// Designed to be fork()+exec()'d by the sweep driver so each candidate gets
// a fully isolated HIP context — a hung kernel can be SIGKILL'd without
// poisoning the parent.
//
// Usage:
//   rocke_kern_time <hsaco> <kernel> <gx> <gy> <gz> <block> <sz_a> <sz_b> <sz_d>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static constexpr int kDefaultWarmup = 3;
static constexpr int kDefaultRepeat = 20;

int main(int argc, char** argv) {
    if (argc < 10) {
        fprintf(stderr,
                "Usage: %s <hsaco> <kernel> <gx> <gy> <gz> <block> <sz_a> <sz_b> <sz_d>\n",
                argv[0]);
        printf("ERROR bad_args\n");
        return 1;
    }

    const char* hsaco_path   = argv[1];
    const char* kernel_name  = argv[2];
    unsigned gx              = (unsigned)std::stoul(argv[3]);
    unsigned gy              = (unsigned)std::stoul(argv[4]);
    unsigned gz              = (unsigned)std::stoul(argv[5]);
    unsigned block_size      = (unsigned)std::stoul(argv[6]);
    size_t sz_a              = std::stoull(argv[7]);
    size_t sz_b              = std::stoull(argv[8]);
    size_t sz_d              = std::stoull(argv[9]);

    int warmup = kDefaultWarmup;
    int repeat = kDefaultRepeat;
    for (int i = 10; i < argc; ++i) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc)
            repeat = std::stoi(argv[++i]);
    }

    // 1. Read HSACO from file.
    std::ifstream f(hsaco_path, std::ios::binary | std::ios::ate);
    if (!f) {
        printf("ERROR cannot_open_hsaco\n");
        return 1;
    }
    auto size = f.tellg();
    f.seekg(0);
    std::vector<char> hsaco(size);
    f.read(hsaco.data(), size);
    if (!f) {
        printf("ERROR read_hsaco\n");
        return 1;
    }
    f.close();

    // 2. Allocate device buffers.
    void* devA = nullptr;
    void* devB = nullptr;
    void* devD = nullptr;
    hipError_t he;
    he = hipMalloc(&devA, sz_a);
    if (he != hipSuccess) { printf("ERROR hipMalloc_A %s\n", hipGetErrorString(he)); return 1; }
    he = hipMalloc(&devB, sz_b);
    if (he != hipSuccess) { printf("ERROR hipMalloc_B %s\n", hipGetErrorString(he)); return 1; }
    he = hipMalloc(&devD, sz_d);
    if (he != hipSuccess) { printf("ERROR hipMalloc_D %s\n", hipGetErrorString(he)); return 1; }
    (void)hipMemset(devA, 0, sz_a);
    (void)hipMemset(devB, 0, sz_b);

    // 3. Load module + function.
    hipModule_t mod = nullptr;
    he = hipModuleLoadData(&mod, hsaco.data());
    if (he != hipSuccess) { printf("ERROR hipModuleLoadData %s\n", hipGetErrorString(he)); return 1; }

    hipFunction_t fn = nullptr;
    he = hipModuleGetFunction(&fn, mod, kernel_name);
    if (he != hipSuccess) { printf("ERROR hipModuleGetFunction %s\n", hipGetErrorString(he)); return 1; }

    // 4. Pack kernel args.
    struct __attribute__((packed)) KernArgs {
        void* A; void* B; void* D;
        uint32_t A_bytes, B_bytes, D_bytes;
    } args = {devA, devB, devD, (uint32_t)sz_a, (uint32_t)sz_b, (uint32_t)sz_d};

    size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END
    };

    auto do_launch = [&]() -> hipError_t {
        return hipModuleLaunchKernel(fn, gx, gy, gz, block_size, 1, 1,
                                     0, nullptr, nullptr, config);
    };

    // 5. Warmup.
    for (int i = 0; i < warmup; ++i) {
        he = do_launch();
        if (he != hipSuccess) { printf("ERROR launch_warmup %s\n", hipGetErrorString(he)); return 1; }
    }
    he = hipDeviceSynchronize();
    if (he != hipSuccess) { printf("ERROR sync_warmup %s\n", hipGetErrorString(he)); return 1; }

    // 6. Timed runs.
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        he = do_launch();
        if (he != hipSuccess) { printf("ERROR launch_timed %s\n", hipGetErrorString(he)); return 1; }
    }
    he = hipDeviceSynchronize();
    if (he != hipSuccess) { printf("ERROR sync_timed %s\n", hipGetErrorString(he)); return 1; }
    auto t1 = std::chrono::steady_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double avg_us = total_us / repeat;

    printf("OK %.3f\n", avg_us);
    return 0;
}
