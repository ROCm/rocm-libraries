// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Grouped GEMM with an activation and a gate tensor, in one launch:
//
//     D_g = silu(A_g * B_g) * G_g        for every group g
//
// This is the gated feed-forward of a mixture-of-experts layer. Group g is one expert:
// A_g holds the tokens routed to it, B_g its gate-projection weights, and G_g the
// up-projection result (same shape as D_g) computed by an earlier GEMM.
//
// NN, column major, bf16 in / bf16 out with f32 accumulation:
//     a = (m, k), lda = m      b = (k, n), ldb = k      c = d = gate = (m, n), ld = m

#include <cmath>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "helper.h"

namespace
{
    // bf16 is the upper 16 bits of an f32. Round to nearest even on the way down.
    uint16_t toBf16(float v)
    {
        uint32_t u;
        std::memcpy(&u, &v, sizeof(u));
        u += 0x7fffu + ((u >> 16) & 1u);
        return static_cast<uint16_t>(u >> 16);
    }

    float fromBf16(uint16_t h)
    {
        uint32_t u = static_cast<uint32_t>(h) << 16;
        float    v;
        std::memcpy(&v, &u, sizeof(v));
        return v;
    }

    std::vector<uint16_t> randomBf16(size_t count, float lo, float hi, std::mt19937& rng)
    {
        std::uniform_real_distribution<float> dist(lo, hi);
        std::vector<uint16_t>                 v(count);
        for(auto& e : v)
            e = toBf16(dist(rng));
        return v;
    }

    void* toDevice(const std::vector<uint16_t>& host)
    {
        void* dev = nullptr;
        CHECK_HIP_ERROR(hipMalloc(&dev, host.size() * sizeof(uint16_t)));
        CHECK_HIP_ERROR(
            hipMemcpy(dev, host.data(), host.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        return dev;
    }
}

int main()
{
    // Three experts with a different number of routed tokens each.
    std::vector<int64_t> m           = {96, 256, 40};
    std::vector<int64_t> n           = {512, 512, 512};
    std::vector<int64_t> k           = {256, 256, 256};
    std::vector<int64_t> batch_count = {1, 1, 1};
    const size_t         groups      = m.size();
    float                alpha       = 1.0f;
    float                beta        = 0.0f;

    hipStream_t       stream;
    hipblasLtHandle_t handle;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

    std::mt19937                       rng(2026);
    std::vector<std::vector<uint16_t>> a(groups), b(groups), gate(groups), d(groups);
    std::vector<void*>                 d_a(groups), d_b(groups), d_c(groups), d_d(groups);
    std::vector<void*>                 d_gate(groups);
    for(size_t g = 0; g < groups; g++)
    {
        a[g]    = randomBf16(m[g] * k[g], -1.0f, 1.0f, rng);
        b[g]    = randomBf16(k[g] * n[g], -0.125f, 0.125f, rng);
        gate[g] = randomBf16(m[g] * n[g], 0.5f, 1.5f, rng);
        d[g].assign(m[g] * n[g], 0);
        d_a[g]    = toDevice(a[g]);
        d_b[g]    = toDevice(b[g]);
        d_gate[g] = toDevice(gate[g]);
        d_c[g]    = toDevice(d[g]); // beta is 0; C only has to be valid memory
        d_d[g]    = toDevice(d[g]);
    }

    const int64_t max_workspace_size = 32 * 1024 * 1024;
    void*         d_workspace        = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));

    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::GroupedGemm groupedgemm(handle,
                                           HIPBLAS_OP_N,
                                           HIPBLAS_OP_N,
                                           HIP_R_16BF,
                                           HIP_R_16BF,
                                           HIP_R_16BF,
                                           HIP_R_16BF,
                                           HIPBLAS_COMPUTE_32F);

    // One epilogue for all groups (it broadcasts): swish with beta 1 is silu, then the gate.
    hipblaslt_ext::GemmEpilogue gatedSilu;
    gatedSilu.setMode(HIPBLASLT_EPILOGUE_SWISH_EXT);
    gatedSilu.setGateMode(hipblaslt_ext::GateMode::Multiply);
    gatedSilu.setGateDataType(HIP_R_16BF);
    std::vector<hipblaslt_ext::GemmEpilogue> epilogue{gatedSilu};

    std::vector<hipblaslt_ext::GemmInputs> inputs(groups);
    for(size_t g = 0; g < groups; g++)
    {
        inputs[g].setA(d_a[g]);
        inputs[g].setB(d_b[g]);
        inputs[g].setC(d_c[g]);
        inputs[g].setD(d_d[g]);
        inputs[g].setAlpha(&alpha);
        inputs[g].setBeta(&beta);
        inputs[g].setGate(d_gate[g]); // (m, n), leading dimension m
    }
    CHECK_HIPBLASLT_ERROR(groupedgemm.setProblem(m, n, k, batch_count, epilogue, inputs));

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(groupedgemm.algoGetHeuristic(1, gemmPref, heuristicResult));
    if(heuristicResult.empty())
    {
        std::cerr << "No valid solution found!" << std::endl;
        return EXIT_FAILURE;
    }

    // The gate pointers travel with the arguments the library packs for each group, so
    // initialize without device user arguments and launch with run(stream).
    groupedgemm.setMaxWorkspaceBytes(max_workspace_size);
    CHECK_HIPBLASLT_ERROR(
        groupedgemm.initialize(heuristicResult[0].algo, d_workspace, false, stream));
    CHECK_HIPBLASLT_ERROR(groupedgemm.run(stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    // Check against a CPU reference: f32 accumulate, silu, gate, then round to bf16.
    double maxRelErr = 0.0;
    for(size_t g = 0; g < groups; g++)
    {
        CHECK_HIP_ERROR(hipMemcpy(
            d[g].data(), d_d[g], d[g].size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
        for(int64_t j = 0; j < n[g]; j++)
        {
            for(int64_t i = 0; i < m[g]; i++)
            {
                float acc = 0.0f;
                for(int64_t l = 0; l < k[g]; l++)
                    acc += fromBf16(a[g][i + l * m[g]]) * fromBf16(b[g][l + j * k[g]]);
                const float silu = acc / (1.0f + std::exp(-acc));
                const float ref  = fromBf16(toBf16(silu * fromBf16(gate[g][i + j * m[g]])));
                const float got  = fromBf16(d[g][i + j * m[g]]);
                maxRelErr
                    = std::max(maxRelErr,
                               static_cast<double>(std::fabs(got - ref))
                                   / std::max(1e-3, static_cast<double>(std::fabs(ref))));
            }
        }
    }
    // bf16 keeps 8 significant bits and the summation order differs from the reference.
    const bool pass = maxRelErr < 5e-2;
    std::cout << "groups: " << groups << ", max relative error: " << maxRelErr << " -> "
              << (pass ? "PASS" : "FAIL") << std::endl;

    for(size_t g = 0; g < groups; g++)
    {
        CHECK_HIP_ERROR(hipFree(d_a[g]));
        CHECK_HIP_ERROR(hipFree(d_b[g]));
        CHECK_HIP_ERROR(hipFree(d_c[g]));
        CHECK_HIP_ERROR(hipFree(d_d[g]));
        CHECK_HIP_ERROR(hipFree(d_gate[g]));
    }
    CHECK_HIP_ERROR(hipFree(d_workspace));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
