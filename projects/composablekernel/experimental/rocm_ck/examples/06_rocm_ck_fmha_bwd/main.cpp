// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side loader for the FMHA BWD kpack example. Loads kernel variants
// from a kpack archive and verifies each one against a CPU reference.
//
// Three kernel families:
//   1. OGradDotO:  D[b][h][s]    = sum_v(dO * O) * p_undrop
//   2. DqDkDv:     dQ, dK, dV    = main backward attention GEMMs
//   3. ConvertDQ:  dQ (fp16/bf16) = reduce + type-convert dQ_acc (fp32)
//
// The full FMHA BWD pipeline runs: OGradDotO -> DqDkDv -> (ConvertDQ if det).

#include "rocm_fmha_bwd_registry.hpp"

#include <rocm_ck/args.hpp>
#include <rocm_ck/datatype_convert.hpp>
#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/gpu_arch.hpp>
#include <rocm_ck/hip_check.hpp>

#include <hip/hip_runtime.h>

#include <rocm_kpack/kpack.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

// Namespace aliases for named slot constants.
namespace ODO = rocm_ck::fmha_bwd_ograd_dot_o_slots;
namespace DKV = rocm_ck::fmha_bwd_dqdkdv_slots;

// ---------------------------------------------------------------------------
// Test parameters
// ---------------------------------------------------------------------------
static constexpr int BATCH      = 2;
static constexpr int NHEAD      = 2;
static constexpr int SEQLEN_Q   = 128;
static constexpr int SEQLEN_K   = 128;
static constexpr int HDIM_Q     = 128;
static constexpr int HDIM_V     = 128;
static constexpr float P_UNDROP = 1.0f;

// ---------------------------------------------------------------------------
// CPU reference: OGradDotO
// ---------------------------------------------------------------------------

/// Compute D = rowSum(dO * O) * p_undrop on the host.
/// Layout: [batch, nhead, seqlen_q, hdim_v] row-major.
static void cpuOGradDotO(const std::vector<float>& O,
                         const std::vector<float>& dO,
                         std::vector<float>& D,
                         int batch,
                         int nhead,
                         int seqlen_q,
                         int hdim_v,
                         float p_undrop)
{
    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < nhead; ++h)
        {
            for(int s = 0; s < seqlen_q; ++s)
            {
                float acc = 0.0f;
                for(int v = 0; v < hdim_v; ++v)
                {
                    int idx = ((b * nhead + h) * seqlen_q + s) * hdim_v + v;
                    acc += dO[idx] * O[idx];
                }
                int d_idx = (b * nhead + h) * seqlen_q + s;
                D[d_idx]  = acc * p_undrop;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference: full FMHA BWD (plain config, no mask/dropout/bias)
// ---------------------------------------------------------------------------

/// Compute the full FMHA backward pass on the host (float precision).
///
/// All tensors use row-major layout [batch, nhead, seqlen, hdim].
/// D must already be computed (via cpuOGradDotO).
///
/// Algorithm:
///   S[i][j]  = scale * sum_k(Q[i][k] * K[j][k])
///   P[i][j]  = exp(S[i][j] - LSE[i])            (numerically stable softmax)
///   dP[i][j] = sum_v(dO[i][v] * V[j][v])
///   dS[i][j] = P[i][j] * (dP[i][j] - D[i])
///   dQ[i][k] = scale * sum_j(dS[i][j] * K[j][k])
///   dK[j][k] = scale * sum_i(dS[i][j] * Q[i][k])
///   dV[j][v] = sum_i(P[i][j] * dO[i][v])
static void cpuFmhaBwd(const std::vector<float>& Q,
                       const std::vector<float>& K,
                       const std::vector<float>& V,
                       const std::vector<float>& O,
                       const std::vector<float>& dO,
                       const std::vector<float>& D,
                       std::vector<float>& LSE,
                       std::vector<float>& dQ,
                       std::vector<float>& dK,
                       std::vector<float>& dV,
                       int batch,
                       int nhead,
                       int seqlen_q,
                       int seqlen_k,
                       int hdim_q,
                       int hdim_v,
                       float scale)
{
    // Helper lambdas for indexing into [batch, nhead, seq, dim].
    auto idx4 = [&](int b, int h, int s, int d, int seq, int dim) {
        return ((b * nhead + h) * seq + s) * dim + d;
    };
    // Index into [batch, nhead, seq] (for D, LSE).
    auto idx3 = [&](int b, int h, int s) { return (b * nhead + h) * seqlen_q + s; };

    const int sq = seqlen_q;
    const int sk = seqlen_k;
    const int dq = hdim_q;
    const int dv = hdim_v;

    // Temporary buffers per (batch, head).
    std::vector<float> S(sq * sk);
    std::vector<float> P(sq * sk);
    std::vector<float> dP(sq * sk);
    std::vector<float> dS(sq * sk);

    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < nhead; ++h)
        {
            // --- S = scale * Q @ K^T ---
            for(int i = 0; i < sq; ++i)
            {
                for(int j = 0; j < sk; ++j)
                {
                    float acc = 0.0f;
                    for(int k = 0; k < dq; ++k)
                    {
                        acc += Q[idx4(b, h, i, k, sq, dq)] * K[idx4(b, h, j, k, sk, dq)];
                    }
                    S[i * sk + j] = scale * acc;
                }
            }

            // --- LSE and P = softmax(S) ---
            // LSE[i] = log(sum_j(exp(S[i][j])))
            // P[i][j] = exp(S[i][j] - LSE[i])
            for(int i = 0; i < sq; ++i)
            {
                float max_s = S[i * sk];
                for(int j = 1; j < sk; ++j)
                    max_s = std::max(max_s, S[i * sk + j]);

                float sum_exp = 0.0f;
                for(int j = 0; j < sk; ++j)
                    sum_exp += std::exp(S[i * sk + j] - max_s);

                float lse_val      = max_s + std::log(sum_exp);
                LSE[idx3(b, h, i)] = lse_val;

                for(int j = 0; j < sk; ++j)
                    P[i * sk + j] = std::exp(S[i * sk + j] - lse_val);
            }

            // --- dP = dO @ V^T ---
            for(int i = 0; i < sq; ++i)
            {
                for(int j = 0; j < sk; ++j)
                {
                    float acc = 0.0f;
                    for(int v = 0; v < dv; ++v)
                    {
                        acc += dO[idx4(b, h, i, v, sq, dv)] * V[idx4(b, h, j, v, sk, dv)];
                    }
                    dP[i * sk + j] = acc;
                }
            }

            // --- dS = P * (dP - D) ---
            for(int i = 0; i < sq; ++i)
            {
                float d_val = D[idx3(b, h, i)];
                for(int j = 0; j < sk; ++j)
                {
                    dS[i * sk + j] = P[i * sk + j] * (dP[i * sk + j] - d_val);
                }
            }

            // --- dQ = scale * dS @ K ---
            for(int i = 0; i < sq; ++i)
            {
                for(int k = 0; k < dq; ++k)
                {
                    float acc = 0.0f;
                    for(int j = 0; j < sk; ++j)
                    {
                        acc += dS[i * sk + j] * K[idx4(b, h, j, k, sk, dq)];
                    }
                    dQ[idx4(b, h, i, k, sq, dq)] = scale * acc;
                }
            }

            // --- dK = scale * dS^T @ Q ---
            for(int j = 0; j < sk; ++j)
            {
                for(int k = 0; k < dq; ++k)
                {
                    float acc = 0.0f;
                    for(int i = 0; i < sq; ++i)
                    {
                        acc += dS[i * sk + j] * Q[idx4(b, h, i, k, sq, dq)];
                    }
                    dK[idx4(b, h, j, k, sk, dq)] = scale * acc;
                }
            }

            // --- dV = P^T @ dO ---
            for(int j = 0; j < sk; ++j)
            {
                for(int v = 0; v < dv; ++v)
                {
                    float acc = 0.0f;
                    for(int i = 0; i < sq; ++i)
                    {
                        acc += P[i * sk + j] * dO[idx4(b, h, i, v, sq, dv)];
                    }
                    dV[idx4(b, h, j, v, sk, dv)] = acc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

/// Fill O and dO with small integers exactly representable in fp16/bf16.
static void makeOGradDotOTestData(
    int batch, int nhead, int seqlen_q, int hdim_v, std::vector<float>& O, std::vector<float>& dO)
{
    const int total = batch * nhead * seqlen_q * hdim_v;
    O.resize(total);
    dO.resize(total);
    for(int i = 0; i < total; ++i)
    {
        O[i]  = static_cast<float>(i % 16);
        dO[i] = static_cast<float>((i * 3) % 16);
    }
}

/// Fill Q, K, V, dO with small integers for the DqDkDv test.
/// Values in [0..7] to keep intermediate products small and avoid
/// overflow in fp16 after accumulation.
static void makeDqDkDvTestData(int batch,
                               int nhead,
                               int seqlen_q,
                               int seqlen_k,
                               int hdim_q,
                               int hdim_v,
                               std::vector<float>& Q,
                               std::vector<float>& K,
                               std::vector<float>& V,
                               std::vector<float>& dO)
{
    const int qkv_q  = batch * nhead * seqlen_q * hdim_q;
    const int qkv_k  = batch * nhead * seqlen_k * hdim_q;
    const int v_tot  = batch * nhead * seqlen_k * hdim_v;
    const int do_tot = batch * nhead * seqlen_q * hdim_v;

    Q.resize(qkv_q);
    K.resize(qkv_k);
    V.resize(v_tot);
    dO.resize(do_tot);

    // Use small values to avoid fp16 overflow in attention scores.
    for(int i = 0; i < qkv_q; ++i)
        Q[i] = static_cast<float>((i * 3 + 1) % 8);
    for(int i = 0; i < qkv_k; ++i)
        K[i] = static_cast<float>((i * 5 + 2) % 8);
    for(int i = 0; i < v_tot; ++i)
        V[i] = static_cast<float>((i * 7 + 3) % 8);
    for(int i = 0; i < do_tot; ++i)
        dO[i] = static_cast<float>((i * 11 + 5) % 8);
}

// ---------------------------------------------------------------------------
// Helper: load kernel from kpack archive
// ---------------------------------------------------------------------------

struct LoadedKernel
{
    hipModule_t module;
    hipFunction_t function;
    void* code_object;
};

static bool
loadKernel(kpack_archive_t archive, const char* name, const char* gpu_arch, LoadedKernel& out)
{
    out                = {};
    size_t code_size   = 0;
    kpack_error_t kerr = kpack_get_kernel(archive, name, gpu_arch, &out.code_object, &code_size);
    if(kerr != KPACK_SUCCESS)
    {
        std::fprintf(stderr, "  %s: no kernel for '%s' (error %d)\n", name, gpu_arch, kerr);
        return false;
    }
    HIP_CHECK(hipModuleLoadData(&out.module, out.code_object));
    HIP_CHECK(hipModuleGetFunction(&out.function, out.module, name));
    return true;
}

static void unloadKernel(LoadedKernel& k)
{
    HIP_CHECK(hipModuleUnload(k.module));
    kpack_free_kernel(k.code_object);
}

// ---------------------------------------------------------------------------
// Helper: launch a kernel with rocm_ck::Args
// ---------------------------------------------------------------------------

static void launchArgs(hipFunction_t func, dim3 grid, int block_size, rocm_ck::Args& args)
{
    size_t args_size   = sizeof(args);
    void* launch_cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                          &args,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &args_size,
                          HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(
        func, grid.x, grid.y, grid.z, block_size, 1, 1, 0, nullptr, nullptr, launch_cfg));
    HIP_CHECK(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Helper: set up a TensorArg
// ---------------------------------------------------------------------------

static void setTensor(rocm_ck::TensorArg& t,
                      const void* ptr,
                      std::initializer_list<rocm_ck::index_t> lens,
                      std::initializer_list<int64_t> strs)
{
    t.ptr = ptr;
    std::memset(t.lengths, 0, sizeof(t.lengths));
    std::memset(t.strides, 0, sizeof(t.strides));
    int i = 0;
    for(auto l : lens)
    {
        if(i < rocm_ck::kMaxRank)
            t.lengths[i++] = l;
    }
    i = 0;
    for(auto s : strs)
    {
        if(i < rocm_ck::kMaxRank)
            t.strides[i++] = s;
    }
}

// ---------------------------------------------------------------------------
// OGradDotO batch-mode runner (using rocm_ck::Args)
// ---------------------------------------------------------------------------

static bool runOGradDotOBatch(const rocm_ck::FmhaBwdOGradDotOVariant& variant,
                              kpack_archive_t archive,
                              const char* gpu_arch,
                              const std::vector<float>& host_O,
                              const std::vector<float>& host_dO,
                              const std::vector<float>& ref_D,
                              int batch,
                              int nhead,
                              int seqlen_q,
                              int hdim_v,
                              float p_undrop)
{
    const auto dtype      = variant.kernel.dtype;
    const int dtype_bytes = rocm_ck::data_type_bits(dtype) / 8;
    const size_t total_el = size_t(batch) * nhead * seqlen_q * hdim_v;
    const size_t total_d  = size_t(batch) * nhead * seqlen_q;
    const size_t buf_size = total_el * dtype_bytes;
    const size_t d_size   = total_d * sizeof(float);

    LoadedKernel lk;
    if(!loadKernel(archive, variant.name, gpu_arch, lk))
        return false;

    // Device buffers
    void* dev_O  = nullptr;
    void* dev_dO = nullptr;
    void* dev_D  = nullptr;
    HIP_CHECK(hipMalloc(&dev_O, buf_size));
    HIP_CHECK(hipMalloc(&dev_dO, buf_size));
    HIP_CHECK(hipMalloc(&dev_D, d_size));

    // Convert and upload
    std::vector<char> typed_O(buf_size), typed_dO(buf_size);
    for(size_t i = 0; i < total_el; ++i)
    {
        rocm_ck::float_to_typed(dtype, host_O[i], typed_O.data() + i * dtype_bytes);
        rocm_ck::float_to_typed(dtype, host_dO[i], typed_dO.data() + i * dtype_bytes);
    }
    HIP_CHECK(hipMemcpy(dev_O, typed_O.data(), buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_dO, typed_dO.data(), buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dev_D, 0, d_size));

    // Strides: [batch, nhead, seqlen_q, hdim_v] row-major
    const int64_t stride_o       = hdim_v;
    const int64_t nhead_stride_o = int64_t(seqlen_q) * hdim_v;
    const int64_t batch_stride_o = int64_t(nhead) * seqlen_q * hdim_v;
    const int64_t nhead_stride_d = seqlen_q;
    const int64_t batch_stride_d = int64_t(nhead) * seqlen_q;

    // Populate Args
    rocm_ck::Args args{};
    setTensor(args.tensors[ODO::O],
              dev_O,
              {static_cast<rocm_ck::index_t>(seqlen_q), static_cast<rocm_ck::index_t>(hdim_v)},
              {stride_o, nhead_stride_o, batch_stride_o});
    setTensor(args.tensors[ODO::DO],
              dev_dO,
              {static_cast<rocm_ck::index_t>(seqlen_q), static_cast<rocm_ck::index_t>(hdim_v)},
              {stride_o, nhead_stride_o, batch_stride_o});
    setTensor(args.tensors[ODO::D],
              dev_D,
              {static_cast<rocm_ck::index_t>(seqlen_q)},
              {nhead_stride_d, batch_stride_d});

    args.scalars[ODO::P_UNDROP].f32 = p_undrop;

    // Launch
    dim3 grid = rocm_ck::ograd_dot_o_grid_size(batch, nhead, seqlen_q, variant.kernel.block_size);
    std::printf("  %s: grid=(%u,%u,%u), block=%d\n",
                variant.name,
                grid.x,
                grid.y,
                grid.z,
                variant.kernel.block_size);

    launchArgs(lk.function, grid, variant.kernel.block_size, args);

    // Verify
    std::vector<float> got_D(total_d);
    HIP_CHECK(hipMemcpy(got_D.data(), dev_D, d_size, hipMemcpyDeviceToHost));

    const float tol = rocm_ck::tolerance_for(dtype);
    bool passed     = true;
    for(size_t i = 0; i < total_d; ++i)
    {
        if(std::fabs(got_D[i] - ref_D[i]) > tol)
        {
            std::fprintf(stderr,
                         "  %s: MISMATCH at D[%zu]: got %f, expected %f "
                         "(diff=%e)\n",
                         variant.name,
                         i,
                         got_D[i],
                         ref_D[i],
                         got_D[i] - ref_D[i]);
            passed = false;
            break;
        }
    }
    std::printf("  %s: %s\n", variant.name, passed ? "PASSED" : "FAILED");

    HIP_CHECK(hipFree(dev_O));
    HIP_CHECK(hipFree(dev_dO));
    HIP_CHECK(hipFree(dev_D));
    unloadKernel(lk);
    return passed;
}

// ---------------------------------------------------------------------------
// OGradDotO group-mode runner (using rocm_ck::Args)
// ---------------------------------------------------------------------------

static bool runOGradDotOGroup(const rocm_ck::FmhaBwdOGradDotOVariant& variant,
                              kpack_archive_t archive,
                              const char* gpu_arch,
                              const std::vector<float>& host_O,
                              const std::vector<float>& host_dO,
                              const std::vector<float>& ref_D,
                              int batch,
                              int nhead,
                              int seqlen_q,
                              int hdim_v,
                              float p_undrop)
{
    const auto dtype      = variant.kernel.dtype;
    const int dtype_bytes = rocm_ck::data_type_bits(dtype) / 8;
    const size_t total_el = size_t(batch) * nhead * seqlen_q * hdim_v;
    const size_t total_d  = size_t(batch) * nhead * seqlen_q;
    const size_t buf_size = total_el * dtype_bytes;
    const size_t d_size   = total_d * sizeof(float);

    LoadedKernel lk;
    if(!loadKernel(archive, variant.name, gpu_arch, lk))
        return false;

    void* dev_O  = nullptr;
    void* dev_dO = nullptr;
    void* dev_D  = nullptr;
    HIP_CHECK(hipMalloc(&dev_O, buf_size));
    HIP_CHECK(hipMalloc(&dev_dO, buf_size));
    HIP_CHECK(hipMalloc(&dev_D, d_size));

    // Re-layout from [batch, nhead, seqlen_q, hdim_v] to
    // [total_seq, nhead, hdim_v] for group mode.
    const int total_seq = batch * seqlen_q;
    std::vector<float> group_O(total_el), group_dO(total_el);
    for(int b = 0; b < batch; ++b)
        for(int h = 0; h < nhead; ++h)
            for(int s = 0; s < seqlen_q; ++s)
                for(int v = 0; v < hdim_v; ++v)
                {
                    int src       = ((b * nhead + h) * seqlen_q + s) * hdim_v + v;
                    int dst       = ((b * seqlen_q + s) * nhead + h) * hdim_v + v;
                    group_O[dst]  = host_O[src];
                    group_dO[dst] = host_dO[src];
                }

    std::vector<char> typed_O(buf_size), typed_dO(buf_size);
    for(size_t i = 0; i < total_el; ++i)
    {
        rocm_ck::float_to_typed(dtype, group_O[i], typed_O.data() + i * dtype_bytes);
        rocm_ck::float_to_typed(dtype, group_dO[i], typed_dO.data() + i * dtype_bytes);
    }
    HIP_CHECK(hipMemcpy(dev_O, typed_O.data(), buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_dO, typed_dO.data(), buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dev_D, 0, d_size));

    // seqstart_q
    std::vector<int32_t> seqstart_q(batch + 1);
    for(int b = 0; b <= batch; ++b)
        seqstart_q[b] = b * seqlen_q;
    int32_t* dev_seqstart_q = nullptr;
    HIP_CHECK(hipMalloc(&dev_seqstart_q, seqstart_q.size() * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(dev_seqstart_q,
                        seqstart_q.data(),
                        seqstart_q.size() * sizeof(int32_t),
                        hipMemcpyHostToDevice));

    // Group layout strides
    const int64_t g_stride_o       = int64_t(nhead) * hdim_v;
    const int64_t g_nhead_stride_o = hdim_v;
    const int64_t g_nhead_stride_d = total_seq;

    // Re-layout reference D: [batch, nhead, seqlen_q] -> [nhead, total_seq]
    std::vector<float> group_ref_D(total_d);
    for(int b = 0; b < batch; ++b)
        for(int h = 0; h < nhead; ++h)
            for(int s = 0; s < seqlen_q; ++s)
            {
                int src          = (b * nhead + h) * seqlen_q + s;
                int dst          = h * total_seq + b * seqlen_q + s;
                group_ref_D[dst] = ref_D[src];
            }

    // Populate Args
    rocm_ck::Args args{};
    setTensor(args.tensors[ODO::O],
              dev_O,
              {static_cast<rocm_ck::index_t>(seqlen_q), static_cast<rocm_ck::index_t>(hdim_v)},
              {g_stride_o, g_nhead_stride_o, 0});
    setTensor(args.tensors[ODO::DO],
              dev_dO,
              {static_cast<rocm_ck::index_t>(seqlen_q), static_cast<rocm_ck::index_t>(hdim_v)},
              {g_stride_o, g_nhead_stride_o, 0});
    setTensor(args.tensors[ODO::D],
              dev_D,
              {static_cast<rocm_ck::index_t>(seqlen_q)},
              {g_nhead_stride_d, 0});
    setTensor(args.tensors[ODO::SEQSTART_Q],
              dev_seqstart_q,
              {static_cast<rocm_ck::index_t>(batch + 1)},
              {1});
    // SEQLEN_Q ptr = nullptr (use seqstart_q for physical length)
    setTensor(args.tensors[ODO::SEQLEN_Q], nullptr, {}, {});

    args.scalars[ODO::P_UNDROP].f32 = p_undrop;

    dim3 grid = rocm_ck::ograd_dot_o_grid_size(batch, nhead, seqlen_q, variant.kernel.block_size);
    std::printf("  %s: grid=(%u,%u,%u), block=%d (group)\n",
                variant.name,
                grid.x,
                grid.y,
                grid.z,
                variant.kernel.block_size);

    launchArgs(lk.function, grid, variant.kernel.block_size, args);

    // Verify
    std::vector<float> got_D(total_d);
    HIP_CHECK(hipMemcpy(got_D.data(), dev_D, d_size, hipMemcpyDeviceToHost));

    const float tol = rocm_ck::tolerance_for(dtype);
    bool passed     = true;
    for(size_t i = 0; i < total_d; ++i)
    {
        if(std::fabs(got_D[i] - group_ref_D[i]) > tol)
        {
            std::fprintf(stderr,
                         "  %s: MISMATCH at D[%zu]: got %f, expected %f "
                         "(diff=%e)\n",
                         variant.name,
                         i,
                         got_D[i],
                         group_ref_D[i],
                         got_D[i] - group_ref_D[i]);
            passed = false;
            break;
        }
    }
    std::printf("  %s: %s\n", variant.name, passed ? "PASSED" : "FAILED");

    HIP_CHECK(hipFree(dev_O));
    HIP_CHECK(hipFree(dev_dO));
    HIP_CHECK(hipFree(dev_D));
    HIP_CHECK(hipFree(dev_seqstart_q));
    unloadKernel(lk);
    return passed;
}

// ---------------------------------------------------------------------------
// DqDkDv batch-mode runner (with numerical verification)
// ---------------------------------------------------------------------------

static bool runDqDkDvBatchVariant(const rocm_ck::FmhaBwdDQDKDVVariant& variant,
                                  kpack_archive_t archive,
                                  const char* gpu_arch,
                                  const std::vector<float>& host_Q,
                                  const std::vector<float>& host_K,
                                  const std::vector<float>& host_V,
                                  const std::vector<float>& host_dO,
                                  const std::vector<float>& host_LSE,
                                  const std::vector<float>& host_D,
                                  const std::vector<float>& ref_dQ,
                                  const std::vector<float>& ref_dK,
                                  const std::vector<float>& ref_dV,
                                  int batch,
                                  int nhead,
                                  int seqlen_q,
                                  int seqlen_k,
                                  int hdim_q,
                                  int hdim_v,
                                  float scale,
                                  bool verify)
{
    const auto dtype      = variant.kernel.dtype;
    const int dtype_bytes = rocm_ck::data_type_bits(dtype) / 8;

    const size_t q_elems   = size_t(batch) * nhead * seqlen_q * hdim_q;
    const size_t k_elems   = size_t(batch) * nhead * seqlen_k * hdim_q;
    const size_t v_elems   = size_t(batch) * nhead * seqlen_k * hdim_v;
    const size_t do_elems  = size_t(batch) * nhead * seqlen_q * hdim_v;
    const size_t lse_elems = size_t(batch) * nhead * seqlen_q;

    LoadedKernel lk;
    if(!loadKernel(archive, variant.name, gpu_arch, lk))
        return false;

    // Allocate device buffers
    void* dev_Q      = nullptr;
    void* dev_K      = nullptr;
    void* dev_V      = nullptr;
    void* dev_LSE    = nullptr;
    void* dev_dO     = nullptr;
    void* dev_D      = nullptr;
    void* dev_dQ_acc = nullptr; // always float
    void* dev_dK     = nullptr;
    void* dev_dV     = nullptr;

    HIP_CHECK(hipMalloc(&dev_Q, q_elems * dtype_bytes));
    HIP_CHECK(hipMalloc(&dev_K, k_elems * dtype_bytes));
    HIP_CHECK(hipMalloc(&dev_V, v_elems * dtype_bytes));
    HIP_CHECK(hipMalloc(&dev_LSE, lse_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dev_dO, do_elems * dtype_bytes));
    HIP_CHECK(hipMalloc(&dev_D, lse_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dev_dQ_acc, q_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dev_dK, k_elems * dtype_bytes));
    HIP_CHECK(hipMalloc(&dev_dV, v_elems * dtype_bytes));

    // Convert and upload typed tensors
    auto upload_typed = [&](const std::vector<float>& src, void* dev, size_t count) {
        std::vector<char> buf(count * dtype_bytes);
        for(size_t i = 0; i < count; ++i)
            rocm_ck::float_to_typed(dtype, src[i], buf.data() + i * dtype_bytes);
        HIP_CHECK(hipMemcpy(dev, buf.data(), count * dtype_bytes, hipMemcpyHostToDevice));
    };

    upload_typed(host_Q, dev_Q, q_elems);
    upload_typed(host_K, dev_K, k_elems);
    upload_typed(host_V, dev_V, v_elems);
    upload_typed(host_dO, dev_dO, do_elems);

    // LSE and D are always float
    HIP_CHECK(
        hipMemcpy(dev_LSE, host_LSE.data(), lse_elems * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_D, host_D.data(), lse_elems * sizeof(float), hipMemcpyHostToDevice));

    // Zero dQ_acc, dK, dV
    HIP_CHECK(hipMemset(dev_dQ_acc, 0, q_elems * sizeof(float)));
    HIP_CHECK(hipMemset(dev_dK, 0, k_elems * dtype_bytes));
    HIP_CHECK(hipMemset(dev_dV, 0, v_elems * dtype_bytes));

    // Strides: [batch, nhead, seq, dim] row-major
    const int64_t stride_q       = hdim_q;
    const int64_t nhead_stride_q = int64_t(seqlen_q) * hdim_q;
    const int64_t batch_stride_q = int64_t(nhead) * seqlen_q * hdim_q;

    const int64_t stride_k       = hdim_q;
    const int64_t nhead_stride_k = int64_t(seqlen_k) * hdim_q;
    const int64_t batch_stride_k = int64_t(nhead) * seqlen_k * hdim_q;

    const int64_t stride_v       = hdim_v;
    const int64_t nhead_stride_v = int64_t(seqlen_k) * hdim_v;
    const int64_t batch_stride_v = int64_t(nhead) * seqlen_k * hdim_v;

    const int64_t stride_do       = hdim_v;
    const int64_t nhead_stride_do = int64_t(seqlen_q) * hdim_v;
    const int64_t batch_stride_do = int64_t(nhead) * seqlen_q * hdim_v;

    const int64_t nhead_stride_lse = seqlen_q;
    const int64_t batch_stride_lse = int64_t(nhead) * seqlen_q;

    // dQ_acc: same layout as Q but float
    const int64_t stride_dq_acc       = hdim_q;
    const int64_t nhead_stride_dq_acc = int64_t(seqlen_q) * hdim_q;
    const int64_t batch_stride_dq_acc = int64_t(nhead) * seqlen_q * hdim_q;

    // dK: same layout as K
    // dV: same layout as V

    // Populate Args
    rocm_ck::Args args{};

    // Each tensor carries its own natural dimensions.
    setTensor(args.tensors[DKV::Q],
              dev_Q,
              {static_cast<rocm_ck::index_t>(seqlen_q), static_cast<rocm_ck::index_t>(hdim_q)},
              {stride_q, nhead_stride_q, batch_stride_q});

    setTensor(args.tensors[DKV::K],
              dev_K,
              {static_cast<rocm_ck::index_t>(seqlen_k), static_cast<rocm_ck::index_t>(hdim_q)},
              {stride_k, nhead_stride_k, batch_stride_k});

    setTensor(args.tensors[DKV::V],
              dev_V,
              {static_cast<rocm_ck::index_t>(seqlen_k), static_cast<rocm_ck::index_t>(hdim_v)},
              {stride_v, nhead_stride_v, batch_stride_v});

    // LSE is 1D per head — no row stride.
    // strides[0]=nhead_stride, strides[1]=batch_stride
    setTensor(args.tensors[DKV::LSE],
              dev_LSE,
              {static_cast<rocm_ck::index_t>(seqlen_q)},
              {nhead_stride_lse, batch_stride_lse});

    setTensor(args.tensors[DKV::DO],
              dev_dO,
              {static_cast<rocm_ck::index_t>(seqlen_q), static_cast<rocm_ck::index_t>(hdim_v)},
              {stride_do, nhead_stride_do, batch_stride_do});

    // D is 1D per head — same stride convention as LSE.
    setTensor(args.tensors[DKV::D],
              dev_D,
              {static_cast<rocm_ck::index_t>(seqlen_q)},
              {nhead_stride_lse, batch_stride_lse});

    setTensor(args.tensors[DKV::DQ_ACC],
              dev_dQ_acc,
              {static_cast<rocm_ck::index_t>(seqlen_q), static_cast<rocm_ck::index_t>(hdim_q)},
              {stride_dq_acc, nhead_stride_dq_acc, batch_stride_dq_acc});

    setTensor(args.tensors[DKV::DK],
              dev_dK,
              {static_cast<rocm_ck::index_t>(seqlen_k), static_cast<rocm_ck::index_t>(hdim_q)},
              {stride_k, nhead_stride_k, batch_stride_k});

    setTensor(args.tensors[DKV::DV],
              dev_dV,
              {static_cast<rocm_ck::index_t>(seqlen_k), static_cast<rocm_ck::index_t>(hdim_v)},
              {stride_v, nhead_stride_v, batch_stride_v});

    // Scalars
    const float raw_scale                 = scale;
    const float log2e                     = 1.4426950408889634f;
    args.scalars[DKV::RAW_SCALE].f32      = raw_scale;
    args.scalars[DKV::SCALE].f32          = raw_scale * log2e;
    args.scalars[DKV::NUM_HEAD_Q].i32     = nhead;
    args.scalars[DKV::NHEAD_RATIO_QK].i32 = 1; // MHA (no GQA)

    // Launch
    dim3 grid = rocm_ck::dqdkdv_grid_size(batch, nhead, seqlen_k, variant.kernel.block_n0);
    std::printf("  %s: grid=(%u,%u,%u), block=%d\n",
                variant.name,
                grid.x,
                grid.y,
                grid.z,
                variant.kernel.block_size);

    launchArgs(lk.function, grid, variant.kernel.block_size, args);

    // Download and verify
    bool passed = true;

    if(verify)
    {
        // dQ_acc (float)
        std::vector<float> got_dQ_acc(q_elems);
        HIP_CHECK(hipMemcpy(
            got_dQ_acc.data(), dev_dQ_acc, q_elems * sizeof(float), hipMemcpyDeviceToHost));

        // dK (typed -> float)
        std::vector<char> typed_dK(k_elems * dtype_bytes);
        HIP_CHECK(hipMemcpy(typed_dK.data(), dev_dK, k_elems * dtype_bytes, hipMemcpyDeviceToHost));
        std::vector<float> got_dK(k_elems);
        for(size_t i = 0; i < k_elems; ++i)
            got_dK[i] = rocm_ck::typed_to_float(dtype, typed_dK.data() + i * dtype_bytes);

        // dV (typed -> float)
        std::vector<char> typed_dV(v_elems * dtype_bytes);
        HIP_CHECK(hipMemcpy(typed_dV.data(), dev_dV, v_elems * dtype_bytes, hipMemcpyDeviceToHost));
        std::vector<float> got_dV(v_elems);
        for(size_t i = 0; i < v_elems; ++i)
            got_dV[i] = rocm_ck::typed_to_float(dtype, typed_dV.data() + i * dtype_bytes);

        // Tolerance: FMHA BWD involves multiple GEMMs, so use a
        // more generous tolerance than the simple OGradDotO check.
        const float tol_dq = (dtype == rocm_ck::DataType::FP16) ? 0.05f : 0.2f;
        const float tol_dk = tol_dq;
        const float tol_dv = tol_dq;

        // Check dQ_acc
        for(size_t i = 0; i < q_elems && passed; ++i)
        {
            float diff    = std::fabs(got_dQ_acc[i] - ref_dQ[i]);
            float ref_mag = std::fabs(ref_dQ[i]) + 1e-6f;
            if(diff > tol_dq && diff / ref_mag > tol_dq)
            {
                std::fprintf(stderr,
                             "  %s: dQ MISMATCH at [%zu]: got %f, "
                             "expected %f (diff=%e)\n",
                             variant.name,
                             i,
                             got_dQ_acc[i],
                             ref_dQ[i],
                             got_dQ_acc[i] - ref_dQ[i]);
                passed = false;
            }
        }

        // Check dK
        for(size_t i = 0; i < k_elems && passed; ++i)
        {
            float diff    = std::fabs(got_dK[i] - ref_dK[i]);
            float ref_mag = std::fabs(ref_dK[i]) + 1e-6f;
            if(diff > tol_dk && diff / ref_mag > tol_dk)
            {
                std::fprintf(stderr,
                             "  %s: dK MISMATCH at [%zu]: got %f, "
                             "expected %f (diff=%e)\n",
                             variant.name,
                             i,
                             got_dK[i],
                             ref_dK[i],
                             got_dK[i] - ref_dK[i]);
                passed = false;
            }
        }

        // Check dV
        for(size_t i = 0; i < v_elems && passed; ++i)
        {
            float diff    = std::fabs(got_dV[i] - ref_dV[i]);
            float ref_mag = std::fabs(ref_dV[i]) + 1e-6f;
            if(diff > tol_dv && diff / ref_mag > tol_dv)
            {
                std::fprintf(stderr,
                             "  %s: dV MISMATCH at [%zu]: got %f, "
                             "expected %f (diff=%e)\n",
                             variant.name,
                             i,
                             got_dV[i],
                             ref_dV[i],
                             got_dV[i] - ref_dV[i]);
                passed = false;
            }
        }

        std::printf("  %s: %s\n", variant.name, passed ? "PASSED" : "FAILED");
    }
    else
    {
        // Compilation proof only -- kernel launched without crash.
        std::printf("  %s: LAUNCHED OK (compilation proof only)\n", variant.name);
    }

    HIP_CHECK(hipFree(dev_Q));
    HIP_CHECK(hipFree(dev_K));
    HIP_CHECK(hipFree(dev_V));
    HIP_CHECK(hipFree(dev_LSE));
    HIP_CHECK(hipFree(dev_dO));
    HIP_CHECK(hipFree(dev_D));
    HIP_CHECK(hipFree(dev_dQ_acc));
    HIP_CHECK(hipFree(dev_dK));
    HIP_CHECK(hipFree(dev_dV));
    unloadKernel(lk);
    return passed;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    // Disable stdout buffering so prints appear before a crash.
    std::setbuf(stdout, nullptr);

    std::printf("FMHA BWD kpack example starting...\n");

    if(argc != 2)
    {
        std::fprintf(stderr, "Usage: %s <path-to-kernels.kpack>\n", argv[0]);
        return 1;
    }

    // --- Open the kpack archive ---
    kpack_archive_t archive = nullptr;
    kpack_error_t kerr      = kpack_open(argv[1], &archive);
    if(kerr != KPACK_SUCCESS)
    {
        std::fprintf(stderr, "Failed to open archive '%s' (error %d)\n", argv[1], kerr);
        return 1;
    }

    size_t arch_count = 0;
    kpack_get_architecture_count(archive, &arch_count);
    std::printf("Opened %s -- architectures:", argv[1]);
    for(size_t i = 0; i < arch_count; ++i)
    {
        const char* arch_name = nullptr;
        kpack_get_architecture(archive, i, &arch_name);
        std::printf("%s %s", (i > 0 ? "," : ""), arch_name);
    }
    std::printf("\n");

    // --- Detect current GPU architecture ---
    std::string gpu_arch = rocm_ck::get_gpu_arch();
    if(gpu_arch.empty())
    {
        std::fprintf(stderr, "Failed to detect GPU architecture\n");
        kpack_close(archive);
        return 1;
    }
    std::printf("Detected GPU: %s\n", gpu_arch.c_str());

    bool all_passed = true;

    // =================================================================
    // Part 1: OGradDotO variants
    // =================================================================

    std::printf("\n=== OGradDotO variants (%d) ===\n", rocm_ck::ALL_OGRAD_DOT_O_VARIANTS_COUNT);

    // Variant selection demo
    std::printf("\nOGradDotO variant selection:\n");
    for(auto dt : {rocm_ck::DataType::FP16, rocm_ck::DataType::BF16})
    {
        const auto* best = rocm_ck::findVariant(rocm_ck::FmhaBwdOGradDotOConfig{
            .signature = {.dtype = dt, .hdim_v = HDIM_V, .mode = rocm_ck::FmhaMode::BATCH},
            .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
        if(best)
            std::printf("  %s d%d batch -> %s\n", rocm_ck::data_type_name(dt), HDIM_V, best->name);
    }

    std::printf("\nRunning OGradDotO variants:\n");
    for(const auto& v : rocm_ck::ALL_OGRAD_DOT_O_VARIANTS)
    {
        const int cur_hdim = v.kernel.hdim_v;

        std::vector<float> host_O, host_dO;
        makeOGradDotOTestData(BATCH, NHEAD, SEQLEN_Q, cur_hdim, host_O, host_dO);

        const int total_d = BATCH * NHEAD * SEQLEN_Q;
        std::vector<float> ref_D(total_d, 0.0f);
        cpuOGradDotO(host_O, host_dO, ref_D, BATCH, NHEAD, SEQLEN_Q, cur_hdim, P_UNDROP);

        bool passed;
        if(v.kernel.mode == rocm_ck::FmhaMode::GROUP)
        {
            passed = runOGradDotOGroup(v,
                                       archive,
                                       gpu_arch.c_str(),
                                       host_O,
                                       host_dO,
                                       ref_D,
                                       BATCH,
                                       NHEAD,
                                       SEQLEN_Q,
                                       cur_hdim,
                                       P_UNDROP);
        }
        else
        {
            passed = runOGradDotOBatch(v,
                                       archive,
                                       gpu_arch.c_str(),
                                       host_O,
                                       host_dO,
                                       ref_D,
                                       BATCH,
                                       NHEAD,
                                       SEQLEN_Q,
                                       cur_hdim,
                                       P_UNDROP);
        }
        if(!passed)
            all_passed = false;
    }

    // =================================================================
    // Part 2: DqDkDv variants
    // =================================================================

    std::printf("\n=== DqDkDv variants (%d) ===\n", rocm_ck::ALL_DQDKDV_VARIANTS_COUNT);

    // Prepare shared test data for DqDkDv (all variants use d128).
    const float scale = 1.0f / std::sqrt(static_cast<float>(HDIM_Q));

    std::vector<float> host_Q, host_K, host_V, host_dO_full;
    makeDqDkDvTestData(
        BATCH, NHEAD, SEQLEN_Q, SEQLEN_K, HDIM_Q, HDIM_V, host_Q, host_K, host_V, host_dO_full);

    // Forward pass reference: compute O, LSE, and D on the CPU.
    // O is needed for D = rowSum(dO * O), and LSE for softmax
    // reconstruction in the backward kernel.
    const size_t o_elems   = size_t(BATCH) * NHEAD * SEQLEN_Q * HDIM_V;
    const size_t lse_elems = size_t(BATCH) * NHEAD * SEQLEN_Q;

    // We compute O as part of the forward reference: O = softmax(Q@K^T * scale) @ V.
    // But cpuFmhaBwd computes LSE internally. We run the full bwd reference to get
    // LSE, dQ, dK, dV -- and also get D from cpuOGradDotO.

    std::vector<float> ref_LSE(lse_elems, 0.0f);
    std::vector<float> ref_dQ(size_t(BATCH) * NHEAD * SEQLEN_Q * HDIM_Q, 0.0f);
    std::vector<float> ref_dK(size_t(BATCH) * NHEAD * SEQLEN_K * HDIM_Q, 0.0f);
    std::vector<float> ref_dV(size_t(BATCH) * NHEAD * SEQLEN_K * HDIM_V, 0.0f);

    // Compute forward O for D computation.
    // O[i][v] = sum_j(P[i][j] * V[j][v]) where P = softmax(Q@K^T * scale)
    std::vector<float> ref_O(o_elems, 0.0f);
    {
        auto idx4 = [](int b, int h, int s, int d, int nh, int seq, int dim) {
            return ((b * nh + h) * seq + s) * dim + d;
        };
        auto idx3 = [](int b, int h, int s, int nh, int sq) { return (b * nh + h) * sq + s; };

        std::vector<float> S(SEQLEN_Q * SEQLEN_K);
        std::vector<float> P(SEQLEN_Q * SEQLEN_K);

        for(int b = 0; b < BATCH; ++b)
        {
            for(int h = 0; h < NHEAD; ++h)
            {
                // S = scale * Q @ K^T
                for(int i = 0; i < SEQLEN_Q; ++i)
                    for(int j = 0; j < SEQLEN_K; ++j)
                    {
                        float acc = 0.0f;
                        for(int k = 0; k < HDIM_Q; ++k)
                            acc += host_Q[idx4(b, h, i, k, NHEAD, SEQLEN_Q, HDIM_Q)] *
                                   host_K[idx4(b, h, j, k, NHEAD, SEQLEN_K, HDIM_Q)];
                        S[i * SEQLEN_K + j] = scale * acc;
                    }

                // softmax -> P, LSE
                for(int i = 0; i < SEQLEN_Q; ++i)
                {
                    float mx = S[i * SEQLEN_K];
                    for(int j = 1; j < SEQLEN_K; ++j)
                        mx = std::max(mx, S[i * SEQLEN_K + j]);

                    float se = 0.0f;
                    for(int j = 0; j < SEQLEN_K; ++j)
                        se += std::exp(S[i * SEQLEN_K + j] - mx);

                    float lse                               = mx + std::log(se);
                    ref_LSE[idx3(b, h, i, NHEAD, SEQLEN_Q)] = lse;

                    for(int j = 0; j < SEQLEN_K; ++j)
                        P[i * SEQLEN_K + j] = std::exp(S[i * SEQLEN_K + j] - lse);
                }

                // O = P @ V
                for(int i = 0; i < SEQLEN_Q; ++i)
                    for(int v = 0; v < HDIM_V; ++v)
                    {
                        float acc = 0.0f;
                        for(int j = 0; j < SEQLEN_K; ++j)
                            acc += P[i * SEQLEN_K + j] *
                                   host_V[idx4(b, h, j, v, NHEAD, SEQLEN_K, HDIM_V)];
                        ref_O[idx4(b, h, i, v, NHEAD, SEQLEN_Q, HDIM_V)] = acc;
                    }
            }
        }
    }

    // Compute D = rowSum(dO * O) * p_undrop (using reference O).
    std::vector<float> ref_D_full(lse_elems, 0.0f);
    cpuOGradDotO(ref_O, host_dO_full, ref_D_full, BATCH, NHEAD, SEQLEN_Q, HDIM_V, P_UNDROP);

    // Full backward reference.
    cpuFmhaBwd(host_Q,
               host_K,
               host_V,
               ref_O,
               host_dO_full,
               ref_D_full,
               ref_LSE,
               ref_dQ,
               ref_dK,
               ref_dV,
               BATCH,
               NHEAD,
               SEQLEN_Q,
               SEQLEN_K,
               HDIM_Q,
               HDIM_V,
               scale);

    // DqDkDv variant selection demo
    std::printf("\nDqDkDv variant selection:\n");
    for(auto dt : {rocm_ck::DataType::FP16, rocm_ck::DataType::BF16})
    {
        const auto* best = rocm_ck::findVariant(rocm_ck::FmhaBwdDQDKDVConfig{
            .signature =
                {.dtype = dt, .hdim_q = HDIM_Q, .hdim_v = HDIM_V, .mode = rocm_ck::FmhaMode::BATCH},
            .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
        if(best)
            std::printf("  %s d%d batch -> %s\n", rocm_ck::data_type_name(dt), HDIM_Q, best->name);
    }

    std::printf("\nRunning DqDkDv variants:\n");
    for(const auto& v : rocm_ck::ALL_DQDKDV_VARIANTS)
    {
        // Group mode is not yet implemented in the device bridge.
        if(v.kernel.mode == rocm_ck::FmhaMode::GROUP)
        {
            std::printf("  %s: SKIPPED (group mode not yet "
                        "implemented)\n",
                        v.name);
            continue;
        }

        // Determine whether to verify numerically.
        // Plain batch fp16/bf16 variants with no mask, no dropout,
        // no deterministic flag get full verification.
        // Others are compilation proof only.
        bool is_plain_batch =
            (!v.kernel.has_mask && !v.kernel.has_dropout && !v.kernel.is_deterministic &&
             v.kernel.bias_type == rocm_ck::FmhaBiasType::NONE);

        bool passed = runDqDkDvBatchVariant(v,
                                            archive,
                                            gpu_arch.c_str(),
                                            host_Q,
                                            host_K,
                                            host_V,
                                            host_dO_full,
                                            ref_LSE,
                                            ref_D_full,
                                            ref_dQ,
                                            ref_dK,
                                            ref_dV,
                                            BATCH,
                                            NHEAD,
                                            SEQLEN_Q,
                                            SEQLEN_K,
                                            HDIM_Q,
                                            HDIM_V,
                                            scale,
                                            is_plain_batch);

        if(!passed)
            all_passed = false;
    }

    // =================================================================
    // Part 3: ConvertDQ (skipped for now)
    // =================================================================

    std::printf("\n=== ConvertDQ variants (%d) ===\n", rocm_ck::ALL_CONVERT_DQ_VARIANTS_COUNT);
    std::printf("  (skipped: requires deterministic DqDkDv "
                "split-K setup)\n");

    // --- Cleanup ---
    kpack_close(archive);

    std::printf("\n%s\n", all_passed ? "All variants PASSED" : "Some variants FAILED");
    return all_passed ? 0 : 1;
}
