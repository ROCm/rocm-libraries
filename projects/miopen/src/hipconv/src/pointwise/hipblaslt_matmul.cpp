#include "pointwise/hipblaslt_matmul.hpp"

#include <hipblaslt/hipblaslt.h>

#include <stdexcept>
#include <string>

namespace hipconv::pointwise
{
namespace
{

void check_status(hipblasStatus_t status, const char* where)
{
    if(status != HIPBLAS_STATUS_SUCCESS)
        throw HipblasltError(status, where);
}

hipDataType to_hip_data_type(DataType dtype)
{
    switch(dtype)
    {
    case DataType::fp16:
        return HIP_R_16F;
    case DataType::bf16:
        return HIP_R_16BF;
    case DataType::fp32:
        return HIP_R_32F;
    default:
        throw std::invalid_argument("unsupported pointwise dtype");
    }
}

// Process-wide lazily-created hipBLASLt handle. Held as a raw handle (not a
// Guard), so it is intentionally never destroyed: this avoids the static
// destruction-order fiasco and the driver reclaims it at process exit.
// TODO: revisit once hipconv gains an explicit lifecycle (e.g. hipconvCreate /
// hipconvDestroy). At that point the handle should live in the hipconv context
// object, be owned by a Guard, and be passed into launch_pointwise_gemm rather
// than pulled from this global accessor.
hipblasLtHandle_t& handle()
{
    static hipblasLtHandle_t h = [] {
        hipblasLtHandle_t created{};
        check_status(hipblasLtCreate(&created), "hipblasLtCreate");
        return created;
    }();
    return h;
}

// RAII wrapper for a hipBLASLt opaque handle. Non-copyable and non-movable: the
// guard uniquely owns the handle and destroys it exactly once. Instances are only
// ever produced as prvalues (guaranteed copy elision), so no move is needed.
template <typename HandleT, hipblasStatus_t (*Destroy)(HandleT)>
struct Guard
{
    HandleT handle{};
    explicit Guard(HandleT h) : handle{h} {}
    Guard(Guard const&)            = delete;
    Guard(Guard&&)                 = delete;
    Guard& operator=(Guard const&) = delete;
    Guard& operator=(Guard&&)      = delete;
    ~Guard()
    {
        if(handle)
            Destroy(handle);
    }
};

using LayoutGuard     = Guard<hipblasLtMatrixLayout_t, &hipblasLtMatrixLayoutDestroy>;
using MatmulDescGuard = Guard<hipblasLtMatmulDesc_t, &hipblasLtMatmulDescDestroy>;

// hipBLASLt default layout is column-major. NHWC/KRSC row-major tensors alias
// column-major views without copying:
//   NHWC X[M,C] row  <=> col [C,M] ld=C
//   KRSC W[K,C] row  <=> col [C,K] ld=C
//   NPQK Y[M,K] row  <=> col [K,M] ld=K
LayoutGuard make_col_layout(hipDataType type, int64_t rows, int64_t cols, int64_t ld)
{
    hipblasLtMatrixLayout_t layout{};
    check_status(hipblasLtMatrixLayoutCreate(
                     &layout, type, static_cast<uint64_t>(rows), static_cast<uint64_t>(cols), ld),
                 "hipblasLtMatrixLayoutCreate");
    return LayoutGuard{layout};
}

MatmulDescGuard make_matmul_desc(hipblasOperation_t trans_a, hipblasOperation_t trans_b)
{
    hipblasLtMatmulDesc_t desc{};
    check_status(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                 "hipblasLtMatmulDescCreate");
    check_status(hipblasLtMatmulDescSetAttribute(
                     desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
                 "HIPBLASLT_MATMUL_DESC_TRANSA");
    check_status(hipblasLtMatmulDescSetAttribute(
                     desc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
                 "HIPBLASLT_MATMUL_DESC_TRANSB");
    return MatmulDescGuard{desc};
}

using PreferenceGuard = Guard<hipblasLtMatmulPreference_t, &hipblasLtMatmulPreferenceDestroy>;

// Resolve the GEMM algorithm via hipBLASLt's heuristic. This is the recommended
// usage (vs. passing a null algo): kernel selection is explicit and robust
// across hipBLASLt versions. We restrict to workspace-free algos because the
// GEMM is launched with a null workspace. `out_algo` is valid only if true is
// returned; otherwise the caller falls back to the null-algo default path.
bool resolve_algo(hipblasLtMatmulDesc_t desc,
                  hipblasLtMatrixLayout_t a,
                  hipblasLtMatrixLayout_t b,
                  hipblasLtMatrixLayout_t c,
                  hipblasLtMatrixLayout_t d,
                  hipblasLtMatmulAlgo_t& out_algo)
{
    PreferenceGuard pref_guard = [] {
        hipblasLtMatmulPreference_t pref{};
        check_status(hipblasLtMatmulPreferenceCreate(&pref), "hipblasLtMatmulPreferenceCreate");
        const uint64_t max_workspace = 0;
        check_status(
            hipblasLtMatmulPreferenceSetAttribute(pref,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &max_workspace,
                                                  sizeof(max_workspace)),
            "HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");
        return PreferenceGuard{pref};
    }();

    hipblasLtMatmulHeuristicResult_t result{};
    int returned      = 0;
    const auto status = hipblasLtMatmulAlgoGetHeuristic(
        handle(), desc, a, b, c, d, pref_guard.handle, 1, &result, &returned);
    if(status != HIPBLAS_STATUS_SUCCESS || returned <= 0)
        return false;

    out_algo = result.algo;
    return true;
}

void run_matmul(hipblasOperation_t trans_a,
                hipblasOperation_t trans_b,
                int64_t m,
                int64_t n,
                int64_t k,
                hipDataType abc_type,
                hipDataType d_type,
                const void* a,
                int64_t lda,
                const void* b,
                int64_t ldb,
                void* d,
                int64_t ldd,
                float alpha,
                float beta,
                hipStream_t stream)
{
    auto a_desc = make_col_layout(
        abc_type, trans_a == HIPBLAS_OP_N ? m : k, trans_a == HIPBLAS_OP_N ? k : m, lda);
    auto b_desc = make_col_layout(
        abc_type, trans_b == HIPBLAS_OP_N ? k : n, trans_b == HIPBLAS_OP_N ? n : k, ldb);
    auto c_desc  = make_col_layout(d_type, m, n, ldd);
    auto d_desc  = make_col_layout(d_type, m, n, ldd);
    auto mm_desc = make_matmul_desc(trans_a, trans_b);

    hipblasLtMatmulAlgo_t algo{};
    const bool have_algo = resolve_algo(
        mm_desc.handle, a_desc.handle, b_desc.handle, c_desc.handle, d_desc.handle, algo);

    check_status(hipblasLtMatmul(handle(),
                                 mm_desc.handle,
                                 &alpha,
                                 a,
                                 a_desc.handle,
                                 b,
                                 b_desc.handle,
                                 &beta,
                                 d,
                                 c_desc.handle,
                                 d,
                                 d_desc.handle,
                                 have_algo ? &algo : nullptr,
                                 nullptr,
                                 0,
                                 stream),
                 "hipblasLtMatmul");
}

} // namespace

void launch_pointwise_gemm(const Conv2dParams& par,
                           const void* in,
                           const void* wei,
                           void* out,
                           hipStream_t stream)
{
    const int64_t m_spatial = static_cast<int64_t>(par.n) * par.h * par.w;
    const int64_t c         = par.c;
    const int64_t k         = par.k;
    const auto abc_type     = to_hip_data_type(par.input_type);

    switch(par.direction)
    {
    case Direction::Fprop:
        // Y[M,K] = X[M,C] * W^T[C,K]  (W KRSC row [K,C])
        // Col: Y^T[K,M] = W^T[K,C] * X^T[C,M] = op(W)[K,C] * op(X)[C,M]
        run_matmul(HIPBLAS_OP_T,
                   HIPBLAS_OP_N,
                   k,
                   m_spatial,
                   c,
                   abc_type,
                   abc_type,
                   wei,
                   c,
                   in,
                   c,
                   out,
                   k,
                   1.0f,
                   0.0f,
                   stream);
        break;

    case Direction::Dgrad:
        // dX[M,C] = dY[M,K] * W[K,C]
        // Col: dX^T[C,M] = W^T[C,K] * dY^T[K,M]
        run_matmul(HIPBLAS_OP_N,
                   HIPBLAS_OP_N,
                   c,
                   m_spatial,
                   k,
                   abc_type,
                   abc_type,
                   wei,
                   c,
                   in,
                   k,
                   out,
                   c,
                   1.0f,
                   0.0f,
                   stream);
        break;

    case Direction::Wgrad:
        // dW[K,C] = dY^T[K,M] * X[M,C]
        // Row dW aliases col [C,K] ld=C; dW^T[C,K] = X^T[C,M] * dY[M,K]
        run_matmul(HIPBLAS_OP_N,
                   HIPBLAS_OP_T,
                   c,
                   k,
                   m_spatial,
                   abc_type,
                   HIP_R_32F,
                   in,
                   c,
                   wei,
                   k,
                   out,
                   c,
                   1.0f,
                   0.0f,
                   stream);
        break;
    }
}

} // namespace hipconv::pointwise
