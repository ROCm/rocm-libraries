#include "explicit_gemm/hipblaslt_matmul.hpp"

#include "hip_util.h"
#include "unreachable.h"

#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace hipconv::explicit_gemm
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
    case DataType::tf32:
        return HIP_R_32F;
    default:
        throw std::invalid_argument("unsupported GEMM dtype");
    }
}

hipblasComputeType_t to_compute_type(DataType dtype)
{
    return dtype == DataType::tf32 ? HIPBLAS_COMPUTE_32F_FAST_TF32 : HIPBLAS_COMPUTE_32F;
}

// Process-wide lazily-created hipBLASLt handle. Held as a raw handle (not a
// Guard), so it is intentionally never destroyed: this avoids the static
// destruction-order fiasco and the driver reclaims it at process exit.
// TODO: revisit once hipconv gains an explicit lifecycle (e.g. hipconvCreate /
// hipconvDestroy). At that point the handle should live in the hipconv context
// object, be owned by a Guard, and be passed into launch_gemm rather
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

MatmulDescGuard make_matmul_desc(hipblasComputeType_t compute_type,
                                 hipblasOperation_t trans_a,
                                 hipblasOperation_t trans_b)
{
    hipblasLtMatmulDesc_t desc{};
    check_status(hipblasLtMatmulDescCreate(&desc, compute_type, HIP_R_32F),
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

// The column-major GEMM a layer runs, in hipBLASLt's terms. A is the weights for fprop and
// dgrad and the input for wgrad; B is the other operand; D is the output.
struct GemmProblem
{
    hipblasOperation_t trans_a;
    hipblasOperation_t trans_b;
    int64_t m;
    int64_t n;
    int64_t k;
    hipDataType abc_type;
    hipDataType d_type;
    hipblasComputeType_t compute_type;
    int64_t lda;
    int64_t ldb;
    int64_t ldd;
    bool a_is_weights;
};

GemmProblem gemm_problem(const ConvParams& par)
{
    const int64_t m_spatial = static_cast<int64_t>(par.n) * par.h * par.w;
    const int64_t c         = par.c;
    const int64_t k         = par.k;
    const auto abc_type     = to_hip_data_type(par.input_type);
    const auto compute_type = to_compute_type(par.input_type);

    switch(par.direction)
    {
    case Direction::Fprop:
        // Y[M,K] = X[M,C] * W^T[C,K]  (W KRSC row [K,C])
        // Col: Y^T[K,M] = W^T[K,C] * X^T[C,M] = op(W)[K,C] * op(X)[C,M]
        return {HIPBLAS_OP_T,
                HIPBLAS_OP_N,
                k,
                m_spatial,
                c,
                abc_type,
                abc_type,
                compute_type,
                c,
                c,
                k,
                true};
    case Direction::Dgrad:
        // dX[M,C] = dY[M,K] * W[K,C]
        // Col: dX^T[C,M] = W^T[C,K] * dY^T[K,M]
        return {HIPBLAS_OP_N,
                HIPBLAS_OP_N,
                c,
                m_spatial,
                k,
                abc_type,
                abc_type,
                compute_type,
                c,
                k,
                c,
                true};
    case Direction::Wgrad:
        // dW[K,C] = dY^T[K,M] * X[M,C]
        // Row dW aliases col [C,K] ld=C; dW^T[C,K] = X^T[C,M] * dY[M,K]
        return {HIPBLAS_OP_N,
                HIPBLAS_OP_T,
                c,
                k,
                m_spatial,
                abc_type,
                HIP_R_32F,
                compute_type,
                c,
                k,
                c,
                false};
    }
    HIPCONV_UNREACHABLE();
}

struct GemmDescs
{
    LayoutGuard a;
    LayoutGuard b;
    LayoutGuard c;
    LayoutGuard d;
    MatmulDescGuard matmul;
};

GemmDescs make_descs(const GemmProblem& g)
{
    const bool a_n = g.trans_a == HIPBLAS_OP_N;
    const bool b_n = g.trans_b == HIPBLAS_OP_N;
    return {make_col_layout(g.abc_type, a_n ? g.m : g.k, a_n ? g.k : g.m, g.lda),
            make_col_layout(g.abc_type, b_n ? g.k : g.n, b_n ? g.n : g.k, g.ldb),
            make_col_layout(g.d_type, g.m, g.n, g.ldd),
            make_col_layout(g.d_type, g.m, g.n, g.ldd),
            make_matmul_desc(g.compute_type, g.trans_a, g.trans_b)};
}

// A failed query means hipBLASLt could not answer; a successful one with no results means
// the heuristic found no algorithm for the GEMM.
struct HeuristicOutcome
{
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
    int returned           = 0;

    bool found() const { return status == HIPBLAS_STATUS_SUCCESS && returned > 0; }
    bool none() const { return status == HIPBLAS_STATUS_SUCCESS && returned == 0; }
};

// The heuristic's top workspace-free algorithm, since the GEMM is launched without a
// workspace. `out_algo` is valid only if the outcome is found().
HeuristicOutcome resolve_algo(const GemmDescs& descs, hipblasLtMatmulAlgo_t& out_algo)
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
    HeuristicOutcome outcome{};
    outcome.status = hipblasLtMatmulAlgoGetHeuristic(handle(),
                                                     descs.matmul.handle,
                                                     descs.a.handle,
                                                     descs.b.handle,
                                                     descs.c.handle,
                                                     descs.d.handle,
                                                     pref_guard.handle,
                                                     1,
                                                     &result,
                                                     &outcome.returned);
    if(outcome.found())
        out_algo = result.algo;
    return outcome;
}

} // namespace

bool has_algorithm(const ConvParams& par)
{
    // Only an empty answer rules the layer out; a failed query is left for the launch to report.
    try
    {
        hipblasLtMatmulAlgo_t algo{};
        return !resolve_algo(make_descs(gemm_problem(par)), algo).none();
    }
    catch(const HipblasltError&)
    {
        return true;
    }
}

void launch_gemm(const ConvParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 hipStream_t stream)
{
    const GemmProblem g   = gemm_problem(par);
    const GemmDescs descs = make_descs(g);

    hipblasLtMatmulAlgo_t algo{};
    const HeuristicOutcome heuristic = resolve_algo(descs, algo);
    // launch() reports this as hipErrorNotSupported; a null algo would only re-ask the same
    // heuristic and fail with HIPBLAS_STATUS_INTERNAL_ERROR.
    if(heuristic.none())
        throw HipError(hipErrorNotSupported, "hipBLASLt has no algorithm for this GEMM");
    check_status(heuristic.status, "hipblasLtMatmulAlgoGetHeuristic");

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    check_status(hipblasLtMatmul(handle(),
                                 descs.matmul.handle,
                                 &alpha,
                                 g.a_is_weights ? wei : in,
                                 descs.a.handle,
                                 g.a_is_weights ? in : wei,
                                 descs.b.handle,
                                 &beta,
                                 out,
                                 descs.c.handle,
                                 out,
                                 descs.d.handle,
                                 &algo,
                                 nullptr,
                                 0,
                                 stream),
                 "hipblasLtMatmul");
}

} // namespace hipconv::explicit_gemm
