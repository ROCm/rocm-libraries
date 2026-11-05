#include "explicit_gemm/hipblaslt_matmul.hpp"

#include "device_runtime.h"
#include "hip_util.h"
#include "unreachable.h"

#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

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
//
// A layer with G groups is a strided batch of G GEMMs, Cg = C/G and Kg = K/G. Group g's views are
// X col [Cg,M] ld=C, W col [Cg,Kg] ld=Cg and Y col [Kg,M] ld=K, at offsets g*Cg, g*Kg*Cg and
// g*Kg, so the activations' batch stride is below their leading dimension.
LayoutGuard make_col_layout(hipDataType type,
                            int64_t rows,
                            int64_t cols,
                            int64_t ld,
                            int32_t batch_count,
                            int64_t stride)
{
    hipblasLtMatrixLayout_t layout{};
    check_status(hipblasLtMatrixLayoutCreate(
                     &layout, type, static_cast<uint64_t>(rows), static_cast<uint64_t>(cols), ld),
                 "hipblasLtMatrixLayoutCreate");
    if(batch_count > 1)
    {
        const auto set = [layout](hipblasLtMatrixLayoutAttribute_t attr,
                                  const void* value,
                                  std::size_t size,
                                  const char* where) {
            const hipblasStatus_t status =
                hipblasLtMatrixLayoutSetAttribute(layout, attr, value, size);
            if(status != HIPBLAS_STATUS_SUCCESS)
                hipblasLtMatrixLayoutDestroy(layout);
            check_status(status, where);
        };
        set(HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch_count,
            sizeof(batch_count),
            "HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT");
        set(HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride,
            sizeof(stride),
            "HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET");
    }
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
    int32_t batch_count;
    int64_t stride_a;
    int64_t stride_b;
    int64_t stride_d;
    uint64_t max_workspace = 0;
};

GemmProblem gemm_shape(const ConvParams& par)
{
    const int64_t m_spatial = static_cast<int64_t>(par.n) * par.h * par.w;
    const int64_t c         = par.c;
    const int64_t k         = par.k;
    const int32_t g         = par.groups;
    const int64_t cg        = par.channels_per_group();
    const int64_t kg        = par.filters_per_group();
    const auto abc_type     = to_hip_data_type(par.input_type);
    const auto compute_type = to_compute_type(par.input_type);

    switch(par.direction)
    {
    case Direction::Fprop:
        // Y[M,K] = X[M,C] * W^T[C,K]  (W KRSC row [K,C])
        // Col: Y^T[K,M] = W^T[K,C] * X^T[C,M] = op(W)[K,C] * op(X)[C,M]
        return {HIPBLAS_OP_T,
                HIPBLAS_OP_N,
                kg,
                m_spatial,
                cg,
                abc_type,
                abc_type,
                compute_type,
                cg,
                c,
                k,
                true,
                g,
                kg * cg,
                cg,
                kg};
    case Direction::Dgrad:
        // dX[M,C] = dY[M,K] * W[K,C]
        // Col: dX^T[C,M] = W^T[C,K] * dY^T[K,M]
        return {HIPBLAS_OP_N,
                HIPBLAS_OP_N,
                cg,
                m_spatial,
                kg,
                abc_type,
                abc_type,
                compute_type,
                cg,
                k,
                c,
                true,
                g,
                kg * cg,
                kg,
                cg};
    case Direction::Wgrad:
        // dW[K,C] = dY^T[K,M] * X[M,C]
        // Row dW aliases col [C,K] ld=C; dW^T[C,K] = X^T[C,M] * dY[M,K]
        return {HIPBLAS_OP_N,
                HIPBLAS_OP_T,
                cg,
                kg,
                m_spatial,
                abc_type,
                to_hip_data_type(par.weight_grad_type),
                compute_type,
                c,
                k,
                cg,
                false,
                g,
                cg,
                kg,
                kg * cg};
    }
    HIPCONV_UNREACHABLE();
}

GemmProblem gemm_problem(const ConvParams& par)
{
    GemmProblem g   = gemm_shape(par);
    g.max_workspace = gemm_workspace_bytes(par.direction);
    return g;
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
    return {make_col_layout(
                g.abc_type, a_n ? g.m : g.k, a_n ? g.k : g.m, g.lda, g.batch_count, g.stride_a),
            make_col_layout(
                g.abc_type, b_n ? g.k : g.n, b_n ? g.n : g.k, g.ldb, g.batch_count, g.stride_b),
            make_col_layout(g.d_type, g.m, g.n, g.ldd, g.batch_count, g.stride_d),
            make_col_layout(g.d_type, g.m, g.n, g.ldd, g.batch_count, g.stride_d),
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

// The heuristic's top algorithm within `max_workspace` bytes. `out` is valid only if the
// outcome is found().
HeuristicOutcome resolve_algo(hipblasLtHandle_t lt,
                              const GemmDescs& descs,
                              uint64_t max_workspace,
                              hipblasLtMatmulHeuristicResult_t& out)
{
    PreferenceGuard pref_guard = [max_workspace] {
        hipblasLtMatmulPreference_t pref{};
        check_status(hipblasLtMatmulPreferenceCreate(&pref), "hipblasLtMatmulPreferenceCreate");
        check_status(
            hipblasLtMatmulPreferenceSetAttribute(pref,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &max_workspace,
                                                  sizeof(max_workspace)),
            "HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");
        return PreferenceGuard{pref};
    }();

    HeuristicOutcome outcome{};
    outcome.status = hipblasLtMatmulAlgoGetHeuristic(lt,
                                                     descs.matmul.handle,
                                                     descs.a.handle,
                                                     descs.b.handle,
                                                     descs.c.handle,
                                                     descs.d.handle,
                                                     pref_guard.handle,
                                                     1,
                                                     &out,
                                                     &outcome.returned);
    return outcome;
}

// The algorithm hipBLASLt's heuristic chose for a GEMM on one device.
struct DeviceAlgo
{
    DeviceAlgo(const GemmProblem& problem, hipblasLtHandle_t lt)
        : descs(make_descs(problem))
        , outcome(resolve_algo(lt, descs, problem.max_workspace, heuristic))
    {
    }

    GemmDescs descs;
    hipblasLtMatmulHeuristicResult_t heuristic{};
    HeuristicOutcome outcome;
};

} // namespace

class GemmPlan
{
public:
    explicit GemmPlan(const ConvParams& par) : problem_(gemm_problem(par)) {}

    const GemmProblem& problem() const { return problem_; }

    // The algorithm for `device`, resolved by its first launch there. A failed query is not
    // kept, so the next launch asks again.
    std::shared_ptr<const DeviceAlgo> algo(int device, hipblasLtHandle_t lt)
    {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if(const auto found = algos_.find(device); found != algos_.end())
                return found->second;
        }
        auto resolved = std::make_shared<const DeviceAlgo>(problem_, lt);
        if(resolved->outcome.status != HIPBLAS_STATUS_SUCCESS)
            return resolved;
        const std::lock_guard<std::mutex> lock(mutex_);
        return algos_.try_emplace(device, std::move(resolved)).first->second;
    }

private:
    GemmProblem problem_;
    std::mutex mutex_;
    std::unordered_map<int, std::shared_ptr<const DeviceAlgo>> algos_;
};

std::shared_ptr<GemmPlan> make_gemm_plan(const ConvParams& par)
{
    return std::make_shared<GemmPlan>(par);
}

void launch_gemm(GemmPlan& plan,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* workspace,
                 hipStream_t stream)
{
    if(workspace == nullptr && plan.problem().max_workspace > 0)
        throw HipError(hipErrorInvalidValue, "an explicit-GEMM launch needs its workspace");
    const int device           = stream_device(stream);
    const hipblasLtHandle_t lt = hipblaslt_handle(device);
    const auto algo            = plan.algo(device, lt);
    // launch() reports this as hipErrorNotSupported; a null algo would only re-ask the same
    // heuristic and fail with HIPBLAS_STATUS_INTERNAL_ERROR.
    if(algo->outcome.none())
        throw HipError(hipErrorNotSupported, "hipBLASLt has no algorithm for this GEMM");
    check_status(algo->outcome.status, "hipblasLtMatmulAlgoGetHeuristic");
    // Pass the algorithm its own workspace size: given one it does not use, hipBLASLt fails
    // some algorithms with HIPBLAS_STATUS_INTERNAL_ERROR (wgrad at n*h*w around 2.8M and up).
    const std::size_t ws_bytes = algo->heuristic.workspaceSize;

    const GemmProblem& g   = plan.problem();
    const GemmDescs& descs = algo->descs;
    const float alpha      = 1.0f;
    const float beta       = 0.0f;
    check_status(hipblasLtMatmul(lt,
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
                                 &algo->heuristic.algo,
                                 ws_bytes > 0 ? workspace : nullptr,
                                 ws_bytes,
                                 stream),
                 "hipblasLtMatmul");
}

} // namespace hipconv::explicit_gemm
