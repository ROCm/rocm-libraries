// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipblaslt_arguments.hpp"
#include <algorithm>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <numeric>
#include <string>
#include <vector>

// api_method value for grouped GEMM through the C API: grouped matrix layouts,
// hipblasLtMatmulAlgoGetHeuristic and hipblasLtMatmul.
constexpr int32_t kApiMethodGroupedC = 3;

// Why the grouped C API cannot run arg, or an empty string if it can.
inline std::string groupedCApiUnsupportedReason(const Arguments& arg)
{
    if(arg.grouped_gemm <= 0)
        return "needs --grouped_gemm";
    if(arg.batch_count > 1)
        return "supports one batch per group";
    if(arg.use_user_args)
        return "does not take user arguments";
    if(arg.algo_method != 0)
        return "supports only --algo_method heuristic";
    if(arg.bias_vector || arg.activation_type != hipblaslt_activation_type::none || arg.use_e
       || arg.gradient)
        return "supports no epilogue, bias or aux output";
    if(arg.scaleB != arg.scaleA
       || (arg.scaleA != hipblaslt_scaling_format::none
           && arg.scaleA != hipblaslt_scaling_format::Block_32_UE8M0
           && arg.scaleA != hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT))
        return "supports only the MX scale modes, the same on A and B";
    if(arg.scaleC || arg.scaleD || arg.scaleE || arg.scaleAlpha_vector || arg.amaxD)
        return "supports no scaling other than MX";
    if(arg.swizzle_a || arg.swizzle_b)
        return "supports no swizzled A or B";
    if(arg.scale_type != HIP_R_32F)
        return "needs scale type f32_r";
    return {};
}

// One grouped problem for the C API: grouped layouts shared by all rotating blocks, and per
// block a descriptor and device arrays of the per-group A, B, C and D pointers, and of the
// MX scale pointers. The descriptor holds the size hints and the launch setup, so each
// block keeps its own.
class GroupedCApiProblem
{
public:
    // a, b, c and d, and scaleA and scaleB with MX, are [block][group] device addresses.
    // Check status() before use.
    GroupedCApiProblem(hipblasLtHandle_t                      handle,
                       const Arguments&                       arg,
                       hipblasOperation_t                     transA,
                       hipblasOperation_t                     transB,
                       const std::vector<int64_t>&            M,
                       const std::vector<int64_t>&            N,
                       const std::vector<int64_t>&            K,
                       const std::vector<int64_t>&            lda,
                       const std::vector<int64_t>&            ldb,
                       const std::vector<int64_t>&            ldc,
                       const std::vector<int64_t>&            ldd,
                       const std::vector<std::vector<void*>>& a,
                       const std::vector<std::vector<void*>>& b,
                       const std::vector<std::vector<void*>>& c,
                       const std::vector<std::vector<void*>>& d,
                       const std::vector<std::vector<void*>>& scaleA,
                       const std::vector<std::vector<void*>>& scaleB,
                       size_t                                 maxWorkspaceBytes)
        : handle_(handle)
    {
        const bool nA = transA == HIPBLAS_OP_N, nB = transB == HIPBLAS_OP_N;
        if(!makeLayout(layA_, arg.a_type, nA ? M : K, nA ? K : M, lda)
           || !makeLayout(layB_, arg.b_type, nB ? K : N, nB ? N : K, ldb)
           || !makeLayout(layC_, arg.c_type, M, N, ldc)
           || !makeLayout(layD_, arg.d_type, M, N, ldd))
            return;

        for(size_t blk = 0; blk < a.size(); blk++)
        {
            descs_.push_back(nullptr);
            if(!ok(hipblasLtMatmulDescCreate(&descs_.back(), arg.compute_type, arg.scale_type))
               || !ok(hipblasLtMatmulDescSetAttribute(
                   descs_.back(), HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)))
               || !ok(hipblasLtMatmulDescSetAttribute(
                   descs_.back(), HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)))
               || !upload(ptrsA_, a[blk]) || !upload(ptrsB_, b[blk]) || !upload(ptrsC_, c[blk])
               || !upload(ptrsD_, d[blk]))
                return;
            if(arg.scaleA != hipblaslt_scaling_format::none
               && !setScales(descs_.back(), arg.scaleA, scaleA[blk], scaleB[blk]))
                return;
        }

        if(!ok(hipblasLtMatmulPreferenceCreate(&pref_)))
            return;
        uint64_t wsLimit = maxWorkspaceBytes;
        if(!ok(hipblasLtMatmulPreferenceSetAttribute(
               pref_, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsLimit, sizeof(wsLimit))))
            return;
        auto hint = [&](hipblasLtMatmulPreferenceAttributes_t attr, int64_t v) {
            return ok(hipblasLtMatmulPreferenceSetAttribute(pref_, attr, &v, sizeof(v)));
        };
        const int64_t groups = int64_t(M.size());
        if(!uniform(M)
           && !(hint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_ROWS_EXT,
                     *std::max_element(M.begin(), M.end()))
                && hint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS, sum(M) / groups)))
            return;
        if(!uniform(N)
           && !(hint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_COLS_EXT,
                     *std::max_element(N.begin(), N.end()))
                && hint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS, sum(N) / groups)))
            return;
        if(!uniform(K))
            hint(HIPBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, sum(K) / groups);
    }

    GroupedCApiProblem(const GroupedCApiProblem&)            = delete;
    GroupedCApiProblem& operator=(const GroupedCApiProblem&) = delete;

    ~GroupedCApiProblem()
    {
        for(auto desc : descs_)
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
        if(pref_)
            hipblasLtMatmulPreferenceDestroy(pref_);
        for(auto l : {layA_, layB_, layC_, layD_})
            if(l)
                hipblasLtMatrixLayoutDestroy(l);
        for(auto* ptrs : {&ptrsA_, &ptrsB_, &ptrsC_, &ptrsD_, &dims_, &scales_})
            for(void* p : *ptrs)
                static_cast<void>(hipFree(p));
    }

    // The first failure while building the problem.
    hipblasStatus_t status() const
    {
        return status_;
    }

    // Algos for block 0. The heuristic also runs on the other blocks' descriptors, since
    // hipblasLtMatmul needs the hints recorded on the descriptor it is given.
    hipblasStatus_t heuristic(int requested, std::vector<hipblasLtMatmulHeuristicResult_t>& out)
    {
        out.assign(std::max(requested, 1), hipblasLtMatmulHeuristicResult_t{});
        int count = 0;
        for(size_t blk = 0; blk < descs_.size(); blk++)
        {
            int  blkCount = 0;
            auto status   = hipblasLtMatmulAlgoGetHeuristic(handle_,
                                                          descs_[blk],
                                                          layA_,
                                                          layB_,
                                                          layC_,
                                                          layD_,
                                                          pref_,
                                                          blk == 0 ? int(out.size()) : 1,
                                                          out.data(),
                                                          blk == 0 ? &count : &blkCount);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return status;
        }
        out.resize(count);
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t run(int                          block,
                        const hipblasLtMatmulAlgo_t& algo,
                        const void*                  alpha,
                        const void*                  beta,
                        void*                        workspace,
                        size_t                       workspaceBytes,
                        hipStream_t                  stream) const
    {
        return hipblasLtMatmul(handle_,
                               descs_[block],
                               alpha,
                               ptrsA_[block],
                               layA_,
                               ptrsB_[block],
                               layB_,
                               beta,
                               ptrsC_[block],
                               layC_,
                               ptrsD_[block],
                               layD_,
                               &algo,
                               workspace,
                               workspaceBytes,
                               stream);
    }

private:
    static bool uniform(const std::vector<int64_t>& v)
    {
        return std::all_of(v.begin(), v.end(), [&](int64_t x) { return x == v[0]; });
    }

    static int64_t sum(const std::vector<int64_t>& v)
    {
        return std::accumulate(v.begin(), v.end(), int64_t(0));
    }

    bool ok(hipblasStatus_t status)
    {
        if(status_ == HIPBLAS_STATUS_SUCCESS)
            status_ = status;
        return status_ == HIPBLAS_STATUS_SUCCESS;
    }

    template <typename T>
    bool upload(std::vector<void*>& owned, const std::vector<T>& host)
    {
        void* dev = nullptr;
        if(hipMalloc(&dev, host.size() * sizeof(T)) != hipSuccess)
            return ok(HIPBLAS_STATUS_ALLOC_FAILED);
        owned.push_back(dev);
        if(hipMemcpy(dev, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice)
           != hipSuccess)
            return ok(HIPBLAS_STATUS_INTERNAL_ERROR);
        return true;
    }

    bool setScales(hipblasLtMatmulDesc_t     desc,
                   hipblaslt_scaling_format  format,
                   const std::vector<void*>& scaleA,
                   const std::vector<void*>& scaleB)
    {
        hipblasLtMatmulMatrixScale_t mode
            = format == hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT
                  ? HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT
                  : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        if(!upload(scales_, scaleA) || !upload(scales_, scaleB))
            return false;
        void* arrayA = scales_[scales_.size() - 2];
        void* arrayB = scales_.back();
        return ok(hipblasLtMatmulDescSetAttribute(
                   desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)))
               && ok(hipblasLtMatmulDescSetAttribute(
                   desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)))
               && ok(hipblasLtMatmulDescSetAttribute(
                   desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &arrayA, sizeof(arrayA)))
               && ok(hipblasLtMatmulDescSetAttribute(
                   desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &arrayB, sizeof(arrayB)));
    }

    // A dimension shared by every group uses the scalar layout attribute, which lets the
    // library check kernels against the real size; otherwise it is a device array.
    bool makeLayout(hipblasLtMatrixLayout_t&    layout,
                    hipDataType                 type,
                    const std::vector<int64_t>& rows,
                    const std::vector<int64_t>& cols,
                    const std::vector<int64_t>& ld)
    {
        const void* arrays[3] = {};
        for(int i = 0; i < 3; i++)
        {
            const auto& v = i == 0 ? rows : i == 1 ? cols : ld;
            if(uniform(v))
                continue;
            if(!upload(dims_, v))
                return false;
            arrays[i] = dims_.back();
        }
        if(!ok(hipblasLtGroupedMatrixLayoutCreate(
               &layout, type, int(rows.size()), arrays[0], arrays[1], arrays[2])))
            return false;
        int32_t  width = HIPBLASLT_INTEGER_WIDTH_64;
        uint64_t r = uint64_t(rows[0]), c = uint64_t(cols[0]);
        int64_t  l = ld[0];
        return ok(hipblasLtMatrixLayoutSetAttribute(
                   layout,
                   HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH,
                   &width,
                   sizeof(width)))
               && ok(hipblasLtMatrixLayoutSetAttribute(
                   layout,
                   HIPBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH,
                   &width,
                   sizeof(width)))
               && (arrays[0]
                   || ok(hipblasLtMatrixLayoutSetAttribute(
                       layout, HIPBLASLT_MATRIX_LAYOUT_ROWS, &r, sizeof(r))))
               && (arrays[1]
                   || ok(hipblasLtMatrixLayoutSetAttribute(
                       layout, HIPBLASLT_MATRIX_LAYOUT_COLS, &c, sizeof(c))))
               && (arrays[2]
                   || ok(hipblasLtMatrixLayoutSetAttribute(
                       layout, HIPBLASLT_MATRIX_LAYOUT_LD, &l, sizeof(l))));
    }

    hipblasLtHandle_t                  handle_;
    hipblasStatus_t                    status_ = HIPBLAS_STATUS_SUCCESS;
    hipblasLtMatrixLayout_t            layA_ = nullptr, layB_ = nullptr, layC_ = nullptr;
    hipblasLtMatrixLayout_t            layD_ = nullptr;
    hipblasLtMatmulPreference_t        pref_ = nullptr;
    std::vector<hipblasLtMatmulDesc_t> descs_;
    std::vector<void*>                 ptrsA_, ptrsB_, ptrsC_, ptrsD_, dims_, scales_;
};
