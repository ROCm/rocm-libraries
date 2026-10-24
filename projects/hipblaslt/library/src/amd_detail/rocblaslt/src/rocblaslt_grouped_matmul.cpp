// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Grouped GEMM through rocblaslt_matmul_algo_get_heuristic and rocblaslt_matmul.
//
// Group shapes, pointers, and scalars live in device memory, so the host never sees the
// real problems. The heuristic and the launch setup use placeholder host problems built
// from the size hints. hipblasLtMatmul packs the real per-group records into the head of
// the workspace with a small kernel, then runs the selected kernel from those records.

#include "definitions.h"
#include "handle.h"
#include "rocblaslt.h"
#include "rocblaslt_mat_utils.hpp"
#include "tensile_host.hpp"
#include "utility.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

namespace
{
    // Must match TensileLite::DeviceUserArguments<float>, the record the grouped kernels read.
    struct __attribute__((packed)) GroupedRecord
    {
        uint32_t m;
        uint32_t n;
        uint32_t batch;
        uint32_t k;
        void*    d;
        void*    c;
        void*    a;
        void*    b;
        uint32_t strideD1;
        uint32_t strideD2;
        uint32_t strideC1;
        uint32_t strideC2;
        uint32_t strideA1;
        uint32_t strideA2;
        uint32_t strideB1;
        uint32_t strideB2;
        int8_t   alpha[16];
        int8_t   beta[16];
        void*    scaleA;
        void*    scaleB;
        void*    scaleC;
        void*    scaleD;
        void*    scaleAlphaVec;
        void*    bias;
        int      biasType;
        uint32_t reserved;
        void*    e;
        uint32_t strideE1;
        uint32_t strideE2;
        float    act0;
        float    act1;
        int      activationType;
    };
    static_assert(sizeof(GroupedRecord) == 196, "grouped record layout changed");

    // The record MX kernels read: the scale tensors follow A and B, and the fields from
    // alpha on match GroupedRecord.
    struct __attribute__((packed)) GroupedRecordMX
    {
        uint32_t m;
        uint32_t n;
        uint32_t batch;
        uint32_t k;
        void*    d;
        void*    c;
        void*    a;
        void*    mxsa;
        void*    b;
        void*    mxsb;
        uint32_t strideD1;
        uint32_t strideD2;
        uint32_t strideC1;
        uint32_t strideC2;
        uint32_t strideA1;
        uint32_t strideA2;
        uint32_t strideMXSA1;
        uint32_t strideMXSA2;
        uint32_t strideB1;
        uint32_t strideB2;
        uint32_t strideMXSB1;
        uint32_t strideMXSB2;
        int8_t   alpha[16];
        int8_t   beta[16];
        void*    scaleA;
        void*    scaleB;
        void*    scaleC;
        void*    scaleD;
        void*    scaleAlphaVec;
        void*    bias;
        int      biasType;
        uint32_t reserved;
        void*    e;
        uint32_t strideE1;
        uint32_t strideE2;
        float    act0;
        float    act1;
        int      activationType;
    };
    static_assert(sizeof(GroupedRecordMX) == 228, "grouped MX record layout changed");

    constexpr size_t kRecordAlignment = 256;
    // Used for the reduction dimension when neither a scalar K nor an average hint is set.
    // It only steers selection and split-K; any value gives correct results.
    constexpr int64_t kDefaultK = 512;
    constexpr int64_t kMXBlock  = 32;

    using ScalingFormat = RocblasltContractionProblem::ScalingFormat;

    bool isMXFormat(ScalingFormat f)
    {
        return f == ScalingFormat::Block_32_UE8M0 || f == ScalingFormat::Block_32_UE8M0_32_8_EXT;
    }

    bool isMX(rocblaslt_matmul_desc desc)
    {
        return isMXFormat(desc->scaleAType);
    }

    // Whether a kernel's in-device scale layout (Tensile MXScaleFormat) is the one the
    // caller's scale mode provides: pre-swizzled for the _EXT mode, unswizzled otherwise.
    bool scaleFormatMatches(rocblaslt_handle             handle,
                            rocblaslt_matmul_desc        desc,
                            const rocblaslt_matmul_algo& algo)
    {
        if(!isMX(desc))
            return true;
        constexpr int kHostPreSwizzle = 1;
        int           format          = getMXScaleFormatFromAlgoIndex(handle, algo);
        if(format < 0)
            return false;
        return (desc->scaleAType == ScalingFormat::Block_32_UE8M0_32_8_EXT)
               == (format == kHostPreSwizzle);
    }

    size_t recordBytes(int groupCount, bool mx)
    {
        size_t bytes = (mx ? sizeof(GroupedRecordMX) : sizeof(GroupedRecord))
                       * static_cast<size_t>(groupCount);
        return (bytes + kRecordAlignment - 1) / kRecordAlignment * kRecordAlignment;
    }

    bool isGrouped(rocblaslt_matrix_layout l)
    {
        return l->batch_mode == HIPBLASLT_BATCH_MODE_GROUPED;
    }

    // Host problem shapes: group 0 is rows0 x cols0, every other group rows x cols.
    struct Placeholders
    {
        int64_t rows0 = 0;
        int64_t cols0 = 0;
        int64_t rows  = 0;
        int64_t cols  = 0;
        int64_t k     = 0;

        bool operator==(const Placeholders& o) const
        {
            return rows0 == o.rows0 && cols0 == o.cols0 && rows == o.rows && cols == o.cols
                   && k == o.k;
        }
    };

    // Reduction dimension of A: columns for OP_N, rows otherwise.
    bool kIsUniform(rocblaslt_matmul_desc desc, rocblaslt_matrix_layout matA)
    {
        return (desc->op_A == HIPBLAS_OP_N ? matA->cols_array : matA->rows_array) == nullptr;
    }

    int64_t uniformK(rocblaslt_matmul_desc desc, rocblaslt_matrix_layout matA)
    {
        return static_cast<int64_t>(desc->op_A == HIPBLAS_OP_N ? matA->n : matA->m);
    }

    rocblaslt_status validateGrouped(rocblaslt_matmul_desc   desc,
                                     rocblaslt_matrix_layout matA,
                                     rocblaslt_matrix_layout matB,
                                     rocblaslt_matrix_layout matC,
                                     rocblaslt_matrix_layout matD)
    {
        if(!isGrouped(matA) || !isGrouped(matB) || !isGrouped(matC) || !isGrouped(matD))
        {
            log_error(__func__, "A, B, C, and D layouts must all be grouped");
            return rocblaslt_status_invalid_value;
        }
        int groupCount = matD->group_count;
        if(matA->group_count != groupCount || matB->group_count != groupCount
           || matC->group_count != groupCount)
        {
            log_error(__func__, "grouped layouts have different group counts");
            return rocblaslt_status_invalid_value;
        }
        for(auto l : {matA, matB, matC, matD})
        {
            if(l->order != HIPBLASLT_ORDER_COL)
            {
                log_error(__func__, "grouped layouts must be column major");
                return rocblaslt_status_invalid_value;
            }
        }
        if(matC->type != matD->type)
        {
            log_error(__func__, "C and D types differ");
            return rocblaslt_status_type_mismatch;
        }
        if(desc->scale_type != HIP_R_32F)
        {
            log_error(__func__, "grouped GEMM supports only a 32-bit float scale type");
            return rocblaslt_status_invalid_value;
        }
        const bool mx = isMX(desc);
        auto       noScaleMode = [](ScalingFormat f) {
            return f == ScalingFormat::None || f == ScalingFormat::Scalar;
        };
        if(mx ? desc->scaleBType != desc->scaleAType
              : !noScaleMode(desc->scaleAType) || !noScaleMode(desc->scaleBType))
        {
            log_error(__func__,
                      "grouped GEMM supports only the HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 "
                      "and HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT scale modes, the "
                      "same on A and B");
            return rocblaslt_status_invalid_value;
        }
        if(desc->epilogue != ROCBLASLT_EPILOGUE_DEFAULT || desc->bias || desc->e
           || (!mx && (desc->scaleA || desc->scaleB)) || desc->scaleC || desc->scaleD
           || desc->scaleE || desc->amaxD)
        {
            log_error(__func__, "grouped GEMM does not support epilogues or scaling other than MX");
            return rocblaslt_status_invalid_value;
        }
        if(mx && kIsUniform(desc, matA) && uniformK(desc, matA) % kMXBlock != 0)
        {
            log_error(__func__, "MX grouped GEMM needs K to be a multiple of 32");
            return rocblaslt_status_invalid_value;
        }
        if(desc->pointermode != rocblaslt_pointer_mode_host
           && desc->pointermode != rocblaslt_pointer_mode_device)
        {
            log_error(__func__, "unsupported pointer mode for grouped GEMM", desc->pointermode);
            return rocblaslt_status_invalid_value;
        }
        if(desc->pointermode == rocblaslt_pointer_mode_host
           && (desc->alpha_batch_stride != 0 || desc->beta_batch_stride != 0))
        {
            log_error(__func__, "per-group alpha/beta need device pointer mode");
            return rocblaslt_status_invalid_value;
        }
        return rocblaslt_status_success;
    }

    int64_t placeholderK(rocblaslt_matmul_desc          desc,
                         rocblaslt_matrix_layout        matA,
                         const rocblaslt_grouped_hints& h)
    {
        int64_t k = kIsUniform(desc, matA) ? uniformK(desc, matA) : h.avg_k ? h.avg_k : kDefaultK;
        if(isMX(desc))
            k = (k + kMXBlock - 1) / kMXBlock * kMXBlock;
        return std::max<int64_t>(k, 1);
    }

    // Largest rows and cols of any group.
    void maxGroupShape(rocblaslt_matrix_layout        matD,
                       const rocblaslt_grouped_hints& h,
                       int64_t&                       maxRows,
                       int64_t&                       maxCols)
    {
        const bool    rowsFixed = matD->rows_array == nullptr;
        const bool    colsFixed = matD->cols_array == nullptr;
        const int64_t totalRows = rowsFixed ? 0 : h.max_total_rows;
        const int64_t totalCols = colsFixed ? 0 : h.max_total_cols;
        maxRows = rowsFixed ? static_cast<int64_t>(matD->m) : (h.max_rows ? h.max_rows : totalRows);
        maxCols = colsFixed ? static_cast<int64_t>(matD->n) : (h.max_cols ? h.max_cols : totalCols);
        if(totalRows > 0)
            maxRows = std::min(maxRows, totalRows);
        if(totalCols > 0)
            maxCols = std::min(maxCols, totalCols);
    }

    // Placeholders whose total tile count bounds the real problems for any tile size, so
    // the grid computed from them covers every launch within the hints. With per-group
    // columns summing to at most S, Mmax x S plus G - 1 groups of Mmax x 1 is such a bound,
    // and usually far tighter than G x Mmax x Nmax.
    rocblaslt_status boundPlaceholders(rocblaslt_matmul_desc          desc,
                                       rocblaslt_matrix_layout        matA,
                                       rocblaslt_matrix_layout        matD,
                                       const rocblaslt_grouped_hints& h,
                                       Placeholders&                  out)
    {
        const int64_t groups    = matD->group_count;
        const int64_t totalRows = matD->rows_array == nullptr ? 0 : h.max_total_rows;
        const int64_t totalCols = matD->cols_array == nullptr ? 0 : h.max_total_cols;
        int64_t       maxRows, maxCols;
        maxGroupShape(matD, h, maxRows, maxCols);
        if(maxRows <= 0 || maxCols <= 0)
        {
            log_error(__func__,
                      "grouped D rows and cols need a fixed size or a "
                      "GROUPED_DESC_D_MAX_*_EXT hint");
            return rocblaslt_status_invalid_value;
        }

        out.k     = placeholderK(desc, matA, h);
        out.rows0 = out.rows = maxRows;
        out.cols0 = out.cols = maxCols;

        double best = double(groups) * double(maxRows) * double(maxCols);
        if(totalCols > 0)
        {
            double cost = double(maxRows) * double(totalCols + groups - 1);
            if(cost < best)
            {
                best      = cost;
                out.cols0 = totalCols;
                out.cols  = 1;
            }
        }
        if(totalRows > 0)
        {
            double cost = double(maxCols) * double(totalRows + groups - 1);
            if(cost < best)
            {
                out.rows0 = totalRows;
                out.rows  = 1;
                out.cols0 = out.cols = maxCols;
            }
        }
        return rocblaslt_status_success;
    }

    // Representative shapes for kernel selection: the average hints where given.
    Placeholders averagePlaceholders(rocblaslt_matmul_desc          desc,
                                     rocblaslt_matrix_layout        matA,
                                     rocblaslt_matrix_layout        matD,
                                     const rocblaslt_grouped_hints& h)
    {
        const int64_t groups = std::max<int64_t>(matD->group_count, 1);
        auto pick = [groups](bool fixed, int64_t uniform, int64_t avg, int64_t maxv, int64_t total) {
            if(fixed)
                return uniform;
            if(avg > 0)
                return avg;
            if(total > 0)
                return std::max<int64_t>((total + groups - 1) / groups, 1);
            return maxv;
        };
        Placeholders p;
        p.rows0 = p.rows = pick(matD->rows_array == nullptr,
                                static_cast<int64_t>(matD->m),
                                h.avg_rows,
                                h.max_rows,
                                h.max_total_rows);
        p.cols0 = p.cols = pick(matD->cols_array == nullptr,
                                static_cast<int64_t>(matD->n),
                                h.avg_cols,
                                h.max_cols,
                                h.max_total_cols);
        p.k = placeholderK(desc, matA, h);
        return p;
    }

    // Builds TensileDataGroupedGemm for the placeholder problems through the existing
    // descriptor-based grouped create.
    rocblaslt_status buildGroupedData(rocblaslt_handle        handle,
                                      rocblaslt_matmul_desc   desc,
                                      rocblaslt_matrix_layout matA,
                                      rocblaslt_matrix_layout matB,
                                      rocblaslt_matrix_layout matC,
                                      rocblaslt_matrix_layout matD,
                                      const Placeholders&     p,
                                      std::shared_ptr<void>&  gemmData,
                                      size_t&                 gemmCount)
    {
        const int groups = matD->group_count;

        // Placeholder scalars are host values; 2 keeps the alpha/beta restriction at "any".
        static const float kPlaceholderScalar = 2.0f;
        _rocblaslt_matmul_desc hostDesc;
        hostDesc.copy(*desc);
        hostDesc.pointermode = rocblaslt_pointer_mode_host;
        if(isMX(desc))
        {
            hostDesc.scaleA = reinterpret_cast<void*>(0x5000);
            hostDesc.scaleB = reinterpret_cast<void*>(0x6000);
        }

        std::vector<_rocblaslt_matrix_layout> layouts(4 * static_cast<size_t>(groups));
        std::vector<rocblaslt_matrix_layout>  la(groups), lb(groups), lc(groups), ld(groups);
        std::vector<rocblaslt_matmul_desc>    descs(groups, &hostDesc);
        std::vector<const void*>              alpha(groups, &kPlaceholderScalar);
        std::vector<const void*>              beta(groups, &kPlaceholderScalar);
        // Never dereferenced on the host; distinct so C == D is not assumed.
        std::vector<const void*> a(groups, reinterpret_cast<const void*>(0x1000));
        std::vector<const void*> b(groups, reinterpret_cast<const void*>(0x2000));
        std::vector<const void*> c(groups, reinterpret_cast<const void*>(0x3000));
        std::vector<void*>       d(groups, reinterpret_cast<void*>(0x4000));

        auto setLayout = [](_rocblaslt_matrix_layout& l, hipDataType type, int64_t rows, int64_t cols) {
            l.type = type;
            l.m    = rows;
            l.n    = cols;
            l.ld   = rows;
        };
        for(int g = 0; g < groups; g++)
        {
            int64_t m = g == 0 ? p.rows0 : p.rows;
            int64_t n = g == 0 ? p.cols0 : p.cols;
            auto*   l = &layouts[4 * static_cast<size_t>(g)];
            if(desc->op_A == HIPBLAS_OP_N)
                setLayout(l[0], matA->type, m, p.k);
            else
                setLayout(l[0], matA->type, p.k, m);
            if(desc->op_B == HIPBLAS_OP_N)
                setLayout(l[1], matB->type, p.k, n);
            else
                setLayout(l[1], matB->type, n, p.k);
            setLayout(l[2], matC->type, m, n);
            setLayout(l[3], matD->type, m, n);
            la[g] = &l[0];
            lb[g] = &l[1];
            lc[g] = &l[2];
            ld[g] = &l[3];
        }

        std::vector<rocblaslt::RocGemmProblemTypeV2> problemTypes;
        auto status = rocblaslt_groupedgemm_create_cpp(handle,
                                                       descs,
                                                       alpha,
                                                       a,
                                                       la,
                                                       b,
                                                       lb,
                                                       beta,
                                                       c,
                                                       lc,
                                                       d,
                                                       ld,
                                                       problemTypes,
                                                       gemmData,
                                                       gemmCount);
        if(status == rocblaslt_status_success && gemmCount != static_cast<size_t>(groups))
            status = rocblaslt_status_internal_error;
        return status;
    }

    // The group shapes are never checked on the host, so an algo must also accept the
    // smallest ones the layouts allow: 1 in each dimension set by a device array, or 32 in
    // K for MX. Kernels that assume a minimum size or a size multiple reject these; a group
    // below a kernel's minimum M or N makes it read before the group's A or B.
    rocblaslt_status buildSmallestData(rocblaslt_handle               handle,
                                       rocblaslt_matmul_desc          desc,
                                       rocblaslt_matrix_layout        matA,
                                       rocblaslt_matrix_layout        matB,
                                       rocblaslt_matrix_layout        matC,
                                       rocblaslt_matrix_layout        matD,
                                       const rocblaslt_grouped_hints& h,
                                       std::shared_ptr<void>&         gemmData)
    {
        int64_t maxRows, maxCols;
        maxGroupShape(matD, h, maxRows, maxCols);
        Placeholders p;
        p.rows0 = p.rows = matD->rows_array == nullptr ? maxRows : 1;
        p.cols0 = p.cols = matD->cols_array == nullptr ? maxCols : 1;
        p.k = kIsUniform(desc, matA) ? placeholderK(desc, matA, h) : isMX(desc) ? kMXBlock : 1;
        size_t gemmCount = 0;
        return buildGroupedData(handle, desc, matA, matB, matC, matD, p, gemmData, gemmCount);
    }

    bool algoSupports(rocblaslt_handle      handle,
                      std::shared_ptr<void> gemmData,
                      rocblaslt_matmul_algo algo)
    {
        size_t ws = 0;
        return rocblaslt_is_algo_supported_cpp(handle,
                                               rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM,
                                               gemmData,
                                               algo,
                                               nullptr,
                                               ws)
               == rocblaslt_status_success;
    }

    struct GroupedDims
    {
        const void* rows;
        const void* cols;
        const void* ld;
        int64_t     m;
        int64_t     n;
        int64_t     ld0;
        int32_t     rowsColsWidth;
        int32_t     ldWidth;
    };

    GroupedDims groupedDims(rocblaslt_matrix_layout l)
    {
        return {l->rows_array,
                l->cols_array,
                l->ld_array,
                static_cast<int64_t>(l->m),
                static_cast<int64_t>(l->n),
                l->ld,
                l->rows_cols_int_width,
                l->ld_int_width};
    }

    struct PackArgs
    {
        GroupedDims        a, b, c, d;
        const void* const* aPtrs;
        const void* const* bPtrs;
        const void* const* cPtrs;
        void* const*       dPtrs;
        // 0: host value, 1: one device value, 2: device array of per-group pointers.
        int32_t      alphaMode;
        int32_t      betaMode;
        float        alphaValue;
        float        betaValue;
        const void*  alphaPtr;
        const void*  betaPtr;
        int32_t      groupCount;
        bool         transA;
        bool         transB;
        // MX only: per-group scale tensor pointers, and the multiples the K blocks and the
        // free dimension of each scale tensor are padded to.
        const void* const* aScalePtrs;
        const void* const* bScalePtrs;
        int32_t            mxKBlockMultiple;
        int32_t            mxFreeMultiple;
    };

    __device__ int64_t loadDim(const void* array, int32_t width, int64_t uniform, int g)
    {
        if(array == nullptr)
            return uniform;
        return width == HIPBLASLT_INTEGER_WIDTH_64 ? static_cast<const int64_t*>(array)[g]
                                                   : static_cast<const int32_t*>(array)[g];
    }

    __device__ void groupShape(const GroupedDims& l, int g, int64_t& rows, int64_t& cols, int64_t& ld)
    {
        rows = loadDim(l.rows, l.rowsColsWidth, l.m, g);
        cols = loadDim(l.cols, l.rowsColsWidth, l.n, g);
        ld   = loadDim(l.ld, l.ldWidth, l.ld0, g);
        if(ld == 0)
            ld = rows > 0 ? rows : 1;
    }

    __device__ float loadScalar(int32_t mode, float value, const void* ptr, int g)
    {
        if(mode == 0)
            return value;
        if(mode == 1)
            return *static_cast<const float*>(ptr);
        return *static_cast<const float* const*>(ptr)[g];
    }

    __device__ int64_t roundUp(int64_t v, int64_t multiple)
    {
        return (v + multiple - 1) / multiple * multiple;
    }

    // Packed strides of an MX scale tensor, which has the sizes of its data tensor with K
    // replaced by padded K blocks and the free dimension padded.
    __device__ void mxScaleStrides(
        const PackArgs& args, int64_t k, int64_t freeDim, bool kIsRows, uint32_t& s1, uint32_t& s2)
    {
        int64_t blocks = roundUp((k + kMXBlock - 1) / kMXBlock, args.mxKBlockMultiple);
        int64_t free   = roundUp(freeDim, args.mxFreeMultiple);
        s1             = static_cast<uint32_t>(kIsRows ? blocks : free);
        s2             = static_cast<uint32_t>(blocks * free);
    }

    template <typename Record>
    __global__ void packGroupedRecords(Record templ, PackArgs args, Record* records)
    {
        int g = blockIdx.x * blockDim.x + threadIdx.x;
        if(g >= args.groupCount)
            return;

        int64_t aRows, aCols, ldA, bRows, bCols, ldB, cRows, cCols, ldC, m, n, ldD;
        groupShape(args.a, g, aRows, aCols, ldA);
        groupShape(args.b, g, bRows, bCols, ldB);
        groupShape(args.c, g, cRows, cCols, ldC);
        groupShape(args.d, g, m, n, ldD);
        int64_t k = args.transA ? aRows : aCols;

        float alpha = loadScalar(args.alphaMode, args.alphaValue, args.alphaPtr, g);
        float beta  = loadScalar(args.betaMode, args.betaValue, args.betaPtr, g);
        // k = 0 gives D = beta * C.
        if(k == 0)
            alpha = 0.f;

        Record r = templ;
        r.m      = static_cast<uint32_t>(m);
        r.n             = static_cast<uint32_t>(n);
        r.batch         = 1;
        r.k             = static_cast<uint32_t>(k);
        r.a             = const_cast<void*>(args.aPtrs[g]);
        r.b             = const_cast<void*>(args.bPtrs[g]);
        r.c             = const_cast<void*>(args.cPtrs[g]);
        r.d             = args.dPtrs[g];
        r.strideD1      = static_cast<uint32_t>(ldD);
        r.strideD2      = static_cast<uint32_t>(ldD * n);
        r.strideC1      = static_cast<uint32_t>(ldC);
        r.strideC2      = static_cast<uint32_t>(ldC * cCols);
        r.strideA1      = static_cast<uint32_t>(ldA);
        r.strideA2      = static_cast<uint32_t>(ldA * aCols);
        r.strideB1      = static_cast<uint32_t>(ldB);
        r.strideB2      = static_cast<uint32_t>(ldB * bCols);
        memset(r.alpha, 0, sizeof(r.alpha));
        memset(r.beta, 0, sizeof(r.beta));
        memcpy(r.alpha, &alpha, sizeof(alpha));
        memcpy(r.beta, &beta, sizeof(beta));
        if constexpr(std::is_same_v<Record, GroupedRecordMX>)
        {
            r.mxsa = const_cast<void*>(args.aScalePtrs[g]);
            r.mxsb = const_cast<void*>(args.bScalePtrs[g]);
            mxScaleStrides(args, k, m, args.transA, r.strideMXSA1, r.strideMXSA2);
            mxScaleStrides(args, k, n, !args.transB, r.strideMXSB1, r.strideMXSB2);
        }
        records[g] = r;
    }

    // MX record with the fields from alpha on taken from the standard record.
    GroupedRecordMX mxTemplate(const GroupedRecord& t)
    {
        GroupedRecordMX r{};
        r.batch          = t.batch;
        r.scaleA         = t.scaleA;
        r.scaleB         = t.scaleB;
        r.scaleC         = t.scaleC;
        r.scaleD         = t.scaleD;
        r.scaleAlphaVec  = t.scaleAlphaVec;
        r.bias           = t.bias;
        r.biasType       = t.biasType;
        r.reserved       = t.reserved;
        r.e              = t.e;
        r.strideE1       = t.strideE1;
        r.strideE2       = t.strideE2;
        r.act0           = t.act0;
        r.act1           = t.act1;
        r.activationType = t.activationType;
        return r;
    }

    // Launch state built on the first hipblasLtMatmul and reused while nothing it depends
    // on changes.
    struct GroupedLaunch
    {
        int32_t                algoIndex = -1;
        void*                  workspace = nullptr;
        size_t                 workspaceSize = 0;
        hipStream_t            stream    = nullptr;
        int32_t                groupCount = 0;
        hipblasOperation_t     opA;
        hipblasOperation_t     opB;
        hipDataType            typeA, typeB, typeC, typeD;
        rocblaslt_compute_type computeType;
        bool                   mx = false;
        Placeholders           placeholders;

        std::shared_ptr<void> gemmData;
        GroupedRecord         templ;
    };
} // namespace

rocblaslt_status
    rocblaslt_grouped_matmul_algo_get_heuristic(rocblaslt_handle                  handle,
                                                rocblaslt_matmul_desc             desc,
                                                rocblaslt_matrix_layout           matA,
                                                rocblaslt_matrix_layout           matB,
                                                rocblaslt_matrix_layout           matC,
                                                rocblaslt_matrix_layout           matD,
                                                rocblaslt_matmul_preference       pref,
                                                int                               requestedAlgoCount,
                                                rocblaslt_matmul_heuristic_result heuristicResultsArray[],
                                                int*                              returnAlgoCount)
{
    *returnAlgoCount = 0;
    if(auto status = validateGrouped(desc, matA, matB, matC, matD);
       status != rocblaslt_status_success)
        return status;
    const int groups = matD->group_count;
    if(groups == 0)
    {
        log_error(__func__, "groupCount must be positive to select an algorithm");
        return rocblaslt_status_invalid_value;
    }

    try
    {
        const rocblaslt_grouped_hints& h = pref->grouped;
        Placeholders                   bound;
        if(auto status = boundPlaceholders(desc, matA, matD, h, bound);
           status != rocblaslt_status_success)
            return status;

        desc->grouped_hints     = h;
        desc->grouped_hints_set = true;
        desc->grouped_launch.reset();

        const size_t recBytes = recordBytes(groups, isMX(desc));
        if(pref->max_workspace_bytes < recBytes)
        {
            log_api(__func__, "workspace too small for grouped records", recBytes);
            return rocblaslt_status_success;
        }
        const size_t tensileMax = pref->max_workspace_bytes - recBytes;

        std::shared_ptr<void> avgData;
        size_t                gemmCount = 0;
        auto status = buildGroupedData(handle,
                                       desc,
                                       matA,
                                       matB,
                                       matC,
                                       matD,
                                       averagePlaceholders(desc, matA, matD, h),
                                       avgData,
                                       gemmCount);
        if(status != rocblaslt_status_success)
            return status;

        // Ask for extra candidates: some may not fit the bound problems, and with MX some
        // expect the other scale layout.
        const int candidateCount = isMX(desc) ? std::max(4 * requestedAlgoCount, 32)
                                              : std::max(requestedAlgoCount, 8);
        std::vector<rocblaslt_matmul_heuristic_result> candidates;
        status = rocblaslt_algo_get_heuristic_cpp(handle,
                                                  rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM,
                                                  avgData,
                                                  tensileMax,
                                                  desc->streamk_tile_scheduling_ext,
                                                  effective_uniform_summation_order(handle, desc)
                                                      != 0,
                                                  candidateCount,
                                                  candidates);
        if(status != rocblaslt_status_success)
            return status;

        std::shared_ptr<void> boundData;
        status = buildGroupedData(
            handle, desc, matA, matB, matC, matD, bound, boundData, gemmCount);
        if(status != rocblaslt_status_success)
            return status;
        std::shared_ptr<void> smallestData;
        status = buildSmallestData(handle, desc, matA, matB, matC, matD, h, smallestData);
        if(status != rocblaslt_status_success)
            return status;

        for(auto& candidate : candidates)
        {
            if(*returnAlgoCount >= requestedAlgoCount)
                break;
            size_t ws = 0;
            if(!scaleFormatMatches(handle, desc, candidate.algo))
                continue;
            if(rocblaslt_is_algo_supported_cpp(handle,
                                               rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM,
                                               boundData,
                                               candidate.algo,
                                               nullptr,
                                               ws)
                   != rocblaslt_status_success
               || ws > tensileMax || !algoSupports(handle, smallestData, candidate.algo))
                continue;
            auto& out                     = heuristicResultsArray[*returnAlgoCount];
            out                           = candidate;
            out.algo.max_workspace_bytes  = pref->max_workspace_bytes;
            out.workspaceSize             = ws + recBytes;
            out.state                     = rocblaslt_status_success;
            (*returnAlgoCount)++;
        }
        log_api(__func__, "groupCount", groups, "returnAlgoCount", *returnAlgoCount);
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

rocblaslt_status rocblaslt_grouped_matmul(rocblaslt_handle             handle,
                                          rocblaslt_matmul_desc        desc,
                                          const void*                  alpha,
                                          const void*                  A,
                                          rocblaslt_matrix_layout      matA,
                                          const void*                  B,
                                          rocblaslt_matrix_layout      matB,
                                          const void*                  beta,
                                          const void*                  C,
                                          rocblaslt_matrix_layout      matC,
                                          void*                        D,
                                          rocblaslt_matrix_layout      matD,
                                          const rocblaslt_matmul_algo* algo,
                                          void*                        workspace,
                                          size_t                       workspaceSizeInBytes,
                                          hipStream_t                  stream)
{
    if(auto status = validateGrouped(desc, matA, matB, matC, matD);
       status != rocblaslt_status_success)
        return status;
    const int groups = matD->group_count;
    if(groups == 0)
        return rocblaslt_status_success;

    if(alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr || C == nullptr
       || D == nullptr)
    {
        log_error(__func__, "invalid data pointer");
        return rocblaslt_status_invalid_pointer;
    }
    if(algo == nullptr || !desc->grouped_hints_set)
    {
        log_error(__func__,
                  "grouped GEMM needs an algo from hipblasLtMatmulAlgoGetHeuristic with this "
                  "descriptor");
        return rocblaslt_status_invalid_value;
    }
    const bool mx = isMX(desc);
    if(mx && (desc->scaleA == nullptr || desc->scaleB == nullptr))
    {
        log_error(__func__, "MX grouped GEMM needs the A and B scale pointer arrays");
        return rocblaslt_status_invalid_pointer;
    }
    if(!scaleFormatMatches(handle, desc, *algo))
    {
        log_error(__func__, "the algo expects a different MX scale layout than the scale mode");
        return rocblaslt_status_invalid_value;
    }
    workspaceSizeInBytes = clampWorkspaceSize(__func__, workspaceSizeInBytes);
    const size_t recBytes = recordBytes(groups, mx);
    if(workspace == nullptr || workspaceSizeInBytes < recBytes)
    {
        log_error(__func__, "workspace too small for grouped records", recBytes);
        return rocblaslt_status_invalid_value;
    }

    try
    {
        Placeholders bound;
        if(auto status = boundPlaceholders(desc, matA, matD, desc->grouped_hints, bound);
           status != rocblaslt_status_success)
            return status;

        const int32_t algoIndex = *reinterpret_cast<const int32_t*>(algo->data);
        auto launch = std::static_pointer_cast<GroupedLaunch>(desc->grouped_launch);
        bool reuse  = launch && launch->algoIndex == algoIndex && launch->workspace == workspace
                     && launch->workspaceSize == workspaceSizeInBytes && launch->stream == stream
                     && launch->groupCount == groups && launch->opA == desc->op_A
                     && launch->opB == desc->op_B && launch->typeA == matA->type
                     && launch->typeB == matB->type && launch->typeC == matC->type
                     && launch->typeD == matD->type && launch->computeType == desc->compute_type
                     && launch->mx == mx && launch->placeholders == bound;
        if(!reuse)
        {
            launch = std::make_shared<GroupedLaunch>();
            size_t gemmCount = 0;
            auto   status    = buildGroupedData(
                handle, desc, matA, matB, matC, matD, bound, launch->gemmData, gemmCount);
            if(status != rocblaslt_status_success)
                return status;

            void*  tensileWs     = static_cast<uint8_t*>(workspace) + recBytes;
            size_t tensileWsSize = workspaceSizeInBytes - recBytes;
            auto   rocAlgo       = *algo;
            size_t required      = 0;
            status = rocblaslt_is_algo_supported_cpp(handle,
                                                     rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM,
                                                     launch->gemmData,
                                                     rocAlgo,
                                                     nullptr,
                                                     required);
            if(status != rocblaslt_status_success)
                return status;
            if(required > tensileWsSize)
            {
                log_error(__func__, "workspace too small", required + recBytes);
                return rocblaslt_status_invalid_value;
            }
            std::shared_ptr<void> smallestData;
            status = buildSmallestData(
                handle, desc, matA, matB, matC, matD, desc->grouped_hints, smallestData);
            if(status != rocblaslt_status_success)
                return status;
            if(!algoSupports(handle, smallestData, *algo))
            {
                log_error(__func__,
                          "the algo does not support every group shape the layouts allow");
                return rocblaslt_status_invalid_value;
            }
            status = rocblaslt_makeArgument_cpp(handle,
                                                rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM,
                                                *algo,
                                                nullptr,
                                                tensileWs,
                                                tensileWsSize,
                                                true,
                                                stream,
                                                launch->gemmData);
            if(status != rocblaslt_status_success)
                return status;

            std::vector<GroupedRecord> defaults(groups);
            status = rocblaslt_get_default_user_args(handle,
                                                     rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM,
                                                     launch->gemmData,
                                                     defaults.data());
            if(status != rocblaslt_status_success)
                return status;

            launch->templ         = defaults[0];
            launch->algoIndex     = algoIndex;
            launch->workspace     = workspace;
            launch->workspaceSize = workspaceSizeInBytes;
            launch->stream        = stream;
            launch->groupCount    = groups;
            launch->opA           = desc->op_A;
            launch->opB           = desc->op_B;
            launch->typeA         = matA->type;
            launch->typeB         = matB->type;
            launch->typeC         = matC->type;
            launch->typeD         = matD->type;
            launch->computeType   = desc->compute_type;
            launch->mx            = mx;
            launch->placeholders  = bound;
            desc->grouped_launch  = launch;
        }

        PackArgs args{};
        args.a          = groupedDims(matA);
        args.b          = groupedDims(matB);
        args.c          = groupedDims(matC);
        args.d          = groupedDims(matD);
        args.aPtrs      = static_cast<const void* const*>(A);
        args.bPtrs      = static_cast<const void* const*>(B);
        args.cPtrs      = static_cast<const void* const*>(C);
        args.dPtrs      = static_cast<void* const*>(D);
        args.groupCount = groups;
        args.transA     = desc->op_A != HIPBLAS_OP_N;
        args.transB     = desc->op_B != HIPBLAS_OP_N;
        if(mx)
        {
            const bool padFree    = mxScaleTensorPadsFreeDim();
            args.aScalePtrs       = static_cast<const void* const*>(desc->scaleA);
            args.bScalePtrs       = static_cast<const void* const*>(desc->scaleB);
            args.mxKBlockMultiple = padFree ? 8 : 128 / kMXBlock;
            args.mxFreeMultiple   = padFree ? 32 : 1;
        }
        if(desc->pointermode == rocblaslt_pointer_mode_host)
        {
            args.alphaMode  = 0;
            args.betaMode   = 0;
            args.alphaValue = *static_cast<const float*>(alpha);
            args.betaValue  = *static_cast<const float*>(beta);
        }
        else
        {
            args.alphaMode = desc->alpha_batch_stride ? 2 : 1;
            args.betaMode  = desc->beta_batch_stride ? 2 : 1;
            args.alphaPtr  = alpha;
            args.betaPtr   = beta;
        }

        unsigned block = 256;
        unsigned grid  = (static_cast<unsigned>(groups) + block - 1) / block;
        if(mx)
            hipLaunchKernelGGL(packGroupedRecords<GroupedRecordMX>,
                               dim3(grid),
                               dim3(block),
                               0,
                               stream,
                               mxTemplate(launch->templ),
                               args,
                               static_cast<GroupedRecordMX*>(workspace));
        else
            hipLaunchKernelGGL(packGroupedRecords<GroupedRecord>,
                               dim3(grid),
                               dim3(block),
                               0,
                               stream,
                               launch->templ,
                               args,
                               static_cast<GroupedRecord*>(workspace));
        if(hipGetLastError() != hipSuccess)
        {
            log_error(__func__, "failed to launch the grouped record pack kernel");
            return rocblaslt_status_internal_error;
        }

        log_api(__func__, "groupCount", groups, "algoIndex", algoIndex, "reuse", reuse);
        return rocblaslt_run_user_args_cpp(
            handle, rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM, launch->gemmData, workspace, stream);
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
}
