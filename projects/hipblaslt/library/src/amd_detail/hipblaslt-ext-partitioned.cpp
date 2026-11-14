// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// PartitionedGemm: one GEMM run as slices on several devices (for example the partitions of a
// DPX card). It owns one hipblaslt_ext::Gemm per partition and only uses the public Gemm API.

#include "hipblaslt/hipblaslt-ext.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

namespace hipblaslt_ext
{
    namespace
    {
        // Size in bytes of one element, or 0 for a type this version cannot offset into
        // (sub-byte and block-scaled types).
        size_t elementBytes(hipDataType type)
        {
            switch(type)
            {
            case HIP_R_64F:
                return 8;
            case HIP_R_32F:
            case HIP_R_32I:
                return 4;
            case HIP_R_16F:
            case HIP_R_16BF:
                return 2;
            case HIP_R_8I:
            case HIP_R_8F_E4M3:
            case HIP_R_8F_E5M2:
            case HIP_R_8F_E4M3_FNUZ:
            case HIP_R_8F_E5M2_FNUZ:
                return 1;
            default:
                return 0;
            }
        }

        // Makes `device` current for the lifetime of the object.
        class ScopedDevice
        {
        public:
            explicit ScopedDevice(int device)
            {
                if(hipGetDevice(&m_previous) != hipSuccess)
                    m_previous = -1;
                if(m_previous != device)
                    m_ok = hipSetDevice(device) == hipSuccess;
            }
            ~ScopedDevice()
            {
                if(m_previous >= 0)
                    static_cast<void>(hipSetDevice(m_previous));
            }
            bool ok() const
            {
                return m_ok;
            }
            ScopedDevice(const ScopedDevice&)            = delete;
            ScopedDevice& operator=(const ScopedDevice&) = delete;

        private:
            int  m_previous = -1;
            bool m_ok       = true;
        };

        const void* advance(const void* base, int64_t elements, size_t bytes)
        {
            return static_cast<const char*>(base) + elements * static_cast<int64_t>(bytes);
        }
    } // namespace

    class PartitionedGemm::PartitionedGemmImpl
    {
    public:
        std::vector<std::unique_ptr<Gemm>> gemms;

        hipblasOperation_t   opA, opB;
        hipDataType          typeA, typeB, typeC, typeD;
        hipblasComputeType_t typeCompute;

        bool                 hasProblem = false;
        int64_t              m = 0, n = 0, k = 0, batch = 1;
        int64_t              ld[4]        = {};
        int64_t              stride[4]    = {};
        SplitDim             dim          = SplitDim::M;
        std::vector<int64_t> sliceOffset, sliceSize;

        size_t count() const
        {
            return gemms.size();
        }

        // Checks that every per-partition vector has one entry per partition (or is empty when
        // `allowEmpty`), and that the streams belong to different devices.
        template <typename V>
        bool sized(V const& v, bool allowEmpty = false) const
        {
            return v.size() == count() || (allowEmpty && v.empty());
        }

        hipblasStatus_t checkStreams(std::vector<hipStream_t> const& streams) const
        {
            if(!sized(streams))
                return HIPBLAS_STATUS_INVALID_VALUE;
            std::set<int> devices;
            for(auto stream : streams)
            {
                int device = -1;
                if(hipStreamGetDevice(stream, &device) != hipSuccess)
                    return HIPBLAS_STATUS_INVALID_VALUE;
                if(!devices.insert(device).second)
                    return HIPBLAS_STATUS_INVALID_VALUE;
            }
            return HIPBLAS_STATUS_SUCCESS;
        }
    };

    PartitionedGemm::PartitionedGemm(const std::vector<hipblasLtHandle_t>& handles,
                                     hipblasOperation_t                    opA,
                                     hipblasOperation_t                    opB,
                                     hipDataType                           typeA,
                                     hipDataType                           typeB,
                                     hipDataType                           typeC,
                                     hipDataType                           typeD,
                                     hipblasComputeType_t                  typeCompute)
        : pimpl(std::make_unique<PartitionedGemmImpl>())
    {
        pimpl->opA         = opA;
        pimpl->opB         = opB;
        pimpl->typeA       = typeA;
        pimpl->typeB       = typeB;
        pimpl->typeC       = typeC;
        pimpl->typeD       = typeD;
        pimpl->typeCompute = typeCompute;
        for(auto handle : handles)
            pimpl->gemms.push_back(
                std::make_unique<Gemm>(handle, opA, opB, typeA, typeB, typeC, typeD, typeCompute));
    }

    PartitionedGemm::~PartitionedGemm()                                     = default;
    PartitionedGemm::PartitionedGemm(PartitionedGemm&&) noexcept            = default;
    PartitionedGemm& PartitionedGemm::operator=(PartitionedGemm&&) noexcept = default;

    size_t PartitionedGemm::getPartitionCount() const
    {
        return pimpl ? pimpl->count() : 0;
    }

    hipblasStatus_t PartitionedGemm::setProblem(int64_t                      m,
                                                int64_t                      n,
                                                int64_t                      k,
                                                int64_t                      batch_count,
                                                int64_t                      lda,
                                                int64_t                      ldb,
                                                int64_t                      ldc,
                                                int64_t                      ldd,
                                                int64_t                      strideA,
                                                int64_t                      strideB,
                                                int64_t                      strideC,
                                                int64_t                      strideD,
                                                SplitDim                     dim,
                                                const PartitionedGemmInputs& inputs)
    {
        auto&        s     = *pimpl;
        const size_t parts = s.count();
        if(parts == 0 || m <= 0 || n <= 0 || k <= 0 || batch_count <= 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        if((s.opA != HIPBLAS_OP_N && s.opA != HIPBLAS_OP_T)
           || (s.opB != HIPBLAS_OP_N && s.opB != HIPBLAS_OP_T))
            return HIPBLAS_STATUS_NOT_SUPPORTED;

        const size_t bytes[4] = {elementBytes(s.typeA),
                                 elementBytes(s.typeB),
                                 elementBytes(s.typeC),
                                 elementBytes(s.typeD)};
        for(size_t b : bytes)
            if(b == 0)
                return HIPBLAS_STATUS_NOT_SUPPORTED;

        // One pointer per partition for every matrix.
        const size_t have[4] = {inputs.a.size(), inputs.b.size(), inputs.c.size(), inputs.d.size()};
        for(int t = 0; t < 4; t++)
            if(have[t] != parts)
                return HIPBLAS_STATUS_INVALID_VALUE;
        if(inputs.alpha == nullptr || inputs.beta == nullptr)
            return HIPBLAS_STATUS_INVALID_VALUE;

        // Slices along the split dimension: the first N-1 take size/N, the last takes the rest.
        const int64_t total = dim == SplitDim::M ? m : n;
        const int64_t base  = total / static_cast<int64_t>(parts);
        if(base == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        std::vector<int64_t> offset(parts), size(parts);
        for(size_t i = 0; i < parts; i++)
        {
            offset[i] = static_cast<int64_t>(i) * base;
            size[i]   = i + 1 < parts ? base : total - offset[i];
        }

        const std::vector<const void*> dPointers(inputs.d.begin(), inputs.d.end());
        const int64_t                  leading[4] = {lda, ldb, ldc, ldd};
        const int64_t                  strides[4] = {strideA, strideB, strideC, strideD};
        for(size_t i = 0; i < parts; i++)
        {
            // Element offset of this partition's slice inside each (whole) matrix. The
            // matrices are column-major.
            int64_t elems[4] = {0, 0, 0, 0};
            if(dim == SplitDim::M)
            {
                elems[0] = s.opA == HIPBLAS_OP_N ? offset[i] : offset[i] * lda;
                elems[2] = offset[i];
                elems[3] = offset[i];
            }
            else
            {
                elems[1] = s.opB == HIPBLAS_OP_N ? offset[i] * ldb : offset[i];
                elems[2] = offset[i] * ldc;
                elems[3] = offset[i] * ldd;
            }

            const void*        ptr[4];
            const void* const* lists[4]
                = {inputs.a.data(), inputs.b.data(), inputs.c.data(), dPointers.data()};
            for(int t = 0; t < 4; t++)
            {
                const void* p = lists[t][i];
                if(p == nullptr)
                    return HIPBLAS_STATUS_INVALID_VALUE;
                ptr[t] = advance(p, elems[t], bytes[t]);
            }

            GemmInputs slice;
            slice.setA(ptr[0]);
            slice.setB(ptr[1]);
            slice.setC(ptr[2]);
            slice.setD(ptr[3]);
            slice.setAlpha(inputs.alpha);
            slice.setBeta(inputs.beta);

            GemmEpilogue    epilogue;
            GemmProblemType type(s.opA, s.opB, s.typeA, s.typeB, s.typeC, s.typeD, s.typeCompute);
            const int64_t   mi = dim == SplitDim::M ? size[i] : m;
            const int64_t   ni = dim == SplitDim::N ? size[i] : n;
            auto            st = s.gemms[i]->setProblem(mi,
                                             ni,
                                             k,
                                             batch_count,
                                             leading[0],
                                             leading[1],
                                             leading[2],
                                             leading[3],
                                             strides[0],
                                             strides[1],
                                             strides[2],
                                             strides[3],
                                             epilogue,
                                             slice,
                                             type);
            if(st != HIPBLAS_STATUS_SUCCESS)
                return st;
        }

        s.m = m, s.n = n, s.k = k, s.batch = batch_count;
        for(int t = 0; t < 4; t++)
        {
            s.ld[t]     = leading[t];
            s.stride[t] = strides[t];
        }
        s.dim         = dim;
        s.sliceOffset = std::move(offset);
        s.sliceSize   = std::move(size);
        s.hasProblem  = true;
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t
        PartitionedGemm::getSlice(size_t partition, int64_t& offset, int64_t& size) const
    {
        auto const& s = *pimpl;
        if(!s.hasProblem || partition >= s.count())
            return HIPBLAS_STATUS_INVALID_VALUE;
        offset = s.sliceOffset[partition];
        size   = s.sliceSize[partition];
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t PartitionedGemm::algoGetHeuristic(
        const int                                      requestedAlgoCount,
        const GemmPreference&                          pref,
        std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults)
    {
        auto& s = *pimpl;
        if(!s.hasProblem)
            return HIPBLAS_STATUS_INVALID_VALUE;

        // Partition 0 proposes; an algorithm stays only if every partition accepts its slice.
        std::vector<hipblasLtMatmulHeuristicResult_t> proposed;
        auto st = s.gemms[0]->algoGetHeuristic(requestedAlgoCount, pref, proposed);
        if(st != HIPBLAS_STATUS_SUCCESS)
            return st;

        heuristicResults.clear();
        for(auto const& result : proposed)
        {
            auto   merged = result;
            bool   ok     = true;
            size_t ws     = result.workspaceSize;
            for(size_t i = 0; i < s.count() && ok; i++)
            {
                auto   algo = result.algo;
                size_t need = 0;
                if(s.gemms[i]->isAlgoSupported(algo, need) != HIPBLAS_STATUS_SUCCESS)
                    ok = false;
                else
                    ws = std::max(ws, need);
            }
            if(ok)
            {
                merged.workspaceSize = ws;
                heuristicResults.push_back(merged);
            }
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t PartitionedGemm::isAlgoSupported(hipblasLtMatmulAlgo_t& algo,
                                                     std::vector<size_t>&   workspaceSizeInBytes)
    {
        auto& s = *pimpl;
        if(!s.hasProblem)
            return HIPBLAS_STATUS_INVALID_VALUE;
        workspaceSizeInBytes.assign(s.count(), 0);
        for(size_t i = 0; i < s.count(); i++)
        {
            auto copy = algo;
            auto st   = s.gemms[i]->isAlgoSupported(copy, workspaceSizeInBytes[i]);
            if(st != HIPBLAS_STATUS_SUCCESS)
                return st;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    void PartitionedGemm::setMaxWorkspaceBytes(size_t workspaceBytes)
    {
        for(auto& gemm : pimpl->gemms)
            gemm->setMaxWorkspaceBytes(workspaceBytes);
    }

    hipblasStatus_t PartitionedGemm::initialize(const hipblasLtMatmulAlgo_t&    algo,
                                                const std::vector<void*>&       workspaces,
                                                bool                            useUserArgs,
                                                const std::vector<hipStream_t>& streams)
    {
        auto& s = *pimpl;
        if(!s.hasProblem || !s.sized(workspaces))
            return HIPBLAS_STATUS_INVALID_VALUE;
        if(auto st = s.checkStreams(streams); st != HIPBLAS_STATUS_SUCCESS)
            return st;
        for(size_t i = 0; i < s.count(); i++)
        {
            int device = -1;
            if(hipStreamGetDevice(streams[i], &device) != hipSuccess)
                return HIPBLAS_STATUS_INVALID_VALUE;
            ScopedDevice guard(device);
            if(!guard.ok())
                return HIPBLAS_STATUS_INVALID_VALUE;
            auto st = s.gemms[i]->initialize(algo, workspaces[i], useUserArgs, streams[i]);
            if(st != HIPBLAS_STATUS_SUCCESS)
                return st;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t PartitionedGemm::run(const std::vector<hipStream_t>& streams,
                                         const std::vector<hipEvent_t>&  start,
                                         const std::vector<hipEvent_t>&  stop)
    {
        auto& s = *pimpl;
        if(!s.hasProblem || !s.sized(start, true) || !s.sized(stop, true))
            return HIPBLAS_STATUS_INVALID_VALUE;
        if(auto st = s.checkStreams(streams); st != HIPBLAS_STATUS_SUCCESS)
            return st;
        for(size_t i = 0; i < s.count(); i++)
        {
            int device = -1;
            if(hipStreamGetDevice(streams[i], &device) != hipSuccess)
                return HIPBLAS_STATUS_INVALID_VALUE;
            ScopedDevice guard(device);
            if(!guard.ok())
                return HIPBLAS_STATUS_INVALID_VALUE;
            auto st = s.gemms[i]->run(
                streams[i], start.empty() ? nullptr : start[i], stop.empty() ? nullptr : stop[i]);
            if(st != HIPBLAS_STATUS_SUCCESS)
                return st;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t PartitionedGemm::run(const std::vector<void*>&       deviceUserArgs,
                                         const std::vector<hipStream_t>& streams,
                                         const std::vector<hipEvent_t>&  start,
                                         const std::vector<hipEvent_t>&  stop)
    {
        // Reserved: the signature is fixed now, the feature comes later.
        for(void* args : deviceUserArgs)
            if(args != nullptr)
                return HIPBLAS_STATUS_NOT_SUPPORTED;
        return run(streams, start, stop);
    }

    hipblasStatus_t
        PartitionedGemm::getDefaultValueForDeviceUserArguments(const std::vector<void*>&)
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
} // namespace hipblaslt_ext
