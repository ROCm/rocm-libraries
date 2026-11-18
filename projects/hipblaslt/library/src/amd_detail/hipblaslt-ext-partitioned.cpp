// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// PartitionedGemm: one GEMM run as slices on several devices (for example the partitions of a
// DPX card). It owns one hipblaslt_ext::Gemm per partition and uses the public Gemm API, plus
// the device of each handle to give every partition the slice whose memory is on its device.

// The library always compiles in every experimental feature, so it enables the switch.
#define ROCM_ENABLE_EXPERIMENTAL_APIS 1
#include "hipblaslt/hipblaslt-experimental.h"

#include "hipblaslt-ext-inputs.hpp"
#include "hipblaslt_internal.hpp"
#include "utility.hpp"

#include <Tensile/SliceAlignment.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace hipblaslt_experimental
{
    using hipblaslt_ext::Gemm;
    using hipblaslt_ext::GemmEpilogue;
    using hipblaslt_ext::GemmInputs;
    using hipblaslt_ext::GemmPreference;
    using hipblaslt_ext::GemmProblemType;

    namespace
    {
        // The pointers of every partition, as setProblem works with them: one entry per matrix
        // copy, taken from the per-partition GemmInputs.
        struct PartitionInputs
        {
            std::vector<const void*> a, b, c;
            std::vector<void*>       d;
            const void*              alpha      = nullptr;
            const void*              beta       = nullptr;
            bool                     storePeerD = false;
        };

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

        // The device that holds the page at `p`, or -1 when that is not known (host or managed
        // memory). Works for hipMalloc and for VMM mappings alike.
        int pageDevice(const void* p)
        {
            hipPointerAttribute_t attr;
            if(hipPointerGetAttributes(&attr, p) != hipSuccess)
            {
                static_cast<void>(hipGetLastError());
                return -1;
            }
            return attr.type == hipMemoryTypeDevice ? attr.device : -1;
        }

        // Whether kernels on device `from` can access memory on device `to`. Capability only: it is
        // the caller's job to enable peer access. Used for a hint only, never to reject a problem.
        bool canReach(int from, int to)
        {
            if(from == to)
                return true;
            int ok = 0;
            if(hipDeviceCanAccessPeer(&ok, from, to) != hipSuccess)
            {
                static_cast<void>(hipGetLastError());
                return false;
            }
            return ok != 0;
        }

        // "dddd:bb:dd" of a device: its PCI domain, bus and device without the function. The
        // partitions of one GPU (DPX, ...) share it and differ in the function.
        std::string gpuOf(int device)
        {
            char id[64] = {};
            if(hipDeviceGetPCIBusId(id, sizeof(id), device) != hipSuccess)
            {
                static_cast<void>(hipGetLastError());
                return "";
            }
            std::string bdf(id);
            const auto  dot = bdf.rfind('.');
            return dot == std::string::npos ? bdf : bdf.substr(0, dot);
        }

        // Copy sets to partitions: allowed[j][p] says copy set j may go to partition p. Fills
        // `owner` (copy set j -> partition) with an assignment that keeps every copy set allowed,
        // trying the given order first. Returns false when there is none.
        bool matchCopies(std::vector<std::vector<bool>> const& allowed,
                         std::vector<size_t>&                  owner,
                         std::vector<bool>&                    used,
                         size_t                                j)
        {
            const size_t parts = allowed.size();
            if(j == parts)
                return true;
            for(size_t step = 0; step < parts; step++)
            {
                const size_t p = (j + step) % parts;
                if(used[p] || !allowed[j][p])
                    continue;
                used[p]  = true;
                owner[j] = p;
                if(matchCopies(allowed, owner, used, j + 1))
                    return true;
                used[p] = false;
            }
            return false;
        }

        // One split matrix as the slice placement sees it: column-major, and the slice is either
        // whole columns (`columns`) or rows of every column. `other` is the size of the dimension
        // that is not split.
        struct SplitMatrix
        {
            const char*        name;
            const void* const* pointers; // one per partition
            int64_t            ld;
            size_t             bytes;
            bool               columns;
            int64_t            other;
        };

        // Fraction of slice [offset, offset + size) of `m`, as partition `i` addresses it, that
        // is on `device`, from a few sampled elements of the first batch.
        double
            localFraction(SplitMatrix const& m, size_t i, int64_t offset, int64_t size, int device)
        {
            constexpr int kSamples = 8;
            int           local    = 0;
            for(int j = 0; j < kSamples; j++)
            {
                const int64_t along  = offset + size * j / kSamples;
                const int64_t across = m.other * j / kSamples;
                const int64_t elem   = m.columns ? along * m.ld : along + across * m.ld;
                local += pageDevice(advance(m.pointers[i], elem, m.bytes)) == device;
            }
            return double(local) / kSamples;
        }
    } // namespace

    class PartitionedGemm::PartitionedGemmImpl
    {
    public:
        std::vector<std::unique_ptr<Gemm>> gemms;
        std::vector<int>                   devices; // device of each partition's handle
        bool                               gfx1250 = true; // every partition is a gfx1250 device

        hipblasOperation_t   opA, opB;
        hipDataType          typeA, typeB, typeC, typeD;
        hipblasComputeType_t typeCompute;

        bool                 hasProblem = false;
        int64_t              m = 0, n = 0, k = 0, batch = 1;
        int64_t              ld[4]        = {};
        int64_t              stride[4]    = {};
        SplitDim             dim          = SplitDim::M;
        std::vector<int64_t> sliceOffset, sliceSize;

        // Partitions whose D slice (or the peer D they also store) is memory of another device:
        // run() publishes their writes. One host word per partition (device, word) is the
        // target of that stream write.
        std::vector<bool>                      publishD;
        std::vector<std::pair<int, uint64_t*>> publishWords;

        ~PartitionedGemmImpl()
        {
            releasePublishWords();
        }

        void releasePublishWords()
        {
            for(auto& [device, word] : publishWords)
                if(word != nullptr)
                {
                    ScopedDevice guard(device);
                    static_cast<void>(hipHostFree(word));
                }
            publishWords.clear();
        }

        // The word that partition `i` writes on `device`, or nullptr when it cannot be allocated.
        uint64_t* publishWord(size_t i, int device)
        {
            if(publishWords.size() != count())
            {
                releasePublishWords();
                publishWords.assign(count(), {-1, nullptr});
            }
            auto& [owner, word] = publishWords[i];
            if(word != nullptr && owner == device)
                return word;
            if(word != nullptr)
            {
                ScopedDevice guard(owner);
                static_cast<void>(hipHostFree(word));
                word = nullptr;
            }
            ScopedDevice guard(device);
            void*        p = nullptr;
            if(!guard.ok()
               || hipHostMalloc(&p, sizeof(uint64_t), hipHostMallocCoherent | hipHostMallocMapped)
                      != hipSuccess)
                return nullptr;
            owner = device;
            word  = static_cast<uint64_t*>(p);
            *word = 0;
            return word;
        }

        size_t count() const
        {
            return gemms.size();
        }

        // Checks that every per-partition vector has one entry per partition (or is empty when
        // `allowEmpty`).
        template <typename V>
        bool sized(V const& v, bool allowEmpty = false) const
        {
            return v.size() == count() || (allowEmpty && v.empty());
        }

        hipblasStatus_t checkStreams(std::vector<hipStream_t> const& streams) const
        {
            if(!sized(streams))
                return HIPBLAS_STATUS_INVALID_VALUE;
            for(size_t i = 0; i < streams.size(); i++)
            {
                int device = -1;
                if(hipStreamGetDevice(streams[i], &device) != hipSuccess)
                    return HIPBLAS_STATUS_INVALID_VALUE;
                // Stream i runs partition i, whose kernels are loaded for its handle's device.
                // In SPX mode every partition is on the same device, so the streams may share
                // a device (and even be one stream, which only serializes the partitions).
                if(device != devices[i])
                {
                    log_error(__func__,
                              "stream",
                              i,
                              "is on device",
                              device,
                              "but its handle is on device",
                              devices[i]);
                    return HIPBLAS_STATUS_INVALID_VALUE;
                }
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
        // A null handle makes the object unusable (no partitions) instead of failing later.
        if(std::find(handles.begin(), handles.end(), nullptr) != handles.end())
            return;
        for(auto handle : handles)
        {
            pimpl->gemms.push_back(
                std::make_unique<Gemm>(handle, opA, opB, typeA, typeB, typeC, typeD, typeCompute));
            pimpl->devices.push_back(handle ? reinterpret_cast<rocblaslt_handle>(handle)->device
                                            : -1);
        }
        // PartitionedGemm is supported on gfx1250 only: setProblem refuses any other device.
        for(int device : pimpl->devices)
        {
            hipDeviceProp_t props;
            if(hipGetDeviceProperties(&props, device) != hipSuccess
               || std::strncmp(props.gcnArchName, "gfx1250", 7) != 0)
                pimpl->gfx1250 = false;
        }
    }

    PartitionedGemm::~PartitionedGemm()                                     = default;
    PartitionedGemm::PartitionedGemm(PartitionedGemm&&) noexcept            = default;
    PartitionedGemm& PartitionedGemm::operator=(PartitionedGemm&&) noexcept = default;

    size_t PartitionedGemm::getPartitionCount() const
    {
        return pimpl ? pimpl->count() : 0;
    }

    hipblasStatus_t PartitionedGemm::setProblem(int64_t                        m,
                                                int64_t                        n,
                                                int64_t                        k,
                                                int64_t                        batch_count,
                                                int64_t                        lda,
                                                int64_t                        ldb,
                                                int64_t                        ldc,
                                                int64_t                        ldd,
                                                int64_t                        strideA,
                                                int64_t                        strideB,
                                                int64_t                        strideC,
                                                int64_t                        strideD,
                                                SplitDim                       dim,
                                                const std::vector<GemmInputs>& gemmInputs,
                                                bool                           storePeerD)
    {
        if(!pimpl) // moved from
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto&        s     = *pimpl;
        const size_t parts = s.count();
        // All or nothing: a failure below must not leave the previous problem usable with some
        // partitions already switched to the new one.
        s.hasProblem = false;
        if(!s.gfx1250)
        {
            log_error(__func__, "PartitionedGemm is supported on gfx1250 devices only");
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }
        if(parts == 0 || m <= 0 || n <= 0 || k <= 0 || batch_count <= 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        // A batch split writes different batches of D from different partitions, so D must have
        // a batch stride; with stride 0 they would all write the same matrix.
        if(dim == SplitDim::Batch && batch_count > 1 && strideD == 0)
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

        // Hints cost device queries, so none of their work runs unless HIPBLASLT_LOG_MASK has 4.
        const bool hints = (get_logger_layer_mode() & rocblaslt_layer_mode_log_hints) != 0;

        // Where the partitions are, for HIPBLASLT_LOG_MASK=4.
        if(hints)
        {
            const std::set<int> distinct(s.devices.begin(), s.devices.end());
            if(distinct.size() == 1)
                log_hints(__func__,
                          "PartitionedGemm: all",
                          parts,
                          "partitions are on device",
                          s.devices[0],
                          "(SPX mode): the result is correct, but there is no speed benefit");
            else
            {
                std::set<std::string> gpus;
                for(int device : s.devices)
                    gpus.insert(gpuOf(device));
                if(gpus.size() == 1 && !gpus.begin()->empty())
                    log_hints(__func__,
                              "PartitionedGemm: the partitions are",
                              parts,
                              "partitions of one GPU");
                else
                    log_hints(__func__,
                              "PartitionedGemm: the partitions are on different GPUs: the slices"
                              " that are not local travel over the interconnect");
            }
        }

        // One GemmInputs per partition. A PartitionedGemm runs a plain GEMM, so every pointer
        // beyond A, B, C, D, alpha and beta must be null.
        if(gemmInputs.size() != parts)
            return HIPBLAS_STATUS_INVALID_VALUE;
        PartitionInputs inputs;
        inputs.storePeerD = storePeerD;
        for(auto const& g : gemmInputs)
        {
            if(g.getBias() || g.getScaleA() || g.getScaleB() || g.getScaleC() || g.getScaleD()
               || g.getScaleAux() || g.getScaleAlphaVec() || g.getScaleAlphaScalar() || g.getAux()
               || g.getAmaxD())
                return HIPBLAS_STATUS_NOT_SUPPORTED;
            if(g.getAlpha() == nullptr || g.getBeta() == nullptr)
                return HIPBLAS_STATUS_INVALID_VALUE;
            inputs.a.push_back(g.getA());
            inputs.b.push_back(g.getB());
            inputs.c.push_back(g.getC());
            inputs.d.push_back(const_cast<void*>(g.getD()));
        }
        inputs.alpha = gemmInputs[0].getAlpha();
        inputs.beta  = gemmInputs[0].getBeta();
        // storePeerD: partition i also writes the other partition's D. Two partitions and two
        // different D buffers only.
        if(inputs.storePeerD)
        {
            if(parts != 2)
                return HIPBLAS_STATUS_INVALID_VALUE;
            // Two separate buffers of the whole D: the same pointer or overlapping extents would
            // have a partition write the other one's slice.
            const uint64_t extent
                = uint64_t((batch_count - 1) * std::abs(strideD) + (n - 1) * ldd + m) * bytes[3];
            const auto d0 = reinterpret_cast<uintptr_t>(inputs.d[0]);
            const auto d1 = reinterpret_cast<uintptr_t>(inputs.d[1]);
            if((d0 > d1 ? d0 - d1 : d1 - d0) < extent)
                return HIPBLAS_STATUS_INVALID_VALUE;
        }

        // A matrix passed as different pointers (copies) is matched to the partitions by device:
        // partition p uses the copy on its device, whatever the order of the list. All copied
        // matrices must agree on one assignment (all in order, or all swapped the same way);
        // otherwise some partition would read or write the wrong device's copy, which is an error.
        // Shared matrices (one pointer repeated) and copies on no partition's device do not count.
        PartitionInputs in = inputs;
        {
            const char*                     names[4] = {"A", "B", "C", "D"};
            const std::vector<const void*>  d0(inputs.d.begin(), inputs.d.end());
            const std::vector<const void*>* lists[4] = {&inputs.a, &inputs.b, &inputs.c, &d0};
            std::vector<std::vector<bool>>  allowed(parts, std::vector<bool>(parts, true));
            std::ostringstream              layout;
            for(int t = 0; t < 4; t++)
            {
                auto const& list = *lists[t];
                if(std::all_of(
                       list.begin(), list.end(), [&](const void* p) { return p == list[0]; }))
                    continue;
                layout << " " << names[t] << ":";
                for(size_t j = 0; j < parts; j++)
                {
                    const int device = pageDevice(list[j]);
                    layout << " " << device;
                    if(std::find(s.devices.begin(), s.devices.end(), device) == s.devices.end())
                        continue;
                    for(size_t p = 0; p < parts; p++)
                        if(s.devices[p] != device)
                            allowed[j][p] = false;
                }
            }
            std::vector<size_t> owner(parts);
            std::vector<bool>   used(parts, false);
            if(!matchCopies(allowed, owner, used, 0))
            {
                std::ostringstream devices;
                for(int device : s.devices)
                    devices << " " << device;
                log_error(__func__,
                          "PartitionedGemm: the copies of A, B, C and D cannot be matched to one"
                          " partition each by device. Partition devices:",
                          devices.str(),
                          "devices of the copies, in list order:",
                          layout.str());
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
            for(size_t j = 0; j < parts; j++)
            {
                in.a[owner[j]] = inputs.a[j];
                in.b[owner[j]] = inputs.b[j];
                in.c[owner[j]] = inputs.c[j];
                in.d[owner[j]] = inputs.d[j];
                if(owner[j] != j)
                    log_hints(__func__,
                              "PartitionedGemm: copy set",
                              j,
                              "is on device",
                              s.devices[owner[j]],
                              "and is used by partition",
                              owner[j]);
            }
        }

        // Slices along the split dimension, with the rule the TensileLite client uses for
        // --split-devices (Tensile/SliceAlignment.hpp): a slice starts `offset * step` elements
        // into each matrix it cuts (the same offsets the loop below applies), and the first N-1
        // slices take size/N rounded up so that every start is a multiple of 256 bytes, when that
        // changes the balance by at most 1%; otherwise size/N. The split dimension is the
        // caller's choice, so alignment never refuses a problem.
        namespace align = TensileLite::SliceAlignment;
        const int64_t total = dim == SplitDim::M ? m : dim == SplitDim::N ? n : batch_count;
        // Elements a slice offset moves per unit of the split dimension, 0 for a matrix the
        // split does not cut.
        int64_t step[4] = {0, 0, 0, 0};
        if(dim == SplitDim::M)
        {
            step[0] = s.opA == HIPBLAS_OP_N ? 1 : lda;
            step[2] = 1;
            step[3] = 1;
        }
        else if(dim == SplitDim::N)
        {
            step[1] = s.opB == HIPBLAS_OP_N ? ldb : 1;
            step[2] = ldc;
            step[3] = ldd;
        }
        else
        {
            step[0] = strideA;
            step[1] = strideB;
            step[2] = strideC;
            step[3] = strideD;
        }
        const char*               names[4] = {"A", "B", "C", "D"};
        std::vector<align::Matrix> cut;
        for(int t = 0; t < 4; t++)
            if(step[t] != 0)
                cut.push_back({names[t], step[t], static_cast<int64_t>(bytes[t]) * 8});
        const align::Result aligned = align::split(total, parts, cut, align::Mode::Auto);
        const int64_t       base    = aligned.share;
        if(base == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        std::vector<int64_t> offset(parts), size(parts);
        for(size_t i = 0; i < parts; i++)
        {
            offset[i] = static_cast<int64_t>(i) * base;
            size[i]   = i + 1 < parts ? base : total - offset[i];
        }
        if(hints)
        {
            const align::Dim alignDim = dim == SplitDim::M   ? align::Dim::M
                                        : dim == SplitDim::N ? align::Dim::N
                                                             : align::Dim::Batch;
            for(auto const& line : align::hints(aligned, alignDim, parts))
                log_hints(__func__, "PartitionedGemm:", line);
            const void* bases[4] = {in.a[0], in.b[0], in.c[0], in.d[0]};
            for(int t = 0; t < 4; t++)
                if(step[t] != 0 && bases[t] != nullptr
                   && reinterpret_cast<uintptr_t>(bases[t]) % align::kAlignBytes != 0)
                    log_hints(__func__,
                              "PartitionedGemm: the base pointer of",
                              names[t],
                              "is not 256-byte aligned, which slows every partition; allocate it"
                              " on a 256-byte boundary");
            // An M split gives every partition a run of `share` rows in each column of C and D;
            // a run under 256 bytes shares cache lines with the neighbouring partition.
            if(dim == SplitDim::M)
                for(int t = 2; t < 4; t++)
                    if(base * static_cast<int64_t>(bytes[t]) < align::kAlignBytes)
                        log_hints(__func__,
                                  "PartitionedGemm: each partition writes",
                                  base * static_cast<int64_t>(bytes[t]),
                                  "bytes per column of",
                                  names[t],
                                  ", under 256: the partitions share cache lines; consider"
                                  " SplitDim::N");
        }

        const std::vector<const void*> dPointers(in.d.begin(), in.d.end());
        const int64_t                  leading[4] = {lda, ldb, ldc, ldd};
        const int64_t                  strides[4] = {strideA, strideB, strideC, strideD};

        // The matrices split by `dim`: op(A) with M, op(B) with N, and C and D with both. A batch
        // split splits every matrix with a nonzero batch stride; a slice is `size` whole
        // matrices, `stride` elements apart, so it is sampled like whole columns of length
        // `stride`.
        std::vector<SplitMatrix> split;
        if(dim == SplitDim::Batch)
        {
            const char*        names[4] = {"A", "B", "C", "D"};
            const void* const* lists[4] = {in.a.data(), in.b.data(), in.c.data(), dPointers.data()};
            for(int t = 0; t < 4; t++)
                if(strides[t] != 0)
                    split.push_back({names[t], lists[t], strides[t], bytes[t], true, strides[t]});
        }
        else
        {
            if(dim == SplitDim::M)
                split.push_back({"A", in.a.data(), lda, bytes[0], s.opA == HIPBLAS_OP_T, k});
            else
                split.push_back({"B", in.b.data(), ldb, bytes[1], s.opB == HIPBLAS_OP_N, k});
            split.push_back(
                {"C", in.c.data(), ldc, bytes[2], dim == SplitDim::N, dim == SplitDim::M ? n : m});
            split.push_back({"D",
                             dPointers.data(),
                             ldd,
                             bytes[3],
                             dim == SplitDim::N,
                             dim == SplitDim::M ? n : m});
        }

        // Partition i computes slice slice[i]. Which partition computes which slice does not
        // change the result (every slice is written at its own place in D), so the slices go
        // where most of their bytes are local: a framework that places the halves of a shared
        // matrix on the other devices than expected still gets local accesses.
        // Only when the partitions are on different devices (DPX): on one device (SPX, meant for
        // checking results only) every order is equally local, so nothing is sampled and
        // partition i computes slice i.
        std::vector<size_t> slice(parts);
        for(size_t i = 0; i < parts; i++)
            slice[i] = i;
        const bool distinct = std::set<int>(s.devices.begin(), s.devices.end()).size() == parts;
        if(distinct)
        {
            std::vector<std::vector<double>> localBytes(parts, std::vector<double>(parts, 0.0));
            for(size_t i = 0; i < parts; i++)
                for(size_t j = 0; j < parts; j++)
                    for(auto const& mat : split)
                        localBytes[i][j] += localFraction(mat, i, offset[j], size[j], s.devices[i])
                                            * double(size[j]) * double(mat.other)
                                            * double(mat.bytes);
            auto localTotal = [&](std::vector<size_t> const& map) {
                double sum = 0;
                for(size_t i = 0; i < parts; i++)
                    sum += localBytes[i][map[i]];
                return sum;
            };
            // Greedy: each partition in turn takes the free slice with the most local bytes. It
            // replaces the plain order only when it is strictly better (for 2 partitions this
            // compares both orders).
            std::vector<size_t> greedy(parts);
            std::vector<bool>   taken(parts, false);
            for(size_t i = 0; i < parts; i++)
            {
                size_t best = parts;
                for(size_t j = 0; j < parts; j++)
                    if(!taken[j] && (best == parts || localBytes[i][j] > localBytes[i][best]))
                        best = j;
                greedy[i]   = best;
                taken[best] = true;
            }
            if(localTotal(greedy) > localTotal(slice))
                slice = greedy;
        }
        // A hint when a split matrix still mostly lives on another device than its partition.
        if(hints && distinct)
        for(size_t i = 0; i < parts; i++)
            for(auto const& mat : split)
            {
                const int64_t o = offset[slice[i]], z = size[slice[i]];
                if(localFraction(mat, i, o, z, s.devices[i]) < 0.5)
                    log_hints(__func__,
                              "PartitionedGemm: partition",
                              i,
                              "runs on device",
                              s.devices[i],
                              "but most of its slice of",
                              mat.name,
                              "is on device",
                              pageDevice(advance(
                                  mat.pointers[i], mat.columns ? o * mat.ld : o, mat.bytes)));
            }
        {
            std::vector<int64_t> o(parts), z(parts);
            for(size_t i = 0; i < parts; i++)
            {
                o[i] = offset[slice[i]];
                z[i] = size[slice[i]];
            }
            offset = std::move(o);
            size   = std::move(z);
        }

        std::vector<bool> publish(parts, false);
        for(size_t i = 0; i < parts; i++)
        {
            // Element offset of this partition's slice inside each (whole) matrix. The
            // matrices are column-major.
            int64_t elems[4] = {0, 0, 0, 0};
            if(dim == SplitDim::Batch)
            {
                for(int t = 0; t < 4; t++)
                    elems[t] = offset[i] * strides[t];
            }
            else if(dim == SplitDim::M)
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
            const void* const* lists[4] = {in.a.data(), in.b.data(), in.c.data(), dPointers.data()};
            for(int t = 0; t < 4; t++)
            {
                const void* p = lists[t][i];
                if(p == nullptr)
                    return HIPBLAS_STATUS_INVALID_VALUE;
                ptr[t] = advance(p, elems[t], bytes[t]);
            }

            // Partition i reads A, B and C and writes D (and the peer D) from device
            // s.devices[i]. Memory on another device must be reachable from it, which is the
            // caller's precondition: without peer access the kernel faults, as with any GEMM given
            // an inaccessible pointer. Nothing is rejected here; with hints on, a few addresses
            // are sampled to say so.
            if(hints)
            {
                const int64_t mi0 = dim == SplitDim::M ? size[i] : m;
                const int64_t ni0 = dim == SplitDim::N ? size[i] : n;
                const int64_t bi0 = dim == SplitDim::Batch ? size[i] : batch_count;
                const int64_t rows[4]
                    = {s.opA == HIPBLAS_OP_N ? mi0 : k, s.opB == HIPBLAS_OP_N ? k : ni0, mi0, mi0};
                const int64_t cols[4]
                    = {s.opA == HIPBLAS_OP_N ? k : mi0, s.opB == HIPBLAS_OP_N ? ni0 : k, ni0, ni0};
                const char* names[4] = {"A", "B", "C", "D"};
                const void* peer
                    = inputs.storePeerD ? advance(dPointers[1 - i], elems[3], bytes[3]) : nullptr;
                for(int t = 0; t < 5; t++)
                {
                    const void* base = t < 4 ? ptr[t] : peer;
                    if(base == nullptr)
                        continue;
                    const int     mat  = t < 4 ? t : 3;
                    const int64_t last = leading[mat] * (cols[mat] - 1) + rows[mat] - 1
                                         + (bi0 - 1) * std::abs(strides[mat]);
                    for(int64_t at : {int64_t(0), last / 2, last})
                    {
                        const int where = pageDevice(advance(base, at, bytes[mat]));
                        if(where >= 0 && !canReach(s.devices[i], where))
                        {
                            log_hints(__func__,
                                      "PartitionedGemm: partition",
                                      i,
                                      "will access",
                                      t < 4 ? names[mat] : "peer D",
                                      "on device",
                                      where,
                                      "from device",
                                      s.devices[i],
                                      "without peer access; the run will fault unless the caller"
                                      " enables it (or places a copy on the partition's device)");
                            break;
                        }
                    }
                }
            }

            GemmInputs slice;
            slice.setA(ptr[0]);
            slice.setB(ptr[1]);
            slice.setC(ptr[2]);
            slice.setD(ptr[3]);
            slice.setAlpha(inputs.alpha);
            slice.setBeta(inputs.beta);
            // The peer is the other partition's D, at this partition's slice offset.
            if(inputs.storePeerD)
                hipblaslt_ext::detail::GemmInputsAccess::setPeerD(
                    slice, const_cast<void*>(advance(dPointers[1 - i], elems[3], bytes[3])));

            GemmEpilogue    epilogue;
            GemmProblemType type(s.opA, s.opB, s.typeA, s.typeB, s.typeC, s.typeD, s.typeCompute);
            const int64_t   mi = dim == SplitDim::M ? size[i] : m;
            const int64_t   ni = dim == SplitDim::N ? size[i] : n;
            const int64_t   bi = dim == SplitDim::Batch ? size[i] : batch_count;
            auto            st = s.gemms[i]->setProblem(mi,
                                             ni,
                                             k,
                                             bi,
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

            // Whoever writes memory of another device publishes it: a D slice that is not on
            // this partition's device (a shared D), or the peer D of storePeerD.
            const int64_t last = leading[3] * (ni - 1) + mi - 1 + (bi - 1) * std::abs(strides[3]);
            auto const    foreign = [&](const void* base) {
                for(int64_t at : {int64_t(0), last})
                {
                    const int where = pageDevice(advance(base, at, bytes[3]));
                    if(where >= 0 && where != s.devices[i])
                        return true;
                }
                return false;
            };
            publish[i]
                = foreign(ptr[3])
                  || (inputs.storePeerD && foreign(advance(dPointers[1 - i], elems[3], bytes[3])));
        }

        // The word a publishing partition writes is allocated here and not in run(): a host
        // allocation is refused while a stream is being captured.
        for(size_t i = 0; i < parts; i++)
            if(publish[i] && s.publishWord(i, s.devices[i]) == nullptr)
                return HIPBLAS_STATUS_INTERNAL_ERROR;
        s.publishD = std::move(publish);
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
        if(!pimpl) // moved from
            return HIPBLAS_STATUS_INVALID_VALUE;
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
        if(!pimpl) // moved from
            return HIPBLAS_STATUS_INVALID_VALUE;
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
        if(!pimpl) // moved from
            return HIPBLAS_STATUS_INVALID_VALUE;
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
        if(!pimpl)
            return;
        for(auto& gemm : pimpl->gemms)
            gemm->setMaxWorkspaceBytes(workspaceBytes);
    }

    hipblasStatus_t PartitionedGemm::initialize(const hipblasLtMatmulAlgo_t&    algo,
                                                const std::vector<void*>&       workspaces,
                                                bool                            useUserArgs,
                                                const std::vector<hipStream_t>& streams)
    {
        if(!pimpl) // moved from
            return HIPBLAS_STATUS_INVALID_VALUE;
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
        if(!pimpl) // moved from
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto& s = *pimpl;
        if(!s.hasProblem || !s.sized(start, true) || !s.sized(stop, true))
            return HIPBLAS_STATUS_INVALID_VALUE;
        if(auto st = s.checkStreams(streams); st != HIPBLAS_STATUS_SUCCESS)
            return st;
        std::vector<int> devices(s.count());
        for(size_t i = 0; i < s.count(); i++)
        {
            if(hipStreamGetDevice(streams[i], &devices[i]) != hipSuccess)
                return HIPBLAS_STATUS_INVALID_VALUE;
            ScopedDevice guard(devices[i]);
            if(!guard.ok())
                return HIPBLAS_STATUS_INVALID_VALUE;
            auto st = s.gemms[i]->run(
                streams[i], start.empty() ? nullptr : start[i], stop.empty() ? nullptr : stop[i]);
            if(st != HIPBLAS_STATUS_SUCCESS)
                return st;
            // This partition wrote memory of another device: a stream write after the slice makes
            // those writes visible to the other devices.
            if(s.publishD[i])
            {
                // Allocated by setProblem; nothing is allocated here, so run() can be captured.
                uint64_t* word = i < s.publishWords.size() ? s.publishWords[i].second : nullptr;
                if(word == nullptr || hipStreamWriteValue64(streams[i], word, 1, 0) != hipSuccess)
                    return HIPBLAS_STATUS_INTERNAL_ERROR;
            }
        }

        // The partitions are not joined here: ordering work after all of them is the caller's
        // (an event on each partition's stream that the other streams wait for, or a host
        // synchronize), as with any stream-based launch.
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
} // namespace hipblaslt_experimental
