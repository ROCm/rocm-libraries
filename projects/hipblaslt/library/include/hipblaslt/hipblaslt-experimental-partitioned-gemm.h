// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief Experimental feature PARTITIONED_GEMM: one GEMM run as slices on the partitions of a
 *  partitioned GPU (for example DPX), one stream per partition. C++ only.
 *
 *  Contract tests: clients/tests/src/experimental/contract_partitioned_gemm_gtest.cpp.
 */

#pragma once

#ifndef HIPBLASLT_EXPERIMENTAL_ENUM_BIT
#error "include <hipblaslt/hipblaslt-experimental.h> instead"
#endif

#ifdef __cplusplus

/*! \ingroup experimental_module
 *  \brief Revision of experimental feature PARTITIONED_GEMM declared in this header.
 *
 *  Incremented on every breaking change to the feature: a removed declaration, a function or
 *  type replaced by a renamed version, a retired enumerator, or a changed contract test.
 *  Additions do not increment it. tools/scripts/check_experimental_api.py enforces this.
 */
#define HIPBLASLT_EXPERIMENTAL_PARTITIONED_GEMM 1

#include "hipblaslt/hipblaslt-ext.hpp"

#include <memory>
#include <vector>

namespace hipblaslt_experimental
{
    /*! \ingroup experimental_module
     *  \brief The dimension along which a PartitionedGemm splits the problem.
     *
     *  \details With ``M``, every partition computes ``m_i x n x k`` and the first operand
     * (A) is split. With ``N``, every partition computes ``m x n_i x k`` and the second
     * operand (B) is split. With ``Batch``, every partition computes whole ``m x n x k``
     * matrices for a range of the batches, and every matrix with a nonzero batch stride is
     * split. The first ``N - 1`` partitions take ``size / N`` rounded up so that every
     * slice starts at a multiple of 256 bytes in every matrix it cuts, and the last partition
     * takes the rest, when that changes the balance by at most 1%. Otherwise they take
     * ``size / N`` and a hint is logged (``HIPBLASLT_LOG_MASK=4``) naming what to pad. The
     * split is never refused because of alignment. Pad the leading dimensions (and batch
     * strides) to multiples of 256 bytes: that is enough for ``N`` and ``Batch``; for ``M``
     * also make ``M`` a multiple of ``parts * 256 / element size``.
     *
     *  Experimental enumerator values are never reused (see
     * HIPBLASLT_EXPERIMENTAL_ENUM_HIGH_WATER).
     */
    enum class SplitDim
    {
        M     = 0, //!< Split rows of op(A), C and D.
        N     = 1, //!< Split columns of op(B), C and D.
        Batch = 2, //!< Split the batch: whole matrices, a range of batches per partition.
    };

    /*! \ingroup experimental_module
     *  \brief Runs one GEMM as slices on several devices.
     *
     *  \details A device that is partitioned (for example DPX) shows up as several HIP
     * devices. A PartitionedGemm takes the full problem, gives each partition a slice of it
     * (see \ref hipblaslt_experimental::SplitDim), and launches the slices on one stream per
     * partition. It owns one hipblaslt_ext::Gemm per partition and does not allocate memory or
     * copy data.
     *
     *  Handle, stream, workspace, pointer and event number ``i`` all belong to partition
     * ``i``. Each handle must be created with its partition's device current, and each stream
     * must belong to the same device as its handle. In SPX mode, where every partition is on
     * the same device, the streams share that device and the result is still correct, but
     * SPX is only meant for checking results without a DPX GPU, not for speed: the
     * partitions just run one after another or concurrently on one device.
     *
     *  Calls follow the order of hipblaslt_ext::Gemm: setProblem, algoGetHeuristic (or
     * #isAlgoSupported), #initialize, then #run. Call setProblem again when a pointer or the
     * shape changes, not on every launch: it queries where the memory is
     * (``hipPointerGetAttributes``) to decide which partition computes which slice.
     *
     *  The current implementation supports plain GEMM: no epilogue, no scale or aux inputs, and
     * no MX types. Device user arguments are declared but not supported yet.
     */
    class PartitionedGemm
    {
    public:
        /*! \ingroup experimental_module
        *  \brief Constructor.
        *
        *  @param[in]
        *  handles                    One hipBLASLt handle per partition. The number of
        * partitions is ``handles.size()``.
        *  @param[in]
        *  opA,opB                    The transpose type of matrices A and B.
        *  @param[in]
        *  typeA,typeB,typeC,typeD    The data type of matrices A, B, C, and D.
        *  @param[in]
        *  typeCompute                The compute type of the GEMM problem.
        */
        HIPBLASLT_EXPORT explicit PartitionedGemm(const std::vector<hipblasLtHandle_t>& handles,
                                                  hipblasOperation_t                    opA,
                                                  hipblasOperation_t                    opB,
                                                  hipDataType                           typeA,
                                                  hipDataType                           typeB,
                                                  hipDataType                           typeC,
                                                  hipDataType                           typeD,
                                                  hipblasComputeType_t typeCompute);
        HIPBLASLT_EXPORT ~PartitionedGemm();

        PartitionedGemm(const PartitionedGemm&)                             = delete;
        PartitionedGemm&                  operator=(const PartitionedGemm&) = delete;
        HIPBLASLT_EXPORT                  PartitionedGemm(PartitionedGemm&&) noexcept;
        HIPBLASLT_EXPORT PartitionedGemm& operator=(PartitionedGemm&&) noexcept;

        //! The number of partitions.
        HIPBLASLT_EXPORT size_t getPartitionCount() const;

        /*! \ingroup experimental_module
        *  \brief Sets up the full GEMM problem and its partitioning.
        *
        *  \details ``m``, ``n``, ``k``, the leading dimensions and the strides describe the
        * full matrices. The pointers come from ``inputs``.
        *
        *  Which partition computes which slice is chosen here: the slices of the split
        * matrices (op(A) for SplitDim::M, op(B) for SplitDim::N, every batched matrix for
        * SplitDim::Batch, and C and D) go to the
        * partitions whose devices hold most of their pages, and otherwise partition ``i``
        * computes slice ``i``. The result is the same either way. A slice that still mostly
        * lives on another device is reported in the hints log (``HIPBLASLT_LOG_MASK=4``).
        *
        *  @param[in]
        *  m,n,k                            The size of the full problem.
        *  @param[in]
        *  batch_count                      The batch count.
        *  @param[in]
        *  lda,ldb,ldc,ldd                  The leading dimensions of the matrices.
        *  @param[in]
        *  strideA,strideB,strideC,strideD  The batch strides of the matrices.
        *  @param[in]
        *  dim                              The dimension to split.
        *  @param[in]
        *  inputs                           One GemmInputs per partition (see below).
        *  @param[in]
        *  storePeerD                       Every partition also stores its slice into the D of
        * the other partition (see below).
        *
        *  \details ``inputs`` holds one GemmInputs per partition, N in total, with ``setA``,
        * ``setB``, ``setC``, ``setD``, ``setAlpha`` and ``setBeta`` set; any other pointer (bias,
        * scales, aux, amax) must stay null, because a PartitionedGemm runs a plain GEMM. Each
        * pointer points at a whole matrix (the library adds the partition's slice offset), and
        * the leading dimensions and batch strides describe that whole matrix. The same pointer
        * in every element shares one matrix between the partitions (it must be accessible from
        * every partition device, see below); different pointers give each partition its own copy, which
        * the caller keeps equal. For D, the same pointer means every partition writes its slice
        * into one D. Copies are matched to the partitions by device, whatever the order of
        * ``inputs``: partition ``p`` uses the copy on its device, so the D on device X receives
        * the slice of the partition on device X. All copied matrices must agree on one
        * assignment; if they do not (for example A listed in one order and B in the other),
        * setProblem returns ``HIPBLAS_STATUS_INVALID_VALUE`` and logs an error. hipBLASLt
        * allocates and copies nothing. ``alpha`` and ``beta`` are host pointers, as in
        * GemmInputs; they are read from the first element, must be set in every element, and
        * apply to every partition.
        *
        *  Precondition: every partition's slices of A, B, C and D (and with ``storePeerD`` the
        * peer D) must be accessible from that partition's device: local memory, or peer access
        * or VMM access that the caller has set up. Otherwise the run faults, as with any GEMM
        * given an inaccessible pointer; setProblem does not check it and does not enable peer
        * access. With ``HIPBLASLT_LOG_MASK=4`` (hints) it samples a few addresses and logs a
        * hint when one looks unreachable (``hipDeviceCanAccessPeer`` says only that peer
        * access is possible, not that it is enabled), but never changes its return value.
        *
        *  Where the memory is (hipMalloc on one device, or a VMM range whose pages are on
        * several devices) is up to the caller: for example, two copies of A back to back in one
        * VMM range are passed as ``va`` and ``va + sizeA``, and one D whose halves are on the
        * two devices as ``va`` in both elements. The library reads where the pages of each slice
        * are and gives every partition the slice that is on its device (see
        * PartitionedGemm::getSlice).
        *
        *  With ``HIPBLASLT_LOG_MASK=4`` setProblem logs where the partitions are: all on one
        * device (SPX mode, correct but only for checking results, not for speed), the partitions of one GPU (DPX or
        * another partition mode, found from the PCI function of the devices), or different GPUs.
        *
        *  With ``storePeerD`` every partition also stores its slice into the D of the other
        * partition, so that both Ds are complete once every partition has finished its slice
        * (see run()). It needs exactly 2 partitions and a replicated D (two different
        * pointers, in any order: the peer of each partition is found by device), and the backend
        * needs kernels that store a second D; without them setProblem succeeds and
        * algoGetHeuristic returns nothing (also with a device library built before this
        * feature). The peer D is written with system-scope stores. run() does not order
        * later work after the partitions: see run().
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the problem was set on every partition.
        *  \retval HIPBLAS_STATUS_INVALID_VALUE     If `inputs` does not hold one GemmInputs
        * per partition, a required pointer is null, a size is not positive, or the split dimension is smaller than the
        * number of partitions.
        *  \retval HIPBLAS_STATUS_NOT_SUPPORTED     If a partition's device is not a gfx1250
        * (the only architecture PartitionedGemm supports), the types or operations are not
        * supported (conjugate transpose, sub-byte types), or a partition rejects its slice.
        */
        HIPBLASLT_EXPORT hipblasStatus_t
            setProblem(int64_t                                       m,
                       int64_t                                       n,
                       int64_t                                       k,
                       int64_t                                       batch_count,
                       int64_t                                       lda,
                       int64_t                                       ldb,
                       int64_t                                       ldc,
                       int64_t                                       ldd,
                       int64_t                                       strideA,
                       int64_t                                       strideB,
                       int64_t                                       strideC,
                       int64_t                                       strideD,
                       SplitDim                                      dim,
                       const std::vector<hipblaslt_ext::GemmInputs>& inputs,
                       bool                                          storePeerD = false);

        /*! \ingroup experimental_module
        *  \brief The slice of partition ``partition`` along the split dimension.
        *
        *  \details It need not be slice number ``partition``: setProblem may give a partition
        * the slice whose memory is on its device.
        *
        *  @param[in]
        *  partition        The partition index.
        *  @param[out]
        *  offset,size      The first index and the number of indices of the slice.
        *
        *  \retval HIPBLAS_STATUS_INVALID_VALUE If the index is out of range or no problem is set.
        */
        HIPBLASLT_EXPORT hipblasStatus_t getSlice(size_t   partition,
                                                  int64_t& offset,
                                                  int64_t& size) const;

        /*! \ingroup experimental_module
        *  \brief Retrieve the algorithms that every partition accepts.
        *
        *  \details The heuristic runs on partition 0. Algorithms that any partition rejects
        * for its slice are dropped. The workspace size of a result is the largest over the
        * partitions.
        */
        HIPBLASLT_EXPORT hipblasStatus_t
            algoGetHeuristic(const int                                      requestedAlgoCount,
                             const hipblaslt_ext::GemmPreference&           pref,
                             std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

        /*! \ingroup experimental_module
        *  \brief Check whether the algorithm supports every partition's slice.
        *
        *  @param[in]
        *  algo                  The algorithm heuristic.
        *  @param[out]
        *  workspaceSizeInBytes  The required workspace size of each partition.
        *
        *  \retval HIPBLAS_STATUS_INVALID_VALUE The algorithm does not support some slice.
        */
        HIPBLASLT_EXPORT hipblasStatus_t isAlgoSupported(hipblasLtMatmulAlgo_t& algo,
                                                         std::vector<size_t>& workspaceSizeInBytes);

        //! Sets the maximum workspace size of every partition.
        HIPBLASLT_EXPORT void setMaxWorkspaceBytes(size_t workspaceBytes);

        /*! \ingroup experimental_module
        *  \brief Creates the kernel arguments of every partition.
        *
        *  @param[in]
        *  algo          The algorithm to use.
        *  @param[in]
        *  workspaces    One workspace pointer per partition (16B aligned).
        *  @param[in]
        *  useUserArgs   Like hipblaslt_ext::Gemm::initialize; it does not affect a standard GEMM.
        *  @param[in]
        *  streams       One stream per partition.
        *
        *  \retval HIPBLAS_STATUS_INVALID_VALUE If a vector size is not the partition count, or
        * a stream is not on the device of its partition's handle.
        */
        HIPBLASLT_EXPORT hipblasStatus_t initialize(const hipblasLtMatmulAlgo_t&    algo,
                                                    const std::vector<void*>&       workspaces,
                                                    bool                            useUserArgs,
                                                    const std::vector<hipStream_t>& streams);

        /*! \ingroup experimental_module
        *  \brief Launches every partition's slice on its stream and returns without waiting.
        *
        *  \details Each partition's slice is launched on its own stream, as Gemm::run does for
        * one stream. The partitions are not joined: work that needs the whole result (for
        * example a kernel that reads all of D) must be ordered after every partition by the
        * caller. Record an event on each partition's stream after this call and make the
        * streams that need the result wait for it (hipStreamWaitEvent), or synchronize the
        * streams from the host. The sample does the first. The same holds under stream
        * capture, where only the kernels of each partition are recorded.
        *
        *  A partition whose D slice is memory of another device (a shared D), or that also
        * stores the peer D (``storePeerD``), publishes its writes on its stream after the
        * slice, so that the other devices see them. A partition that writes only its own
        * device's memory does not. ``setProblem`` decides this from the device that owns the
        * first and the last element of the partition's D slice (and of its peer D). A D slice
        * whose first and last element are on the partition's own device but which crosses
        * another device in between (possible with a VMM mapping) is not published: making
        * those writes visible is then the caller's. Shared A, B and C are the caller's too:
        * whoever wrote them last must have made them visible to every partition's device
        * before this call.
        *
        *  @param[in]
        *  streams       One stream per partition.
        *  @param[in]
        *  start,stop    Optional events, one per partition (or empty).
        */
        HIPBLASLT_EXPORT hipblasStatus_t run(const std::vector<hipStream_t>& streams,
                                             const std::vector<hipEvent_t>&  start = {},
                                             const std::vector<hipEvent_t>&  stop  = {});

        /*! \ingroup experimental_module
        *  \brief Reserved. Runs with device user arguments, one buffer per partition.
        *
        *  \retval HIPBLAS_STATUS_NOT_SUPPORTED If any pointer in ``deviceUserArgs`` is not
        * null. The call is declared now so that adding it later changes no signature.
        */
        HIPBLASLT_EXPORT hipblasStatus_t run(const std::vector<void*>&       deviceUserArgs,
                                             const std::vector<hipStream_t>& streams,
                                             const std::vector<hipEvent_t>&  start = {},
                                             const std::vector<hipEvent_t>&  stop  = {});

        /*! \ingroup experimental_module
        *  \brief Reserved. Fills one host ``UserArguments`` buffer per partition.
        *
        *  \retval HIPBLAS_STATUS_NOT_SUPPORTED Always, in this version.
        */
        HIPBLASLT_EXPORT hipblasStatus_t
            getDefaultValueForDeviceUserArguments(const std::vector<void*>& hostDeviceUserArgs);

    private:
        class PartitionedGemmImpl;
        std::unique_ptr<PartitionedGemmImpl> pimpl;
    };
} // namespace hipblaslt_experimental

#endif // __cplusplus
