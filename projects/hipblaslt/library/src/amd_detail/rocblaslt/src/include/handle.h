/*! \file */
/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef HANDLE_H
#define HANDLE_H

#include "rocblaslt.h"
//#include "rocblaslt_ostream.hpp"
#include <atomic>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>

struct _rocblaslt_attribute
{
    _rocblaslt_attribute() {};

    ~_rocblaslt_attribute();

    void clear();

    const void* data();

    size_t length();

    size_t get(void* out, size_t size);

    template <typename T>
    size_t get(T* out)
    {
        return get(out, sizeof(T));
    }

    void set(const void* in, size_t size);

    template <typename T>
    void set(const T* in)
    {
        set(in, sizeof(T));
    }

private:
    void*  _data      = nullptr;
    size_t _data_size = 0;
};

/********************************************************************************
 * \brief rocblaslt_handle is a structure holding the rocblaslt library context.
 * It must be initialized using rocblaslt_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocblaslt_destroy_handle().
 *******************************************************************************/
struct _rocblaslt_handle
{
    // constructor
    _rocblaslt_handle();
    // destructor
    ~_rocblaslt_handle();

    // device id
    int device;
    // device properties
    hipDeviceProp_t properties;
    // device wavefront size
    int wavefront_size;
    // asic revision
    int asic_rev;

    // GSU (MBSK) reduction flags, and the amax reduction counter. Shared across
    // streams, indexed by problem, exactly as before.
    void* Synchronizer = nullptr;
    // Stream-K inter-workgroup flags. Private to a (stream, problem) pair; see
    // streamKFlagsForStream below.
    void* StreamKFlags = nullptr;
    // pointer mode ; default mode is host
    rocblaslt_pointer_mode pointer_mode = rocblaslt_pointer_mode_host;

    // Handle-level SM-count-target override. Set via hipblasLtSetSmCountTarget
    // (the analogue of cublasSetSmCountTarget). 0 (default) means "no
    // override; use all CUs the device exposes". Negative values are rejected
    // by the setter. A per-matmul descriptor or preference attribute, when
    // non-zero, takes precedence over this handle-level value.
    int32_t sm_count_target = 0;

    // Handle-level uniform-summation-order request. 0 off, 1 on; see hipblaslt.h.
    int32_t uniform_summation_order = 0;

#ifdef HIPBLASLT_USE_ROCROLLER
    void* rocroller_handle = nullptr;
    int   useRocRoller     = -1;
#endif

    // HIPBLASLT_CHECK_NUMERICS state. Read once in the ctor; opt-in via env.
    // See check_numerics_matrix.hpp for the scanner protocol.
    hipblaslt_check_numerics_mode check_numerics = hipblaslt_check_numerics_mode_no_check;

    // 4-byte device slot; scanner does atomicCAS(0, call_id) so the slot
    // holds the FIRST matmul's call_id whose D contained a NaN.
    uint32_t*             check_numerics_flag       = nullptr;
    // Host-mapped alias of the flag under STOP_ON_FIRST (null otherwise).
    uint32_t*             check_numerics_flag_host  = nullptr;
    // Per-matmul counter; 1-indexed so 0 stays the "no NaN" sentinel.
    std::atomic<uint32_t> check_numerics_call_id{0};
    uint32_t              check_numerics_scan_every = 1u;
    uint32_t              check_numerics_scan_from  = 1u;
    uint32_t              check_numerics_scan_until = ~uint32_t(0);
    bool                  check_numerics_stop_on_first = false;
    // Sticky bypass for scan_D once any caller observes a NaN.
    std::atomic<bool>     check_numerics_short_circuit{false};

    // Kernels treat these buffers as inter-workgroup flags that they set, spin
    // on, and reset themselves, so a flag region may only be touched by one
    // kernel at a time. Two GEMMs running concurrently on different streams
    // would otherwise clear a flag the other was still spinning on, leaving that
    // workgroup spinning forever with no error reported.
    //
    // There are two consumers and they want opposite things, so they get two
    // buffers rather than one compromise.
    //
    // GSU reduction (MBSK) needs synchronizerSizePerWG * numTiles * batch ints,
    // which runs into the tens of thousands on the shapes it is selected for.
    // Shrinking that would price MBSK candidates out of selection, so this
    // region keeps the size and the per-problem-only layout it has always had
    // and stays shared across streams. Solution selection is therefore
    // unchanged, and so is the cross-stream exposure of this path: no better
    // than today, but no worse.
    static constexpr size_t c_syncGsuSlotElements  = 409600;
    static constexpr size_t c_syncGsuSlotBytes     = c_syncGsuSlotElements * sizeof(int);
    static constexpr size_t c_syncGsuSlots         = 16;
    static constexpr size_t c_syncGsuTotalElements = c_syncGsuSlots * c_syncGsuSlotElements;

    // Stream-K indexes its flags by workgroup id (StreamK.py emits "flag offset
    // based on CTA index"), so a slot holds one int per Stream-K workgroup and
    // no more; skGrid is 224 on gfx950, which has 256 CUs. At that size every
    // stream can afford its own region, which is what closes the deadlock.
    static constexpr size_t c_syncSkSlotElements = 2048;
    static constexpr size_t c_syncSkSlotBytes    = c_syncSkSlotElements * sizeof(int);
    // Problem slots inside one stream's block: grouped GEMM does select Stream-K
    // solutions, so each problem in a group needs its own region.
    static constexpr size_t c_syncSkSlotsPerStream = 16;
    // Distinct streams one handle can serve. A block is claimed on a stream's
    // first Stream-K matmul and held until the handle is destroyed, so this
    // bounds distinct streams per handle rather than concurrent ones. PyTorch
    // draws streams from a fixed pool that tops out at 32 distinct hipStream_t
    // however many it is asked for, and a 32-way fan-out needs 33 (the streams
    // plus the default one it forks from); 64 is that working set doubled.
    static constexpr size_t c_syncSkStreamSlots = 64;
    // Allocated once at handle creation: allocating device memory mid-run would
    // break hipGraph capture, and a fixed layout keeps the lookup lock-free.
    // This table is 8 MiB, on top of the 25 MiB GSU one.
    static constexpr size_t c_syncSkTotalElements
        = c_syncSkStreamSlots * c_syncSkSlotsPerStream * c_syncSkSlotElements;

    // Hands back the Stream-K flag region reserved for (`stream`,
    // `problemIndex`) in `out`. Blocks are claimed lock-free; the scan is over
    // at most c_syncSkStreamSlots entries and hits on the first comparison once
    // a stream owns its block.
    //
    // Returns rocblaslt_status_internal_error once every block is taken, or for
    // a problem index past the block. Handing back a shared region instead would
    // silently reintroduce the cross-stream deadlock this separation exists to
    // prevent, and the caller would report success while a workgroup spins
    // forever.
    rocblaslt_status streamKFlagsForStream(hipStream_t stream, size_t problemIndex, void** out)
    {
        *out = nullptr;
        if(StreamKFlags == nullptr)
            return rocblaslt_status_success;

        if(problemIndex >= c_syncSkSlotsPerStream)
            return rocblaslt_status_internal_error;

        // hipStreamPerThread is a sentinel that resolves to a different stream
        // for every host thread, so every thread would present the same key and
        // share one block. Key those by thread instead. The legacy null stream
        // needs no such treatment: it really is one stream shared by all threads.
        const void* key = static_cast<const void*>(stream);
        if(stream == hipStreamPerThread)
        {
            static thread_local char perThreadKey;
            key = &perThreadKey;
        }

        for(size_t i = 0; i < c_syncSkStreamSlots; ++i)
        {
            const void* owner = m_skSlotOwner[i].load(std::memory_order_acquire);
            if(owner == nullptr)
            {
                const void* expected = nullptr;
                // Whoever wins the exchange owns the block; the loser reads back
                // the winner's key and falls through to the comparison below.
                owner = m_skSlotOwner[i].compare_exchange_strong(
                            expected, key, std::memory_order_acq_rel, std::memory_order_acquire)
                            ? key
                            : expected;
            }
            if(owner == key)
            {
                *out = static_cast<char*>(StreamKFlags)
                       + (i * c_syncSkSlotsPerStream + problemIndex) * c_syncSkSlotBytes;
                return rocblaslt_status_success;
            }
        }

        return rocblaslt_status_internal_error;
    }

    // Value-initialised so every block starts unowned.
    std::atomic<const void*> m_skSlotOwner[c_syncSkStreamSlots] = {};
};

/********************************************************************************
 * \brief rocblaslt_matrix_layout is a structure holding the rocblaslt matrix
 * content. It must be initialized using rocblaslt_matrix_layout_create()
 * and the retured handle must be passed
 * to all subsequent library function calls that involve the matrix.
 * It should be destroyed at the end using rocblaslt_matrix_layout_destroy().
 *******************************************************************************/
struct _rocblaslt_matrix_layout
{
    // constructor
    _rocblaslt_matrix_layout() = default;
    // destructor
    ~_rocblaslt_matrix_layout() = default;

    // num rows
    uint64_t m = 0;
    // num cols
    uint64_t n = 0;
    // leading dimension
    int64_t ld = 0;
    // data type of the matrix
    hipDataType      type;
    int32_t          batch_count  = 1;
    int64_t          batch_stride = 0;
    int64_t          batch_offset = 0;
    hipblasLtOrder_t order        = HIPBLASLT_ORDER_COL;
    // Batch Mode
    hipblasLtBatchMode_t batch_mode = HIPBLASLT_BATCH_MODE_STRIDED;    
};

/********************************************************************************
 * \brief rocblaslt_matmul_desc holds the description of the matrix
 *multiplication operation. It is initialized and destroyed with
 *rocblaslt_matmul_desc_create() and rocblaslt_matmul_desc_destroy() functions
 *respectively.
 *******************************************************************************/
struct _rocblaslt_matmul_desc
{
    // constructor
    _rocblaslt_matmul_desc() {};
    // destructor
    ~_rocblaslt_matmul_desc() {};

    // operation applied to the matrix A
    hipblasOperation_t op_A = HIPBLAS_OP_N;
    // operation applied to the matrix B
    hipblasOperation_t op_B = HIPBLAS_OP_N;
    // epilogue operation
    rocblaslt_epilogue epilogue = ROCBLASLT_EPILOGUE_DEFAULT;
    // alpha,beta pointer mode
    rocblaslt_pointer_mode pointermode = rocblaslt_pointer_mode_host;
    // bias vector pointer
    void*       bias      = nullptr;
    void*       scaleA    = nullptr;
    void*       scaleB    = nullptr;
    void*       scaleC    = nullptr;
    void*       scaleD    = nullptr;
    void*       scaleE    = nullptr;
    void*       amaxD     = nullptr;
    hipDataType bias_type = HIPBLASLT_DATATYPE_INVALID;
    // E
    void*       e        = nullptr;
    hipDataType aux_type = HIPBLASLT_DATATYPE_INVALID;
    int64_t     lde      = 0;
    int64_t     stride_e = 0;
    //
    rocblaslt_compute_type compute_type;
    rocblaslt_compute_type compute_type_original;
    hipDataType            compute_input_typeA = HIPBLASLT_DATATYPE_INVALID;
    hipDataType            compute_input_typeB = HIPBLASLT_DATATYPE_INVALID;
    hipDataType            scale_type          = HIPBLASLT_DATATYPE_INVALID;

    RocblasltContractionProblem::ScalingFormat scaleAType
        = RocblasltContractionProblem::ScalingFormat::None;
    RocblasltContractionProblem::ScalingFormat scaleBType
        = RocblasltContractionProblem::ScalingFormat::None;

    float act0 = 0.f;
    float act1 = 0.f;

    // User-supplied target compute-unit count for kernel selection /
    // persistent-grid sizing. 0 (default) means "no constraint; use all
    // CUs the device exposes". Negative values are rejected by the setter.
    int32_t sm_count_target = 0;

    // StreamK tile scheduling mode (hybrid SK3/SK4 tile scheduling for SK5).
    // Exposed via the _EXT attribute namespace (no equivalent in the base C API).
    // Tri-state: 0 = OFF (default; static SK3 unless sm_count_target > 0),
    // 1 = ON (force SK4 dynamic), 2 = AUTO (heuristic picks per launch).
    int32_t streamk_tile_scheduling_ext = 0;

    // Uniform summation order. 0 inherits the handle request; 1 enables.
    // See hipblaslt.h for the guarantee.
    int32_t uniform_summation_order = 0;

    // Added this new bias_stride parameter to capture the stride in bias vector to get unique bias vector for each batch in strided batch case. 
    // Default value is 0 which means same bias vector will be used across all batches (broadcast).
    int32_t bias_stride = 0;

    std::shared_ptr<void> m_data; // Tensile data

    void copy(const _rocblaslt_matmul_desc& src)
    {
        this->op_A                    = src.op_A;
        this->op_B                    = src.op_B;
        this->epilogue                = src.epilogue;
        this->bias                    = src.bias;
        this->scaleA                  = src.scaleA;
        this->scaleB                  = src.scaleB;
        this->scaleC                  = src.scaleC;
        this->scaleD                  = src.scaleD;
        this->scaleE                  = src.scaleE;
        this->scaleAType              = src.scaleAType;
        this->scaleBType              = src.scaleBType;
        this->pointermode             = src.pointermode;
        this->amaxD                   = src.amaxD;
        this->bias_type               = src.bias_type;
        this->e                       = src.e;
        this->aux_type                = src.aux_type;
        this->lde                     = src.lde;
        this->stride_e                = src.stride_e;
        this->compute_type            = src.compute_type;
        this->compute_type_original   = src.compute_type_original;
        this->compute_input_typeA     = src.compute_input_typeA;
        this->compute_input_typeB     = src.compute_input_typeB;
        this->scale_type              = src.scale_type;
        this->act0                    = src.act0;
        this->act1                    = src.act1;
        this->sm_count_target         = src.sm_count_target;
        this->streamk_tile_scheduling_ext = src.streamk_tile_scheduling_ext;
        this->uniform_summation_order = src.uniform_summation_order;
        this->bias_stride             = src.bias_stride;
    }
};

/********************************************************************************
 * \brief rocblaslt_matmul_preference holds the description of the matrix
 * multiplication preference.
 * It is initialized and destroyed with rocblaslt_matmul_preference_create()
 * and rocblaslt_matmul_preference_destroy() functions respectively.
 *******************************************************************************/
struct _rocblaslt_matmul_preference
{
    // constructor
    _rocblaslt_matmul_preference() {};
    // destructor
    ~_rocblaslt_matmul_preference() {};
    //
    uint32_t search_mode         = 0;
    uint64_t max_workspace_bytes = 0;

    // User-supplied target compute-unit count used as a heuristic-selection
    // hint. 0 (default) means "no constraint; use all CUs the device exposes".
    // Negative values are rejected by the setter.
    int32_t sm_count_target = 0;

    int64_t alg_config_id     = 0;
    int64_t alg_max_id        = 0;
    int64_t search_iterations = 0;
};

// Resolve the effective sm_count_target hint for a matmul launch using
// the documented precedence (per-matmul preference > per-matmul desc >
// handle). Returns 0 when nothing has been set, meaning "use all CUs
// the device exposes".
inline int32_t effective_sm_count_target(const _rocblaslt_handle*            handle,
                                         const _rocblaslt_matmul_desc*       desc,
                                         const _rocblaslt_matmul_preference* pref)
{
    if(pref && pref->sm_count_target > 0)
        return pref->sm_count_target;
    if(desc && desc->sm_count_target > 0)
        return desc->sm_count_target;
    if(handle)
        return handle->sm_count_target;
    return 0;
}

// Resolve the effective uniform-summation-order request for a matmul launch
// using first-on-wins precedence (C++ preference true, then desc==1, then
// handle==1). 0 on desc/pref means inherit. There is no per-GEMM opt-out
// when the handle is on.
inline int32_t effective_uniform_summation_order(const _rocblaslt_handle*      handle,
                                                 const _rocblaslt_matmul_desc* desc,
                                                 bool                          pref_uso = false)
{
    if(pref_uso)
        return 1;
    if(desc && desc->uniform_summation_order)
        return 1;
    if(handle && handle->uniform_summation_order)
        return 1;
    return 0;
}

#endif // HANDLE_H
