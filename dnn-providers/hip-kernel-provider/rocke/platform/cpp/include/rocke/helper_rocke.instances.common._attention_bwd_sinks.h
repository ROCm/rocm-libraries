/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_sinks.h -- C port of
 * library/kernels/common/_attention_bwd_sinks.py: the dQ sinks of the
 * attention backward tile step.
 *
 *   Python (_attention_bwd_sinks.py)    C (this header)
 *   ----------------------------------  ----------------------------------------
 *   DqSlot                              rocke_attn_bwd_dq_slot_t
 *   PendingDq (+ flat, names,           rocke_attn_bwd_pending_dq_t (+ _flat_len,
 *     from_flat)                          _flat, _name, _from_flat)
 *   DqSink + the four built sinks       rocke_attn_bwd_dq_sink_t, one struct with a
 *                                       kind tag (the Python class tree has four
 *                                       leaves; PackedAtomicWorkspaceSink derives
 *                                       from AtomicWorkspaceSink)
 *   AtomicWorkspaceSink(b, ws, seq=,    rocke_attn_bwd_dq_sink_init_atomic
 *     scale=)
 *   PackedAtomicWorkspaceSink           rocke_attn_bwd_dq_sink_init_packed_atomic
 *   NullSink()                          rocke_attn_bwd_dq_sink_init_null
 *   RegisterAccumulateSink(b, out,      rocke_attn_bwd_dq_sink_init_register
 *     scale=)
 *   SplitSliceSink(...)                 rocke_attn_bwd_dq_sink_init_split_slice
 *                                       (always fails: it raises in Python)
 *   sink.begin_head / consume / finish  rocke_attn_bwd_dq_sink_begin_head /
 *                                       _consume / _finish
 *   RegisterAccumulateSink.state /      rocke_attn_bwd_dq_sink_register_state_len /
 *     set_state                           _state / _set_state
 *   split_slice_workspace_bytes         rocke_attn_bwd_split_slice_workspace_bytes
 *
 * A sink copies the WorkspaceAddr / SeqInfo / StridedTensorAddr it is given
 * (all are immutable after init in Python). A register sink owns heap arrays
 * (its fragment layout and accumulators): release with
 * rocke_attn_bwd_dq_sink_free and do not copy it. Python's _key identity of a
 * row or column (the int itself, or id() of an SSA value) is the (host int,
 * or Value pointer) pair here.
 *
 * Python calls PackedAtomicWorkspaceSink.consume with a PackedRows as `rows`;
 * rocke_attn_bwd_dq_sink_consume takes both: `rows` (an i32 value, every
 * other sink) and `packed` (the PackedRows of the packed sink); the one the
 * sink kind does not use may be NULL.
 *
 * Errors (Python ValueError / RuntimeError / NotImplementedError) are recorded
 * on the builder's sticky error (ROCKE_ERR_VALUE, ROCKE_ERR_NOTIMPL for
 * NotImplementedError) and the entry returns false. Builder calls are made in
 * Python's order.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_SINKS_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_SINKS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* DqSlot: row offset inside the q step, head-dimension column, fp32 partial. */
typedef struct rocke_attn_bwd_dq_slot
{
    rocke_attn_bwd_iv_t row;
    rocke_attn_bwd_iv_t col;
    rocke_value_t* value;
} rocke_attn_bwd_dq_slot_t;

/* PendingDq (non-owning values). n_heads / rows_per_head NULL are None. */
typedef struct rocke_attn_bwd_pending_dq
{
    rocke_value_t* const* values;
    int n_values;
    rocke_value_t* q0;
    rocke_value_t* h;
    rocke_value_t* n_heads;
    rocke_value_t* rows_per_head;
} rocke_attn_bwd_pending_dq_t;

/* flat(): values, q0, h, then n_heads when present. */
int rocke_attn_bwd_pending_dq_flat_len(const rocke_attn_bwd_pending_dq_t* p);
void rocke_attn_bwd_pending_dq_flat(const rocke_attn_bwd_pending_dq_t* p, rocke_value_t** out);
/* names(prefix)[i]: "<prefix>dq<i>" for the values, then "<prefix>dqrow",
 * "<prefix>dqhead" and, when n_heads is present, "<prefix>dqnh". Writes the
 * i-th name (0 <= i < flat_len) into out; false when out_cap is too small. */
bool rocke_attn_bwd_pending_dq_name(
    const rocke_attn_bwd_pending_dq_t* p, const char* prefix, int i, char* out, size_t out_cap);
/* from_flat(vals): a partial of p's shape from loop-carried vals (n_vals must
 * be flat_len(p)); the result points into vals and keeps p's rows_per_head. */
bool rocke_attn_bwd_pending_dq_from_flat(rocke_ir_builder_t* b,
                                         const rocke_attn_bwd_pending_dq_t* p,
                                         rocke_value_t* const* vals,
                                         int n_vals,
                                         rocke_attn_bwd_pending_dq_t* out);

typedef enum rocke_attn_bwd_dq_sink_kind
{
    ROCKE_ATTN_BWD_SINK_ATOMIC = 0,
    ROCKE_ATTN_BWD_SINK_PACKED_ATOMIC = 1,
    ROCKE_ATTN_BWD_SINK_NULL = 2,
    ROCKE_ATTN_BWD_SINK_REGISTER = 3
} rocke_attn_bwd_dq_sink_kind_t;

typedef struct rocke_attn_bwd_dq_sink
{
    rocke_attn_bwd_dq_sink_kind_t kind;
    bool emits_g4; /* False when the tile step must not emit G4 (NullSink) */
    rocke_value_t* scale; /* NULL: None */

    /* atomic sinks */
    rocke_attn_bwd_workspace_addr_t ws;
    rocke_attn_bwd_seq_info_t seq;
    rocke_value_t* head; /* NULL: None */
    bool has_slab;
    rocke_attn_bwd_ws_slab_t slab;

    /* register sink */
    rocke_attn_bwd_strided_tensor_addr_t out;
    bool has_layout; /* Python _layout is not None */
    int n_slots;
    rocke_attn_bwd_dq_slot_t* slots; /* (row, col) of the first fragment; value unused */
    rocke_value_t** acc;
    int n_acc;
    rocke_value_t* q0; /* NULL: None */
} rocke_attn_bwd_dq_sink_t;

bool rocke_attn_bwd_dq_sink_init_atomic(rocke_attn_bwd_dq_sink_t* s,
                                        rocke_ir_builder_t* b,
                                        const rocke_attn_bwd_workspace_addr_t* ws,
                                        const rocke_attn_bwd_seq_info_t* seq,
                                        rocke_value_t* scale);
bool rocke_attn_bwd_dq_sink_init_packed_atomic(rocke_attn_bwd_dq_sink_t* s,
                                               rocke_ir_builder_t* b,
                                               const rocke_attn_bwd_workspace_addr_t* ws,
                                               const rocke_attn_bwd_seq_info_t* seq,
                                               rocke_value_t* scale);
void rocke_attn_bwd_dq_sink_init_null(rocke_attn_bwd_dq_sink_t* s);
bool rocke_attn_bwd_dq_sink_init_register(rocke_attn_bwd_dq_sink_t* s,
                                          rocke_ir_builder_t* b,
                                          const rocke_attn_bwd_strided_tensor_addr_t* out,
                                          rocke_value_t* scale);
/* SplitSliceSink: recorded, not built; constructing one raises. Always false. */
bool rocke_attn_bwd_dq_sink_init_split_slice(rocke_ir_builder_t* b);
void rocke_attn_bwd_dq_sink_free(rocke_attn_bwd_dq_sink_t* s);

bool rocke_attn_bwd_dq_sink_begin_head(rocke_attn_bwd_dq_sink_t* s,
                                       rocke_ir_builder_t* b,
                                       rocke_value_t* h);
/* consume(b, dq_frag, rows, h, clamp=). See the header note on rows / packed.
 * h may be NULL only for the packed sink (Python's h is unused there). */
bool rocke_attn_bwd_dq_sink_consume(rocke_attn_bwd_dq_sink_t* s,
                                    rocke_ir_builder_t* b,
                                    const rocke_attn_bwd_dq_slot_t* dq_frag,
                                    int n_frag,
                                    rocke_value_t* rows,
                                    const rocke_attn_bwd_packed_rows_t* packed,
                                    rocke_value_t* h,
                                    bool clamp);
bool rocke_attn_bwd_dq_sink_finish(rocke_attn_bwd_dq_sink_t* s, rocke_ir_builder_t* b);

/* RegisterAccumulateSink.state(): the accumulators (n = state_len). */
int rocke_attn_bwd_dq_sink_register_state_len(const rocke_attn_bwd_dq_sink_t* s);
void rocke_attn_bwd_dq_sink_register_state(const rocke_attn_bwd_dq_sink_t* s, rocke_value_t** out);
/* set_state(values): ValueError unless n == the accumulator count. */
bool rocke_attn_bwd_dq_sink_register_set_state(rocke_attn_bwd_dq_sink_t* s,
                                               rocke_ir_builder_t* b,
                                               rocke_value_t* const* values,
                                               int n);

/* split_slice_workspace_bytes: A(4 * n_split * B * s_q * h_q * D) rounded up
 * to 256 bytes. false (and a recorded ValueError) on a negative argument or
 * when the result does not fit int64 (Python integers are unbounded). */
bool rocke_attn_bwd_split_slice_workspace_bytes(rocke_ir_builder_t* b,
                                                int64_t n_split,
                                                int64_t batch,
                                                int64_t s_q,
                                                int64_t h_q,
                                                int64_t head_size,
                                                int64_t* out);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_SINKS_H */
