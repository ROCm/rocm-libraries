/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_fused_moe_internal.h -- PRIVATE shared state + phase-function
 * contract for the C99 port of the five MoE-specific kernel builders in
 * rocke/instances/common/fused_moe.py:
 *   build_moe_gather, build_moe_silu_mul, build_moe_silu_mul_packed,
 *   build_moe_static_scatter_gather, build_moe_topk_weighted_reduce.
 *
 * WHY THIS HEADER EXISTS.
 *   Each Python builder is a self-contained function whose prologue derives one
 *   block of enclosing locals (geometry scalars, param Values, SSA constants,
 *   thread/grid decode) that the chunk-loop body then reads on every iteration.
 *   The bodies are written as inline loops rather than named nested closures,
 *   BUT the module also has three genuinely-shared module-level helpers that
 *   every builder calls and that must be ported once and reused:
 *     _effective_vec(spec_vec, block_size, n)  -> int   (pure)
 *     _chunk_distribution(block_size, vec)     -> TileDistribution (pure)
 *     _silu_mul_f32(b, g, u, one_f32, c_neg_log2e) -> f32 Value  (emits SSA)
 *
 *   In C there is no closure capture. The faithful port turns each builder body
 *   into a small family of free functions taking a POINTER to one shared context
 *   struct that holds EXACTLY the locals the prologue computes and the body
 *   reads. The driver populates the ctx in the same order the Python prologue
 *   computes its locals, then calls the phase functions in Python order.
 *
 *   Two ctx flavors cover the five builders:
 *     - rocke_moe_stream_ctx_t : the four "stream a row through interleaved/
 *       block-partitioned vec chunks" builders -- gather, silu_mul,
 *       silu_mul_packed, topk_reduce. They share a near-identical prologue
 *       (one CTA per bucket, BS/EPT/VEC consts, bid/tid, row bases) and an
 *       inner chunk loop. The ctx is the union of fields any of the four need;
 *       a `kind` discriminant + the param-Value set in use select the variant.
 *     - rocke_moe_ssg_ctx_t : build_moe_static_scatter_gather, which has the
 *       extra atomic-slot-claim + LDS-broadcast prologue and an 11-param ABI;
 *       it gets its own ctx so the stream ctx stays clean.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing .c TU binds
 *   to. It is DESIGNED TO BE COMPLETE: every local the Python body shares across
 *   phases is a field here. A body agent implementing a phase MUST be able to
 *   read/write only ctx fields and call the prototypes below WITHOUT editing
 *   this header. If a phase genuinely needs a value not present, that is a design
 *   bug to fix here once, deliberately.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `bucket_base`
 *   -> `ctx->bucket_base`; Python `c_neg_log2e` -> `ctx->c_neg_log2e`; Python
 *   ALL-CAPS geometry `H`/`BS`/`EPT`/`VEC` -> ctx->H/BS/EPT/VEC). Phase
 *   functions carry a `rocke_moe_` prefix and the builder tag.
 *
 * PORTING NOTE (silu_mul tile path).
 *   build_moe_silu_mul / _packed use the CK-Tile distribution surface
 *   tile.load(b, distribution=, ps=) / make_static_distributed_tensor /
 *   distribution.iterate_ys() / tile.store(b, out_dt, ps=). The ported
 *   tensor_view header currently exposes load_vec/store_vec but not the
 *   distribution-aware tile load/store; the tile-path phase functions below
 *   depend on that surface. Implementers should drive it through the
 *   distribution + tile_window structs already in rocke/helper_*.h (the missing
 *   distribution-aware tile.load/store is a one-time helper addition, flagged
 *   here so a body agent does not rediscover it).
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. Included only by the
 * instance_fused_moe*.c translation units. Public callers use
 * rocke/instance_fused_moe.h.
 */
#ifndef ROCKE_INSTANCE_FUSED_MOE_INTERNAL_H
#define ROCKE_INSTANCE_FUSED_MOE_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.distribution.h" /* rocke_tile_distribution_t */
#include "rocke/helper_rocke.helpers.tensor_view.h" /* rocke_tensor_view_t      */
#include "rocke/instance_fused_moe.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Max chunks = EPT/VEC the streaming bodies unroll. EPT = n/block_size and
 * VEC>=1, so chunks <= n/block_size; with block_size>=64 and the covered shapes
 * (n up to 32768) chunks stays small. 512 is generous headroom (e.g.
 * I=32768,BS=1024,VEC=1 -> 32 chunks; I=32768,BS=64,VEC=8 -> 64 chunks). */
#define ROCKE_MOE_MAX_CHUNKS 512

/* Which of the four streaming builders a rocke_moe_stream_ctx_t drives. Selects
 * the param set, the address-layout (interleaved vs block-partitioned), and the
 * per-chunk op chain. */
typedef enum rocke_moe_stream_kind
{
    ROCKE_MOE_STREAM_GATHER = 0, /* build_moe_gather                       */
    ROCKE_MOE_STREAM_SILU_MUL, /* build_moe_silu_mul                     */
    ROCKE_MOE_STREAM_SILU_MUL_PACKED, /* build_moe_silu_mul_packed              */
    ROCKE_MOE_STREAM_REDUCE /* build_moe_topk_weighted_reduce         */
} rocke_moe_stream_kind_t;

/* ===================================================================== *
 *  rocke_moe_stream_ctx_t  --  shared state for the four streaming builders.
 *
 *  Field order follows the common Python prologue top-to-bottom. Not every
 *  field is live for every `kind`; comments tag the owning builder(s). Fields
 *  unused by a given kind are left NULL/0 by the prologue.
 * ===================================================================== */
typedef struct rocke_moe_stream_ctx
{
    /* ---- inputs / resolved environment -- */
    rocke_ir_builder_t* b; /* the IRBuilder `b`                 */
    const rocke_fused_moe_spec_t* spec; /* the FusedMoeSpec                  */
    const char* arch; /* NULL-normalised "gfx950"          */
    rocke_moe_stream_kind_t kind; /* which builder this ctx drives     */

    /* ---- geometry scalars (Python ALL-CAPS prologue locals) -- *
     * N is the streamed axis: hidden for gather/reduce, intermediate for the
     * two silu_mul builders. EPT/VEC are derived off N accordingly. */
    int N; /* H (gather/reduce) or I_DIM (silu_mul[/_packed])         */
    int BS; /* spec.block_size                                          */
    int EPT; /* elems_per_thread_hidden or _inter (= N / BS)             */
    int VEC; /* _effective_vec(spec.vec, BS, N)                          */
    int chunks; /* EPT // VEC  (loop trip count; <= ROCKE_MOE_MAX_CHUNKS)     */
    const char* dtype; /* spec.dtype                                          */
    const rocke_type_t* ty; /* io_ir_type(dtype)                               */

    /* ---- kernel params (Values). The live subset depends on kind. -- */
    /* gather:    X, SortedTokenIds, GroupedInput, p_tokens, p_hidden        */
    /* silu_mul:  GateOut, UpOut, Hidden, p_total_pairs, p_inter             */
    /* packed:    GateUp, Hidden, p_total_pairs, p_inter                     */
    /* reduce:    DownOut, SortedTokenIds, SortedWeights, Y, p_total_pairs,  */
    /*            p_hidden, p_tokens                                          */
    rocke_value_t* X; /* gather                                    */
    rocke_value_t* GateOut; /* silu_mul                                  */
    rocke_value_t* UpOut; /* silu_mul                                  */
    rocke_value_t* GateUp; /* silu_mul_packed                           */
    rocke_value_t* DownOut; /* reduce                                    */
    rocke_value_t* SortedTokenIds; /* gather, reduce                            */
    rocke_value_t* SortedWeights; /* reduce                                    */
    rocke_value_t* GroupedInput; /* gather                                    */
    rocke_value_t* Hidden; /* silu_mul, silu_mul_packed                 */
    rocke_value_t* Y; /* reduce (f32 accumulator)                  */
    rocke_value_t* p_tokens; /* gather, reduce (ABI scalar, unused body)  */
    rocke_value_t* p_hidden; /* gather, reduce (ABI scalar, unused body)  */
    rocke_value_t* p_total_pairs; /* silu_mul[/_packed], reduce (ABI scalar)   */
    rocke_value_t* p_inter; /* silu_mul[/_packed] (ABI scalar, unused)   */

    /* ---- thread / grid decode (SSA) -- */
    rocke_value_t* bid; /* block_id_x()                              */
    rocke_value_t* tid; /* thread_id_x()                             */
    rocke_value_t* c_vec; /* const_i32(VEC)                            */

    /* ---- per-CTA row bases + indirect loads (SSA) -- *
     * gather/reduce: token_id (to_sgpr_u32 of load_sorted_token_id), the
     *   valid_token test, bucket_base = bid*N, and the source/dest row base
     *   (src_row_base = token_id*H for gather; y_row_base = token_id*H for
     *   reduce). reduce also pins `weight`.
     * silu_mul:        row_base = bid*N.
     * silu_mul_packed: gate_base = bid*2I, up_base = gate_base+I, out_base=bid*I.
     */
    rocke_value_t* token_id; /* gather, reduce                            */
    rocke_value_t* valid_token; /* gather, reduce (cmp_ge token_id, 0)       */
    rocke_value_t* weight; /* reduce (load_sorted_topk_weight)          */
    rocke_value_t* bucket_base; /* gather (bid*H), reduce (bid*H)            */
    rocke_value_t* src_row_base; /* gather (token_id*H)                       */
    rocke_value_t* y_row_base; /* reduce (token_id*H)                       */
    rocke_value_t* lane_chunk_base; /* reduce: tid*EPT (block-partitioned base)  */
    rocke_value_t* row_base; /* silu_mul (bid*I)                          */
    rocke_value_t* gate_base; /* packed (bid*2I)                           */
    rocke_value_t* up_base; /* packed (gate_base + I)                    */
    rocke_value_t* out_base; /* packed (bid*I)                            */

    /* ---- silu_mul f32 constants (SSA) -- */
    rocke_value_t* c_neg_log2e; /* const_f32(-1.4426950408889634)            */
    rocke_value_t* one_f32; /* const_f32(1.0)                            */

    /* ---- silu_mul CK-Tile distribution tile path (VEC>1) -- *
     * Built once in the silu_mul prologue; reused per chunk. chunk_elems =
     * BS*VEC. The views wrap the param ptrs at a (chunk_elems,) shape; the
     * per-chunk windows are anchored at the chunk origin. ps = [[tid]]. */
    const rocke_tile_distribution_t* distribution; /* _chunk_distribution(BS,VEC)*/
    int chunk_elems; /* BS * VEC                  */
    rocke_tensor_view_t gate_view; /* silu_mul: GateOut view ; packed: GateUp  */
    rocke_tensor_view_t up_view; /* silu_mul: UpOut view  (packed reuses gate)*/
    rocke_tensor_view_t out_view; /* Hidden view                              */
    rocke_value_t* ps_tid; /* the single P value tid (ps = [[ps_tid]]) */
} rocke_moe_stream_ctx_t;

/* ===================================================================== *
 *  rocke_moe_ssg_ctx_t  --  shared state for build_moe_static_scatter_gather.
 *
 *  Field order follows the Python prologue (lines 759-849).
 * ===================================================================== */
typedef struct rocke_moe_ssg_ctx
{
    /* ---- inputs / resolved environment -- */
    rocke_ir_builder_t* b;
    const rocke_fused_moe_spec_t* spec;
    const char* arch;

    /* ---- geometry scalars -- */
    int H; /* spec.hidden                                              */
    int BS; /* spec.block_size                                          */
    int EPT; /* elems_per_thread_hidden (= H / BS)                       */
    int VEC; /* _effective_vec(spec.vec, BS, H)                          */
    int chunks; /* EPT // VEC                                               */
    const char* dtype;
    const rocke_type_t* ty; /* io_ir_type(dtype)                               */

    /* ---- kernel params (Values), full 12-entry ABI -- */
    rocke_value_t* TopkIds; /* ptr<i32> readonly                         */
    rocke_value_t* TopkWeights; /* ptr<f32> readonly                         */
    rocke_value_t* Counter; /* ptr<i32> (atomic target)                  */
    rocke_value_t* X; /* ptr<dtype> readonly                       */
    rocke_value_t* SortedTokenIds; /* ptr<i32> writeonly                        */
    rocke_value_t* SortedWeights; /* ptr<f32> writeonly                        */
    rocke_value_t* GroupedInput; /* ptr<dtype> writeonly                      */
    rocke_value_t* tokens; /* i32 scalar (live: num_pairs)              */
    rocke_value_t* topk; /* i32 scalar (live: t_idx, num_pairs)       */
    rocke_value_t* num_experts; /* i32 scalar (live: valid_e)                */
    rocke_value_t* p_hidden; /* i32 scalar (ABI, unused body)             */
    rocke_value_t* slot_size; /* i32 scalar (live: slot base)              */

    /* ---- thread / grid decode + prologue scalars (SSA) -- */
    rocke_value_t* bid; /* block_id_x()                              */
    rocke_value_t* tid; /* thread_id_x()                             */
    rocke_value_t* out_row_slot; /* smem_alloc i32[1] "sg_out_row"            */
    rocke_value_t* num_pairs; /* tokens * topk                             */
    rocke_value_t* in_bounds; /* cmp_lt(bid, num_pairs)                    */
    rocke_value_t* c_vec; /* const_i32(VEC)                            */

    /* ---- inside the in_bounds / valid_e / is_lead nest (SSA) -- *
     * Populated by the claim phase; consumed by the copy phase. */
    rocke_value_t* eid; /* global_load_i32(TopkIds, bid)             */
    rocke_value_t* valid_e; /* (eid>=0) && (eid<num_experts)             */
    rocke_value_t* t_idx; /* bid / topk                                */
    rocke_value_t* is_lead; /* cmp_eq(tid, 0)                            */
    /* lead-only locals (live only inside the is_lead scf_if):                */
    rocke_value_t* local; /* atomic_add(Counter, eid, 1)              */
    rocke_value_t* base; /* eid * slot_size                           */
    rocke_value_t* out_row_lead; /* base + local                             */
    rocke_value_t* w; /* global_load_f32(TopkWeights, bid)         */
    /* post-sync broadcast + row bases (live in the copy phase):             */
    rocke_value_t* out_row; /* to_sgpr_u32(vec_extract(smem_load,0))     */
    rocke_value_t* src_row_base; /* t_idx * H                                 */
    rocke_value_t* dst_row_base; /* out_row * H                               */
} rocke_moe_ssg_ctx_t;

/* ===================================================================== *
 *  SHARED MODULE-LEVEL HELPERS (ported once; pure or SSA-emitting).
 *  Mirror the Python module-level functions used by every builder.
 * ===================================================================== */

/* _effective_vec(spec_vec, block_size, n): largest 2^k in {1,2,4,8} not
 * exceeding min(spec_vec,8) such that n % (block_size*ev)==0; returns 1 when
 * none of the larger widths divide. Pure. */
int rocke_moe_effective_vec(int spec_vec, int block_size, int n);

/* _chunk_distribution(block_size, vec): the one (block_size, vec) interleaved-
 * chunk TileDistribution -- TileDistributionEncoding(Hs=((BS,vec),),
 * Ps2RHs_major=((1,),), Ps2RHs_minor=((0,),), Ys2RHs_major=(1,),
 * Ys2RHs_minor=(1,)) -> make_static_tile_distribution. Arena-owned; NULL +
 * sticky error on failure. */
const rocke_tile_distribution_t*
    rocke_moe_chunk_distribution(rocke_ir_builder_t* b, int block_size, int vec);

/* _silu_mul_f32(b, g, u, one_f32, c_neg_log2e): the f32 SwiGLU chain in the
 * exact Python op order -- sig = rcp(fadd(one_f32, exp2(fmul(c_neg_log2e, g))));
 * silu = fmul(g, sig); return fmul(silu, u). Returns the f32 result Value. */
rocke_value_t* rocke_moe_silu_mul_f32(rocke_ir_builder_t* b,
                                      rocke_value_t* g,
                                      rocke_value_t* u,
                                      rocke_value_t* one_f32,
                                      rocke_value_t* c_neg_log2e);

/* ===================================================================== *
 *  GATHER PHASE FUNCTIONS  (build_moe_gather, lines 326-441).
 * ===================================================================== */

/* Prologue (lines 368-412): is_valid_spec gate; derive H/BS/EPT/VEC/dtype/ty;
 * set kernel max_workgroup_size=BS; declare the 5 params (X, SortedTokenIds,
 * GroupedInput, tokens, hidden) with their noalias/readonly/writeonly/align
 * attrs; bid/tid; load+pin token_id; valid_token; bucket_base; src_row_base;
 * c_vec; chunks. Fills ctx (kind==GATHER). Returns false (sticky error) on a
 * rejected spec. */
bool rocke_moe_gather_prologue(rocke_moe_stream_ctx_t* ctx);

/* Body (lines 416-439): scf_if(valid_token){ for k in chunks: interleaved
 * h_col = k*BS*VEC + tid*VEC; src_off=src_row_base+h_col; dst_off=bucket_base+
 * h_col; VEC==1 -> native global_load_f16/bf16 + global_store; else
 * global_load_vN + global_store_vN }. Returns ctx->b->kernel / NULL. */
rocke_kernel_def_t* rocke_moe_gather_body(rocke_moe_stream_ctx_t* ctx);

/* ===================================================================== *
 *  SILU_MUL PHASE FUNCTIONS  (build_moe_silu_mul, lines 449-573).
 * ===================================================================== */

/* Prologue (lines 491-523): gate; derive I_DIM/BS/EPT/VEC/dtype/ty;
 * max_workgroup_size=BS; declare 5 params (GateOut, UpOut, Hidden, total_pairs,
 * intermediate); bid/tid; row_base=bid*I; c_neg_log2e; one_f32; c_vec; chunks.
 * Fills ctx (kind==SILU_MUL). Returns false on rejected spec. */
bool rocke_moe_silu_mul_prologue(rocke_moe_stream_ctx_t* ctx);

/* Scalar body (lines 526-538): VEC==1 fallback. for k in chunks: off=row_base+
 * (k*BS*VEC + tid*VEC); g=load_scalar_as_f32(GateOut,off); u=load_scalar_as_f32
 * (UpOut,off); h=silu_mul_f32(...); store_scalar_from_f32(Hidden,off,h).
 * Returns ctx->b->kernel. */
rocke_kernel_def_t* rocke_moe_silu_mul_body_scalar(rocke_moe_stream_ctx_t* ctx);

/* Tile body (lines 540-573): VEC>1. Build distribution + chunk_elems + the
 * three (chunk_elems,) global views (filled into ctx). for k in chunks: anchor
 * gate/up/out tile windows at row_base + k*BS*VEC; load gate/up dt via the
 * distribution at ps=[[tid]]; out_dt = make_static_distributed_tensor; for y in
 * distribution.iterate_ys(): out_dt.set(y, silu_mul_f32(g_dt.get(y),
 * u_dt.get(y), ...)); store out_dt. Returns ctx->b->kernel. */
rocke_kernel_def_t* rocke_moe_silu_mul_body_tile(rocke_moe_stream_ctx_t* ctx);

/* ===================================================================== *
 *  SILU_MUL_PACKED PHASE FUNCTIONS  (build_moe_silu_mul_packed, 587-708).
 * ===================================================================== */

/* Prologue (lines 612-648): gate; I_DIM/BS/EPT/VEC/dtype/ty;
 * max_workgroup_size=BS; declare 4 params (GateUp, Hidden, total_pairs,
 * intermediate); bid/tid; two_i/i_const; gate_base=bid*2I; up_base=gate_base+I;
 * out_base=bid*I; c_neg_log2e; one_f32; c_vec; chunks. Fills ctx
 * (kind==SILU_MUL_PACKED). Returns false on rejected spec. */
bool rocke_moe_silu_mul_packed_prologue(rocke_moe_stream_ctx_t* ctx);

/* Scalar body (lines 658-670): VEC==1. for k in chunks: i_col=k*BS*VEC+tid*VEC;
 * g_off=gate_base+i_col; u_off=up_base+i_col; o_off=out_base+i_col; g/u via
 * load_scalar_as_f32(GateUp,...); h=silu_mul_f32; store_scalar_from_f32(Hidden,
 * o_off,h). Returns ctx->b->kernel. */
rocke_kernel_def_t* rocke_moe_silu_mul_packed_body_scalar(rocke_moe_stream_ctx_t* ctx);

/* Tile body (lines 674-708): VEC>1. Build distribution + chunk_elems + the
 * single GateUp view (used for both gate+up windows at gate_base/up_base) and
 * the Hidden out_view. for k in chunks: gate/up/out origins at *_base + k*BS*VEC;
 * load gate/up dt; out_dt over iterate_ys via silu_mul_f32; store. Returns
 * ctx->b->kernel. */
rocke_kernel_def_t* rocke_moe_silu_mul_packed_body_tile(rocke_moe_stream_ctx_t* ctx);

/* ===================================================================== *
 *  STATIC_SCATTER_GATHER PHASE FUNCTIONS  (lines 731-849).
 * ===================================================================== */

/* Prologue (lines 759-803): gate; H/BS/EPT/VEC/dtype/ty; max_workgroup_size=BS;
 * declare the 12 params in ABI order with their attrs; bid/tid; out_row_slot=
 * smem_alloc i32[1]; num_pairs=tokens*topk; in_bounds=cmp_lt(bid,num_pairs);
 * c_vec; chunks. Fills ctx. Returns false on rejected spec. */
bool rocke_moe_ssg_prologue(rocke_moe_ssg_ctx_t* ctx);

/* Slot-claim phase (lines 805-819): opens scf_if(in_bounds) -> eid load ->
 * valid_e -> scf_if(valid_e) -> t_idx, is_lead -> scf_if(is_lead){ local=
 * atomic_add(Counter,eid,1); base=eid*slot_size; out_row_lead=base+local;
 * smem_store out_row_lead; w=load(TopkWeights,bid); store SortedTokenIds[
 * out_row_lead]=t_idx; store SortedWeights[out_row_lead]=w }. Fills the claim
 * ctx fields. The scf scopes opened here stay open for the copy phase (the
 * driver must keep the region nest; see note). */
void rocke_moe_ssg_claim(rocke_moe_ssg_ctx_t* ctx);

/* Broadcast + copy phase (lines 820-847): sync(); out_row=to_sgpr_u32(vec_
 * extract(smem_load out_row_slot,0)); src_row_base=t_idx*H; dst_row_base=
 * out_row*H; for k in chunks: interleaved h_col=k*BS*VEC+tid*VEC; copy X[src]->
 * GroupedInput[dst] (VEC==1 native f16/bf16 else vN). Returns ctx->b->kernel /
 * NULL after closing the in_bounds/valid_e scopes. */
rocke_kernel_def_t* rocke_moe_ssg_copy(rocke_moe_ssg_ctx_t* ctx);

/* ===================================================================== *
 *  TOPK_WEIGHTED_REDUCE PHASE FUNCTIONS  (lines 880-997).
 * ===================================================================== */

/* Prologue (lines 911-958, 979-980): gate; H/BS/EPT/VEC/dtype/ty;
 * max_workgroup_size=BS; declare 7 params (DownOut, SortedTokenIds,
 * SortedWeights, Y, total_pairs, hidden, tokens); bid/tid; pin token_id; weight;
 * valid_token; bucket_base=bid*H; y_row_base=token_id*H; chunks; lane_chunk_base
 * =tid*EPT (BLOCK-PARTITIONED, not interleaved). Fills ctx (kind==REDUCE).
 * Returns false on rejected spec. */
bool rocke_moe_reduce_prologue(rocke_moe_stream_ctx_t* ctx);

/* Body (lines 981-995): scf_if(valid_token){ for k in chunks: h_col=
 * lane_chunk_base + k*VEC; src_off=bucket_base+h_col; dst_off=y_row_base+h_col;
 * VEC==1 -> v=load_scalar_as_f32(DownOut); atomic_add(Y,dst,weight*v); else
 * v_vec=load_vN; for i in VEC: atomic_add(Y, dst+i, weight*cast_to_f32(
 * vec_extract(v_vec,i))) }. Honors spec.bf16_accumulator (v2bf16 atomic path)
 * when set. Returns ctx->b->kernel / NULL. */
rocke_kernel_def_t* rocke_moe_reduce_body(rocke_moe_stream_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_FUSED_MOE_INTERNAL_H */
