/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._matmul_nbits_decode_gemv.h
 *   -- C99 port of rocke.instances.common._matmul_nbits_decode_gemv.
 *
 * Ported symbol:
 *   build_decode_gemv_matmul_nbits(spec, arch="gfx1151") -> KernelDef
 *       rocke_build_decode_gemv_matmul_nbits(b, spec, arch)
 *
 * The Python body is the decode-phase ``lm_head`` GEMV specialization of the
 * ``MatMulNBits`` instance: very large N with a tiny M (decode token, M=1). It
 * is a dedicated arch-agnostic scalar body -- one thread per output column, no
 * WMMA -- so unlike the large_n / skinny_n bodies it never touches the WMMA
 * atom or warp-tile geometry. It reads only a handful of fields off the spec:
 *
 *     N, K, group_size, block_size, scale_dtype
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python ``MatMulNBitsSpec`` dataclass (and
 * its TileSpec, WarpTileBlockSizeMixin, etc.) is a peer port that is NOT
 * required to build the decode-GEMV body, which only consults the five fields
 * above. To stay faithful AND self-contained, this header declares a minimal
 * value struct, ``rocke_matmul_nbits_decode_gemv_spec_t``, carrying exactly the
 * fields the body reads. A caller holding a full ported MatMulNBitsSpec simply
 * copies those five fields across.
 *
 * The body mirrors the Python builder-call sequence byte-faithfully so the
 * resulting IR is identical (same op order, same operands, same attrs).
 *
 * DEPENDENCY (peer, may be unresolved at -fsyntax-only):
 *   - rocke/ir.h                          (IRBuilder + Value API)
 *   - the i4 unpack helper: rocke.helpers.i4_dequant.unpack_i4_byte_to_pair_f32
 *     -> rocke_unpack_i4_byte_to_pair_f32(b, byte, &lo, &hi). Declared (extern)
 *     below so this body compiles against the not-yet-ported sibling; its real
 *     definition lives in the i4_dequant peer port.
 *
 * Error model: identical to the rest of the C port -- a NULL/failed builder
 * makes the call a no-op returning NULL; any internal builder failure leaves
 * the sticky error on `b` and returns NULL.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_MATMUL_NBITS_DECODE_GEMV_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_MATMUL_NBITS_DECODE_GEMV_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Scale storage wire dtype (canonicalised by the Python ``_scale_wire_dtype``).
 * The decode body picks rocke_f16() for ROCKE_NBITS_SCALE_F16 and rocke_f32()
 * otherwise, mirroring:
 *     scale_t = F16 if _scale_wire_dtype(spec.scale_dtype) == "f16" else F32
 */
typedef enum rocke_matmul_nbits_scale_wire_dtype
{
    ROCKE_NBITS_SCALE_F16 = 0, /* "f16" / "fp16"  */
    ROCKE_NBITS_SCALE_F32 = 1 /* "f32" / "fp32"  */
} rocke_matmul_nbits_scale_wire_dtype_t;

/* Minimal spec view used by the decode-GEMV body. Field names mirror the Python
 * MatMulNBitsSpec; only the fields the body reads are carried.
 *
 *   N           -> spec.N            (compile-time, baked as const)
 *   K           -> spec.K            (compile-time, baked as const)
 *   group_size  -> spec.group_size   (group-of-32 by v1 contract)
 *   block_size  -> spec.block_size   (derived block size; max_workgroup_size)
 *   scale_wire  -> _scale_wire_dtype(spec.scale_dtype)
 */
typedef struct rocke_matmul_nbits_decode_gemv_spec
{
    int N;
    int K;
    int group_size;
    int block_size;
    rocke_matmul_nbits_scale_wire_dtype_t scale_wire;
} rocke_matmul_nbits_decode_gemv_spec_t;

/* build_decode_gemv_matmul_nbits(spec, arch="gfx1151") -> KernelDef.
 *
 * Builds the decode-GEMV IR into the supplied (already rocke_ir_builder_init'd)
 * builder `b`, byte-faithfully to the Python, and returns the kernel
 * (b->kernel) on success or NULL with b's sticky error set. The Python defaults
 * arch to V1_ARCH ("gfx1151"); `arch` is accepted for signature parity but the
 * body is arch-agnostic (no atom/geometry lookup), so it does not affect the IR.
 * `arch` NULL is treated as "gfx1151".
 *
 * Like the Python (which expects ``IRBuilder(spec.kernel_name())``), this does
 * NOT re-init the builder; the caller owns its lifetime and should have created
 * it with the spec's kernel name. */
rocke_kernel_def_t* rocke_build_decode_gemv_matmul_nbits(
    rocke_ir_builder_t* b, const rocke_matmul_nbits_decode_gemv_spec_t* spec, const char* arch);

/* ---- peer helper (rocke.helpers.i4_dequant) ----
 *
 * unpack_i4_byte_to_pair_f32(b, packed_byte) -> (low_f32, high_f32).
 * Declared here (extern) so the decode body compiles against the i4_dequant
 * peer port. The real definition is provided by that port; under
 * -fsyntax-only it is an unresolved peer (expected). Writes the two f32 lanes
 * through `out_lo` / `out_hi`. */
void rocke_unpack_i4_byte_to_pair_f32(rocke_ir_builder_t* b,
                                      rocke_value_t* packed_byte,
                                      rocke_value_t** out_lo,
                                      rocke_value_t** out_hi);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_MATMUL_NBITS_DECODE_GEMV_H */
