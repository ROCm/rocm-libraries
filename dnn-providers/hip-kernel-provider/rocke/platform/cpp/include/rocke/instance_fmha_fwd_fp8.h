/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_fmha_fwd_fp8.h -- C99 port of the FP8 FMHA forward kernel
 * instance builder rocke/instances/common/fmha_fwd_fp8.py (CK Tile 01_fmha
 * fp8 parity).
 *
 * K and V are stored in fp8e4m3 (or bf8e5m2) and dequantised on load with
 * per-tensor scales; the MFMA-tiled body promotes them to f32 for the QK / PV
 * math and emits the cshuffle to the activation dtype (f16 / bf16) at the end.
 * The output O is stored back in the activation dtype, not in fp8.
 *
 *   Python (fmha_fwd_fp8.py)              C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   KvFp8DType (Literal)                  rocke_kv_fp8_dtype_t (enum)
 *   @dataclass FmhaFwdFp8Spec             rocke_fmha_fwd_fp8_spec_t
 *     .kernel_name()                      rocke_fmha_fwd_fp8_kernel_name(...)
 *   is_valid_spec(spec, arch)             rocke_fmha_fwd_fp8_is_valid_spec(...)
 *   _declare_params(kb, spec)             (internal to the .c)
 *   build_fmha_fwd_fp8(spec, arch)        rocke_build_fmha_fwd_fp8(b, spec, arch)
 *   fmha_fwd_fp8_grid(spec)               rocke_fmha_fwd_fp8_grid(...)
 *   fmha_fwd_fp8_signature(spec)          rocke_fmha_fwd_fp8_signature(...)
 *   (+ convenience: build -> lower .ll)   rocke_fmha_fwd_fp8_lower_to_llvm(...)
 *
 * The build entry mirrors the Python build op-for-op (same builder-call order,
 * same attrs) so the emitted IR is byte-identical to build_fmha_fwd_fp8 on the
 * default arch="gfx950". The heavy lifting (the QK -> online-softmax -> PV
 * pipeline) is delegated to the already-ported helper
 * rocke_mfma_attention_fwd_inner_body (helper_rocke.helpers.mfma_attention.h),
 * exactly as the Python instance delegates to mfma_attention_fwd_inner_body.
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass carries defaults;
 * in C the caller fills a rocke_fmha_fwd_fp8_spec_t. rocke_fmha_fwd_fp8_spec_default()
 * returns a struct with every field set to the Python dataclass default (the
 * `common` field must still be filled by the caller).
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate returns bool + a reason
 * string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_FMHA_FWD_FP8_H
#define ROCKE_INSTANCE_FMHA_FWD_FP8_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common._fmha_common.h" /* rocke_fmha_common_spec_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * KvFp8DType
 * ------------------------------------------------------------------ *
 *
 * KvFp8DType = Literal["fp8e4m3", "bf8e5m2"]. Recovered to the canonical
 * lowercase spelling via rocke_kv_fp8_dtype_name() (the spelling is fed straight
 * into the helper's kv_dtype and the kernel name). */
typedef enum rocke_kv_fp8_dtype
{
    ROCKE_KV_FP8_E4M3 = 0, /* "fp8e4m3" */
    ROCKE_KV_BF8_E5M2 /* "bf8e5m2" */
} rocke_kv_fp8_dtype_t;

/* Canonical lowercase spelling ("fp8e4m3"/"bf8e5m2"); NULL for out-of-range. */
const char* rocke_kv_fp8_dtype_name(rocke_kv_fp8_dtype_t d);

/* ------------------------------------------------------------------ *
 * FmhaFwdFp8Spec
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True)
 * class FmhaFwdFp8Spec:
 *     common: FmhaCommonSpec
 *     kv_dtype: KvFp8DType = "fp8e4m3"
 *     seqlen_q: int = 1
 *     seqlen_k: int = 0
 *     fp8_fnuz: bool = False
 *     waves_per_eu: Optional[int] = 4
 *     name: str = "rocke_fmha_fwd_fp8"
 *
 * waves_per_eu is Optional[int]: the Python None (use the LLVM heuristic) maps
 * to has_waves_per_eu=false; a concrete int maps to has_waves_per_eu=true with
 * the value in waves_per_eu. */
typedef struct rocke_fmha_fwd_fp8_spec
{
    rocke_fmha_common_spec_t common;
    rocke_kv_fp8_dtype_t kv_dtype; /* default ROCKE_KV_FP8_E4M3        */
    int seqlen_q; /* default 1                      */
    int seqlen_k; /* default 0                      */
    bool fp8_fnuz; /* default false                  */
    bool has_waves_per_eu; /* false == Python None           */
    int waves_per_eu; /* default 4 (when has_waves_per_eu) */
    const char* name; /* default "rocke_fmha_fwd_fp8"  */
} rocke_fmha_fwd_fp8_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still fill `common`. */
rocke_fmha_fwd_fp8_spec_t rocke_fmha_fwd_fp8_spec_default(void);

/* FmhaFwdFp8Spec.kernel_name() -> kernel_name_join(name, "H{head_size}",
 * "HQ{num_query_heads}", "HK{num_kv_heads}", common.dtype, kv_dtype,
 * "Q{seqlen_q}", common.mask_mode). NUL-terminated into out (capacity out_cap).
 * Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t rocke_fmha_fwd_fp8_kernel_name(const rocke_fmha_fwd_fp8_spec_t* spec,
                                              char* out,
                                              size_t out_cap);

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ *
 *
 * is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * Python-matching message and the function returns false. On accept returns
 * true and writes "ok". Mirrors the Python is_valid_spec gates exactly:
 *   - unknown arch                          (KeyError -> reason)
 *   - validate_common_spec failure
 *   - kv_dtype not fp8e4m3 / bf8e5m2
 *   - fnuz target without fp8_fnuz opt-in   (G3)
 *   - seqlen_q <= 0 / not multiple of BLOCK_M
 *   - head_size % 16 != 0
 *   - missing f16 MMA dtype combo / 16x16x16 shape
 *   - LDS budget (BLOCK_M*BLOCK_M*2 bytes) over cap                            */
bool rocke_fmha_fwd_fp8_is_valid_spec(const rocke_fmha_fwd_fp8_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap);

/* ------------------------------------------------------------------ *
 * build_fmha_fwd_fp8
 * ------------------------------------------------------------------ *
 *
 * build_fmha_fwd_fp8(spec, arch="gfx950") -> KernelDef. Builds the IR into the
 * supplied (already initialised) kernel-builder driven by the embedded
 * FmhaKernelBuilder pattern, op-for-op with the Python build, and returns the
 * kernel on success or NULL with the embedded builder's sticky error set.
 *
 * The caller passes a rocke_fmha_kernel_builder_t whose rocke_fmha_kernel_builder_init
 * was already called with spec.kernel_name() + spec.common; the caller owns its
 * lifetime (rocke_fmha_kernel_builder_free). `arch` NULL => "gfx950". */
rocke_kernel_def_t* rocke_build_fmha_fwd_fp8(rocke_fmha_kernel_builder_t* kb,
                                             const rocke_fmha_fwd_fp8_spec_t* spec,
                                             const char* arch);

/* Convenience: init `kb` with spec.kernel_name() + spec.common, then build. The
 * caller owns `kb` and frees it with rocke_fmha_kernel_builder_free(). Returns the
 * kernel or NULL. `arch` NULL => "gfx950". */
rocke_kernel_def_t* rocke_build_fmha_fwd_fp8_new(rocke_fmha_kernel_builder_t* kb,
                                                 const rocke_fmha_fwd_fp8_spec_t* spec,
                                                 const char* arch);

/* ------------------------------------------------------------------ *
 * fmha_fwd_fp8_grid
 * ------------------------------------------------------------------ *
 *
 * fmha_fwd_fp8_grid(spec) -> (seqlen_q // BLOCK_M, num_query_heads, 1). Writes
 * the triple into out[3]; returns ROCKE_OK (ROCKE_ERR_VALUE on NULL spec/out). */
rocke_status_t rocke_fmha_fwd_fp8_grid(const rocke_fmha_fwd_fp8_spec_t* spec, int out[3]);

/* ------------------------------------------------------------------ *
 * fmha_fwd_fp8_signature
 * ------------------------------------------------------------------ *
 *
 * fmha_fwd_fp8_signature(spec): declare the kernel ABI into a sig-probe
 * FmhaKernelBuilder ("rocke_fmha_fwd_fp8_sig_probe") and emit its
 * SignatureBuilder shape. On ROCKE_OK *out_items / *out_count hold the
 * arena-owned manifest; `arena` backs the SignatureBuilder storage. The probe
 * builder is created + freed internally. */
rocke_status_t rocke_fmha_fwd_fp8_signature(const rocke_fmha_fwd_fp8_spec_t* spec,
                                            rocke_arena_t* arena,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count);

/* ------------------------------------------------------------------ *
 * convenience lower-to-.ll
 * ------------------------------------------------------------------ *
 *
 * Given a spec, init a builder, build, and lower to LLVM .ll text. `arch` NULL
 * => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the
 * caller frees with free(); on failure it is left NULL and (if err != NULL,
 * capacity err_cap) a diagnostic is written. Internally owns + frees its builder. */
rocke_status_t rocke_fmha_fwd_fp8_lower_to_llvm(const rocke_fmha_fwd_fp8_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_FMHA_FWD_FP8_H */
