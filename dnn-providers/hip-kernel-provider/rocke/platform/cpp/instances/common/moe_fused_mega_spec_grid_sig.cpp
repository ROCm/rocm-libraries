// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_fused_mega_spec_grid_sig.c -- VALUE-TYPE + PURE-METADATA chunk
 * of the C99 port of rocke/instances/common/moe_fused_mega.py.
 *
 * SCOPE (this translation unit): everything in moe_fused_mega.py that emits NO
 * IR -- the FusedMegaKernelSpec value type and its derivations:
 *
 *   Python (moe_fused_mega.py)              C99 (this file)
 *   -------------------------------------   --------------------------------------
 *   FusedMegaKernelSpec defaults            rocke_moe_fused_mega_kernel_spec_default()
 *   FusedMegaKernelSpec.__post_init__       rocke_moe_fused_mega_kernel_spec_finalize()
 *   spec._data_spec() + gate_up_tile()      (folded into the universal-spec
 *     -> gate_up_universal_spec()             builder below)
 *                                           rocke_moe_fused_mega_gate_up_universal_spec()
 *   spec._data_spec() + down_tile()         rocke_moe_fused_mega_down_universal_spec()
 *     -> down_universal_spec()
 *   spec.kernel_name()                      rocke_moe_fused_mega_kernel_name()
 *   moe_fused_mega_grid(...)                rocke_moe_fused_mega_grid()
 *   moe_fused_mega_signature(spec)          rocke_moe_fused_mega_signature()
 *
 * NONE of these touch the IR builder; they are bit-for-bit value producers whose
 * results are later baked into the IR by the build chunk. A byte-identical
 * builder-call sequence in the build chunk follows from byte-identical values
 * here.
 */

#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/instance_gemm_universal.h"
#include "rocke/instance_moe_fused_mega.h"
#include "rocke/instance_moe_fused_mega_internal.h"

/* ===================================================================== *
 *  FusedMegaKernelSpec defaults + __post_init__
 *
 *  Python (moe_fused_mega.py):
 *      @dataclass(frozen=True)
 *      class FusedMegaKernelSpec:
 *          name: str
 *          tile_m: int = 16
 *          tile_n_inter: int = 256
 *          tile_k_gu: int = 32
 *          warp_m: int = 1
 *          warp_n: int = 4
 *          warp_tile_m: int = 16
 *          warp_tile_n: int = 16
 *          warp_tile_k: int = 32
 *          tile_n_down: int = 256
 *          tile_k_down: int = 64
 *          trait: TraitSpec = field(default_factory=lambda:
 *                                   TraitSpec(epilogue="default"))
 *          wave_size: int = 64
 *          block_size: int = 0
 *          dtype: str = "fp16"
 *
 *          def __post_init__(self):
 *              if self.block_size == 0:
 *                  object.__setattr__(self, "block_size",
 *                      self.warp_m * self.warp_n * self.wave_size)
 *
 *  NOTE: the trait default is TraitSpec(epilogue="default") -- i.e. every
 *  TraitSpec field at its gemm_universal default EXCEPT epilogue, which is
 *  "default" (not the gemm "cshuffle"). We reuse rocke_gemm_universal_spec_default
 *  to get the canonical TraitSpec defaults, then override epilogue.
 * ===================================================================== */

rocke_moe_fused_mega_kernel_spec_t rocke_moe_fused_mega_kernel_spec_default(void)
{
    rocke_moe_fused_mega_kernel_spec_t s;
    rocke_gemm_universal_spec_t g;

    memset(&s, 0, sizeof(s));

    s.name = NULL; /* caller must set (Python: required field, no default) */

    s.tile_m = 16; /* default 16 */
    s.tile_n_inter = 256; /* default 256 */
    s.tile_k_gu = 32; /* default 32 */
    s.warp_m = 1; /* default 1 */
    s.warp_n = 4; /* default 4 */
    s.warp_tile_m = 16; /* default 16 */
    s.warp_tile_n = 16; /* default 16 */
    s.warp_tile_k = 32; /* default 32 */
    s.tile_n_down = 256; /* default 256 */
    s.tile_k_down = 64; /* default 64 */

    /* trait = TraitSpec(epilogue="default"): canonical TraitSpec defaults with
     * epilogue overridden to "default". */
    g = rocke_gemm_universal_spec_default();
    s.trait = g.trait;
    s.trait.epilogue = "default";

    s.wave_size = 64; /* default 64 */
    s.block_size = 0; /* derived at finalize() */
    s.dtype = "fp16"; /* default "fp16" */

    return s;
}

void rocke_moe_fused_mega_kernel_spec_finalize(rocke_moe_fused_mega_kernel_spec_t* spec)
{
    if(spec == NULL)
    {
        return;
    }
    /* __post_init__: block_size = warp_m * warp_n * wave_size when 0.
     * (NOTE: this is NOT the WarpTileBlockSizeMixin derivation -- there is no
     * warp_k factor here; the Python spells out warp_m*warp_n*wave_size.) */
    if(spec->block_size == 0)
    {
        spec->block_size = spec->warp_m * spec->warp_n * spec->wave_size;
    }
}

/* ===================================================================== *
 *  gate_up_universal_spec() / down_universal_spec()
 *
 *  Python:
 *      def _data_spec(self) -> DataSpec:
 *          dt = "fp16" if self.dtype in ("f16","fp16") else self.dtype
 *          return DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt)
 *
 *      def gate_up_tile(self) -> TileSpec:
 *          return TileSpec(tile_m, tile_n=tile_n_inter, tile_k=tile_k_gu,
 *                          warp_m, warp_n, warp_tile_m, warp_tile_n, warp_tile_k)
 *      def down_tile(self) -> TileSpec:
 *          return TileSpec(tile_m, tile_n=tile_n_down, tile_k=tile_k_down,
 *                          warp_m, warp_n, warp_tile_m, warp_tile_n, warp_tile_k)
 *
 *      def gate_up_universal_spec(self) -> UniversalGemmSpec:
 *          return UniversalGemmSpec(name=self.name + "_gu",
 *              tile=self.gate_up_tile(), trait=self.trait, data=self._data_spec(),
 *              wave_size=self.wave_size, block_size=self.block_size, batched=True)
 *      def down_universal_spec(self) -> UniversalGemmSpec:
 *          return UniversalGemmSpec(name=self.name + "_down",
 *              tile=self.down_tile(), trait=self.trait, data=self._data_spec(),
 *              wave_size=self.wave_size, block_size=self.block_size, batched=True)
 *
 *  Both fold the _data_spec()/tile derivations directly into the universal spec.
 *  The UniversalGemmSpec is otherwise constructed positionally with the gemm
 *  dataclass defaults for the fields not listed (dtype_acc="fp32",
 *  layout="RCR", _fused_epilogue=None). We start from
 *  rocke_gemm_universal_spec_default() to inherit those, then overwrite.
 *
 *  The "_gu"/"_down" suffixed names need stable storage. The Python builds a new
 *  str; here the *out struct's `name` is a const char* -- the caller owns it.
 *  But _data_spec / the tile / trait derivations require no storage. For the
 *  name we point at a small static-thread-local buffer per call site? No: to
 *  stay allocation-free and faithful, the name is written into a buffer the
 *  CALLER must keep alive. The build chunk passes its own scratch. Here we use a
 *  module-static reusable buffer keyed by suffix; a single build holds at most
 *  one gate_up + one down spec live at a time, so two distinct static buffers
 *  suffice and never alias.
 * ===================================================================== */

/* Reusable name storage. The Python kernel_name() of the gate_up spec is the
 * only consumer that reads `name` again later, and the build chunk reads
 * u_gu->name / u_down->name for diagnostics; one buffer per role is enough for
 * a single in-flight build. */
static char rocke_moe_mega_gu_name[256];
static char rocke_moe_mega_down_name[256];

static void rocke_moe_mega_fill_universal_common(const rocke_moe_fused_mega_kernel_spec_t* spec,
                                                 rocke_gemm_universal_spec_t* out)
{
    const char* dt;

    *out = rocke_gemm_universal_spec_default();

    /* _data_spec(): dt = "fp16" if dtype in ("f16","fp16") else dtype. */
    if(spec->dtype != NULL && (strcmp(spec->dtype, "f16") == 0 || strcmp(spec->dtype, "fp16") == 0))
    {
        dt = "fp16";
    }
    else
    {
        dt = spec->dtype;
    }
    out->data.dtype_a = dt;
    out->data.dtype_b = dt;
    out->data.dtype_c = dt;
    /* dtype_acc / layout keep the gemm defaults (fp32 / RCR). */

    out->trait = spec->trait;
    out->wave_size = spec->wave_size;
    out->block_size = spec->block_size;
    out->batched = true;
}

void rocke_moe_fused_mega_gate_up_universal_spec(const rocke_moe_fused_mega_kernel_spec_t* spec,
                                                 rocke_gemm_universal_spec_t* out)
{
    if(spec == NULL || out == NULL)
    {
        return;
    }
    rocke_moe_mega_fill_universal_common(spec, out);

    /* name = self.name + "_gu" */
    rocke_moe_mega_gu_name[0] = '\0';
    if(spec->name != NULL)
    {
        snprintf(rocke_moe_mega_gu_name, sizeof(rocke_moe_mega_gu_name), "%s_gu", spec->name);
    }
    else
    {
        snprintf(rocke_moe_mega_gu_name, sizeof(rocke_moe_mega_gu_name), "_gu");
    }
    out->name = rocke_moe_mega_gu_name;

    /* gate_up_tile() */
    out->tile.tile_m = spec->tile_m;
    out->tile.tile_n = spec->tile_n_inter;
    out->tile.tile_k = spec->tile_k_gu;
    out->tile.warp_m = spec->warp_m;
    out->tile.warp_n = spec->warp_n;
    out->tile.warp_k = 1; /* TileSpec default (not exposed by the mega spec) */
    out->tile.warp_tile_m = spec->warp_tile_m;
    out->tile.warp_tile_n = spec->warp_tile_n;
    out->tile.warp_tile_k = spec->warp_tile_k;
}

void rocke_moe_fused_mega_down_universal_spec(const rocke_moe_fused_mega_kernel_spec_t* spec,
                                              rocke_gemm_universal_spec_t* out)
{
    if(spec == NULL || out == NULL)
    {
        return;
    }
    rocke_moe_mega_fill_universal_common(spec, out);

    /* name = self.name + "_down" */
    rocke_moe_mega_down_name[0] = '\0';
    if(spec->name != NULL)
    {
        snprintf(rocke_moe_mega_down_name, sizeof(rocke_moe_mega_down_name), "%s_down", spec->name);
    }
    else
    {
        snprintf(rocke_moe_mega_down_name, sizeof(rocke_moe_mega_down_name), "_down");
    }
    out->name = rocke_moe_mega_down_name;

    /* down_tile() */
    out->tile.tile_m = spec->tile_m;
    out->tile.tile_n = spec->tile_n_down;
    out->tile.tile_k = spec->tile_k_down;
    out->tile.warp_m = spec->warp_m;
    out->tile.warp_n = spec->warp_n;
    out->tile.warp_k = 1; /* TileSpec default */
    out->tile.warp_tile_m = spec->warp_tile_m;
    out->tile.warp_tile_n = spec->warp_tile_n;
    out->tile.warp_tile_k = spec->warp_tile_k;
}

/* ===================================================================== *
 *  kernel_name()
 *
 *  Python:
 *      def kernel_name(self) -> str:
 *          return self.gate_up_universal_spec().kernel_name() + "_fused_mega"
 * ===================================================================== */

rocke_status_t rocke_moe_fused_mega_kernel_name(const rocke_moe_fused_mega_kernel_spec_t* spec,
                                                char* out,
                                                size_t out_cap)
{
    rocke_gemm_universal_spec_t u_gu;
    char gu[224];
    rocke_status_t st;
    size_t need;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    rocke_moe_fused_mega_gate_up_universal_spec(spec, &u_gu);
    st = rocke_gemm_universal_kernel_name(&u_gu, gu, sizeof(gu));
    if(st != ROCKE_OK)
    {
        return st;
    }

    /* + "_fused_mega" */
    need = strlen(gu) + strlen("_fused_mega") + 1;
    if(need > out_cap)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(out, out_cap, "%s_fused_mega", gu);
    return ROCKE_OK;
}

/* ===================================================================== *
 *  moe_fused_mega_grid()
 *
 *  Python:
 *      def moe_fused_mega_grid(num_m_blocks, inter, spec):
 *          sub_gu = spec.tile_n_inter
 *          gx = (inter + sub_gu - 1) // sub_gu
 *          return (gx, num_m_blocks, 1)
 * ===================================================================== */

void rocke_moe_fused_mega_grid(int num_m_blocks,
                               int inter,
                               const rocke_moe_fused_mega_kernel_spec_t* spec,
                               int* out_gx,
                               int* out_gy,
                               int* out_gz)
{
    int sub_gu;
    int gx;

    if(spec == NULL)
    {
        return;
    }
    sub_gu = spec->tile_n_inter;
    gx = (inter + sub_gu - 1) / sub_gu;

    if(out_gx != NULL)
    {
        *out_gx = gx;
    }
    if(out_gy != NULL)
    {
        *out_gy = num_m_blocks;
    }
    if(out_gz != NULL)
    {
        *out_gz = 1;
    }
}

/* ===================================================================== *
 *  moe_fused_mega_signature()
 *
 *  Python:
 *      def moe_fused_mega_signature(spec):
 *          dt = spec.dtype if spec.dtype in ("f16","fp16","bf16") else "f16"
 *          return (SignatureBuilder()
 *              .ptr("A", dt).ptr("WGate", dt).ptr("WUp", dt).ptr("WDown", dt)
 *              .ptr("SortedTokenIds", "i32").ptr("SortedWeights", "f32")
 *              .ptr("BlockExpertIds", "i32").ptr("Y", "f32")
 *              .scalar("M","i32").scalar("N","i32").scalar("K","i32")
 *              .scalar("H_out","i32")
 *              .scalar("stride_a","i32").scalar("stride_b_gate","i32")
 *              .scalar("stride_b_up","i32").scalar("stride_b_down","i32")
 *              .scalar("slot_size","i32").scalar("tokens","i32")
 *              .build())
 *
 *  8 ptr params + 11 scalar params = 19 entries. The public API writes into a
 *  caller array (no arena), so the {name,type} strings are static literals
 *  matching ptr_type_str(dtype, "global") = "ptr<<canon>, global>" (f16/fp16 ->
 *  "f16"). dt fallback: spec.dtype if in {f16,fp16,bf16} else "f16".
 * ===================================================================== */

#define ROCKE_MOE_MEGA_SIG_COUNT 19

rocke_status_t rocke_moe_fused_mega_signature(const rocke_moe_fused_mega_kernel_spec_t* spec,
                                              rocke_sig_entry_t* out,
                                              size_t out_cap,
                                              size_t* out_count)
{
    const char* ptr_dt_type; /* ptr<<dt>, global> for A/WGate/WUp/WDown */
    size_t i = 0;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(out_cap < ROCKE_MOE_MEGA_SIG_COUNT)
    {
        return ROCKE_ERR_VALUE;
    }

    /* dt = spec.dtype if in ("f16","fp16","bf16") else "f16"; ptr_type_str then
     * canonicalises f16/fp16 -> "f16". So:
     *   dtype in {f16,fp16}        -> "ptr<f16, global>"
     *   dtype == "bf16"            -> "ptr<bf16, global>"
     *   otherwise                  -> "ptr<f16, global>" */
    if(spec->dtype != NULL && strcmp(spec->dtype, "bf16") == 0)
    {
        ptr_dt_type = "ptr<bf16, global>";
    }
    else
    {
        /* f16 / fp16 / fallback all canonicalise to f16. */
        ptr_dt_type = "ptr<f16, global>";
    }

    /* 8 pointer params (Python order). */
    out[i].name = "A";
    out[i].type = ptr_dt_type;
    ++i;
    out[i].name = "WGate";
    out[i].type = ptr_dt_type;
    ++i;
    out[i].name = "WUp";
    out[i].type = ptr_dt_type;
    ++i;
    out[i].name = "WDown";
    out[i].type = ptr_dt_type;
    ++i;
    out[i].name = "SortedTokenIds";
    out[i].type = "ptr<i32, global>";
    ++i;
    out[i].name = "SortedWeights";
    out[i].type = "ptr<f32, global>";
    ++i;
    out[i].name = "BlockExpertIds";
    out[i].type = "ptr<i32, global>";
    ++i;
    out[i].name = "Y";
    out[i].type = "ptr<f32, global>";
    ++i;

    /* 11 scalar params (Python order), all i32. */
    out[i].name = "M";
    out[i].type = "i32";
    ++i;
    out[i].name = "N";
    out[i].type = "i32";
    ++i;
    out[i].name = "K";
    out[i].type = "i32";
    ++i;
    out[i].name = "H_out";
    out[i].type = "i32";
    ++i;
    out[i].name = "stride_a";
    out[i].type = "i32";
    ++i;
    out[i].name = "stride_b_gate";
    out[i].type = "i32";
    ++i;
    out[i].name = "stride_b_up";
    out[i].type = "i32";
    ++i;
    out[i].name = "stride_b_down";
    out[i].type = "i32";
    ++i;
    out[i].name = "slot_size";
    out[i].type = "i32";
    ++i;
    out[i].name = "tokens";
    out[i].type = "i32";
    ++i;

    if(out_count != NULL)
    {
        *out_count = i; /* == ROCKE_MOE_MEGA_SIG_COUNT (19) */
    }
    return ROCKE_OK;
}
