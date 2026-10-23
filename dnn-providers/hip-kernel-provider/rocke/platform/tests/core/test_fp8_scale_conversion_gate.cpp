// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_fp8_scale_conversion_gate.cpp -- the fused scale-and-convert ops
 * (arith.cvt_scalef32_pk_f32_fp8 / _bf8) lower only for a target that has the
 * instruction. Every other target is refused at lowering with one exact
 * message, because LLVM can select the intrinsic for such a target and write
 * an encoding the GPU does not have. Mirrors
 * tests/core/test_fp8_scale_conversion_gate.py, which pins the same text for
 * the Python engine.
 */

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(expr)                                                 \
    do                                                              \
    {                                                               \
        if(!(expr))                                                 \
        {                                                           \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); \
            return 1;                                               \
        }                                                           \
    } while(0)

struct Variant
{
    bool bf8;
    const char* op;
    const char* instruction;
    const char* fallback;
    const char* intrinsic;
};

static const Variant kVariants[] = {
    {false,
     "arith.cvt_scalef32_pk_f32_fp8",
     "v_cvt_scalef32_pk_f32_fp8",
     "cvt_pk_f32_fp8x4",
     "@llvm.amdgcn.cvt.scalef32.pk.f32.fp8("},
    {true,
     "arith.cvt_scalef32_pk_f32_bf8",
     "v_cvt_scalef32_pk_f32_bf8",
     "cvt_pk_f32_bf8x4",
     "@llvm.amdgcn.cvt.scalef32.pk.f32.bf8("},
};

/* D[0:4] = convert(S[0:4]) * scale, through the fused op. */
static bool build(rocke_ir_builder_t* b, bool bf8)
{
    const rocke_type_t* elem = bf8 ? rocke_bf8e5m2() : rocke_fp8e4m3();
    rocke_value_t* src = rocke_b_param(b, "S", rocke_ptr_type(b, elem, "global"), NULL);
    rocke_value_t* dst = rocke_b_param(b, "D", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* scale = rocke_b_param(b, "scale", rocke_f32(), NULL);
    rocke_value_t* idx = rocke_b_const_i32(b, 0);
    rocke_value_t* v = rocke_b_global_load_vN(b, src, idx, elem, 4, 0);
    rocke_value_t* r = bf8 ? rocke_b_cvt_scalef32_pk_f32_bf8x4(b, v, scale)
                           : rocke_b_cvt_scalef32_pk_f32_fp8x4(b, v, scale);
    rocke_b_global_store_vN(b, dst, idx, r, 4, 0);
    rocke_b_ret(b);
    return rocke_ir_builder_ok(b);
}

static int check_variant(const Variant& t)
{
    /* Every wired target other than gfx950 (the Python test reads the same
     * list from the registry). */
    static const char* const kRefused[]
        = {"gfx90a", "gfx942", "gfx1151", "gfx1201", "gfx11-generic", "gfx12-generic", "gfx1250"};
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "fp8_scale_conversion_gate") == ROCKE_OK);
    CHECK(build(&b, t.bf8));
    const rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);

    /* The one target with the instruction still lowers it. */
    char* text = NULL;
    char err[ROCKE_ERR_MSG_CAP] = {0};
    CHECK(rocke_lower_kernel_to_llvm_ex(
              kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx950", &text, err, sizeof(err))
          == ROCKE_OK);
    CHECK(text != NULL && std::strstr(text, t.intrinsic) != NULL);
    std::free(text);

    for(const char* arch : kRefused)
    {
        char want[ROCKE_ERR_MSG_CAP];
        std::snprintf(want,
                      sizeof(want),
                      "%s needs %s, which %s does not have (gfx950 does); convert with %s and "
                      "multiply by the scale instead",
                      t.op,
                      t.instruction,
                      arch,
                      t.fallback);
        text = NULL;
        err[0] = '\0';
        const rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_LLVM20, arch, &text, err, sizeof(err));
        if(st != ROCKE_ERR_VALUE || text != NULL || std::strcmp(err, want) != 0)
        {
            std::fprintf(stderr, "%s on %s: status %d, message '%s'\n", t.op, arch, (int)st, err);
            std::free(text);
            rocke_ir_builder_free(&b);
            return 1;
        }
    }
    rocke_ir_builder_free(&b);
    return 0;
}

int main(void)
{
    for(const Variant& t : kVariants)
    {
        if(check_variant(t) != 0)
        {
            return 1;
        }
    }
    return 0;
}
