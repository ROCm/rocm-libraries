// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Native twin of tests/core/test_mma_scale_sel_lds_dma_offset.py: the C++
// builder's scale_sel and LDS-DMA offset entry points are not bound to Python,
// so their guard rails and both C++ lowerings are checked here.
#include "rocke/error.hpp"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#define CHECK(expr)                                                 \
    do                                                              \
    {                                                               \
        if(!(expr))                                                 \
        {                                                           \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); \
            return 1;                                               \
        }                                                           \
    } while(0)

#define CHECK_THROWS(stmt, needle)                    \
    do                                                \
    {                                                 \
        try                                           \
        {                                             \
            stmt;                                     \
            CHECK(false);                             \
        }                                             \
        catch(const ckc::Error& error)                \
        {                                             \
            CHECK(error.code() == ROCKE_ERR_VALUE);   \
            CHECK(std::strstr(error.what(), needle)); \
        }                                             \
    } while(0)

static const char* const kMx16 = "mfma_scale_f32_16x16x128_f8f6f4";

struct Mx16Params
{
    rocke_value_t *a, *b, *c, *sa, *sb, *out;
};

/* Kernel params for the 16x16x128 MX atom; `c_lanes` and `scale` let a case
 * hand the builder a malformed accumulator or scale. */
static Mx16Params
    mx16_params(rocke_ir_builder_t* b, int c_lanes = 4, const rocke_type_t* scale = rocke_i32())
{
    Mx16Params p;
    p.a = rocke_b_param(b, "a", rocke_vector_type(b, rocke_i32(), 8), nullptr);
    p.b = rocke_b_param(b, "b", rocke_vector_type(b, rocke_i32(), 8), nullptr);
    p.c = rocke_b_param(b, "c", rocke_vector_type(b, rocke_f32(), c_lanes), nullptr);
    p.sa = rocke_b_param(b, "sa", scale, nullptr);
    p.sb = rocke_b_param(b, "sb", scale, nullptr);
    p.out = rocke_b_param(b, "out", rocke_ptr_type(b, rocke_f32(), "global"), nullptr);
    return p;
}

/* Builds out[0] = mfma_scale(a, b, c, sa, sb)[0] with scale_sel (sel_a, sel_b);
 * the store keeps the MMA live. Returns the tile.mma op. */
static rocke_op_t* build_mx16(rocke_ir_builder_t* b, int sel_a, int sel_b)
{
    Mx16Params p = mx16_params(b);
    rocke_value_t* r
        = rocke_b_mfma_scale_f32_16x16x128_f8f6f4_sel(b, p.a, p.b, p.c, p.sa, p.sb, sel_a, sel_b);
    rocke_op_t* mma = b->kernel->body->ops[b->kernel->body->num_ops - 1];
    rocke_b_global_store(b, p.out, rocke_b_const_i32(b, 0), rocke_b_vec_extract(b, r, 0), 4);
    rocke_b_ret(b);
    return mma;
}

/* Both C++ lowerers refuse the kernel with `status`, and their messages hold
 * `needle`. Consumes the builder (HIP reports on its error channel). */
static int expect_lowerings_fail(rocke_ir_builder_t* b,
                                 const char* gfx,
                                 rocke_status_t status,
                                 const char* needle)
{
    for(int f = 0; f < rocke_llvm_flavor_count(); ++f)
    {
        char* ll = nullptr;
        char err[ROCKE_ERR_MSG_CAP] = {};
        auto flavor = rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(f));
        CHECK(rocke_lower_kernel_to_llvm_ex(b->kernel, flavor, gfx, &ll, err, sizeof(err))
              == status);
        CHECK(std::strstr(err, needle));
        std::free(ll);
    }
    rocke_strbuf_t hip;
    CHECK(rocke_strbuf_init(&hip, 0) == 0);
    rocke_lower_hip_opts_t opts = {};
    opts.arch = gfx;
    CHECK(rocke_lower_kernel_to_hip(b, b->kernel, &opts, &hip) == status);
    CHECK(std::strstr(rocke_ir_builder_error(b), needle));
    rocke_strbuf_free(&hip);
    return 0;
}

static int test_scale_sel_lowers()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "mx16_scale_sel") == ROCKE_OK);
    rocke_op_t* mma = build_mx16(&b, 2, 3);
    const rocke_attr_value_t* sel = rocke_attr_get(&mma->attrs, "scale_sel");
    CHECK(sel && sel->kind == ROCKE_ATTR_INT_LIST && sel->u.ilist.count == 2);
    CHECK(sel->u.ilist.ints[0] == 2 && sel->u.ilist.ints[1] == 3);
    for(int f = 0; f < rocke_llvm_flavor_count(); ++f)
    {
        char* ll = nullptr;
        auto flavor = rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(f));
        CHECK(rocke_lower_kernel_to_llvm(b.kernel, flavor, "gfx950", &ll) == ROCKE_OK);
        CHECK(std::strstr(ll, "@llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4("));
        if(flavor == ROCKE_LLVM_FLAVOR_LLVM23)
        {
            // llvm23: (A, B, C, cbsz, blgp, opsel_a, sa, opsel_b, sb)
            CHECK(std::strstr(ll, "i32 0, i32 0, i32 2, i32 "));
            CHECK(!std::strstr(ll, "i32 0, i32 0, i32 0, i32 2"));
        }
        else
        {
            // llvm20/22: (A, B, C, cbsz, blgp, 0, opsel_a, sa, opsel_b, sb, 0)
            CHECK(std::strstr(ll, "i32 0, i32 0, i32 0, i32 2, i32 "));
        }
        CHECK(std::strstr(ll, ", i32 3, i32 "));
        std::free(ll);
    }
    rocke_strbuf_t hip;
    CHECK(rocke_strbuf_init(&hip, 0) == 0);
    rocke_lower_hip_opts_t opts = {};
    opts.arch = "gfx950";
    CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &hip) == ROCKE_OK);
    CHECK(std::strstr(hip.data, "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4("));
    CHECK(std::strstr(hip.data, ", 0, 0, 2, "));
    CHECK(std::strstr(hip.data, ", 3, "));
    rocke_strbuf_free(&hip);
    rocke_ir_builder_free(&b);
    return 0;
}

static int test_builder_rejects_bad_scale_sel()
{
    const int bad[][2] = {{4, 0}, {0, -1}, {-1, 0}, {0, 4}};
    for(const auto& sel : bad)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "mx16_bad_sel") == ROCKE_OK);
        Mx16Params p = mx16_params(&b);
        CHECK_THROWS(rocke_b_mfma_scale_f32_16x16x128_f8f6f4_sel(
                         &b, p.a, p.b, p.c, p.sa, p.sb, sel[0], sel[1]),
                     "scale_sel must be two byte indices in 0..3");
        rocke_ir_builder_free(&b);
    }
    // scale_sel on an unscaled atom.
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "f16_scale_sel") == ROCKE_OK);
    auto* a = rocke_b_param(&b, "a", rocke_vector_type(&b, rocke_f16(), 8), nullptr);
    auto* bb = rocke_b_param(&b, "b", rocke_vector_type(&b, rocke_f16(), 8), nullptr);
    auto* c = rocke_b_param(&b, "c", rocke_vector_type(&b, rocke_f32(), 4), nullptr);
    CHECK_THROWS(rocke_b_mma_scale_sel(&b, "mfma_f32_16x16x32_f16", a, bb, c, nullptr, 0, 0, 0),
                 "scale_sel applies only to a scaled MFMA atom");
    rocke_ir_builder_free(&b);
    return 0;
}

static int test_builder_rejects_bad_mx16_operands()
{
    for(int variant = 0; variant < 2; ++variant)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "mx16_bad_operands") == ROCKE_OK);
        Mx16Params p = variant == 0 ? mx16_params(&b, 16) : mx16_params(&b, 4, rocke_f32());
        rocke_value_t* extra[] = {p.sa, p.sb};
        CHECK_THROWS(rocke_b_mma(&b, kMx16, p.a, p.b, p.c, extra, 2),
                     "requires A and B of 32 bytes");
        rocke_ir_builder_free(&b);
    }
    return 0;
}

static int test_lowerings_refuse()
{
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "mx16_gfx942") == ROCKE_OK);
        build_mx16(&b, 0, 0);
        CHECK(expect_lowerings_fail(&b, "gfx942", ROCKE_ERR_NOTIMPL, "requires gfx950") == 0);
        rocke_ir_builder_free(&b);
    }
    // A scale_sel attr that bypassed the builder (e.g. parsed IR).
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "mx16_injected_sel") == ROCKE_OK);
        rocke_op_t* mma = build_mx16(&b, 0, 0);
        const int64_t bad[2] = {4, 0};
        rocke_attr_set_int_list(&b, &mma->attrs, "scale_sel", bad, 2);
        CHECK(expect_lowerings_fail(
                  &b, "gfx950", ROCKE_ERR_VALUE, "scale_sel must be two byte indices in 0..3")
              == 0);
        rocke_ir_builder_free(&b);
    }
    // Formats other than fp8e4m3 are lowered by Rust only.
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "mx16_fp6") == ROCKE_OK);
        rocke_op_t* mma = build_mx16(&b, 0, 0);
        rocke_attr_set_str(&b, &mma->attrs, "a_format", "fp6e2m3");
        CHECK(expect_lowerings_fail(&b, "gfx950", ROCKE_ERR_VALUE, "(fp8e4m3 only)") == 0);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

static int test_lds_dma_offset()
{
    for(int addr_form = 0; addr_form < 2; ++addr_form)
    {
        for(int offset : {0, 64, 4095, -1, 4096})
        {
            rocke_ir_builder_t b;
            CHECK(rocke_ir_builder_init(&b, "lds_dma_offset") == ROCKE_OK);
            auto* p = rocke_b_param(&b, "p", rocke_ptr_type(&b, rocke_f16(), "global"), nullptr);
            auto* rsrc = rocke_b_buffer_rsrc(&b, p, rocke_b_const_i32(&b, 4096));
            const int shape[2] = {64, 64};
            auto* smem = rocke_b_smem_alloc(&b, rocke_f16(), shape, 2, "stage");
            auto* lds = rocke_b_smem_addr_of(&b, smem);
            auto* zero = rocke_b_const_i32(&b, 0);
            auto build = [&] {
                if(addr_form)
                    rocke_b_async_buffer_load_lds_addr_off(&b, rsrc, lds, zero, zero, 1, 2, offset);
                else
                    rocke_b_async_buffer_load_lds_off(&b, rsrc, lds, zero, zero, 1, 2, offset);
            };
            if(offset < 0 || offset > 4095)
            {
                CHECK_THROWS(build(), "offset must be 0..4095 bytes");
                rocke_ir_builder_free(&b);
                continue;
            }
            build();
            auto* body = b.kernel->body;
            const rocke_attr_value_t* attr
                = rocke_attr_get(&body->ops[body->num_ops - 1]->attrs, "offset");
            CHECK(offset == 0 ? attr == nullptr
                              : attr && attr->kind == ROCKE_ATTR_INT && attr->u.i == offset);
            rocke_b_ret(&b);
            char needle[32];
            std::snprintf(needle, sizeof(needle), "i32 %d, i32 2)", offset);
            for(int f = 0; f < rocke_llvm_flavor_count(); ++f)
            {
                char* ll = nullptr;
                auto flavor = rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(f));
                CHECK(rocke_lower_kernel_to_llvm(b.kernel, flavor, "gfx950", &ll) == ROCKE_OK);
                CHECK(std::strstr(ll, "@llvm.amdgcn.raw.ptr.buffer.load.lds("));
                CHECK(std::strstr(ll, needle));
                std::free(ll);
            }
            rocke_ir_builder_free(&b);
        }
    }
    return 0;
}

int main()
{
    CHECK(test_scale_sel_lowers() == 0);
    CHECK(test_builder_rejects_bad_scale_sel() == 0);
    CHECK(test_builder_rejects_bad_mx16_operands() == 0);
    CHECK(test_lowerings_refuse() == 0);
    CHECK(test_lds_dma_offset() == 0);
    return 0;
}
