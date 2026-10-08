// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/future_intrinsic_lowering.cpp -- C++ parity for the Python
 * TestNewTargetIntrinsics gate.
 *
 * One case per intrinsic added for the LLVM 23 / future-operator surface. Each
 * case builds the smallest kernel that emits a single intrinsic and pins the
 * full `declare` plus call-site text, so a signature or immediate-encoding
 * regression names the intrinsic that broke. Asserting only that a mangled name
 * appears somewhere in the module cannot tell a correct call from one with the
 * wrong operand order, types, or immediates.
 *
 * The expected strings must stay byte-identical to the Python engine's -- that
 * equality is the parity contract this gate exists to defend.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/strbuf.h"

namespace
{

int g_failures = 0;
const char* g_case = "";

void fail(const char* what, int line)
{
    fprintf(stderr, "FAIL [%s]: %s (%s:%d)\n", g_case, what, __FILE__, line);
    ++g_failures;
}

void expect_contains(const std::string& ir, const char* needle, int line)
{
    if(ir.find(needle) == std::string::npos)
        fail(needle, line);
}

void expect_count(const std::string& ir, const char* needle, size_t want, int line)
{
    size_t seen = 0;
    for(size_t at = ir.find(needle); at != std::string::npos; at = ir.find(needle, at + 1))
        ++seen;
    if(seen != want)
    {
        char msg[512];
        snprintf(
            msg, sizeof(msg), "expected %zu occurrence(s) of \"%s\", saw %zu", want, needle, seen);
        fail(msg, line);
    }
}

#define EXPECT_IR(ir, needle) expect_contains((ir), (needle), __LINE__)
#define EXPECT_IR_COUNT(ir, needle, n) expect_count((ir), (needle), (n), __LINE__)
#define EXPECT_NO_IR(ir, needle)                   \
    do                                             \
    {                                              \
        if((ir).find(needle) != std::string::npos) \
            fail("unexpected: " needle, __LINE__); \
    } while(0)

/* Build a single-intrinsic kernel and return its lowered LLVM IR text. */
template <typename BuildFn>
std::string lower_one(const char* name,
                      BuildFn build,
                      const char* arch = "gfx950",
                      rocke_llvm_flavor_t flavor = ROCKE_LLVM_FLAVOR_AUTO)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        fail("rocke_ir_builder_init", __LINE__);
        return std::string();
    }
    build(&b);
    rocke_b_ret(&b);

    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    const rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
        rocke_ir_builder_kernel(&b), flavor, arch, &ll, err, sizeof(err));
    std::string ir;
    if(st != ROCKE_OK || ll == nullptr)
    {
        char msg[ROCKE_ERR_MSG_CAP + 64];
        snprintf(msg, sizeof(msg), "lower failed (status %d): %s", (int)st, err);
        fail(msg, __LINE__);
    }
    else
        ir.assign(ll);
    std::free(ll);
    rocke_ir_builder_free(&b);
    return ir;
}

/* Build the same kernel through the public HIP-source lowerer. */
template <typename BuildFn>
std::string lower_one_hip(const char* name, BuildFn build, const char* arch)
{
    rocke_ir_builder_t b;
    rocke_strbuf_t out;
    rocke_lower_hip_opts_t opts{};
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        fail("rocke_ir_builder_init", __LINE__);
        return std::string();
    }
    build(&b);
    rocke_b_ret(&b);
    rocke_strbuf_init(&out, 0);
    opts.arch = arch;
    const rocke_status_t st
        = rocke_lower_kernel_to_hip(&b, rocke_ir_builder_kernel(&b), &opts, &out);
    std::string hip;
    if(st != ROCKE_OK)
    {
        char msg[128];
        snprintf(msg, sizeof(msg), "HIP lower failed (status %d)", (int)st);
        fail(msg, __LINE__);
    }
    else
        hip.assign(rocke_strbuf_cstr(&out));
    rocke_strbuf_free(&out);
    rocke_ir_builder_free(&b);
    return hip;
}

rocke_value_t* global_ptr_param(rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem);

void case_gfx1250_standalone_bridge()
{
    const std::string ir = lower_one(
        "gfx1250_bridge",
        [](rocke_ir_builder_t* b) {
            const int shape[] = {1};
            rocke_value_t* dst = global_ptr_param(b, "dst", rocke_i16());
            rocke_value_t* smem = rocke_b_smem_alloc(b, rocke_i64(), shape, 1, "barrier");
            rocke_value_t* local = rocke_b_smem_addr_of(b, smem);
            rocke_value_t* members = rocke_b_const_i32(b, 2);
            rocke_value_t* d4 = rocke_b_zero_vec(b, rocke_i32(), 4);
            rocke_value_t* d8 = rocke_b_zero_vec(b, rocke_i32(), 8);

            rocke_b_s_wait_tensorcnt(b, 3);
            rocke_b_s_barrier_signal(b, 1);
            rocke_b_s_barrier_wait(b, 2);
            rocke_b_s_barrier_init(b, local, members);
            rocke_b_s_barrier_signal_var(b, local, members);
            rocke_b_s_barrier_join(b, local);
            rocke_b_s_wakeup_barrier(b, local);
            rocke_b_s_barrier_leave(b, 4);
            rocke_b_s_delay_alu(b, 0x1234);
            rocke_b_s_wait_alu(b, 0x2345);
            rocke_b_s_clause(b, 0x3456);
            rocke_b_s_wait_xcnt(b, 0x4567);
            rocke_b_global_store_async_from_lds(b, dst, local, 16, -4, 31);
            rocke_b_global_load_tr16_b128(b, dst, rocke_i16());
            rocke_b_tensor_load_to_lds(b, d4, d8, d4, d4, d8, 5);
            rocke_b_tensor_store_from_lds(b, d4, d8, d4, d4, d8, 6);
        },
        "gfx1250",
        ROCKE_LLVM_FLAVOR_LLVM23);

    EXPECT_IR(ir, "call void @llvm.amdgcn.s.wait.tensorcnt(i16 3)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.barrier.signal(i32 1)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.wakeup.barrier(ptr addrspace(3)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.barrier.leave(i16 4)");
    EXPECT_IR(ir, "call void asm sideeffect \"s_delay_alu 4660\", \"\"()");
    EXPECT_IR(ir, "call void asm sideeffect \"s_wait_xcnt 17767\", \"\"()");
    EXPECT_IR(ir, "call void @llvm.amdgcn.global.store.async.from.lds.b128(");
    EXPECT_IR(ir, "call <8 x i16> @llvm.amdgcn.global.load.tr.b128.v8i16(");
    EXPECT_IR(ir, "call void @llvm.amdgcn.tensor.load.to.lds(");
    EXPECT_IR(ir, "call void @llvm.amdgcn.tensor.store.from.lds(");
}

/* Build one kernel and return the C++ engine's HIP source. */
template <typename BuildFn>
std::string lower_one_hip(const char* name, BuildFn build)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        fail("rocke_ir_builder_init", __LINE__);
        return std::string();
    }
    build(&b);
    rocke_b_ret(&b);

    rocke_strbuf_t out;
    if(rocke_strbuf_init(&out, 256) != 0)
    {
        fail("rocke_strbuf_init", __LINE__);
        rocke_ir_builder_free(&b);
        return std::string();
    }
    rocke_lower_hip_opts_t opts{};
    opts.include_prologue = false;
    opts.include_prologue_set = true;
    opts.arch = "gfx950";
    const rocke_status_t st
        = rocke_lower_kernel_to_hip(&b, rocke_ir_builder_kernel(&b), &opts, &out);
    std::string hip;
    if(st != ROCKE_OK)
        fail("rocke_lower_kernel_to_hip", __LINE__);
    else
        hip.assign(rocke_strbuf_cstr(&out));
    rocke_strbuf_free(&out);
    rocke_ir_builder_free(&b);
    return hip;
}
rocke_value_t* global_ptr_param(rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem)
{
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), nullptr);
}

/* ---- ds_swizzle (raw offset + XOR-butterfly encoding) ---- */
void case_ds_swizzle_raw_offset()
{
    const std::string ir = lower_one("dssw", [](rocke_ir_builder_t* b) {
        rocke_b_ds_swizzle(b, rocke_b_const_i32(b, 1), 0x041F);
    });
    EXPECT_IR(ir, "declare i32 @llvm.amdgcn.ds.swizzle(i32, i32 immarg)");
    /* The raw immediate reaches the call verbatim (0x041F == 1055). */
    EXPECT_IR(ir, "call i32 @llvm.amdgcn.ds.swizzle(i32 1, i32 1055)");
}

void case_ds_swizzle_xor()
{
    /* offset = (xor_mask << 10) | 0x1F -> (2 << 10) | 31 == 2079 (0x081F). */
    const std::string ir = lower_one("dsswx", [](rocke_ir_builder_t* b) {
        rocke_b_ds_swizzle_xor(b, rocke_b_const_i32(b, 1), 2);
    });
    EXPECT_IR(ir, "call i32 @llvm.amdgcn.ds.swizzle(i32 1, i32 2079)");
    EXPECT_IR(ir, "declare i32 @llvm.amdgcn.ds.swizzle(i32, i32 immarg)");
}

/* ---- quad_perm ---- */
void case_quad_perm()
{
    /* [1,0,3,2] -> 1 | (0 << 2) | (3 << 4) | (2 << 6) == 177. */
    const std::string ir = lower_one("qperm", [](rocke_ir_builder_t* b) {
        rocke_b_quad_perm(b, rocke_b_const_i32(b, 1), 1, 0, 3, 2);
    });
    EXPECT_IR(ir,
              "declare i32 @llvm.amdgcn.update.dpp.i32("
              "i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg)");
    EXPECT_IR(ir,
              "call i32 @llvm.amdgcn.update.dpp.i32("
              "i32 1, i32 1, i32 177, i32 15, i32 15, i1 true)");
}

void case_quad_perm_hip()
{
    const std::string hip = lower_one_hip("qperm_hip", [](rocke_ir_builder_t* b) {
        rocke_b_quad_perm(b, rocke_b_const_i32(b, 1), 1, 0, 3, 2);
    });
    EXPECT_IR(hip, "__builtin_amdgcn_update_dpp(c1, c1, 177, 15, 15, 1)");
}

void expect_quad_perm_rejected(const char* name, bool use_f32, int p0, int p1, int p2, int p3)
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, name);
    bool rejected = false;
    try
    {
        rocke_value_t* data = use_f32 ? rocke_b_const_f32(&b, 1.0) : rocke_b_const_i32(&b, 1);
        rocke_value_t* r = rocke_b_quad_perm(&b, data, p0, p1, p2, p3);
        rejected = (r == nullptr || rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE);
    }
    catch(...)
    {
        rejected = true;
    }
    if(!rejected)
        fail(name, __LINE__);
    rocke_ir_builder_free(&b);
}

void case_quad_perm_rejects_invalid_input()
{
    expect_quad_perm_rejected("quad_perm must reject selector -1", false, -1, 0, 3, 2);
    expect_quad_perm_rejected("quad_perm must reject selector 4", false, 1, 0, 3, 4);
    expect_quad_perm_rejected("quad_perm must reject non-i32 data", true, 1, 0, 3, 2);
}

void case_quad_perm_hip_rejects_missing_ctrl()
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "qperm_missing_ctrl");
    rocke_value_t* data = rocke_b_const_i32(&b, 1);
    rocke_value_t* operands[] = {data};
    const rocke_type_t* result_types[] = {rocke_i32()};
    rocke_b_op(&b,
               ROCKE_OP_TILE_QUAD_PERM,
               operands,
               1,
               result_types,
               1,
               nullptr,
               nullptr,
               0,
               "qperm",
               nullptr);
    rocke_b_ret(&b);

    rocke_strbuf_t out;
    rocke_strbuf_init(&out, 256);
    rocke_lower_hip_opts_t opts{};
    opts.include_prologue = false;
    opts.include_prologue_set = true;
    opts.arch = "gfx950";
    const rocke_status_t st
        = rocke_lower_kernel_to_hip(&b, rocke_ir_builder_kernel(&b), &opts, &out);
    if(st != ROCKE_ERR_KEY)
        fail("quad_perm HIP lowering must reject missing ctrl", __LINE__);
    rocke_strbuf_free(&out);
    rocke_ir_builder_free(&b);
}

/* Build a kernel whose quad_perm carries a raw, out-of-range `ctrl`. The
 * builder validates selectors, so this shape can only arrive from IR that
 * skipped it (deserialized, rewritten by a pass, hand-built) -- exactly the
 * case masking to eight bits used to swallow: 256 became 0 ([0,0,0,0], a
 * lane-0 broadcast) and -1 became 255 ([3,3,3,3]). Both are legal permutes,
 * so the kernel computed wrong numbers instead of failing. */
void build_quad_perm_raw_ctrl(rocke_ir_builder_t* b, int64_t ctrl)
{
    rocke_value_t* data = rocke_b_const_i32(b, 1);
    rocke_value_t* operands[] = {data};
    const rocke_type_t* result_types[] = {rocke_i32()};
    rocke_attr_map_t attrs;
    rocke_attr_map_init(&attrs);
    rocke_attr_set_int(b, &attrs, "ctrl", ctrl);
    rocke_b_op(b,
               ROCKE_OP_TILE_QUAD_PERM,
               operands,
               1,
               result_types,
               1,
               &attrs,
               nullptr,
               0,
               "qperm",
               nullptr);
    rocke_b_ret(b);
}

void expect_quad_perm_ctrl_rejected(int64_t ctrl)
{
    /* HIP lowerer. */
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "qperm_bad_ctrl_hip");
        build_quad_perm_raw_ctrl(&b, ctrl);

        rocke_strbuf_t out;
        rocke_strbuf_init(&out, 256);
        rocke_lower_hip_opts_t opts{};
        opts.include_prologue = false;
        opts.include_prologue_set = true;
        opts.arch = "gfx950";
        const rocke_status_t st
            = rocke_lower_kernel_to_hip(&b, rocke_ir_builder_kernel(&b), &opts, &out);
        if(st != ROCKE_ERR_VALUE)
            fail("quad_perm HIP lowering must reject out-of-range ctrl", __LINE__);
        rocke_strbuf_free(&out);
        rocke_ir_builder_free(&b);
    }
    /* LLVM lowerer. */
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "qperm_bad_ctrl_ll");
        build_quad_perm_raw_ctrl(&b, ctrl);

        char* ll = nullptr;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = '\0';
        const rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            rocke_ir_builder_kernel(&b), ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &ll, err, sizeof(err));
        if(st != ROCKE_ERR_VALUE)
            fail("quad_perm LLVM lowering must reject out-of-range ctrl", __LINE__);
        std::free(ll);
        rocke_ir_builder_free(&b);
    }
}

void case_quad_perm_rejects_out_of_range_ctrl()
{
    expect_quad_perm_ctrl_rejected(256);
    expect_quad_perm_ctrl_rejected(-1);
}

/* ---- mov_dpp8 ---- */
void case_mov_dpp8_i32()
{
    const std::string ir = lower_one("dpp8i", [](rocke_ir_builder_t* b) {
        rocke_b_mov_dpp8(b, rocke_b_const_i32(b, 1), 0x765432);
    });
    EXPECT_IR(ir, "declare i32 @llvm.amdgcn.mov.dpp8.i32(i32, i32 immarg)");
    /* 0x765432 == 7754802, passed through as the 24-bit lane-select imm. */
    EXPECT_IR(ir, "call i32 @llvm.amdgcn.mov.dpp8.i32(i32 1, i32 7754802)");
}

void case_mov_dpp8_f32()
{
    const std::string ir = lower_one("dpp8f", [](rocke_ir_builder_t* b) {
        rocke_b_mov_dpp8(b, rocke_b_const_f32(b, 1.0), 0x765432);
    });
    EXPECT_IR(ir, "declare float @llvm.amdgcn.mov.dpp8.f32(float, i32 immarg)");
    EXPECT_IR(ir, "call float @llvm.amdgcn.mov.dpp8.f32(float");
}

void case_mov_dpp8_both_types_coexist()
{
    /* Both variants used to declare a bare @llvm.amdgcn.mov.dpp8, so a kernel
     * using each type emitted two conflicting declares for one symbol. */
    const std::string ir = lower_one("dpp8_both", [](rocke_ir_builder_t* b) {
        rocke_b_mov_dpp8(b, rocke_b_const_i32(b, 1), 0x11);
        rocke_b_mov_dpp8(b, rocke_b_const_f32(b, 1.0), 0x11);
    });
    EXPECT_IR(ir, "declare i32 @llvm.amdgcn.mov.dpp8.i32(");
    EXPECT_IR(ir, "declare float @llvm.amdgcn.mov.dpp8.f32(");
    EXPECT_NO_IR(ir, "@llvm.amdgcn.mov.dpp8(");
}

/* ---- wave_reduce ---- */
void case_wave_reduce()
{
    struct Variant
    {
        const char* op;
        const char* llvm_ty;
        const char* suffix;
        bool is_float;
    };
    static const Variant variants[] = {
        {"fmax", "float", "f32", true},
        {"fadd", "float", "f32", true},
        {"add", "i32", "i32", false},
        {"max", "i32", "i32", false},
        {"min", "i32", "i32", false},
    };

    for(const Variant& v : variants)
    {
        const std::string ir = lower_one("wred", [&v](rocke_ir_builder_t* b) {
            rocke_value_t* x = v.is_float ? rocke_b_const_f32(b, 1.0) : rocke_b_const_i32(b, 1);
            rocke_b_wave_reduce(b, x, v.op, 0);
        });
        char buf[256];
        snprintf(buf,
                 sizeof(buf),
                 "declare %s @llvm.amdgcn.wave.reduce.%s.%s(%s, i32 immarg)",
                 v.llvm_ty,
                 v.op,
                 v.suffix,
                 v.llvm_ty);
        EXPECT_IR(ir, buf);
        snprintf(buf,
                 sizeof(buf),
                 "call %s @llvm.amdgcn.wave.reduce.%s.%s(%s",
                 v.llvm_ty,
                 v.op,
                 v.suffix,
                 v.llvm_ty);
        EXPECT_IR(ir, buf);
        /* Trailing i32 is the strategy immediate (0 == default). */
        EXPECT_IR(ir, ", i32 0)");
    }
}

void case_wave_reduce_strategy()
{
    const std::string ir = lower_one("wred_strat", [](rocke_ir_builder_t* b) {
        rocke_b_wave_reduce(b, rocke_b_const_f32(b, 1.0), "fmax", 2);
    });
    EXPECT_IR(ir, "@llvm.amdgcn.wave.reduce.fmax.f32(float");
    EXPECT_IR(ir, ", i32 2)");
}

/* ---- readlane / writelane ---- */
void case_readlane()
{
    const std::string i32_ir = lower_one("rlane_i32", [](rocke_ir_builder_t* b) {
        rocke_b_readlane(b, rocke_b_const_i32(b, 7), rocke_b_const_i32(b, 0));
    });
    EXPECT_IR(i32_ir, "declare i32 @llvm.amdgcn.readlane.i32(i32, i32)");
    EXPECT_IR(i32_ir, "call i32 @llvm.amdgcn.readlane.i32(i32 7, i32 0)");

    const std::string f32_ir = lower_one("rlane_f32", [](rocke_ir_builder_t* b) {
        rocke_b_readlane(b, rocke_b_const_f32(b, 1.0), rocke_b_const_i32(b, 0));
    });
    EXPECT_IR(f32_ir, "declare float @llvm.amdgcn.readlane.f32(float, i32)");
    EXPECT_IR(f32_ir, "call float @llvm.amdgcn.readlane.f32(float");
}

void case_writelane()
{
    const std::string ir = lower_one("wlane", [](rocke_ir_builder_t* b) {
        rocke_b_writelane(
            b, rocke_b_const_i32(b, 7), rocke_b_const_i32(b, 0), rocke_b_const_i32(b, 9));
    });
    EXPECT_IR(ir, "declare i32 @llvm.amdgcn.writelane.i32(i32, i32, i32)");
    /* Operand order is (uniform_val, lane, passthrough). */
    EXPECT_IR(ir, "call i32 @llvm.amdgcn.writelane.i32(i32 7, i32 0, i32 9)");
}

/* ---- permlane16 / permlane64 / permlane32_swap ---- */
void case_permlane16()
{
    const std::string ir = lower_one("pl16", [](rocke_ir_builder_t* b) {
        rocke_b_permlane16(b,
                           rocke_b_const_i32(b, 0),
                           rocke_b_const_i32(b, 1),
                           rocke_b_const_i32(b, 2),
                           rocke_b_const_i32(b, 3),
                           false,
                           false);
    });
    /* The data type is an overloaded position, so the mangled name carries its
     * suffix. LLVM accepts the bare "permlane16" and auto-upgrades it, but
     * rewrites it to this on the way out, so the bare form is the one that
     * would not survive a round trip. */
    EXPECT_IR(ir,
              "declare i32 @llvm.amdgcn.permlane16.i32"
              "(i32, i32, i32, i32, i1 immarg, i1 immarg)");
    EXPECT_IR(ir,
              "call i32 @llvm.amdgcn.permlane16.i32"
              "(i32 0, i32 1, i32 2, i32 3, i1 false, i1 false)");
}

void case_permlane16_flags()
{
    const std::string ir = lower_one("pl16_flags", [](rocke_ir_builder_t* b) {
        rocke_b_permlane16(b,
                           rocke_b_const_i32(b, 0),
                           rocke_b_const_i32(b, 1),
                           rocke_b_const_i32(b, 2),
                           rocke_b_const_i32(b, 3),
                           true,
                           true);
    });
    EXPECT_IR(ir,
              "call i32 @llvm.amdgcn.permlane16.i32"
              "(i32 0, i32 1, i32 2, i32 3, i1 true, i1 true)");
}

void case_permlane64()
{
    const std::string ir = lower_one(
        "pl64", [](rocke_ir_builder_t* b) { rocke_b_permlane64(b, rocke_b_const_i32(b, 1)); });
    EXPECT_IR(ir, "declare i32 @llvm.amdgcn.permlane64.i32(i32)");
    EXPECT_IR(ir, "call i32 @llvm.amdgcn.permlane64.i32(i32 1)");
}

void case_permlane32_swap()
{
    const std::string ir = lower_one("psw", [](rocke_ir_builder_t* b) {
        rocke_value_t* lo = nullptr;
        rocke_value_t* hi = nullptr;
        rocke_b_permlane32_swap(b, rocke_b_const_i32(b, 1), rocke_b_const_i32(b, 2), &lo, &hi);
    });
    /* No name suffix: unlike its permlane siblings this one is not overloaded.
     * The flags are still immarg. */
    EXPECT_IR(ir,
              "declare { i32, i32 } @llvm.amdgcn.permlane32.swap"
              "(i32, i32, i1 immarg, i1 immarg)");
    EXPECT_IR(ir,
              "call { i32, i32 } @llvm.amdgcn.permlane32.swap"
              "(i32 1, i32 2, i1 false, i1 false)");
    /* Both halves are extracted from the returned struct. */
    EXPECT_IR_COUNT(ir, "extractvalue { i32, i32 }", 2);
}

/* ---- alignbyte / s_wqm ---- */
void case_alignbyte()
{
    const std::string ir = lower_one("algn", [](rocke_ir_builder_t* b) {
        rocke_b_alignbyte(
            b, rocke_b_const_i32(b, 1), rocke_b_const_i32(b, 2), rocke_b_const_i32(b, 8));
    });
    EXPECT_IR(ir, "declare i32 @llvm.amdgcn.alignbyte(i32, i32, i32)");
    EXPECT_IR(ir, "call i32 @llvm.amdgcn.alignbyte(i32 1, i32 2, i32 8)");
}

void case_s_wqm()
{
    const std::string i32_ir = lower_one(
        "wqm_i32", [](rocke_ir_builder_t* b) { rocke_b_s_wqm(b, rocke_b_const_i32(b, 0xF)); });
    /* Result and operand are separately overloaded, so the canonical name
     * repeats the type: s.wqm.i32.i32. */
    EXPECT_IR(i32_ir, "declare i32 @llvm.amdgcn.s.wqm.i32.i32(i32)");
    EXPECT_IR(i32_ir, "call i32 @llvm.amdgcn.s.wqm.i32.i32(i32 15)");

    const std::string i64_ir = lower_one(
        "wqm_i64", [](rocke_ir_builder_t* b) { rocke_b_s_wqm(b, rocke_b_const_i64(b, 0xF)); });
    EXPECT_IR(i64_ir, "declare i64 @llvm.amdgcn.s.wqm.i64.i64(i64)");
    EXPECT_IR(i64_ir, "call i64 @llvm.amdgcn.s.wqm.i64.i64(i64 15)");
}

/* ---- av.load / av.store (agent-scope 128-bit vector mem) ---- */
void case_av_load_b128()
{
    const std::string ir = lower_one("avld", [](rocke_ir_builder_t* b) {
        rocke_b_av_load_b128(b, global_ptr_param(b, "p", rocke_i32()));
    });
    /* A global pointer param is ptr addrspace(1) in the header, so the
     * overload -- declare and call -- has to be the p1 one. */
    EXPECT_IR(ir, "declare <4 x i32> @llvm.amdgcn.av.load.b128.p1(ptr addrspace(1), metadata)");
    EXPECT_IR(ir, "call <4 x i32> @llvm.amdgcn.av.load.b128.p1(ptr addrspace(1) %p, metadata !3)");
    /* The scope operand must be backed by a real metadata node. */
    EXPECT_IR(ir, "!3 = !{!\"agent\"}");
}

void case_av_store_b128()
{
    const std::string ir = lower_one("avst", [](rocke_ir_builder_t* b) {
        rocke_value_t* p = global_ptr_param(b, "p", rocke_i32());
        rocke_b_av_store_b128(b, p, rocke_b_av_load_b128(b, p));
    });
    EXPECT_IR(ir,
              "declare void @llvm.amdgcn.av.store.b128.p1(ptr addrspace(1), <4 x i32>, metadata)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.av.store.b128.p1(ptr addrspace(1) %p, <4 x i32>");
    EXPECT_IR(ir, "metadata !3)");
    EXPECT_IR(ir, "!3 = !{!\"agent\"}");
}

/* ---- s_alloc_vgpr ---- */
void case_s_alloc_vgpr()
{
    const std::string ir
        = lower_one("valloc", [](rocke_ir_builder_t* b) { rocke_b_s_alloc_vgpr(b, 8); });
    EXPECT_IR(ir, "declare i1 @llvm.amdgcn.s.alloc.vgpr(i32)");
    EXPECT_IR(ir, "call i1 @llvm.amdgcn.s.alloc.vgpr(i32 8)");
    /* The intrinsic returns i1; the IR value is an i32, so a zext is required. */
    EXPECT_IR(ir, "zext i1 ");
}

/* ---- async markers / event waits / prefetch ---- */
void case_asyncmark()
{
    const std::string ir = lower_one("amark", [](rocke_ir_builder_t* b) { rocke_b_asyncmark(b); });
    EXPECT_IR(ir, "declare void @llvm.amdgcn.asyncmark()");
    EXPECT_IR(ir, "call void @llvm.amdgcn.asyncmark()");
}

void case_wait_asyncmark()
{
    const std::string ir
        = lower_one("await", [](rocke_ir_builder_t* b) { rocke_b_wait_asyncmark(b, 3); });
    EXPECT_IR(ir, "declare void @llvm.amdgcn.wait.asyncmark(i16 immarg)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.wait.asyncmark(i16 3)");
}

void case_s_wait_event()
{
    const std::string ir
        = lower_one("sevt", [](rocke_ir_builder_t* b) { rocke_b_s_wait_event(b, 1); });
    EXPECT_IR(ir, "declare void @llvm.amdgcn.s.wait.event(i16 immarg)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.wait.event(i16 1)");
}

void case_s_prefetch_inst()
{
    const std::string ir = lower_one("sprefetch", [](rocke_ir_builder_t* b) {
        rocke_b_s_prefetch_inst(
            b, global_ptr_param(b, "code", rocke_i32()), rocke_b_const_i32(b, 64));
    });
    /* The operand is llvm_anyptr_ty, so the call and its declare have to name
     * the pointer's real space. A bare `ptr` for this addrspace(1) param is
     * what LLVM rejects with "'%code' defined with type 'ptr addrspace(1)' but
     * expected 'ptr'". */
    EXPECT_IR(ir, "declare void @llvm.amdgcn.s.prefetch.inst.p1(ptr addrspace(1), i32)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.prefetch.inst.p1(ptr addrspace(1) %code, i32 64)");
}

/* ---- gfx1250 data prefetch (Python tests/core/test_gfx1250_prefetch.py) ---- */

/* Lower a kernel that is expected to be rejected; return the status and fill
 * `err` with the lowerer's message. */
template <typename BuildFn>
rocke_status_t lower_expect_error(const char* name,
                                  BuildFn build,
                                  const char* arch,
                                  rocke_llvm_flavor_t flavor,
                                  std::string* err_out)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        fail("rocke_ir_builder_init", __LINE__);
        return ROCKE_OK;
    }
    build(&b);
    rocke_b_ret(&b);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    const rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
        rocke_ir_builder_kernel(&b), flavor, arch, &ll, err, sizeof(err));
    err_out->assign(err);
    std::free(ll);
    rocke_ir_builder_free(&b);
    return st;
}

rocke_value_t* private_ptr_param(rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem)
{
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "private"), nullptr);
}

/* Mirrors _build_all in the Python test: every op once, each pointer space
 * s_prefetch_data accepts, and non-default immediates so a dropped attr shows. */
void build_gfx1250_data_prefetch(rocke_ir_builder_t* b)
{
    rocke_param_opts_t readonly{};
    readonly.readonly = true;
    readonly.readonly_set = true;
    readonly.align = 16;
    readonly.align_set = true;
    rocke_param_opts_t constant{};
    constant.addr_space = "constant";

    rocke_value_t* src
        = rocke_b_param(b, "src", rocke_ptr_type(b, rocke_f32(), "global"), &readonly);
    rocke_value_t* table
        = rocke_b_param(b, "table", rocke_ptr_type(b, rocke_i32(), "global"), &constant);
    rocke_value_t* flat = private_ptr_param(b, "flat", rocke_f32());
    rocke_value_t* nbytes = rocke_b_param(b, "nbytes", rocke_i32(), nullptr);
    rocke_value_t* lines = rocke_b_const_i32(b, 4);

    rocke_b_enable_scalar_prefetch(b);
    rocke_b_s_prefetch_data(b, src, lines);
    rocke_b_s_prefetch_data(b, table, lines);
    rocke_b_s_prefetch_data(b, flat, lines);
    rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, src, nbytes);
    rocke_b_s_buffer_prefetch_data(b, rsrc, lines, /*offset=*/256);
    rocke_b_global_prefetch(b, src, /*cachepolicy=*/3);
    rocke_b_flat_prefetch(b, flat, /*cachepolicy=*/5);
}

void case_gfx1250_data_prefetch()
{
    const std::string ir = lower_one(
        "gfx1250_prefetch", build_gfx1250_data_prefetch, "gfx1250", ROCKE_LLVM_FLAVOR_LLVM23);

    /* hwreg(MODE=1, offset=24, size=1) == 1 | (24 << 6) == 1537. */
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.setreg(i32 1537, i32 1)");
    /* s.prefetch.data is llvm_anyptr_ty: the overload follows the param's
     * header type, including the constant-space override on `table`. */
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.prefetch.data.p1(ptr addrspace(1) %src, i32 4)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.prefetch.data.p4(ptr addrspace(4) %table, i32 4)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.prefetch.data.p0(ptr %flat, i32 4)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.buffer.prefetch.data(ptr addrspace(8) %");
    EXPECT_IR(ir, ", i32 256, i32 4)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %src, i32 3)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.flat.prefetch(ptr %flat, i32 5)");

    EXPECT_IR_COUNT(ir, "declare void @llvm.amdgcn.s.setreg(i32 immarg, i32)", 1);
    EXPECT_IR_COUNT(ir, "declare void @llvm.amdgcn.s.prefetch.data.p0(ptr, i32)", 1);
    EXPECT_IR_COUNT(ir, "declare void @llvm.amdgcn.s.prefetch.data.p1(ptr addrspace(1), i32)", 1);
    EXPECT_IR_COUNT(ir, "declare void @llvm.amdgcn.s.prefetch.data.p4(ptr addrspace(4), i32)", 1);
    EXPECT_IR_COUNT(ir,
                    "declare void @llvm.amdgcn.s.buffer.prefetch.data("
                    "ptr addrspace(8), i32 immarg, i32)",
                    1);
    EXPECT_IR_COUNT(
        ir, "declare void @llvm.amdgcn.global.prefetch(ptr addrspace(1), i32 immarg)", 1);
    EXPECT_IR_COUNT(ir, "declare void @llvm.amdgcn.flat.prefetch(ptr, i32 immarg)", 1);
}

void case_gfx1250_data_prefetch_declares_only_used()
{
    const std::string ir = lower_one(
        "only_global",
        [](rocke_ir_builder_t* b) {
            rocke_b_global_prefetch(b, global_ptr_param(b, "src", rocke_f32()), 0);
        },
        "gfx1250",
        ROCKE_LLVM_FLAVOR_LLVM23);
    EXPECT_IR(ir, "call void @llvm.amdgcn.global.prefetch(");
    EXPECT_NO_IR(ir, "@llvm.amdgcn.s.setreg");
    EXPECT_NO_IR(ir, "@llvm.amdgcn.s.prefetch.data");
    EXPECT_NO_IR(ir, "@llvm.amdgcn.s.buffer.prefetch");
    EXPECT_NO_IR(ir, "@llvm.amdgcn.flat.prefetch");
}

void case_gfx1250_data_prefetch_distinct_from_inst()
{
    const std::string ir = lower_one(
        "both",
        [](rocke_ir_builder_t* b) {
            rocke_value_t* src = global_ptr_param(b, "src", rocke_f32());
            rocke_value_t* n = rocke_b_const_i32(b, 2);
            rocke_b_s_prefetch_inst(b, src, n);
            rocke_b_s_prefetch_data(b, src, n);
        },
        "gfx1250",
        ROCKE_LLVM_FLAVOR_LLVM23);
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.prefetch.inst.p1(");
    EXPECT_IR(ir, "call void @llvm.amdgcn.s.prefetch.data.p1(");
}

/* Each op is gfx1250 + llvm23 only; the message names the op that tripped. */
void case_gfx1250_data_prefetch_gated()
{
    struct Gate
    {
        const char* name;
        void (*emit)(rocke_ir_builder_t*, rocke_value_t*, rocke_value_t*);
    };
    static const Gate gates[] = {
        {"s_setreg",
         [](rocke_ir_builder_t* b, rocke_value_t*, rocke_value_t*) {
             rocke_b_s_setreg(b, 1537, rocke_b_const_i32(b, 1));
         }},
        {"s_prefetch_data",
         [](rocke_ir_builder_t* b, rocke_value_t* s, rocke_value_t*) {
             rocke_b_s_prefetch_data(b, s, rocke_b_const_i32(b, 1));
         }},
        {"s_buffer_prefetch_data",
         [](rocke_ir_builder_t* b, rocke_value_t* s, rocke_value_t*) {
             rocke_b_s_buffer_prefetch_data(b,
                                            rocke_b_buffer_rsrc(b, s, rocke_b_const_i32(b, 64)),
                                            rocke_b_const_i32(b, 1),
                                            0);
         }},
        {"global_prefetch",
         [](rocke_ir_builder_t* b, rocke_value_t* s, rocke_value_t*) {
             rocke_b_global_prefetch(b, s, 0);
         }},
        {"flat_prefetch",
         [](rocke_ir_builder_t* b, rocke_value_t*, rocke_value_t* f) {
             rocke_b_flat_prefetch(b, f, 0);
         }},
    };
    for(const Gate& g : gates)
    {
        const auto build = [&g](rocke_ir_builder_t* b) {
            rocke_value_t* src = global_ptr_param(b, "src", rocke_f32());
            rocke_value_t* flat = private_ptr_param(b, "flat", rocke_f32());
            g.emit(b, src, flat);
        };
        const struct
        {
            const char* arch;
            rocke_llvm_flavor_t flavor;
            const char* suffix;
        } bad[] = {
            {"gfx950", ROCKE_LLVM_FLAVOR_LLVM22, " requires gfx1250"},
            {"gfx1250", ROCKE_LLVM_FLAVOR_LLVM22, " requires LLVM flavor llvm23"},
        };
        for(const auto& t : bad)
        {
            std::string err;
            const rocke_status_t st = lower_expect_error("gate", build, t.arch, t.flavor, &err);
            const std::string want = std::string(g.name) + t.suffix;
            if(st != ROCKE_ERR_VALUE || err.find(want) == std::string::npos)
            {
                char msg[ROCKE_ERR_MSG_CAP + 128];
                snprintf(msg,
                         sizeof(msg),
                         "%s on %s: status %d, err \"%s\"",
                         g.name,
                         t.arch,
                         (int)st,
                         err.c_str());
                fail(msg, __LINE__);
            }
        }
    }
}

/* Builder-side operand and immediate validation. */
void case_gfx1250_data_prefetch_builder_rejects()
{
    struct Reject
    {
        const char* what;
        void (*emit)(rocke_ir_builder_t*, rocke_value_t*, rocke_value_t*);
    };
    static const Reject rejects[] = {
        {"s_setreg simm16 65536",
         [](rocke_ir_builder_t* b, rocke_value_t*, rocke_value_t*) {
             rocke_b_s_setreg(b, 65536, rocke_b_const_i32(b, 1));
         }},
        {"s_setreg simm16 -1",
         [](rocke_ir_builder_t* b, rocke_value_t*, rocke_value_t*) {
             rocke_b_s_setreg(b, -1, rocke_b_const_i32(b, 1));
         }},
        {"s_setreg i64 value",
         [](rocke_ir_builder_t* b, rocke_value_t*, rocke_value_t*) {
             rocke_b_s_setreg(b, 1537, rocke_b_const_i64(b, 1));
         }},
        {"s_prefetch_data non-pointer",
         [](rocke_ir_builder_t* b, rocke_value_t*, rocke_value_t*) {
             rocke_b_s_prefetch_data(b, rocke_b_const_i64(b, 0), rocke_b_const_i32(b, 1));
         }},
        {"s_prefetch_data i64 length",
         [](rocke_ir_builder_t* b, rocke_value_t* s, rocke_value_t*) {
             rocke_b_s_prefetch_data(b, s, rocke_b_const_i64(b, 1));
         }},
        {"s_buffer_prefetch_data non-rsrc",
         [](rocke_ir_builder_t* b, rocke_value_t*, rocke_value_t*) {
             rocke_b_s_buffer_prefetch_data(
                 b, rocke_b_zero_vec(b, rocke_i32(), 8), rocke_b_const_i32(b, 1), 0);
         }},
        {"global_prefetch cachepolicy 32",
         [](rocke_ir_builder_t* b, rocke_value_t* s, rocke_value_t*) {
             rocke_b_global_prefetch(b, s, 32);
         }},
        {"global_prefetch flat pointer",
         [](rocke_ir_builder_t* b, rocke_value_t*, rocke_value_t* f) {
             rocke_b_global_prefetch(b, f, 0);
         }},
        {"flat_prefetch global pointer",
         [](rocke_ir_builder_t* b, rocke_value_t* s, rocke_value_t*) {
             rocke_b_flat_prefetch(b, s, 0);
         }},
    };
    for(const Reject& r : rejects)
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "bad");
        bool rejected = false;
        try
        {
            rocke_value_t* src = global_ptr_param(&b, "src", rocke_f32());
            rocke_value_t* flat = private_ptr_param(&b, "flat", rocke_f32());
            r.emit(&b, src, flat);
            rejected = rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE;
        }
        catch(...)
        {
            rejected = true;
        }
        if(!rejected)
            fail(r.what, __LINE__);
        rocke_ir_builder_free(&b);
    }
}

/* Serialized or pass-rewritten IR reaches the lowerer without the builder's
 * checks, so the lowerer re-validates immediates and pointer spaces. */
void emit_raw_op(rocke_ir_builder_t* b,
                 rocke_opcode_t opcode,
                 rocke_value_t* operand,
                 const char* attr,
                 int64_t value)
{
    rocke_attr_map_t attrs;
    rocke_attr_map_init(&attrs);
    rocke_attr_set_int(b, &attrs, attr, value);
    rocke_value_t* operands[] = {operand};
    rocke_b_op(b, opcode, operands, 1, nullptr, 0, &attrs, nullptr, 0, nullptr, nullptr);
}

void case_gfx1250_data_prefetch_lowerer_rechecks()
{
    struct Recheck
    {
        const char* want;
        void (*build)(rocke_ir_builder_t*);
    };
    static const Recheck rechecks[] = {
        {"global_prefetch cachepolicy must be in 0..31, got 40",
         [](rocke_ir_builder_t* b) {
             emit_raw_op(b,
                         ROCKE_OP_TILE_GLOBAL_PREFETCH,
                         global_ptr_param(b, "src", rocke_f32()),
                         "cachepolicy",
                         40);
         }},
        {"flat_prefetch cachepolicy must be in 0..31, got -1",
         [](rocke_ir_builder_t* b) {
             emit_raw_op(b,
                         ROCKE_OP_TILE_FLAT_PREFETCH,
                         private_ptr_param(b, "flat", rocke_f32()),
                         "cachepolicy",
                         -1);
         }},
        {"s_setreg simm16 must fit an unsigned i16",
         [](rocke_ir_builder_t* b) {
             emit_raw_op(b, ROCKE_OP_TILE_S_SETREG, rocke_b_const_i32(b, 1), "simm16", 70000);
         }},
        /* The builder sees a global PtrType, but the param ABI moved it to
         * constant space: the header says ptr addrspace(4), which
         * global.prefetch cannot take. */
        {"global_prefetch ptr must be a global pointer, got ptr addrspace(4)",
         [](rocke_ir_builder_t* b) {
             rocke_param_opts_t constant{};
             constant.addr_space = "constant";
             rocke_b_global_prefetch(
                 b,
                 rocke_b_param(b, "table", rocke_ptr_type(b, rocke_i32(), "global"), &constant),
                 0);
         }},
    };
    for(const Recheck& r : rechecks)
    {
        std::string err;
        const rocke_status_t st
            = lower_expect_error("recheck", r.build, "gfx1250", ROCKE_LLVM_FLAVOR_LLVM23, &err);
        if(st != ROCKE_ERR_VALUE || err.find(r.want) == std::string::npos)
        {
            char msg[ROCKE_ERR_MSG_CAP + 128];
            snprintf(msg,
                     sizeof(msg),
                     "want \"%s\", status %d, err \"%s\"",
                     r.want,
                     (int)st,
                     err.c_str());
            fail(msg, __LINE__);
        }
    }
}

/* ---- gfx1250 workgroup clusters ---- */

/* Mirrors _build_all in test_gfx1250_cluster.py: every read on every axis,
 * stored so the pure reads stay live, then one cluster barrier. */
void build_gfx1250_cluster(rocke_ir_builder_t* b)
{
    rocke_value_t* out = global_ptr_param(b, "out", rocke_i32());
    static const char* const axes[] = {"x", "y", "z"};
    int slot = 0;
    const auto store = [&](rocke_value_t* v) {
        rocke_b_global_store(b, out, rocke_b_const_i32(b, slot++), v, 4);
    };
    for(const char* axis : axes)
    {
        store(rocke_b_cluster_id(b, axis));
        store(rocke_b_cluster_workgroup_id(b, axis));
        store(rocke_b_cluster_workgroup_max_id(b, axis));
    }
    store(rocke_b_cluster_workgroup_flat_id(b));
    store(rocke_b_cluster_workgroup_max_flat_id(b));
    rocke_b_cluster_barrier(b);
}

void case_gfx1250_cluster()
{
    const std::string ir
        = lower_one("gfx1250_cluster", build_gfx1250_cluster, "gfx1250", ROCKE_LLVM_FLAVOR_LLVM23);

    static const char* const stems[]
        = {"cluster.id", "cluster.workgroup.id", "cluster.workgroup.max.id"};
    static const char* const axes[] = {"x", "y", "z"};
    for(const char* stem : stems)
    {
        for(const char* axis : axes)
        {
            char call[128];
            char decl[128];
            snprintf(call, sizeof(call), " = call i32 @llvm.amdgcn.%s.%s()", stem, axis);
            snprintf(decl, sizeof(decl), "declare i32 @llvm.amdgcn.%s.%s()", stem, axis);
            expect_count(ir, call, 1, __LINE__);
            expect_count(ir, decl, 1, __LINE__);
        }
    }
    EXPECT_IR_COUNT(ir, " = call i32 @llvm.amdgcn.cluster.workgroup.flat.id()", 1);
    EXPECT_IR_COUNT(ir, " = call i32 @llvm.amdgcn.cluster.workgroup.max.flat.id()", 1);
    EXPECT_IR_COUNT(ir, "declare i32 @llvm.amdgcn.cluster.workgroup.flat.id()", 1);
    EXPECT_IR_COUNT(ir, "declare i32 @llvm.amdgcn.cluster.workgroup.max.flat.id()", 1);
    EXPECT_IR_COUNT(ir, "declare void @llvm.amdgcn.s.cluster.barrier()", 1);
    /* The fences bracket the barrier with nothing in between. */
    EXPECT_IR(ir,
              "  fence syncscope(\"cluster\") release\n"
              "  call void @llvm.amdgcn.s.cluster.barrier()\n"
              "  fence syncscope(\"cluster\") acquire\n");
    /* The cluster barrier is not the workgroup split barrier. */
    EXPECT_NO_IR(ir, "@llvm.amdgcn.s.barrier.signal");
    EXPECT_NO_IR(ir, "@llvm.amdgcn.s.barrier.wait");
}

void case_gfx1250_cluster_size()
{
    const std::string ir = lower_one(
        "cluster_size",
        [](rocke_ir_builder_t* b) {
            rocke_value_t* out = global_ptr_param(b, "out", rocke_i32());
            rocke_b_global_store(b, out, rocke_b_const_i32(b, 0), rocke_b_cluster_size(b, "y"), 4);
        },
        "gfx1250",
        ROCKE_LLVM_FLAVOR_LLVM23);
    EXPECT_IR_COUNT(ir, " = call i32 @llvm.amdgcn.cluster.workgroup.max.id.y()", 1);
    EXPECT_IR(ir, " = add nsw i32 %cwmax");
    EXPECT_NO_IR(ir, "@llvm.amdgcn.cluster.id.");
    EXPECT_NO_IR(ir, "@llvm.amdgcn.s.cluster.barrier");
}

/* Each op is gfx1250 + llvm23 only; the message names the op that tripped. */
void case_gfx1250_cluster_gated()
{
    struct Gate
    {
        const char* name;
        void (*emit)(rocke_ir_builder_t*, rocke_value_t*);
    };
    static const Gate gates[] = {
        {"cluster_id",
         [](rocke_ir_builder_t* b, rocke_value_t* o) {
             rocke_b_global_store(b, o, rocke_b_const_i32(b, 0), rocke_b_cluster_id(b, "x"), 4);
         }},
        {"cluster_workgroup_id",
         [](rocke_ir_builder_t* b, rocke_value_t* o) {
             rocke_b_global_store(
                 b, o, rocke_b_const_i32(b, 0), rocke_b_cluster_workgroup_id(b, "y"), 4);
         }},
        {"cluster_workgroup_max_id",
         [](rocke_ir_builder_t* b, rocke_value_t* o) {
             rocke_b_global_store(
                 b, o, rocke_b_const_i32(b, 0), rocke_b_cluster_workgroup_max_id(b, "z"), 4);
         }},
        {"cluster_workgroup_flat_id",
         [](rocke_ir_builder_t* b, rocke_value_t* o) {
             rocke_b_global_store(
                 b, o, rocke_b_const_i32(b, 0), rocke_b_cluster_workgroup_flat_id(b), 4);
         }},
        {"cluster_workgroup_max_flat_id",
         [](rocke_ir_builder_t* b, rocke_value_t* o) {
             rocke_b_global_store(
                 b, o, rocke_b_const_i32(b, 0), rocke_b_cluster_workgroup_max_flat_id(b), 4);
         }},
        {"cluster_barrier",
         [](rocke_ir_builder_t* b, rocke_value_t*) { rocke_b_cluster_barrier(b); }},
    };
    for(const Gate& g : gates)
    {
        const auto build
            = [&g](rocke_ir_builder_t* b) { g.emit(b, global_ptr_param(b, "out", rocke_i32())); };
        const struct
        {
            const char* arch;
            rocke_llvm_flavor_t flavor;
            const char* suffix;
        } bad[] = {
            {"gfx950", ROCKE_LLVM_FLAVOR_LLVM22, " requires gfx1250"},
            {"gfx1201", ROCKE_LLVM_FLAVOR_LLVM23, " requires gfx1250"},
            {"gfx1250", ROCKE_LLVM_FLAVOR_LLVM22, " requires LLVM flavor llvm23"},
        };
        for(const auto& t : bad)
        {
            std::string err;
            const rocke_status_t st = lower_expect_error("gate", build, t.arch, t.flavor, &err);
            const std::string want = std::string(g.name) + t.suffix;
            if(st != ROCKE_ERR_VALUE || err.find(want) == std::string::npos)
            {
                char msg[ROCKE_ERR_MSG_CAP + 128];
                snprintf(msg,
                         sizeof(msg),
                         "%s on %s: status %d, err \"%s\"",
                         g.name,
                         t.arch,
                         (int)st,
                         err.c_str());
                fail(msg, __LINE__);
            }
        }
    }
}

/* Builder axis validation; the text matches the Python builder. */
void case_gfx1250_cluster_builder_rejects()
{
    struct Reject
    {
        const char* want;
        rocke_value_t* (*emit)(rocke_ir_builder_t*);
    };
    static const Reject rejects[] = {
        {"cluster_id axis must be x, y, or z, got 'w'",
         [](rocke_ir_builder_t* b) { return rocke_b_cluster_id(b, "w"); }},
        {"cluster_workgroup_id axis must be x, y, or z, got 'X'",
         [](rocke_ir_builder_t* b) { return rocke_b_cluster_workgroup_id(b, "X"); }},
        {"cluster_workgroup_max_id axis must be x, y, or z, got ''",
         [](rocke_ir_builder_t* b) { return rocke_b_cluster_workgroup_max_id(b, ""); }},
        {"cluster_workgroup_max_id axis must be x, y, or z, got 'xy'",
         [](rocke_ir_builder_t* b) { return rocke_b_cluster_size(b, "xy"); }},
    };
    for(const Reject& r : rejects)
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "bad");
        bool rejected = false;
        std::string err;
        try
        {
            rocke_value_t* v = r.emit(&b);
            rejected = v == nullptr && rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE;
            const char* msg = rocke_ir_builder_error(&b);
            err = msg ? msg : "";
        }
        catch(const std::exception& e)
        {
            /* Builder errors surface as ckc::ValueError inside the library. */
            rejected = true;
            err = e.what();
        }
        if(!rejected || err.find(r.want) == std::string::npos)
        {
            char msg[ROCKE_ERR_MSG_CAP + 128];
            snprintf(msg, sizeof(msg), "want \"%s\", err \"%s\"", r.want, err.c_str());
            fail(msg, __LINE__);
        }
        rocke_ir_builder_free(&b);
    }
}

/* Serialized IR reaches the lowerer without the builder's axis check. */
void case_gfx1250_cluster_lowerer_rechecks()
{
    std::string err;
    const rocke_status_t st = lower_expect_error(
        "recheck",
        [](rocke_ir_builder_t* b) {
            rocke_attr_map_t attrs;
            rocke_attr_map_init(&attrs);
            rocke_attr_set_str(b, &attrs, "axis", "w");
            const rocke_type_t* i32 = rocke_i32();
            rocke_op_t* op = rocke_b_op(b,
                                        ROCKE_OP_GPU_CLUSTER_WORKGROUP_ID,
                                        nullptr,
                                        0,
                                        &i32,
                                        1,
                                        &attrs,
                                        nullptr,
                                        0,
                                        "cwid",
                                        nullptr);
            rocke_b_global_store(b,
                                 global_ptr_param(b, "out", rocke_i32()),
                                 rocke_b_const_i32(b, 0),
                                 rocke_op_result(b, op),
                                 4);
        },
        "gfx1250",
        ROCKE_LLVM_FLAVOR_LLVM23,
        &err);
    const char* want = "cluster_workgroup_id axis must be x, y, or z, got 'w'";
    if(st != ROCKE_ERR_VALUE || err.find(want) == std::string::npos)
    {
        char msg[ROCKE_ERR_MSG_CAP + 128];
        snprintf(
            msg, sizeof(msg), "want \"%s\", status %d, err \"%s\"", want, (int)st, err.c_str());
        fail(msg, __LINE__);
    }
}

/* The reads are pure (DCE/CSE-able); the barrier is not. Must match Python
 * PURE_OP_NAMES. */
void case_gfx1250_cluster_purity()
{
    static const rocke_opcode_t pure[] = {
        ROCKE_OP_GPU_CLUSTER_ID,
        ROCKE_OP_GPU_CLUSTER_WORKGROUP_ID,
        ROCKE_OP_GPU_CLUSTER_WORKGROUP_MAX_ID,
        ROCKE_OP_GPU_CLUSTER_WORKGROUP_FLAT_ID,
        ROCKE_OP_GPU_CLUSTER_WORKGROUP_MAX_FLAT_ID,
    };
    for(rocke_opcode_t op : pure)
        if(!rocke_opcode_is_pure(op))
            fail(rocke_opcode_name(op), __LINE__);
    if(rocke_opcode_is_pure(ROCKE_OP_TILE_CLUSTER_BARRIER))
        fail("tile.cluster_barrier must not be pure", __LINE__);
}

/* ---- gfx1250 cluster multicast loads ---- */

rocke_value_t* i32_param(rocke_ir_builder_t* b, const char* name)
{
    return rocke_b_param(b, name, rocke_i32(), nullptr);
}

rocke_value_t* lds_stage(rocke_ir_builder_t* b)
{
    const int shape[] = {64};
    return rocke_b_smem_addr_of(b, rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "stage"));
}

/* Mirrors _build_all in test_gfx1250_multicast.py. */
void build_gfx1250_multicast(rocke_ir_builder_t* b)
{
    rocke_value_t* src = global_ptr_param(b, "src", rocke_i32());
    rocke_value_t* dst = global_ptr_param(b, "dst", rocke_i32());
    rocke_value_t* mask = i32_param(b, "mask");
    rocke_value_t* local = lds_stage(b);
    rocke_value_t* v1 = rocke_b_cluster_load(b, src, mask, 4, 0);
    rocke_value_t* v2 = rocke_b_cluster_load(b, src, mask, 8, 1);
    rocke_value_t* v4 = rocke_b_cluster_load(b, src, mask, 16, 8);
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    rocke_b_global_store(b, dst, zero, v1, 4);
    rocke_b_global_store(b, dst, zero, rocke_b_vec_extract(b, v2, 1), 4);
    rocke_b_global_store(b, dst, zero, rocke_b_vec_extract(b, v4, 3), 4);
    rocke_b_cluster_load_async_to_lds(b, src, local, mask, 1, 0, 0);
    rocke_b_cluster_load_async_to_lds(b, src, local, mask, 4, 16, 0);
    rocke_b_cluster_load_async_to_lds(b, src, local, mask, 8, 0, 3);
    rocke_b_cluster_load_async_to_lds(b, src, local, mask, 16, -32, 31);
}

void case_gfx1250_multicast()
{
    const std::string ir = lower_one(
        "gfx1250_multicast", build_gfx1250_multicast, "gfx1250", ROCKE_LLVM_FLAVOR_LLVM23);
    EXPECT_IR(ir,
              " = call i32 @llvm.amdgcn.cluster.load.b32.i32("
              "ptr addrspace(1) %src, i32 0, i32 %mask)");
    EXPECT_IR(ir,
              " = call <2 x i32> @llvm.amdgcn.cluster.load.b64.v2i32("
              "ptr addrspace(1) %src, i32 1, i32 %mask)");
    EXPECT_IR(ir,
              " = call <4 x i32> @llvm.amdgcn.cluster.load.b128.v4i32("
              "ptr addrspace(1) %src, i32 8, i32 %mask)");
    EXPECT_IR_COUNT(
        ir, "declare i32 @llvm.amdgcn.cluster.load.b32.i32(ptr addrspace(1), i32 immarg, i32)", 1);
    EXPECT_IR_COUNT(
        ir,
        "declare <2 x i32> @llvm.amdgcn.cluster.load.b64.v2i32(ptr addrspace(1), i32 immarg, i32)",
        1);
    EXPECT_IR_COUNT(ir,
                    "declare <4 x i32> @llvm.amdgcn.cluster.load.b128.v4i32("
                    "ptr addrspace(1), i32 immarg, i32)",
                    1);
    static const struct
    {
        const char* suffix;
        const char* imms;
    } asyncs[] = {
        {"b8", "i32 0, i32 0, i32 %mask)"},
        {"b32", "i32 16, i32 0, i32 %mask)"},
        {"b64", "i32 0, i32 3, i32 %mask)"},
        {"b128", "i32 -32, i32 31, i32 %mask)"},
    };
    for(const auto& a : asyncs)
    {
        char call[160];
        char decl[192];
        snprintf(call,
                 sizeof(call),
                 "call void @llvm.amdgcn.cluster.load.async.to.lds.%s(ptr addrspace(1) %%src, "
                 "ptr addrspace(3) ",
                 a.suffix);
        snprintf(decl,
                 sizeof(decl),
                 "declare void @llvm.amdgcn.cluster.load.async.to.lds.%s(ptr addrspace(1), "
                 "ptr addrspace(3), i32 immarg, i32 immarg, i32)",
                 a.suffix);
        expect_count(ir, call, 1, __LINE__);
        expect_count(ir, decl, 1, __LINE__);
        /* The immediates trail the call on the same line. */
        const size_t at = ir.find(call);
        if(at != std::string::npos)
        {
            const std::string line = ir.substr(at, ir.find('\n', at) - at);
            if(line.find(a.imms) == std::string::npos)
                fail(line.c_str(), __LINE__);
        }
    }
}

void case_gfx1250_multicast_declares_only_used()
{
    const std::string ir = lower_one(
        "only_b32",
        [](rocke_ir_builder_t* b) {
            rocke_value_t* src = global_ptr_param(b, "src", rocke_i32());
            rocke_value_t* dst = global_ptr_param(b, "dst", rocke_i32());
            rocke_b_global_store(b,
                                 dst,
                                 rocke_b_const_i32(b, 0),
                                 rocke_b_cluster_load(b, src, rocke_b_const_i32(b, 3), 4, 0),
                                 4);
        },
        "gfx1250",
        ROCKE_LLVM_FLAVOR_LLVM23);
    EXPECT_IR(ir, "@llvm.amdgcn.cluster.load.b32.i32(");
    EXPECT_NO_IR(ir, "@llvm.amdgcn.cluster.load.b64");
    EXPECT_NO_IR(ir, "@llvm.amdgcn.cluster.load.b128");
    EXPECT_NO_IR(ir, "@llvm.amdgcn.cluster.load.async");
}

void case_gfx1250_multicast_gated()
{
    struct Gate
    {
        const char* name;
        void (*build)(rocke_ir_builder_t*);
    };
    static const Gate gates[] = {
        {"cluster_load",
         [](rocke_ir_builder_t* b) {
             rocke_value_t* src = global_ptr_param(b, "src", rocke_i32());
             rocke_b_global_store(b,
                                  src,
                                  rocke_b_const_i32(b, 0),
                                  rocke_b_cluster_load(b, src, rocke_b_const_i32(b, 1), 4, 0),
                                  4);
         }},
        {"cluster_load_async_to_lds",
         [](rocke_ir_builder_t* b) {
             rocke_b_cluster_load_async_to_lds(b,
                                               global_ptr_param(b, "src", rocke_i32()),
                                               lds_stage(b),
                                               rocke_b_const_i32(b, 1),
                                               4,
                                               0,
                                               0);
         }},
    };
    const struct
    {
        const char* arch;
        rocke_llvm_flavor_t flavor;
        const char* suffix;
    } bad[] = {
        {"gfx950", ROCKE_LLVM_FLAVOR_LLVM22, " requires gfx1250"},
        {"gfx1201", ROCKE_LLVM_FLAVOR_LLVM23, " requires gfx1250"},
        {"gfx1250", ROCKE_LLVM_FLAVOR_LLVM22, " requires LLVM flavor llvm23"},
    };
    for(const Gate& g : gates)
    {
        for(const auto& t : bad)
        {
            std::string err;
            const rocke_status_t st = lower_expect_error("gate", g.build, t.arch, t.flavor, &err);
            const std::string want = std::string(g.name) + t.suffix;
            if(st != ROCKE_ERR_VALUE || err.find(want) == std::string::npos)
            {
                char msg[ROCKE_ERR_MSG_CAP + 128];
                snprintf(msg,
                         sizeof(msg),
                         "%s on %s: status %d, err \"%s\"",
                         g.name,
                         t.arch,
                         (int)st,
                         err.c_str());
                fail(msg, __LINE__);
            }
        }
    }
}

void case_gfx1250_multicast_builder_rejects()
{
    struct Reject
    {
        const char* want;
        void (*emit)(rocke_ir_builder_t*, rocke_value_t*, rocke_value_t*, rocke_value_t*);
    };
    using B = rocke_ir_builder_t*;
    using V = rocke_value_t*;
    static const Reject rejects[] = {
        {"cluster_load width_bytes must be 4, 8, or 16 (got 1)",
         [](B b, V s, V, V) { rocke_b_cluster_load(b, s, rocke_b_const_i32(b, 1), 1, 0); }},
        {"cluster_load width_bytes must be 4, 8, or 16 (got 32)",
         [](B b, V s, V, V) { rocke_b_cluster_load(b, s, rocke_b_const_i32(b, 1), 32, 0); }},
        {"cluster_load cachepolicy",
         [](B b, V s, V, V) { rocke_b_cluster_load(b, s, rocke_b_const_i32(b, 1), 4, 32); }},
        {"cluster_load ptr must be a global pointer",
         [](B b, V, V f, V) { rocke_b_cluster_load(b, f, rocke_b_const_i32(b, 1), 4, 0); }},
        {"cluster_load mask must be i32",
         [](B b, V s, V, V) { rocke_b_cluster_load(b, s, rocke_b_const_i64(b, 1), 4, 0); }},
        {"cluster_load_async_to_lds width_bytes must be 1, 4, 8, or 16 (got 2)",
         [](B b, V s, V, V l) {
             rocke_b_cluster_load_async_to_lds(b, s, l, rocke_b_const_i32(b, 1), 2, 0, 0);
         }},
        {"cluster_load_async_to_lds cachepolicy",
         [](B b, V s, V, V l) {
             rocke_b_cluster_load_async_to_lds(b, s, l, rocke_b_const_i32(b, 1), 4, 0, -1);
         }},
        {"cluster_load_async_to_lds src_ptr must be a global pointer",
         [](B b, V, V f, V l) {
             rocke_b_cluster_load_async_to_lds(b, f, l, rocke_b_const_i32(b, 1), 4, 0, 0);
         }},
        {"cluster_load_async_to_lds local pointer must be",
         [](B b, V s, V, V) {
             rocke_b_cluster_load_async_to_lds(b, s, s, rocke_b_const_i32(b, 1), 4, 0, 0);
         }},
        {"cluster_load_async_to_lds mask must be i32",
         [](B b, V s, V, V l) {
             rocke_b_cluster_load_async_to_lds(b, s, l, rocke_b_const_i64(b, 1), 4, 0, 0);
         }},
    };
    for(const Reject& r : rejects)
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "bad");
        bool rejected = false;
        std::string err;
        try
        {
            rocke_value_t* src = global_ptr_param(&b, "src", rocke_i32());
            rocke_value_t* flat = private_ptr_param(&b, "flat", rocke_i32());
            r.emit(&b, src, flat, lds_stage(&b));
            rejected = rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE;
            const char* msg = rocke_ir_builder_error(&b);
            err = msg ? msg : "";
        }
        catch(const std::exception& e)
        {
            rejected = true;
            err = e.what();
        }
        if(!rejected || err.find(r.want) == std::string::npos)
        {
            char msg[ROCKE_ERR_MSG_CAP + 128];
            snprintf(msg, sizeof(msg), "want \"%s\", err \"%s\"", r.want, err.c_str());
            fail(msg, __LINE__);
        }
        rocke_ir_builder_free(&b);
    }
}

/* Raw ops carry attrs the builder would reject; the lowerer must catch them. */
void emit_raw_cluster_load(rocke_ir_builder_t* b, int64_t width, int64_t cachepolicy, int lanes)
{
    rocke_attr_map_t attrs;
    rocke_attr_map_init(&attrs);
    rocke_attr_set_int(b, &attrs, "width_bytes", width);
    rocke_attr_set_int(b, &attrs, "cachepolicy", cachepolicy);
    rocke_value_t* src = global_ptr_param(b, "src", rocke_i32());
    rocke_value_t* operands[] = {src, rocke_b_const_i32(b, 1)};
    const rocke_type_t* rty = lanes == 1 ? rocke_i32() : rocke_vector_type(b, rocke_i32(), lanes);
    rocke_op_t* op = rocke_b_op(
        b, ROCKE_OP_TILE_CLUSTER_LOAD, operands, 2, &rty, 1, &attrs, nullptr, 0, "cld", nullptr);
    rocke_value_t* v = rocke_op_result(b, op);
    rocke_b_global_store(
        b, src, rocke_b_const_i32(b, 0), lanes == 1 ? v : rocke_b_vec_extract(b, v, 0), 4);
}

void emit_raw_cluster_load_async(rocke_ir_builder_t* b,
                                 int64_t width,
                                 int64_t offset,
                                 int64_t cachepolicy)
{
    rocke_attr_map_t attrs;
    rocke_attr_map_init(&attrs);
    rocke_attr_set_int(b, &attrs, "width_bytes", width);
    rocke_attr_set_int(b, &attrs, "offset_bytes", offset);
    rocke_attr_set_int(b, &attrs, "cachepolicy", cachepolicy);
    rocke_value_t* operands[]
        = {global_ptr_param(b, "src", rocke_i32()), lds_stage(b), rocke_b_const_i32(b, 1)};
    rocke_b_op(b,
               ROCKE_OP_TILE_CLUSTER_LOAD_ASYNC_TO_LDS,
               operands,
               3,
               nullptr,
               0,
               &attrs,
               nullptr,
               0,
               nullptr,
               nullptr);
}

void case_gfx1250_multicast_lowerer_rechecks()
{
    struct Recheck
    {
        const char* want;
        void (*build)(rocke_ir_builder_t*);
    };
    static const Recheck rechecks[] = {
        {"cluster_load cachepolicy must be in 0..31, got 40",
         [](rocke_ir_builder_t* b) { emit_raw_cluster_load(b, 4, 40, 1); }},
        {"cluster_load width_bytes must be 4, 8, or 16, got 2",
         [](rocke_ir_builder_t* b) { emit_raw_cluster_load(b, 2, 0, 1); }},
        {"cluster_load result must be vec<i32x2>, got i32",
         [](rocke_ir_builder_t* b) { emit_raw_cluster_load(b, 8, 0, 1); }},
        {"cluster_load result must be i32, got vec<i32x4>",
         [](rocke_ir_builder_t* b) { emit_raw_cluster_load(b, 4, 0, 4); }},
        {"cluster_load_async_to_lds width_bytes must be 1, 4, 8, or 16, got 2",
         [](rocke_ir_builder_t* b) { emit_raw_cluster_load_async(b, 2, 0, 0); }},
        {"cluster_load_async_to_lds offset_bytes must fit signed i32, got 2147483648",
         [](rocke_ir_builder_t* b) { emit_raw_cluster_load_async(b, 4, INT64_C(1) << 31, 0); }},
        {"cluster_load_async_to_lds cachepolicy must be in 0..31, got 32",
         [](rocke_ir_builder_t* b) { emit_raw_cluster_load_async(b, 4, 0, 32); }},
    };
    for(const Recheck& r : rechecks)
    {
        std::string err;
        const rocke_status_t st
            = lower_expect_error("recheck", r.build, "gfx1250", ROCKE_LLVM_FLAVOR_LLVM23, &err);
        if(st != ROCKE_ERR_VALUE || err.find(r.want) == std::string::npos)
        {
            char msg[ROCKE_ERR_MSG_CAP + 128];
            snprintf(msg,
                     sizeof(msg),
                     "want \"%s\", status %d, err \"%s\"",
                     r.want,
                     (int)st,
                     err.c_str());
            fail(msg, __LINE__);
        }
    }
}

/* Multicast writes other workgroups' registers/LDS; DCE must keep both. */
void case_gfx1250_multicast_purity()
{
    if(rocke_opcode_is_pure(ROCKE_OP_TILE_CLUSTER_LOAD))
        fail("tile.cluster_load must not be pure", __LINE__);
    if(rocke_opcode_is_pure(ROCKE_OP_TILE_CLUSTER_LOAD_ASYNC_TO_LDS))
        fail("tile.cluster_load_async_to_lds must not be pure", __LINE__);
}

/* ---- gfx1250 TDM descriptor (tdm_descriptor_2d + tensor transfers) ---- */

struct TdmShape
{
    int elem_bytes = 4;
    int tile_dim0 = 64;
    int tile_dim1 = 16;
    int workgroup_mask = 0;
    int pad_interval = -1;
    int pad_amount = 0;
};

/* Mirrors _build_round_trip in test_gfx1250_tdm.py: global -> LDS -> global
 * through one 64x16 i32 tile. */
void build_gfx1250_tdm(rocke_ir_builder_t* b, const TdmShape& s)
{
    const int shape[] = {64 * 16};
    rocke_value_t* src = global_ptr_param(b, "src", rocke_i32());
    rocke_value_t* dst = global_ptr_param(b, "dst", rocke_i32());
    rocke_value_t* dim0 = i32_param(b, "dim0");
    rocke_value_t* dim1 = i32_param(b, "dim1");
    rocke_value_t* stride = i32_param(b, "stride");
    rocke_value_t* lds
        = rocke_b_smem_addr_of(b, rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "tile"));
    rocke_value_t* g[5];
    rocke_b_tdm_descriptor_2d(b,
                              src,
                              lds,
                              s.elem_bytes,
                              dim0,
                              dim1,
                              stride,
                              s.tile_dim0,
                              s.tile_dim1,
                              s.workgroup_mask,
                              s.pad_interval,
                              s.pad_amount,
                              g);
    rocke_b_tensor_load_to_lds(b, g[0], g[1], g[2], g[3], g[4], 0);
    rocke_b_s_wait_tensorcnt(b, 0);
    rocke_b_tdm_descriptor_2d(b,
                              dst,
                              lds,
                              s.elem_bytes,
                              dim0,
                              dim1,
                              stride,
                              s.tile_dim0,
                              s.tile_dim1,
                              s.workgroup_mask,
                              s.pad_interval,
                              s.pad_amount,
                              g);
    rocke_b_tensor_store_from_lds(b, g[0], g[1], g[2], g[3], g[4], 0);
    rocke_b_s_wait_tensorcnt(b, 0);
}

void build_gfx1250_tdm_default(rocke_ir_builder_t* b)
{
    build_gfx1250_tdm(b, TdmShape{});
}

/* The operand of every readfirstlane call, in emission order. The trailing
 * space in the needle skips the declaration, whose operand list is "(i32)". */
std::vector<std::string> tdm_readfirstlane_args(const std::string& ir)
{
    static const char needle[] = "@llvm.amdgcn.readfirstlane.i32(i32 ";
    std::vector<std::string> args;
    for(size_t at = ir.find(needle); at != std::string::npos; at = ir.find(needle, at + 1))
    {
        const size_t start = at + sizeof(needle) - 1;
        const size_t end = ir.find(')', start);
        if(end == std::string::npos)
            break;
        args.push_back(ir.substr(start, end - start));
    }
    return args;
}

void expect_arg(const std::vector<std::string>& args, size_t i, const char* want, int line)
{
    if(i >= args.size() || args[i] != want)
    {
        char msg[256];
        snprintf(msg,
                 sizeof(msg),
                 "readfirstlane arg %zu: want \"%s\", saw \"%s\"",
                 i,
                 want,
                 i < args.size() ? args[i].c_str() : "<missing>");
        fail(msg, line);
    }
}

void case_gfx1250_tdm_round_trip()
{
    const std::string ir
        = lower_one("gfx1250_tdm", build_gfx1250_tdm_default, "gfx1250", ROCKE_LLVM_FLAVOR_LLVM23);
    /* Group 0: global address split, hi[24:0] with type field (bits 31:30) = 2. */
    EXPECT_IR(ir, " = ptrtoint ptr addrspace(1) %src to i64");
    EXPECT_IR(ir, " = ptrtoint ptr addrspace(1) %dst to i64");
    EXPECT_IR_COUNT(ir, ", 32\n", 2);
    EXPECT_IR_COUNT(ir, ", 33554431\n", 2);
    EXPECT_IR_COUNT(ir, ", -2147483648\n", 2);
    /* The smem i64 address is truncated to the 32-bit LDS byte address. */
    EXPECT_IR(ir, " = trunc i64 %lds_addr");
    /* Group 1: tensor dims split 16/16; tile_dim0=64 lands in dword 3 high half. */
    EXPECT_IR(ir, " = shl i32 %dim0, 16\n");
    EXPECT_IR(ir, " = lshr i32 %dim0, 16\n");
    EXPECT_IR(ir, " = shl i32 %dim1, 16\n");
    EXPECT_IR(ir, " = lshr i32 %dim1, 16\n");
    EXPECT_IR_COUNT(ir, ", 4194304\n", 2);

    const std::vector<std::string> args = tdm_readfirstlane_args(ir);
    if(args.size() != 24)
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "want 24 readfirstlane args, saw %zu", args.size());
        fail(msg, __LINE__);
        return;
    }
    /* Group 0 dword 0 = 1; group 1 = flags (elem 4 B -> data_size 2), ...,
     * tile_dim1, row_stride, dim-1 stride 1<<16, 0. Both descriptors match. */
    static const struct
    {
        size_t index;
        const char* want;
    } constant_words[]
        = {{0, "1"}, {4, "131072"}, {8, "16"}, {9, "%stride"}, {10, "65536"}, {11, "0"}};
    for(size_t base : {size_t(0), size_t(12)})
        for(const auto& w : constant_words)
            expect_arg(args, base + w.index, w.want, __LINE__);

    EXPECT_IR_COUNT(ir, "call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %vp", 1);
    EXPECT_IR_COUNT(ir, "call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %vp", 1);
    EXPECT_IR_COUNT(ir, "call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)", 2);
    /* Groups 2 and 3 share one zero v4i32; each call ends with the cpol 0. */
    for(const char* call : {"call void @llvm.amdgcn.tensor.load.to.lds(",
                            "call void @llvm.amdgcn.tensor.store.from.lds("})
    {
        const size_t at = ir.find(call);
        if(at == std::string::npos)
            continue;
        const std::string line = ir.substr(at, ir.find('\n', at) - at);
        const size_t g2 = line.find("<4 x i32> %cz");
        const size_t g3 = g2 == std::string::npos ? g2 : line.find("<4 x i32> %cz", g2 + 1);
        if(g2 == std::string::npos || g3 == std::string::npos
           || line.substr(g2, line.find(',', g2) - g2) != line.substr(g3, line.find(',', g3) - g3)
           || line.find("<8 x i32> %cz") == std::string::npos
           || line.compare(line.size() - 8, 8, ", i32 0)") != 0)
            fail(line.c_str(), __LINE__);
    }
}

void case_gfx1250_tdm_flag_words()
{
    static const struct
    {
        TdmShape shape;
        const char* flags;
    } cases[] = {
        /* elem 2 B, wg mask 3, pad on (interval 2, amount 5). */
        {{2, 64, 16, 3, 2, 5}, "177274883"},
        /* elem 8 B, pad interval 7 / amount 127 sets bit 31: printed signed. */
        {{8, 64, 16, 0, 7, 127}, "-2949120"},
        /* elem 1 B, pad enabled with interval 0 still sets bit 20. */
        {{1, 64, 16, 0, 0, 0}, "1048576"},
    };
    for(const auto& c : cases)
    {
        const TdmShape s = c.shape;
        const std::string ir = lower_one(
            "tdm_flags",
            [&s](rocke_ir_builder_t* b) { build_gfx1250_tdm(b, s); },
            "gfx1250",
            ROCKE_LLVM_FLAVOR_LLVM23);
        expect_arg(tdm_readfirstlane_args(ir), 4, c.flags, __LINE__);
    }
    /* tile_dim0 = 0xFFFF shifts into bit 31 of dword 3: printed signed. */
    TdmShape wide;
    wide.tile_dim0 = 0xFFFF;
    const std::string ir = lower_one(
        "tdm_wide",
        [&wide](rocke_ir_builder_t* b) { build_gfx1250_tdm(b, wide); },
        "gfx1250",
        ROCKE_LLVM_FLAVOR_LLVM23);
    EXPECT_IR_COUNT(ir, ", -65536\n", 2);
}

void case_gfx1250_tdm_i32_lds_address()
{
    const std::string ir = lower_one(
        "i32_lds",
        [](rocke_ir_builder_t* b) {
            rocke_value_t* src = global_ptr_param(b, "src", rocke_i32());
            rocke_value_t* lds = i32_param(b, "lds");
            rocke_value_t* n = rocke_b_const_i32(b, 64);
            rocke_value_t* g[5];
            rocke_b_tdm_descriptor_2d(b, src, lds, 4, n, n, n, 64, 1, 0, -1, 0, g);
            rocke_b_tensor_load_to_lds(b, g[0], g[1], g[2], g[3], g[4], 0);
        },
        "gfx1250",
        ROCKE_LLVM_FLAVOR_LLVM23);
    EXPECT_NO_IR(ir, "trunc i64 %lds");
    expect_arg(tdm_readfirstlane_args(ir), 1, "%lds", __LINE__);
}

void case_gfx1250_tdm_group_types()
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "types");
    rocke_value_t* src = global_ptr_param(&b, "src", rocke_i32());
    rocke_value_t* n = rocke_b_const_i32(&b, 8);
    rocke_value_t* g[5];
    if(!rocke_b_tdm_descriptor_2d(&b, src, rocke_b_const_i32(&b, 0), 4, n, n, n, 8, 8, 0, -1, 0, g))
        fail("tdm_descriptor_2d failed", __LINE__);
    else
    {
        static const int lanes[] = {4, 8, 4, 4, 8};
        for(int i = 0; i < 5; ++i)
        {
            const rocke_type_t* t = g[i]->type;
            if(t->kind != ROCKE_TYPE_VECTOR || t->count != lanes[i]
               || !rocke_type_eq(t->elem, rocke_i32()))
                fail("tdm group type", __LINE__);
        }
        if(g[2] != g[3])
            fail("groups 2 and 3 must share one zero vector", __LINE__);
    }
    rocke_ir_builder_free(&b);
}

void case_global_ptr_to_i64()
{
    /* Plain address cast: available on every arch and flavor. */
    const std::string ir = lower_one(
        "p2i",
        [](rocke_ir_builder_t* b) {
            rocke_value_t* src = global_ptr_param(b, "src", rocke_i32());
            rocke_value_t* dst = global_ptr_param(b, "dst", rocke_i64());
            rocke_b_global_store(
                b, dst, rocke_b_const_i32(b, 0), rocke_b_global_ptr_to_i64(b, src), 8);
        },
        "gfx950",
        ROCKE_LLVM_FLAVOR_LLVM22);
    EXPECT_IR(ir, " = ptrtoint ptr addrspace(1) %src to i64\n");

    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "p2i_bad");
    bool rejected = false;
    std::string err;
    try
    {
        rocke_b_global_ptr_to_i64(&b, private_ptr_param(&b, "flat", rocke_f32()));
        rejected = rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE;
        const char* msg = rocke_ir_builder_error(&b);
        err = msg ? msg : "";
    }
    catch(const std::exception& e)
    {
        rejected = true;
        err = e.what();
    }
    if(!rejected || err.find("global_ptr_to_i64 ptr must be a global pointer") == std::string::npos)
        fail("global_ptr_to_i64 must reject a non-global pointer", __LINE__);
    rocke_ir_builder_free(&b);
}

void case_gfx1250_tdm_gated()
{
    const struct
    {
        const char* arch;
        rocke_llvm_flavor_t flavor;
        const char* want;
    } bad[] = {
        {"gfx950", ROCKE_LLVM_FLAVOR_LLVM22, "tensor_load_to_lds requires gfx1250"},
        {"gfx1201", ROCKE_LLVM_FLAVOR_LLVM23, "tensor_load_to_lds requires gfx1250"},
        {"gfx1250", ROCKE_LLVM_FLAVOR_LLVM22, "tensor_load_to_lds requires LLVM flavor llvm23"},
    };
    for(const auto& t : bad)
    {
        std::string err;
        const rocke_status_t st
            = lower_expect_error("gate", build_gfx1250_tdm_default, t.arch, t.flavor, &err);
        if(st != ROCKE_ERR_VALUE || err.find(t.want) == std::string::npos)
        {
            char msg[ROCKE_ERR_MSG_CAP + 128];
            snprintf(msg, sizeof(msg), "%s: status %d, err \"%s\"", t.arch, (int)st, err.c_str());
            fail(msg, __LINE__);
        }
    }
}

void case_gfx1250_tdm_builder_rejects()
{
    using B = rocke_ir_builder_t*;
    using V = rocke_value_t*;
    struct Reject
    {
        const char* want;
        /* (builder, global src, private ptr, i32 n) */
        void (*emit)(B, V, V, V);
    };
    static const Reject rejects[] = {
        {"tdm_descriptor_2d global_ptr must be a global pointer",
         [](B b, V, V f, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(b, f, n, 4, n, n, n, 8, 8, 0, -1, 0, g);
         }},
        {"tdm_descriptor_2d lds_addr must be i64 or i32",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(
                 b, s, rocke_b_const_f32(b, 0.0), 4, n, n, n, 8, 8, 0, -1, 0, g);
         }},
        {"tdm_descriptor_2d tensor_dim0 must be i32",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(
                 b, s, n, 4, rocke_b_const_i64(b, 8), n, n, 8, 8, 0, -1, 0, g);
         }},
        {"tdm_descriptor_2d tensor_dim1 must be i32",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(
                 b, s, n, 4, n, rocke_b_const_i64(b, 8), n, 8, 8, 0, -1, 0, g);
         }},
        {"tdm_descriptor_2d row_stride must be i32",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(
                 b, s, n, 4, n, n, rocke_b_const_i64(b, 8), 8, 8, 0, -1, 0, g);
         }},
        {"tdm_descriptor_2d elem_bytes must be 1, 2, 4, or 8",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(b, s, n, 3, n, n, n, 8, 8, 0, -1, 0, g);
         }},
        {"tdm_descriptor_2d tile_dim0 must be in 1..65535",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(b, s, n, 4, n, n, n, 0, 8, 0, -1, 0, g);
         }},
        {"tdm_descriptor_2d tile_dim0 must be in 1..65535",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(b, s, n, 4, n, n, n, 1 << 16, 8, 0, -1, 0, g);
         }},
        {"tdm_descriptor_2d tile_dim1 must be in 1..65535",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(b, s, n, 4, n, n, n, 8, 0, 0, -1, 0, g);
         }},
        {"tdm_descriptor_2d workgroup_mask must be in 0..65535",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(b, s, n, 4, n, n, n, 8, 8, 1 << 16, -1, 0, g);
         }},
        {"tdm_descriptor_2d workgroup_mask must be in 0..65535",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(b, s, n, 4, n, n, n, 8, 8, -1, -1, 0, g);
         }},
        {"tdm_descriptor_2d pad_interval must be in 0..7",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(b, s, n, 4, n, n, n, 8, 8, 0, 8, 0, g);
         }},
        {"tdm_descriptor_2d pad_amount must be in 0..127",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(b, s, n, 4, n, n, n, 8, 8, 0, 0, 128, g);
         }},
        {"needs pad_interval",
         [](B b, V s, V, V n) {
             V g[5];
             rocke_b_tdm_descriptor_2d(b, s, n, 4, n, n, n, 8, 8, 0, -1, 1, g);
         }},
    };
    for(const Reject& r : rejects)
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "bad");
        bool rejected = false;
        std::string err;
        try
        {
            rocke_value_t* src = global_ptr_param(&b, "src", rocke_i32());
            rocke_value_t* flat = private_ptr_param(&b, "flat", rocke_f32());
            r.emit(&b, src, flat, rocke_b_const_i32(&b, 8));
            rejected = rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE;
            const char* msg = rocke_ir_builder_error(&b);
            err = msg ? msg : "";
        }
        catch(const std::exception& e)
        {
            /* Builder errors surface as ckc::ValueError inside the library. */
            rejected = true;
            err = e.what();
        }
        if(!rejected || err.find(r.want) == std::string::npos)
        {
            char msg[ROCKE_ERR_MSG_CAP + 128];
            snprintf(msg, sizeof(msg), "want \"%s\", err \"%s\"", r.want, err.c_str());
            fail(msg, __LINE__);
        }
        rocke_ir_builder_free(&b);
    }
}

/* The address cast is side-effect free; the transfers are not. */
void case_gfx1250_tdm_purity()
{
    if(!rocke_opcode_is_pure(ROCKE_OP_TILE_GLOBAL_PTR_TO_I64))
        fail("tile.global_ptr_to_i64 must be pure", __LINE__);
    if(rocke_opcode_is_pure(ROCKE_OP_TILE_TENSOR_LOAD_TO_LDS))
        fail("tile.tensor_load_to_lds must not be pure", __LINE__);
    if(rocke_opcode_is_pure(ROCKE_OP_TILE_TENSOR_STORE_FROM_LDS))
        fail("tile.tensor_store_from_lds must not be pure", __LINE__);
}

/* ---- gfx1250 cluster launch shape (cluster_dims kernel attr) ---- */

void set_cluster_dims(rocke_ir_builder_t* b, const int64_t* dims, int count)
{
    rocke_attr_set_int_list(b, &rocke_ir_builder_kernel(b)->attrs, "cluster_dims", dims, count);
}

/* Mirrors TestGfx1250ClusterDims.test_attr_emitted_last_on_the_kernel: the
 * shape becomes the last string attribute of the kernel's attribute group. */
void case_gfx1250_cluster_dims()
{
    static const struct
    {
        int64_t dims[3];
        const char* want;
    } shapes[] = {
        {{2, 2, 1}, " \"amdgpu-cluster-dims\"=\"2,2,1\" norecurse nounwind }"},
        {{15, 1, 1}, " \"amdgpu-cluster-dims\"=\"15,1,1\" norecurse nounwind }"},
        {{4, 4, 1}, " \"amdgpu-cluster-dims\"=\"4,4,1\" norecurse nounwind }"},
        {{2, 2, 4}, " \"amdgpu-cluster-dims\"=\"2,2,4\" norecurse nounwind }"},
    };
    for(const auto& t : shapes)
    {
        const std::string ir = lower_one(
            "cluster_dims",
            [&t](rocke_ir_builder_t* b) { set_cluster_dims(b, t.dims, 3); },
            "gfx1250",
            ROCKE_LLVM_FLAVOR_LLVM23);
        EXPECT_IR(ir, t.want);
        EXPECT_IR_COUNT(ir, "amdgpu-cluster-dims", 1);
    }
    /* A kernel without the attr is untouched. */
    const std::string plain
        = lower_one("plain", [](rocke_ir_builder_t*) {}, "gfx1250", ROCKE_LLVM_FLAVOR_LLVM23);
    EXPECT_NO_IR(plain, "amdgpu-cluster-dims");
}

/* Shape and target validation; the text matches ir.check_cluster_dims and the
 * Python lowerer's gate. */
void case_gfx1250_cluster_dims_rejects()
{
    static const struct
    {
        int64_t dims[4];
        int count;
        const char* arch;
        rocke_llvm_flavor_t flavor;
        const char* want;
    } rejects[] = {
        {{2, 2},
         2,
         "gfx1250",
         ROCKE_LLVM_FLAVOR_LLVM23,
         "cluster_dims must be three integers (x, y, z)"},
        {{2, 2, 1, 1},
         4,
         "gfx1250",
         ROCKE_LLVM_FLAVOR_LLVM23,
         "cluster_dims must be three integers (x, y, z)"},
        {{0, 1, 1},
         3,
         "gfx1250",
         ROCKE_LLVM_FLAVOR_LLVM23,
         "cluster_dims (0, 1, 1): each dimension must be in 1..15"},
        {{1, -2, 1},
         3,
         "gfx1250",
         ROCKE_LLVM_FLAVOR_LLVM23,
         "cluster_dims (1, -2, 1): each dimension must be in 1..15"},
        {{16, 1, 1},
         3,
         "gfx1250",
         ROCKE_LLVM_FLAVOR_LLVM23,
         "cluster_dims (16, 1, 1): each dimension must be in 1..15"},
        {{1, 1, 16},
         3,
         "gfx1250",
         ROCKE_LLVM_FLAVOR_LLVM23,
         "cluster_dims (1, 1, 16): each dimension must be in 1..15"},
        {{4, 4, 2},
         3,
         "gfx1250",
         ROCKE_LLVM_FLAVOR_LLVM23,
         "cluster_dims (4, 4, 2): 32 workgroups exceeds the cluster limit of 16"},
        {{3, 3, 2},
         3,
         "gfx1250",
         ROCKE_LLVM_FLAVOR_LLVM23,
         "cluster_dims (3, 3, 2): 18 workgroups exceeds the cluster limit of 16"},
        {{2, 1, 1},
         3,
         "gfx950",
         ROCKE_LLVM_FLAVOR_LLVM22,
         "cluster_dims requires gfx1250, got gfx950"},
        {{2, 1, 1},
         3,
         "gfx1201",
         ROCKE_LLVM_FLAVOR_LLVM23,
         "cluster_dims requires gfx1250, got gfx1201"},
        {{2, 1, 1},
         3,
         "gfx1250",
         ROCKE_LLVM_FLAVOR_LLVM22,
         "cluster_dims requires LLVM flavor llvm23, got llvm22"},
    };
    for(const auto& r : rejects)
    {
        std::string err;
        const rocke_status_t st = lower_expect_error(
            "cluster_dims_bad",
            [&r](rocke_ir_builder_t* b) { set_cluster_dims(b, r.dims, r.count); },
            r.arch,
            r.flavor,
            &err);
        if(st != ROCKE_ERR_VALUE || err.find(r.want) == std::string::npos)
        {
            char msg[ROCKE_ERR_MSG_CAP + 256];
            snprintf(msg,
                     sizeof(msg),
                     "want \"%s\", status %d, err \"%s\"",
                     r.want,
                     (int)st,
                     err.c_str());
            fail(msg, __LINE__);
        }
    }
}

/* ---- memory capabilities ---- */

/* The embedded table must match arch_specs.json, which test_memory_caps_mirror
 * reads on the Python side. The gfx1250 feature flags are set only where the
 * feature passed a functional run on a device, which today is gfx1250 alone. */
void case_memory_caps()
{
    struct expected_caps
    {
        const char* gfx;
        rocke_memory_caps_t caps;
    };
    static const expected_caps table[] = {
        {"gfx11-generic", {false, false, false, false, 4, false, false, false, false}},
        {"gfx1151", {false, false, false, false, 4, false, false, false, false}},
        {"gfx1201", {false, false, false, false, 4, false, false, false, false}},
        {"gfx1250", {false, true, false, true, 4, true, true, true, true}},
        {"gfx90a", {false, false, false, false, 4, false, false, false, false}},
        {"gfx942", {true, true, false, false, 4, false, false, false, false}},
        {"gfx950", {true, true, true, false, 4, false, false, false, false}},
    };
    for(const expected_caps& row : table)
    {
        const rocke_arch_target_t* t = rocke_arch_target_from_gfx(row.gfx);
        if(t == nullptr)
        {
            fail(row.gfx, __LINE__);
            continue;
        }
        const rocke_memory_caps_t& got = t->memory;
        const rocke_memory_caps_t& want = row.caps;
        if(got.has_async_lds != want.has_async_lds
           || got.has_async_global_lds != want.has_async_global_lds
           || got.has_ds_read_tr != want.has_ds_read_tr || got.has_tdm != want.has_tdm
           || got.buffer_load_max_dwords != want.buffer_load_max_dwords
           || got.has_scalar_data_prefetch != want.has_scalar_data_prefetch
           || got.has_global_prefetch != want.has_global_prefetch
           || got.has_cluster_launch != want.has_cluster_launch
           || got.has_multicast_load != want.has_multicast_load)
            fail(row.gfx, __LINE__);
    }
}

/* ---- async buffer / global -> LDS ---- */
void case_buffer_load_lds_async()
{
    const std::string ir = lower_one("buf_async", [](rocke_ir_builder_t* b) {
        rocke_value_t* X = global_ptr_param(b, "X", rocke_f16());
        rocke_value_t* N = rocke_b_param(b, "N_bytes", rocke_i32(), nullptr);
        rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, X, N);
        const int shape[] = {64, 8};
        rocke_value_t* lds = rocke_b_smem_alloc(b, rocke_f16(), shape, 2, "stage");
        rocke_b_buffer_load_lds_async(b,
                                      rsrc,
                                      rocke_b_smem_addr_of(b, lds),
                                      rocke_b_const_i32(b, 0),
                                      rocke_b_const_i32(b, 0),
                                      /*dwords=*/4,
                                      /*coherency=*/2);
    });
    EXPECT_IR(ir, "@llvm.amdgcn.raw.ptr.buffer.load.async.lds");
    /* dwords=4 -> 16 bytes per lane; trailing imm is coherency (CACHE_STREAM=2). */
    EXPECT_IR(ir, "i32 16, i32 0, i32 0, i32 0, i32 2)");
    /* smem_addr_of yields an i64 LDS address, but the intrinsic declares
     * ptr addrspace(3); passing the i64 through is what LLVM rejects with
     * "defined with type 'i64' but expected 'ptr addrspace(3)'". */
    EXPECT_IR(ir, " = inttoptr i64 ");
    EXPECT_IR(ir, "ptr addrspace(3) %lds_ptr");
}

void case_global_load_async_to_lds_b8()
{
    /* width_bytes=1 selects the LLVM 23 `.b8` async copy. The opcode is a
     * gfx1250 one, but its lowering is arch-independent, so this runs on the
     * default gfx950 backend -- the C++ engine has no gfx1250 ISA backend. */
    const std::string ir = lower_one("gl_async_b8", [](rocke_ir_builder_t* b) {
        rocke_value_t* src = global_ptr_param(b, "src", rocke_i32());
        const int shape[] = {64};
        rocke_value_t* lds = rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "stage");
        rocke_value_t* const idx[] = {rocke_b_const_i32(b, 0)};
        rocke_b_global_load_async_to_lds(b,
                                         src,
                                         rocke_b_const_i32(b, 0),
                                         lds,
                                         idx,
                                         /*num_lds_indices=*/1,
                                         /*width_bytes=*/1,
                                         /*coherency=*/0,
                                         /*offset_bytes=*/0);
    });
    EXPECT_IR(ir,
              "declare void @llvm.amdgcn.global.load.async.to.lds.b8("
              "ptr addrspace(1) nocapture, ptr addrspace(3) nocapture, "
              "i32 immarg, i32 immarg)");
    EXPECT_IR(ir, "call void @llvm.amdgcn.global.load.async.to.lds.b8(");
    /* Per-lane source/destination addresses are computed with GEPs. */
    EXPECT_IR(ir, "getelementptr inbounds");
}

/* ---- HIP zero-extension source signedness ---- */
void case_hip_zext_uses_unsigned_source_cast()
{
    const std::string hip = lower_one_hip(
        "zext_i8",
        [](rocke_ir_builder_t* b) {
            rocke_value_t* byte = rocke_b_param(b, "byte", rocke_i8(), nullptr);
            rocke_b_zext(b, byte, rocke_i32());
            rocke_b_zext(b, byte, rocke_i64());
        },
        "gfx950");
    EXPECT_IR_COUNT(hip, "(int)(uint8_t)byte", 1);
    EXPECT_IR_COUNT(hip, "(int64_t)(uint8_t)byte", 1);
    EXPECT_NO_IR(hip, "(int)byte");
    EXPECT_NO_IR(hip, "(int64_t)byte");
}

/* ---- gfx1250 native SCALE / SCALE16 FP8/FP4 WMMA ---- */
void case_gfx1250_scaled_wmma()
{
    struct Variant
    {
        bool scale16;
        const rocke_type_t* (*scale_type)(void);
        const char* intrinsic;
        const char* builtin;
        const char* scale_llvm_type;
    };
    static const Variant variants[] = {
        {false,
         rocke_i32,
         "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32",
         "__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4",
         "i32"},
        {true,
         rocke_i64,
         "llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32",
         "__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4",
         "i64"},
    };

    for(int fmt : {0, 4})
        for(const Variant& v : variants)
        {
            const auto build = [&v, fmt](rocke_ir_builder_t* b) {
                rocke_value_t* matrix = global_ptr_param(b, "matrix", rocke_i32());
                rocke_value_t* accum = global_ptr_param(b, "accum", rocke_f32());
                rocke_value_t* scales = global_ptr_param(b, "scales", v.scale_type());
                rocke_value_t* lane = rocke_b_thread_id_x(b);
                rocke_value_t* lo = rocke_b_global_load_vN(b, matrix, lane, rocke_i32(), 8, 0);
                rocke_value_t* eight = rocke_b_const_i32(b, 8);
                rocke_value_t* hi_idx = rocke_b_add(b, lane, eight);
                rocke_value_t* hi = rocke_b_global_load_vN(b, matrix, hi_idx, rocke_i32(), 8, 0);
                rocke_value_t* fragment = rocke_b_vec_concat(b, lo, hi);
                rocke_value_t* c = rocke_b_global_load_vN(b, accum, lane, rocke_f32(), 8, 0);
                rocke_value_t* scale = rocke_b_global_load(b, scales, lane, v.scale_type(), 1);
                const auto block = v.scale16 ? ROCKE_MMA_SCALE_K16 : ROCKE_MMA_SCALE_K32;
                const rocke_mma_scale_filter_t scales_query = {"e8m0", "e8m0", block};
                const rocke_arch_target_t* target = rocke_arch_target_from_gfx("gfx1250");
                const rocke_mma_op_t* atom
                    = rocke_mma_catalog_op_for_shape(&target->mma,
                                                     "wmma_scaled",
                                                     fmt == 4 ? "fp4" : "fp8",
                                                     fmt == 4 ? "fp4e2m1" : "fp8",
                                                     "fp32",
                                                     16,
                                                     16,
                                                     128,
                                                     &scales_query);
                if(!atom)
                    fail("scaled WMMA aliases must resolve the requested atom", __LINE__);
                rocke_value_t* extra[2] = {scale, scale};
                rocke_value_t* d = rocke_b_mma(b, atom->op_id, fragment, fragment, c, extra, 2);
                rocke_b_global_store(b, accum, lane, rocke_b_vec_extract(b, d, 0), 1);
            };
            const char* name = v.scale16 ? "wmma_scale16" : "wmma_scale";
            const std::string ir = lower_one(name, build, "gfx1250", ROCKE_LLVM_FLAVOR_LLVM23);
            EXPECT_IR(ir, v.intrinsic);
            const std::string scale_arg = std::string(", ") + v.scale_llvm_type + " %";
            EXPECT_IR(ir, scale_arg.c_str());

            const std::string hip = lower_one_hip(name, build, "gfx1250");
            const std::string hip_call = std::string(v.builtin) + "(" + std::to_string(fmt) + ",";
            EXPECT_IR(hip, hip_call.c_str());
            const std::string matrix_arg = "i32 " + std::to_string(fmt) + ", <16 x i32>";
            EXPECT_IR_COUNT(ir, matrix_arg.c_str(), 2);
        }
}

/* ---- opcode table alignment ---- */

/* These opcodes were spliced into rocke_opcode_t's family groups rather than
 * appended, which is only safe while the two opcode-INDEXED tables in
 * core_types.cpp (rocke_opcode_names / rocke_opcode_pure) get their new row in
 * the same position. Both are sized ROCKE_OP__COUNT, so a missing row does not
 * fail the build: it shifts every later row by one and zero-fills the tail,
 * silently renaming ops. Pin each new opcode to its dotted name so that shift
 * is a test failure instead of a mislabelled op in serialized IR.
 *
 * ROCKE_OP_CF_RETURN is the last enumerator, so it catches a shift introduced
 * anywhere ahead of it -- including by an opcode this list does not name. */
void case_opcode_names_are_aligned()
{
    static const struct
    {
        rocke_opcode_t opcode;
        const char* name;
    } k_expect[] = {
        {ROCKE_OP_TILE_DS_SWIZZLE, "tile.ds_swizzle"},
        {ROCKE_OP_TILE_DS_SWIZZLE_XOR, "tile.ds_swizzle_xor"},
        {ROCKE_OP_TILE_MOV_DPP8, "tile.mov_dpp8"},
        {ROCKE_OP_TILE_QUAD_PERM, "tile.quad_perm"},
        {ROCKE_OP_TILE_WAVE_REDUCE, "tile.wave_reduce"},
        {ROCKE_OP_TILE_READLANE, "tile.readlane"},
        {ROCKE_OP_TILE_WRITELANE, "tile.writelane"},
        {ROCKE_OP_TILE_PERMLANE16, "tile.permlane16"},
        {ROCKE_OP_TILE_PERMLANE64, "tile.permlane64"},
        {ROCKE_OP_TILE_ALIGNBYTE, "tile.alignbyte"},
        {ROCKE_OP_TILE_S_WQM, "tile.s_wqm"},
        {ROCKE_OP_TILE_AV_LOAD_B128, "tile.av_load_b128"},
        {ROCKE_OP_TILE_AV_STORE_B128, "tile.av_store_b128"},
        {ROCKE_OP_TILE_S_ALLOC_VGPR, "tile.s_alloc_vgpr"},
        {ROCKE_OP_TILE_ASYNCMARK, "tile.asyncmark"},
        {ROCKE_OP_TILE_WAIT_ASYNCMARK, "tile.wait_asyncmark"},
        {ROCKE_OP_TILE_S_WAIT_EVENT, "tile.s_wait_event"},
        {ROCKE_OP_TILE_S_WAIT_ASYNCCNT, "tile.s_wait_asynccnt"},
        {ROCKE_OP_TILE_S_WAIT_TENSORCNT, "tile.s_wait_tensorcnt"},
        {ROCKE_OP_TILE_S_BARRIER_SIGNAL, "tile.s_barrier_signal"},
        {ROCKE_OP_TILE_S_BARRIER_WAIT, "tile.s_barrier_wait"},
        {ROCKE_OP_TILE_S_BARRIER_INIT, "tile.s_barrier_init"},
        {ROCKE_OP_TILE_S_BARRIER_SIGNAL_VAR, "tile.s_barrier_signal_var"},
        {ROCKE_OP_TILE_S_BARRIER_JOIN, "tile.s_barrier_join"},
        {ROCKE_OP_TILE_S_WAKEUP_BARRIER, "tile.s_wakeup_barrier"},
        {ROCKE_OP_TILE_S_BARRIER_LEAVE, "tile.s_barrier_leave"},
        {ROCKE_OP_TILE_S_PREFETCH_INST, "tile.s_prefetch_inst"},
        {ROCKE_OP_TILE_S_SETREG, "tile.s_setreg"},
        {ROCKE_OP_TILE_S_PREFETCH_DATA, "tile.s_prefetch_data"},
        {ROCKE_OP_TILE_S_BUFFER_PREFETCH_DATA, "tile.s_buffer_prefetch_data"},
        {ROCKE_OP_TILE_GLOBAL_PREFETCH, "tile.global_prefetch"},
        {ROCKE_OP_TILE_FLAT_PREFETCH, "tile.flat_prefetch"},
        {ROCKE_OP_TILE_CLUSTER_BARRIER, "tile.cluster_barrier"},
        {ROCKE_OP_TILE_CLUSTER_LOAD, "tile.cluster_load"},
        {ROCKE_OP_TILE_CLUSTER_LOAD_ASYNC_TO_LDS, "tile.cluster_load_async_to_lds"},
        {ROCKE_OP_TILE_GLOBAL_PTR_TO_I64, "tile.global_ptr_to_i64"},
        {ROCKE_OP_GPU_CLUSTER_ID, "gpu.cluster_id"},
        {ROCKE_OP_GPU_CLUSTER_WORKGROUP_ID, "gpu.cluster_workgroup_id"},
        {ROCKE_OP_GPU_CLUSTER_WORKGROUP_MAX_ID, "gpu.cluster_workgroup_max_id"},
        {ROCKE_OP_GPU_CLUSTER_WORKGROUP_FLAT_ID, "gpu.cluster_workgroup_flat_id"},
        {ROCKE_OP_GPU_CLUSTER_WORKGROUP_MAX_FLAT_ID, "gpu.cluster_workgroup_max_flat_id"},
        {ROCKE_OP_TILE_BUFFER_LOAD_LDS_ASYNC, "tile.buffer_load_lds_async"},
        {ROCKE_OP_TILE_GLOBAL_LOAD_ASYNC_TO_LDS, "tile.global_load_async_to_lds"},
        {ROCKE_OP_TILE_GLOBAL_STORE_ASYNC_FROM_LDS, "tile.global_store_async_from_lds"},
        {ROCKE_OP_TILE_GLOBAL_LOAD_TR16_B128, "tile.global_load_tr16_b128"},
        {ROCKE_OP_TILE_TENSOR_LOAD_TO_LDS, "tile.tensor_load_to_lds"},
        {ROCKE_OP_TILE_TENSOR_STORE_FROM_LDS, "tile.tensor_store_from_lds"},
        {ROCKE_OP_CF_RETURN, "cf.return"},
    };
    for(const auto& e : k_expect)
    {
        if(strcmp(rocke_opcode_name(e.opcode), e.name) != 0)
        {
            char msg[256];
            snprintf(msg,
                     sizeof(msg),
                     "opcode %d is named \"%s\", expected \"%s\"",
                     (int)e.opcode,
                     rocke_opcode_name(e.opcode),
                     e.name);
            fail(msg, __LINE__);
        }
        if(rocke_opcode_from_name(e.name) != e.opcode)
            fail(e.name, __LINE__);
        if(e.opcode == ROCKE_OP_TILE_QUAD_PERM && !rocke_opcode_is_pure(e.opcode))
            fail("tile.quad_perm must be pure", __LINE__);
    }
}

/* Malformed result lists must reach the public lowering error boundary. */
void case_wmma_result_count()
{
    for(bool integer : {false, true})
    {
        for(int count : {0, 2, 1})
        {
            rocke_ir_builder_t b;
            if(rocke_ir_builder_init(&b, "wmma_result_count") != ROCKE_OK)
            {
                fail("rocke_ir_builder_init", __LINE__);
                return;
            }
            auto* a = rocke_b_zero_vec(&b, integer ? rocke_i32() : rocke_f16(), integer ? 4 : 16);
            auto* c = rocke_b_zero_vec(&b, integer ? rocke_i32() : rocke_f32(), 8);
            auto* d = rocke_b_mma(&b,
                                  integer ? "wmma_i32_16x16x16_iu8" : "wmma_f32_16x16x16_f16",
                                  a,
                                  a,
                                  c,
                                  nullptr,
                                  0);
            rocke_value_t* results[] = {d, d};
            d->op->num_results = count;
            d->op->results = count ? results : nullptr;
            rocke_b_ret(&b);
            char* out = nullptr;
            char err[ROCKE_ERR_MSG_CAP] = {};
            const auto status = rocke_lower_kernel_to_llvm_ex(
                b.kernel, ROCKE_LLVM_FLAVOR_LLVM23, "gfx1151", &out, err, sizeof(err));
            if(count == 1)
            {
                if(status != ROCKE_OK || !out)
                    fail("valid WMMA failed", __LINE__);
                else
                    EXPECT_IR(std::string(out), integer ? "call <8 x i32>" : "call <8 x float>");
            }
            else
            {
                char expected[80];
                snprintf(expected,
                         sizeof(expected),
                         "tile.mma: expected exactly one result, got %d",
                         count);
                if(status != ROCKE_ERR_VALUE || out || strcmp(err, expected) != 0)
                    fail("malformed WMMA did not return its result-count error", __LINE__);
            }
            std::free(out);
            rocke_ir_builder_free(&b);
        }
    }
}

void case_lowbit_dtype_aliases()
{
    const char* aliases[][2] = {
        {"fp8", "fp8e4m3"},
        {"bf8", "bf8e5m2"},
        {"fp6", "fp6e2m3"},
        {"bf6", "fp6e3m2"},
        {"fp4", "fp4e2m1"},
    };
    char scratch[32];
    for(const auto& pair : aliases)
    {
        if(strcmp(rocke_normalize_dtype(pair[0], scratch, sizeof(scratch)), pair[1]) != 0
           || strcmp(rocke_normalize_dtype(pair[1], scratch, sizeof(scratch)), pair[1]) != 0)
            fail("low-bit alias must resolve to explicit catalog key", __LINE__);
        const auto* arch = rocke_arch_target_from_gfx("gfx950");
        const int k = strcmp(pair[0], "fp4") == 0 ? 128 : 96;
        if((strcmp(pair[0], "fp4") == 0 || strcmp(pair[0], "fp6") == 0)
           && !rocke_mma_catalog_has_shape(&arch->mma, "mma", pair[0], pair[1], "fp32", 16, 16, k))
            fail("low-bit aliases must resolve native catalog rows", __LINE__);
    }
}

struct TestCase
{
    const char* name;
    void (*fn)();
};

const TestCase k_cases[] = {
    {"lowbit_dtype_aliases", case_lowbit_dtype_aliases},
    {"wmma_result_count", case_wmma_result_count},
    {"ds_swizzle_raw_offset", case_ds_swizzle_raw_offset},
    {"quad_perm_hip", case_quad_perm_hip},
    {"quad_perm_rejects_invalid_input", case_quad_perm_rejects_invalid_input},
    {"quad_perm_hip_rejects_missing_ctrl", case_quad_perm_hip_rejects_missing_ctrl},
    {"quad_perm_rejects_out_of_range_ctrl", case_quad_perm_rejects_out_of_range_ctrl},
    {"ds_swizzle_xor", case_ds_swizzle_xor},
    {"mov_dpp8_i32", case_mov_dpp8_i32},
    {"quad_perm", case_quad_perm},
    {"mov_dpp8_f32", case_mov_dpp8_f32},
    {"mov_dpp8_both_types_coexist", case_mov_dpp8_both_types_coexist},
    {"wave_reduce", case_wave_reduce},
    {"wave_reduce_strategy", case_wave_reduce_strategy},
    {"readlane", case_readlane},
    {"writelane", case_writelane},
    {"permlane16", case_permlane16},
    {"permlane16_flags", case_permlane16_flags},
    {"permlane64", case_permlane64},
    {"permlane32_swap", case_permlane32_swap},
    {"alignbyte", case_alignbyte},
    {"s_wqm", case_s_wqm},
    {"av_load_b128", case_av_load_b128},
    {"av_store_b128", case_av_store_b128},
    {"s_alloc_vgpr", case_s_alloc_vgpr},
    {"asyncmark", case_asyncmark},
    {"wait_asyncmark", case_wait_asyncmark},
    {"s_wait_event", case_s_wait_event},
    {"s_prefetch_inst", case_s_prefetch_inst},
    {"gfx1250_data_prefetch", case_gfx1250_data_prefetch},
    {"gfx1250_data_prefetch_declares_only_used", case_gfx1250_data_prefetch_declares_only_used},
    {"gfx1250_data_prefetch_distinct_from_inst", case_gfx1250_data_prefetch_distinct_from_inst},
    {"gfx1250_data_prefetch_gated", case_gfx1250_data_prefetch_gated},
    {"gfx1250_data_prefetch_builder_rejects", case_gfx1250_data_prefetch_builder_rejects},
    {"gfx1250_data_prefetch_lowerer_rechecks", case_gfx1250_data_prefetch_lowerer_rechecks},
    {"gfx1250_cluster", case_gfx1250_cluster},
    {"gfx1250_cluster_size", case_gfx1250_cluster_size},
    {"gfx1250_cluster_gated", case_gfx1250_cluster_gated},
    {"gfx1250_cluster_builder_rejects", case_gfx1250_cluster_builder_rejects},
    {"gfx1250_cluster_lowerer_rechecks", case_gfx1250_cluster_lowerer_rechecks},
    {"gfx1250_cluster_purity", case_gfx1250_cluster_purity},
    {"gfx1250_multicast", case_gfx1250_multicast},
    {"gfx1250_multicast_declares_only_used", case_gfx1250_multicast_declares_only_used},
    {"gfx1250_multicast_gated", case_gfx1250_multicast_gated},
    {"gfx1250_multicast_builder_rejects", case_gfx1250_multicast_builder_rejects},
    {"gfx1250_multicast_lowerer_rechecks", case_gfx1250_multicast_lowerer_rechecks},
    {"gfx1250_multicast_purity", case_gfx1250_multicast_purity},
    {"gfx1250_tdm_round_trip", case_gfx1250_tdm_round_trip},
    {"gfx1250_tdm_flag_words", case_gfx1250_tdm_flag_words},
    {"gfx1250_tdm_i32_lds_address", case_gfx1250_tdm_i32_lds_address},
    {"gfx1250_tdm_group_types", case_gfx1250_tdm_group_types},
    {"global_ptr_to_i64", case_global_ptr_to_i64},
    {"gfx1250_tdm_gated", case_gfx1250_tdm_gated},
    {"gfx1250_tdm_builder_rejects", case_gfx1250_tdm_builder_rejects},
    {"gfx1250_tdm_purity", case_gfx1250_tdm_purity},
    {"gfx1250_cluster_dims", case_gfx1250_cluster_dims},
    {"gfx1250_cluster_dims_rejects", case_gfx1250_cluster_dims_rejects},
    {"memory_caps", case_memory_caps},
    {"buffer_load_lds_async", case_buffer_load_lds_async},
    {"global_load_async_to_lds_b8", case_global_load_async_to_lds_b8},
    {"hip_zext_uses_unsigned_source_cast", case_hip_zext_uses_unsigned_source_cast},
    {"gfx1250_scaled_wmma", case_gfx1250_scaled_wmma},
    {"gfx1250_standalone_bridge", case_gfx1250_standalone_bridge},
    {"opcode_names_are_aligned", case_opcode_names_are_aligned},
};

} // namespace

int main(void)
{
    for(const TestCase& tc : k_cases)
    {
        g_case = tc.name;
        tc.fn();
    }
    g_case = "";

    const size_t num_cases = sizeof(k_cases) / sizeof(k_cases[0]);
    if(g_failures != 0)
    {
        fprintf(stderr, "%zu case(s) run, %d failure(s)\n", num_cases, g_failures);
        return 1;
    }
    printf("future_intrinsic_lowering: %zu case(s) OK\n", num_cases);
    return 0;
}
