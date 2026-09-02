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
#include <string>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

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
std::string lower_one(const char* name, BuildFn build, const char* arch = "gfx950")
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
        rocke_ir_builder_kernel(&b), ROCKE_LLVM_FLAVOR_AUTO, arch, &ll, err, sizeof(err));
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
        {ROCKE_OP_TILE_S_PREFETCH_INST, "tile.s_prefetch_inst"},
        {ROCKE_OP_TILE_BUFFER_LOAD_LDS_ASYNC, "tile.buffer_load_lds_async"},
        {ROCKE_OP_TILE_GLOBAL_LOAD_ASYNC_TO_LDS, "tile.global_load_async_to_lds"},
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
    }
}

struct TestCase
{
    const char* name;
    void (*fn)();
};

const TestCase k_cases[] = {
    {"ds_swizzle_raw_offset", case_ds_swizzle_raw_offset},
    {"ds_swizzle_xor", case_ds_swizzle_xor},
    {"mov_dpp8_i32", case_mov_dpp8_i32},
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
    {"buffer_load_lds_async", case_buffer_load_lds_async},
    {"global_load_async_to_lds_b8", case_global_load_async_to_lds_b8},
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
