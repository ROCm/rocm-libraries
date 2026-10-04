// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_nontemporal_hip.cpp -- the C++ side of the temporal hint
 * (rocke_mem_opts_t.temporal_hint) on global_load_vN / global_store_vN.
 *
 * With no arguments it self-checks: a ROCKE_TEMPORAL_STREAMING op lowers to
 * __builtin_nontemporal_load / __builtin_nontemporal_store, a default one
 * does not, the unaligned memcpy load path and an under-aligned store do not
 * yet lower the hint (ROCKE_ERR_NOTIMPL; both still lower without it), a
 * non-bool attr is rejected
 * rather than coerced, an out-of-range hint or a struct_size of 0 (opts not
 * built with ROCKE_MEM_OPTS_INIT) puts the builder in its error state, a
 * temporal_hint lying past the caller's struct_size (an older, shorter struct)
 * is never read, the io helpers' _ex forms (load_vec, load_vec_as_f32,
 * store_vec) forward the hint to the op they emit, and the original io helper
 * signatures still build and emit no nontemporal access (HIP or LLVM).
 *
 * With `--hip <case> <arch>` it prints the lowered HIP source of one copy
 * kernel built exactly like tests/core/test_nontemporal_lowering.py's
 * _copy_kernel, so that test can byte-compare the two engines.
 */
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/strbuf.h"

namespace
{

int g_failures = 0;

void fail(const char* what, const char* where, int line)
{
    fprintf(stderr, "FAIL [%s]: %s (%s:%d)\n", where, what, __FILE__, line);
    ++g_failures;
}

rocke_mem_opts_t hint_opts(bool streaming)
{
    rocke_mem_opts_t o = ROCKE_MEM_OPTS_INIT;
    o.temporal_hint = streaming ? ROCKE_TEMPORAL_STREAMING : ROCKE_TEMPORAL_DEFAULT;
    return o;
}

/* One copy kernel: S -> D, n elements per thread. */
struct CopyCase
{
    const char* name;
    bool f16; /* false -> bf16 */
    int n;
    int load_align; /* <=0 -> default */
    bool load_streaming;
    bool store_streaming;
    int store_align; /* <=0 -> default */
};

const CopyCase CASES[] = {
    {"both", false, 8, 0, true, true, 0},
    {"load", false, 8, 0, true, false, 0},
    {"store", false, 8, 0, false, true, 0},
    {"plain", false, 8, 0, false, false, 0},
    /* align 2 < 16-byte payload: the memcpy load path. */
    {"memcpy_nt", true, 8, 2, true, false, 0},
    {"memcpy_plain", true, 8, 2, false, false, 0},
    /* align 2 < 16-byte payload on the store side. */
    {"store_underaligned_nt", true, 8, 0, false, true, 2},
    {"store_underaligned_plain", true, 8, 0, false, false, 2},
};

const CopyCase* find_case(const char* name)
{
    for(const CopyCase& c : CASES)
        if(strcmp(c.name, name) == 0)
            return &c;
    return nullptr;
}

rocke_value_t*
    copy_param(rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem, bool readonly)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.readonly = readonly;
    o.readonly_set = readonly;
    o.align = 16;
    o.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &o);
}

/* Which op, if any, gets its nontemporal attr rewritten to the integer 1 (a
 * hand-built or deserialized IR could carry it). */
enum class BadAttr
{
    none,
    load,
    store
};

/* Builder calls in _copy_kernel's order so both engines assign the same ids. */
void build(rocke_ir_builder_t* b, const CopyCase& c, BadAttr bad)
{
    const rocke_type_t* elem = c.f16 ? rocke_f16() : rocke_bf16();
    rocke_value_t* src = copy_param(b, "S", elem, true);
    rocke_value_t* dst = copy_param(b, "D", elem, false);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* off = rocke_b_mul(b, tid, rocke_b_const_i32(b, c.n));
    const rocke_mem_opts_t load_opts = hint_opts(c.load_streaming);
    const rocke_mem_opts_t store_opts = hint_opts(c.store_streaming);
    rocke_value_t* v = rocke_b_global_load_vN_ex(b, src, off, elem, c.n, c.load_align, &load_opts);
    if(bad == BadAttr::load && v && v->op)
        rocke_attr_set_int(b, &v->op->attrs, "nontemporal", 1);
    rocke_b_global_store_vN_ex(b, dst, off, v, c.n, c.store_align, &store_opts);
    if(bad == BadAttr::store)
    {
        /* The store returns no value; find its op in the entry region. */
        const rocke_region_t* body = rocke_ir_builder_kernel(b)->body;
        for(int i = 0; i < body->num_ops; ++i)
            if(body->ops[i]->opcode == ROCKE_OP_MEMREF_GLOBAL_STORE_VN)
                rocke_attr_set_int(b, &body->ops[i]->attrs, "nontemporal", 1);
    }
    rocke_b_ret(b);
}

/* Lower a built kernel to HIP; returns the status and fills `hip` on success. */
rocke_status_t lower_built(rocke_ir_builder_t* b, const char* arch, std::string* hip)
{
    if(!rocke_ir_builder_ok(b))
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(b));
        return ROCKE_ERR_VALUE;
    }
    rocke_strbuf_t out;
    rocke_strbuf_init(&out, 0);
    rocke_lower_hip_opts_t opts{};
    opts.arch = arch;
    const rocke_status_t st = rocke_lower_kernel_to_hip(b, rocke_ir_builder_kernel(b), &opts, &out);
    if(st == ROCKE_OK && hip)
        hip->assign(rocke_strbuf_cstr(&out));
    rocke_strbuf_free(&out);
    return st;
}

/* Lower a built kernel to LLVM IR; "" on failure. */
std::string lower_built_llvm(rocke_ir_builder_t* b, const char* arch)
{
    std::string ll;
    if(!rocke_ir_builder_ok(b))
        return ll;
    char* text = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = 0;
    if(rocke_lower_kernel_to_llvm_ex(
           rocke_ir_builder_kernel(b), ROCKE_LLVM_FLAVOR_AUTO, arch, &text, err, sizeof err)
           == ROCKE_OK
       && text)
        ll.assign(text);
    else
        fprintf(stderr, "LLVM lowering failed: %s\n", err);
    free(text);
    return ll;
}

/* Lower one case to HIP; returns the status and fills `hip` on success. */
rocke_status_t lower(const CopyCase& c, const char* arch, BadAttr bad, std::string* hip)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "nt_copy") != ROCKE_OK)
        return ROCKE_ERR_VALUE;
    build(&b, c, bad);
    const rocke_status_t st = lower_built(&b, arch, hip);
    rocke_ir_builder_free(&b);
    return st;
}

bool has(const std::string& s, const char* needle)
{
    return s.find(needle) != std::string::npos;
}

int count(const std::string& s, const char* needle)
{
    int k = 0;
    for(size_t at = s.find(needle); at != std::string::npos; at = s.find(needle, at + 1))
        ++k;
    return k;
}

/* load_vec -> store_vec, plus a load_vec_as_f32 whose lanes are stored back.
 * nt < 0 builds it with the original (opts-less) helper signatures; otherwise
 * with the _ex forms, each helper STREAMING by its own bit of `nt`
 * (1 load_vec, 2 load_vec_as_f32, 4 store_vec). Lowers to HIP (llvm=false)
 * or LLVM IR (llvm=true); returns "" on failure. */
std::string lower_io_helpers(const char* arch, int nt, bool llvm)
{
    std::string text;
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "nt_io") != ROCKE_OK)
        return text;
    rocke_value_t* src = copy_param(&b, "S", rocke_bf16(), true);
    rocke_value_t* dst = copy_param(&b, "D", rocke_bf16(), false);
    rocke_value_t* acc = copy_param(&b, "A", rocke_f32(), false);
    rocke_value_t* off = rocke_b_mul(&b, rocke_b_thread_id_x(&b), rocke_b_const_i32(&b, 8));
    rocke_value_t* v = nullptr;
    rocke_value_t* f[8] = {};
    int loaded = 0;
    if(nt < 0)
    {
        v = rocke_b_load_vec(&b, src, off, "bf16", 8);
        loaded = rocke_b_load_vec_as_f32(&b, src, off, "bf16", 8, f);
    }
    else
    {
        const rocke_mem_opts_t vec_opts = hint_opts(nt & 1);
        const rocke_mem_opts_t f32_opts = hint_opts(nt & 2);
        v = rocke_b_load_vec_ex(&b, src, off, "bf16", 8, &vec_opts);
        loaded = rocke_b_load_vec_as_f32_ex(&b, src, off, "bf16", 8, f, &f32_opts);
    }
    if(loaded)
        rocke_b_global_store(&b, acc, off, f[7], 0);
    if(nt < 0)
        rocke_b_store_vec(&b, dst, off, v, 8);
    else
    {
        const rocke_mem_opts_t store_opts = hint_opts(nt & 4);
        rocke_b_store_vec_ex(&b, dst, off, v, 8, &store_opts);
    }
    rocke_b_ret(&b);
    if(llvm)
        text = lower_built_llvm(&b, arch);
    else if(lower_built(&b, arch, &text) != ROCKE_OK)
        text.clear();
    rocke_ir_builder_free(&b);
    return text;
}

/* Which builder receives the out-of-range hint. */
enum class BadHint
{
    load_vN,
    store_vN,
    load_vec,
    load_vec_as_f32,
    store_vec
};

/* Invalid opts (an out-of-range temporal_hint, or a struct_size that was never
 * set) must leave the builder in its error state (ROCKE_ERR_VALUE) with an
 * error naming the problem, never be treated as streaming or as defaults. */
void check_bad_opts(BadHint which,
                    const rocke_mem_opts_t& bad,
                    const char* expect,
                    const char* what)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "nt_bad") != ROCKE_OK)
    {
        fail("builder init failed", what, __LINE__);
        return;
    }
    rocke_value_t* p = copy_param(&b, "P", rocke_bf16(), false);
    rocke_value_t* off = rocke_b_thread_id_x(&b);
    rocke_value_t* v = rocke_b_global_load_vN(&b, p, off, rocke_bf16(), 8, 0);
    if(!rocke_ir_builder_ok(&b))
        fail("valid setup load must succeed", what, __LINE__);
    const int ops_before = rocke_ir_builder_kernel(&b)->body->num_ops;
    rocke_value_t* f[8] = {};
    /* The engine reports builder errors as the sticky status or as a thrown
     * ckc::Error, depending on the boundary; accept either, as other tests do. */
    bool rejected = false;
    try
    {
        bool produced = false;
        switch(which)
        {
        case BadHint::load_vN:
            produced = rocke_b_global_load_vN_ex(&b, p, off, rocke_bf16(), 8, 0, &bad) != nullptr;
            break;
        case BadHint::store_vN:
            rocke_b_global_store_vN_ex(&b, p, off, v, 8, 0, &bad);
            break;
        case BadHint::load_vec:
            produced = rocke_b_load_vec_ex(&b, p, off, "bf16", 8, &bad) != nullptr;
            break;
        case BadHint::load_vec_as_f32:
            produced = rocke_b_load_vec_as_f32_ex(&b, p, off, "bf16", 8, f, &bad) != 0;
            break;
        case BadHint::store_vec:
            rocke_b_store_vec_ex(&b, p, off, v, 8, &bad);
            break;
        }
        rejected = !produced && rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE
                   && has(rocke_ir_builder_error(&b), expect);
    }
    catch(const ckc::Error& error)
    {
        rejected = error.code() == ROCKE_ERR_VALUE && has(error.what(), expect);
    }
    if(!rejected)
        fail("invalid rocke_mem_opts_t must be a ROCKE_ERR_VALUE builder error", what, __LINE__);
    if(rocke_ir_builder_kernel(&b)->body->num_ops != ops_before)
        fail("invalid rocke_mem_opts_t must not record an op", what, __LINE__);
    rocke_ir_builder_free(&b);
}

/* A caller built against an older, shorter rocke_mem_opts_t passes a smaller
 * struct_size. Simulate one whose header ended before temporal_hint: the
 * library must not read that field (here deliberately set to STREAMING) and
 * must lower both ops with the default policy. */
void check_short_struct(const char* arch)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "nt_short") != ROCKE_OK)
    {
        fail("builder init failed", arch, __LINE__);
        return;
    }
    rocke_mem_opts_t shorter = ROCKE_MEM_OPTS_INIT;
    shorter.struct_size = (uint32_t)offsetof(rocke_mem_opts_t, temporal_hint);
    shorter.temporal_hint = ROCKE_TEMPORAL_STREAMING;
    rocke_value_t* src = copy_param(&b, "S", rocke_bf16(), true);
    rocke_value_t* dst = copy_param(&b, "D", rocke_bf16(), false);
    rocke_value_t* off = rocke_b_mul(&b, rocke_b_thread_id_x(&b), rocke_b_const_i32(&b, 8));
    rocke_value_t* v = rocke_b_global_load_vN_ex(&b, src, off, rocke_bf16(), 8, 0, &shorter);
    rocke_b_global_store_vN_ex(&b, dst, off, v, 8, 0, &shorter);
    rocke_b_ret(&b);
    const std::string ll = lower_built_llvm(&b, arch);
    rocke_ir_builder_free(&b);
    if(ll.empty())
        fail("a shorter struct_size must still build and lower", arch, __LINE__);
    else if(has(ll, "nontemporal"))
        fail("a temporal_hint past struct_size must not be read", arch, __LINE__);
}

void self_check(const char* arch)
{
    std::string hip;
    if(lower(*find_case("both"), arch, BadAttr::none, &hip) != ROCKE_OK)
        fail("streaming copy kernel failed to lower", arch, __LINE__);
    if(!has(hip, "__builtin_nontemporal_load(reinterpret_cast<const bf16x8*>("))
        fail("streaming load is not __builtin_nontemporal_load", arch, __LINE__);
    if(!has(hip, "__builtin_nontemporal_store("))
        fail("streaming store is not __builtin_nontemporal_store", arch, __LINE__);

    for(const char* one : {"load", "store"})
    {
        hip.clear();
        if(lower(*find_case(one), arch, BadAttr::none, &hip) != ROCKE_OK)
            fail("single-hint kernel failed to lower", arch, __LINE__);
        const bool load = strcmp(one, "load") == 0;
        if(has(hip, "__builtin_nontemporal_load(") != load)
            fail("load hint leaked to/from the other op", arch, __LINE__);
        if(has(hip, "__builtin_nontemporal_store(") == load)
            fail("store hint leaked to/from the other op", arch, __LINE__);
    }

    hip.clear();
    if(lower(*find_case("plain"), arch, BadAttr::none, &hip) != ROCKE_OK)
        fail("plain copy kernel failed to lower", arch, __LINE__);
    if(has(hip, "__builtin_nontemporal"))
        fail("default ops must not use the nontemporal builtins", arch, __LINE__);

    /* The HIP memcpy path does not yet lower nontemporal (NOTIMPL), and
     * the same kernel without it still lowers through memcpy. */
    if(lower(*find_case("memcpy_nt"), arch, BadAttr::none, nullptr) != ROCKE_ERR_NOTIMPL)
        fail("nontemporal on the memcpy load path must be ROCKE_ERR_NOTIMPL", arch, __LINE__);
    hip.clear();
    if(lower(*find_case("memcpy_plain"), arch, BadAttr::none, &hip) != ROCKE_OK
       || !has(hip, "__builtin_memcpy("))
        fail("default unaligned load must still take the memcpy path", arch, __LINE__);

    /* An under-aligned streaming store is refused the same way (its vector
     * pointer cast would over-promise alignment); without the hint it lowers. */
    if(lower(*find_case("store_underaligned_nt"), arch, BadAttr::none, nullptr)
       != ROCKE_ERR_NOTIMPL)
        fail("nontemporal on an under-aligned store must be ROCKE_ERR_NOTIMPL", arch, __LINE__);
    if(lower(*find_case("store_underaligned_plain"), arch, BadAttr::none, nullptr) != ROCKE_OK)
        fail("default under-aligned store must still lower", arch, __LINE__);

    if(lower(*find_case("load"), arch, BadAttr::load, nullptr) != ROCKE_ERR_VALUE)
        fail("a non-bool nontemporal attr on the load must be ROCKE_ERR_VALUE", arch, __LINE__);
    if(lower(*find_case("store"), arch, BadAttr::store, nullptr) != ROCKE_ERR_VALUE)
        fail("a non-bool nontemporal attr on the store must be ROCKE_ERR_VALUE", arch, __LINE__);

    /* Each io helper _ex forwards the hint to exactly the op it emits. */
    for(int nt = 0; nt < 8; ++nt)
    {
        hip = lower_io_helpers(arch, nt, /*llvm=*/false);
        if(hip.empty())
            fail("io-helper kernel failed to lower", arch, __LINE__);
        if(count(hip, "__builtin_nontemporal_load(") != (nt & 1) + ((nt >> 1) & 1))
            fail("load_vec_ex / load_vec_as_f32_ex did not forward the hint", arch, __LINE__);
        if(has(hip, "__builtin_nontemporal_store(") != bool(nt & 4))
            fail("store_vec_ex did not forward the hint", arch, __LINE__);
    }

    /* The _ex helpers with STREAMING reach LLVM `!nontemporal` on every
     * global vector access (2 loads + 1 store). */
    const std::string ll_streaming = lower_io_helpers(arch, 7, /*llvm=*/true);
    if(ll_streaming.empty() || count(ll_streaming, ", !nontemporal !5") != 3
       || !has(ll_streaming, "!5 = !{i32 1}"))
        fail("STREAMING io helpers must emit !nontemporal in LLVM IR", arch, __LINE__);

    /* The original io helper signatures still build and keep the default
     * policy: no nontemporal access in either backend. */
    hip = lower_io_helpers(arch, -1, /*llvm=*/false);
    if(hip.empty() || has(hip, "__builtin_nontemporal"))
        fail("original io helpers must emit no nontemporal HIP access", arch, __LINE__);
    const std::string ll_plain = lower_io_helpers(arch, -1, /*llvm=*/true);
    if(ll_plain.empty() || has(ll_plain, "nontemporal"))
        fail("original io helpers must emit no !nontemporal", arch, __LINE__);
    if(ll_plain != lower_io_helpers(arch, 0, /*llvm=*/true))
        fail("original io helpers must equal the _ex helpers with DEFAULT", arch, __LINE__);
}

} // namespace

int main(int argc, char** argv)
{
    if(argc == 4 && strcmp(argv[1], "--hip") == 0)
    {
        const CopyCase* c = find_case(argv[2]);
        if(!c)
        {
            fprintf(stderr, "unknown case %s\n", argv[2]);
            return 2;
        }
        std::string hip;
        const rocke_status_t st = lower(*c, argv[3], BadAttr::none, &hip);
        if(st != ROCKE_OK)
        {
            fprintf(stderr, "HIP lowering failed (status %d)\n", (int)st);
            return 1;
        }
        fputs(hip.c_str(), stdout);
        return 0;
    }
    if(argc != 1)
    {
        fprintf(stderr, "usage: %s [--hip <case> <arch>]\n", argv[0]);
        return 2;
    }
    for(const char* arch : {"gfx942", "gfx950"})
    {
        self_check(arch);
        check_short_struct(arch);
    }
    rocke_mem_opts_t out_of_range = ROCKE_MEM_OPTS_INIT;
    out_of_range.temporal_hint = static_cast<rocke_temporal_hint_t>(2);
    rocke_mem_opts_t no_size = {};
    no_size.temporal_hint = ROCKE_TEMPORAL_STREAMING;
    const struct
    {
        BadHint which;
        const char* what;
    } helpers[] = {{BadHint::load_vN, "global_load_vN_ex"},
                   {BadHint::store_vN, "global_store_vN_ex"},
                   {BadHint::load_vec, "load_vec_ex"},
                   {BadHint::load_vec_as_f32, "load_vec_as_f32_ex"},
                   {BadHint::store_vec, "store_vec_ex"}};
    for(const auto& h : helpers)
    {
        check_bad_opts(h.which, out_of_range, "invalid temporal_hint 2", h.what);
        check_bad_opts(h.which, no_size, "invalid rocke_mem_opts_t.struct_size 0", h.what);
    }
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("nontemporal HIP lowering: OK\n");
    return 0;
}
