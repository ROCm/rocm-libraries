// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_nontemporal_hip.cpp -- the C++ HIP lowerer's side of the
 * `nontemporal=` flag on global_load_vN / global_store_vN.
 *
 * With no arguments it self-checks: a flagged op lowers to
 * __builtin_nontemporal_load / __builtin_nontemporal_store, an unflagged one
 * does not, the unaligned memcpy load path rejects the flag (and still lowers
 * without it), and a non-bool attr is rejected rather than coerced.
 *
 * With `--hip <case> <arch>` it prints the lowered HIP source of one copy
 * kernel built exactly like tests/core/test_nontemporal_lowering.py's
 * _copy_kernel, so that test can byte-compare the two engines.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/strbuf.h"

namespace
{

int g_failures = 0;

void fail(const char* what, const char* where, int line)
{
    fprintf(stderr, "FAIL [%s]: %s (%s:%d)\n", where, what, __FILE__, line);
    ++g_failures;
}

/* One copy kernel: S -> D, n elements per thread. */
struct CopyCase
{
    const char* name;
    bool f16; /* false -> bf16 */
    int n;
    int load_align; /* <=0 -> default */
    int load_nt;
    int store_nt;
};

const CopyCase CASES[] = {
    {"both", false, 8, 0, 1, 1},
    {"load", false, 8, 0, 1, 0},
    {"store", false, 8, 0, 0, 1},
    {"plain", false, 8, 0, 0, 0},
    /* align 2 < 16-byte payload: the memcpy load path. */
    {"memcpy_nt", true, 8, 2, 1, 0},
    {"memcpy_plain", true, 8, 2, 0, 0},
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
    rocke_value_t* v = rocke_b_global_load_vN_ex(b, src, off, elem, c.n, c.load_align, c.load_nt);
    if(bad == BadAttr::load && v && v->op)
        rocke_attr_set_int(b, &v->op->attrs, "nontemporal", 1);
    rocke_b_global_store_vN_ex(b, dst, off, v, c.n, 0, c.store_nt);
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

/* Lower one case to HIP; returns the status and fills `hip` on success. */
rocke_status_t lower(const CopyCase& c, const char* arch, BadAttr bad, std::string* hip)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "nt_copy") != ROCKE_OK)
        return ROCKE_ERR_VALUE;
    build(&b, c, bad);
    rocke_status_t st = ROCKE_ERR_VALUE;
    if(rocke_ir_builder_ok(&b))
    {
        rocke_strbuf_t out;
        rocke_strbuf_init(&out, 0);
        rocke_lower_hip_opts_t opts{};
        opts.arch = arch;
        st = rocke_lower_kernel_to_hip(&b, rocke_ir_builder_kernel(&b), &opts, &out);
        if(st == ROCKE_OK && hip)
            hip->assign(rocke_strbuf_cstr(&out));
        rocke_strbuf_free(&out);
    }
    else
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
    rocke_ir_builder_free(&b);
    return st;
}

bool has(const std::string& s, const char* needle)
{
    return s.find(needle) != std::string::npos;
}

void self_check(const char* arch)
{
    std::string hip;
    if(lower(*find_case("both"), arch, BadAttr::none, &hip) != ROCKE_OK)
        fail("flagged copy kernel failed to lower", arch, __LINE__);
    if(!has(hip, "__builtin_nontemporal_load(reinterpret_cast<const bf16x8*>("))
        fail("flagged load is not __builtin_nontemporal_load", arch, __LINE__);
    if(!has(hip, "__builtin_nontemporal_store("))
        fail("flagged store is not __builtin_nontemporal_store", arch, __LINE__);

    for(const char* one : {"load", "store"})
    {
        hip.clear();
        if(lower(*find_case(one), arch, BadAttr::none, &hip) != ROCKE_OK)
            fail("single-flag kernel failed to lower", arch, __LINE__);
        const bool load = strcmp(one, "load") == 0;
        if(has(hip, "__builtin_nontemporal_load(") != load)
            fail("load flag leaked to/from the other op", arch, __LINE__);
        if(has(hip, "__builtin_nontemporal_store(") == load)
            fail("store flag leaked to/from the other op", arch, __LINE__);
    }

    hip.clear();
    if(lower(*find_case("plain"), arch, BadAttr::none, &hip) != ROCKE_OK)
        fail("plain copy kernel failed to lower", arch, __LINE__);
    if(has(hip, "__builtin_nontemporal"))
        fail("unflagged ops must not use the nontemporal builtins", arch, __LINE__);

    /* The memcpy path has no nontemporal form: the flag is rejected there, and
     * the same kernel without it still lowers through memcpy. */
    if(lower(*find_case("memcpy_nt"), arch, BadAttr::none, nullptr) != ROCKE_ERR_VALUE)
        fail("nontemporal on the memcpy load path must be ROCKE_ERR_VALUE", arch, __LINE__);
    hip.clear();
    if(lower(*find_case("memcpy_plain"), arch, BadAttr::none, &hip) != ROCKE_OK
       || !has(hip, "__builtin_memcpy("))
        fail("unflagged unaligned load must still take the memcpy path", arch, __LINE__);

    if(lower(*find_case("load"), arch, BadAttr::load, nullptr) != ROCKE_ERR_VALUE)
        fail("a non-bool nontemporal attr on the load must be ROCKE_ERR_VALUE", arch, __LINE__);
    if(lower(*find_case("store"), arch, BadAttr::store, nullptr) != ROCKE_ERR_VALUE)
        fail("a non-bool nontemporal attr on the store must be ROCKE_ERR_VALUE", arch, __LINE__);
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
        self_check(arch);
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("nontemporal HIP lowering: OK\n");
    return 0;
}
