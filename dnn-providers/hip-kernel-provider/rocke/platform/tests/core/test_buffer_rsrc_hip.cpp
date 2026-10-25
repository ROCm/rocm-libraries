// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_buffer_rsrc_hip.cpp -- the C++ side of the buffer resource
 * descriptor word the HIP lowerer writes into __builtin_amdgcn_make_buffer_rsrc.
 *
 * DWORD3 of a buffer descriptor (the builtin's flags argument) has an
 * ISA-specific encoding: the gfx9 word makes every raw buffer access on gfx11 /
 * gfx12 read 0 or drop the write. So the HIP lowerer must take it from the
 * target, exactly as the LLVM lowerer does.
 *
 * With no arguments it self-checks, for every target the lowerers accept, that
 * the HIP flags word is the expected value for that target family and is the
 * same number the LLVM lowerer passes to llvm.amdgcn.make.buffer.rsrc.
 *
 * With `--hip <arch>` it prints the lowered HIP source of one copy kernel built
 * exactly like tests/core/test_buffer_rsrc_hip.py's _copy_kernel, so that test
 * can byte-compare the two engines.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/strbuf.h"

namespace
{

int g_failures = 0;

void fail(const std::string& what, const char* arch)
{
    fprintf(stderr, "FAIL [%s]: %s\n", arch, what.c_str());
    ++g_failures;
}

/* A one-wave f16 copy through two buffer descriptors: each lane loads one
 * dword (two halves) of S, bounded by load_bytes, and stores it to D, bounded
 * by store_bytes. Builder calls follow _copy_kernel's order so both engines
 * assign the same value names. */
void build(rocke_ir_builder_t* b)
{
    const rocke_type_t* ptr = rocke_ptr_type(b, rocke_f16(), "global");
    rocke_value_t* src = rocke_b_param(b, "S", ptr, NULL);
    rocke_value_t* dst = rocke_b_param(b, "D", ptr, NULL);
    rocke_value_t* load_bytes = rocke_b_param(b, "load_bytes", rocke_i32(), NULL);
    rocke_value_t* store_bytes = rocke_b_param(b, "store_bytes", rocke_i32(), NULL);
    /* Separate statements: argument evaluation order is unspecified. */
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* voffset = rocke_b_mul(b, tid, rocke_b_const_i32(b, 4));
    rocke_value_t* soffset = rocke_b_const_i32(b, 0);
    rocke_value_t* src_rsrc = rocke_b_buffer_rsrc(b, src, load_bytes);
    rocke_value_t* v = rocke_b_buffer_load_vN_f16(b, src_rsrc, voffset, soffset, 1);
    rocke_value_t* dst_rsrc = rocke_b_buffer_rsrc(b, dst, store_bytes);
    rocke_b_buffer_store_vN_f16(b, dst_rsrc, voffset, soffset, v, 1);
    rocke_b_ret(b);
}

/* Lower the copy kernel for `arch`: HIP source into `hip`, LLVM 20 IR into
 * `ll`. Returns false (after reporting) if either lowering fails. */
bool lower(const char* arch, std::string* hip, std::string* ll)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "buffer_rsrc_copy") != ROCKE_OK)
    {
        fail("builder init failed", arch);
        return false;
    }
    build(&b);
    bool ok = rocke_ir_builder_ok(&b);
    if(!ok)
        fail(std::string("builder error: ") + rocke_ir_builder_error(&b), arch);

    if(ok)
    {
        rocke_strbuf_t out;
        rocke_strbuf_init(&out, 0);
        rocke_lower_hip_opts_t opts{};
        opts.arch = arch;
        const rocke_status_t st
            = rocke_lower_kernel_to_hip(&b, rocke_ir_builder_kernel(&b), &opts, &out);
        if(st == ROCKE_OK)
            hip->assign(rocke_strbuf_cstr(&out));
        else
        {
            fail("HIP lowering failed (status " + std::to_string((int)st) + ")", arch);
            ok = false;
        }
        rocke_strbuf_free(&out);
    }

    if(ok && ll)
    {
        char* text = nullptr;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        if(rocke_lower_kernel_to_llvm_ex(
               rocke_ir_builder_kernel(&b), ROCKE_LLVM_FLAVOR_LLVM20, arch, &text, err, sizeof err)
               == ROCKE_OK
           && text)
            ll->assign(text);
        else
        {
            fail(std::string("LLVM lowering failed: ") + err, arch);
            ok = false;
        }
        free(text);
    }
    rocke_ir_builder_free(&b);
    return ok;
}

int count(const std::string& s, const std::string& needle)
{
    int k = 0;
    for(size_t at = s.find(needle); at != std::string::npos; at = s.find(needle, at + 1))
        ++k;
    return k;
}

/* Both descriptors carry the target's word, in both lowerers. */
void check(const char* arch, unsigned word3)
{
    std::string hip, ll;
    if(!lower(arch, &hip, &ll))
        return;
    char hip_flags[64];
    snprintf(hip_flags, sizeof hip_flags, "/*flags=*/(int)0x%08X);", word3);
    if(count(hip, hip_flags) != 2)
        fail(std::string("HIP source does not pass ") + hip_flags + " to both descriptors", arch);
    if(count(hip, "make_buffer_rsrc(") != 2)
        fail("HIP source does not build exactly two descriptors", arch);
    const std::string ll_word = ", i32 " + std::to_string(word3) + ")";
    const std::string ll_call = "@llvm.amdgcn.make.buffer.rsrc.p1(";
    int ll_matches = 0;
    for(size_t at = ll.find(ll_call); at != std::string::npos; at = ll.find(ll_call, at + 1))
    {
        const size_t eol = ll.find('\n', at);
        const std::string line = ll.substr(at, eol == std::string::npos ? eol : eol - at);
        if(line.size() >= ll_word.size()
           && line.compare(line.size() - ll_word.size(), ll_word.size(), ll_word) == 0)
            ++ll_matches;
    }
    if(ll_matches != 2)
        fail("LLVM IR does not pass" + ll_word + " to both descriptor calls", arch);
}

} // namespace

int main(int argc, char** argv)
{
    if(argc == 3 && strcmp(argv[1], "--hip") == 0)
    {
        std::string hip;
        if(!lower(argv[2], &hip, nullptr))
            return 1;
        fputs(hip.c_str(), stdout);
        return 0;
    }
    if(argc != 1)
    {
        fprintf(stderr, "usage: %s [--hip <arch>]\n", argv[0]);
        return 2;
    }
    /* gfx9: 32-bit uint data format, bounds-checked. */
    for(const char* arch : {"gfx90a", "gfx942", "gfx950"})
        check(arch, 0x00027000u);
    /* gfx11 / gfx12: the gfx10+ raw-buffer word. gfx1250 currently inherits it
     * from the gfx12 backend. */
    for(const char* arch : {"gfx1151", "gfx11-generic", "gfx1201", "gfx12-generic", "gfx1250"})
        check(arch, 0x31014000u);
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("buffer descriptor word per target: OK\n");
    return 0;
}
