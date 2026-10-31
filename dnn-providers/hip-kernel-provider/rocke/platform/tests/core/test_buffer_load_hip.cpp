// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_buffer_load_hip.cpp -- the HIP lowering of the dtype-generic
 * scalar buffer load (tile.buffer_load, IRBuilder.buffer_load).
 *
 * For every element type the builder accepts, it lowers a one-load kernel to
 * HIP and checks the text Python's lower_hip._op_tile_buffer_load emits:
 * 2-byte types (f16, bf16) load through __builtin_amdgcn_raw_buffer_load_b16
 * into an unsigned short, 4-byte types (f32, i32) through _b32 into an
 * unsigned int, then a memcpy of that many bytes into the element type.
 *
 * With `--hip <arch> <dtype>` it prints the lowered HIP source of one kernel,
 * so the Python and C++ engines can be byte-compared by hand.
 */
#include <cstdio>
#include <cstring>
#include <string>

#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/strbuf.h"

namespace
{

int g_failures = 0;

void fail(const std::string& what, const char* arch, const char* dtype)
{
    fprintf(stderr, "FAIL [%s %s]: %s\n", arch, dtype, what.c_str());
    ++g_failures;
}

struct Case
{
    const char* dtype;
    const rocke_type_t* (*type)(void);
    const char* hip_type; /* Python _HIP_TYPE[dtype] */
    bool two_byte;
};

const Case kCases[] = {
    {"f16", rocke_f16, "fp16", true},
    {"bf16", rocke_bf16, "bf16", true},
    {"f32", rocke_f32, "float", false},
    {"i32", rocke_i32, "int", false},
};

/* One wave; each lane loads one element of S through a buffer descriptor
 * bounded by num_bytes. */
void build(rocke_ir_builder_t* b, const Case& c)
{
    const rocke_type_t* ptr = rocke_ptr_type(b, c.type(), "global");
    rocke_value_t* src = rocke_b_param(b, "S", ptr, NULL);
    rocke_value_t* num_bytes = rocke_b_param(b, "num_bytes", rocke_i32(), NULL);
    /* Separate statements: argument evaluation order is unspecified. */
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* voffset = rocke_b_mul(b, tid, rocke_b_const_i32(b, c.two_byte ? 2 : 4));
    rocke_value_t* soffset = rocke_b_const_i32(b, 0);
    rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, src, num_bytes);
    rocke_b_buffer_load(b, rsrc, voffset, soffset, c.type());
    rocke_b_ret(b);
}

/* Lower the kernel for `arch` to HIP source. Returns false (after reporting)
 * if building or lowering fails. */
bool lower(const char* arch, const Case& c, std::string* hip)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "buffer_load_hip") != ROCKE_OK)
    {
        fail("builder init failed", arch, c.dtype);
        return false;
    }
    build(&b, c);
    bool ok = rocke_ir_builder_ok(&b);
    if(!ok)
        fail(std::string("builder error: ") + rocke_ir_builder_error(&b), arch, c.dtype);
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
            fail("HIP lowering failed (status " + std::to_string((int)st) + ")", arch, c.dtype);
            ok = false;
        }
        rocke_strbuf_free(&out);
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

void check(const char* arch, const Case& c)
{
    std::string hip;
    if(!lower(arch, c, &hip))
        return;
    const std::string raw = c.two_byte ? "unsigned short" : "unsigned int";
    const std::string call
        = std::string("__builtin_amdgcn_raw_buffer_load_") + (c.two_byte ? "b16" : "b32") + "(";
    if(count(hip, "__builtin_amdgcn_raw_buffer_load_") != 1 || count(hip, call) != 1)
    {
        fail("HIP source does not make exactly one " + call + ") load", arch, c.dtype);
        return;
    }
    /* "<raw> _bl_<name> = (<raw>)<call>..." then
     * "<hip_type> <name>; __builtin_memcpy(&<name>, &_bl_<name>, <bytes>);" */
    const std::string decl = raw + " _bl_";
    const size_t at = hip.find(decl);
    const size_t name_end = at == std::string::npos ? at : hip.find(' ', at + decl.size());
    if(name_end == std::string::npos)
    {
        fail("HIP source has no '" + decl + "<name>' temporary", arch, c.dtype);
        return;
    }
    const std::string name = hip.substr(at + decl.size(), name_end - at - decl.size());
    const std::string load = decl + name + " = (" + raw + ")" + call;
    const std::string copy = std::string(c.hip_type) + " " + name + "; __builtin_memcpy(&" + name
                             + ", &_bl_" + name + ", " + (c.two_byte ? "2" : "4") + ");";
    if(count(hip, load) != 1)
        fail("HIP source does not contain '" + load + "'", arch, c.dtype);
    if(count(hip, copy) != 1)
        fail("HIP source does not contain '" + copy + "'", arch, c.dtype);
}

} // namespace

int main(int argc, char** argv)
{
    if(argc == 4 && strcmp(argv[1], "--hip") == 0)
    {
        for(const Case& c : kCases)
        {
            if(strcmp(c.dtype, argv[3]) != 0)
                continue;
            std::string hip;
            if(!lower(argv[2], c, &hip))
                return 1;
            fputs(hip.c_str(), stdout);
            return 0;
        }
        fprintf(stderr, "unknown dtype %s\n", argv[3]);
        return 2;
    }
    if(argc != 1)
    {
        fprintf(stderr, "usage: %s [--hip <arch> <f16|bf16|f32|i32>]\n", argv[0]);
        return 2;
    }
    for(const char* arch : {"gfx942", "gfx950"})
        for(const Case& c : kCases)
            check(arch, c);
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("tile.buffer_load HIP lowering per element type: OK\n");
    return 0;
}
