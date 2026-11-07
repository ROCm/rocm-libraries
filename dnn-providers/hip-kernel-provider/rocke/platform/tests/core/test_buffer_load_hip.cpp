// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_buffer_load_hip.cpp -- the HIP lowering of the dtype-generic
 * scalar buffer load (tile.buffer_load, IRBuilder.buffer_load).
 *
 * For every element type the builder accepts, it lowers a one-load kernel to
 * HIP and requires the two full lines of Python's
 * lower_hip._op_tile_buffer_load template, spelled with the kernel's own value
 * names: 2-byte types (f16, bf16) load through
 * __builtin_amdgcn_raw_buffer_load_b16 into an unsigned short, 4-byte types
 * (f32, i32) through _b32 into an unsigned int, with the operands in the order
 * (rsrc, voffset, soffset, 0); then a memcpy of that many bytes into the
 * element type.
 *
 * The expected text is that template written out here, not Python's output.
 * The byte-for-byte two-engine comparison is test_buffer_load_scalar.py, whose
 * C++ side needs the rocke_engine binding. With `--hip <arch> <dtype>` this
 * program prints the lowered HIP source of one kernel, so the Python and C++
 * engines can be byte-compared by hand.
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

/* HIP spellings of the load's operands and result: the IR value name without
 * its leading '%', as the HIP lowerer (Python _name) writes it. */
struct LoadNames
{
    std::string rsrc, voffset, soffset, result;
};

std::string hip_name(const rocke_value_t* v)
{
    if(!v || !v->name)
        return "";
    return v->name[0] == '%' ? v->name + 1 : v->name;
}

/* One wave; each lane loads one element of S through a buffer descriptor
 * bounded by num_bytes. */
LoadNames build(rocke_ir_builder_t* b, const Case& c)
{
    const rocke_type_t* ptr = rocke_ptr_type(b, c.type(), "global");
    rocke_value_t* src = rocke_b_param(b, "S", ptr, NULL);
    rocke_value_t* num_bytes = rocke_b_param(b, "num_bytes", rocke_i32(), NULL);
    /* Separate statements: argument evaluation order is unspecified. */
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* voffset = rocke_b_mul(b, tid, rocke_b_const_i32(b, c.two_byte ? 2 : 4));
    rocke_value_t* soffset = rocke_b_const_i32(b, 0);
    rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, src, num_bytes);
    rocke_value_t* result = rocke_b_buffer_load(b, rsrc, voffset, soffset, c.type());
    rocke_b_ret(b);
    return {hip_name(rsrc), hip_name(voffset), hip_name(soffset), hip_name(result)};
}

/* Lower the kernel for `arch` to HIP source. Returns false (after reporting)
 * if building or lowering fails. `names`, when given, receives the load's
 * value names. */
bool lower(const char* arch, const Case& c, std::string* hip, LoadNames* names = nullptr)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "buffer_load_hip") != ROCKE_OK)
    {
        fail("builder init failed", arch, c.dtype);
        return false;
    }
    const LoadNames built = build(&b, c);
    if(names)
        *names = built;
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
    LoadNames n;
    if(!lower(arch, c, &hip, &n))
        return;
    /* A swap of two operands only changes the text if their names differ. */
    if(n.rsrc.empty() || n.voffset.empty() || n.soffset.empty() || n.result.empty()
       || n.rsrc == n.voffset || n.rsrc == n.soffset || n.voffset == n.soffset)
    {
        fail("load operands need distinct, non-empty names (rsrc '" + n.rsrc + "', voffset '"
                 + n.voffset + "', soffset '" + n.soffset + "', result '" + n.result + "')",
             arch,
             c.dtype);
        return;
    }
    if(count(hip, "__builtin_amdgcn_raw_buffer_load_") != 1)
    {
        fail("HIP source does not make exactly one raw buffer load", arch, c.dtype);
        return;
    }
    /* Python: "<raw> _bl_<res> = (<raw>)__builtin_amdgcn_raw_buffer_load_<bN>(
     *          <rsrc>, <voffset>, <soffset>, 0);"
     * then    "<hip_type> <res>; __builtin_memcpy(&<res>, &_bl_<res>, <bytes>);" */
    const std::string raw = c.two_byte ? "unsigned short" : "unsigned int";
    const std::string tmp = "_bl_" + n.result;
    const std::string load = raw + " " + tmp + " = (" + raw + ")__builtin_amdgcn_raw_buffer_load_"
                             + (c.two_byte ? "b16" : "b32") + "(" + n.rsrc + ", " + n.voffset + ", "
                             + n.soffset + ", 0);";
    const std::string copy = std::string(c.hip_type) + " " + n.result + "; __builtin_memcpy(&"
                             + n.result + ", &" + tmp + ", " + (c.two_byte ? "2" : "4") + ");";
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
