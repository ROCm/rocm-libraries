/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/elementary_emit.c -- C-side emitter for the elementary-function
 * programs. Enumerates the configs of elementary_emit.py in the same order and
 * builds each kernel with rocke/elementary.h (rocke_elementary_emit, or
 * rocke::elementary::Math + family for the run-time predicate forms).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/elementary.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

namespace
{

struct cfg_t
{
    bool emit;
    std::string name; /* function (emit) or family */
    bool f64;
    bool anchored;
    std::vector<std::string> variant;
};

std::vector<cfg_t> configs()
{
    std::vector<cfg_t> out;
    for(int i = 0; i < rocke_elementary_num_functions(); ++i)
        for(int d = 0; d < 2; ++d)
            for(int a = 0; a < 2; ++a)
                out.push_back({true, rocke_elementary_function(i), d == 0, a == 0, {}});
    for(int i = 0; i < rocke_elementary_num_families(); ++i)
    {
        const char* fam = rocke_elementary_family(i);
        const char* const* mem = nullptr;
        int n = rocke_elementary_family_members(fam, &mem);
        std::vector<std::vector<std::string>> variants;
        variants.push_back(std::vector<std::string>(mem, mem + n));
        if(n > 1)
            for(int k = 0; k < n; ++k)
                variants.push_back({mem[k]});
        for(int d = 0; d < 2; ++d)
        {
            for(const auto& v : variants)
                out.push_back({false, fam, d == 0, true, v});
            out.push_back({false, fam, d == 0, false, variants[0]});
        }
    }
    return out;
}

rocke_kernel_def_t* build(rocke_ir_builder_t* b, const cfg_t& c)
{
    using namespace rocke::elementary;
    const rocke_type_t* ty = c.f64 ? rocke_f64() : rocke_f32();
    int nb = c.f64 ? 8 : 4;
    std::string tag;
    for(const auto& s : c.variant)
        tag += (tag.empty() ? "" : "_") + s;
    std::string kname = std::string("elementary_") + (c.emit ? "emit" : "family") + "_" + c.name
                        + "_" + (c.f64 ? "f64" : "f32") + "_" + (c.anchored ? "a" : "u")
                        + (tag.empty() ? "" : "_" + tag);
    if(rocke_ir_builder_init(b, kname.c_str()) != ROCKE_OK)
        return nullptr;
    rocke_param_opts_t po;
    memset(&po, 0, sizeof po);
    po.align = nb;
    po.align_set = true;
    rocke_value_t* y = rocke_b_param(b, "y", rocke_ptr_type(b, ty, "global"), &po);
    rocke_value_t* x = rocke_b_param(b, "x", ty, nullptr);
    rocke_value_t* r = nullptr;
    if(c.emit)
        r = rocke_elementary_emit(b, c.name.c_str(), x, c.anchored);
    else
    {
        rocke_value_t* code = rocke_b_param(b, "code", rocke_i32(), nullptr);
        std::vector<Pred> preds;
        for(size_t k = 0; k < c.variant.size(); ++k)
        {
            rocke_value_t* kc = rocke_b_const_i32(b, (int64_t)k + 1);
            preds.push_back({c.variant[k].c_str(), MV(rocke_b_cmp_eq(b, code, kc))});
        }
        Math m(b, c.anchored ? rocke_elementary_opaque_zero(b) : nullptr);
        if(c.anchored)
            m.materialize(family_constants(c.name.c_str(), preds));
        rocke_value_t* xx = c.f64 ? x : rocke_b_cvt_f32_to_f64(b, x);
        r = family(m, c.name.c_str(), MV(xx), preds).v;
        if(!c.f64)
            r = rocke_b_cvt_f64_to_f32(b, r);
    }
    rocke_value_t* z = rocke_b_const_i32(b, 0);
    rocke_b_global_store(b, y, z, r, nb);
    if(!rocke_ir_builder_ok(b))
        return nullptr;
    return rocke_ir_builder_kernel(b);
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: elementary_emit <config_index> [ll|ir|verify]\n");
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    std::vector<cfg_t> cfgs = configs();
    if(idx < 0 || idx >= (int)cfgs.size())
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const char* arch = "gfx950";
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof b);
    rocke_kernel_def_t* kernel = build(&b, cfgs[idx]);
    if(kernel == NULL)
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }
    int rc = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            rc = 1;
        }
        else
        {
            fputs(llvm_text, stdout);
            free(llvm_text);
        }
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rc = 1;
        }
        else
        {
            fputs(text, stdout);
            free(text);
        }
    }
    else
    {
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }
    rocke_ir_builder_free(&b);
    return rc;
}
