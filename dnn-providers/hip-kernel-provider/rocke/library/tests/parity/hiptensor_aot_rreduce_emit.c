/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/hiptensor_aot_rreduce_emit.c -- C-side emitter for the tuned reduction classes
 * of hipTensor's kernel pack. Enumerates the configs of hiptensor_aot_rreduce_emit.py in the
 * same order (_hiptensor_aot_rows.rreduce_rows: the fixed set, the classes of every pack target
 * from the engine's route table, the finalize classes) and builds each kernel with
 * rocke/instance_hiptensor_aot.h.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/instance_hiptensor_aot.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

namespace
{

struct Fixed
{
    const char* mode;
    const char* a;
    const char* d;
    const char* op;
    int nr, block, vec, lanes, unroll, rows, nt, una, idx64, wave;
};

/* _hiptensor_aot_rows.RREDUCE_FIXED */
const Fixed FIXED[] = {
    {"row", "f32", "f32", "add", 1, 256, 4, 64, 2, 0, 0, 0, 0, 64},
    {"row", "f16", "f16", "max", 1, 512, 8, 512, 1, 0, 1, 0, 0, 64},
    {"row", "bf16", "bf16", "min", 2, 256, 8, 8, 8, 0, 1, 0, 0, 64},
    {"row", "f64", "f64", "mul", 3, 256, 2, 64, 4, 0, 1, 0, 0, 64},
    {"row", "f32", "f32", "max", 4, 128, 1, 1, 1, 0, 0, 0, 0, 64},
    {"row", "f16", "f16", "add", 8, 1024, 2, 16, 4, 0, 0, 1, 0, 64},
    {"row", "f32", "f32", "add", 1, 256, 4, 256, 2, 0, 0, 0, 1, 64},
    {"row", "f64", "f64", "max", 2, 64, 2, 32, 2, 0, 0, 1, 1, 64},
    {"row", "f16", "f16", "min", 1, 64, 8, 4, 2, 0, 0, 0, 0, 32},
    {"row", "bf16", "bf16", "mul", 2, 128, 4, 64, 1, 0, 0, 0, 0, 32},
    {"row", "f16", "f64", "max", 1, 256, 8, 64, 2, 0, 0, 0, 0, 64},
    {"col", "f32", "f32", "add", 1, 256, 4, 32, 2, 0, 0, 0, 0, 64},
    {"col", "f16", "f16", "max", 1, 1024, 4, 32, 2, 8, 1, 0, 0, 64},
    {"col", "bf16", "bf16", "min", 2, 1024, 8, 8, 1, 16, 1, 0, 0, 64},
    {"col", "f64", "f64", "mul", 3, 256, 2, 2, 4, 8, 1, 0, 0, 64},
    {"col", "f32", "f32", "max", 2, 512, 2, 16, 1, 0, 0, 0, 0, 64},
    {"col", "f16", "f16", "add", 4, 64, 1, 64, 1, 0, 0, 0, 0, 64},
    {"col", "f64", "f64", "add", 8, 128, 1, 1, 8, 0, 0, 1, 0, 64},
    {"col", "f32", "f32", "min", 1, 256, 4, 64, 2, 1, 0, 0, 1, 64},
    {"col", "bf16", "bf16", "max", 1, 256, 2, 16, 2, 4, 0, 1, 1, 64},
    {"col", "f16", "f16", "mul", 2, 128, 8, 8, 2, 0, 0, 0, 0, 32},
    {"col", "f32", "f32", "add", 1, 256, 4, 128, 2, 0, 0, 0, 0, 64},
    {"col", "f64", "f16", "add", 1, 256, 2, 32, 2, 0, 0, 0, 0, 64},
};

struct FinalFixed
{
    const char* a;
    const char* d;
    const char* op;
    int lanes, idx64, block, wave;
};

/* _hiptensor_aot_rows.RREDUCE_FINAL_FIXED */
const FinalFixed FINAL_FIXED[] = {
    {"f16", "f64", "max", 8, 0, 128, 64},
    {"f32", "f32", "min", 16, 0, 64, 32},
    {"f64", "f16", "add", 2, 1, 256, 64},
};

/* _hiptensor_aot_rows.RREDUCE_PACK_ARCHES */
const char* const PACK_ARCHES[] = {"gfx90a", "gfx942", "gfx950"};

struct Row
{
    bool final;
    rocke_htaot_rreduce_spec_t cls;
    rocke_htaot_rreduce_final_spec_t fin;
    const char* arch;
};

bool eqs(const char* a, const char* b)
{
    return strcmp(a, b) == 0;
}

bool same(const rocke_htaot_rreduce_spec_t& x, const rocke_htaot_rreduce_spec_t& y)
{
    return eqs(x.mode, y.mode) && eqs(x.a, y.a) && eqs(x.d, y.d) && eqs(x.op, y.op) && x.nr == y.nr
           && x.block == y.block && x.vec == y.vec && x.lanes == y.lanes && x.unroll == y.unroll
           && x.rows == y.rows && x.nt == y.nt && x.una == y.una && x.idx64 == y.idx64
           && x.wave == y.wave;
}

bool rows(std::vector<Row>* out)
{
    for(const Fixed& f : FIXED)
    {
        Row r;
        memset(&r, 0, sizeof r);
        r.cls = rocke_htaot_rreduce_spec_default();
        r.cls.mode = f.mode;
        r.cls.a = f.a;
        r.cls.d = f.d;
        r.cls.op = f.op;
        r.cls.nr = f.nr;
        r.cls.block = f.block;
        r.cls.vec = f.vec;
        r.cls.lanes = f.lanes;
        r.cls.unroll = f.unroll;
        r.cls.rows = f.rows;
        r.cls.nt = f.nt != 0;
        r.cls.una = f.una != 0;
        r.cls.idx64 = f.idx64 != 0;
        r.cls.wave = f.wave;
        r.arch = f.wave == 64 ? "gfx950" : "gfx1151";
        out->push_back(r);
    }
    for(const char* arch : PACK_ARCHES)
    {
        char err[512] = {0};
        int n = rocke_htaot_rreduce_pack_classes(arch, nullptr, 0, err, sizeof err);
        if(n < 0)
        {
            fprintf(stderr, "%s\n", err);
            return false;
        }
        std::vector<rocke_htaot_rreduce_spec_t> cls(n > 0 ? n : 1);
        rocke_htaot_rreduce_pack_classes(arch, cls.data(), n, err, sizeof err);
        for(int i = 0; i < n; ++i)
        {
            /* a spec already listed (the fixed set, another target's pack) is one config */
            bool dup = false;
            for(const Row& o : *out)
                dup = dup || (!o.final && same(o.cls, cls[i]));
            if(dup)
                continue;
            Row r;
            memset(&r, 0, sizeof r);
            r.cls = cls[i];
            r.arch = arch;
            out->push_back(r);
        }
    }
    for(const FinalFixed& f : FINAL_FIXED)
    {
        Row r;
        memset(&r, 0, sizeof r);
        r.final = true;
        r.fin = rocke_htaot_rreduce_final_spec_default();
        r.fin.a = f.a;
        r.fin.d = f.d;
        r.fin.op = f.op;
        r.fin.lanes = f.lanes;
        r.fin.idx64 = f.idx64 != 0;
        r.fin.block = f.block;
        r.fin.wave = f.wave;
        r.arch = f.wave == 64 ? "gfx950" : "gfx1151";
        out->push_back(r);
    }
    int nf = rocke_htaot_rreduce_final_classes(nullptr, 0);
    std::vector<rocke_htaot_rreduce_final_spec_t> fins(nf);
    rocke_htaot_rreduce_final_classes(fins.data(), nf);
    for(int i = 0; i < nf; ++i)
    {
        Row r;
        memset(&r, 0, sizeof r);
        r.final = true;
        r.fin = fins[i];
        r.arch = "gfx950";
        out->push_back(r);
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: hiptensor_aot_rreduce_emit <config_index> [ll|ir|verify]\n");
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    std::vector<Row> all;
    if(!rows(&all))
        return 1;
    if(idx < 0 || idx >= (int)all.size())
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const Row& row = all[idx];
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof b);
    rocke_kernel_def_t* kernel = row.final ? rocke_build_htaot_rreduce_final(&b, &row.fin)
                                           : rocke_build_htaot_rreduce(&b, &row.cls);
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
            kernel, ROCKE_LLVM_FLAVOR_AUTO, row.arch, &llvm_text, err, sizeof err);
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
