/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/tensor_pack_emit.c -- C-side emitter for the tensor pack
 * (rocke/instance_tensor_gett.h: TensorPackSpec, build_tensor_pack). Builds the rows of
 * tensor_pack_emit.py (same order).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <string>
#include <vector>

#include "rocke/instance_tensor_gett.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

namespace
{
using namespace rocke::gett;

bool make(int idx, TensorPackSpec* s)
{
    switch(idx)
    {
    case 0: /* pairs_f16 */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"K1", "M1", "K0", "M0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        return true;
    }
    case 1: /* pairs_bf16 */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"K1", "M1", "K0", "M0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "bf16");
        return true;
    }
    case 2: /* pairs_g1 */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"K1", "M1", "K0", "M0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        s->groups = 1;
        return true;
    }
    case 3: /* pairs_g4_b128 */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"K1", "M1", "K0", "M0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        s->groups = 4;
        s->block = 128;
        return true;
    }
    case 4: /* pairs_b512 */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"K1", "M1", "K0", "M0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        s->block = 512;
        return true;
    }
    case 5: /* pairs_b64 */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"K1", "M1", "K0", "M0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        s->block = 64;
        return true;
    }
    case 6: /* tiled_f16 */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"M1", "M0", "K1", "K0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        return true;
    }
    case 7: /* tiled_bf16 */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"M1", "M0", "K1", "K0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "bf16");
        return true;
    }
    case 8: /* tiled_ragged */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 26}, {"M0", 40}, {"M1", 36}};
        *s = TensorPackSpec::make(
            {"M1", "M0", "K1", "K0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        return true;
    }
    case 9: /* tiled_kfirst */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 72}, {"M1", 64}};
        *s = TensorPackSpec::make(
            {"K1", "K0", "M1", "M0"}, {"M1", "M0", "K1", "K0"}, ext, nullptr, "fp16");
        return true;
    }
    case 10: /* elements_odd */
    {
        std::map<std::string, int64_t> ext = {{"K0", 3}, {"M0", 5}, {"M1", 8}};
        *s = TensorPackSpec::make({"M1", "K0", "M0"}, {"K0", "M1", "M0"}, ext, nullptr, "fp16");
        return true;
    }
    case 11: /* elements_bf16 */
    {
        std::map<std::string, int64_t> ext = {{"K0", 3}, {"M0", 5}, {"M1", 8}};
        *s = TensorPackSpec::make({"M1", "K0", "M0"}, {"K0", "M1", "M0"}, ext, nullptr, "bf16");
        return true;
    }
    case 12: /* force_elements */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"K1", "M1", "K0", "M0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        s->form = "elements";
        return true;
    }
    case 13: /* force_pairs */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"K1", "M1", "K0", "M0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        s->form = "pairs";
        return true;
    }
    case 14: /* strided_pairs */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        std::map<std::string, int64_t> st = {{"K0", 64}, {"K1", 52000}, {"M0", 1}, {"M1", 1600}};
        *s = TensorPackSpec::make(
            {"K1", "M1", "K0", "M0"}, {"K1", "K0", "M1", "M0"}, ext, &st, "fp16");
        return true;
    }
    case 15: /* strided_tiled */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        std::map<std::string, int64_t> st = {{"K0", 1}, {"K1", 32}, {"M0", 800}, {"M1", 40000}};
        *s = TensorPackSpec::make(
            {"M1", "M0", "K1", "K0"}, {"K1", "K0", "M1", "M0"}, ext, &st, "fp16");
        return true;
    }
    case 16: /* strided_elems */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        std::map<std::string, int64_t> st = {{"K0", 3}, {"K1", 80}, {"M0", 2000}, {"M1", 100000}};
        *s = TensorPackSpec::make(
            {"M1", "M0", "K1", "K0"}, {"K1", "K0", "M1", "M0"}, ext, &st, "fp16");
        return true;
    }
    case 17: /* bad_numel */
    {
        std::map<std::string, int64_t> ext = {{"K0", 3}, {"M0", 5}};
        *s = TensorPackSpec::make({"M0", "K0"}, {"K0", "M0"}, ext, nullptr, "fp16");
        return true;
    }
    case 18: /* bad_tiled_block */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"M1", "M0", "K1", "K0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        s->form = "tiled";
        s->block = 128;
        return true;
    }
    case 19: /* bad_pairs */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"M1", "M0", "K1", "K0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        s->form = "pairs";
        return true;
    }
    case 20: /* bad_groups */
    {
        std::map<std::string, int64_t> ext = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}};
        *s = TensorPackSpec::make(
            {"K1", "M1", "K0", "M0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        s->groups = 3;
        return true;
    }
    case 21: /* bad_modes */
    {
        std::map<std::string, int64_t> ext
            = {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"M2", 2}};
        *s = TensorPackSpec::make(
            {"K1", "M1", "K0", "M0"}, {"K1", "K0", "M1", "M2"}, ext, nullptr, "fp16");
        return true;
    }
    case 22: /* bad_span */
    {
        std::map<std::string, int64_t> ext = {{"K0", 1024}, {"K1", 2}, {"M0", 1024}, {"M1", 1024}};
        *s = TensorPackSpec::make(
            {"M1", "M0", "K1", "K0"}, {"K1", "K0", "M1", "M0"}, ext, nullptr, "fp16");
        return true;
    }
    default:
        return false;
    }
}

rocke_kernel_def_t* build(rocke_ir_builder_t* b, int idx, int* known)
{
    TensorPackSpec s;
    if(!make(idx, &s))
        return NULL;
    *known = 1;
    std::string why;
    if(!pack_is_valid_spec(s, "gfx950", &why))
    {
        fprintf(stderr, "invalid spec: %s\n", why.c_str());
        exit(1);
    }
    return build_tensor_pack(b, s, "gfx950");
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: tensor_pack_emit <config_index> [ll|ir|verify]\n");
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    const char* arch = "gfx950";
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof b);
    int known = 0;
    rocke_kernel_def_t* kernel = build(&b, idx, &known);
    if(!known)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
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
