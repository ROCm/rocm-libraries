// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_barrier_fence.cpp -- the fence attr of tile.sync / tile.sync_lds_only /
 * tile.s_barrier_bare in the C++ engine (mirror of
 * tests/core/test_barrier_fence.py).
 *
 * LLVM: fence="compiler" follows the s.barrier call with a compiler memory
 * barrier (inline asm with a memory clobber); fence="workgroup" also brackets
 * it with workgroup release / acquire fences. HIP: both add
 * __asm__ volatile("" ::: "memory") after the barrier; s_barrier_bare under
 * "workgroup" also gets the two __builtin_amdgcn_fence calls __syncthreads
 * carries. An unfenced barrier keeps its historical emission.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/strbuf.h"

static const char* const REL = "  fence syncscope(\"workgroup\") release";
static const char* const BAR = " call void @llvm.amdgcn.s.barrier()";
static const char* const ACQ = "  fence syncscope(\"workgroup\") acquire";
static const char* const CLOB = "  call void asm sideeffect \"\", \"~{memory}\"()";
static const char* const HIP_CLOB = "__asm__ volatile(\"\" ::: \"memory\");";

/* fence mode of the six barriers built by build(): 0 none, 1 compiler,
 * 2 workgroup */
static const int MODES[6] = {0, 1, 0, 2, 0, 2};

static int count(const char* text, const char* needle)
{
    int n = 0;
    for(const char* p = strstr(text, needle); p; p = strstr(p + 1, needle))
    {
        ++n;
    }
    return n;
}

static int split_lines(char* text, char** lines, int cap)
{
    int n = 0;
    for(char* p = text; p && *p && n < cap;)
    {
        lines[n++] = p;
        char* nl = strchr(p, '\n');
        if(!nl)
        {
            break;
        }
        *nl = '\0';
        p = nl + 1;
    }
    return n;
}

static void build(rocke_ir_builder_t* b)
{
    rocke_b_sync(b);
    rocke_b_sync_fence(b, "compiler");
    rocke_b_sync_lds_only(b);
    rocke_b_sync_lds_only_fence(b, "workgroup");
    rocke_b_s_barrier_bare_fence(b, "none");
    rocke_b_s_barrier_bare_fence(b, "workgroup");
}

static int check_llvm(void)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "fence_probe") != ROCKE_OK)
    {
        return 1;
    }
    build(&b);
    char* text = NULL;
    int bad = !rocke_ir_builder_ok(&b);
    if(!bad)
    {
        bad = rocke_lower_kernel_to_llvm(
                  rocke_ir_builder_kernel(&b), ROCKE_LLVM_FLAVOR_LLVM22, "gfx942", &text)
                  != ROCKE_OK
              || text == NULL;
    }
    if(!bad)
    {
        bad |= count(text, "fence syncscope(\"workgroup\") release") != 2;
        bad |= count(text, "fence syncscope(\"workgroup\") acquire") != 2;
        bad |= count(text, "~{memory}") != 3;
        static char* lines[4096];
        int n = split_lines(text, lines, 4096);
        int k = 0;
        for(int i = 0; i < n; ++i)
        {
            if(strcmp(lines[i], BAR) != 0)
            {
                continue;
            }
            if(k >= 6)
            {
                bad = 1;
                break;
            }
            int mode = MODES[k];
            bool rel = i > 0 && strcmp(lines[i - 1], REL) == 0;
            bool acq = i + 1 < n && strcmp(lines[i + 1], ACQ) == 0;
            int clob_at = mode == 2 ? i + 2 : i + 1;
            bool clob = clob_at < n && strcmp(lines[clob_at], CLOB) == 0;
            if(rel != (mode == 2) || acq != (mode == 2) || clob != (mode != 0))
            {
                fprintf(stderr, "barrier %d: fence placement wrong\n", k);
                bad = 1;
            }
            ++k;
        }
        bad |= k != 6;
    }
    free(text);
    rocke_ir_builder_free(&b);
    return bad;
}

static int check_hip(void)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "fence_probe") != ROCKE_OK)
    {
        return 1;
    }
    build(&b);
    rocke_strbuf_t text;
    if(rocke_strbuf_init(&text, 256) != 0)
    {
        rocke_ir_builder_free(&b);
        return 1;
    }
    rocke_lower_hip_opts_t opts = {};
    opts.arch = "gfx942";
    opts.include_prologue = false;
    opts.include_prologue_set = true;
    int bad = rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &text) != ROCKE_OK;
    if(!bad)
    {
        const char* s = rocke_strbuf_cstr(&text);
        bad |= count(s, HIP_CLOB) != 3;
        bad |= count(s, "__builtin_amdgcn_fence(__ATOMIC_RELEASE, \"workgroup\");") != 1;
        bad |= count(s, "__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, \"workgroup\");") != 1;
        bad |= count(s, "__syncthreads();") != 4;
        bad |= count(s, "__builtin_amdgcn_s_barrier();") != 2;
    }
    rocke_strbuf_free(&text);
    rocke_ir_builder_free(&b);
    return bad;
}

static int check_rejections(void)
{
    /* an unknown mode is a builder error */
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "fence_bad") != ROCKE_OK)
    {
        return 1;
    }
    bool rejected = false;
    try
    {
        rocke_b_sync_fence(&b, "global");
        rejected = rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE;
    }
    catch(const ckc::Error& error)
    {
        rejected = error.code() == ROCKE_ERR_VALUE;
    }
    int bad = !rejected;
    rocke_ir_builder_free(&b);

    /* an unknown attr value (serialized IR does not pass the builder) fails
     * the lowering */
    if(rocke_ir_builder_init(&b, "fence_bad_attr") != ROCKE_OK)
    {
        return 1;
    }
    rocke_attr_map_t attrs = rocke_i_attrs(&b);
    rocke_attr_set_str(&b, &attrs, "fence", "global");
    rocke_i_op0(&b, ROCKE_OP_TILE_SYNC, NULL, 0, &attrs);
    char* text = NULL;
    rejected = false;
    try
    {
        rejected = rocke_lower_kernel_to_llvm(
                       rocke_ir_builder_kernel(&b), ROCKE_LLVM_FLAVOR_LLVM22, "gfx942", &text)
                   != ROCKE_OK;
    }
    catch(const ckc::Error& error)
    {
        rejected = error.code() == ROCKE_ERR_VALUE;
    }
    bad |= !rejected;
    free(text);
    rocke_ir_builder_free(&b);
    return bad;
}

int main(void)
{
    if(check_llvm() || check_hip() || check_rejections())
    {
        fprintf(stderr, "barrier fence lowering contract failed\n");
        return 1;
    }
    return 0;
}
