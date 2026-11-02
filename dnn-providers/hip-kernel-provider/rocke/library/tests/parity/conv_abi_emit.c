/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_abi_emit.c -- C-side emitter for the conv kernarg ABI
 * parity family. Prints the same list as conv_abi_emit.py for each config
 * index, one "name kind" line per entry, so the two can be byte-compared.
 *
 * Config index table (must stay in sync with the Python emitter):
 *   0  fwd              1  fwd 3-D
 *   2  wgrad            3  wgrad 3-D
 *   4  wgrad two_stage  5  wgrad 3-D two_stage
 *   6  dgrad            7  dgrad 3-D
 *   8  direct fwd       9  direct dgrad
 *  10  fwd problem block          11  fwd problem block 3-D
 *  12  deep_fused_conv_pool signature (2-D)
 *  13  fwd two_stage (rejected)   14  direction "bwd" (rejected)
 *  15  direct wgrad                16  direct "bwd" (rejected)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/helper_rocke.instances.common.deep_fused_conv_pool.h"
#include "rocke/instance_conv_abi.h"

static void print_list(const rocke_conv_arg_list_t* l)
{
    for(int i = 0; i < l->count; ++i)
        printf("%s %s\n", l->items[i].name, rocke_conv_arg_kind_str(l->items[i].kind));
}

static int print_deep_signature(void)
{
    rocke_arena_t arena;
    rocke_deep_fused_conv_pool_spec_t spec;
    const rocke_sig_entry_t* items = NULL;
    size_t count = 0;
    if(rocke_arena_init(&arena, 0) != 0)
        return 1;
    /* The signature depends only on whether conv0 is 3-D. */
    memset(&spec, 0, sizeof(spec));
    spec.problem.conv.is_3d = false;
    if(rocke_deep_fused_conv_pool_signature(&arena, &spec, &items, &count) != ROCKE_OK)
    {
        rocke_arena_destroy(&arena);
        return 1;
    }
    for(size_t i = 0; i < count; ++i)
        printf("%s %s\n", items[i].name, items[i].type);
    rocke_arena_destroy(&arena);
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [ll]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0)
    {
        /* The lists are not IR; there is nothing to serialize or verify. */
        fprintf(stderr, "unknown mode '%s'\n", mode);
        return 2;
    }

    rocke_conv_arg_list_t list;
    bool ok = true;
    switch(idx)
    {
    case 0:
        ok = rocke_conv_arg_names("fwd", false, false, &list);
        break;
    case 1:
        ok = rocke_conv_arg_names("fwd", true, false, &list);
        break;
    case 2:
        ok = rocke_conv_arg_names("wgrad", false, false, &list);
        break;
    case 3:
        ok = rocke_conv_arg_names("wgrad", true, false, &list);
        break;
    case 4:
        ok = rocke_conv_arg_names("wgrad", false, true, &list);
        break;
    case 5:
        ok = rocke_conv_arg_names("wgrad", true, true, &list);
        break;
    case 6:
        ok = rocke_conv_arg_names("dgrad", false, false, &list);
        break;
    case 7:
        ok = rocke_conv_arg_names("dgrad", true, false, &list);
        break;
    case 8:
        ok = rocke_conv_direct_arg_names("fwd", &list);
        break;
    case 9:
        ok = rocke_conv_direct_arg_names("dgrad", &list);
        break;
    case 10:
        rocke_conv_fwd_problem_block(false, &list);
        break;
    case 11:
        rocke_conv_fwd_problem_block(true, &list);
        break;
    case 12:
        return print_deep_signature();
    case 13:
        ok = rocke_conv_arg_names("fwd", false, true, &list);
        break;
    case 14:
        ok = rocke_conv_arg_names("bwd", false, false, &list);
        break;
    case 15:
        ok = rocke_conv_direct_arg_names("wgrad", &list);
        break;
    case 16:
        ok = rocke_conv_direct_arg_names("bwd", &list);
        break;
    default:
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    if(!ok)
    {
        printf("REJECTED\n");
        return 0;
    }
    print_list(&list);
    return 0;
}
