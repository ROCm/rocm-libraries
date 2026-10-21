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
 *  12  fwd two_stage (rejected)   13  direction "bwd" (rejected)
 *  14  direct wgrad                15  direct "bwd" (rejected)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/instance_conv_abi.h"

static void print_list(const rocke_conv_arg_list_t* l)
{
    for(int i = 0; i < l->count; ++i)
        printf("%s %s\n", l->items[i].name, rocke_conv_arg_kind_str(l->items[i].kind));
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
        ok = rocke_conv_arg_names("fwd", false, true, &list);
        break;
    case 13:
        ok = rocke_conv_arg_names("bwd", false, false, &list);
        break;
    case 14:
        ok = rocke_conv_direct_arg_names("wgrad", &list);
        break;
    case 15:
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
