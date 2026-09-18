// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* The public dtype API needs no architecture or IR-builder header.
 * Mirrors tests/core/test_dtypes.py, including unknown-name pass-through.
 */
#include "rocke/dtypes.h"

#include <stdio.h>
#include <string.h>

int main()
{
    const char* cases[][2] = {{" HALF ", "fp16"},
                              {"bfloat16", "bf16"},
                              {" Float\t", "fp32"},
                              {"FP8", "fp8e4m3"},
                              {"BF8", "bf8e5m2"},
                              {"FP6", "fp6e2m3"},
                              {"BF6", "fp6e3m2"},
                              {"FP4", "fp4e2m1"},
                              {"int32", ROCKE_DTYPE_I32},
                              {" Custom_Format ", "custom_format"},
                              {"", ""}};
    for(const auto& entry : cases)
    {
        char scratch[64];
        if(strcmp(rocke_normalize_dtype(entry[0], scratch, sizeof(scratch)), entry[1]) != 0
           || strcmp(rocke_normalize_dtype(entry[1], scratch, sizeof(scratch)), entry[1]) != 0)
        {
            fprintf(stderr, "dtype normalization failed: %s\n", entry[0]);
            return 1;
        }
    }
    char scratch[64];
    if(strcmp(rocke_normalize_dtype(NULL, scratch, sizeof(scratch)), "") != 0
       || strcmp(rocke_normalize_dtype(" BF6 ", NULL, 0), "fp6e3m2") != 0)
        return 1;
    return 0;
}
