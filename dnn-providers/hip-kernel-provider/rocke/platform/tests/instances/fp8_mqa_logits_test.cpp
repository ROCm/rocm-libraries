// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only coverage for the gfx942 FP8-logits C ABI and builder.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/instance_gfx942_fp8_mqa_logits.h"

static int check(bool condition, const char* message)
{
    if(!condition)
    {
        fprintf(stderr, "FAIL: %s\n", message);
        return 1;
    }
    return 0;
}

static int check_lower(rocke_fp8_mqa_logits_spec_t spec)
{
    char* llvm_text = NULL;
    char err[ROCKE_ERR_MSG_CAP];
    rocke_status_t status;
    err[0] = '\0';
    status = rocke_fp8_mqa_logits_lower_to_llvm(
        &spec, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof(err));
    if(status != ROCKE_OK || llvm_text == NULL)
    {
        fprintf(stderr, "FAIL: lower status=%d err=%s\n", (int)status, err);
        free(llvm_text);
        return 1;
    }
    if(strstr(llvm_text, "@llvm.amdgcn.mfma.f32.16x16x32.fp8") == NULL)
    {
        fprintf(stderr, "FAIL: expected FP8 MFMA intrinsic\n");
        free(llvm_text);
        return 1;
    }
    free(llvm_text);
    return 0;
}

int main(void)
{
    rocke_fp8_mqa_logits_spec_t spec = rocke_fp8_mqa_logits_spec_default();
    char name[256];
    char reason[256];
    int grid[3];
    rocke_arena_t arena = {0};
    const rocke_sig_entry_t* signature = NULL;
    size_t signature_count = 0;
    int failed = 0;

    failed |= check(rocke_fp8_mqa_logits_block_size(&spec) == 256, "default block size");
    failed |= check(rocke_fp8_mqa_logits_kernel_name(&spec, name, sizeof(name)) == ROCKE_OK,
                    "kernel name status");
    failed |= check(strcmp(name, "rocke_fp8_mqa_logits_H64_D128_BKV128_R2_W4") == 0,
                    "default kernel name");
    failed |= check(rocke_fp8_mqa_logits_is_valid_spec(&spec, "gfx942", reason, sizeof(reason)),
                    "default spec validity");
    failed |= check(!rocke_fp8_mqa_logits_is_valid_spec(&spec, "gfx950", reason, sizeof(reason)),
                    "target rejection");
    failed |= check(rocke_fp8_mqa_logits_grid(16, 3, &spec, grid) == ROCKE_OK, "grid status");
    failed |= check(grid[0] == 8 && grid[1] == 3 && grid[2] == 1, "grid values");
    failed |= check(rocke_fp8_mqa_logits_grid(15, 1, &spec, grid) == ROCKE_ERR_VALUE,
                    "grid padding rejection");
    failed |= check(rocke_fp8_mqa_logits_num_splits(2, 128, 2, 128, 1) == 1,
                    "small-window split count");

    failed |= check(rocke_arena_init(&arena, 0) == 0, "signature arena init");
    if(!failed)
    {
        failed |= check(rocke_fp8_mqa_logits_signature(&arena, &spec, &signature, &signature_count)
                            == ROCKE_OK,
                        "signature status");
        failed |= check(signature_count == 11, "signature count");
        failed |= check(strcmp(signature[0].name, "Q") == 0
                            && strcmp(signature[10].name, "num_splits") == 0,
                        "signature order");
    }
    rocke_arena_destroy(&arena);

    failed |= check_lower(spec);
    spec.waves_per_block = 2;
    failed |= check_lower(spec);
    spec = rocke_fp8_mqa_logits_spec_default();
    spec.head_dim = 64;
    failed |= check_lower(spec);
    return failed ? 1 : 0;
}
