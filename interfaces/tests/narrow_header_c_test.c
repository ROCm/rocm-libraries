// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm/interfaces/experimental/blas_narrow_v2.h"

int main(void) {
    rocm_blas_v2_provider table = {0};
    return table.header.struct_size != 0;
}
