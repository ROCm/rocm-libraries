/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/arch_num_cus.h -- the static CU count per arch of
 * helpers/split_k.py (arch_num_cus: _ARCH_NUM_CUS, 256 when unknown). Build
 * time policies that size persistent grids without a device (the emulated-f32
 * GEMM bodies, hipTensor's kernel pack) read it.
 */
#ifndef ROCKE_ARCH_NUM_CUS_H
#define ROCKE_ARCH_NUM_CUS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Python _DEFAULT_NUM_CUS */
#define ROCKE_DEFAULT_NUM_CUS 256

/* Python arch_num_cus(arch). NULL -> the default. */
int rocke_arch_num_cus(const char* arch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_ARCH_NUM_CUS_H */
