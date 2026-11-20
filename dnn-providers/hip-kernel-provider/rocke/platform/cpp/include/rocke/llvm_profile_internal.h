/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/llvm_profile_internal.h -- PRIVATE compiler-lane and LLVM emission-profile model.
 * Shared by the native LLVM lowerer and ISA adapter; not a public ABI.
 */
#ifndef ROCKE_LLVM_PROFILE_INTERNAL_H
#define ROCKE_LLVM_PROFILE_INTERNAL_H

#include <stdbool.h>

#include "rocke/lower_llvm.h"

namespace ckc
{

typedef enum rocke_ll_datalayout_profile
{
    ROCKE_LL_DATALAYOUT_PROFILE_LLVM20,
    ROCKE_LL_DATALAYOUT_PROFILE_LLVM22,
    ROCKE_LL_DATALAYOUT_PROFILE_LLVM23
} rocke_ll_datalayout_profile_t;

typedef enum rocke_ll_declaration_abi
{
    ROCKE_LL_DECLARATION_ABI_LLVM20,
    ROCKE_LL_DECLARATION_ABI_LLVM22,
    ROCKE_LL_DECLARATION_ABI_LLVM23,
    ROCKE_LL_DECLARATION_ABI_LLVM24
} rocke_ll_declaration_abi_t;

typedef enum rocke_ll_emission_capability
{
    ROCKE_LL_CAP_LLVM23_TARGET_OPS = 1u << 0,
    ROCKE_LL_CAP_NAMED_BARRIER_AS15 = 1u << 1
} rocke_ll_emission_capability_t;

typedef struct rocke_ll_emission_profile
{
    rocke_ll_datalayout_profile_t datalayout;
    rocke_ll_declaration_abi_t declarations;
    unsigned capabilities;
} rocke_ll_emission_profile_t;

/* Borrowed process-lifetime profile. LLVM24 retains LLVM23's datalayout and
 * target-op capability but has a distinct named-barrier declaration ABI. */
const rocke_ll_emission_profile_t* rocke_ll_emission_profile(rocke_llvm_flavor_t flavor);
bool rocke_ll_profile_has(rocke_llvm_flavor_t flavor, rocke_ll_emission_capability_t capability);
bool rocke_ll_flavor_is_modern(rocke_llvm_flavor_t flavor);
bool rocke_ll_flavor_at_least(rocke_llvm_flavor_t flavor, rocke_llvm_flavor_t required);
rocke_llvm_flavor_t rocke_ll_flavor_for_llvm_major(unsigned major);
rocke_llvm_flavor_t rocke_ll_explicit_flavor(const char* value);

extern const char* const ROCKE_LL_DATALAYOUT_LLVM20;
extern const char* const ROCKE_LL_DATALAYOUT_LLVM22;
extern const char* const ROCKE_LL_DATALAYOUT_LLVM23;
extern const char* const ROCKE_LL_DATALAYOUT;
extern const char* const ROCKE_LL_TRIPLE;

/* Borrowed process-lifetime string, or NULL for AUTO/an unknown flavor. */
const char* rocke_ll_datalayout_for_flavor(rocke_llvm_flavor_t flavor);

} // namespace ckc

#endif // ROCKE_LLVM_PROFILE_INTERNAL_H
