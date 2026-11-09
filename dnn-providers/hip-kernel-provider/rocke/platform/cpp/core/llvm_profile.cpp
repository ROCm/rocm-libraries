// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * llvm_profile.cpp -- native compiler-lane and structural LLVM emission-profile registry.
 * This neutral layer is shared by the LLVM lowerer and ISA adapter.
 */
#include "rocke/llvm_profile_internal.h"

#include <cstring>
#include <string>

#include "rocke/error.hpp"

namespace
{

static const ckc::rocke_ll_emission_profile_t ROCKE_LL_PROFILE_LLVM20
    = {ckc::ROCKE_LL_DATALAYOUT_PROFILE_LLVM20, ckc::ROCKE_LL_DECLARATION_ABI_LLVM20, 0};
static const ckc::rocke_ll_emission_profile_t ROCKE_LL_PROFILE_LLVM22
    = {ckc::ROCKE_LL_DATALAYOUT_PROFILE_LLVM22, ckc::ROCKE_LL_DECLARATION_ABI_LLVM22, 0};
static const ckc::rocke_ll_emission_profile_t ROCKE_LL_PROFILE_LLVM23
    = {ckc::ROCKE_LL_DATALAYOUT_PROFILE_LLVM23,
       ckc::ROCKE_LL_DECLARATION_ABI_LLVM23,
       ckc::ROCKE_LL_CAP_LLVM23_TARGET_OPS};
static const ckc::rocke_ll_emission_profile_t ROCKE_LL_PROFILE_LLVM24
    = {ckc::ROCKE_LL_DATALAYOUT_PROFILE_LLVM23,
       ckc::ROCKE_LL_DECLARATION_ABI_LLVM24,
       ckc::ROCKE_LL_CAP_LLVM23_TARGET_OPS | ckc::ROCKE_LL_CAP_NAMED_BARRIER_AS15};

static const struct
{
    rocke_llvm_flavor_t flavor;
    const char* name;
    unsigned min_llvm_major;
    const ckc::rocke_ll_emission_profile_t* profile;
} ROCKE_LL_FLAVOR_LADDER[] = {
    {ROCKE_LLVM_FLAVOR_LLVM20, "llvm20", 0, &ROCKE_LL_PROFILE_LLVM20},
    {ROCKE_LLVM_FLAVOR_LLVM22, "llvm22", 21, &ROCKE_LL_PROFILE_LLVM22},
    {ROCKE_LLVM_FLAVOR_LLVM23, "llvm23", 23, &ROCKE_LL_PROFILE_LLVM23},
    {ROCKE_LLVM_FLAVOR_LLVM24, "llvm24", 24, &ROCKE_LL_PROFILE_LLVM24},
};

static const int ROCKE_LL_FLAVOR_LADDER_COUNT
    = (int)(sizeof(ROCKE_LL_FLAVOR_LADDER) / sizeof(ROCKE_LL_FLAVOR_LADDER[0]));

} // namespace

int rocke_llvm_flavor_count(void)
{
    return ROCKE_LL_FLAVOR_LADDER_COUNT;
}

const char* rocke_llvm_flavor_at(int index)
{
    if(index < 0 || index >= ROCKE_LL_FLAVOR_LADDER_COUNT)
        return "";
    return ROCKE_LL_FLAVOR_LADDER[index].name;
}

const char* rocke_llvm_flavor_name(rocke_llvm_flavor_t flavor)
{
    for(int i = 0; i < ROCKE_LL_FLAVOR_LADDER_COUNT; ++i)
    {
        if(ROCKE_LL_FLAVOR_LADDER[i].flavor == flavor)
            return ROCKE_LL_FLAVOR_LADDER[i].name;
    }
    return "";
}

rocke_llvm_flavor_t rocke_llvm_flavor_from_name(const char* name)
{
    if(name)
    {
        for(int i = 0; i < ROCKE_LL_FLAVOR_LADDER_COUNT; ++i)
        {
            if(std::strcmp(name, ROCKE_LL_FLAVOR_LADDER[i].name) == 0)
                return ROCKE_LL_FLAVOR_LADDER[i].flavor;
        }
    }
    return ROCKE_LLVM_FLAVOR_AUTO;
}

bool rocke_llvm_flavor_is_known(rocke_llvm_flavor_t flavor)
{
    for(int i = 0; i < ROCKE_LL_FLAVOR_LADDER_COUNT; ++i)
    {
        if(ROCKE_LL_FLAVOR_LADDER[i].flavor == flavor)
            return true;
    }
    return false;
}

namespace ckc
{

const rocke_ll_emission_profile_t* rocke_ll_emission_profile(rocke_llvm_flavor_t flavor)
{
    for(int i = 0; i < ROCKE_LL_FLAVOR_LADDER_COUNT; ++i)
    {
        if(ROCKE_LL_FLAVOR_LADDER[i].flavor == flavor)
            return ROCKE_LL_FLAVOR_LADDER[i].profile;
    }
    return NULL;
}

bool rocke_ll_profile_has(rocke_llvm_flavor_t flavor, rocke_ll_emission_capability_t capability)
{
    const rocke_ll_emission_profile_t* profile = rocke_ll_emission_profile(flavor);
    return profile && (profile->capabilities & (unsigned)capability) != 0;
}

bool rocke_ll_flavor_is_modern(rocke_llvm_flavor_t flavor)
{
    const rocke_ll_emission_profile_t* profile = rocke_ll_emission_profile(flavor);
    return profile && profile->datalayout != ROCKE_LL_DATALAYOUT_PROFILE_LLVM20;
}

bool rocke_ll_flavor_at_least(rocke_llvm_flavor_t flavor, rocke_llvm_flavor_t required)
{
    int flavor_rank = -1;
    int required_rank = -1;
    for(int i = 0; i < ROCKE_LL_FLAVOR_LADDER_COUNT; ++i)
    {
        if(ROCKE_LL_FLAVOR_LADDER[i].flavor == flavor)
            flavor_rank = i;
        if(ROCKE_LL_FLAVOR_LADDER[i].flavor == required)
            required_rank = i;
    }
    return flavor_rank >= 0 && required_rank >= 0 && flavor_rank >= required_rank;
}

rocke_llvm_flavor_t rocke_ll_explicit_flavor(const char* value)
{
    if(!value || value[0] == '\0')
        return ROCKE_LLVM_FLAVOR_AUTO;
    rocke_llvm_flavor_t flavor = rocke_llvm_flavor_from_name(value);
    if(flavor == ROCKE_LLVM_FLAVOR_AUTO)
    {
        std::string expected = "(";
        for(int i = 0; i < ROCKE_LL_FLAVOR_LADDER_COUNT; ++i)
        {
            if(i)
                expected += ", ";
            expected += "'" + std::string(ROCKE_LL_FLAVOR_LADDER[i].name) + "'";
        }
        expected += ")";
        throw ckc::ValueError("ROCKE_LLVM_FLAVOR='" + std::string(value)
                              + "' is not a recognized flavor; expected one of " + expected);
    }
    return flavor;
}

rocke_llvm_flavor_t rocke_ll_flavor_for_llvm_major(unsigned major)
{
    const int latest = ROCKE_LL_FLAVOR_LADDER_COUNT - 1;
    if(major > ROCKE_LL_FLAVOR_LADDER[latest].min_llvm_major)
    {
        throw ckc::ValueError(
            ckc::format_error("LLVM compiler major %u is newer than the latest qualified lane %s",
                              major,
                              ROCKE_LL_FLAVOR_LADDER[latest].name));
    }
    for(int i = latest; i >= 0; --i)
    {
        if(major >= ROCKE_LL_FLAVOR_LADDER[i].min_llvm_major)
            return ROCKE_LL_FLAVOR_LADDER[i].flavor;
    }
    return ROCKE_LLVM_FLAVOR_LLVM20;
}

const char* const ROCKE_LL_DATALAYOUT_LLVM20
    = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
      "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32"
      "-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
      "-n32:64-S32-A5-G1-ni:7:8:9";

const char* const ROCKE_LL_DATALAYOUT_LLVM22
    = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
      "-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32"
      "-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
      "-n32:64-S32-A5-G1-ni:7:8:9";

const char* const ROCKE_LL_DATALAYOUT_LLVM23
    = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
      "-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32"
      "-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32"
      "-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
      "-n32:64-S32-A5-G1-ni:7:8:9";

const char* const ROCKE_LL_DATALAYOUT = ROCKE_LL_DATALAYOUT_LLVM20;
const char* const ROCKE_LL_TRIPLE = "amdgcn-amd-amdhsa";

const char* rocke_ll_datalayout_for_flavor(rocke_llvm_flavor_t flavor)
{
    const rocke_ll_emission_profile_t* profile = rocke_ll_emission_profile(flavor);
    if(!profile)
        return NULL;
    if(profile->datalayout == ROCKE_LL_DATALAYOUT_PROFILE_LLVM20)
        return ROCKE_LL_DATALAYOUT_LLVM20;
    if(profile->datalayout == ROCKE_LL_DATALAYOUT_PROFILE_LLVM23)
        return ROCKE_LL_DATALAYOUT_LLVM23;
    return ROCKE_LL_DATALAYOUT_LLVM22;
}

} // namespace ckc
