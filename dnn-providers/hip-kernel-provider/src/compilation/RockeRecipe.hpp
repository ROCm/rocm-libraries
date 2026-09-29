// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime_api.h>
#include <memory>
#include <rocke/recipe_launch.h>
#include <string>
#include <vector>

namespace hip_kernel_provider::compilation
{
using RecipeBytes = std::vector<unsigned char>;
using RecipeLaunchPlan = std::unique_ptr<rocke_launch_plan_t, decltype(&rocke_launch_plan_free)>;

bool recipeAdmits(const RecipeBytes& bytes,
                  const std::string& key,
                  const std::string& arch,
                  int64_t sequence);

// Owns metadata and module for the entire prepared-plan lifetime.
class RockeRecipe
{
public:
    RockeRecipe(const RecipeBytes& bytes,
                const std::string& key,
                const std::string& arch,
                int64_t sequence);
    ~RockeRecipe();
    RockeRecipe(const RockeRecipe&) = delete;
    RockeRecipe& operator=(const RockeRecipe&) = delete;
    const rocke_launch_plan_t* plan() const
    {
        return _plan.get();
    }
    void launch(void** extra, hipStream_t stream) const;

private:
    RecipeLaunchPlan _plan{nullptr, rocke_launch_plan_free};
    rocke_launch_dims_t _grid{}, _block{};
    unsigned _lds = 0;
    hipModule_t _module = nullptr;
    hipFunction_t _function = nullptr;
};
} // namespace hip_kernel_provider::compilation
