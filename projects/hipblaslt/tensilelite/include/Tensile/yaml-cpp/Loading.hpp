// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/Tensile.hpp>

#include <tensilelitehost/export.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace TensileLite
{
    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        YamlCppLoadLibraryFile(std::string const&                  filename,
                               const std::vector<LazyLoadingInit>& preloaded = {});

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        YamlCppLoadLibraryData(std::vector<uint8_t> const& data, std::string filename = "");
} // namespace TensileLite
