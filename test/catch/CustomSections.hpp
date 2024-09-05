
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>

/**
 * Parameterizes the current test with `GENERATE` and enters a `SECTION` for each GPU
 * target supported by rocRoller.
 *
 * Declares a `GPUArchitectureTarget` variable in the current scope named `varname`.
 */
#define SUPPORTED_ARCH_SECTION(varname)                                                       \
    std::string varname_ = GENERATE(                                                          \
        from_range(rocRoller::GPUArchitectureLibrary::getInstance()->getAllSupportedISAs())); \
    GPUArchitectureTarget varname;                                                            \
    varname = varname_;                                                                       \
    SECTION(varname_)
