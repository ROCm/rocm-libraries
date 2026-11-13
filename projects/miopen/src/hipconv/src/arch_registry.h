#pragma once

// Per-architecture dispatch table, keyed by gfx name.
//
// Defined by the generated arch_registry_gen.cpp, whose extern references to
// each arch's ConvAlgorithms force the linker to pull the arch objects out of
// the static library (no --whole-archive needed).

#include "algorithm.h"         // ConvAlgorithm
#include "hipconv/hipconv.hpp" // hipconv::Algorithm
#include "internal_api.h"

#include <cstddef>
#include <span>
#include <string_view>

namespace hipconv
{

// One algorithm offered by an architecture. impl is never null.
struct ArchAlgorithm
{
    Algorithm algorithm;
    const ConvAlgorithm* impl;
};

// One architecture: its gfx name and its algorithms in preference order.
struct ArchEntry
{
    std::string_view name;
    std::span<const ArchAlgorithm> algorithms;
};

} // namespace hipconv

// Global-scope symbols (unmangled by namespace) so the generated TU's extern
// references keep pulling the arch objects out of the static library.
extern HIPCONV_INTERNAL_API const hipconv::ArchEntry hipconv_arch_registry[];
extern HIPCONV_INTERNAL_API const std::size_t hipconv_arch_registry_size;
