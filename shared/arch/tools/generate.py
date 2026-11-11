#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate build-directory C++ and CMake views of the packaged JSON catalog."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from rocm_arch import catalog_digest, load_catalog  # noqa: E402


def generate(directory, output):
    records = load_catalog(directory)
    digest = catalog_digest(records)
    processors = [name for name, kind in records if kind == "processor"]
    generics = [name for name, kind in records if kind == "generic"]
    header = """// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Generated from architecture definitions. Do not edit.
#pragma once
#include <array>
#include <cstddef>
#if __cplusplus >= 201703L
#define ROCM_ARCH_INLINE inline
#else
#define ROCM_ARCH_INLINE
#endif
namespace rocm { namespace arch {
"""
    header += f'ROCM_ARCH_INLINE constexpr char catalog_digest[] = "{digest}";\n'
    header += "namespace names {\n"
    identifiers = set()
    for name, _ in records:
        identifier = name.replace("-", "_")
        if identifier in identifiers:
            raise ValueError(f"C++ name collision: {name}")
        identifiers.add(identifier)
        header += f'ROCM_ARCH_INLINE constexpr char {identifier}[] = "{name}";\n'
    header += "} // namespace names\n"
    header += (
        "// Identity ordinals are catalog-local; persist canonical names instead.\n"
    )
    header += "enum class architecture_t {\n"
    header += "".join(f"    {name.replace('-', '_')},\n" for name in processors)
    header += "    Count\n};\n"
    values = ", ".join(
        f"architecture_t::{name.replace('-', '_')}" for name in processors
    )
    header += f"ROCM_ARCH_INLINE constexpr std::array<architecture_t, {len(processors)}> architecture_ids = {{{{{values}}}}};\n"
    for symbol, names in (("architectures", processors), ("generic_targets", generics)):
        literals = ", ".join(f'names::{name.replace("-", "_")}' for name in names)
        header += f"ROCM_ARCH_INLINE constexpr std::array<const char*, {len(names)}> {symbol} = {{{{{literals}}}}};\n"
    header += """static constexpr bool equal(const char* lhs, const char* rhs)
{
    if (!lhs || !rhs) return false;
    while (*lhs && *lhs == *rhs) { ++lhs; ++rhs; }
    return *lhs == *rhs;
}
static constexpr const char* name(architecture_t architecture)
{
    auto index = static_cast<std::size_t>(architecture);
    return index < architectures.size() ? architectures[index] : nullptr;
}
static constexpr architecture_t from_name(const char* name)
{
    for (std::size_t i = 0; i < architectures.size(); ++i)
        if (equal(name, architectures[i])) return architecture_ids[i];
    return architecture_t::Count;
}
static constexpr bool is_known(const char* name)
{
    for (std::size_t i = 0; i < architectures.size(); ++i)
        if (equal(name, architectures[i])) return true;
    for (std::size_t i = 0; i < generic_targets.size(); ++i)
        if (equal(name, generic_targets[i])) return true;
    return false;
}
}} // namespace rocm::arch
#undef ROCM_ARCH_INLINE
"""
    cmake = "# Generated from architecture definitions. Do not edit.\n"
    cmake += f'set(ROCM_ARCH_CATALOG_DIGEST "{digest}")\n'
    cmake += f'set(_ROCM_ARCH_PROCESSORS "{";".join(processors)}")\n'
    cmake += f'set(_ROCM_ARCH_GENERICS "{";".join(generics)}")\n'
    for relative, content in (
        ("include/rocm/arch/catalog.hpp", header),
        ("rocm-arch-catalog.cmake", cmake),
    ):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        generate(args.catalog, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"catalog generation failed: {error}\n")


if __name__ == "__main__":
    main()
