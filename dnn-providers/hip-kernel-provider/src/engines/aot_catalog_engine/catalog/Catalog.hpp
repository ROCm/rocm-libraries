// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Loose-file catalog: scans <catalogDir>/<arch>/<family>/family.json and parses
// each into a Family. Forked from PR #9207's dispatcher/AotCatalog but with the
// per-arch kpack manifest scan replaced by a direct family.json -> .co lookup
// (co_file is relative to the family.json directory and absolutized on load).
//
// loadForDevice() has a NO-THROW contract: any malformed family is logged and
// skipped so one bad JSON file can never take down the whole engine.

#pragma once

#include <string>
#include <vector>

#include "catalog/CatalogTypes.hpp"

namespace aot_catalog_engine::catalog
{

class Catalog
{
public:
    Catalog() = default;

    // Test-injection ctor: build directly from families, bypassing filesystem
    // and JSON parsing (used by unit tests).
    explicit Catalog(std::vector<Family> families)
        : _families(std::move(families))
    {
    }

    // Load every family for `arch` under `catalogDir`. Never throws; on any
    // error the offending family is skipped and the rest still load.
    static Catalog loadForDevice(const std::string& catalogDir, const std::string& arch);

    const std::vector<Family>& families() const
    {
        return _families;
    }

    bool empty() const
    {
        return _families.empty();
    }

    // A kernel plus its owning family, produced by candidate selection.
    struct Candidate
    {
        const Family* family = nullptr;
        const KernelEntry* kernel = nullptr;
    };

    // All kernels whose family op_kind == opKind and whose constraints are
    // satisfied by `problem`. Pointers stay valid for this Catalog's lifetime.
    std::vector<Candidate> candidatesFor(const std::string& opKind,
                                         const ProblemShape& problem) const;

private:
    std::vector<Family> _families;
};

// Runtime catalog root: the HIPDNN_AOT_CATALOG_DIR environment variable if set,
// else the HIPDNN_AOT_CATALOG_DIR compile define baked in by CMake.
std::string defaultCatalogDir();

} // namespace aot_catalog_engine::catalog
