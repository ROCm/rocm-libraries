// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "catalog/Catalog.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <nlohmann/json.hpp>

#include "catalog/AotDebug.hpp"
#include "catalog/ModulePath.hpp"
#include "catalog/Selection.hpp"

#ifndef HIPDNN_AOT_CATALOG_DIR
#error "HIPDNN_AOT_CATALOG_DIR must be defined (set via CMake compile definition)"
#endif
#ifndef HIPDNN_AOT_CATALOG_RELDIR
#error "HIPDNN_AOT_CATALOG_RELDIR must be defined (set via CMake compile definition)"
#endif

namespace aot_catalog_engine::catalog
{

namespace fs = std::filesystem;
using nlohmann::json;

namespace
{

// ---- Small JSON accessors (throw std::runtime_error, caught per family) ------

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error("aot-catalog: " + message);
}

const json& requireMember(const json& obj, const std::string& key)
{
    if(!obj.is_object() || !obj.contains(key))
    {
        fail("missing required field '" + key + "'");
    }
    return obj.at(key);
}

std::string getRequiredString(const json& obj, const std::string& key)
{
    const json& value = requireMember(obj, key);
    if(!value.is_string())
    {
        fail("field '" + key + "' must be a string");
    }
    return value.get<std::string>();
}

// ---- Value + constraint parsing ---------------------------------------------

ShapeValue parseShapeValue(const json& value)
{
    if(value.is_boolean())
    {
        return ShapeValue{value.get<bool>()};
    }
    if(value.is_number_integer())
    {
        return ShapeValue{value.get<int64_t>()};
    }
    if(value.is_number_float())
    {
        return ShapeValue{value.get<double>()};
    }
    if(value.is_string())
    {
        return ShapeValue{value.get<std::string>()};
    }
    fail("constraint value must be a bool, number, or string");
}

ConstraintRule parseRule(const json& obj)
{
    if(!obj.is_object())
    {
        fail("constraint rule must be an object");
    }

    ConstraintRule rule;
    if(obj.contains("equals"))
    {
        rule.equals = parseShapeValue(obj.at("equals"));
    }
    if(obj.contains("not_equals"))
    {
        rule.notEquals = parseShapeValue(obj.at("not_equals"));
    }
    if(obj.contains("one_of"))
    {
        const json& arr = obj.at("one_of");
        if(!arr.is_array())
        {
            fail("'one_of' must be an array");
        }
        for(const auto& element : arr)
        {
            rule.oneOf.push_back(parseShapeValue(element));
        }
    }
    if(obj.contains("min"))
    {
        rule.min = obj.at("min").get<int64_t>();
    }
    if(obj.contains("max"))
    {
        rule.max = obj.at("max").get<int64_t>();
    }
    if(obj.contains("multiple_of"))
    {
        rule.multipleOf = obj.at("multiple_of").get<int64_t>();
    }

    if(rule.empty())
    {
        fail("constraint rule has no recognized predicate");
    }
    return rule;
}

Constraints parseConstraints(const json& obj)
{
    Constraints constraints;
    if(!obj.is_object())
    {
        fail("'constraints' must be an object");
    }
    for(const auto& [key, value] : obj.items())
    {
        constraints.emplace(key, parseRule(value));
    }
    return constraints;
}

// ---- Grid parsing -----------------------------------------------------------

GridValue parseGridValue(const json& value)
{
    GridValue gridValue;
    if(value.is_string())
    {
        gridValue.symbol = value.get<std::string>();
    }
    else if(value.is_number_integer())
    {
        gridValue.literal = value.get<int64_t>();
    }
    else
    {
        fail("grid value must be a symbol string or an integer literal");
    }
    return gridValue;
}

GridAxis parseGridAxis(const json& value)
{
    GridAxis axis;

    // Shorthand: a bare symbol/int is a VALUE axis.
    if(value.is_string() || value.is_number_integer())
    {
        axis.kind = GridAxisKind::VALUE;
        axis.value = parseGridValue(value);
        return axis;
    }

    if(!value.is_object())
    {
        fail("grid axis must be a string, integer, or object");
    }

    auto parseDivPair = [&](const json& pair, GridAxis& out) {
        if(!pair.is_array() || pair.size() != 2)
        {
            fail("grid div must be a 2-element [numerator, denominator] array");
        }
        out.numerator = parseGridValue(pair.at(0));
        out.denominator = parseGridValue(pair.at(1));
    };

    if(value.contains("value"))
    {
        axis.kind = GridAxisKind::VALUE;
        axis.value = parseGridValue(value.at("value"));
    }
    else if(value.contains("ceil_div"))
    {
        axis.kind = GridAxisKind::CEIL_DIV;
        parseDivPair(value.at("ceil_div"), axis);
    }
    else if(value.contains("floor_div"))
    {
        axis.kind = GridAxisKind::FLOOR_DIV;
        parseDivPair(value.at("floor_div"), axis);
    }
    else
    {
        fail("grid axis object needs one of 'value', 'ceil_div', 'floor_div'");
    }

    if(value.contains("add"))
    {
        axis.addend = parseGridValue(value.at("add"));
    }
    return axis;
}

GridFormula parseGridFormula(const json& obj)
{
    GridFormula grid;
    grid.x = parseGridAxis(requireMember(obj, "x"));
    grid.y = parseGridAxis(requireMember(obj, "y"));
    grid.z = parseGridAxis(requireMember(obj, "z"));
    return grid;
}

void parseBlock(const json& value, uint32_t (&block)[3])
{
    if(!value.is_array() || value.size() != 3)
    {
        fail("'block' must be a 3-element array");
    }
    for(size_t i = 0; i < 3; ++i)
    {
        const int64_t dim = value.at(i).get<int64_t>();
        if(dim <= 0)
        {
            fail("'block' dimensions must be positive");
        }
        block[i] = static_cast<uint32_t>(dim);
    }
}

// ---- Argument ABI parsing ---------------------------------------------------

KernelArgument parseArgument(const json& obj)
{
    KernelArgument arg;
    arg.name = getRequiredString(obj, "name");
    const std::string type = getRequiredString(obj, "type");

    if(type == "ptr")
    {
        arg.kind = ArgKind::POINTER;
    }
    else if(type == "f32")
    {
        arg.kind = ArgKind::SCALAR;
        arg.scalarType = ScalarType::F32;
    }
    else if(type == "i32")
    {
        arg.kind = ArgKind::SCALAR;
        arg.scalarType = ScalarType::I32;
    }
    else if(type == "i64")
    {
        arg.kind = ArgKind::SCALAR;
        arg.scalarType = ScalarType::I64;
    }
    else
    {
        fail("argument '" + arg.name + "' has unknown type '" + type
             + "' (expected ptr|f32|i32|i64)");
    }

    // Optional ABI cross-check: if the author states size_bytes, it must equal
    // the natural width we compute for the type.
    if(obj.contains("size_bytes"))
    {
        const int64_t declared = obj.at("size_bytes").get<int64_t>();
        const auto expected = static_cast<int64_t>(argSizeBytes(arg));
        if(declared != expected)
        {
            fail("argument '" + arg.name + "' size_bytes=" + std::to_string(declared)
                 + " disagrees with type width " + std::to_string(expected));
        }
    }
    return arg;
}

std::vector<KernelArgument> parseArgsSignature(const json& value)
{
    if(!value.is_array())
    {
        fail("'args_signature' must be an array");
    }
    std::vector<KernelArgument> signature;
    signature.reserve(value.size());
    for(const auto& element : value)
    {
        signature.push_back(parseArgument(element));
    }
    return signature;
}

LaunchMetadata parseLaunch(const json& obj)
{
    LaunchMetadata launch;
    launch.grid = parseGridFormula(requireMember(obj, "grid"));
    parseBlock(requireMember(obj, "block"), launch.block);
    if(obj.contains("shared_mem_bytes"))
    {
        const int64_t shared = obj.at("shared_mem_bytes").get<int64_t>();
        if(shared < 0)
        {
            fail("'shared_mem_bytes' must be non-negative");
        }
        launch.sharedMemBytes = static_cast<uint32_t>(shared);
    }
    launch.argsSignature = parseArgsSignature(requireMember(obj, "args_signature"));
    return launch;
}

KernelEntry parseKernel(const json& obj, const fs::path& familyDir)
{
    KernelEntry entry;
    entry.symbol = getRequiredString(obj, "symbol");

    const std::string coFile = getRequiredString(obj, "co_file");
    const fs::path coPath = familyDir / coFile;
    if(!fs::exists(coPath))
    {
        fail("co_file '" + coFile + "' not found beside family.json (" + coPath.string() + ")");
    }
    entry.coPath = coPath.string();

    if(obj.contains("constraints"))
    {
        entry.constraints = parseConstraints(obj.at("constraints"));
    }
    if(obj.contains("workspace_bytes"))
    {
        const int64_t workspace = obj.at("workspace_bytes").get<int64_t>();
        if(workspace < 0)
        {
            fail("'workspace_bytes' must be non-negative");
        }
        entry.workspaceBytes = static_cast<size_t>(workspace);
    }

    // Launch metadata lives at the kernel top level (grid/block/args_signature).
    entry.launch = parseLaunch(obj);
    return entry;
}

Family parseFamily(const json& obj, const fs::path& familyDir, const std::string& arch)
{
    Family family;
    family.name = getRequiredString(obj, "family");
    family.opKind = getRequiredString(obj, "op_kind");
    family.arch = arch;

    if(obj.contains("dtype"))
    {
        const json& dtype = obj.at("dtype");
        if(!dtype.is_array())
        {
            fail("'dtype' must be an array of strings");
        }
        for(const auto& element : dtype)
        {
            family.dtypes.push_back(element.get<std::string>());
        }
    }

    const json& kernels = requireMember(obj, "kernels");
    if(!kernels.is_array() || kernels.empty())
    {
        fail("'kernels' must be a non-empty array");
    }
    for(const auto& kernelObj : kernels)
    {
        family.kernels.push_back(parseKernel(kernelObj, familyDir));
    }
    return family;
}

} // namespace

const char* catalogDirSourceName(CatalogDirSource source)
{
    switch(source)
    {
    case CatalogDirSource::Env: return "env HIPDNN_AOT_CATALOG_DIR";
    case CatalogDirSource::SelfLocated: return "self-located beside plugin .so";
    case CatalogDirSource::Baked: return "baked install path (self-location failed)";
    default: return "unknown";
    }
}

CatalogDirResolution resolveCatalogDir()
{
    // 1. Explicit author override always wins.
    const std::string envDir = hipdnn_data_sdk::utilities::getEnv("HIPDNN_AOT_CATALOG_DIR");
    if(!envDir.empty())
    {
        return {envDir, CatalogDirSource::Env};
    }

    // 2. Beside the loaded plugin .so. Used UNCONDITIONALLY when the module dir
    //    resolves -- even if the catalog there is missing/empty -- so a locally
    //    built or force-loaded plugin reads ITS OWN build-tree catalog and never
    //    silently falls through to a system install's (the KA cross-contamination
    //    footgun). Build and install trees both place the catalog at exactly
    //    <plugin-dir>/HIPDNN_AOT_CATALOG_RELDIR, so one offset serves both.
    const std::string moduleDir = thisModuleDir();
    if(!moduleDir.empty())
    {
        const fs::path dir = fs::path(moduleDir) / HIPDNN_AOT_CATALOG_RELDIR;
        return {dir.string(), CatalogDirSource::SelfLocated};
    }

    // 3. Module dir unresolvable (dladdr/GetModuleHandleEx failed): last-resort
    //    baked absolute install path.
    return {HIPDNN_AOT_CATALOG_DIR, CatalogDirSource::Baked};
}

std::string defaultCatalogDir()
{
    return resolveCatalogDir().dir;
}

Catalog Catalog::loadForDevice(const std::string& catalogDir, const std::string& arch)
{
    std::vector<Family> families;

    const fs::path archDir = fs::path(catalogDir) / arch;
    std::error_code ec;
    if(!fs::is_directory(archDir, ec))
    {
        HIPDNN_PLUGIN_LOG_INFO("aot-catalog: no catalog directory for arch "
                               << arch << " at " << archDir.string() << " (engine will decline)");
        AOT_DEBUG("no catalog directory for arch " << arch << " at " << archDir.string()
                                                   << " -> engine declines every graph on this arch. "
                                                   << "Check the catalog was built/installed and holds a '"
                                                   << arch << "' subdir.");
        return Catalog{std::move(families)};
    }
    AOT_DEBUG("scanning arch dir " << archDir.string());

    for(const auto& entry : fs::directory_iterator(archDir, ec))
    {
        if(!entry.is_directory())
        {
            continue;
        }
        const fs::path familyJson = entry.path() / "family.json";
        if(!fs::exists(familyJson))
        {
            continue;
        }

        try
        {
            std::ifstream stream(familyJson);
            if(!stream)
            {
                fail("cannot open " + familyJson.string());
            }
            const json obj = json::parse(stream);
            families.push_back(parseFamily(obj, entry.path(), arch));
            HIPDNN_PLUGIN_LOG_INFO("aot-catalog: loaded family '" << families.back().name << "' ("
                                                                  << families.back().kernels.size()
                                                                  << " kernels) from "
                                                                  << familyJson.string());
            AOT_DEBUG("loaded family '" << families.back().name << "' op_kind='"
                                        << families.back().opKind << "' ("
                                        << families.back().kernels.size() << " kernels) from "
                                        << familyJson.string());
        }
        catch(const std::exception& e)
        {
            // NO-THROW contract: log and skip the offending family.
            HIPDNN_PLUGIN_LOG_ERROR("aot-catalog: skipping " << familyJson.string() << ": "
                                                             << e.what());
            AOT_DEBUG("SKIPPED family " << familyJson.string() << ": " << e.what());
        }
    }

    AOT_DEBUG("loaded " << families.size() << " family(ies) for arch " << arch);
    return Catalog{std::move(families)};
}

std::vector<Catalog::Candidate> Catalog::candidatesFor(const std::string& opKind,
                                                       const ProblemShape& problem) const
{
    std::vector<Candidate> candidates;
    for(const auto& family : _families)
    {
        if(family.opKind != opKind)
        {
            continue;
        }
        for(const auto& kernel : family.kernels)
        {
            if(satisfies(kernel.constraints, problem))
            {
                candidates.push_back(Candidate{&family, &kernel});
            }
        }
    }
    return candidates;
}

} // namespace aot_catalog_engine::catalog
