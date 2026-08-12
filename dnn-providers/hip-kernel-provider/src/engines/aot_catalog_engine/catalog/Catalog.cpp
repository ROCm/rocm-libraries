// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "catalog/Catalog.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>

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

// Reject any field name not in `allowed`, except the '_'-prefixed comment
// convention the checked-in family.json files use (e.g. "_comment",
// "_constraints_comment"). family.json selection is fail-CLOSED by design, but a
// misspelled FIELD NAME fails OPEN: e.g. a kernel entry spelling "constraints" as
// "constraint" silently loses its whole predicate set and matches every problem
// of its op kind, and "mutiple_of" inside a rule silently drops that divisibility
// check. There is no schema validator in the load path, so this is the only place
// that guarantee is enforced -- fail loudly on any key we do not recognize.
void rejectUnknownKeys(const json& obj,
                       const char* context,
                       std::initializer_list<const char*> allowed)
{
    if(!obj.is_object())
    {
        return; // shape is validated by the caller's field accessors
    }
    for(const auto& item : obj.items())
    {
        const std::string& key = item.key();
        if(!key.empty() && key.front() == '_')
        {
            continue; // '_'-prefixed keys are the comment convention -- allowed
        }
        bool recognized = false;
        for(const char* candidate : allowed)
        {
            if(key == candidate)
            {
                recognized = true;
                break;
            }
        }
        if(!recognized)
        {
            fail("unknown field '" + key + "' in " + context
                 + " (misspelled? only listed fields and '_'-prefixed comments are allowed)");
        }
    }
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
    rejectUnknownKeys(
        obj, "constraint rule", {"equals", "not_equals", "one_of", "min", "max", "multiple_of"});

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

// ---- Workspace-expression parsing -------------------------------------------

// One recognized operator key -> (WsOp, exact-or-minimum arity). exactArity>0
// means the array must hold exactly that many children; exactArity==0 means
// "variadic, >= minArity".
struct WsOpSpec
{
    WsOp op;
    size_t exactArity; // 0 => variadic
    size_t minArity;
};

WorkspaceExpr parseWorkspaceExpr(const json& value)
{
    WorkspaceExpr expr;

    // Shorthand scalars: a bare integer is a LITERAL (the back-compat path --
    // every existing "workspace_bytes": 0 lands here), a bare string a SYMBOL.
    if(value.is_number_integer())
    {
        expr.op = WsOp::LITERAL;
        expr.literal = value.get<int64_t>();
        if(expr.literal < 0)
        {
            fail("'workspace_bytes' literal must be non-negative");
        }
        return expr;
    }
    if(value.is_string())
    {
        expr.op = WsOp::SYMBOL;
        expr.symbol = value.get<std::string>();
        return expr;
    }

    if(!value.is_object())
    {
        fail("workspace expression must be an integer, a symbol string, or an operator object");
    }

    // An operator node is an object with EXACTLY one recognized op key whose
    // value is an array of child expressions. Stricter than grid axes (which
    // allow an 'add' companion key): the JSON-AST discipline is one op per node.
    static const std::map<std::string, WsOpSpec> s_kOps = {
        {"mul", {WsOp::MUL, 0, 1}},
        {"add", {WsOp::ADD, 0, 1}},
        {"min", {WsOp::MIN, 0, 1}},
        {"max", {WsOp::MAX, 0, 1}},
        {"sub", {WsOp::SUB, 2, 2}},
        {"ceil_div", {WsOp::CEIL_DIV, 2, 2}},
        {"floor_div", {WsOp::FLOOR_DIV, 2, 2}},
        {"align_up", {WsOp::ALIGN_UP, 2, 2}},
    };

    const WsOpSpec* spec = nullptr;
    std::string opKey;
    for(const auto& [key, node] : value.items())
    {
        auto it = s_kOps.find(key);
        if(it == s_kOps.end())
        {
            fail("workspace expression has unknown operator key '" + key
                 + "' (expected one of mul|add|sub|min|max|ceil_div|floor_div|align_up)");
        }
        if(spec != nullptr)
        {
            std::string message
                = "workspace expression object must hold exactly one operator key, found '";
            message += opKey;
            message += "' and '";
            message += key;
            message += "'";
            fail(message);
        }
        spec = &it->second;
        opKey = key;
    }
    if(spec == nullptr)
    {
        fail("workspace expression object needs exactly one operator key");
    }

    const json& operands = value.at(opKey);
    if(!operands.is_array())
    {
        fail("workspace operator '" + opKey + "' needs an array of operands");
    }
    if(spec->exactArity != 0 && operands.size() != spec->exactArity)
    {
        fail("workspace operator '" + opKey + "' takes exactly " + std::to_string(spec->exactArity)
             + " operands, got " + std::to_string(operands.size()));
    }
    if(operands.size() < spec->minArity)
    {
        fail("workspace operator '" + opKey + "' needs at least " + std::to_string(spec->minArity)
             + " operand(s)");
    }

    expr.op = spec->op;
    expr.args.reserve(operands.size());
    for(const auto& operand : operands)
    {
        expr.args.push_back(parseWorkspaceExpr(operand));
    }
    return expr;
}

void parseBlock(const json& value, std::array<uint32_t, 3>& block)
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
    rejectUnknownKeys(obj, "args_signature entry", {"name", "type", "size_bytes"});

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
    rejectUnknownKeys(obj,
                      "kernel entry",
                      {"symbol",
                       "co_file",
                       "constraints",
                       "workspace_bytes",
                       "shared_mem_bytes",
                       "grid",
                       "block",
                       "args_signature"});

    KernelEntry entry;
    entry.symbol = getRequiredString(obj, "symbol");

    const std::string coFile = getRequiredString(obj, "co_file");
    const fs::path coPath = familyDir / coFile;
    if(!fs::exists(coPath))
    {
        fail("co_file '" + coFile + "' not found beside family.json (" + coPath.string() + ")");
    }
    entry.coPath = coPath.string();

    // 'constraints' is REQUIRED and must be non-empty. A kernel with no constraints
    // would ASSERT it handles every ProblemShape (selection omits absent keys), so an
    // accidentally-missing or empty map is a silent fail-OPEN. Require it explicitly.
    entry.constraints = parseConstraints(requireMember(obj, "constraints"));
    if(entry.constraints.empty())
    {
        fail("kernel '" + entry.symbol
             + "' has an empty 'constraints' map; a kernel that constrains nothing claims to "
               "handle every problem shape -- list at least a dtype constraint");
    }
    // 'workspace_bytes' is either a static integer (constant scratch, the common
    // case -- absent field defaults to LITERAL 0) or a JSON-AST expression over
    // the problem's grid symbols + `elem_size`, evaluated per-problem at launch.
    if(obj.contains("workspace_bytes"))
    {
        entry.workspace = parseWorkspaceExpr(obj.at("workspace_bytes"));
    }

    // Launch metadata lives at the kernel top level (grid/block/args_signature).
    entry.launch = parseLaunch(obj);
    return entry;
}

Family parseFamily(const json& obj, const fs::path& familyDir, const std::string& arch)
{
    rejectUnknownKeys(obj, "family", {"family", "op_kind", "arch", "dtype", "kernels"});

    Family family;
    family.name = getRequiredString(obj, "family");
    family.opKind = getRequiredString(obj, "op_kind");
    family.arch = arch;

    // The catalog directory (<catalog>/<arch>/<family>) is the authority for arch.
    // If the file also declares "arch", it must agree -- a mismatch means the file was
    // copied into the wrong arch folder, which would silently load kernels built for
    // another GPU. Cross-check rather than trust the (ignored-for-selection) field.
    if(obj.contains("arch"))
    {
        const std::string declaredArch = getRequiredString(obj, "arch");
        if(declaredArch != arch)
        {
            fail("family '" + family.name + "' declares arch '" + declaredArch
                 + "' but lives in the '" + arch + "' catalog directory");
        }
    }

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
    case CatalogDirSource::ENV:
        return "env HIPDNN_AOT_CATALOG_DIR";
    case CatalogDirSource::SELF_LOCATED:
        return "self-located beside plugin .so";
    case CatalogDirSource::BAKED:
        return "baked install path (self-location failed)";
    default:
        return "unknown";
    }
}

CatalogDirResolution resolveCatalogDir()
{
    // 1. Explicit author override always wins.
    const std::string envDir = hipdnn_data_sdk::utilities::getEnv("HIPDNN_AOT_CATALOG_DIR");
    if(!envDir.empty())
    {
        return {envDir, CatalogDirSource::ENV};
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
        return {dir.string(), CatalogDirSource::SELF_LOCATED};
    }

    // 3. Module dir unresolvable (dladdr/GetModuleHandleEx failed): last-resort
    //    baked absolute install path.
    return {HIPDNN_AOT_CATALOG_DIR, CatalogDirSource::BAKED};
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
        AOT_DEBUG("no catalog directory for arch "
                  << arch << " at " << archDir.string()
                  << " -> engine declines every graph on this arch. "
                  << "Check the catalog was built/installed and holds a '" << arch << "' subdir.");
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
