// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "dispatcher/AotCatalog.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <nlohmann/json.hpp>

#include "dispatcher/AotBundlePaths.hpp"
#include "dispatcher/PluginModuleDir.hpp"

namespace rocke_client::dispatcher
{
namespace
{

using Json = nlohmann::json;

Json loadJson(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if(!stream.is_open())
    {
        throw std::runtime_error("failed to open " + path.string());
    }
    Json value;
    stream >> value;
    return value;
}

std::string getRequiredString(const Json& value, const char* context)
{
    if(!value.is_string() || value.get_ref<const std::string&>().empty())
    {
        throw std::runtime_error(std::string(context) + " must be a non-empty string");
    }
    return value.get<std::string>();
}

std::int64_t getRequiredInt(const Json& value, const char* context)
{
    if(!value.is_number_integer())
    {
        throw std::runtime_error(std::string(context) + " must be an integer");
    }
    return value.get<std::int64_t>();
}

AttrValue parseAttrValue(const Json& value, const char* context)
{
    if(value.is_boolean())
    {
        return AttrValue{value.get<bool>()};
    }
    if(value.is_number_float())
    {
        return AttrValue{value.get<double>()};
    }
    if(value.is_number_integer())
    {
        return AttrValue{value.get<std::int64_t>()};
    }
    if(value.is_string())
    {
        return AttrValue{value.get<std::string>()};
    }
    throw std::runtime_error(std::string(context) + " has unsupported JSON value kind");
}

AttributeRule parseRule(const Json& value, const std::string& name)
{
    if(!value.is_object())
    {
        throw std::runtime_error("attribute constraint for " + name + " must be an object");
    }

    AttributeRule rule;
    if(value.contains("equals"))
    {
        rule.equals = parseAttrValue(value.at("equals"), (name + ".equals").c_str());
    }
    if(value.contains("not_equals"))
    {
        rule.notEquals = parseAttrValue(value.at("not_equals"), (name + ".not_equals").c_str());
    }
    if(value.contains("one_of"))
    {
        const auto& oneOf = value.at("one_of");
        if(!oneOf.is_array() || oneOf.empty())
        {
            throw std::runtime_error("attribute constraint for " + name
                                     + ".one_of must be a non-empty array");
        }
        std::vector<AttrValue> values;
        values.reserve(oneOf.size());
        for(const auto& item : oneOf)
        {
            values.emplace_back(parseAttrValue(item, (name + ".one_of").c_str()));
        }
        rule.oneOf = std::move(values);
    }
    if(rule.empty())
    {
        throw std::runtime_error("attribute constraint for " + name + " has no operator");
    }
    return rule;
}

AttributeConstraints parseAttributeConstraints(const Json& value)
{
    if(!value.is_object())
    {
        throw std::runtime_error("selection.attribute_constraints must be an object");
    }

    AttributeConstraints constraints;
    for(const auto& item : value.items())
    {
        constraints.emplace(item.key(), parseRule(item.value(), item.key()));
    }
    return constraints;
}

GridValue parseGridValue(const Json& value, const char* context)
{
    if(value.is_string())
    {
        return GridValue{.symbol = value.get<std::string>(), .literal = 0};
    }
    if(value.is_number_integer())
    {
        return GridValue{.symbol = std::nullopt, .literal = value.get<std::int64_t>()};
    }
    throw std::runtime_error(std::string(context) + " must be an integer or symbol string");
}

GridAxis parseGridAxis(const Json& value, const char* context)
{
    if(value.is_string() || value.is_number_integer())
    {
        GridAxis axis;
        axis.kind = GridAxis::Kind::VALUE;
        axis.value = parseGridValue(value, context);
        return axis;
    }

    if(value.is_object() && value.contains("ceil_div"))
    {
        const auto& args = value.at("ceil_div");
        if(!args.is_array() || args.size() != 2)
        {
            throw std::runtime_error(std::string(context) + ".ceil_div must have two arguments");
        }
        GridAxis axis;
        axis.kind = GridAxis::Kind::CEIL_DIV;
        axis.numerator = parseGridValue(args.at(0), context);
        axis.denominator = parseGridValue(args.at(1), context);
        return axis;
    }

    throw std::runtime_error(std::string(context) + " has unsupported grid expression");
}

GridFormula parseGridFormula(const Json& value)
{
    if(!value.is_object())
    {
        throw std::runtime_error("launch.grid_formula must be an object");
    }
    return {.x = parseGridAxis(value.at("x"), "launch.grid_formula.x"),
            .y = parseGridAxis(value.at("y"), "launch.grid_formula.y"),
            .z = parseGridAxis(value.at("z"), "launch.grid_formula.z")};
}

std::array<unsigned int, 3> parseBlock(const Json& value)
{
    if(!value.is_array() || value.size() != 3)
    {
        throw std::runtime_error("launch.block must be a three-element array");
    }
    return {value.at(0).get<unsigned int>(),
            value.at(1).get<unsigned int>(),
            value.at(2).get<unsigned int>()};
}

ArgKind parseArgKind(const Json& value)
{
    const std::string kind = getRequiredString(value, "argument kind");
    if(kind == "pointer")
    {
        return ArgKind::POINTER;
    }
    if(kind == "scalar")
    {
        return ArgKind::SCALAR;
    }
    throw std::runtime_error(R"(args_signature kind must be "pointer" or "scalar", got ")" + kind
                             + R"(")");
}

ScalarType parseScalarType(const Json& value)
{
    const std::string type = getRequiredString(value, "argument type");
    if(type == "f32")
    {
        return ScalarType::F32;
    }
    if(type == "i32")
    {
        return ScalarType::I32;
    }
    if(type == "i64")
    {
        return ScalarType::I64;
    }
    throw std::runtime_error(R"(args_signature scalar type must be "f32", "i32", or "i64", got ")"
                             + type + R"(")");
}

// Cross-check the manifest's declared size_bytes/alignment against the ABI width
// implied by kind/dtype, so a malformed bundle fails at load rather than packing
// a wrongly sized argument at launch.
void validateArgAbi(const Json& item, const KernelArgument& arg)
{
    const std::size_t expected = argSizeBytes(arg);
    const auto declaredSize = item.at("size_bytes").get<std::size_t>();
    if(declaredSize != expected)
    {
        throw std::runtime_error("args_signature size_bytes for '" + arg.name + "' must be "
                                 + std::to_string(expected) + ", got "
                                 + std::to_string(declaredSize));
    }
    const auto declaredAlignment = item.value("alignment", expected);
    if(declaredAlignment != expected)
    {
        throw std::runtime_error("args_signature alignment for '" + arg.name + "' must be "
                                 + std::to_string(expected) + ", got "
                                 + std::to_string(declaredAlignment));
    }
}

std::vector<KernelArgument> parseArgsSignature(const Json& value)
{
    if(!value.is_array() || value.empty())
    {
        throw std::runtime_error("args_signature must be a non-empty array");
    }

    std::vector<KernelArgument> args;
    args.reserve(value.size());
    for(const auto& item : value)
    {
        KernelArgument arg;
        arg.name = getRequiredString(item.at("name"), "argument name");
        arg.kind = parseArgKind(item.at("kind"));
        if(arg.kind == ArgKind::SCALAR)
        {
            arg.scalarType = parseScalarType(item.at("type"));
        }
        validateArgAbi(item, arg);
        args.emplace_back(std::move(arg));
    }
    return args;
}

LaunchMetadata parseLaunchMetadata(const Json& entry)
{
    const auto& launch = entry.at("launch");
    LaunchMetadata meta;
    meta.grid = parseGridFormula(launch.at("grid_formula"));
    meta.block = parseBlock(launch.at("block"));
    meta.sharedMemBytes = launch.value("shared_mem_bytes", std::size_t{0});
    meta.argsSignature = parseArgsSignature(entry.at("args_signature"));
    return meta;
}

CompileSpec parseCompileSpec(const Json& value)
{
    CompileSpec spec;
    spec.dtype = getRequiredString(value.at("dtype"), "compile_spec.dtype");
    spec.canonicalLayout
        = getRequiredString(value.at("canonical_layout"), "compile_spec.canonical_layout");
    spec.seqlenQ = getRequiredInt(value.at("seqlen_q"), "compile_spec.seqlen_q");
    spec.seqlenK = getRequiredInt(value.at("seqlen_k"), "compile_spec.seqlen_k");
    spec.numQueryHeads
        = getRequiredInt(value.at("num_query_heads"), "compile_spec.num_query_heads");
    spec.numKvHeads = getRequiredInt(value.at("num_kv_heads"), "compile_spec.num_kv_heads");
    spec.headSize = getRequiredInt(value.at("head_size"), "compile_spec.head_size");
    spec.blockSizeQ = getRequiredInt(value.at("block_size_q"), "compile_spec.block_size_q");
    spec.blockSizeK = getRequiredInt(value.at("block_size_k"), "compile_spec.block_size_k");
    spec.maskMode = getRequiredString(value.at("mask_mode"), "compile_spec.mask_mode");
    return spec;
}

BatchRange parseBatchRange(const Json& selection)
{
    const Json* batch = nullptr;
    if(selection.contains("batch"))
    {
        batch = &selection.at("batch");
    }
    else
    {
        batch = &selection.at("shape_constraints").at("batch");
    }
    return {.min = getRequiredInt(batch->at("min"), "selection.batch.min"),
            .max = getRequiredInt(batch->at("max"), "selection.batch.max")};
}

AotInstance parseManifestEntry(const Json& entry,
                               const std::string& manifestArch,
                               const std::filesystem::path& kpackPath)
{
    const auto& selection = entry.at("selection");

    AotInstance instance;
    instance.name = getRequiredString(entry.at("name"), "entry.name");
    instance.op = getRequiredString(entry.at("op"), "entry.op");
    instance.family = getRequiredString(entry.at("family"), "entry.family");
    instance.arch = manifestArch;
    instance.compileSpec = parseCompileSpec(entry.at("compile_spec"));
    instance.batch = parseBatchRange(selection);
    instance.attributeConstraints
        = parseAttributeConstraints(selection.at("attribute_constraints"));
    instance.runtime.cacheKey = getRequiredString(entry.at("cache_key"), "entry.cache_key");
    instance.runtime.tocKey = getRequiredString(entry.at("toc_key"), "entry.toc_key");
    instance.runtime.symbol = getRequiredString(entry.at("symbol"), "entry.symbol");
    instance.runtime.kpackPath = kpackPath.string();
    instance.runtime.launch = parseLaunchMetadata(entry);
    return instance;
}

std::vector<std::filesystem::path> getManifestFilePaths(const std::filesystem::path& root)
{
    std::vector<std::filesystem::path> files;
    if(!std::filesystem::is_directory(root))
    {
        return files;
    }

    for(const auto& archDir : std::filesystem::directory_iterator(root))
    {
        if(!archDir.is_directory())
        {
            continue;
        }
        for(const auto& item : std::filesystem::directory_iterator(archDir.path()))
        {
            if(item.is_regular_file()
               && item.path().filename().string().starts_with("rocke_client_")
               && item.path().extension() == ".json")
            {
                files.emplace_back(item.path());
            }
        }
    }
    std::ranges::sort(files);
    return files;
}

std::vector<AotInstance> parseManifestEntries(const std::filesystem::path& manifestPath)
{
    const Json manifest = loadJson(manifestPath);
    const auto arch = getRequiredString(manifest.at("arch"), "manifest.arch");
    const auto kpackPath
        = manifestPath.parent_path() / getRequiredString(manifest.at("kpack"), "manifest.kpack");
    if(!std::filesystem::is_regular_file(kpackPath))
    {
        throw std::runtime_error("bundle kpack file is missing: " + kpackPath.string());
    }

    const auto& entries = manifest.at("entries");
    if(!entries.is_array() || entries.empty())
    {
        throw std::runtime_error(manifestPath.string() + " must contain non-empty entries");
    }

    std::vector<AotInstance> instances;
    instances.reserve(entries.size());
    for(const auto& entry : entries)
    {
        instances.emplace_back(parseManifestEntry(entry, arch, kpackPath));
    }
    return instances;
}

} // namespace

AotCatalog::AotCatalog(std::vector<AotInstance> instances)
    : _instances(std::move(instances))
{
}

AotCatalog AotCatalog::loadForDevice(const std::string& arch)
{
    // No-throw contract: all errors produce a log and return an empty catalog.
    // Do NOT throw -- this may be called from the noexcept selectInstance path.
    try
    {
        const std::filesystem::path pluginDir = currentPluginDirectory();
        const std::filesystem::path manifestPath = aotManifestPath(pluginDir, arch);

        if(!std::filesystem::exists(manifestPath))
        {
            HIPDNN_PLUGIN_LOG_INFO("rocke-client: no AOT bundle for '"
                                   << arch << "' at " << manifestPath.string()
                                   << "; engine will decline all graphs");
            return AotCatalog{};
        }

        auto instances = parseManifestEntries(manifestPath);
        HIPDNN_PLUGIN_LOG_INFO("rocke-client dispatcher: loaded "
                               << instances.size() << " AOT kpack instances for '" << arch
                               << "' from " << manifestPath.string());
        return AotCatalog{std::move(instances)};
    }
    catch(const std::exception& ex)
    {
        HIPDNN_PLUGIN_LOG_ERROR("rocke-client: failed to load AOT bundle for '"
                                << arch << "': " << ex.what()
                                << "; engine will decline all graphs");
        return AotCatalog{};
    }
    catch(...)
    {
        HIPDNN_PLUGIN_LOG_ERROR("rocke-client: failed to load AOT bundle for '"
                                << arch << "': unknown error; engine will decline all graphs");
        return AotCatalog{};
    }
}

std::vector<AotInstance> loadManifestsFromDirectory(const std::filesystem::path& root)
{
    std::vector<AotInstance> instances;
    for(const auto& manifest : getManifestFilePaths(root))
    {
        try
        {
            auto parsed = parseManifestEntries(manifest);
            instances.insert(instances.end(),
                             std::make_move_iterator(parsed.begin()),
                             std::make_move_iterator(parsed.end()));
        }
        catch(const std::exception& e)
        {
            // One malformed bundle must not drop the rest of the catalog.
            HIPDNN_PLUGIN_LOG_WARN("rocke-client dispatcher: skipping malformed AOT bundle "
                                   << manifest << ": " << e.what());
        }
    }
    return instances;
}

std::vector<std::reference_wrapper<const AotInstance>>
    AotCatalog::candidatesFor(const std::string& op, const std::string& arch) const
{
    std::vector<std::reference_wrapper<const AotInstance>> candidates;
    for(const auto& instance : _instances)
    {
        if(instance.op == op && instance.arch == arch)
        {
            candidates.emplace_back(instance);
        }
    }
    return candidates;
}

} // namespace rocke_client::dispatcher
