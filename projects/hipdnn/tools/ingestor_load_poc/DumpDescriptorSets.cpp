// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// ALMIOPEN-2812 POC correctness oracle: loads a descriptor tree with loadDescriptorCatalog,
// writes every catalog entry (path, root, conflict state and parsed descriptor), then
// resolves the catalog with resolveDescriptorSets and writes every DescriptorSet. All
// canonical JSON, one record per line. Object keys are sorted (nlohmann's default
// std::map); catalog entries are sorted by key; every vector is written in the order the
// loader produced it, which is stricter than sorting: the loader already promises a
// deterministic order, so a step that reorders anything shows up as a changed dump.
//
// Usage: hipdnn_poc_dump_descriptor_sets <root> [<root>...] > dump.jsonl
// A final summary line counts catalog packs, sets, packs and kernels.

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

namespace
{

using nlohmann::json;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

json toJson(const ingestor::DescriptorId& id)
{
    return ingestor::toString(id);
}

json toJson(const ingestor::MetadataValue& value)
{
    // Tagged with the alternative index so int 1 and double 1.0 cannot collapse.
    return json::array({value.index(), std::visit([](const auto& v) { return json(v); }, value)});
}

json toJson(const ingestor::MetadataValues& values)
{
    json out = json::object();
    for(const auto& [name, value] : values)
    {
        out[name] = toJson(value);
    }
    return out;
}

template <typename T>
json idList(const std::vector<T>& ids)
{
    json out = json::array();
    for(const auto& id : ids)
    {
        out.push_back(toJson(id));
    }
    return out;
}

json toJson(const ingestor::KernelSource& source)
{
    json signature = json::array();
    for(const auto& argument : source.signature)
    {
        signature.push_back({{"kind", argument.kind},
                             {"size", argument.size},
                             {"offset", argument.offset},
                             {"name", argument.name}});
    }
    return {{"kind", static_cast<int>(source.kind)},
            {"sourceFile", source.sourceFile},
            {"entryPoint", source.entryPoint},
            {"library", source.library},
            {"tocKey", source.tocKey},
            {"symbol", source.symbol},
            {"sha256", source.sha256},
            {"signature", std::move(signature)}};
}

json toJson(const ingestor::KernelDescriptor& kernel)
{
    return {{"id", toJson(kernel.id)},
            {"name", kernel.name},
            {"source", toJson(kernel.source)},
            {"metadata", toJson(kernel.metadata)},
            {"priority", kernel.priority},
            {"arch", kernel.arch},
            {"originDirectory", kernel.originDirectory.generic_string()},
            {"treeRoot", kernel.treeRoot.generic_string()}};
}

json toJson(const ingestor::KernelDescriptorPack& pack)
{
    json kernels = json::array();
    for(const auto& kernel : pack.kernels)
    {
        kernels.push_back(toJson(kernel));
    }
    return {{"id", toJson(pack.id)},
            {"name", pack.name},
            {"matcherIds", idList(pack.matcherIds)},
            {"engineId", toJson(pack.engineId)},
            {"dispatchId", toJson(pack.dispatchId)},
            {"arch", pack.arch},
            {"kernelIds", idList(pack.kernelIds)},
            {"kernels", std::move(kernels)}};
}

json toJson(const ingestor::EngineDescriptor& engine)
{
    return {{"id", toJson(engine.id)},
            {"name", engine.name},
            {"heuristicId", engine.heuristicId ? toJson(*engine.heuristicId) : json(nullptr)},
            {"metadataSchemaId", toJson(engine.metadataSchemaId)},
            {"knobs", engine.knobs},
            {"behaviorNotes", engine.behaviorNotes},
            {"sdkVersion", engine.sdkVersion.str()},
            {"numericalNotes", engine.numericalNotes},
            {"graphMatchNativeSymbol", engine.graphMatchNativeSymbol}};
}

json toJson(const ingestor::MetadataSchema& schema)
{
    json fields = json::array();
    for(const auto& field : schema.fields)
    {
        fields.push_back(
            {{"name", field.name},
             {"type", static_cast<int>(field.type)},
             {"defaultValue", field.defaultValue ? toJson(*field.defaultValue) : json(nullptr)}});
    }
    return {{"id", toJson(schema.id)}, {"name", schema.name}, {"fields", std::move(fields)}};
}

json toJson(const ingestor::HeuristicDescriptor& heuristic)
{
    return {{"id", toJson(heuristic.id)},
            {"name", heuristic.name},
            {"kind", static_cast<int>(heuristic.kind)},
            {"payload", heuristic.payload}};
}

json toJson(const ingestor::MatchDescriptor& matcher)
{
    return {{"id", toJson(matcher.id)},
            {"name", matcher.name},
            {"scope", static_cast<int>(matcher.scope)},
            {"matchSymbol", matcher.matchSymbol}};
}

json toJson(const ingestor::DispatchDescriptor& dispatch)
{
    return {{"id", toJson(dispatch.id)},
            {"name", dispatch.name},
            {"dispatchSymbol", dispatch.dispatchSymbol}};
}

template <typename T>
json listJson(const std::vector<T>& items)
{
    json out = json::array();
    for(const auto& item : items)
    {
        out.push_back(toJson(item));
    }
    return out;
}

json toJson(const ingestor::DescriptorSet& set)
{
    return {{"engine", toJson(set.engine)},
            {"schema", toJson(set.schema)},
            {"heuristic", set.heuristic ? toJson(*set.heuristic) : json(nullptr)},
            {"matchers", listJson(set.matchers)},
            {"dispatches", listJson(set.dispatches)},
            {"packs", listJson(set.packs)}};
}

json keyJson(const ingestor::DescriptorId& key)
{
    return toJson(key);
}

json keyJson(const ingestor::ArchKey& key)
{
    return json::array({toJson(key.first), key.second});
}

/// One line per catalog entry, sorted by key text: covers trees whose sets never resolve
/// (authoring-form fixtures), where the set dump alone would be empty on both sides.
template <typename Map>
void dumpCatalogMap(const char* type, const Map& map)
{
    std::vector<json> lines;
    for(const auto& [key, entry] : map)
    {
        lines.push_back({{"catalog", type},
                         {"key", keyJson(key)},
                         {"path", entry.path.generic_string()},
                         {"treeRoot", entry.treeRoot.generic_string()},
                         {"conflicted", entry.conflicted},
                         {"settled", entry.settled},
                         {"descriptor", toJson(entry.descriptor)}});
    }
    std::sort(lines.begin(), lines.end(), [](const json& a, const json& b) {
        return a["key"].dump() < b["key"].dump();
    });
    for(const auto& line : lines)
    {
        std::cout << line.dump() << '\n';
    }
}

} // namespace

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <root> [<root>...]\n";
        return 2;
    }
    std::vector<std::filesystem::path> roots(argv + 1, argv + argc);

    auto catalog = ingestor::loadDescriptorCatalog(roots);
    dumpCatalogMap("kmd", catalog.schemas);
    dumpCatalogMap("uhd", catalog.heuristics);
    dumpCatalogMap("ued", catalog.engines);
    dumpCatalogMap("umd", catalog.matchers);
    dumpCatalogMap("udd", catalog.dispatches);
    dumpCatalogMap("kdp", catalog.packs);
    dumpCatalogMap("ukd", catalog.kernels);
    const size_t catalogPacks = catalog.packs.size();

    size_t packCount = 0;
    size_t kernelCount = 0;
    const auto sets = ingestor::resolveDescriptorSets(std::move(catalog));
    for(const auto& set : sets)
    {
        std::cout << toJson(set).dump() << '\n';
        packCount += set.packs.size();
        for(const auto& pack : set.packs)
        {
            kernelCount += pack.kernels.size();
        }
    }
    std::cout << json{{"catalog_packs", catalogPacks},
                      {"sets", sets.size()},
                      {"packs", packCount},
                      {"kernels", kernelCount}}
                     .dump()
              << '\n';
    return 0;
}

#else // HIPDNN_ENABLE_KERNEL_INGESTOR

#include <iostream>

int main()
{
    std::cerr << "built without HIPDNN_ENABLE_KERNEL_INGESTOR\n";
    return 1;
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
