// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Graph/GraphUtilities.hpp>
#include <rocRoller/KernelGraph/Transforms/ScheduleMultiplyAndLDS.hpp>
#include <rocRoller/KernelGraph/Transforms/ScheduleMultiplyAndLDS_detail.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Concepts.hpp>
#include <rocRoller/Utilities/Logging.hpp>

namespace rocRoller::KernelGraph
{
    namespace ScheduleMultiplyAndLDSDetail
    {
        Chains makeChains(KernelGraph const& graph, std::vector<int> nodes)
        {
            std::ranges::sort(nodes, TopologicalCompare(graph));
            Log::debug("makeChains({})", ShowValue(nodes));

            for(int i = 0; i + 1 < nodes.size(); i++)
            {
                auto order = graph.control.compareNodes(UpdateCache, nodes[i], nodes[i + 1]);
                AssertFatal(order == ControlGraph::NodeOrdering::LeftFirst,
                            ShowValue(order),
                            ShowValue(nodes[i]),
                            ShowValue(nodes[i + 1]));
            }

            Chains rv;

            if(nodes.empty())
                return rv;

            std::vector<int> currentChain;

            currentChain.push_back(nodes[0]);

            for(int i = 1; i < nodes.size(); i++)
            {

                if(!graph.control.findEdge(currentChain.back(), nodes[i]))
                {
                    rv.push_back(std::move(currentChain));
                    currentChain.clear();
                }

                currentChain.push_back(nodes[i]);
            }

            rv.push_back(std::move(currentChain));

            return rv;
        }

        int getLDSNode(KernelGraph const& graph, int node)
        {
            while(auto body = graph.control.getOutputNodeIndices<ControlGraph::Body>(node).only())
            {
                node = *body;
            }

            return node;
        }

        DataType getLDSType(KernelGraph const& graph, int node)
        {
            node = getLDSNode(graph, node);

            auto ldsType = getDataType(graph.control.getNode(node));
            AssertFatal(ldsType != DataType::None);
            return ldsType;
        };

        std::string toString(std::map<DataType, int> const& typeCounts)
        {
            auto fmtPair = [](std::pair<DataType, int> const& pair) -> std::string {
                return fmt::format("{}: {}", toString(pair.first), pair.second);
            };

            auto formatted = typeCounts | std::views::transform(fmtPair);

            return fmt::format("[{}]", fmt::join(formatted, ","));
        }

        std::string toString(ChainTypes const& chainTypes)
        {
            auto ts = [](auto val) { return toString(val); };

            auto formatted = chainTypes | std::views::transform(ts);

            return fmt::format("{{{}}}", fmt::join(formatted, ", "));
        }

        ChainTypes filterLastCoordinateReads(KernelGraph const& graph, Chain& chain)
        {
            ControlFlowRWTracer tracer(graph);

            auto coordDataType = [&](int coord) -> DataType {
                for(auto rw : tracer.coordinatesReadWrite(coord))
                {
                    auto rwType = getDataType(graph.control.getNode(rw.control));
                    if(rwType != DataType::None)
                        return rwType;
                }

                return DataType::None;
            };

            // Map from coord to the last op that uses that coord.
            std::unordered_map<int, int> lastUses;

            for(auto op : chain)
            {
                auto records = tracer.opReadWrite(op);
                for(auto rec : records)
                {
                    if(rec.rw == ControlFlowRWTracer::READ)
                        lastUses[rec.coordinate] = op;
                }
            }

            // Map from op to the count of each data type used by it.
            std::unordered_map<int, std::map<DataType, int>> lastUsedTypes;

            for(auto const& [coord, op] : lastUses)
                lastUsedTypes[op][coordDataType(coord)]++;

            std::vector<int> newChain;
            ChainTypes       rv;

            int skipCount = 0;
            for(auto op : chain)
            {
                auto iter = lastUsedTypes.find(op);

                if(iter != lastUsedTypes.end())
                {
                    Log::debug("filterLastCoordinateReads: Skipped {}.", skipCount);
                    skipCount = 0;

                    newChain.push_back(op);
                    rv.push_back(iter->second);
                }
                else
                {
                    skipCount++;
                }
            }
            Log::debug("filterLastCoordinateReads: Skipped {}.", skipCount);
            Log::debug("-------------------------------------------------");

            chain = newChain;

            return rv;
        }

        Chains findMultiplyChains(KernelGraph const& graph)
        {
            auto isMultiply = [&](int idx) -> bool {
                return graph.control.get<ControlGraph::Multiply>(idx).has_value();
            };

            auto multiplies = graph.control.getNodes().filter(isMultiply).to<std::vector>();

            auto chains = makeChains(graph, std::move(multiplies));

            return chains;
        }

        std::tuple<Chains, std::vector<ChainTypes>>
            findMultiplyChainsAndCoords(KernelGraph const& graph)
        {
            auto chains = findMultiplyChains(graph);

            std::vector<ChainTypes> chainTypes;

            for(auto& chain : chains)
                chainTypes.push_back(filterLastCoordinateReads(graph, chain));

            return {chains, chainTypes};
        }

        void getImmediateBodyParents(KernelGraph const& graph, std::vector<int>& nodes)
        {
            auto isSetCoordinate = [&](int idx) -> bool {
                return graph.control.get<ControlGraph::SetCoordinate>(idx).has_value();
            };

            for(auto& node : nodes)
            {
                while(auto bodyParent = graph.control.getInputNodeIndices<ControlGraph::Body>(node)
                                            .filter(isSetCoordinate)
                                            .only())
                {
                    node = *bodyParent;
                }
            }
        }

        Chains findLoadLDSChains(KernelGraph const& graph)
        {
            auto isLoadLDSTile = [&](int idx) -> bool {
                return graph.control.get<ControlGraph::LoadLDSTile>(idx).has_value();
            };

            auto nodes = graph.control.getNodes().filter(isLoadLDSTile).to<std::vector>();
            getImmediateBodyParents(graph, nodes);

            return makeChains(graph, std::move(nodes));
        }

        std::string chainTagTable(KernelGraph const& graph, Chain chain)
        {
            ControlFlowRWTracer tracer(graph);

            std::vector<int>                              coords;
            std::map<int, DataType>                       coordDataTypes;
            std::map<int, rocRoller::LayoutType>          coordLayouts;
            std::map<int, ControlFlowRWTracer::ReadWrite> useTypes;
            std::map<int, int>                            lastUses;

            {
                std::set<int> coordSet;
                for(auto& op : chain)
                {
                    while(auto child
                          = graph.control.getOutputNodeIndices<ControlGraph::Body>(op).only())
                    {
                        op = *child;
                    }
                }

                std::unordered_set<int> nodes(chain.begin(), chain.end());
                for(auto const& rec : tracer.coordinatesReadWrite())
                {
                    if(nodes.contains(rec.control))
                        coordSet.insert(rec.coordinate);
                }

                for(auto op : chain)
                {
                    auto records = tracer.opReadWrite(op);
                    for(auto rec : records)
                    {
                        lastUses[rec.coordinate] = op;
                        auto iter                = useTypes.find(rec.coordinate);
                        if(iter == useTypes.end())
                            useTypes[rec.coordinate] = rec.rw;
                        else
                            iter->second = combine(iter->second, rec.rw);
                    }
                }

                for(auto const& rec : tracer.coordinatesReadWrite())
                {
                    if(rec.rw == ControlFlowRWTracer::WRITE && coordSet.contains(rec.coordinate)
                       && !coordDataTypes.contains(rec.coordinate))
                    {
                        auto node  = graph.control.getNode(rec.control);
                        auto dtype = getDataType(node);
                        if(dtype != DataType::None)
                            coordDataTypes[rec.coordinate] = dtype;
                    }
                }

                for(auto coord : coordSet)
                    if(!coordDataTypes.contains(coord))
                        coordDataTypes[coord] = DataType::None;

                coordLayouts = [&]() {
                    auto getCoordLayout = [&](int coord) {
                        auto mt = graph.coordinates.get<CoordinateGraph::MacroTile>(coord);

                        LayoutType lt = LayoutType::None;
                        if(mt)
                            lt = mt->layoutType;

                        return std::make_pair(coord, lt);
                    };

                    auto theView = coordSet | std::views::transform(getCoordLayout);

                    return std::map(theView.begin(), theView.end());
                }();

                auto coordOrder = [&](int a, int b) {
                    auto typeA = useTypes[a], typeB = useTypes[b];

                    if(typeA != typeB)
                        return typeA < typeB;

                    return TopologicalCompare(graph)(lastUses[a], lastUses[b]);
                };

                {
                    auto isWantedCoord = [&](int coord) {
                        auto dtype = coordDataTypes.at(coord);
                        if(dtype == DataType::None)
                            return false;

                        auto const& info = DataTypeInfo::Get(dtype);

                        return info.packing > 1;
                    };

                    auto tmp = coordSet | std::views::filter(isWantedCoord);
                    coords   = std::vector(tmp.begin(), tmp.end());
                }

                std::ranges::sort(coords, coordOrder);
            }

            auto getCoordLayout = [&](int coord) { return coordLayouts.at(coord); };

            auto value     = [](auto const& x) { return x.second; };
            auto lastNodes = [&]() {
                auto tmp = lastUses | std::views::transform(value);
                return std::unordered_set(tmp.begin(), tmp.end());
            }();

            auto isLast = [&](int x) { return lastNodes.contains(x); };
            auto lasts  = [&]() {
                auto tmp = chain | std::views::filter(isLast);
                return std::vector(tmp.begin(), tmp.end());
            }();

            auto formatElement = [](auto el) -> std::string { return fmt::format("{:^6}", el); };

            auto getDtype = [&](int coord) { return TypeAbbrev(coordDataTypes[coord]); };

            auto msg = fmt::format("|{}|{}",
                                   formatElement(""),
                                   fmt::join(coords | std::views::transform(formatElement), "|"));

            std::string line(msg.size(), '=');

            msg += fmt::format("\n{}\n", line);

            msg += fmt::format("|{}|{}\n{}\n",
                               formatElement(""),
                               fmt::join(coords | std::views::transform(getDtype)
                                             | std::views::transform(formatElement),
                                         "|"),
                               line);

            msg += fmt::format("\n{}\n", line);

            msg += fmt::format("|{}|{}\n{}\n",
                               formatElement(""),
                               fmt::join(coords | std::views::transform(getCoordLayout)
                                             | std::views::transform(abbrev)
                                             | std::views::transform(formatElement),
                                         "|"),
                               line);

            for(auto op : chain)
            {
                auto records    = tracer.opReadWrite(op);
                auto lookupNode = [&](int coord) -> std::string {
                    for(auto const& rec : records)
                    {
                        if(rec.coordinate == coord)
                        {
                            switch(rec.rw)
                            {
                            case ControlFlowRWTracer::READ:
                                return "V";
                            case ControlFlowRWTracer::WRITE:
                                return "^";
                            case ControlFlowRWTracer::READWRITE:
                                return "X";
                            default:
                                break;
                            }
                        }
                    }
                    return " ";
                };

                msg += fmt::format("|{}|{}\n",
                                   formatElement(op),
                                   fmt::join(coords | std::views::transform(lookupNode)
                                                 | std::views::transform(formatElement),
                                             "|"));
            }

            msg += fmt::format("Lasts: ({})({})\n", lasts.size(), fmt::join(lasts, ", "));

            return msg;
        }

        void logChainTagTable(KernelGraph const& graph, Chain chain)
        {
            Log::debug("\n{}", chainTagTable(graph, chain));
        }

        std::string showChain(Chain const& chain)
        {
            auto ts = [](int x) -> std::string { return fmt::format("{}", x); };
            return fmt::format("({})", fmt::join(chain | std::views::transform(ts), ", "));
        }

        std::string showChains(Chains const& chains)
        {
            std::ostringstream msg;

            for(auto const& chain : chains)
            {
                msg << " - {";
                streamJoin(msg, chain, ", ");
                msg << "} (" << chain.size() << ")" << std::endl;
            }

            return msg.str();
        }

        std::string showGroups(Groups const& groups)
        {
            std::string rv;

            for(auto const& group : groups)
            {
                rv += fmt::format("-=-=-=-= Group: =-=-=-=-\n{}\n", showChains(group));
            }

            rv += fmt::format("{} groups\n", groups.size());

            return rv;
        }

        bool canJoin(KernelGraph const& graph, Chains const& group, Chain const& chain)
        {
            auto groupParentPair = containingAncestors(group.at(0).at(0), graph).take(1).only();
            auto chainParentPair = containingAncestors(chain.at(0), graph).take(1).only();

            std::optional<int> groupParent
                = groupParentPair ? std::optional(groupParentPair->first) : std::nullopt;
            std::optional<int> chainParent
                = chainParentPair ? std::optional(chainParentPair->first) : std::nullopt;

            auto showParent = [](auto x) -> std::string {
                if(x)
                    return std::to_string(x.value());
                else
                    return "-";
            };

            if(groupParent != chainParent)
            {
                Log::debug("Chain {{{}}} can't join group {{{}}} due to different parents ({}/{})",
                           fmt::join(chain, ", "),
                           showChains(group),
                           showParent(chainParent),
                           showParent(groupParent));
                return false;
            }

            for(auto const& groupChain : group)
            {
                auto order = graph.control.compareNodes(UpdateCache, groupChain.at(0), chain.at(0));
                if(order != ControlGraph::NodeOrdering::Undefined)
                {
                    Log::debug(
                        "Chain {{{}}} can't join group {{{}}} due to defined order ({} {} {})",
                        fmt::join(chain, ", "),
                        showChains(group),
                        groupChain.at(0),
                        toString(order),
                        chain.at(0));
                    return false;
                }
            }
            return true;
        }

        Groups identifyParallelChains(KernelGraph const& graph, Groups groups)
        {
            Groups rv;
            if(groups.empty())
                return rv;

            rv.reserve(groups[0].size());
            for(auto& chain : groups[0])
            {
                rv.push_back({std::move(chain)});
            }

            // Skip the first input group
            for(auto& inputGroup : groups | std::views::drop(1))
            {
                for(auto& chain : inputGroup)
                {
                    bool didJoin = false;
                    for(auto& outputGroup : rv)
                    {
                        if(canJoin(graph, outputGroup, chain))
                        {
                            Log::debug(
                                "({}) joining {}", showChain(chain), showChains(outputGroup));
                            outputGroup.push_back(std::move(chain));
                            didJoin = true;
                            break;
                        }
                    }
                    if(!didJoin)
                        Log::debug("({}) joined nothing.", showChain(chain));
                }
            }

            for(auto& group : rv)
                if(group.size() == 1)
                    group.clear();

            return rv;
        }

        std::vector<ParallelChainSet>
            identifyParallelMultiplyAndLDSChainsWithTypes(KernelGraph const& graph)
        {
            auto [multiplyChains, multiplyCoordTypes] = findMultiplyChainsAndCoords(graph);
            auto ldsChains                            = findLoadLDSChains(graph);

            Log::debug("Multiply chains: \n{}", showChains(multiplyChains));
            Log::debug("LDS chains: \n{}", showChains(ldsChains));

            auto chainSets = identifyParallelChains(graph, {multiplyChains, std::move(ldsChains)});

            if(Log::logger().should_log(LogLevel::Debug))
            {
                for(auto const& group : chainSets)
                {
                    for(auto const& chain : group)
                    {
                        logChainTagTable(graph, chain);
                    }
                }
            }

            auto ts = [](auto const& x) { return toString(x); };

            Log::debug("#########################\nchainSets: [{}]\nmultiplyCoordTypes: [{}]",
                       showGroups(chainSets),
                       fmt::join(multiplyCoordTypes | std::views::transform(ts), ", "));

            if(chainSets.size() != multiplyCoordTypes.size())
            {
                Log::debug("#########################\nchainSets: [{}]\nmultiplyCoordTypes: [{}]",
                           showGroups(chainSets),
                           fmt::join(multiplyCoordTypes | std::views::transform(ts), ", "));

                Log::debug("{} != {}", chainSets.size(), multiplyCoordTypes.size());
                return {};
            }

            std::vector<ParallelChainSet> rv;

            auto getType = [&](int node) { return getLDSType(graph, node); };

            for(auto const& [chains, types] : zip(chainSets, multiplyCoordTypes))
            {
                if(chains.empty())
                    continue;

                auto ldsTypes = chains.at(1) | std::views::transform(getType);

                rv.push_back(
                    {chains.at(0), chains.at(1), types, {ldsTypes.begin(), ldsTypes.end()}});
            }

            return rv;
        }

        std::map<DataType, int> getMultiplyHeadStarts(KernelGraph&            graph,
                                                      ParallelChainSet const& chainSet)
        {
            // Give the multiply nodes a head start so we can cover the extra
            // ops at the beginning of the loop. This should be replaced with a
            // more robust heuristic.
            std::map<DataType, int> totals;

            for(auto const& types : chainSet.multiplyTagTypes)
            {
                for(auto const& [type, count] : types)
                    totals[type] += count;
            }

            std::map<DataType, int> rv;

            for(auto [inputType, inputCount] : totals)
            {
                if(inputType == DataType::None)
                    continue;

                auto const& info = DataTypeInfo::Get(inputType);

                auto packed = info.packedVariableType();

                auto outputType = packed.value_or(inputType).dataType;

                if(info.isIntegral)
                {
                    rv[outputType] = inputCount / 4;
                }
                else
                {
                    rv[outputType] = (inputCount * 3) / 8;
                }
            }

            Log::debug("Multiply head starts: {{{}}}", toString(rv));

            return rv;
        }

        void distributeChains(KernelGraph& graph, std::vector<ParallelChainSet> const& chainSets)
        {
            for(auto const& chainSet : chainSets)
            {
                std::map<DataType, int> multiplyTypeCounts = getMultiplyHeadStarts(graph, chainSet);
                std::map<DataType, int> ldsTypeCounts;

                AssertFatal(chainSet.multiplyChain.size() == chainSet.multiplyTagTypes.size(),
                            ShowValue(chainSet.multiplyChain.size()),
                            ShowValue(chainSet.multiplyTagTypes.size()));

                auto ldsIter = chainSet.ldsChain.begin();
                AssertFatal(ldsIter != chainSet.ldsChain.end());

                auto ldsType = getLDSType(graph, *ldsIter);
                Log::debug("First LDS: {} ({})", *ldsIter, toString(ldsType));

                for(auto const& [multiply, multiplyTypes] :
                    zip(chainSet.multiplyChain, chainSet.multiplyTagTypes))
                {
                    for(auto const& [dtype, count] : multiplyTypes)
                    {
                        multiplyTypeCounts[dtype] += count;
                    }

                    auto ldsCount = 0;
                    while(ldsIter != chainSet.ldsChain.end()
                          && multiplyTypeCounts[ldsType] > ldsTypeCounts[ldsType])
                    {
                        Log::debug("{} -> {} Mul: {}, LDS: {} ({})",
                                   multiply,
                                   *ldsIter,
                                   toString(multiplyTypeCounts),
                                   toString(ldsTypeCounts),
                                   getLDSNode(graph, *ldsIter));
                        graph.control.chain<ControlGraph::Sequence>(multiply, *ldsIter);

                        ldsTypeCounts[ldsType]++;

                        ++ldsIter;
                        if(ldsIter != chainSet.ldsChain.end())
                            ldsType = getLDSType(graph, *ldsIter);

                        ++ldsCount;
                    }

                    if(ldsCount == 0 && ldsIter != chainSet.ldsChain.end())
                        Log::debug("Skipping multiply {}. LDS: {} ({})",
                                   multiply,
                                   *ldsIter,
                                   toString(ldsType));
                }

                Log::debug("Multiply type counts: {{{}}}", toString(multiplyTypeCounts));
                Log::debug("LDS type counts: {{{}}}", toString(ldsTypeCounts));
            }
        }
    }

    KernelGraph ScheduleMultiplyAndLDS::apply(KernelGraph const& original)
    {
        using namespace ScheduleMultiplyAndLDSDetail;

        auto rv = original;

        auto chainSets = identifyParallelMultiplyAndLDSChainsWithTypes(rv);

        if(Log::getLogger()->should_log(LogLevel::Debug))
        {
            for(auto chainSet : chainSets)
            {
                auto ts = [](auto x) { return toString(x); };
                Log::debug("Chains:\nMultiplies: {{{}}}({})\nTypes: {}({})\nLDS: "
                           "{{{}}}({})\nTypes: {}({})",
                           fmt::join(chainSet.multiplyChain, ", "),
                           chainSet.multiplyChain.size(),
                           toString(chainSet.multiplyTagTypes),
                           chainSet.multiplyTagTypes.size(),
                           fmt::join(chainSet.ldsChain, ", "),
                           chainSet.ldsChain.size(),
                           fmt::join(chainSet.ldsChainTypes | std::views::transform(ts), ", "),
                           chainSet.ldsChainTypes.size());
            }
        }

        distributeChains(rv, chainSets);
        return rv;
    }
}
