// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "harness/SupportClaimsAutoGen.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <process.h>
#define hipdnn_getpid _getpid
#else
#include <unistd.h>
#define hipdnn_getpid getpid
#endif

#include "harness/SupportMatrixCollector.hpp"

namespace hipdnn_integration_tests
{

CondensedSupportData condenseSupportClaims(const std::vector<GraphSupportRecord>& records,
                                           std::string_view engineName)
{
    using Tuple = std::tuple<std::string, std::string, std::string>; // op, io, layout

    std::set<Tuple> S;
    std::set<Tuple> U;
    std::set<std::string> opsInS;

    for(const auto& record : records)
    {
        if(record.opChain.empty())
        {
            continue;
        }
        const Tuple tuple{record.opChain, record.ioDtype, record.layout};
        const bool engineSupports = record.supportingEngines.find(std::string(engineName))
                                    != record.supportingEngines.end();
        if(engineSupports)
        {
            S.insert(tuple);
            opsInS.insert(record.opChain);
        }
        else
        {
            U.insert(tuple);
        }
    }

    // For each op_chain in S, compute its safe (io × layout) rectangle:
    // start with the full observed io/layout sets for the op, then
    // greedily drop axis values until the rectangle is disjoint from U;
    // finally shrink to the densest sub-rectangle entirely in S so
    // unobserved combinations aren't claimed.
    struct SafeRect
    {
        std::vector<std::string> ios;
        std::vector<std::string> layouts;

        bool operator<(const SafeRect& other) const
        {
            return std::tie(ios, layouts) < std::tie(other.ios, other.layouts);
        }
    };

    std::map<std::string, SafeRect> opToRect;

    for(const auto& op : opsInS)
    {
        std::set<std::string> ios;
        std::set<std::string> layouts;
        for(const auto& tuple : S)
        {
            if(std::get<0>(tuple) == op)
            {
                ios.insert(std::get<1>(tuple));
                layouts.insert(std::get<2>(tuple));
            }
        }

        // Greedy shrink: while any (io, layout) in the cross-product is
        // in U, drop the axis value with the most U-hits. Bounded by the
        // sum of axis cardinalities, so trivially terminating.
        while(true)
        {
            std::map<std::string, size_t> ioHits;
            std::map<std::string, size_t> layoutHits;
            for(const auto& io : ios)
            {
                for(const auto& layout : layouts)
                {
                    if(U.find({op, io, layout}) != U.end())
                    {
                        ++ioHits[io];
                        ++layoutHits[layout];
                    }
                }
            }
            if(ioHits.empty())
            {
                break;
            }
            std::string worstIo;
            size_t worstIoCount = 0;
            for(const auto& [io, count] : ioHits)
            {
                if(count > worstIoCount || (count == worstIoCount && io < worstIo))
                {
                    worstIo = io;
                    worstIoCount = count;
                }
            }
            std::string worstLayout;
            size_t worstLayoutCount = 0;
            for(const auto& [layout, count] : layoutHits)
            {
                if(count > worstLayoutCount || (count == worstLayoutCount && layout < worstLayout))
                {
                    worstLayout = layout;
                    worstLayoutCount = count;
                }
            }
            if(worstIoCount >= worstLayoutCount)
            {
                ios.erase(worstIo);
            }
            else
            {
                layouts.erase(worstLayout);
            }
            if(ios.empty() || layouts.empty())
            {
                break;
            }
        }

        if(ios.empty() || layouts.empty())
        {
            // No clean rectangle — the op's S tuples are too entangled
            // with U to safely group. Skip; engineer can hand-write a
            // tighter matcher if they want coverage.
            continue;
        }

        // Final shrink: keep only ios/layouts whose every cross with the
        // other axis is in S. Unobserved combinations would risk Rule A
        // on first run, so we don't claim them.
        bool clean = true;
        for(const auto& io : ios)
        {
            for(const auto& layout : layouts)
            {
                if(S.find({op, io, layout}) == S.end())
                {
                    clean = false;
                }
            }
        }
        if(!clean)
        {
            std::set<std::string> denseIos;
            for(const auto& io : ios)
            {
                bool keep = true;
                for(const auto& layout : layouts)
                {
                    if(S.find({op, io, layout}) == S.end())
                    {
                        keep = false;
                        break;
                    }
                }
                if(keep)
                {
                    denseIos.insert(io);
                }
            }
            std::set<std::string> denseLayouts;
            for(const auto& layout : layouts)
            {
                bool keep = true;
                for(const auto& io : denseIos)
                {
                    if(S.find({op, io, layout}) == S.end())
                    {
                        keep = false;
                        break;
                    }
                }
                if(keep)
                {
                    denseLayouts.insert(layout);
                }
            }
            ios = std::move(denseIos);
            layouts = std::move(denseLayouts);
            if(ios.empty() || layouts.empty())
            {
                continue;
            }
        }

        SafeRect rect;
        rect.ios.assign(ios.begin(), ios.end());
        rect.layouts.assign(layouts.begin(), layouts.end());
        opToRect.emplace(op, std::move(rect));
    }

    // Group op_chains sharing the same safe rectangle into one matcher.
    std::map<SafeRect, std::vector<std::string>> rectToOps;
    for(const auto& [op, rect] : opToRect)
    {
        rectToOps[rect].push_back(op);
    }

    CondensedSupportData out;
    for(auto& [rect, ops] : rectToOps)
    {
        std::sort(ops.begin(), ops.end());
        SupportMatcher matcher;
        matcher.opChains = std::move(ops);
        matcher.ioDtypes = rect.ios;
        matcher.layouts = rect.layouts;
        out.matchers.push_back(std::move(matcher));
    }
    std::sort(
        out.matchers.begin(),
        out.matchers.end(),
        [](const SupportMatcher& a, const SupportMatcher& b) { return a.opChains < b.opChains; });

    out.unsupportedObservations = std::move(U);
    return out;
}

std::string renderSupportBlockToml(const CondensedSupportData& condensed,
                                   const std::string& archToken,
                                   const std::optional<std::string>& platform)
{
    std::ostringstream out;
    out << "[[supported]]\n";
    out << "arch = \"" << archToken << "\"\n";
    if(platform.has_value())
    {
        out << "platform = \"" << *platform << "\"\n";
    }
    out << "\n";

    for(const auto& matcher : condensed.matchers)
    {
        out << "[[supported.matchers]]\n";
        out << "op_chains = [\n";
        for(const auto& op : matcher.opChains)
        {
            out << "    \"" << op << "\",\n";
        }
        out << "]\n";

        out << "io_dtypes = [";
        for(size_t i = 0; i < matcher.ioDtypes.size(); ++i)
        {
            if(i > 0)
            {
                out << ", ";
            }
            out << "\"" << matcher.ioDtypes[i] << "\"";
        }
        out << "]\n";

        out << "layouts = [";
        for(size_t i = 0; i < matcher.layouts.size(); ++i)
        {
            if(i > 0)
            {
                out << ", ";
            }
            out << "\"" << matcher.layouts[i] << "\"";
        }
        out << "]\n\n";
    }
    return out.str();
}

namespace
{

std::string defaultHeader(const std::string& engineName)
{
    std::ostringstream out;
    out << "# " << engineName << ".supported.toml\n"
        << "# Generated by hipdnn_integration_tests --write-support-claims.\n"
        << "# Do not hand-edit — regenerate on hardware via the tool.\n"
        << "# Schema reference: dnn-providers/integration-tests/docs/"
           "support-claims-schema.md\n"
        << "\n"
        << "[meta]\n"
        << "version = 1\n"
        << "engine  = \"" << engineName << "\"\n"
        << "\n";
    return out.str();
}

std::string extractQuoted(const std::string& line)
{
    const auto first = line.find('"');
    if(first == std::string::npos)
    {
        return {};
    }
    const auto second = line.find('"', first + 1);
    if(second == std::string::npos)
    {
        return {};
    }
    return line.substr(first + 1, second - first - 1);
}

// Return {header_through_meta, body_without_matching_block(s)} from an
// existing sidecar. Block boundaries are identified by [[supported]]
// table-array starts in column 0. Anything before the first [[supported]]
// is preserved as header. Crude but deterministic and doesn't depend on
// tomlplusplus round-tripping (which would drop comments).
std::pair<std::string, std::string> stripBlockFor(const std::filesystem::path& sidecarPath,
                                                  const std::string& archToken,
                                                  const std::optional<std::string>& platform)
{
    std::ifstream in(sidecarPath);
    if(!in.is_open())
    {
        throw std::runtime_error("SupportClaimsWriter: cannot open " + sidecarPath.string());
    }

    std::vector<std::string> lines;
    std::string line;
    while(std::getline(in, line))
    {
        lines.push_back(line);
    }
    in.close();

    std::vector<size_t> blockStarts;
    size_t firstBlockLine = lines.size();
    for(size_t i = 0; i < lines.size(); ++i)
    {
        if(lines[i].rfind("[[supported]]", 0) == 0)
        {
            blockStarts.push_back(i);
            firstBlockLine = std::min(firstBlockLine, i);
        }
    }

    std::ostringstream header;
    for(size_t i = 0; i < firstBlockLine; ++i)
    {
        header << lines[i] << "\n";
    }

    std::ostringstream body;
    for(size_t bi = 0; bi < blockStarts.size(); ++bi)
    {
        const size_t start = blockStarts[bi];
        const size_t end = (bi + 1 < blockStarts.size()) ? blockStarts[bi + 1] : lines.size();
        std::string blockArch;
        std::optional<std::string> blockPlatform;
        for(size_t i = start + 1; i < end; ++i)
        {
            const auto& l = lines[i];
            if(l.rfind("arch", 0) == 0)
            {
                blockArch = extractQuoted(l);
            }
            else if(l.rfind("platform", 0) == 0)
            {
                blockPlatform = extractQuoted(l);
            }
            else if(l.rfind("[[supported.matchers]]", 0) == 0)
            {
                break;
            }
        }
        const bool archMatches = blockArch == archToken;
        const bool platformMatches = (blockPlatform.has_value() == platform.has_value())
                                     && (!blockPlatform.has_value() || *blockPlatform == *platform);
        if(archMatches && platformMatches)
        {
            continue; // drop — caller will emit the replacement
        }
        for(size_t i = start; i < end; ++i)
        {
            body << lines[i] << "\n";
        }
    }
    return {header.str(), body.str()};
}

} // namespace

void SupportClaimsWriter::writeSidecar(const std::filesystem::path& sidecarPath,
                                       const std::string& engineName,
                                       const std::string& archToken,
                                       const std::optional<std::string>& platform,
                                       const CondensedSupportData& condensed)
{
    const std::string newBlock = renderSupportBlockToml(condensed, archToken, platform);

    std::string preservedBody;
    std::string header;
    if(std::filesystem::exists(sidecarPath))
    {
        std::tie(header, preservedBody) = stripBlockFor(sidecarPath, archToken, platform);
    }
    else
    {
        header = defaultHeader(engineName);
    }

    const auto tmpPath
        = sidecarPath.parent_path()
          / (sidecarPath.filename().string() + ".tmp." + std::to_string(hipdnn_getpid()));
    std::ofstream tmp(tmpPath, std::ios::out | std::ios::trunc);
    if(!tmp.is_open())
    {
        throw std::runtime_error("SupportClaimsWriter: failed to open " + tmpPath.string()
                                 + " for writing");
    }
    tmp << header;
    if(!preservedBody.empty())
    {
        tmp << preservedBody;
        if(preservedBody.back() != '\n')
        {
            tmp << "\n";
        }
    }
    tmp << newBlock;
    tmp.flush();
    tmp.close();

    // std::filesystem::rename is atomic on Linux and on Windows when the
    // destination doesn't exist; if it does, swap via remove+rename.
    std::error_code ec;
    std::filesystem::rename(tmpPath, sidecarPath, ec);
    if(ec)
    {
        std::filesystem::remove(sidecarPath, ec);
        std::filesystem::rename(tmpPath, sidecarPath, ec);
        if(ec)
        {
            std::filesystem::remove(tmpPath);
            throw std::runtime_error("SupportClaimsWriter: failed to commit " + sidecarPath.string()
                                     + ": " + ec.message());
        }
    }
}

bool generateSupportClaimsForCurrentArch(const std::filesystem::path& sidecarPath,
                                         const std::string& engineName,
                                         const std::string& archToken,
                                         const std::optional<std::string>& platform)
{
    const auto records = SupportMatrixCollector::get().getRecords();
    if(records.empty())
    {
        std::cerr << "[--write-support-claims] no observations recorded — refusing to write "
                     "an empty block (run without --gtest_filter and ensure the integration "
                     "suite ran).\n";
        return false;
    }

    const auto condensed = condenseSupportClaims(records, engineName);
    if(condensed.matchers.empty())
    {
        std::cerr << "[--write-support-claims] engine '" << engineName
                  << "' returned support for zero observed tuples on arch=" << archToken
                  << "; nothing to write.\n";
        return false;
    }

    SupportClaimsWriter::writeSidecar(sidecarPath, engineName, archToken, platform, condensed);

    std::cerr << "[--write-support-claims] wrote " << condensed.matchers.size()
              << " matcher(s) for arch=" << archToken
              << " platform=" << (platform.has_value() ? *platform : "any") << " to " << sidecarPath
              << "\n";

    if(!condensed.unsupportedObservations.empty())
    {
        std::cerr << "[--write-support-claims] " << condensed.unsupportedObservations.size()
                  << " observed (op_chain, io_dtype, layout) tuples were in U (engine returned "
                     "no support); they are NOT included in any matcher. Review the sidecar "
                     "diff to confirm the carve-outs are intentional. Examples:\n";
        size_t shown = 0;
        for(const auto& [op, io, layout] : condensed.unsupportedObservations)
        {
            if(shown++ >= 10)
            {
                std::cerr << "    ... (" << condensed.unsupportedObservations.size() - 10
                          << " more)\n";
                break;
            }
            std::cerr << "    (\"" << op << "\", \"" << io << "\", \"" << layout << "\")\n";
        }
    }
    return true;
}

} // namespace hipdnn_integration_tests
