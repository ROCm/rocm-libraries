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

namespace
{

using Cell = std::pair<std::string, std::string>; // (io_dtype, layout)

// One axis-aligned rectangle in (io, layout) space, defined as the
// cross-product of two axis subsets. Sortable for deterministic grouping.
struct Rect
{
    std::vector<std::string> ios;
    std::vector<std::string> layouts;

    bool operator<(const Rect& other) const
    {
        return std::tie(ios, layouts) < std::tie(other.ios, other.layouts);
    }
};

// Find the largest safe rectangle: the (io_subset × layout_subset) that
// covers the most cells in `targets` while containing zero cells in
// `forbidden`. Brute-force over the power set of each axis — fine because
// axis cardinalities are ≤8 in practice (3 dtypes × 6 layouts = 18
// candidate values total, 2^18 worst case ≈ 256k iterations).
//
// Tie-break order (deterministic so two engineers on the same hardware
// produce identical sidecars):
//   1. Most target cells covered (greedy gain).
//   2. Smallest rect size (fewer unobserved cells claimed — reduces
//      Rule A risk on first run).
//   3. Lexicographic order on (ios, layouts) — stable tiebreaker.
//
// Returns an empty rect if no candidate covers any target cell.
Rect findLargestSafeRectangle(const std::set<Cell>& targets,
                              const std::set<Cell>& forbidden,
                              const std::vector<std::string>& allIos,
                              const std::vector<std::string>& allLayouts)
{
    Rect best;
    size_t bestScore = 0;
    size_t bestSize = SIZE_MAX;

    const size_t ioN = allIos.size();
    const size_t layoutN = allLayouts.size();
    for(size_t ioMask = 1; ioMask < (size_t{1} << ioN); ++ioMask)
    {
        std::vector<std::string> ios;
        for(size_t i = 0; i < ioN; ++i)
        {
            if((ioMask >> i) & size_t{1})
            {
                ios.push_back(allIos[i]);
            }
        }
        for(size_t layoutMask = 1; layoutMask < (size_t{1} << layoutN); ++layoutMask)
        {
            std::vector<std::string> layouts;
            for(size_t i = 0; i < layoutN; ++i)
            {
                if((layoutMask >> i) & size_t{1})
                {
                    layouts.push_back(allLayouts[i]);
                }
            }

            // Safety: reject if any cell in the cross-product is forbidden.
            bool safe = true;
            for(const auto& io : ios)
            {
                for(const auto& layout : layouts)
                {
                    if(forbidden.find({io, layout}) != forbidden.end())
                    {
                        safe = false;
                        break;
                    }
                }
                if(!safe)
                {
                    break;
                }
            }
            if(!safe)
            {
                continue;
            }

            size_t score = 0;
            for(const auto& io : ios)
            {
                for(const auto& layout : layouts)
                {
                    if(targets.find({io, layout}) != targets.end())
                    {
                        ++score;
                    }
                }
            }
            if(score == 0)
            {
                continue;
            }

            const size_t size = ios.size() * layouts.size();
            const bool better = score > bestScore || (score == bestScore && size < bestSize)
                                || (score == bestScore && size == bestSize
                                    && std::tie(ios, layouts) < std::tie(best.ios, best.layouts));
            if(better)
            {
                bestScore = score;
                bestSize = size;
                best.ios = ios;
                best.layouts = layouts;
            }
        }
    }
    return best;
}

// Greedy rectangle cover: repeatedly pick the largest safe rectangle and
// remove the covered targets, until none remain. RFC 0012 §7's
// Coverage+Safety contract requires the result to cover every cell in
// `targets` without touching `forbidden`; the loop terminates because
// each iteration removes ≥1 target cell (or bails on empty result).
//
// Note: a single op_chain may need multiple rectangles when its S cells
// are interleaved with U cells — the previous greedy axis-shrink
// implementation couldn't emit more than one rectangle per op_chain and
// silently dropped coverage. Reviewer's counterexample:
//   U = {(fp16,NCHW), (fp32,NHWC)}, S = {(fp16,NHWC), (fp32,NCHW)}
//   → correct cover is {fp16}×{NHWC} ∪ {fp32}×{NCHW} (two rectangles).
std::vector<Rect> findRectangleCover(std::set<Cell> targets,
                                     const std::set<Cell>& forbidden,
                                     const std::vector<std::string>& allIos,
                                     const std::vector<std::string>& allLayouts)
{
    std::vector<Rect> result;
    while(!targets.empty())
    {
        Rect r = findLargestSafeRectangle(targets, forbidden, allIos, allLayouts);
        if(r.ios.empty() || r.layouts.empty())
        {
            // No safe rectangle exists that covers any remaining target.
            // The leftover targets are isolated between forbidden cells —
            // surface them in the diagnostic stream and bail.
            break;
        }
        for(const auto& io : r.ios)
        {
            for(const auto& layout : r.layouts)
            {
                targets.erase({io, layout});
            }
        }
        result.push_back(std::move(r));
    }
    return result;
}

} // namespace

CondensedSupportData condenseSupportClaims(const std::vector<GraphSupportRecord>& records,
                                           std::string_view engineName)
{
    using Tuple = std::tuple<std::string, std::string, std::string>; // op, io, layout

    std::set<Tuple> S;
    std::set<Tuple> U;
    std::set<std::string> opsInS;
    std::set<std::string> allIosSet;
    std::set<std::string> allLayoutsSet;

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
        allIosSet.insert(record.ioDtype);
        allLayoutsSet.insert(record.layout);
    }

    // Sort axis values once; rectangle enumeration uses these as the
    // bitmask alphabet so output order is deterministic across runs.
    const std::vector<std::string> allIos(allIosSet.begin(), allIosSet.end());
    const std::vector<std::string> allLayouts(allLayoutsSet.begin(), allLayoutsSet.end());

    // For each op_chain in S, compute its rectangle cover. Iterate ops
    // in sorted order — keeps emit order stable for identical inputs.
    std::map<std::string, std::vector<Rect>> opToRects;
    for(const auto& op : opsInS)
    {
        std::set<Cell> S_op;
        std::set<Cell> U_op;
        for(const auto& [tupleOp, tupleIo, tupleLayout] : S)
        {
            if(tupleOp == op)
            {
                S_op.emplace(tupleIo, tupleLayout);
            }
        }
        for(const auto& [tupleOp, tupleIo, tupleLayout] : U)
        {
            if(tupleOp == op)
            {
                U_op.emplace(tupleIo, tupleLayout);
            }
        }

        auto cover = findRectangleCover(S_op, U_op, allIos, allLayouts);
        if(!cover.empty())
        {
            opToRects.emplace(op, std::move(cover));
        }
    }

    // Group by rectangle: each emitted matcher is one rectangle plus the
    // set of op_chains whose cover includes it. Two ops sharing a
    // rectangle land in the same matcher; ops with disjoint covers land
    // in separate matchers.
    std::map<Rect, std::set<std::string>> rectToOps;
    for(const auto& [op, rects] : opToRects)
    {
        for(const auto& rect : rects)
        {
            rectToOps[rect].insert(op);
        }
    }

    CondensedSupportData out;
    for(const auto& [rect, ops] : rectToOps)
    {
        SupportMatcher matcher;
        matcher.opChains.assign(ops.begin(), ops.end());
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

// RFC 0012 §8.3 shrinkage-refusal precondition. Before overwriting the
// sidecar, check the existing block for (archToken, platform): if any
// of its matchers' cross-product tuples are entirely absent from the
// current observation set, the engineer is running a partial baseline
// and would silently drop a previously valid claim. Refuse and tell
// them what's missing.
//
// Returns true if it's safe to proceed (no shrinkage detected, or no
// existing block to compare against). Returns false if the run would
// shrink the sidecar — caller aborts.
bool checkShrinkagePrecondition(const std::filesystem::path& sidecarPath,
                                const std::string& engineName,
                                const std::string& archToken,
                                const std::optional<std::string>& platform,
                                const std::vector<GraphSupportRecord>& records)
{
    if(!std::filesystem::exists(sidecarPath))
    {
        // First-time bring-up — nothing to shrink.
        return true;
    }

    std::optional<SupportClaims> existing;
    try
    {
        existing.emplace(sidecarPath, engineName);
    }
    catch(const std::exception& ex)
    {
        // A broken/mismatched sidecar is a different failure mode —
        // surface it directly rather than silently overwriting.
        std::cerr << "[--write-support-claims] cannot read existing sidecar for "
                     "shrinkage check: "
                  << ex.what() << "\n"
                  << "  Either fix the sidecar's [meta] or delete it to start fresh.\n";
        return false;
    }

    const std::string platformStr = platform.has_value() ? *platform : std::string{};
    const auto* existingBlock = existing->blockFor(archToken, platformStr);
    if(existingBlock == nullptr)
    {
        // No prior block for this (arch, platform) — first-time bring-up
        // for this asic.
        return true;
    }

    // Build the observed-tuple set from the current run.
    std::set<std::tuple<std::string, std::string, std::string>> observed;
    for(const auto& r : records)
    {
        if(r.opChain.empty())
        {
            continue;
        }
        observed.emplace(r.opChain, r.ioDtype, r.layout);
    }

    // Collect any matchers whose cross-product has zero overlap with the
    // observed set. List them all rather than bailing on the first —
    // gives the engineer the full picture in one error.
    std::vector<const SupportMatcher*> zeroCoverage;
    for(const auto& matcher : existingBlock->matchers)
    {
        bool anyObserved = false;
        matcher.forEachTuple(
            [&](const std::string& op, const std::string& io, const std::string& layout) {
                if(observed.find({op, io, layout}) != observed.end())
                {
                    anyObserved = true;
                    return false;
                }
                return true;
            });
        if(!anyObserved)
        {
            zeroCoverage.push_back(&matcher);
        }
    }

    if(zeroCoverage.empty())
    {
        return true;
    }

    std::cerr << "[--write-support-claims] REFUSING to overwrite sidecar (RFC 0012 §8.3): "
              << zeroCoverage.size() << " existing matcher(s) for arch=" << archToken
              << " platform=" << (platform.has_value() ? *platform : "any")
              << " had zero observed coverage in this run. Overwriting now would silently "
                 "drop previously valid claims.\n";
    for(const auto* matcher : zeroCoverage)
    {
        std::cerr << "  - " << matcher->sourceLocation << "\n"
                  << "    op_chains[0] = \""
                  << (matcher->opChains.empty() ? "" : matcher->opChains.front())
                  << "\"  io_dtypes=" << matcher->ioDtypes.size()
                  << "  layouts=" << matcher->layouts.size() << "\n";
    }
    std::cerr << "  This usually means a partial run (--gtest_filter set, or the suite ran on "
                 "different hardware than the existing block was generated for). Investigate "
                 "before regenerating: either rerun unfiltered on matching hardware, or hand-"
                 "remove the stale block from the sidecar and regenerate.\n";
    return false;
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

    if(!checkShrinkagePrecondition(sidecarPath, engineName, archToken, platform, records))
    {
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
