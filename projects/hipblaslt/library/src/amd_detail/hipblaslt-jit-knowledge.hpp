// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-prediction.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

// The tuned rows of TensileLite's logic files, as Tensile.JitKnowledge writes
// them at build time: a header that indexes one compressed block per hardware
// branch and ProblemType. A block is read and inflated by the first match
// that needs it, and then kept.
namespace hipblaslt_jit::knowledge
{
    // The epilogue a ProblemType supports, or a request needs.
    struct Features
    {
        std::vector<int>      bias; // bias data types; empty without a bias
        bool                  activation = false;
        std::string           scaleAB; // UseScaleAB, or empty
        std::set<std::string> flags; // UseE, Gradient, ...
    };

    struct Problem
    {
        std::string           coreKey; // the fields a ProblemType must equal
        Features              features;
        std::array<size_t, 4> size{}; // M, N, batch, K
    };

    // The hardware facts the branches test.
    struct Device
    {
        int                cuCount = 0;
        std::optional<int> pciChipId; // only where chip ID rows apply
        std::vector<int>   fallbackChipIds; // chip IDs whose rows it may also use
    };

    // One tuned parameter set near the problem.
    struct Seed
    {
        std::array<size_t, 2>        macroTile{};
        std::array<size_t, 2>        waves{};
        std::array<size_t, 4>        instruction{}; // M, N, K, blocks
        size_t                       depthU = 0;
        ExecutionPolicy              policy;
        int64_t                      globalSplitU = 0;
        std::vector<TuningParameter> parameters; // all the set's parameters
        std::vector<std::pair<std::string, int64_t>> asserts; // alignment asserts
        size_t                                       branch = 0;
        std::string                                  source; // logic file and solution index
        std::array<size_t, 4>                        row{}; // the nearest tuned size
        double                                       distance = 0; // log2 Euclidean
        size_t                                       rank     = 0; // equal ranks tie
    };

    struct Branch
    {
        std::string      kind; // pci, cu or generic
        int              cuCount = 0; // 0: any
        std::vector<int> pciChipIds;
    };

    struct Group
    {
        size_t      branch = 0;
        std::string name; // the ProblemType's name in its logic files
        std::string coreKey;
        Features    features;
        uint64_t    offset = 0, length = 0, rows = 0, sets = 0;
    };

    struct Match
    {
        std::vector<Seed> seeds; // nearest first
        std::string       group; // the group they came from, or empty
        std::string       corrupt; // why a block failed to decode in this call, or empty
    };

    class Database
    {
    public:
        // Reads the header. Throws std::runtime_error saying why the file
        // cannot be used: missing, unreadable, another format or schema.
        explicit Database(std::filesystem::path path);
        ~Database();

        const std::filesystem::path& path() const noexcept;
        const std::string&           arch() const noexcept;
        const std::string&           libraryArch() const noexcept;
        const std::string&           contentHash() const noexcept;
        const std::vector<Branch>&   branches() const noexcept;
        const std::vector<Group>&    groups() const noexcept;

        // Up to `count` sets of the first group, in branch order, that matches
        // the device and covers the problem, at most two per macro tile and
        // waves. A group whose block does not decode is skipped from then on,
        // and only the call that decoded it reports why.
        Match nearest(const Problem& problem, const Device& device, size_t count) const;

        // Inflates a group's block on first use. False, with the reason, if it
        // does not decode.
        bool load(size_t group, std::string& error) const;
        bool loaded(size_t group) const;

    private:
        struct State;
        std::filesystem::path  m_path;
        std::string            m_arch, m_libraryArch, m_contentHash;
        std::vector<Branch>    m_branches;
        std::vector<Group>     m_groups;
        uint64_t               m_blocks = 0; // where the blocks start
        std::unique_ptr<State> m_state;
    };
}
