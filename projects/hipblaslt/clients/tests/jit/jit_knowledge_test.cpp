// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-json.hpp"
#include "hipblaslt-jit-knowledge.hpp"
#include "hipblaslt-jit-prediction.hpp"
#include "hipblaslt-jit-problem-type.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <msgpack.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

namespace knowledge = hipblaslt_jit::knowledge;
namespace fs        = std::filesystem;

// Lowering needs the Tensile host; the requests here are not GEMMs.
TensileLite::ContractionProblemGemm
    hipblaslt_jit::lowerForJit(const hipblaslt_ext::experimental::jit::detail::GemmRequest&)
{
    throw std::runtime_error("lowerForJit is not linked into this test");
}

namespace
{
    void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }

    using Packer = msgpack::packer<msgpack::sbuffer>;

    // A fixture set: its name is its source file, so results read by name.
    struct Set
    {
        std::string name;
        size_t      mt0, mt1, depthU = 64;
        std::string strategy = "None", assignment = "StaticGrid";
        int         gsu      = 1;
        std::vector<std::array<size_t, 4>> rows;
    };

    struct Group
    {
        size_t           branch;
        std::string      name, core;
        std::vector<int> bias;
        bool             activation = false, useE = false;
        std::vector<Set> sets;
        bool             corrupt = false;
    };

    void features(Packer& p, const Group& g)
    {
        p.pack_map(uint32_t(!g.bias.empty() + g.activation + g.useE));
        if(!g.bias.empty())
            p.pack(std::string("Bias")), p.pack(g.bias);
        if(g.activation)
            p.pack(std::string("Activation")), p.pack(true);
        if(g.useE)
            p.pack(std::string("UseE")), p.pack(true);
    }

    std::vector<char> block(const Group& g)
    {
        if(g.corrupt)
            return {'n', 'o', 't', ' ', 'z', 'l', 'i', 'b'};
        msgpack::sbuffer buffer;
        Packer           p(buffer);
        p.pack_map(3);
        p.pack(std::string("param_dictionary"));
        p.pack_array(uint32_t(g.sets.size() * 4));
        for(const auto& s : g.sets)
        {
            const std::vector<size_t> mi{16, 16, 32, 1, 1, s.mt0 / 32, s.mt1 / 32, 2, 2};
            p.pack_array(2), p.pack(std::string("MatrixInstruction")), p.pack(mi);
            p.pack_array(2), p.pack(std::string("DepthU")), p.pack(s.depthU);
            p.pack_array(2), p.pack(std::string("ScaleFactor")), p.pack(1.5);
            p.pack_array(2), p.pack(std::string("TileProcessingStrategy")), p.pack(s.strategy);
        }
        p.pack(std::string("sets"));
        p.pack_array(uint32_t(g.sets.size()));
        size_t rows = 0;
        for(size_t i = 0; i < g.sets.size(); ++i)
        {
            const auto& s = g.sets[i];
            rows += s.rows.size();
            p.pack_map(11);
            p.pack(std::string("macro_tile")), p.pack(std::vector<size_t>{s.mt0, s.mt1});
            p.pack(std::string("waves")), p.pack(std::vector<int>{2, 2});
            p.pack(std::string("instruction")), p.pack(std::vector<int>{16, 16, 32, 1});
            p.pack(std::string("depth_u")), p.pack(s.depthU);
            p.pack(std::string("nt")), p.pack(std::vector<int>{0, 0});
            p.pack(std::string("policy"));
            p.pack_map(2);
            p.pack(std::string("strategy")), p.pack(s.strategy);
            p.pack(std::string("assignment")), p.pack(s.assignment);
            p.pack(std::string("gsu")), p.pack(s.gsu);
            p.pack(std::string("gsu_algorithm")), p.pack(std::string("MultipleBuffer"));
            p.pack(std::string("params"));
            p.pack(std::vector<size_t>{4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3});
            p.pack(std::string("asserts"));
            p.pack_map(1);
            p.pack(std::string("AssertFree0ElementMultiple")), p.pack(8);
            p.pack(std::string("source"));
            p.pack_map(3);
            p.pack(std::string("file")), p.pack(s.name);
            p.pack(std::string("index")), p.pack(i);
            p.pack(std::string("kernel")), p.pack(std::string("ignored"));
        }
        p.pack(std::string("rows"));
        p.pack_array(uint32_t(rows));
        for(size_t i = 0; i < g.sets.size(); ++i)
            for(const auto& r : g.sets[i].rows)
                p.pack(std::vector<size_t>{r[0], r[1], r[2], r[3], i, 0});
        std::vector<char> out(compressBound(buffer.size()));
        uLongf            size = out.size();
        require(compress2(reinterpret_cast<Bytef*>(out.data()),
                          &size,
                          reinterpret_cast<const Bytef*>(buffer.data()),
                          buffer.size(),
                          6)
                    == Z_OK,
                "compress2 failed");
        out.resize(size);
        return out;
    }

    // Branches: 0 MI355X (0x75a3), 1 0x75a0 with 256 CUs, 2 256 CUs, 3 generic.
    std::vector<Group> fixture()
    {
        const std::array<size_t, 4> q{1024, 1024, 1, 1024};
        std::vector<Group>          groups;
        groups.push_back({0, "pci", "K1", {}, false, false, {{"pci", 128, 128, 64, "None",
                                                              "StaticGrid", 1, {q}}}});
        groups.push_back({1, "fallback", "K5", {}, false, false, {{"fallback", 64, 64, 64, "None",
                                                                   "StaticGrid", 1, {q}}}});
        groups.push_back({2, "cu", "K3", {}, false, false, {{"cu", 64, 64, 64, "None",
                                                             "StaticGrid", 1, {q}}}});
        Group generic{3, "generic", "K1", {}, false, false, {}};
        generic.sets = {
            {"A", 128, 128, 64, "None", "StaticGrid", 1, {q, {8192, 8192, 1, 8192}}},
            {"D", 192, 128, 64, "StreamK", "Hybrid", 0, {{2048, 1024, 1, 1024}}},
            {"C", 128, 128, 32, "None", "StaticGrid", 1, {{512, 1024, 1, 1024}}},
            {"B", 128, 128, 128, "None", "StaticGrid", 1, {{1024, 2048, 1, 1024}}},
            {"E", 64, 64, 64, "None", "StaticGrid", -1, {{1024, 1024, 1, 2048}}},
            {"F", 32, 32, 64, "None", "StaticGrid", 4, {{1024, 1024, 1, 512}}},
            {"G", 256, 256, 64, "None", "StaticGrid", 1, {{4096, 4096, 1, 4096}}},
        };
        groups.push_back(generic);
        groups.push_back({3, "bias_act", "K1", {0}, true, false, {{"BA", 64, 64, 64, "None",
                                                                   "StaticGrid", 1, {q}}}});
        groups.push_back({3, "bias_act_e", "K1", {0, 4}, true, true, {{"BAE", 64, 64, 64, "None",
                                                                       "StaticGrid", 1, {q}}}});
        groups.push_back({3, "other", "K2", {}, false, false, {{"K2", 64, 64, 64, "None",
                                                                "StaticGrid", 1, {q}}}});
        groups.push_back({3, "generic_k3", "K3", {}, false, false, {{"GK3", 64, 64, 64, "None",
                                                                     "StaticGrid", 1, {q}}}});
        groups.push_back({3, "corrupt", "K4", {}, false, false, {{"bad", 64, 64, 64, "None",
                                                                  "StaticGrid", 1, {q}}}, true});
        groups.push_back({3, "k4_e", "K4", {}, false, true, {{"K4E", 64, 64, 64, "None",
                                                              "StaticGrid", 1, {q}}}});
        return groups;
    }

    void write(const fs::path& path,
               const std::vector<Group>& groups,
               const std::string&        hash   = "0123456789abcdef",
               uint32_t                  schema = 1,
               const char*               magic  = "HJKN",
               const std::string&        arch   = "gfx950")
    {
        std::vector<std::vector<char>> blocks;
        for(const auto& g : groups)
            blocks.push_back(block(g));
        msgpack::sbuffer header;
        Packer           p(header);
        p.pack_map(7);
        p.pack(std::string("generator")), p.pack(std::string("tensilelite-logic-knowledge"));
        p.pack(std::string("source_commit")), p.pack(std::string("fixture"));
        p.pack(std::string("content_hash")), p.pack(hash);
        p.pack(std::string("arch")), p.pack(arch);
        p.pack(std::string("library_arch")), p.pack(arch);
        p.pack(std::string("branches"));
        p.pack_array(4);
        const auto branch = [&](const char* kind, int cu, std::vector<int> ids) {
            p.pack_map(3);
            p.pack(std::string("kind")), p.pack(std::string(kind));
            p.pack(std::string("cu_count"));
            cu ? p.pack(cu) : p.pack_nil();
            p.pack(std::string("pci_ids")), p.pack(ids);
        };
        branch("pci", 0, {0x75a3});
        branch("pci", 256, {0x75a0});
        branch("cu", 256, {});
        branch("generic", 0, {});
        p.pack(std::string("index"));
        p.pack_array(uint32_t(groups.size()));
        size_t offset = 0;
        for(size_t i = 0; i < groups.size(); ++i)
        {
            const auto& g    = groups[i];
            size_t      rows = 0;
            for(const auto& s : g.sets)
                rows += s.rows.size();
            p.pack_map(7);
            p.pack(std::string("branch")), p.pack(g.branch);
            p.pack(std::string("problem_type"));
            p.pack_map(2);
            p.pack(std::string("name")), p.pack(g.name);
            p.pack(std::string("features"));
            features(p, g);
            p.pack(std::string("core_key")), p.pack(g.core);
            p.pack(std::string("offset")), p.pack(offset);
            p.pack(std::string("length")), p.pack(blocks[i].size());
            p.pack(std::string("rows")), p.pack(rows);
            p.pack(std::string("sets")), p.pack(g.sets.size());
            offset += blocks[i].size();
        }
        std::ofstream out(path, std::ios::binary);
        const auto    word = [&](uint32_t v) {
            const char bytes[4] = {char(v), char(v >> 8), char(v >> 16), char(v >> 24)};
            out.write(bytes, 4);
        };
        out.write(magic, 4);
        word(schema);
        word(uint32_t(header.size()));
        out.write(header.data(), header.size());
        for(const auto& b : blocks)
            out.write(b.data(), b.size());
        require(bool(out), "Cannot write " + path.string());
    }

    knowledge::Problem problem(std::string core, knowledge::Features features = {})
    {
        return {std::move(core), std::move(features), {1024, 1024, 1, 1024}};
    }

    std::vector<std::string> names(const knowledge::Match& match)
    {
        std::vector<std::string> result;
        for(const auto& seed : match.seeds)
            result.push_back(seed.source.substr(0, seed.source.find('#')));
        return result;
    }

    std::string join(const std::vector<std::string>& values)
    {
        std::string out;
        for(const auto& v : values)
            out += (out.empty() ? "" : ",") + v;
        return out;
    }

    void expect(const knowledge::Match& match, const std::string& want, const std::string& what)
    {
        const auto got = join(names(match));
        require(got == want, what + ": expected [" + want + "], got [" + got + "]");
    }

    std::string thrown(const fs::path& path)
    {
        try
        {
            knowledge::Database database(path);
        }
        catch(const std::runtime_error& e)
        {
            return e.what();
        }
        return "";
    }

    const knowledge::Device generic{304, std::nullopt, {}};

    void testOrder(const fs::path& path)
    {
        knowledge::Database database(path);
        require(database.arch() == "gfx950" && database.contentHash() == "0123456789abcdef"
                    && database.branches().size() == 4 && database.groups().size() == 10,
                "The header does not round-trip");
        auto match = database.nearest(problem("K1"), generic, 8);
        // A is exact; C, E and F tie at distance 1 with no tile waste; B is a third
        // 128x128 set; D wastes tile area; G is farthest.
        expect(match, "A,C,E,F,D,G", "log-distance order");
        std::vector<size_t> ranks;
        for(const auto& seed : match.seeds)
            ranks.push_back(seed.rank);
        require(ranks == std::vector<size_t>{0, 1, 1, 1, 2, 3}, "Equal seeds must tie");
        require(match.seeds[0].distance == 0 && match.seeds[0].row[0] == 1024
                    && match.seeds[1].distance == 1 && match.group == "generic (branch 3)",
                "The nearest row of each set is reported");
        expect(database.nearest(problem("K1"), generic, 3), "A,C,E", "the count");
        expect(database.nearest(problem("K1"), generic, 8), "A,C,E,F,D,G", "a repeated match");

        const auto& a = match.seeds[0];
        const std::vector<hipblaslt_jit::TuningParameter> parameters{
            {"MatrixInstruction", "[16,16,32,1,1,4,4,2,2]"},
            {"DepthU", "64"},
            {"ScaleFactor", "1.5"},
            {"TileProcessingStrategy", "\"None\""},
        };
        require(a.parameters.size() == parameters.size(), "A has four parameters");
        for(size_t i = 0; i < parameters.size(); ++i)
            require(a.parameters[i].name == parameters[i].name
                        && a.parameters[i].json == parameters[i].json,
                    "Parameter " + parameters[i].name + " is " + a.parameters[i].json);
        require(a.macroTile == std::array<size_t, 2>{128, 128} && a.depthU == 64
                    && a.instruction == std::array<size_t, 4>{16, 16, 32, 1}
                    && a.asserts.size() == 1 && a.asserts[0].second == 8 && a.branch == 3
                    && a.source == "A#0",
                "Seed A's fields");
        const auto& d = match.seeds[4];
        require(d.policy.strategy == hipblaslt_jit::ExecutionPolicy::Strategy::StreamK
                    && d.policy.assignment == hipblaslt_jit::ExecutionPolicy::Assignment::Hybrid
                    && d.globalSplitU == 0 && match.seeds[2].globalSplitU == -1,
                "Policies and GlobalSplitU are kept");
    }

    void testProblemTypes(const fs::path& path)
    {
        knowledge::Database database(path);
        expect(database.nearest(problem("K1", {{0}, false, "", {}}), generic, 8), "BA",
               "the covering type with the fewest extras");
        expect(database.nearest(problem("K1", {{4}, true, "", {}}), generic, 8), "BAE",
               "a bias type only the larger type covers");
        expect(database.nearest(problem("K1", {{}, false, "Scalar", {}}), generic, 8), "",
               "an epilogue no type covers");
        expect(database.nearest(problem("K2"), generic, 8), "K2", "another core type");
        const auto none = database.nearest(problem("K9"), generic, 8);
        require(none.seeds.empty() && none.group.empty(), "An unknown core type has no seeds");
    }

    void testBranches(const fs::path& path)
    {
        knowledge::Database     database(path);
        const knowledge::Device mi355x{256, 0x75a3, {0x75a0}};
        const knowledge::Device mi350{256, 0x75a2, {0x75a0}};
        const knowledge::Device cu256{256, std::nullopt, {}};
        expect(database.nearest(problem("K1"), mi355x, 1), "pci", "an exact PCI row");
        expect(database.nearest(problem("K1"), mi350, 1), "A", "exact rows before fallback chips");
        expect(database.nearest(problem("K5"), mi350, 1), "fallback", "a fallback chip");
        expect(database.nearest(problem("K5"), cu256, 1), "", "no chip ID, no PCI rows");
        expect(database.nearest(problem("K3"), cu256, 1), "cu", "a CU row");
        expect(database.nearest(problem("K3"), generic, 1), "GK3", "a CU row needs its CU count");
        expect(database.nearest(problem("K3"), mi355x, 1), "cu",
               "a branch without the type falls through");
    }

    void testLazy(const fs::path& path)
    {
        knowledge::Database database(path);
        for(size_t g = 0; g < database.groups().size(); ++g)
            require(!database.loaded(g), "No block decodes when the header loads");
        database.nearest(problem("K1"), generic, 8);
        for(size_t g = 0; g < database.groups().size(); ++g)
            require(database.loaded(g) == (database.groups()[g].name == "generic"),
                    "Only the requested group decodes, not " + database.groups()[g].name);

        auto first = database.nearest(problem("K4"), generic, 8);
        require(first.corrupt.rfind("corrupt (branch 3): ", 0) == 0,
                "The decoding call reports the corrupt block: " + first.corrupt);
        expect(first, "K4E", "the next covering type after a corrupt block");
        auto second = database.nearest(problem("K4"), generic, 8);
        require(second.corrupt.empty(), "A corrupt block is reported once");
        expect(second, "K4E", "the corrupt block stays disabled");
        std::string error;
        require(!database.load(8, error) && !error.empty() && database.load(9, error),
                "Only the corrupt group fails to load");
    }

    void testHeaders(const fs::path& dir)
    {
        const auto groups = fixture();
        require(thrown(dir / "missing.dat.zlib").rfind("no file at ", 0) == 0, "A missing file");
        write(dir / "schema.dat.zlib", groups, "0123456789abcdef", 2);
        require(thrown(dir / "schema.dat.zlib").find("has schema 2, not 1") != std::string::npos,
                "Another schema");
        write(dir / "magic.dat.zlib", groups, "0123456789abcdef", 1, "NOPE");
        require(thrown(dir / "magic.dat.zlib").find("is not a JIT knowledge file")
                    != std::string::npos,
                "Another format");
        std::ofstream(dir / "short.dat.zlib") << "HJ";
        require(thrown(dir / "short.dat.zlib").find("cannot read the preamble")
                    != std::string::npos,
                "A truncated file");
        std::ofstream(dir / "header.dat.zlib", std::ios::binary)
            .write("HJKN\x01\0\0\0\x04\0\0\0\xc0\xc0\xc0\xc0", 16);
        require(thrown(dir / "header.dat.zlib").find("has a malformed header")
                    != std::string::npos,
                "A malformed header");
    }

    void setEnvironment(const char* name, const char* value)
    {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        value ? setenv(name, value, 1) : unsetenv(name);
#endif
    }

    struct NotGemm final : hipblaslt_jit::OperationRequest
    {
        std::string_view kind() const noexcept override
        {
            return "not-gemm";
        }
    };

    std::vector<std::string> loadLines(const fs::path& log)
    {
        std::vector<std::string> lines;
        std::ifstream            in(log);
        for(std::string line; std::getline(in, line);)
            if(line.find("\"cat\":\"knowledge\",\"ev\":\"load\"") != std::string::npos)
                lines.push_back(line);
        return lines;
    }

    void expectLine(const std::string& line, const std::vector<std::string>& fields)
    {
        for(const auto& field : fields)
            require(line.find(field) != std::string::npos, "No " + field + " in " + line);
    }

    // main routes knowledge records to log before any is written.
    void testLibrary(const fs::path& dir, const fs::path& log)
    {
        const auto groups = fixture();
        const auto tree   = dir / "library";
        for(const char* arch : {"gfx950", "gfx942", "gfx1250"})
            fs::create_directories(tree / arch);
        const auto file = [&](const char* arch) {
            return tree / arch / ("hipblaslt-jit-knowledge-" + std::string(arch) + ".dat.zlib");
        };
        write(file("gfx950"), groups, "aaaa");
        write(file("gfx942"), groups, "bbbb");
        write(file("gfx1250"), groups, "cccc", 2, "HJKN", "gfx1250");

        const auto                  catalog = hipblaslt_jit::makeCatalogKnowledge();
        NotGemm                     request;
        hipblaslt_jit::DeviceTarget target;
        target.cuCount     = 256;
        const auto queries = [&](const hipblaslt_jit::TuningKnowledge& source, const char* arch) {
            target.libraryArch = arch;
            for(int i = 0; i < 2; ++i)
                require(source.seeds(request, target).size()
                            == catalog->seeds(request, target).size(),
                        "A request that is not a GEMM has only catalog seeds");
        };

        const auto installed = hipblaslt_jit::makeTuningLibraryKnowledge(tree, true);
        require(installed->id() == "tensilelite-logic.v1" && installed->version() == "gfx950=aaaa",
                "Installed version " + installed->version());
        for(const char* arch : {"gfx950", "gfx942", "gfx1250", "gfx90a"})
            queries(*installed, arch);
        auto lines = loadLines(log);
        require(lines.size() == 4, "One record per architecture, not " + std::to_string(lines.size()));
        expectLine(lines[0], {"\"arch\":\"gfx950\"", "\"status\":\"loaded\"",
                              "\"content_hash\":\"aaaa\"", "\"groups\":10"});
        expectLine(lines[1], {"\"arch\":\"gfx942\"", "\"status\":\"catalog\"", "holds gfx950 knowledge"});
        expectLine(lines[2], {"\"arch\":\"gfx1250\"", "\"status\":\"catalog\"", "has schema 2, not 1"});
        expectLine(lines[3], {"\"arch\":\"gfx90a\"", "\"status\":\"catalog\"", "no file at "});

        write(file("gfx950"), groups, "eeee");
        require(hipblaslt_jit::makeTuningLibraryKnowledge(tree, true)->version() == "gfx950=eeee",
                "The version follows the content");
        fs::create_directories(dir / "flat");
        fs::copy_file(file("gfx950"),
                      dir / "flat" / file("gfx950").filename(),
                      fs::copy_options::overwrite_existing);
        require(hipblaslt_jit::makeTuningLibraryKnowledge(dir / "flat", false)->version()
                    == "gfx950=eeee",
                "A directory that holds the files");

        const auto missing = hipblaslt_jit::makeTuningLibraryKnowledge(dir / "absent", true);
        require(missing->id() == catalog->id() && missing->version() == catalog->version(),
                "Without files the catalog's id and version");
        queries(*missing, "gfx950");
        setEnvironment("HIPBLASLT_JIT_KNOWLEDGE", "none");
        const auto off = hipblaslt_jit::makeTuningLibraryKnowledge(tree, true);
        setEnvironment("HIPBLASLT_JIT_KNOWLEDGE", nullptr);
        require(off->id() == catalog->id() && off->version() == catalog->version(),
                "HIPBLASLT_JIT_KNOWLEDGE=none keeps the catalog's id and version");
        queries(*off, "gfx950");
        lines = loadLines(log);
        require(lines.size() == 6, "Two more records, not " + std::to_string(lines.size() - 4));
        expectLine(lines[4], {"\"status\":\"catalog\"", "no file at "});
        expectLine(lines[5], {"\"status\":\"catalog\"", "HIPBLASLT_JIT_KNOWLEDGE=none"});
    }

    // Reports the first-use decode of a database's largest block.
    int decode(const fs::path& path)
    {
        const auto          opened = std::chrono::steady_clock::now();
        knowledge::Database database(path);
        const double        header = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - opened)
                                  .count();
        const auto& groups  = database.groups();
        const auto  largest = std::max_element(
            groups.begin(), groups.end(), [](const auto& a, const auto& b) {
                return a.length < b.length;
            });
        require(largest != groups.end(), "The database has no groups");
        std::string error;
        const auto  start = std::chrono::steady_clock::now();
        require(database.load(largest - groups.begin(), error), error);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count();
        knowledge::Problem query{largest->coreKey, largest->features, {4096, 4096, 1, 4096}};
        knowledge::Device  device{0, std::nullopt, {}};
        const auto& branch = database.branches()[largest->branch];
        device.cuCount     = branch.cuCount;
        if(!branch.pciChipIds.empty())
            device.pciChipId = branch.pciChipIds.front();
        const auto   matchStart = std::chrono::steady_clock::now();
        const auto   match      = database.nearest(query, device, 8);
        const double matchMs    = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - matchStart)
                                   .count();
        std::cout << "decode " << path.filename().string() << ": header " << header
                  << " ms for " << groups.size() << " groups; largest " << largest->name
                  << " (branch " << largest->branch << ") " << largest->rows << " rows, "
                  << largest->sets << " sets, " << largest->length << " bytes compressed: "
                  << ms << " ms first use; nearest " << matchMs << " ms, " << match.seeds.size()
                  << " seeds" << std::endl;
        return match.seeds.empty();
    }

    std::string policyName(hipblaslt_jit::ExecutionPolicy::Strategy strategy)
    {
        using Strategy = hipblaslt_jit::ExecutionPolicy::Strategy;
        return strategy == Strategy::StreamK        ? "StreamK"
               : strategy == Strategy::DataParallel ? "DataParallel"
                                                    : "None";
    }

    std::string policyName(hipblaslt_jit::ExecutionPolicy::Assignment assignment)
    {
        using Assignment = hipblaslt_jit::ExecutionPolicy::Assignment;
        return assignment == Assignment::Hybrid             ? "Hybrid"
               : assignment == Assignment::DynamicWorkQueue ? "DynamicWorkQueue"
                                                            : "StaticGrid";
    }

    // Prints the seeds of a plain GEMM on a device without a PCI chip ID, one JSON
    // line each, for the compile-only routes of architectures this host lacks.
    int nearest(const fs::path& path, const std::string& core, char** sizes, int cuCount)
    {
        namespace json = hipblaslt_jit::json;
        knowledge::Database database(path);
        knowledge::Problem  query{core, {}, {}};
        for(size_t i = 0; i < 4; ++i)
            query.size[i] = std::stoull(sizes[i]);
        const auto match = database.nearest(query, {cuCount, std::nullopt, {}}, 8);
        for(const auto& seed : match.seeds)
        {
            json::Members parameters, asserts;
            for(const auto& parameter : seed.parameters)
                parameters.emplace_back(parameter.name, parameter.json);
            for(const auto& [name, value] : seed.asserts)
                asserts.emplace_back(name, json::literal(value));
            std::cout << json::object({
                {"group", json::quote(match.group)},
                {"macro_tile", json::array(seed.macroTile)},
                {"depth_u", json::literal(seed.depthU)},
                {"strategy", json::quote(policyName(seed.policy.strategy))},
                {"assignment", json::quote(policyName(seed.policy.assignment))},
                {"gsu", json::literal(seed.globalSplitU)},
                {"parameters", json::object(parameters)},
                {"asserts", json::object(asserts)},
                {"branch", json::literal(seed.branch)},
                {"source", json::quote(seed.source)},
                {"row", json::array(seed.row)},
                {"distance", json::literal(seed.distance)},
            }) << '\n';
        }
        return match.seeds.empty();
    }
}

int main(int argc, char** argv)
{
    try
    {
        if(argc == 3 && std::string(argv[1]) == "--decode")
            return decode(argv[2]);
        if(argc == 9 && std::string(argv[1]) == "--nearest")
            return nearest(argv[2], argv[3], argv + 4, std::stoi(argv[8]));
        if(argc != 2)
        {
            std::cerr << "Usage: " << argv[0] << " FRESH_OUTPUT_DIRECTORY\n"
                      << "       " << argv[0] << " --decode KNOWLEDGE_FILE\n"
                      << "       " << argv[0]
                      << " --nearest KNOWLEDGE_FILE CORE_KEY M N BATCH K CU_COUNT\n";
            return 2;
        }
        const fs::path dir = argv[1];
        fs::create_directories(dir);
        const auto log = dir / "debug.log";
        fs::remove(log);
        setEnvironment("HIPBLASLT_JIT", "1");
        setEnvironment("HIPBLASLT_JIT_DEBUG", "knowledge");
        setEnvironment("HIPBLASLT_JIT_DEBUG_FILE", log.string().c_str());
        setEnvironment("HIPBLASLT_JIT_KNOWLEDGE", nullptr);
        write(dir / "fixture.dat.zlib", fixture());
        testOrder(dir / "fixture.dat.zlib");
        testProblemTypes(dir / "fixture.dat.zlib");
        testBranches(dir / "fixture.dat.zlib");
        testLazy(dir / "fixture.dat.zlib");
        testHeaders(dir);
        testLibrary(dir, log);
        std::cout << "PASS jit-knowledge" << std::endl;
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "FAIL jit-knowledge: " << e.what() << std::endl;
        return 1;
    }
}
