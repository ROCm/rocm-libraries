// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-knowledge.hpp"
#include "hipblaslt-jit-json.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string_view>
#include <zlib.h>
#ifdef TENSILE_MSGPACK
#include <msgpack.hpp>
#endif

namespace hipblaslt_jit::knowledge
{
    namespace
    {
        constexpr char     magic[4]   = {'H', 'J', 'K', 'N'};
        constexpr uint32_t schema     = 1;
        constexpr size_t   preamble   = 12;
        constexpr size_t   perShape   = 2;
        constexpr auto     generator  = "tensilelite-logic-knowledge";
        constexpr auto     infinity   = std::numeric_limits<double>::infinity();

        [[noreturn]] void fail(const std::string& message)
        {
            throw std::runtime_error(message);
        }

        std::vector<uint8_t>
            readAt(const std::filesystem::path& path, uint64_t offset, uint64_t length)
        {
            std::ifstream in(path, std::ios::binary);
            if(!in)
                fail("cannot open " + path.string());
            std::vector<uint8_t> bytes(length);
            in.seekg(static_cast<std::streamoff>(offset));
            if(!in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(length)))
                fail("cannot read " + std::to_string(length) + " bytes at offset "
                     + std::to_string(offset));
            return bytes;
        }

        std::vector<uint8_t> inflateBlock(const std::vector<uint8_t>& compressed)
        {
            z_stream stream{};
            if(inflateInit(&stream) != Z_OK)
                fail("zlib cannot start");
            std::vector<uint8_t> out(compressed.size() * 4 + 1024);
            stream.next_in  = const_cast<Bytef*>(compressed.data());
            stream.avail_in = static_cast<uInt>(compressed.size());
            int status      = Z_OK;
            while(status == Z_OK)
            {
                if(stream.total_out == out.size())
                    out.resize(out.size() * 2);
                stream.next_out  = out.data() + stream.total_out;
                stream.avail_out = static_cast<uInt>(out.size() - stream.total_out);
                status           = inflate(&stream, Z_NO_FLUSH);
            }
            const auto total = stream.total_out;
            const auto rest  = stream.avail_in;
            inflateEnd(&stream);
            if(status != Z_STREAM_END || rest)
                fail("the block is not one complete zlib stream");
            out.resize(total);
            return out;
        }

        // 1 for an exact match, 2 for a fallback chip, 0 for none.
        int matches(const Branch& branch, const Device& device)
        {
            const auto has = [&](int id) {
                return std::find(branch.pciChipIds.begin(), branch.pciChipIds.end(), id)
                       != branch.pciChipIds.end();
            };
            if(branch.cuCount && branch.cuCount != device.cuCount)
                return 0;
            if(branch.pciChipIds.empty())
                return 1;
            if(!device.pciChipId)
                return 0;
            if(has(*device.pciChipId))
                return 1;
            return std::any_of(device.fallbackChipIds.begin(), device.fallbackChipIds.end(), has)
                       ? 2
                       : 0;
        }

        // How many features the group has beyond the request's, if it covers them.
        std::optional<size_t> extras(const Features& group, const Features& request)
        {
            size_t extra = 0;
            for(int type : request.bias)
                if(std::find(group.bias.begin(), group.bias.end(), type) == group.bias.end())
                    return std::nullopt;
            extra += request.bias.empty() && !group.bias.empty();
            if(request.activation && !group.activation)
                return std::nullopt;
            extra += group.activation && !request.activation;
            if(!request.scaleAB.empty() && group.scaleAB != request.scaleAB)
                return std::nullopt;
            extra += request.scaleAB.empty() && !group.scaleAB.empty();
            for(const auto& flag : request.flags)
                if(!group.flags.count(flag))
                    return std::nullopt;
            for(const auto& flag : group.flags)
                extra += !request.flags.count(flag);
            return extra;
        }

        struct Set
        {
            std::array<size_t, 2>                        macroTile{}, waves{};
            std::array<size_t, 4>                        instruction{};
            size_t                                       depthU = 0;
            ExecutionPolicy                              policy;
            int64_t                                      globalSplitU = 0;
            std::vector<uint32_t>                        parameters; // dictionary ids
            std::vector<std::pair<std::string, int64_t>> asserts;
            std::string                                  source;
        };

        struct Row
        {
            std::array<float, 4>    log{};
            std::array<uint32_t, 4> size{};
            uint32_t                set = 0;
        };

        struct Block
        {
            std::vector<TuningParameter> dictionary;
            std::vector<Set>             sets;
            std::vector<Row>             rows;
        };

        std::string label(const Group& group)
        {
            return group.name + " (branch " + std::to_string(group.branch) + ")";
        }

#ifdef TENSILE_MSGPACK
        ExecutionPolicy::Strategy strategy(std::string_view name)
        {
            using S = ExecutionPolicy::Strategy;
            if(name == "None")
                return S::None;
            if(name == "DataParallel")
                return S::DataParallel;
            if(name == "StreamK")
                return S::StreamK;
            fail("unknown TileProcessingStrategy " + std::string(name));
        }

        ExecutionPolicy::Assignment assignment(std::string_view name)
        {
            using A = ExecutionPolicy::Assignment;
            if(name == "StaticGrid")
                return A::StaticGrid;
            if(name == "DynamicWorkQueue")
                return A::DynamicWorkQueue;
            if(name == "Hybrid")
                return A::Hybrid;
            fail("unknown WorkAssignment " + std::string(name));
        }

        using Object = msgpack::object;

        const Object* find(const Object& object, std::string_view name)
        {
            if(object.type != msgpack::type::MAP)
                return nullptr;
            for(uint32_t i = 0; i < object.via.map.size; ++i)
            {
                const auto& key = object.via.map.ptr[i].key;
                if(key.type == msgpack::type::STR
                   && std::string_view(key.via.str.ptr, key.via.str.size) == name)
                    return &object.via.map.ptr[i].val;
            }
            return nullptr;
        }

        const Object& member(const Object& object, std::string_view name)
        {
            if(const auto* value = find(object, name))
                return *value;
            fail("no \"" + std::string(name) + "\" field");
        }

        std::string text(const Object& object)
        {
            if(object.type != msgpack::type::STR)
                fail("a string field holds another type");
            return {object.via.str.ptr, object.via.str.size};
        }

        int64_t integer(const Object& object)
        {
            if(object.type == msgpack::type::POSITIVE_INTEGER
               && object.via.u64 <= uint64_t(std::numeric_limits<int64_t>::max()))
                return static_cast<int64_t>(object.via.u64);
            if(object.type == msgpack::type::NEGATIVE_INTEGER)
                return object.via.i64;
            if(object.type == msgpack::type::NIL)
                return 0;
            fail("an integer field holds another type");
        }

        size_t count(const Object& object)
        {
            const auto value = integer(object);
            if(value < 0)
                fail("a count is negative");
            return static_cast<size_t>(value);
        }

        const msgpack::object_array& array(const Object& object, size_t size = 0)
        {
            if(object.type != msgpack::type::ARRAY || (size && object.via.array.size != size))
                fail("an array field has another type or length");
            return object.via.array;
        }

        template <size_t N>
        std::array<size_t, N> sizes(const Object& object)
        {
            const auto&           values = array(object, N);
            std::array<size_t, N> result{};
            for(size_t i = 0; i < N; ++i)
                result[i] = count(values.ptr[i]);
            return result;
        }

        std::string toJson(const Object& object)
        {
            switch(object.type)
            {
            case msgpack::type::NIL:
                return "null";
            case msgpack::type::BOOLEAN:
                return object.via.boolean ? "true" : "false";
            case msgpack::type::POSITIVE_INTEGER:
                return std::to_string(object.via.u64);
            case msgpack::type::NEGATIVE_INTEGER:
                return std::to_string(object.via.i64);
            case msgpack::type::FLOAT32:
            case msgpack::type::FLOAT64:
            {
                if(!std::isfinite(object.via.f64))
                    fail("a parameter value is not finite");
                auto value = json::literal(object.via.f64);
                if(value.find_first_of(".e") == std::string::npos)
                    value += ".0";
                return value;
            }
            case msgpack::type::STR:
                return json::quote(text(object));
            case msgpack::type::ARRAY:
            {
                std::string out = "[";
                for(uint32_t i = 0; i < object.via.array.size; ++i)
                    out += (i ? "," : "") + toJson(object.via.array.ptr[i]);
                return out + "]";
            }
            case msgpack::type::MAP:
            {
                std::string out = "{";
                for(uint32_t i = 0; i < object.via.map.size; ++i)
                    out += (i ? "," : "") + json::quote(text(object.via.map.ptr[i].key)) + ":"
                           + toJson(object.via.map.ptr[i].val);
                return out + "}";
            }
            default:
                fail("a parameter value has no JSON form");
            }
        }

        Features features(const Object& object)
        {
            Features result;
            if(object.type != msgpack::type::MAP)
                fail("the features are not a map");
            for(uint32_t i = 0; i < object.via.map.size; ++i)
            {
                const auto  name  = text(object.via.map.ptr[i].key);
                const auto& value = object.via.map.ptr[i].val;
                if(name == "Bias")
                {
                    const auto& types = array(value);
                    for(uint32_t t = 0; t < types.size; ++t)
                        result.bias.push_back(static_cast<int>(integer(types.ptr[t])));
                }
                else if(name == "Activation")
                    result.activation = true;
                else if(name == "UseScaleAB")
                    result.scaleAB = text(value);
                else
                    result.flags.insert(name);
            }
            return result;
        }

        msgpack::object_handle parse(const std::vector<uint8_t>& bytes)
        {
            size_t offset = 0;
            auto   handle
                = msgpack::unpack(reinterpret_cast<const char*>(bytes.data()), bytes.size(), offset);
            if(offset != bytes.size())
                fail("trailing data after the MessagePack object");
            return handle;
        }

        std::unique_ptr<const Block> decode(const std::filesystem::path& path,
                                            uint64_t                     start,
                                            const Group&                 group)
        {
            const auto handle = parse(inflateBlock(readAt(path, start + group.offset, group.length)));
            const auto& root  = handle.get();
            auto        block = std::make_unique<Block>();

            const auto& dictionary = array(member(root, "param_dictionary"));
            for(uint32_t i = 0; i < dictionary.size; ++i)
            {
                const auto& entry = array(dictionary.ptr[i], 2);
                block->dictionary.push_back({text(entry.ptr[0]), toJson(entry.ptr[1])});
            }

            const auto& sets = array(member(root, "sets"));
            for(uint32_t i = 0; i < sets.size; ++i)
            {
                const auto& object = sets.ptr[i];
                Set         set;
                set.macroTile    = sizes<2>(member(object, "macro_tile"));
                set.waves        = sizes<2>(member(object, "waves"));
                set.instruction  = sizes<4>(member(object, "instruction"));
                set.depthU       = count(member(object, "depth_u"));
                set.globalSplitU = integer(member(object, "gsu"));
                const auto& policy = member(object, "policy");
                set.policy.strategy   = strategy(text(member(policy, "strategy")));
                set.policy.assignment = assignment(text(member(policy, "assignment")));
                const auto& ids       = array(member(object, "params"));
                for(uint32_t p = 0; p < ids.size; ++p)
                {
                    const auto id = count(ids.ptr[p]);
                    if(id >= block->dictionary.size())
                        fail("a set names a parameter outside the dictionary");
                    set.parameters.push_back(static_cast<uint32_t>(id));
                }
                const auto& asserts = member(object, "asserts");
                if(asserts.type != msgpack::type::MAP)
                    fail("the asserts are not a map");
                for(uint32_t a = 0; a < asserts.via.map.size; ++a)
                    set.asserts.emplace_back(text(asserts.via.map.ptr[a].key),
                                             integer(asserts.via.map.ptr[a].val));
                const auto& source = member(object, "source");
                set.source = text(member(source, "file")) + "#"
                             + std::to_string(integer(member(source, "index")));
                if(!set.macroTile[0] || !set.macroTile[1])
                    fail("a set has an empty macro tile");
                block->sets.push_back(std::move(set));
            }

            const auto& rows = array(member(root, "rows"));
            block->rows.reserve(rows.size);
            for(uint32_t i = 0; i < rows.size; ++i)
            {
                const auto& values = array(rows.ptr[i], 6);
                Row         row;
                for(size_t d = 0; d < 4; ++d)
                {
                    const auto size = count(values.ptr[d]);
                    if(size > std::numeric_limits<uint32_t>::max())
                        fail("a row size is out of range");
                    row.size[d] = static_cast<uint32_t>(size);
                    row.log[d]  = static_cast<float>(std::log2(std::max<double>(1, size)));
                }
                const auto set = count(values.ptr[4]);
                if(set >= block->sets.size())
                    fail("a row names a set outside the block");
                row.set = static_cast<uint32_t>(set);
                block->rows.push_back(row);
            }
            if(block->rows.size() != group.rows || block->sets.size() != group.sets)
                fail("the block does not hold the rows and sets its index entry counts");
            return block;
        }
#endif

        std::vector<Seed> select(const Block& block,
                                 const Problem& problem,
                                 size_t         branch,
                                 size_t         wanted)
        {
            std::array<double, 4> query{};
            for(size_t d = 0; d < 4; ++d)
                query[d] = static_cast<float>(std::log2(std::max<double>(1, problem.size[d])));
            std::vector<double>   best(block.sets.size(), infinity);
            std::vector<uint32_t> nearest(block.sets.size());
            for(size_t r = 0; r < block.rows.size(); ++r)
            {
                const auto& row      = block.rows[r];
                double      distance = 0;
                for(size_t d = 0; d < 4; ++d)
                    distance += (row.log[d] - query[d]) * (row.log[d] - query[d]);
                if(distance < best[row.set])
                {
                    best[row.set]    = distance;
                    nearest[row.set] = static_cast<uint32_t>(r);
                }
            }
            // The fraction of the macro tiles' area that covers the output.
            const double m = std::max<size_t>(1, problem.size[0]);
            const double n = std::max<size_t>(1, problem.size[1]);
            std::vector<double>   waste(block.sets.size());
            std::vector<uint32_t> order;
            for(uint32_t s = 0; s < block.sets.size(); ++s)
            {
                if(best[s] == infinity)
                    continue;
                const double tm = block.sets[s].macroTile[0], tn = block.sets[s].macroTile[1];
                waste[s] = std::ceil(m / tm) * tm * std::ceil(n / tn) * tn / (m * n);
                order.push_back(s);
            }
            std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                return std::tie(best[a], waste[a], a) < std::tie(best[b], waste[b], b);
            });

            std::vector<Seed>                        seeds;
            std::map<std::array<size_t, 4>, size_t> shapes;
            for(const auto s : order)
            {
                if(seeds.size() == wanted)
                    break;
                const auto& set = block.sets[s];
                if(++shapes[{set.macroTile[0], set.macroTile[1], set.waves[0], set.waves[1]}]
                   > perShape)
                    continue;
                Seed seed;
                seed.macroTile    = set.macroTile;
                seed.waves        = set.waves;
                seed.instruction  = set.instruction;
                seed.depthU       = set.depthU;
                seed.policy       = set.policy;
                seed.globalSplitU = set.globalSplitU;
                for(const auto id : set.parameters)
                    seed.parameters.push_back(block.dictionary[id]);
                seed.asserts = set.asserts;
                seed.branch  = branch;
                seed.source  = set.source;
                const auto& row = block.rows[nearest[s]];
                seed.row        = {row.size[0], row.size[1], row.size[2], row.size[3]};
                seed.distance   = std::sqrt(best[s]);
                seeds.push_back(std::move(seed));
            }
            // Seeds tie when they are as near and waste as much of their tiles.
            for(size_t i = 1; i < seeds.size(); ++i)
            {
                const auto& a = seeds[i - 1];
                const auto& b = seeds[i];
                const double aw = std::ceil(m / a.macroTile[0]) * a.macroTile[0]
                                  * std::ceil(n / a.macroTile[1]) * a.macroTile[1];
                const double bw = std::ceil(m / b.macroTile[0]) * b.macroTile[0]
                                  * std::ceil(n / b.macroTile[1]) * b.macroTile[1];
                seeds[i].rank = a.rank + (a.distance != b.distance || aw != bw);
            }
            return seeds;
        }
    }

    struct Database::State
    {
        struct Slot
        {
            std::once_flag               once;
            std::unique_ptr<const Block> block;
            std::string                  error;
            std::atomic<bool>            ready{false};
        };
        explicit State(size_t groups)
            : slots(groups)
        {
        }
        std::vector<Slot> slots;

        // The group's block, or nullptr. fresh gets the error only in the call that decoded it.
        const Block* get(const Database& database, size_t group, std::string* fresh)
        {
            auto& slot = slots.at(group);
            std::call_once(slot.once, [&] {
#ifdef TENSILE_MSGPACK
                try
                {
                    slot.block = decode(database.m_path, database.m_blocks, database.m_groups[group]);
                    slot.ready = true;
                }
                catch(const std::bad_alloc&)
                {
                    throw;
                }
                catch(const std::exception& e)
                {
                    slot.error = e.what();
                }
#else
                slot.error = "hipBLASLt was built without MessagePack support";
#endif
                if(fresh)
                    *fresh = slot.error;
            });
            return slot.block.get();
        }
    };

    Database::Database(std::filesystem::path path)
        : m_path(std::move(path))
    {
#ifndef TENSILE_MSGPACK
        fail("hipBLASLt was built without MessagePack support");
#else
        std::error_code error;
        if(!std::filesystem::is_regular_file(m_path, error))
            fail("no file at " + m_path.string());
        std::ifstream in(m_path, std::ios::binary);
        char          head[preamble];
        if(!in || !in.read(head, preamble))
            fail("cannot read the preamble of " + m_path.string());
        if(std::memcmp(head, magic, sizeof(magic)))
            fail(m_path.string() + " is not a JIT knowledge file");
        const auto word = [&](size_t at) {
            const auto* p = reinterpret_cast<const unsigned char*>(head + at);
            return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16
                   | uint32_t(p[3]) << 24;
        };
        if(word(4) != schema)
            fail(m_path.string() + " has schema " + std::to_string(word(4)) + ", not "
                 + std::to_string(schema));
        try
        {
            const auto  handle = parse(readAt(m_path, preamble, word(8)));
            const auto& header = handle.get();
            if(text(member(header, "generator")) != generator)
                fail("another generator wrote it");
            m_arch        = text(member(header, "arch"));
            m_libraryArch = text(member(header, "library_arch"));
            m_contentHash = text(member(header, "content_hash"));
            const auto& branches = array(member(header, "branches"));
            for(uint32_t i = 0; i < branches.size; ++i)
            {
                Branch branch;
                branch.kind      = text(member(branches.ptr[i], "kind"));
                branch.cuCount   = static_cast<int>(integer(member(branches.ptr[i], "cu_count")));
                const auto& ids = array(member(branches.ptr[i], "pci_ids"));
                for(uint32_t p = 0; p < ids.size; ++p)
                    branch.pciChipIds.push_back(static_cast<int>(integer(ids.ptr[p])));
                m_branches.push_back(std::move(branch));
            }
            const auto& index = array(member(header, "index"));
            for(uint32_t i = 0; i < index.size; ++i)
            {
                const auto& entry = index.ptr[i];
                Group       group;
                group.branch = count(member(entry, "branch"));
                if(group.branch >= m_branches.size())
                    fail("an index entry names an unknown branch");
                const auto& type = member(entry, "problem_type");
                group.name       = text(member(type, "name"));
                group.features   = features(member(type, "features"));
                group.coreKey    = text(member(entry, "core_key"));
                group.offset     = count(member(entry, "offset"));
                group.length     = count(member(entry, "length"));
                group.rows       = count(member(entry, "rows"));
                group.sets       = count(member(entry, "sets"));
                m_groups.push_back(std::move(group));
            }
        }
        catch(const std::runtime_error& e)
        {
            fail(m_path.string() + " has a malformed header: " + e.what());
        }
        m_blocks = preamble + word(8);
        m_state  = std::make_unique<State>(m_groups.size());
#endif
    }

    Database::~Database() = default;

    const std::filesystem::path& Database::path() const noexcept
    {
        return m_path;
    }
    const std::string& Database::arch() const noexcept
    {
        return m_arch;
    }
    const std::string& Database::libraryArch() const noexcept
    {
        return m_libraryArch;
    }
    const std::string& Database::contentHash() const noexcept
    {
        return m_contentHash;
    }
    const std::vector<Branch>& Database::branches() const noexcept
    {
        return m_branches;
    }
    const std::vector<Group>& Database::groups() const noexcept
    {
        return m_groups;
    }

    bool Database::load(size_t group, std::string& error) const
    {
        const auto* block = m_state->get(*this, group, nullptr);
        error             = block ? "" : m_state->slots[group].error;
        return block;
    }

    bool Database::loaded(size_t group) const
    {
        return m_state->slots.at(group).ready;
    }

    Match Database::nearest(const Problem& problem, const Device& device, size_t count) const
    {
        Match match;
        // Exact branches first, then fallback chips, as the Tensile library selects rows.
        for(const int pass : {1, 2})
            for(size_t b = 0; b < m_branches.size(); ++b)
            {
                if(matches(m_branches[b], device) != pass)
                    continue;
                std::vector<std::pair<size_t, size_t>> covering; // {extras, group}
                for(size_t g = 0; g < m_groups.size(); ++g)
                    if(m_groups[g].branch == b && m_groups[g].coreKey == problem.coreKey)
                        if(const auto extra = extras(m_groups[g].features, problem.features))
                            covering.emplace_back(*extra, g);
                std::sort(covering.begin(), covering.end());
                for(const auto& [extra, g] : covering)
                {
                    std::string fresh;
                    const auto* block = m_state->get(*this, g, &fresh);
                    if(!fresh.empty())
                        match.corrupt = label(m_groups[g]) + ": " + fresh;
                    if(!block)
                        continue;
                    match.group = label(m_groups[g]);
                    match.seeds = select(*block, problem, b, count);
                    return match;
                }
            }
        return match;
    }
}
