// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-msgpack.hpp"
#ifdef TENSILE_MSGPACK
#include <msgpack.hpp>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#endif

namespace hipblaslt_jit::msgpack_io
{
#ifdef TENSILE_MSGPACK
    namespace
    {
        using Packer = msgpack::packer<msgpack::sbuffer>;

        [[noreturn]] void malformed(const std::string& message)
        {
            throw std::runtime_error(message);
        }

        template <class F>
        Status guarded(const char* what, F&& f)
        {
            try
            {
                f();
                return {};
            }
            catch(const std::bad_alloc&)
            {
                throw;
            }
            catch(const std::exception& e)
            {
                return {Status::Code::Failed, Stage::Configure, std::string(what) + ": " + e.what()};
            }
        }

        msgpack::object_handle parse(const std::vector<uint8_t>& bytes, const char* what)
        {
            size_t offset = 0;
            auto   handle
                = msgpack::unpack(reinterpret_cast<const char*>(bytes.data()), bytes.size(), offset);
            if(offset != bytes.size())
                malformed(std::string(what) + " has trailing data");
            return handle;
        }

        std::vector<uint8_t> contents(const msgpack::sbuffer& buffer)
        {
            const auto* data = reinterpret_cast<const uint8_t*>(buffer.data());
            return {data, data + buffer.size()};
        }

        std::string_view text(const msgpack::object& object, const char* what)
        {
            if(object.type != msgpack::type::STR)
                malformed(std::string(what) + " is not a string");
            return {object.via.str.ptr, object.via.str.size};
        }

        bool named(const msgpack::object& key, std::string_view name)
        {
            return key.type == msgpack::type::STR
                   && std::string_view(key.via.str.ptr, key.via.str.size) == name;
        }

        const msgpack::object_array& array(const msgpack::object& object, const char* what)
        {
            if(object.type != msgpack::type::ARRAY)
                malformed(std::string(what) + " is not an array");
            return object.via.array;
        }

        const msgpack::object* find(const msgpack::object& object, std::string_view name)
        {
            if(object.type != msgpack::type::MAP)
                return nullptr;
            for(uint32_t i = 0; i < object.via.map.size; ++i)
                if(named(object.via.map.ptr[i].key, name))
                    return &object.via.map.ptr[i].val;
            return nullptr;
        }

        const msgpack::object& member(const msgpack::object& object, std::string_view name, const char* what)
        {
            if(object.type != msgpack::type::MAP)
                malformed(std::string(what) + " is not a map");
            if(const auto* value = find(object, name))
                return *value;
            malformed(std::string(what) + " has no \"" + std::string(name) + "\"");
        }

        void string(Packer& packer, std::string_view value)
        {
            packer.pack_str(static_cast<uint32_t>(value.size()));
            packer.pack_str_body(value.data(), static_cast<uint32_t>(value.size()));
        }

        // Copies a map, letting field write each value.
        template <class Field>
        void copyMap(Packer& packer, const msgpack::object& object, Field&& field)
        {
            packer.pack_map(object.via.map.size);
            for(uint32_t i = 0; i < object.via.map.size; ++i)
            {
                packer.pack(object.via.map.ptr[i].key);
                field(object.via.map.ptr[i]);
            }
        }

        struct Entry
        {
            const msgpack::object* solution;
            const msgpack::object* row;
        };

        // The legacy one-solution shape Tensile.SingleSolution writes.
        Entry entryParts(const msgpack::object& root)
        {
            const auto& solutions = array(member(root, "solutions", "The entry"), "Its solutions");
            if(solutions.size != 1)
                malformed("The entry does not hold exactly one solution");
            member(solutions.ptr[0], "index", "Its solution");
            const auto& library = member(root, "library", "The entry");
            if(text(member(library, "type", "Its library"), "Its library type") != "Problem")
                malformed("The entry library is not a Problem library");
            const auto& rows = array(member(library, "rows", "Its library"), "Its rows");
            if(rows.size != 1)
                malformed("The entry library does not have exactly one row");
            const auto& single = member(rows.ptr[0], "library", "Its row");
            if(text(member(single, "type", "Its row library"), "Its row library type") != "Single")
                malformed("The entry row does not select a Single library");
            member(single, "index", "Its Single library");
            member(rows.ptr[0], "predicate", "Its row");
            return {&solutions.ptr[0], &rows.ptr[0]};
        }

        void packRow(Packer& packer, msgpack::sbuffer& buffer, const MasterRow& row)
        {
            if(row.prefix.empty())
                malformed("A master row has no prefix");
            // The predicate is spliced in as encoded bytes, so it must be one object.
            parse(row.predicate, "A master row predicate");
            packer.pack_map(2);
            string(packer, "predicate");
            packer.pack_map(2);
            string(packer, "type");
            string(packer, "And");
            string(packer, "value");
            packer.pack_array(static_cast<uint32_t>(row.sizes.size() + 1));
            for(size_t i = 0; i < row.sizes.size(); ++i)
            {
                packer.pack_map(3);
                string(packer, "type");
                string(packer, "SizeEqual");
                string(packer, "index");
                packer.pack(static_cast<uint64_t>(i));
                string(packer, "value");
                packer.pack(static_cast<uint64_t>(row.sizes[i]));
            }
            buffer.write(reinterpret_cast<const char*>(row.predicate.data()), row.predicate.size());
            string(packer, "library");
            packer.pack_map(2);
            string(packer, "type");
            string(packer, "Placeholder");
            string(packer, "value");
            string(packer, row.prefix);
        }
    }

    Status rewriteEntryIndex(const std::vector<uint8_t>& entry,
                             int32_t                     index,
                             std::vector<uint8_t>&       rewritten)
    {
        rewritten.clear();
        return guarded("Invalid solution library entry", [&] {
            const auto  handle = parse(entry, "The entry");
            const auto& root   = handle.get();
            entryParts(root);
            msgpack::sbuffer buffer;
            Packer           packer(buffer);
            const auto       withIndex = [&](const msgpack::object& object) {
                copyMap(packer, object, [&](const msgpack::object_kv& field) {
                    if(named(field.key, "index"))
                        packer.pack(index);
                    else
                        packer.pack(field.val);
                });
            };
            copyMap(packer, root, [&](const msgpack::object_kv& field) {
                if(named(field.key, "solutions"))
                {
                    packer.pack_array(1);
                    withIndex(field.val.via.array.ptr[0]);
                }
                else if(named(field.key, "library"))
                    copyMap(packer, field.val, [&](const msgpack::object_kv& libraryField) {
                        if(!named(libraryField.key, "rows"))
                        {
                            packer.pack(libraryField.val);
                            return;
                        }
                        packer.pack_array(1);
                        copyMap(packer,
                                libraryField.val.via.array.ptr[0],
                                [&](const msgpack::object_kv& rowField) {
                                    if(named(rowField.key, "library"))
                                        withIndex(rowField.val);
                                    else
                                        packer.pack(rowField.val);
                                });
                    });
                else
                    packer.pack(field.val);
            });
            rewritten = contents(buffer);
        });
    }

    Status readEntry(const std::vector<uint8_t>& entry, EntryFields& fields)
    {
        fields = {};
        return guarded("Invalid solution library entry", [&] {
            const auto  handle = parse(entry, "The entry");
            const auto  parts  = entryParts(handle.get());
            EntryFields result;
            result.index      = member(*parts.solution, "index", "Its solution").as<int64_t>();
            result.kernelName = std::string(
                text(member(*parts.solution, "kernelName", "Its solution"), "Its kernel name"));
            msgpack::sbuffer buffer;
            msgpack::pack(buffer, member(*parts.row, "predicate", "Its row"));
            result.predicate = contents(buffer);
            fields           = std::move(result);
        });
    }

    Status appendEntryPredicates(const std::vector<uint8_t>&          entry,
                                 const std::vector<IndexedPredicate>& predicates,
                                 std::vector<uint8_t>&                appended)
    {
        appended.clear();
        return guarded("Invalid solution library entry", [&] {
            const auto  handle = parse(entry, "The entry");
            const auto& root   = handle.get();
            const auto  parts  = entryParts(root);
            for(const auto* predicate : {&member(*parts.solution, "problemPredicate", "Its solution"),
                                         &member(*parts.row, "predicate", "Its row")})
            {
                if(text(member(*predicate, "type", "A predicate"), "A predicate type") != "And")
                    malformed("An entry predicate is not an And");
                array(member(*predicate, "value", "An And predicate"), "Its terms");
            }
            msgpack::sbuffer buffer;
            Packer           packer(buffer);
            const auto       withPredicates = [&](const msgpack::object& predicate) {
                copyMap(packer, predicate, [&](const msgpack::object_kv& field) {
                    if(!named(field.key, "value"))
                    {
                        packer.pack(field.val);
                        return;
                    }
                    const auto& terms = field.val.via.array;
                    packer.pack_array(static_cast<uint32_t>(terms.size + predicates.size()));
                    for(uint32_t i = 0; i < terms.size; ++i)
                        packer.pack(terms.ptr[i]);
                    for(const auto& added : predicates)
                    {
                        packer.pack_map(3);
                        string(packer, "type");
                        string(packer, added.type);
                        string(packer, "index");
                        packer.pack(added.index);
                        string(packer, "value");
                        packer.pack(added.value);
                    }
                });
            };
            // Copies a map, passing the value of field name to inner.
            const auto copyWith = [&](const msgpack::object& object, const char* name, auto&& inner) {
                copyMap(packer, object, [&](const msgpack::object_kv& field) {
                    if(named(field.key, name))
                        inner(field.val);
                    else
                        packer.pack(field.val);
                });
            };
            const auto withRows = [&](const msgpack::object& rows) {
                packer.pack_array(1);
                copyWith(rows.via.array.ptr[0], "predicate", withPredicates);
            };
            copyMap(packer, root, [&](const msgpack::object_kv& field) {
                if(named(field.key, "solutions"))
                {
                    packer.pack_array(1);
                    copyWith(field.val.via.array.ptr[0], "problemPredicate", withPredicates);
                }
                else if(named(field.key, "library"))
                    copyWith(field.val, "rows", withRows);
                else
                    packer.pack(field.val);
            });
            appended = contents(buffer);
        });
    }

    Status appendMasterRows(const std::vector<uint8_t>&   master,
                            const std::vector<MasterRow>& rows,
                            std::vector<uint8_t>&         appended)
    {
        appended.clear();
        return guarded("Invalid JIT master library", [&] {
            msgpack::sbuffer buffer;
            Packer           packer(buffer);
            const auto       packRows = [&](const msgpack::object_array* existing) {
                packer.pack_array(static_cast<uint32_t>((existing ? existing->size : 0) + rows.size()));
                for(uint32_t i = 0; existing && i < existing->size; ++i)
                    packer.pack(existing->ptr[i]);
                for(const auto& row : rows)
                    packRow(packer, buffer, row);
            };
            if(master.empty())
            {
                packer.pack_map(2);
                string(packer, "solutions");
                packer.pack_array(0);
                string(packer, "library");
                packer.pack_map(2);
                string(packer, "type");
                string(packer, "Problem");
                string(packer, "rows");
                packRows(nullptr);
            }
            else
            {
                const auto  handle  = parse(master, "The master");
                const auto& root    = handle.get();
                const auto& library = member(root, "library", "The master");
                if(text(member(library, "type", "Its library"), "Its library type") != "Problem")
                    malformed("The master library is not a Problem library");
                array(member(library, "rows", "Its library"), "Its rows");
                copyMap(packer, root, [&](const msgpack::object_kv& field) {
                    if(!named(field.key, "library"))
                    {
                        packer.pack(field.val);
                        return;
                    }
                    copyMap(packer, field.val, [&](const msgpack::object_kv& libraryField) {
                        if(named(libraryField.key, "rows"))
                            packRows(&libraryField.val.via.array);
                        else
                            packer.pack(libraryField.val);
                    });
                });
            }
            appended = contents(buffer);
        });
    }

    Status readMasterPrefixes(const std::vector<uint8_t>& master, std::vector<std::string>& prefixes)
    {
        prefixes.clear();
        return guarded("Invalid JIT master library", [&] {
            const auto  handle  = parse(master, "The master");
            const auto& library = member(handle.get(), "library", "The master");
            const auto& rows    = array(member(library, "rows", "Its library"), "Its rows");
            std::vector<std::string> result;
            for(uint32_t i = 0; i < rows.size; ++i)
            {
                const auto& selected = member(rows.ptr[i], "library", "A row");
                if(text(member(selected, "type", "A row library"), "A row library type")
                   == "Placeholder")
                    result.emplace_back(
                        text(member(selected, "value", "A placeholder"), "A placeholder value"));
            }
            prefixes = std::move(result);
        });
    }

    Status writeMapping(const std::map<int32_t, std::string>& mapping, std::vector<uint8_t>& bytes)
    {
        bytes.clear();
        return guarded("Invalid JIT library mapping", [&] {
            msgpack::sbuffer buffer;
            Packer           packer(buffer);
            packer.pack_map(static_cast<uint32_t>(mapping.size()));
            for(const auto& [index, prefix] : mapping)
            {
                if(index < 0)
                    malformed("negative index " + std::to_string(index));
                packer.pack(index);
                string(packer, prefix);
            }
            bytes = contents(buffer);
        });
    }

    Status readMapping(const std::vector<uint8_t>& bytes, std::map<int32_t, std::string>& mapping)
    {
        mapping.clear();
        return guarded("Invalid JIT library mapping", [&] {
            const auto  handle = parse(bytes, "The mapping");
            const auto& root   = handle.get();
            if(root.type != msgpack::type::MAP)
                malformed("The mapping is not a map");
            std::map<int32_t, std::string> result;
            for(uint32_t i = 0; i < root.via.map.size; ++i)
            {
                const auto& field = root.via.map.ptr[i];
                int64_t     index = field.key.type == msgpack::type::STR
                                        ? std::stoll(std::string(text(field.key, "A key")))
                                        : field.key.as<int64_t>();
                if(index < 0 || index > INT32_MAX)
                    malformed("index " + std::to_string(index) + " is out of range");
                if(!result.emplace(static_cast<int32_t>(index), text(field.val, "A prefix")).second)
                    malformed("index " + std::to_string(index) + " appears twice");
            }
            mapping = std::move(result);
        });
    }

    Status writeAllocator(int64_t next, std::vector<uint8_t>& bytes)
    {
        bytes.clear();
        return guarded("Invalid JIT index allocator", [&] {
            msgpack::sbuffer buffer;
            Packer           packer(buffer);
            packer.pack_map(1);
            string(packer, "next");
            packer.pack(next);
            bytes = contents(buffer);
        });
    }

    Status readAllocator(const std::vector<uint8_t>& bytes, int64_t& next)
    {
        return guarded("Invalid JIT index allocator", [&] {
            const auto handle = parse(bytes, "The allocator");
            next              = member(handle.get(), "next", "The allocator").as<int64_t>();
        });
    }
#else
    namespace
    {
        Status unavailable()
        {
            return {Status::Code::Failed,
                    Stage::Configure,
                    "The JIT solution library requires a MessagePack build"};
        }
    }

    Status rewriteEntryIndex(const std::vector<uint8_t>&, int32_t, std::vector<uint8_t>&)
    {
        return unavailable();
    }
    Status readEntry(const std::vector<uint8_t>&, EntryFields&)
    {
        return unavailable();
    }
    Status appendEntryPredicates(const std::vector<uint8_t>&,
                                 const std::vector<IndexedPredicate>&,
                                 std::vector<uint8_t>&)
    {
        return unavailable();
    }
    Status appendMasterRows(const std::vector<uint8_t>&,
                            const std::vector<MasterRow>&,
                            std::vector<uint8_t>&)
    {
        return unavailable();
    }
    Status readMasterPrefixes(const std::vector<uint8_t>&, std::vector<std::string>&)
    {
        return unavailable();
    }
    Status writeMapping(const std::map<int32_t, std::string>&, std::vector<uint8_t>&)
    {
        return unavailable();
    }
    Status readMapping(const std::vector<uint8_t>&, std::map<int32_t, std::string>&)
    {
        return unavailable();
    }
    Status writeAllocator(int64_t, std::vector<uint8_t>&)
    {
        return unavailable();
    }
    Status readAllocator(const std::vector<uint8_t>&, int64_t&)
    {
        return unavailable();
    }
#endif
}
