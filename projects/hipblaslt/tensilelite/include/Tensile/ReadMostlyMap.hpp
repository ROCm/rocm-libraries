// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace TensileLite
{
    /**
     * Insert-only hash map whose lookups take no lock and write no shared memory.
     *
     * Entries are never removed and values never replaced, so the map holds one
     * node per distinct key. find() loads the current table and its slots
     * through atomic pointers, so it may run concurrently with a writer.
     * Writers (insert(), forEachValue()) must be serialized by the caller. A
     * writer may change a stored value only if Value is safe to change while
     * readers use it, such as a std::atomic.
     *
     * A concurrent find() may still be probing an outgrown table, so tables are
     * kept until the map is destroyed. Tables double as they grow, which keeps
     * the retired ones smaller in total than the current one.
     *
     * find() and insert() accept any key type that Hash and KeyEqual accept.
     */
    template <typename Key,
              typename Value,
              typename Hash     = std::hash<Key>,
              typename KeyEqual = std::equal_to<Key>>
    class ReadMostlyMap
    {
    public:
        ReadMostlyMap() = default;

        ReadMostlyMap(ReadMostlyMap const&)            = delete;
        ReadMostlyMap& operator=(ReadMostlyMap const&) = delete;

        /**
         * Returns the value stored for `key`, or nullptr. The pointer stays
         * valid until the map is destroyed.
         */
        template <typename K>
        Value const* find(K const& key) const
        {
            // Acquire pairs with the writer's release stores, so a published
            // table or node is seen fully built.
            Table const* table = m_table.load(std::memory_order_acquire);
            if(!table)
                return nullptr;

            // Linear probing from hash & mask, which is hash % capacity since the
            // capacity is a power of two. An empty slot ends the search.
            size_t const hash = mix(Hash{}(key));
            for(size_t i = hash & table->mask;; i = (i + 1) & table->mask)
            {
                Node const* node = table->slots[i].load(std::memory_order_acquire);
                if(!node)
                    return nullptr;
                // The stored hash is compared first; keys only when it matches.
                if(node->hash == hash && KeyEqual{}(node->key, key))
                    return &node->value;
            }
        }

        /**
         * Writer only. Stores Value(args...) for `key` unless the key is
         * present, and returns the stored value. A present key keeps its value
         * and allocates nothing.
         */
        template <typename K, typename... Args>
        Value& insert(K&& key, Args&&... args)
        {
            size_t const hash = mix(Hash{}(key));
            if(m_current)
            {
                for(size_t i = hash & m_current->mask;; i = (i + 1) & m_current->mask)
                {
                    Node* node = m_current->slots[i].load(std::memory_order_relaxed);
                    if(!node)
                        break;
                    if(node->hash == hash && KeyEqual{}(node->key, key))
                        return node->value;
                }
            }

            // At most half full, so a probe always reaches an empty slot.
            if(!m_current || (m_nodes.size() + 1) * 2 > m_current->mask + 1)
                grow();
            m_nodes.push_back(std::make_unique<Node>(
                Key(std::forward<K>(key)), hash, std::forward<Args>(args)...));
            Node* node = m_nodes.back().get();
            place(*m_current, node);
            return node->value;
        }

        /// Writer only. Calls f(Value&) for every entry.
        template <typename F>
        void forEachValue(F&& f)
        {
            for(auto& node : m_nodes)
                f(node->value);
        }

    private:
        static constexpr size_t c_initialCapacity = 8;

        struct Node
        {
            template <typename... Args>
            Node(Key k, size_t h, Args&&... args)
                : key(std::move(k))
                , value(std::forward<Args>(args)...)
                , hash(h)
            {
            }

            Key const    key;
            Value        value;
            size_t const hash;
        };

        struct Table
        {
            explicit Table(size_t capacity)
                : mask(capacity - 1)
                , slots(new std::atomic<Node*>[capacity])
            {
                for(size_t i = 0; i < capacity; ++i)
                    slots[i].store(nullptr, std::memory_order_relaxed);
            }

            size_t                                mask; // capacity - 1
            std::unique_ptr<std::atomic<Node*>[]> slots;
        };

        // Linear probing uses the low bits, so spread every input bit into them.
        static size_t mix(size_t h)
        {
            uint64_t x = h;
            x ^= x >> 30;
            x *= 0xbf58476d1ce4e5b9ull;
            x ^= x >> 27;
            x *= 0x94d049bb133111ebull;
            x ^= x >> 31;
            return static_cast<size_t>(x);
        }

        static void place(Table& table, Node* node)
        {
            size_t i = node->hash & table.mask;
            while(table.slots[i].load(std::memory_order_relaxed))
                i = (i + 1) & table.mask;
            table.slots[i].store(node, std::memory_order_release);
        }

        // Doubling keeps the capacity a power of two and the retired tables
        // smaller in total than the current one. Only node pointers are copied.
        void grow()
        {
            size_t const capacity = m_current ? (m_current->mask + 1) * 2 : c_initialCapacity;
            auto         table    = std::make_unique<Table>(capacity);
            if(m_current)
                for(size_t i = 0; i <= m_current->mask; ++i)
                    if(Node* node = m_current->slots[i].load(std::memory_order_relaxed))
                        place(*table, node);

            m_tables.push_back(std::move(table));
            m_current = m_tables.back().get();
            m_table.store(m_current, std::memory_order_release);
        }

        std::atomic<Table const*> m_table{nullptr};

        // Writer state.
        Table*                              m_current = nullptr;
        std::vector<std::unique_ptr<Table>> m_tables;
        std::vector<std::unique_ptr<Node>>  m_nodes;
    };

    /**
     * Cache whose entries are never replaced or removed. find() takes no lock;
     * add() serializes writers on a mutex and keeps the first value stored for
     * a key.
     */
    template <typename Value, typename Key>
    class InsertOnceCache
    {
    public:
        explicit InsertOnceCache(Value const& nullValue)
            : m_nullValue(nullValue)
        {
        }

        /// The value stored for `key`, or the null value.
        Value find(Key const& key) const
        {
            Value const* found = m_map.find(key);
            return found ? *found : m_nullValue;
        }

        void add(Value const& value, Key const& key)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_map.insert(key, value);
        }

    private:
        ReadMostlyMap<Key, Value> m_map;
        std::mutex                m_mutex;
        Value                     m_nullValue;
    };
} // namespace TensileLite
