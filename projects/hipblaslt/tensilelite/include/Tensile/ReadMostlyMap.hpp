// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace TensileLite
{
    /**
     * Hash map whose lookups take no lock and write no shared memory.
     *
     * An entry is immutable once published. find() loads the current table and
     * its slots through atomic pointers, so it may run concurrently with
     * insertOrAssign() and clear(). Writers must be serialized by the caller.
     *
     * A concurrent find() may still be reading a replaced entry or an outgrown
     * table, so both are kept until the map is destroyed. Tables double as they
     * grow, which keeps the retired ones smaller in total than the current one;
     * each replaced or cleared entry keeps its node, so use it only for data
     * that is rarely overwritten.
     *
     * find() accepts any key type that Hash and KeyEqual accept.
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

        void insertOrAssign(Key key, Value value)
        {
            size_t const hash = mix(Hash{}(key));
            m_nodes.push_back(std::make_unique<Node>(Node{std::move(key), std::move(value), hash}));
            Node const* node = m_nodes.back().get();

            if(m_current)
            {
                for(size_t i = hash & m_current->mask;; i = (i + 1) & m_current->mask)
                {
                    Node const* old = m_current->slots[i].load(std::memory_order_relaxed);
                    if(!old)
                        break;
                    if(old->hash == hash && KeyEqual{}(old->key, node->key))
                    {
                        // Readers may still hold `old`, so it is replaced, not edited.
                        m_current->slots[i].store(node, std::memory_order_release);
                        return;
                    }
                }
            }

            // At most half full, so a probe always reaches an empty slot.
            if(!m_current || (m_size + 1) * 2 > m_current->mask + 1)
                grow();
            place(*m_current, node);
            ++m_size;
        }

        void clear()
        {
            m_current = nullptr;
            m_size    = 0;
            m_table.store(nullptr, std::memory_order_release);
        }

    private:
        static constexpr size_t c_initialCapacity = 8;

        struct Node
        {
            Key    key;
            Value  value;
            size_t hash;
        };

        struct Table
        {
            explicit Table(size_t capacity)
                : mask(capacity - 1)
                , slots(new std::atomic<Node const*>[capacity])
            {
                for(size_t i = 0; i < capacity; ++i)
                    slots[i].store(nullptr, std::memory_order_relaxed);
            }

            size_t                                      mask; // capacity - 1
            std::unique_ptr<std::atomic<Node const*>[]> slots;
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

        static void place(Table& table, Node const* node)
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
                    if(Node const* node = m_current->slots[i].load(std::memory_order_relaxed))
                        place(*table, node);

            m_tables.push_back(std::move(table));
            m_current = m_tables.back().get();
            m_table.store(m_current, std::memory_order_release);
        }

        std::atomic<Table const*> m_table{nullptr};

        // Writer state.
        Table*                              m_current = nullptr;
        size_t                              m_size    = 0;
        std::vector<std::unique_ptr<Table>> m_tables;
        std::vector<std::unique_ptr<Node>>  m_nodes;
    };
} // namespace TensileLite
