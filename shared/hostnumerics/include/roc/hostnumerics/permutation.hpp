// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace roc::hostnumerics {
// An ordering of [0, size()) in which every value appears exactly once. Axis
// orders use it: entry i names the axis at position i, from slowest to fastest.
//
// Every constructor is explicit and validates its entries, so a list of flat
// indices or extents is never silently read as an order.
class Permutation {
   public:
    explicit Permutation(std::vector<size_t> entries) : m_entries(std::move(entries)) {
        validate();
    }

    explicit Permutation(std::span<const size_t> entries)
        : Permutation(std::vector<size_t>(entries.begin(), entries.end())) {}

    explicit Permutation(std::initializer_list<size_t> entries)
        : Permutation(std::vector<size_t>(entries)) {}

    // (0, 1, ..., size - 1). As an axis order, the last axis is fastest.
    static Permutation identity(size_t size) {
        std::vector<size_t> entries(size);
        std::iota(entries.begin(), entries.end(), size_t{0});
        return Permutation(std::move(entries), Validated{});
    }

    // (size - 1, ..., 1, 0). As an axis order, the first axis is fastest.
    static Permutation reverse(size_t size) {
        std::vector<size_t> entries(size);
        std::iota(entries.rbegin(), entries.rend(), size_t{0});
        return Permutation(std::move(entries), Validated{});
    }

    size_t size() const noexcept {
        return m_entries.size();
    }

    // Requires position < size().
    size_t operator[](size_t position) const noexcept {
        return m_entries[position];
    }

    auto begin() const noexcept {
        return m_entries.cbegin();
    }

    auto end() const noexcept {
        return m_entries.cend();
    }

    // The returned view remains valid while this Permutation is alive and unmodified.
    std::span<const size_t> entries() const noexcept {
        return m_entries;
    }

    // The permutation q with q[p[i]] == i, so applying p and then q restores the
    // original order.
    Permutation inverse() const {
        std::vector<size_t> entries(size());
        for (size_t position = 0; position < size(); ++position)
            entries[m_entries[position]] = position;
        return Permutation(std::move(entries), Validated{});
    }

    // Reorder values so that result[i] == values[(*this)[i]]. For example,
    // (2, 0, 1) applied to (a, b, c) gives (c, a, b).
    template <typename T>
    std::vector<T> apply(std::span<const T> values) const {
        return applyValues(values);
    }

    template <typename T>
    std::vector<T> apply(const std::vector<T>& values) const {
        return applyValues(values);
    }

    bool operator==(const Permutation&) const noexcept = default;

   private:
    struct Validated {};

    Permutation(std::vector<size_t> entries, Validated) noexcept : m_entries(std::move(entries)) {}

    template <typename Values>
    std::vector<typename Values::value_type> applyValues(const Values& values) const {
        if (values.size() != size())
            throw std::runtime_error("Cannot apply a permutation of size " +
                                     std::to_string(size()) + " to " +
                                     std::to_string(values.size()) + " values.");
        std::vector<typename Values::value_type> result;
        result.reserve(size());
        for (size_t entry : m_entries) result.push_back(values[entry]);
        return result;
    }

    void validate() const {
        std::vector<bool> seen(size(), false);
        for (size_t position = 0; position < size(); ++position) {
            const size_t entry = m_entries[position];
            if (entry >= size() || seen[entry])
                throw std::runtime_error("Permutation entry " + std::to_string(entry) +
                                         " at position " + std::to_string(position) +
                                         " is repeated or outside [0, " + std::to_string(size()) +
                                         "); each value must appear exactly once.");
            seen[entry] = true;
        }
    }

    std::vector<size_t> m_entries;
};

}  // namespace roc::hostnumerics
