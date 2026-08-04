// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <roc/host_validation/tensor.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace roc::host_validation {
inline uint64_t counterRandom(uint64_t seed, uint64_t stream, uint64_t index) {
    uint64_t value = seed ^ (stream + 0x9e3779b97f4a7c15ULL) ^ (index * 0xbf58476d1ce4e5b9ULL);
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline int indexedUniformInteger(uint64_t seed, uint64_t stream, uint64_t index, int lower,
                                 int upper) {
    if (lower > upper)
        throw std::invalid_argument("indexedUniformInteger lower bound exceeds upper bound.");
    const uint64_t range =
        static_cast<uint64_t>(static_cast<int64_t>(upper) - static_cast<int64_t>(lower) + 1);
    return static_cast<int>(static_cast<int64_t>(lower) +
                            static_cast<int64_t>(counterRandom(seed, stream, index) % range));
}

enum class DataPattern {
    Zero,
    RandomInteger,
    UniformInteger,
    AlternatingRandomInteger,
    UniformReal,
    Sine,
    Cosine,
    Constant,
};

template <typename T, typename Generator>
void generate(MatrixView<T> destination, Generator&& generator) {
    for (size_t column = 0; column < destination.columns(); ++column) {
        for (size_t row = 0; row < destination.rows(); ++row)
            destination(row, column) = static_cast<T>(generator(row, column));
    }
}

/**
 * Small host-data generator used by the proof-of-concept cut.
 *
 * The class centralizes the random stream and the common distributions so
 * callers do not each own another set of initialization helpers. A
 * counter-based, cross-language generator should replace this engine before
 * the API is declared stable.
 */
class RandomGenerator {
   public:
    explicit RandomGenerator(uint32_t seed) : m_generator(seed) {}

    template <typename T>
    T binary() {
        return static_cast<T>(m_binaryDistribution(m_generator) ? 1.0 : -1.0);
    }

    template <typename T>
    T uniformReal(double lower, double upper) {
        std::uniform_real_distribution<double> distribution(lower, upper);
        return static_cast<T>(distribution(m_generator));
    }

    template <typename T>
    T uniformInteger(int lower, int upper) {
        static_assert(std::is_constructible_v<T, int> || std::is_arithmetic_v<T>);
        std::uniform_int_distribution<int> distribution(lower, upper);
        return static_cast<T>(distribution(m_generator));
    }

    template <typename T>
    T choose(std::span<const T> values) {
        if (values.empty())
            throw std::invalid_argument("RandomGenerator::choose requires non-empty values.");

        std::uniform_int_distribution<size_t> distribution(0, values.size() - 1);
        return values[distribution(m_generator)];
    }

    template <typename T>
    void fillBinary(std::span<T> values) {
        for (auto& value : values) value = binary<T>();
    }

    template <typename T>
    void fillUniformReal(std::span<T> values, double lower, double upper) {
        for (auto& value : values) value = uniformReal<T>(lower, upper);
    }

   private:
    std::mt19937 m_generator;
    std::uniform_int_distribution<> m_binaryDistribution{0, 1};
};

template <typename T>
void fill(std::span<T> values, DataPattern pattern, RandomGenerator& generator,
          double parameter0 = 0.0, double parameter1 = 0.0) {
    for (size_t index = 0; index < values.size(); ++index) {
        double value = 0.0;
        switch (pattern) {
            case DataPattern::Zero:
                value = 0.0;
                break;
            case DataPattern::RandomInteger:
                value = generator.uniformInteger<int>(1, 10);
                break;
            case DataPattern::UniformInteger:
                value = generator.uniformInteger<int>(static_cast<int>(parameter0),
                                                      static_cast<int>(parameter1));
                break;
            case DataPattern::AlternatingRandomInteger:
                value = generator.uniformInteger<int>(1, 10);
                if ((index & 1) == 0) value = -value;
                break;
            case DataPattern::UniformReal:
                value = generator.uniformReal<double>(parameter0, parameter1);
                break;
            case DataPattern::Sine:
                value = std::sin(static_cast<double>(index));
                break;
            case DataPattern::Cosine:
                value = std::cos(static_cast<double>(index));
                break;
            case DataPattern::Constant:
                value = parameter0;
                break;
        }
        values[index] = static_cast<T>(value);
    }
}

template <typename Destination, typename Source>
std::vector<Destination> convertValues(std::span<const Source> source) {
    std::vector<Destination> result(source.size());
    for (size_t i = 0; i < source.size(); ++i) result[i] = static_cast<Destination>(source[i]);
    return result;
}
}  // namespace roc::host_validation
