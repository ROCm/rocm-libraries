/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_TEST_CAPTURE_BUNDLE_H
#define RPP_TEST_CAPTURE_BUNDLE_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Library-agnostic capture bundle: one directory per GTest case holding manifest.json and one .npy
// per array. Knows nothing about RPP; the adapter (rpp_capture.hpp) maps descriptors onto it.
// The format is documented in utilities/test_viz/README.md.

namespace rpptest::capture {

// NumPy dtype string for T. Specialize for types that are not arithmetic (e.g. a half class).
template <typename T>
struct NpyDescr {
    static std::string get() {
        static_assert(std::is_arithmetic_v<T>, "specialize NpyDescr for this type");
        const char kind = std::is_floating_point_v<T> ? 'f' : (std::is_signed_v<T> ? 'i' : 'u');
        return std::string(sizeof(T) == 1 ? "|" : "<") + kind + std::to_string(sizeof(T));
    }
};

// A dense row-major array of one element type.
struct Array {
    std::string descr;
    std::vector<std::size_t> shape;
    std::vector<unsigned char> bytes;

    template <typename T>
    static Array of(const std::vector<std::size_t>& shape) {
        std::size_t count = 1;
        for (std::size_t e : shape) count *= e;
        return {NpyDescr<T>::get(), shape, std::vector<unsigned char>(count * sizeof(T))};
    }

    template <typename T>
    void put(std::size_t i, const T& v) {
        std::memcpy(bytes.data() + i * sizeof(T), static_cast<const void*>(&v), sizeof(T));
    }
};

// Element status in a Comparison.
enum Status : std::uint8_t { kNotCompared = 0, kPass = 1, kFail = 2 };

// One comparator call: reference and actual over the same canonical shape, plus the signed
// difference and per-element status over the elements the comparator checked.
struct Comparison {
    std::string kind;
    std::vector<std::string> axes;
    Array reference, actual;
    Array diff;    // float32, actual - reference; NaN where not compared
    Array status;  // uint8 Status
    std::size_t compared = 0, failed = 0;

    template <typename TRef, typename TAct>
    static Comparison of(std::string kind, std::vector<std::string> axes,
                         const std::vector<std::size_t>& shape) {
        Comparison c;
        c.kind = std::move(kind);
        c.axes = std::move(axes);
        c.reference = Array::of<TRef>(shape);
        c.actual = Array::of<TAct>(shape);
        c.diff = Array::of<float>(shape);
        c.status = Array::of<std::uint8_t>(shape);
        const std::size_t count = c.status.bytes.size();
        for (std::size_t i = 0; i < count; ++i)
            c.diff.put(i, std::numeric_limits<float>::quiet_NaN());
        return c;
    }

    void mark(std::size_t i, double difference, bool pass) {
        diff.put(i, static_cast<float>(difference));
        status.put(i, static_cast<std::uint8_t>(pass ? kPass : kFail));
        ++compared;
        if (!pass) ++failed;
    }
};

// True when RPP_TEST_CAPTURE_DIR is set; adapters skip all work otherwise.
bool enabled();

// Queues a comparison for the running test; written out when the test ends.
void submit(Comparison&& comparison);

// Registers the GTest listener that writes bundles. No-op unless enabled().
void install();

}  // namespace rpptest::capture

#endif  // RPP_TEST_CAPTURE_BUNDLE_H
