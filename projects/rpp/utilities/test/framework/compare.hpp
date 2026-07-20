#ifndef RPP_TEST_COMPARE_H
#define RPP_TEST_COMPARE_H

#include <cmath>
#include <cstddef>

namespace rpptest {

// Tolerance-based element-wise comparison. Returns true if every element is
// within tolerance of the reference.
template <typename T>
inline bool within_tolerance(const T* actual, const T* reference, std::size_t count,
                             double tolerance) {
    for (std::size_t i = 0; i < count; ++i) {
        if (std::abs(static_cast<double>(actual[i]) - static_cast<double>(reference[i])) >
            tolerance)
            return false;
    }
    return true;
}

}  // namespace rpptest

#endif  // RPP_TEST_COMPARE_H
