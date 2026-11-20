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

#ifndef RPP_TEST_FRAMEWORK_CAPTURE_H
#define RPP_TEST_FRAMEWORK_CAPTURE_H

// Seam between the comparators and the capture tooling (src/capture/, utilities/test_viz/).
// rpp_tests compiles these hooks to nothing; rpp_tests_capture (-DRPP_TEST_CAPTURE=ON) defines
// RPPTEST_CAPTURE and gets the real recorders.

#ifdef RPPTEST_CAPTURE

#include "capture/rpp_capture.hpp"

#else

namespace rpptest::capture {

template <typename... Args>
inline void record_roi(const Args&...) {}
template <typename... Args>
inline void record_nd(const Args&...) {}
template <typename... Args>
inline void record_voxel_roi(const Args&...) {}
template <typename... Args>
inline void record_reduction(const Args&...) {}
inline void install() {}

}  // namespace rpptest::capture

#endif  // RPPTEST_CAPTURE

#endif  // RPP_TEST_FRAMEWORK_CAPTURE_H
