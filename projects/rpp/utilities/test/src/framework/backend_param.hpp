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

#ifndef RPP_TEST_BACKEND_PARAM_H
#define RPP_TEST_BACKEND_PARAM_H

#include <rpp/rpp.h>

#include <ostream>
#include <string>
#include <vector>

namespace rpptest {

// Backends the suite can be instantiated against. HIP is added only when the
// installed rpp was built with the HIP backend.
inline std::vector<RppBackend> available_backends() {
    std::vector<RppBackend> backends = {RPP_HOST_BACKEND};
#if RPP_BACKEND_HIP
    backends.push_back(RPP_HIP_BACKEND);
#endif
    return backends;
}

inline std::string backend_name(RppBackend backend) {
    return backend == RPP_HIP_BACKEND ? "HIP" : "HOST";
}

// numThreads value that keeps rppCreate's default.
inline constexpr Rpp32u kDefaultThreads = 0;

// HOST thread counts every op is tested with: the default, a single thread, and more threads than
// the default batch of 2. rppCreate caps the count at std::thread::hardware_concurrency(), so a
// small machine may run the larger count with fewer.
inline const std::vector<Rpp32u> kHostThreadCounts{kDefaultThreads, 1, 4};

// A backend plus the numThreads its handle is created with. Each config type derives from this,
// so RppHandle can be built straight from a config.
struct BackendVariant {
    RppBackend backend;
    Rpp32u numThreads = kDefaultThreads;

    // Label token: "HOST" for the default thread count, "HOST_T1" for an explicit one.
    std::string name() const {
        return numThreads == kDefaultThreads
                   ? backend_name(backend)
                   : backend_name(backend) + "_T" + std::to_string(numThreads);
    }
};

inline std::vector<BackendVariant> backend_variants() {
    std::vector<BackendVariant> variants;
    for (RppBackend backend : available_backends()) {
        if (backend == RPP_HOST_BACKEND)
            for (Rpp32u n : kHostThreadCounts) variants.push_back({backend, n});
        else
            variants.push_back({backend, kDefaultThreads});  // HIP ignores numThreads
    }
    return variants;
}

}  // namespace rpptest

// Every parameter is already spelled out in the test name, so print nothing instead of gtest's
// ", where GetParam() = ..." suffix (a raw byte dump for config structs).
inline void PrintTo(const RppBackend&, std::ostream*) {}

#endif  // RPP_TEST_BACKEND_PARAM_H
