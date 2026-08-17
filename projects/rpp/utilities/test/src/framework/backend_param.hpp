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

#include <string>
#include <vector>

namespace rpptest {

// Backends the suite can be instantiated against. HIP is added only when the
// installed rpp was built with the HIP backend.
inline std::vector<RppBackend> available_backends() {
    std::vector<RppBackend> backends = {RPP_HOST_BACKEND};
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
    backends.push_back(RPP_HIP_BACKEND);
#endif
    return backends;
}

inline std::string backend_name(RppBackend backend) {
    return backend == RPP_HIP_BACKEND ? "HIP" : "HOST";
}

}  // namespace rpptest

#endif  // RPP_TEST_BACKEND_PARAM_H
