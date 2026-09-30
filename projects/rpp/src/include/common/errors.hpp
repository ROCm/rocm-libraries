/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#ifndef GUARD_RPP_ERRORS_HPP
#define GUARD_RPP_ERRORS_HPP

#include <exception>
#include <string>
#include <tuple>

#include "object.hpp"
#include "returns.hpp"
#include "rpp.h"

#if defined(GPU_SUPPORT)
#include <hip/hip_runtime_api.h>
#endif

namespace rpp {

struct Exception : std::exception {
    std::string message;
    rppStatus_t status;
    Exception(const std::string& msg = "") : message(msg), status(rppStatusUnknownError) {}

    Exception(rppStatus_t s, const std::string& msg = "") : message(msg), status(s) {}

    Exception SetContext(const std::string& file, int line) {
        message = file + ":" + std::to_string(line) + ": " + message;
        return *this;
    }

    // Workaround for HIP, this must be inline
    const char* what() const noexcept override {
        return message.c_str();
    }
};

inline std::string HIPErrorMessage(int error, const std::string& msg = "") {
#if defined(GPU_SUPPORT)
    return msg + " " + hipGetErrorString(static_cast<hipError_t>(error));
#endif
    return msg;
}

#define RPP_THROW(...)                                                    \
    do {                                                                  \
        throw rpp::Exception(__VA_ARGS__).SetContext(__FILE__, __LINE__); \
    } while (false)
#define RPP_THROW_HIP_STATUS(...) \
    RPP_THROW(rppStatusUnknownError, rpp::HIPErrorMessage(__VA_ARGS__))

// TODO(paul): Debug builds should leave the exception uncaught
template <class F>
rppStatus_t try_(F f, bool output = true) {
    try {
        f();
    } catch (const Exception& ex) {
        if (output) {
            fprintf(stderr, "RPP Error: %s\n", ex.what());
        }
        return ex.status;
    } catch (const std::exception& ex) {
        if (output) {
            fprintf(stderr, "RPP Error: %s\n", ex.what());
        }
        return rppStatusUnknownError;
    } catch (...) {
        return rppStatusUnknownError;
    }
    return rppStatusSuccess;
}

template <class T>
auto deref(T&& x, rppStatus_t err = rppStatusBadParm) -> decltype((x == nullptr), get_object(*x)) {
    if (x == nullptr) {
        RPP_THROW(err, "Dereferencing nullptr");
    }
    return get_object(*x);
}

template <class... Ts>
auto tie_deref(Ts&... xs) RPP_RETURNS(std::tie(rpp::deref(xs)...));

}  // namespace rpp

#endif
