// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/error.hpp -- the C++ exception hierarchy for the rocke engine.
 *
 * This is the C++-native error channel that mirrors the Python exceptions raised
 * by rocke. Internal engine code throws a ckc::Error subclass exactly where the
 * Python reference would `raise`; the public extern "C" entry points catch it at
 * the boundary and translate it back into the legacy rocke_status_t + builder->err
 * message, so the extern "C" ABI (and every external caller -- provider,
 * emitters, bindings) is unchanged.
 *
 *   Python exception          ckc::Error subclass     rocke_status_t code
 *   -----------------------    --------------------    -----------------
 *   ValueError                 ckc::ValueError         ROCKE_ERR_VALUE
 *   TypeError                  ckc::TypeError          ROCKE_ERR_TYPE
 *   KeyError                   ckc::KeyError           ROCKE_ERR_KEY
 *   MemoryError / OOM          ckc::OOMError           ROCKE_ERR_OOM
 *   NotImplementedError        ckc::NotImplError       ROCKE_ERR_NOTIMPL
 *
 * Each carries a human-readable message and its rocke_status_t code. The message
 * text is kept byte-identical to the legacy *_set_err strings so that parity /
 * validation reject reasons continue to match the Python reference verbatim.
 *
 * Formatting: ckc::format_error mirrors the vsnprintf-based, ROCKE_ERR_MSG_CAP-
 * bounded message formatting used throughout the engine, so throw sites can keep
 * their existing printf-style format strings unchanged.
 */
#ifndef ROCKE_ERROR_HPP
#define ROCKE_ERROR_HPP

#include <cstdarg>
#include <cstdio>
#include <exception>
#include <string>

#include "rocke/ir.h" /* rocke_status_t, ROCKE_ERR_MSG_CAP */

namespace ckc
{

/* Base of the engine exception hierarchy. Carries the message and the
 * rocke_status_t code the boundary shim reports back across the C ABI. */
class Error : public std::exception
{
public:
    Error(rocke_status_t code, std::string msg)
        : code_(code)
        , msg_(std::move(msg))
    {
    }

    const char* what() const noexcept override
    {
        return msg_.c_str();
    }
    rocke_status_t code() const noexcept
    {
        return code_;
    }
    const std::string& message() const noexcept
    {
        return msg_;
    }

private:
    rocke_status_t code_;
    std::string msg_;
};

/* maps to Python ValueError */
class ValueError : public Error
{
public:
    explicit ValueError(std::string msg)
        : Error(ROCKE_ERR_VALUE, std::move(msg))
    {
    }
};

/* maps to Python TypeError */
class TypeError : public Error
{
public:
    explicit TypeError(std::string msg)
        : Error(ROCKE_ERR_TYPE, std::move(msg))
    {
    }
};

/* maps to Python KeyError (unknown op_id / param) */
class KeyError : public Error
{
public:
    explicit KeyError(std::string msg)
        : Error(ROCKE_ERR_KEY, std::move(msg))
    {
    }
};

/* allocation failure */
class OOMError : public Error
{
public:
    explicit OOMError(std::string msg)
        : Error(ROCKE_ERR_OOM, std::move(msg))
    {
    }
};

/* maps to Python NotImplementedError */
class NotImplError : public Error
{
public:
    explicit NotImplError(std::string msg)
        : Error(ROCKE_ERR_NOTIMPL, std::move(msg))
    {
    }
};

/* printf-style message formatting, bounded to ROCKE_ERR_MSG_CAP exactly like the
 * legacy *_set_err sites (truncation of an over-long reason string is
 * intentional and harmless -- these are reject/error paths only). Returns the
 * formatted std::string so a throw site can write:
 *     throw ckc::ValueError(ckc::format_error("bad N=%d", n));
 */
inline std::string format_error(const char* fmt, ...)
{
    char buf[ROCKE_ERR_MSG_CAP];
    if(fmt == nullptr)
    {
        buf[0] = '\0';
    }
    else
    {
        va_list ap;
        va_start(ap, fmt);
        (void)vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        buf[sizeof(buf) - 1] = '\0';
    }
    return std::string(buf);
}

/* Throw the ckc::Error subclass that corresponds to a rocke_status_t code, with
 * the given (already-formatted) message. Centralizes the code->exception mapping
 * so per-subsystem set_err shims can convert to throwing with one call. */
[[noreturn]] inline void raise_status(rocke_status_t code, const char* msg)
{
    std::string m = (msg != nullptr) ? std::string(msg) : std::string();
    switch(code)
    {
    case ROCKE_ERR_TYPE:
        throw TypeError(std::move(m));
    case ROCKE_ERR_KEY:
        throw KeyError(std::move(m));
    case ROCKE_ERR_OOM:
        throw OOMError(std::move(m));
    case ROCKE_ERR_NOTIMPL:
        throw NotImplError(std::move(m));
    case ROCKE_ERR_VALUE:
    default:
        throw ValueError(std::move(m));
    }
}

} /* namespace ckc */

#endif /* ROCKE_ERROR_HPP */
