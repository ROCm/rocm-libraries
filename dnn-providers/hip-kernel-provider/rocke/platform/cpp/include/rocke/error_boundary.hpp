// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/error_boundary.hpp -- helpers that bridge the C++ exception channel
 * (ckc::Error) back to the legacy rocke_status_t + builder->err model at the
 * public extern "C" entry boundary.
 *
 * Internal engine code throws a ckc::Error where the Python reference raises;
 * the public entry points stay extern "C" and translate a caught exception into
 * the status code / message their existing callers expect. These helpers keep
 * that translation in one place so the boundary stays uniform:
 *
 *   - rocke_guard_builder(b, fn): for entries that return a pointer (kernel def,
 *     value, ...) and own/borrow a builder `b`. Runs fn(); on a thrown
 *     ckc::Error it records the status+message on `b` and returns nullptr.
 *
 *   - rocke_guard_status(b, fn): for entries that return a rocke_status_t and have a
 *     builder `b`. Runs fn(); on a thrown ckc::Error it records the message on
 *     `b` and returns the exception's code.
 *
 * The legacy non-throwing paths are untouched: if fn() does not throw, its
 * return value is passed straight through. Both models coexist during the
 * conversion.
 */
#ifndef ROCKE_ERROR_BOUNDARY_HPP
#define ROCKE_ERROR_BOUNDARY_HPP

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"

#include <exception>
#include <new>

namespace ckc
{

/* Run `fn`, returning its (pointer) result. If `fn` throws a ckc::Error, record
 * its status+message on builder `b` (when non-null) and return nullptr. */
template <class Fn>
auto guard_builder(rocke_ir_builder_t* b, Fn&& fn) -> decltype(fn())
{
    using R = decltype(fn());
    try
    {
        return fn();
    }
    catch(const ckc::Error& e)
    {
        rocke_i_set_err_msg(b, e.code(), e.what());
        return static_cast<R>(nullptr);
    }
    catch(const std::bad_alloc& e)
    {
        rocke_i_set_err_msg(b, ROCKE_ERR_OOM, e.what());
        return static_cast<R>(nullptr);
    }
    catch(const std::exception& e)
    {
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, e.what());
        return static_cast<R>(nullptr);
    }
    catch(...)
    {
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "unknown C++ exception at extern \"C\" boundary");
        return static_cast<R>(nullptr);
    }
}

/* Run `fn` (returns rocke_status_t). If `fn` throws a ckc::Error, record its
 * message on builder `b` (when non-null) and return the exception's code. */
template <class Fn>
rocke_status_t guard_status(rocke_ir_builder_t* b, Fn&& fn)
{
    try
    {
        return fn();
    }
    catch(const ckc::Error& e)
    {
        rocke_i_set_err_msg(b, e.code(), e.what());
        return e.code();
    }
    catch(const std::bad_alloc& e)
    {
        rocke_i_set_err_msg(b, ROCKE_ERR_OOM, e.what());
        return ROCKE_ERR_OOM;
    }
    catch(const std::exception& e)
    {
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, e.what());
        return ROCKE_ERR_VALUE;
    }
    catch(...)
    {
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "unknown C++ exception at extern \"C\" boundary");
        return ROCKE_ERR_VALUE;
    }
}

} /* namespace ckc */

#endif /* ROCKE_ERROR_BOUNDARY_HPP */
