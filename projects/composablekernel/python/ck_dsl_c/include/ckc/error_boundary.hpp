// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * ckc/error_boundary.hpp -- helpers that bridge the C++ exception channel
 * (ckc::Error) back to the legacy ckc_status_t + builder->err model at the
 * public extern "C" entry boundary.
 *
 * Internal engine code throws a ckc::Error where the Python reference raises;
 * the public entry points stay extern "C" and translate a caught exception into
 * the status code / message their existing callers expect. These helpers keep
 * that translation in one place so the boundary stays uniform:
 *
 *   - ckc_guard_builder(b, fn): for entries that return a pointer (kernel def,
 *     value, ...) and own/borrow a builder `b`. Runs fn(); on a thrown
 *     ckc::Error it records the status+message on `b` and returns nullptr.
 *
 *   - ckc_guard_status(b, fn): for entries that return a ckc_status_t and have a
 *     builder `b`. Runs fn(); on a thrown ckc::Error it records the message on
 *     `b` and returns the exception's code.
 *
 * The legacy non-throwing paths are untouched: if fn() does not throw, its
 * return value is passed straight through. Both models coexist during the
 * conversion.
 */
#ifndef CKC_ERROR_BOUNDARY_HPP
#define CKC_ERROR_BOUNDARY_HPP

#include "ckc/error.hpp"
#include "ckc/ir.h"
#include "ckc/ir_internal.h"

namespace ckc {

/* Run `fn`, returning its (pointer) result. If `fn` throws a ckc::Error, record
 * its status+message on builder `b` (when non-null) and return nullptr. */
template <class Fn>
auto guard_builder(ckc_ir_builder_t* b, Fn&& fn) -> decltype(fn())
{
    using R = decltype(fn());
    try
    {
        return fn();
    }
    catch(const ckc::Error& e)
    {
        ckc_i_set_err_msg(b, e.code(), e.what());
        return static_cast<R>(nullptr);
    }
}

/* Run `fn` (returns ckc_status_t). If `fn` throws a ckc::Error, record its
 * message on builder `b` (when non-null) and return the exception's code. */
template <class Fn>
ckc_status_t guard_status(ckc_ir_builder_t* b, Fn&& fn)
{
    try
    {
        return fn();
    }
    catch(const ckc::Error& e)
    {
        ckc_i_set_err_msg(b, e.code(), e.what());
        return e.code();
    }
}

} /* namespace ckc */

#endif /* CKC_ERROR_BOUNDARY_HPP */
