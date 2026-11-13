// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_tensor_view_buffer_errors.cpp -- the buffer address space of
 * the C++ tensor view refuses what Python's rocke.helpers.tensor_view refuses,
 * with the same error kind and the same message
 * (tests/core/test_tensor_view_buffer_errors.py pins the Python side with the
 * same rows). The tensor_view_buffer parity family only sees that both engines
 * reject these calls; this test pins why.
 *
 * Python NotImplementedError, ValueError and TypeError are ckc::Error codes
 * ROCKE_ERR_NOTIMPL, ROCKE_ERR_VALUE and ROCKE_ERR_TYPE.
 */
#include <cstdio>
#include <cstring>
#include <string>

#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/ir.h"

namespace
{

int failures = 0;

void expect(bool ok, const std::string& what)
{
    if(!ok)
    {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

const rocke_type_t* type_by_name(const char* name)
{
    if(std::strcmp(name, "f16") == 0)
        return rocke_f16();
    if(std::strcmp(name, "bf16") == 0)
        return rocke_bf16();
    if(std::strcmp(name, "f32") == 0)
        return rocke_f32();
    if(std::strcmp(name, "i32") == 0)
        return rocke_i32();
    if(std::strcmp(name, "i64") == 0)
        return rocke_i64();
    return rocke_f64();
}

/* One refused call. The rows match _CASES in
 * tests/core/test_tensor_view_buffer_errors.py. */
struct error_case
{
    const char* op;
    const char* dtype;
    int n;
    rocke_status_t code;
    const char* message;
};

const error_case k_cases[] = {
    {"load_scalar",
     "bf16",
     1,
     ROCKE_ERR_NOTIMPL,
     "buffer scalar load not yet wired for dtype bf16 (only f16 is exposed by the IR builder "
     "today)"},
    {"load_scalar", "f64", 1, ROCKE_ERR_NOTIMPL, "buffer-space load/store not wired for dtype f64"},
    {"store_scalar",
     "f32",
     1,
     ROCKE_ERR_NOTIMPL,
     "buffer scalar store not yet wired for dtype f32"},
    {"load_vec", "f16", 3, ROCKE_ERR_VALUE, "buffer load f16 supports n in {2, 4, 8} (got 3)"},
    {"load_vec", "f32", 4, ROCKE_ERR_NOTIMPL, "buffer vec load not yet wired for dtype f32"},
    {"store_vec", "f16", 16, ROCKE_ERR_VALUE, "buffer store f16 supports n in {2, 4, 8} (got 16)"},
    {"store_vec", "bf16", 4, ROCKE_ERR_NOTIMPL, "buffer vec store not yet wired for dtype bf16"},
    {"load_vec_at",
     "f16",
     6,
     ROCKE_ERR_VALUE,
     "buffer load_vec_at f16 supports n in {2, 4, 8} (got 6)"},
    {"load_vec_at", "i32", 4, ROCKE_ERR_NOTIMPL, "buffer load_vec_at not wired for dtype i32"},
    {"store_vec_at_mask",
     "f16",
     1,
     ROCKE_ERR_VALUE,
     "buffer store_vec_at f16 supports n in {2, 4, 8} (got 1)"},
    {"store_vec_at", "f32", 4, ROCKE_ERR_NOTIMPL, "buffer store_vec_at not wired for dtype f32"},
    {"load_vec_at_mask_global",
     "f16",
     4,
     ROCKE_ERR_NOTIMPL,
     "mask= requires addr_space='buffer' (got 'global'); software masking for global/lds is "
     "not yet wired"},
    {"store_vec_at_mask_global",
     "f16",
     4,
     ROCKE_ERR_NOTIMPL,
     "mask= requires addr_space='buffer' (got 'global')"},
    {"load_vec_no_rsrc",
     "f16",
     4,
     ROCKE_ERR_TYPE,
     "buffer TensorView's base must be a BufferResource"},
    {"buffer_of_global",
     "f16",
     1,
     ROCKE_ERR_TYPE,
     "TensorView.buffer requires addr_space='buffer', got 'global'"},
};

/* C-only rows: Python cannot pass a missing value, so these have no .py twin.
 * A buffer store with a NULL value is refused, as global_store_vN refuses one,
 * instead of emitting a load (the vec_at forms) or a store with no value. */
const error_case k_null_value_cases[] = {
    {"store_scalar", "f16", 1, ROCKE_ERR_VALUE, "buffer store_scalar: null value"},
    {"store_vec", "f16", 4, ROCKE_ERR_VALUE, "buffer store_vec: null value"},
    {"store_vec_at", "f16", 4, ROCKE_ERR_VALUE, "buffer store_vec_at: null value"},
    {"store_vec_at_mask", "f16", 4, ROCKE_ERR_VALUE, "buffer store_vec_at: null value"},
};

bool op_is(const error_case& c, const char* op)
{
    return std::strcmp(c.op, op) == 0;
}

/* Issue the case's call on a (4, 64) view of c.dtype: a buffer view over a
 * BufferResource, a buffer view without one, or a global view. */
void issue(rocke_ir_builder_t* b, const error_case& c, bool null_value)
{
    static const int shape[2] = {4, 64};
    const rocke_type_t* dtype = type_by_name(c.dtype);
    rocke_value_t* x = rocke_b_param(b, "X", rocke_ptr_type(b, dtype, "global"), NULL);
    rocke_value_t* n_bytes = rocke_b_param(b, "N_bytes", rocke_i32(), NULL);
    rocke_tv_buffer_resource_t rsrc;
    rocke_tensor_view_t view;
    if(rocke_tv_make_buffer_resource(b, &rsrc, x, n_bytes, NULL) != ROCKE_OK)
        return;
    if(op_is(c, "load_vec_at_mask_global") || op_is(c, "store_vec_at_mask_global")
       || op_is(c, "buffer_of_global"))
    {
        if(rocke_make_global_view(&view, x, shape, 2, dtype, NULL) != ROCKE_OK)
            return;
    }
    else if(rocke_tv_make_buffer_view(
                &view, op_is(c, "load_vec_no_rsrc") ? NULL : &rsrc, shape, 2, dtype, NULL)
            != ROCKE_OK)
        return;
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* row = rocke_b_const_i32(b, 1);
    rocke_value_t* idx[2] = {row, tid};
    rocke_value_t* value = null_value ? NULL : rocke_b_const_i32(b, 0);
    rocke_value_t* c_lim = rocke_b_const_i32(b, 48);
    rocke_value_t* mask = rocke_b_cmp_lt(b, tid, c_lim);

    if(op_is(c, "load_scalar"))
        rocke_tensor_view_load_scalar(b, &view, idx, 2);
    else if(op_is(c, "store_scalar"))
        rocke_tensor_view_store_scalar(b, &view, idx, 2, value, 0);
    else if(op_is(c, "load_vec") || op_is(c, "load_vec_no_rsrc"))
        rocke_tensor_view_load_vec(b, &view, idx, 2, c.n);
    else if(op_is(c, "store_vec"))
        rocke_tensor_view_store_vec(b, &view, idx, 2, value, c.n);
    else if(op_is(c, "load_vec_at"))
        rocke_tensor_view_load_vec_at(b, &view, tid, c.n);
    else if(op_is(c, "store_vec_at"))
        rocke_tensor_view_store_vec_at(b, &view, tid, value, c.n);
    else if(op_is(c, "load_vec_at_mask_global"))
        rocke_tensor_view_load_vec_at_masked(b, &view, tid, c.n, mask);
    else if(op_is(c, "store_vec_at_mask") || op_is(c, "store_vec_at_mask_global"))
        rocke_tensor_view_store_vec_at_masked(b, &view, tid, value, c.n, mask);
    else if(op_is(c, "buffer_of_global"))
        rocke_tensor_view_buffer(b, &view);
    else
        throw ckc::ValueError(std::string("no error case op named '") + c.op + "'");
}

void check(const error_case& c, bool null_value)
{
    const std::string what = std::string(c.op) + " " + c.dtype + " n=" + std::to_string(c.n)
                             + (null_value ? " (null value)" : "");
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "tensor_view_buffer_errors") != ROCKE_OK)
    {
        expect(false, what + ": builder init failed");
        return;
    }
    rocke_status_t code = ROCKE_OK;
    std::string message;
    try
    {
        issue(&b, c, null_value);
        code = rocke_ir_builder_status(&b);
        if(code != ROCKE_OK)
            message = rocke_ir_builder_error(&b);
    }
    catch(const ckc::Error& error)
    {
        code = error.code();
        message = error.what();
    }
    rocke_ir_builder_free(&b);
    expect(code == c.code,
           what + ": status " + std::to_string((int)code) + " != " + std::to_string((int)c.code));
    expect(message == c.message, what + ": message '" + message + "' != '" + c.message + "'");
}

} // namespace

int main()
{
    for(const error_case& c : k_cases)
        check(c, false);
    for(const error_case& c : k_null_value_cases)
        check(c, true);
    if(failures)
        std::fprintf(stderr, "%d tensor_view buffer error check(s) failed\n", failures);
    return failures ? 1 : 0;
}
