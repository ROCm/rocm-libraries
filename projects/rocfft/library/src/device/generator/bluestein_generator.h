// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "generator.h"

enum BluesteinOperationType
{
    BFN_LOAD_CC_FWD_CHIRP,
    BFN_LOAD_RC_FWD_CHIRP,
    BFN_LOAD_CC_FWD_CHIRP_MUL,
    BFN_LOAD_RC_FWD_CHIRP_MUL,
    BFN_LOAD_CC_INV_CHIRP_MUL,
    BFN_LOAD_RC_INV_CHIRP_MUL,
    BFN_STORE_CC_FWD_CHIRP,
    BFN_STORE_RC_FWD_CHIRP,
    BFN_STORE_CC_FWD_CHIRP_MUL,
    BFN_STORE_RC_FWD_CHIRP_MUL,
    BFN_STORE_CC_INV_CHIRP_MUL,
    BFN_STORE_RC_INV_CHIRP_MUL,
};

struct BluesteinData
{
public:
    BluesteinData() {}

    //
    // templates
    //
    Variable scalar_type{"scalar_type", "typename"};
    Variable callback_type{"cbtype", "CallbackType"};

    //
    // internal variables
    //
    Variable chirp{"chirp", "const scalar_type", true, true};
    Variable data_idx{"data_idx", "size_t"};
    Variable data_voffset{"data_voffset", "size_t"};
    Variable data_soffset{"data_soffset", "size_t"};
    Variable data_rw_flag{"data_rw_flag", "bool"};
    Variable buf_in{"buf_in", "scalar_type", true};
    Variable buf_inre{"buf_inre", "real_type_t<scalar_type>", true, true};
    Variable buf_inim{"buf_inim", "real_type_t<scalar_type>", true, true};
    Variable buf_out{"buf_out", "scalar_type", true};
    Variable buf_outre{"buf_outre", "real_type_t<scalar_type>", true, true};
    Variable buf_outim{"buf_outim", "real_type_t<scalar_type>", true, true};
    Variable data_buf{"data_buf", "scalar_type", true};
    Variable data_bufre{"data_bufre", "real_type_t<scalar_type>", true, true};
    Variable data_bufim{"data_bufim", "real_type_t<scalar_type>", true, true};
    Variable data_elem{"data_elem", "scalar_type"};
    Variable length_N_blue{"length_N_blue", "const size_t"};
    Variable length_M_blue{"length_M_blue", "const size_t"};
    Variable global_stride_in_0{"global_stride_in_0", "const size_t"};
    Variable global_stride_in_1{"global_stride_in_1", "const size_t"};
    Variable global_idist{"global_idist", "const size_t"};
    Variable global_stride_out_0{"global_stride_out_0", "const size_t"};
    Variable global_stride_out_1{"global_stride_out_1", "const size_t"};
    Variable global_odist{"global_odist", "const size_t"};
    Variable transform_idx{"transform_idx", "const size_t"};

    //
    // variables borrowed from stockham_gen_base.h
    //
    Variable global_data_id{"global_data_id", "size_t"};
    Variable global_transf_id{"global_transf_id", "size_t"};
    Variable load_cb_data{"load_cb_data", "void*"};
    Variable load_cb_fn{"load_cb_fn", "void", true, true};
    Variable store_cb_data{"store_cb_data", "void*"};
    Variable store_cb_fn{"store_cb_fn", "void", true, true};
};

class BluesteinFunction
{
public:
    Expression get_load_op(const BluesteinOperationType& type, const LoadGlobal& x, bool planar)
    {
        return get_load_op(type, x.args[1], planar);
    }

    Expression
        get_load_op(const BluesteinOperationType& type, const LoadGlobalPlanar& x, bool planar)
    {
        return get_load_op(type, x.args[2], planar);
    }

    Expression get_load_op(const BluesteinOperationType& type, const IntrinsicLoad& x, bool planar)
    {
        return get_intrinsic_load_op(type, x.args[1], x.args[2], x.args[3], planar);
    }

    Expression
        get_load_op(const BluesteinOperationType& type, const IntrinsicLoadPlanar& x, bool planar)
    {
        return get_intrinsic_load_op(type, x.args[2], x.args[3], x.args[4], planar);
    }

    Statement get_store_op(const BluesteinOperationType& type, const StoreGlobal& x, bool planar)
    {
        return get_store_op(type, x.index, x.value, planar);
    }

    Statement
        get_store_op(const BluesteinOperationType& type, const StoreGlobalPlanar& x, bool planar)
    {
        return get_store_op(type, x.index, x.value, planar);
    }

    Statement get_store_op(const BluesteinOperationType& type, const IntrinsicStore& x, bool planar)
    {
        return get_intrinsic_store_op(type, x.voffset, x.soffset, x.rw_flag, x.value, planar);
    }

    Statement
        get_store_op(const BluesteinOperationType& type, const IntrinsicStorePlanar& x, bool planar)
    {
        return get_intrinsic_store_op(type, x.voffset, x.soffset, x.rw_flag, x.value, planar);
    }

    std::string get_op_name(const BluesteinOperationType& type)
    {
        switch(type)
        {
        case BFN_LOAD_CC_FWD_CHIRP:
            return function_name[BFN_LOAD_CC_FWD_CHIRP];
        case BFN_LOAD_RC_FWD_CHIRP:
            return function_name[BFN_LOAD_RC_FWD_CHIRP];
        case BFN_LOAD_CC_FWD_CHIRP_MUL:
            return function_name[BFN_LOAD_CC_FWD_CHIRP_MUL];
        case BFN_LOAD_RC_FWD_CHIRP_MUL:
            return function_name[BFN_LOAD_RC_FWD_CHIRP_MUL];
        case BFN_LOAD_CC_INV_CHIRP_MUL:
            return function_name[BFN_LOAD_CC_INV_CHIRP_MUL];
        case BFN_LOAD_RC_INV_CHIRP_MUL:
            return function_name[BFN_LOAD_RC_INV_CHIRP_MUL];
        case BFN_STORE_CC_FWD_CHIRP:
            return function_name[BFN_STORE_CC_FWD_CHIRP];
        case BFN_STORE_RC_FWD_CHIRP:
            return function_name[BFN_STORE_RC_FWD_CHIRP];
        case BFN_STORE_CC_FWD_CHIRP_MUL:
            return function_name[BFN_STORE_CC_FWD_CHIRP_MUL];
        case BFN_STORE_RC_FWD_CHIRP_MUL:
            return function_name[BFN_STORE_RC_FWD_CHIRP_MUL];
        case BFN_STORE_CC_INV_CHIRP_MUL:
            return function_name[BFN_STORE_CC_INV_CHIRP_MUL];
        case BFN_STORE_RC_INV_CHIRP_MUL:
            return function_name[BFN_STORE_RC_INV_CHIRP_MUL];
        }
    }

    std::string get_intrinsic_op_name(const BluesteinOperationType& type)
    {
        switch(type)
        {
        case BFN_LOAD_CC_FWD_CHIRP:
            return intrinsic_function_name[BFN_LOAD_CC_FWD_CHIRP];
        case BFN_LOAD_RC_FWD_CHIRP:
            return intrinsic_function_name[BFN_LOAD_RC_FWD_CHIRP];
        case BFN_LOAD_CC_FWD_CHIRP_MUL:
            return intrinsic_function_name[BFN_LOAD_CC_FWD_CHIRP_MUL];
        case BFN_LOAD_RC_FWD_CHIRP_MUL:
            return intrinsic_function_name[BFN_LOAD_RC_FWD_CHIRP_MUL];
        case BFN_LOAD_CC_INV_CHIRP_MUL:
            return intrinsic_function_name[BFN_LOAD_CC_INV_CHIRP_MUL];
        case BFN_LOAD_RC_INV_CHIRP_MUL:
            return intrinsic_function_name[BFN_LOAD_RC_INV_CHIRP_MUL];
        case BFN_STORE_CC_FWD_CHIRP:
            return intrinsic_function_name[BFN_STORE_CC_FWD_CHIRP];
        case BFN_STORE_RC_FWD_CHIRP:
            return intrinsic_function_name[BFN_STORE_RC_FWD_CHIRP];
        case BFN_STORE_CC_FWD_CHIRP_MUL:
            return intrinsic_function_name[BFN_STORE_CC_FWD_CHIRP_MUL];
        case BFN_STORE_RC_FWD_CHIRP_MUL:
            return intrinsic_function_name[BFN_STORE_RC_FWD_CHIRP_MUL];
        case BFN_STORE_CC_INV_CHIRP_MUL:
            return intrinsic_function_name[BFN_STORE_CC_INV_CHIRP_MUL];
        case BFN_STORE_RC_INV_CHIRP_MUL:
            return intrinsic_function_name[BFN_STORE_RC_INV_CHIRP_MUL];
        }
    }

private:
    Expression get_load_op(const BluesteinOperationType& type, const Expression& index, bool planar)
    {
        std::unique_ptr<Expression> op;

        switch(type)
        {
        case BFN_LOAD_CC_FWD_CHIRP:
        {
            op = std::make_unique<Expression>(CallExpr{
                get_op_name(BFN_LOAD_CC_FWD_CHIRP) + render_template(),
                {data.chirp, data.global_transf_id, data.length_N_blue, data.length_M_blue}});
            break;
        }
        case BFN_LOAD_RC_FWD_CHIRP:
        {
            op = std::make_unique<Expression>(
                planar
                    ? CallExpr{get_op_name(BFN_LOAD_RC_FWD_CHIRP) + render_template(),
                               {data.global_transf_id,
                                data.buf_inre,
                                data.buf_inim,
                                data.load_cb_fn,
                                data.load_cb_data}}
                    : CallExpr{
                        get_op_name(BFN_LOAD_RC_FWD_CHIRP) + render_template(),
                        {data.global_transf_id, data.buf_in, data.load_cb_fn, data.load_cb_data}});
            break;
        }
        case BFN_LOAD_CC_FWD_CHIRP_MUL:
        {
            op = std::make_unique<Expression>(
                planar ? CallExpr{get_op_name(BFN_LOAD_CC_FWD_CHIRP_MUL) + render_template(),
                                  {data.chirp,
                                   data.global_transf_id,
                                   index,
                                   data.length_N_blue,
                                   data.buf_inre,
                                   data.buf_inim,
                                   data.load_cb_fn,
                                   data.load_cb_data}}
                       : CallExpr{get_op_name(BFN_LOAD_CC_FWD_CHIRP_MUL) + render_template(),
                                  {data.chirp,
                                   data.global_transf_id,
                                   index,
                                   data.length_N_blue,
                                   data.buf_in,
                                   data.load_cb_fn,
                                   data.load_cb_data}});
            break;
        }
        case BFN_LOAD_RC_FWD_CHIRP_MUL:
        {
            op = std::make_unique<Expression>(
                planar ? CallExpr{get_op_name(BFN_LOAD_RC_FWD_CHIRP_MUL) + render_template(),
                                  {data.global_data_id,
                                   data.buf_inre,
                                   data.buf_inim,
                                   data.load_cb_fn,
                                   data.load_cb_data}}
                       : CallExpr{
                           get_op_name(BFN_LOAD_RC_FWD_CHIRP_MUL) + render_template(),
                           {data.global_data_id, data.buf_in, data.load_cb_fn, data.load_cb_data}});
            break;
        }
        case BFN_LOAD_CC_INV_CHIRP_MUL:
        {
            op = std::make_unique<Expression>(
                planar ? CallExpr{get_op_name(BFN_LOAD_CC_INV_CHIRP_MUL) + render_template(),
                                  {data.global_transf_id,
                                   data.global_data_id,
                                   data.length_M_blue,
                                   data.buf_inre,
                                   data.buf_inim,
                                   data.load_cb_fn,
                                   data.load_cb_data}}
                       : CallExpr{get_op_name(BFN_LOAD_CC_INV_CHIRP_MUL) + render_template(),
                                  {data.global_transf_id,
                                   data.global_data_id,
                                   data.length_M_blue,
                                   data.buf_in,
                                   data.load_cb_fn,
                                   data.load_cb_data}});
            break;
        }
        case BFN_LOAD_RC_INV_CHIRP_MUL:
        {
            op = std::make_unique<Expression>(
                planar ? CallExpr{get_op_name(BFN_LOAD_RC_INV_CHIRP_MUL) + render_template(),
                                  {data.global_data_id,
                                   data.buf_inre,
                                   data.buf_inim,
                                   data.load_cb_fn,
                                   data.load_cb_data}}
                       : CallExpr{
                           get_op_name(BFN_LOAD_RC_INV_CHIRP_MUL) + render_template(),
                           {data.global_data_id, data.buf_in, data.load_cb_fn, data.load_cb_data}});
            break;
        }
        default:
            throw std::runtime_error("unsupported bluestein fuse operation");
        }

        return *op;
    }

    Expression get_intrinsic_load_op(const BluesteinOperationType& type,
                                     const Expression&             voffset,
                                     const Expression&             soffset,
                                     const Expression&             rw_flag,
                                     bool                          planar)
    {
        std::unique_ptr<Expression> op;

        switch(type)
        {
        case BFN_LOAD_CC_FWD_CHIRP:
        {
            op = std::make_unique<Expression>(
                CallExpr{get_intrinsic_op_name(BFN_LOAD_CC_FWD_CHIRP) + render_template(),
                         {data.chirp,
                          data.global_transf_id,
                          rw_flag,
                          data.length_N_blue,
                          data.length_M_blue}});
            break;
        }
        case BFN_LOAD_CC_FWD_CHIRP_MUL:
        {
            op = std::make_unique<Expression>(
                planar
                    ? CallExpr{get_intrinsic_op_name(BFN_LOAD_CC_FWD_CHIRP_MUL) + render_template(),
                               {data.chirp,
                                data.global_transf_id,
                                voffset,
                                soffset,
                                rw_flag,
                                data.length_N_blue,
                                data.buf_inre,
                                data.buf_inim,
                                data.load_cb_fn,
                                data.load_cb_data}}
                    : CallExpr{get_intrinsic_op_name(BFN_LOAD_CC_FWD_CHIRP_MUL) + render_template(),
                               {data.chirp,
                                data.global_transf_id,
                                voffset,
                                soffset,
                                rw_flag,
                                data.length_N_blue,
                                data.buf_in,
                                data.load_cb_fn,
                                data.load_cb_data}});
            break;
        }
        case BFN_LOAD_CC_INV_CHIRP_MUL:
        {
            op = std::make_unique<Expression>(
                planar
                    ? CallExpr{get_intrinsic_op_name(BFN_LOAD_CC_INV_CHIRP_MUL) + render_template(),
                               {data.global_transf_id,
                                data.global_data_id,
                                Literal{"0"},
                                rw_flag,
                                data.length_M_blue,
                                data.buf_inre,
                                data.buf_inim,
                                data.load_cb_fn,
                                data.load_cb_data}}
                    : CallExpr{get_intrinsic_op_name(BFN_LOAD_CC_INV_CHIRP_MUL) + render_template(),
                               {data.global_transf_id,
                                data.global_data_id,
                                Literal{"0"},
                                rw_flag,
                                data.length_M_blue,
                                data.buf_in,
                                data.load_cb_fn,
                                data.load_cb_data}});
            break;
        }
        default:
            throw std::runtime_error("unsupported bluestein fuse operation");
        }

        return *op;
    }

    Statement get_store_op(const BluesteinOperationType& type,
                           const Expression&             index,
                           const Expression&             value,
                           bool                          planar)
    {
        std::unique_ptr<Statement> op;

        switch(type)
        {
        case BFN_STORE_CC_FWD_CHIRP:
        {
            op = std::make_unique<Statement>(
                planar ? Call{get_op_name(BFN_STORE_CC_FWD_CHIRP) + render_template(),
                              {data.global_transf_id,
                               data.buf_outre,
                               data.buf_outim,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}}
                       : Call{get_op_name(BFN_STORE_CC_FWD_CHIRP) + render_template(),
                              {data.global_transf_id,
                               data.buf_out,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}});
            break;
        }
        case BFN_STORE_RC_FWD_CHIRP:
        {
            op = std::make_unique<Statement>(
                planar ? Call{get_op_name(BFN_STORE_RC_FWD_CHIRP) + render_template(),
                              {data.global_transf_id,
                               data.buf_outre,
                               data.buf_outim,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}}
                       : Call{get_op_name(BFN_STORE_RC_FWD_CHIRP) + render_template(),
                              {data.global_transf_id,
                               data.buf_out,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}});
            break;
        }
        case BFN_STORE_CC_FWD_CHIRP_MUL:
        {
            op = std::make_unique<Statement>(
                planar ? Call{get_op_name(BFN_STORE_CC_FWD_CHIRP_MUL) + render_template(),
                              {data.global_data_id,
                               data.buf_outre,
                               data.buf_outim,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}}
                       : Call{get_op_name(BFN_STORE_CC_FWD_CHIRP_MUL) + render_template(),
                              {data.global_data_id,
                               data.buf_out,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}});
            break;
        }
        case BFN_STORE_RC_FWD_CHIRP_MUL:
        {
            op = std::make_unique<Statement>(
                planar ? Call{get_op_name(BFN_STORE_RC_FWD_CHIRP_MUL) + render_template(),
                              {data.global_data_id,
                               data.length_M_blue,
                               data.buf_outre,
                               data.buf_outim,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}}
                       : Call{get_op_name(BFN_STORE_RC_FWD_CHIRP_MUL) + render_template(),
                              {data.global_data_id,
                               data.length_M_blue,
                               data.buf_out,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}});
            break;
        }
        case BFN_STORE_CC_INV_CHIRP_MUL:
        {
            op = std::make_unique<Statement>(
                planar ? Call{get_op_name(BFN_STORE_CC_INV_CHIRP_MUL) + render_template(),
                              {data.global_data_id,
                               data.buf_outre,
                               data.buf_outim,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}}
                       : Call{get_op_name(BFN_STORE_CC_INV_CHIRP_MUL) + render_template(),
                              {data.global_data_id,
                               data.buf_out,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}});
            break;
        }
        case BFN_STORE_RC_INV_CHIRP_MUL:
        {
            op = std::make_unique<Statement>(
                planar ? Call{get_op_name(BFN_STORE_RC_INV_CHIRP_MUL) + render_template(),
                              {data.chirp,
                               data.global_transf_id,
                               index,
                               data.length_N_blue,
                               data.length_M_blue,
                               data.buf_outre,
                               data.buf_outim,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}}
                       : Call{get_op_name(BFN_STORE_RC_INV_CHIRP_MUL) + render_template(),
                              {data.chirp,
                               data.global_transf_id,
                               index,
                               data.length_N_blue,
                               data.length_M_blue,
                               data.buf_out,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}});
            break;
        }
        default:
            throw std::runtime_error("unsupported bluestein fuse operation");
        }

        return *op;
    }

    Statement get_intrinsic_store_op(const BluesteinOperationType& type,
                                     const Expression&             voffset,
                                     const Expression&             soffset,
                                     const Expression&             rw_flag,
                                     const Expression&             value,
                                     bool                          planar)
    {
        std::unique_ptr<Statement> op;

        switch(type)
        {
        case BFN_STORE_CC_FWD_CHIRP:
        {
            op = std::make_unique<Statement>(
                planar ? Call{get_intrinsic_op_name(BFN_STORE_CC_FWD_CHIRP) + render_template(),
                              {data.global_transf_id,
                               Literal{"0"},
                               rw_flag,
                               data.buf_outre,
                               data.buf_outim,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}}
                       : Call{get_intrinsic_op_name(BFN_STORE_CC_FWD_CHIRP) + render_template(),
                              {data.global_transf_id,
                               Literal{"0"},
                               rw_flag,
                               data.buf_out,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}});
            break;
        }
        case BFN_STORE_CC_FWD_CHIRP_MUL:
        {
            op = std::make_unique<Statement>(
                planar ? Call{get_intrinsic_op_name(BFN_STORE_CC_FWD_CHIRP_MUL) + render_template(),
                              {data.global_data_id,
                               Literal{"0"},
                               rw_flag,
                               data.buf_outre,
                               data.buf_outim,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}}
                       : Call{get_intrinsic_op_name(BFN_STORE_CC_FWD_CHIRP_MUL) + render_template(),
                              {data.global_data_id,
                               Literal{"0"},
                               rw_flag,
                               data.buf_out,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}});
            break;
        }
        case BFN_STORE_CC_INV_CHIRP_MUL:
        {
            op = std::make_unique<Statement>(
                planar ? Call{get_intrinsic_op_name(BFN_STORE_CC_INV_CHIRP_MUL) + render_template(),
                              {data.global_data_id,
                               Literal{"0"},
                               rw_flag,
                               data.buf_outre,
                               data.buf_outim,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}}
                       : Call{get_intrinsic_op_name(BFN_STORE_CC_INV_CHIRP_MUL) + render_template(),
                              {data.global_data_id,
                               Literal{"0"},
                               rw_flag,
                               data.buf_out,
                               value,
                               data.store_cb_fn,
                               data.store_cb_data}});
            break;
        }
        default:
            throw std::runtime_error("unsupported bluestein fuse operation");
        }

        return *op;
    }

    std::string render_template()
    {
        return "<" + data.scalar_type.render() + ", " + data.callback_type.render() + ">";
    }

    const std::vector<std::string> function_name = {"bluestein_load_cc_fwd_chirp_device",
                                                    "bluestein_load_rc_fwd_chirp_device",
                                                    "bluestein_load_cc_fwd_chirp_mul_device",
                                                    "bluestein_load_rc_fwd_chirp_mul_device",
                                                    "bluestein_load_cc_inv_chirp_mul_device",
                                                    "bluestein_load_rc_inv_chirp_mul_device",
                                                    "bluestein_store_cc_fwd_chirp_device",
                                                    "bluestein_store_rc_fwd_chirp_device",
                                                    "bluestein_store_cc_fwd_chirp_mul_device",
                                                    "bluestein_store_rc_fwd_chirp_mul_device",
                                                    "bluestein_store_cc_inv_chirp_mul_device",
                                                    "bluestein_store_rc_inv_chirp_mul_device"};

    const std::vector<std::string> intrinsic_function_name
        = {"bluestein_intrinsic_load_cc_fwd_chirp_device",
           "bluestein_intrinsic_load_rc_fwd_chirp_device",
           "bluestein_intrinsic_load_cc_fwd_chirp_mul_device",
           "bluestein_intrinsic_load_rc_fwd_chirp_mul_device",
           "bluestein_intrinsic_load_cc_inv_chirp_mul_device",
           "bluestein_intrinsic_load_rc_inv_chirp_mul_device",
           "bluestein_intrinsic_store_cc_fwd_chirp_device",
           "bluestein_intrinsic_store_rc_fwd_chirp_device",
           "bluestein_intrinsic_store_cc_fwd_chirp_mul_device",
           "bluestein_intrinsic_store_rc_fwd_chirp_mul_device",
           "bluestein_intrinsic_store_cc_inv_chirp_mul_device",
           "bluestein_intrinsic_store_rc_inv_chirp_mul_device"};

    BluesteinData data;
};

class BluesteinKernel
{
public:
    BluesteinKernel(ComputeScheme     scheme,
                    BluesteinFuseType type,
                    int               direction,
                    bool              planar_load,
                    bool              planar_store,
                    bool              intrinsic)
        : scheme(scheme)
        , type(type)
        , direction(direction)
        , planar_load(planar_load)
        , planar_store(planar_store)
        , intrinsic(intrinsic)
    {
    }

    Function generate_device_load_function()
    {
        switch(type)
        {
        case BFT_NONE:
            break;
        case BFT_FWD_CHIRP:
            if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
                return generate_fwd_chirp_load_cc();
            if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
                return generate_fwd_chirp_load_rc();
            break;
        case BFT_FWD_CHIRP_MUL:
            if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
                return generate_fwd_chirp_mul_load_cc();
            if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
                return generate_fwd_chirp_mul_load_rc();
            break;
        case BFT_INV_CHIRP_MUL:
            if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
                return generate_inv_chirp_mul_load_cc();
            if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
                return generate_inv_chirp_mul_load_rc();
            break;
        }

        throw std::runtime_error("unsupported bluestein fuse scheme");
    }

    Function generate_device_store_function()
    {
        switch(type)
        {
        case BFT_NONE:
            break;
        case BFT_FWD_CHIRP:
            if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
                return generate_fwd_chirp_store_cc();
            if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
                return generate_fwd_chirp_store_rc();
            break;
        case BFT_FWD_CHIRP_MUL:
            if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
                return generate_fwd_chirp_mul_store_cc();
            if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
                return generate_fwd_chirp_mul_store_rc();
            break;
        case BFT_INV_CHIRP_MUL:
            if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
                return generate_inv_chirp_mul_store_cc();
            if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
                return generate_inv_chirp_mul_store_rc();
            break;
        }

        throw std::runtime_error("unsupported bluestein fuse scheme");
    }

private:
    TemplateList get_template_list()
    {
        TemplateList tpls;
        tpls.append(blueData.scalar_type);
        tpls.append(blueData.callback_type);

        return tpls;
    }

    void append_data_buf(ArgumentList& args, bool planar)
    {
        if(planar)
        {
            args.append(blueData.data_bufre);
            args.append(blueData.data_bufim);
        }
        else
        {
            args.append(blueData.data_buf);
        }
    }

    void append_data_index(ArgumentList& args, bool intrinsic)
    {
        if(intrinsic)
        {
            args.append(blueData.data_voffset);
            args.append(blueData.data_soffset);
            args.append(blueData.data_rw_flag);
        }
        else
        {
            args.append(blueData.data_idx);
        }
    }

    std::unique_ptr<Expression> get_load_expression()
    {
        if(planar_load)
        {
            if(intrinsic)
                return std::make_unique<Expression>(IntrinsicLoadPlanar({
                    blueData.data_bufre,
                    blueData.data_bufim,
                    blueData.data_voffset,
                    blueData.data_soffset,
                    blueData.data_rw_flag,
                }));
            else
                return std::make_unique<Expression>(LoadGlobalPlanar({
                    blueData.data_bufre,
                    blueData.data_bufim,
                    blueData.data_idx,
                }));
        }
        else
        {
            if(intrinsic)
                return std::make_unique<Expression>(IntrinsicLoad({
                    blueData.data_buf,
                    blueData.data_voffset,
                    blueData.data_soffset,
                    blueData.data_rw_flag,
                }));
            else
                return std::make_unique<Expression>(LoadGlobal{
                    blueData.data_buf,
                    blueData.data_idx,
                });
        }
    }

    std::unique_ptr<Expression> get_load_expression(const Expression& index)
    {
        if(planar_load)
        {
            if(intrinsic)
                return std::make_unique<Expression>(IntrinsicLoadPlanar({
                    blueData.data_bufre,
                    blueData.data_bufim,
                    index,
                    0,
                    blueData.data_rw_flag,
                }));
            else
                return std::make_unique<Expression>(LoadGlobalPlanar({
                    blueData.data_bufre,
                    blueData.data_bufim,
                    index,
                }));
        }
        else
        {
            if(intrinsic)
                return std::make_unique<Expression>(IntrinsicLoad({
                    blueData.data_buf,
                    index,
                    0,
                    blueData.data_rw_flag,
                }));
            else
                return std::make_unique<Expression>(LoadGlobal{
                    blueData.data_buf,
                    index,
                });
        }
    }

    std::unique_ptr<Statement> get_store_statement()
    {
        if(planar_store)
        {
            if(intrinsic)
            {
                return std::make_unique<Statement>(IntrinsicStorePlanar(blueData.data_bufre,
                                                                        blueData.data_bufim,
                                                                        blueData.data_voffset,
                                                                        blueData.data_soffset,
                                                                        blueData.data_elem,
                                                                        blueData.data_rw_flag));
            }
            else
            {
                return std::make_unique<Statement>(StoreGlobalPlanar(blueData.data_bufre,
                                                                     blueData.data_bufim,
                                                                     blueData.data_idx,
                                                                     blueData.data_elem));
            }
        }
        else
        {
            if(intrinsic)
                return std::make_unique<Statement>(IntrinsicStore(blueData.data_buf,
                                                                  blueData.data_voffset,
                                                                  blueData.data_soffset,
                                                                  blueData.data_elem,
                                                                  blueData.data_rw_flag));
            else
                return std::make_unique<Statement>(
                    StoreGlobal(blueData.data_buf, blueData.data_idx, blueData.data_elem));
        }
    }

    std::unique_ptr<Statement> get_store_statement(const Expression& index)
    {
        if(planar_store)
        {
            if(intrinsic)
            {
                return std::make_unique<Statement>(IntrinsicStorePlanar(blueData.data_bufre,
                                                                        blueData.data_bufim,
                                                                        index,
                                                                        0,
                                                                        blueData.data_elem,
                                                                        blueData.data_rw_flag));
            }
            else
            {
                return std::make_unique<Statement>(StoreGlobalPlanar(
                    blueData.data_bufre, blueData.data_bufim, index, blueData.data_elem));
            }
        }
        else
        {
            if(intrinsic)
                return std::make_unique<Statement>(IntrinsicStore(
                    blueData.data_buf, index, 0, blueData.data_elem, blueData.data_rw_flag));
            else
                return std::make_unique<Statement>(
                    StoreGlobal(blueData.data_buf, index, blueData.data_elem));
        }
    }

    Function generate_fwd_chirp_load_cc()
    {
        Function f{intrinsic ? function.get_intrinsic_op_name(BFN_LOAD_CC_FWD_CHIRP)
                             : function.get_op_name(BFN_LOAD_CC_FWD_CHIRP)};

        f.templates = get_template_list();
        ArgumentList args;
        args.append(blueData.chirp);
        args.append(blueData.transform_idx);
        if(intrinsic)
            args.append(blueData.data_rw_flag);
        args.append(blueData.length_N_blue);
        args.append(blueData.length_M_blue);
        f.arguments   = args;
        f.return_type = "scalar_type";
        f.qualifier   = "__device__";

        StatementList& body = f.body;
        if(intrinsic)
        {
            body += If{(blueData.transform_idx < blueData.length_N_blue) && blueData.data_rw_flag,
                       {
                           ReturnExpr(blueData.chirp[blueData.transform_idx]),
                       }};
            body += ElseIf{
                (blueData.transform_idx >= blueData.length_M_blue - blueData.length_N_blue + 1)
                    && blueData.data_rw_flag,
                {
                    ReturnExpr(blueData.chirp[blueData.length_M_blue - blueData.transform_idx]),
                }};
            body += Else{{
                ReturnExpr(CallExpr{"scalar_type", {0, 0}}),
            }};
        }
        else
        {
            body += If{blueData.transform_idx < blueData.length_N_blue,
                       {
                           ReturnExpr(blueData.chirp[blueData.transform_idx]),
                       }};
            body += ElseIf{
                blueData.transform_idx >= blueData.length_M_blue - blueData.length_N_blue + 1,
                {
                    ReturnExpr(blueData.chirp[blueData.length_M_blue - blueData.transform_idx]),
                }};
            body += Else{{
                ReturnExpr(CallExpr{"scalar_type", {0, 0}}),
            }};
        }

        return f;
    }

    Function generate_fwd_chirp_load_rc()
    {
        Function f{intrinsic ? function.get_intrinsic_op_name(BFN_LOAD_RC_FWD_CHIRP)
                             : function.get_op_name(BFN_LOAD_RC_FWD_CHIRP)};

        f.templates = get_template_list();
        ArgumentList args;
        append_data_index(args, intrinsic);
        append_data_buf(args, planar_load);
        args.append(blueData.load_cb_fn);
        args.append(blueData.load_cb_data);
        f.arguments   = args;
        f.return_type = "scalar_type";
        f.qualifier   = "__device__";

        auto load_expression = get_load_expression();

        StatementList& body = f.body;
        body += CallbackLoadDeclaration{blueData.scalar_type.render(),
                                        blueData.callback_type.render()};
        body += ReturnExpr{*load_expression};

        return f;
    }

    Function generate_fwd_chirp_mul_load_cc()
    {
        Function f{intrinsic ? function.get_intrinsic_op_name(BFN_LOAD_CC_FWD_CHIRP_MUL)
                             : function.get_op_name(BFN_LOAD_CC_FWD_CHIRP_MUL)};

        f.templates = get_template_list();
        ArgumentList args;
        args.append(blueData.chirp);
        args.append(blueData.transform_idx);
        append_data_index(args, intrinsic);
        args.append(blueData.length_N_blue);
        append_data_buf(args, planar_load);
        args.append(blueData.load_cb_fn);
        args.append(blueData.load_cb_data);
        f.arguments   = args;
        f.return_type = "scalar_type";
        f.qualifier   = "__device__";

        Variable elem_scalar{"elem_scalar", "scalar_type"};
        Variable aux_real{"aux_real", "real_type_t<scalar_type>"};

        auto load_expression = get_load_expression();

        std::unique_ptr<Expression> mul_assign_expression_x, mul_assign_expression_y;
        if(direction == -1) // forward
        {
            mul_assign_expression_x = std::make_unique<Expression>(
                elem_scalar.x() * blueData.chirp[blueData.transform_idx].x()
                + elem_scalar.y() * blueData.chirp[blueData.transform_idx].y());
            mul_assign_expression_y = std::make_unique<Expression>(
                -aux_real * blueData.chirp[blueData.transform_idx].y()
                + elem_scalar.y() * blueData.chirp[blueData.transform_idx].x());
        }
        else if(direction == +1) // inverse
        {
            mul_assign_expression_x = std::make_unique<Expression>(
                elem_scalar.x() * blueData.chirp[blueData.transform_idx].x()
                - elem_scalar.y() * blueData.chirp[blueData.transform_idx].y());
            mul_assign_expression_y = std::make_unique<Expression>(
                -aux_real * blueData.chirp[blueData.transform_idx].y()
                - elem_scalar.y() * blueData.chirp[blueData.transform_idx].x());
        }

        StatementList& body = f.body;
        body += CallbackLoadDeclaration{blueData.scalar_type.render(),
                                        blueData.callback_type.render()};
        body += If{blueData.transform_idx >= blueData.length_N_blue,
                   {
                       Call{"return", {CallExpr{"scalar_type", {0, 0}}}},
                   }};
        body += Else{{
            Declaration{elem_scalar},
            Declaration{aux_real},
            Assign{elem_scalar, *load_expression},
            Assign{aux_real, elem_scalar.x()},
            Assign{elem_scalar.x(), *mul_assign_expression_x},
            Assign{elem_scalar.y(), *mul_assign_expression_y},
            ReturnExpr(elem_scalar),
        }};

        return f;
    }

    Function generate_fwd_chirp_mul_load_rc()
    {
        Function f{intrinsic ? function.get_intrinsic_op_name(BFN_LOAD_RC_FWD_CHIRP_MUL)
                             : function.get_op_name(BFN_LOAD_RC_FWD_CHIRP_MUL)};

        f.templates = get_template_list();
        ArgumentList args;
        append_data_index(args, intrinsic);
        append_data_buf(args, planar_load);
        args.append(blueData.load_cb_fn);
        args.append(blueData.load_cb_data);
        f.arguments   = args;
        f.return_type = "scalar_type";
        f.qualifier   = "__device__";

        auto load_expression = get_load_expression();

        StatementList& body = f.body;
        body += CallbackLoadDeclaration{blueData.scalar_type.render(),
                                        blueData.callback_type.render()};
        body += ReturnExpr(*load_expression);

        return f;
    }

    Function generate_inv_chirp_mul_load_cc()
    {
        Function f{intrinsic ? function.get_intrinsic_op_name(BFN_LOAD_CC_INV_CHIRP_MUL)
                             : function.get_op_name(BFN_LOAD_CC_INV_CHIRP_MUL)};

        f.templates = get_template_list();
        ArgumentList args;
        args.append(blueData.transform_idx);
        append_data_index(args, intrinsic);
        args.append(blueData.length_M_blue);
        append_data_buf(args, planar_load);
        args.append(blueData.load_cb_fn);
        args.append(blueData.load_cb_data);
        f.arguments   = args;
        f.return_type = "scalar_type";
        f.qualifier   = "__device__";

        Variable aux_scalar{"aux_scalar", "scalar_type"};
        Variable elem_scalar{"elem_scalar", "scalar_type"};
        Variable aux_real{"aux_real", "real_type_t<scalar_type>"};

        auto load_expression_1 = get_load_expression(blueData.transform_idx);

        std::unique_ptr<Expression> load_expression_2;
        if(intrinsic)
            load_expression_2 = get_load_expression(blueData.data_voffset + blueData.data_soffset
                                                    + blueData.length_M_blue);
        else
            load_expression_2 = get_load_expression(blueData.data_idx + blueData.length_M_blue);

        StatementList& body = f.body;
        body += CallbackLoadDeclaration{blueData.scalar_type.render(),
                                        blueData.callback_type.render()};
        body += Declaration{elem_scalar};
        body += Declaration{aux_scalar};
        body += Declaration{aux_real};
        body += Assign{elem_scalar, *load_expression_1};
        body += Assign{aux_scalar, *load_expression_2};
        body += Assign{aux_real, elem_scalar.x()};
        body += Assign{elem_scalar.x(),
                       elem_scalar.x() * aux_scalar.x() - elem_scalar.y() * aux_scalar.y()};
        body += Assign{elem_scalar.y(),
                       aux_real * aux_scalar.y() + elem_scalar.y() * aux_scalar.x()};

        body += ReturnExpr(elem_scalar);

        return f;
    }

    Function generate_inv_chirp_mul_load_rc()
    {
        Function f{intrinsic ? function.get_intrinsic_op_name(BFN_LOAD_RC_INV_CHIRP_MUL)
                             : function.get_op_name(BFN_LOAD_RC_INV_CHIRP_MUL)};

        f.templates = get_template_list();
        ArgumentList args;
        append_data_index(args, intrinsic);
        append_data_buf(args, planar_load);
        args.append(blueData.load_cb_fn);
        args.append(blueData.load_cb_data);
        f.arguments   = args;
        f.return_type = "scalar_type";
        f.qualifier   = "__device__";

        auto load_expression = get_load_expression();

        StatementList& body = f.body;
        body += CallbackLoadDeclaration{blueData.scalar_type.render(),
                                        blueData.callback_type.render()};
        body += ReturnExpr(*load_expression);

        return f;
    }

    Function generate_fwd_chirp_store_cc()
    {
        Function f{intrinsic ? function.get_intrinsic_op_name(BFN_STORE_CC_FWD_CHIRP)
                             : function.get_op_name(BFN_STORE_CC_FWD_CHIRP)};

        f.templates = get_template_list();
        ArgumentList args;
        append_data_index(args, intrinsic);
        append_data_buf(args, planar_store);
        args.append(blueData.data_elem);
        args.append(blueData.store_cb_fn);
        args.append(blueData.store_cb_data);
        f.arguments = args;
        f.qualifier = "__device__";

        auto store_statement = get_store_statement();

        StatementList& body = f.body;
        body += CallbackStoreDeclaration{blueData.scalar_type.render(),
                                         blueData.callback_type.render()};
        body += *store_statement;

        return f;
    }

    Function generate_fwd_chirp_store_rc()
    {
        Function f{intrinsic ? function.get_intrinsic_op_name(BFN_STORE_RC_FWD_CHIRP)
                             : function.get_op_name(BFN_STORE_RC_FWD_CHIRP)};

        f.templates = get_template_list();
        ArgumentList args;
        append_data_index(args, intrinsic);
        append_data_buf(args, planar_store);
        args.append(blueData.data_elem);
        args.append(blueData.store_cb_fn);
        args.append(blueData.store_cb_data);
        f.arguments = args;
        f.qualifier = "__device__";

        auto store_statement = get_store_statement();

        StatementList& body = f.body;
        body += CallbackStoreDeclaration{blueData.scalar_type.render(),
                                         blueData.callback_type.render()};
        body += *store_statement;

        return f;
    }

    Function generate_fwd_chirp_mul_store_cc()
    {
        Function f{intrinsic ? function.get_intrinsic_op_name(BFN_STORE_CC_FWD_CHIRP_MUL)
                             : function.get_op_name(BFN_STORE_CC_FWD_CHIRP_MUL)};

        f.templates = get_template_list();
        ArgumentList args;
        append_data_index(args, intrinsic);
        append_data_buf(args, planar_store);
        args.append(blueData.data_elem);
        args.append(blueData.store_cb_fn);
        args.append(blueData.store_cb_data);
        f.arguments = args;
        f.qualifier = "__device__";

        auto store_statement = get_store_statement();

        StatementList& body = f.body;
        body += CallbackStoreDeclaration{blueData.scalar_type.render(),
                                         blueData.callback_type.render()};
        body += *store_statement;

        return f;
    }

    Function generate_fwd_chirp_mul_store_rc()
    {
        Function f{intrinsic ? function.get_intrinsic_op_name(BFN_STORE_RC_FWD_CHIRP_MUL)
                             : function.get_op_name(BFN_STORE_RC_FWD_CHIRP_MUL)};

        f.templates = get_template_list();
        ArgumentList args;
        append_data_index(args, intrinsic);
        args.append(blueData.length_M_blue);
        append_data_buf(args, planar_store);
        args.append(blueData.data_elem);
        args.append(blueData.store_cb_fn);
        args.append(blueData.store_cb_data);
        f.arguments = args;
        f.qualifier = "__device__";

        std::unique_ptr<Statement> store_statement;
        if(intrinsic)
            store_statement = get_store_statement(blueData.data_voffset + blueData.data_soffset
                                                  + blueData.length_M_blue);
        else
            store_statement = get_store_statement(blueData.data_idx + blueData.length_M_blue);

        StatementList& body = f.body;
        body += CallbackStoreDeclaration{blueData.scalar_type.render(),
                                         blueData.callback_type.render()};
        body += *store_statement;

        return f;
    }

    Function generate_inv_chirp_mul_store_cc()
    {
        Function f{intrinsic ? function.get_intrinsic_op_name(BFN_STORE_CC_INV_CHIRP_MUL)
                             : function.get_op_name(BFN_STORE_CC_INV_CHIRP_MUL)};

        f.templates = get_template_list();
        ArgumentList args;
        append_data_index(args, intrinsic);
        append_data_buf(args, planar_store);
        args.append(blueData.data_elem);
        args.append(blueData.store_cb_fn);
        args.append(blueData.store_cb_data);
        f.arguments = args;
        f.qualifier = "__device__";

        auto store_statement = get_store_statement();

        StatementList& body = f.body;
        body += CallbackStoreDeclaration{blueData.scalar_type.render(),
                                         blueData.callback_type.render()};
        body += *store_statement;

        return f;
    }

    Function generate_inv_chirp_mul_store_rc()
    {
        Function f{intrinsic ? function.get_intrinsic_op_name(BFN_STORE_RC_INV_CHIRP_MUL)
                             : function.get_op_name(BFN_STORE_RC_INV_CHIRP_MUL)};

        f.templates = get_template_list();
        ArgumentList args;
        args.append(blueData.chirp);
        args.append(blueData.transform_idx);
        append_data_index(args, intrinsic);
        args.append(blueData.length_N_blue);
        args.append(blueData.length_M_blue);
        append_data_buf(args, planar_store);
        args.append(blueData.data_elem);
        args.append(blueData.store_cb_fn);
        args.append(blueData.store_cb_data);
        f.arguments = args;
        f.qualifier = "__device__";

        Variable aux_real{"aux_real", "real_type_t<scalar_type>"};

        auto store_statement = get_store_statement();

        std::unique_ptr<Expression> mul_assign_expression;
        if(direction == +1) // inverse
            mul_assign_expression = std::make_unique<Expression>(
                -aux_real * blueData.chirp[blueData.transform_idx].y()
                + blueData.data_elem.y() * blueData.chirp[blueData.transform_idx].x());
        else if(direction == -1) // forward
            mul_assign_expression = std::make_unique<Expression>(
                aux_real * blueData.chirp[blueData.transform_idx].y()
                - blueData.data_elem.y() * blueData.chirp[blueData.transform_idx].x());

        StatementList& body = f.body;
        body += CallbackStoreDeclaration{blueData.scalar_type.render(),
                                         blueData.callback_type.render()};
        body += If{
            blueData.transform_idx < blueData.length_N_blue,
            {
                Assign{blueData.data_elem,
                       blueData.data_elem
                           * Parens{Literal{"1.0 / (real_type_t<scalar_type>) "
                                            + blueData.length_M_blue.render()}}},
                Declaration{aux_real},
                Assign{aux_real, blueData.data_elem.x()},
                Assign{blueData.data_elem.x(),
                       blueData.data_elem.x() * blueData.chirp[blueData.transform_idx].x()
                           + blueData.data_elem.y() * blueData.chirp[blueData.transform_idx].y()},
                Assign{blueData.data_elem.y(), *mul_assign_expression},
                *store_statement,
            }};

        return f;
    }

    ComputeScheme     scheme;
    BluesteinFuseType type;
    BluesteinData     blueData;
    int               direction;
    bool              planar_load;
    bool              planar_store;
    bool              intrinsic;
    BluesteinFunction function;
};

static Function generate_bluestein_device_load_function(const ComputeScheme     scheme,
                                                        const BluesteinFuseType type,
                                                        int                     direction,
                                                        bool                    planar,
                                                        bool                    intrinsic)
{
    auto blueKernel = BluesteinKernel(scheme, type, direction, planar, false, intrinsic);
    return blueKernel.generate_device_load_function();
}

static Function generate_bluestein_device_store_function(const ComputeScheme     scheme,
                                                         const BluesteinFuseType type,
                                                         int                     direction,
                                                         bool                    planar,
                                                         bool                    intrinsic)
{
    auto blueKernel = BluesteinKernel(scheme, type, direction, false, planar, intrinsic);
    return blueKernel.generate_device_store_function();
}

struct MakeBluesteinVisitor : public BaseVisitor
{
    MakeBluesteinVisitor()
        : BaseVisitor()
    {
    }

    Function visit_Function(const Function& x) override
    {
        Function y{x};

        y.arguments.append(blueData.length_N_blue);
        y.arguments.append(blueData.length_M_blue);
        y.arguments.append(blueData.global_stride_in_0);
        y.arguments.append(blueData.global_stride_in_1);
        y.arguments.append(blueData.global_idist);
        y.arguments.append(blueData.global_stride_out_0);
        y.arguments.append(blueData.global_stride_out_1);
        y.arguments.append(blueData.global_odist);

        return BaseVisitor::visit_Function(y);
    }

    BluesteinData          blueData;
    BluesteinFunction      blueFunction;
    BluesteinOperationType load_op;
    BluesteinOperationType store_op;
};

struct MakeBluesteinCCVisitor : public MakeBluesteinVisitor
{
    MakeBluesteinCCVisitor()
        : MakeBluesteinVisitor()
    {
    }

    Expression visit_LoadGlobal(const LoadGlobal& x) override
    {
        return blueFunction.get_load_op(load_op, x, false);
    }

    Expression visit_LoadGlobalPlanar(const LoadGlobalPlanar& x) override
    {
        return blueFunction.get_load_op(load_op, x, true);
    }

    Expression visit_IntrinsicLoad(const IntrinsicLoad& x) override
    {
        return blueFunction.get_load_op(load_op, x, false);
    }

    Expression visit_IntrinsicLoadPlanar(const IntrinsicLoadPlanar& x) override
    {
        return blueFunction.get_load_op(load_op, x, true);
    }

    StatementList visit_StoreGlobal(const StoreGlobal& x) override
    {
        auto stmts = StatementList();
        stmts += blueFunction.get_store_op(store_op, x, false);
        return stmts;
    }

    StatementList visit_StoreGlobalPlanar(const StoreGlobalPlanar& x) override
    {
        auto stmts = StatementList();
        stmts += blueFunction.get_store_op(store_op, x, true);
        return stmts;
    }

    StatementList visit_IntrinsicStore(const IntrinsicStore& x) override
    {
        auto stmts = StatementList();
        stmts += blueFunction.get_store_op(store_op, x, false);
        return stmts;
    }

    StatementList visit_IntrinsicStorePlanar(const IntrinsicStorePlanar& x) override
    {
        auto stmts = StatementList();
        stmts += blueFunction.get_store_op(store_op, x, true);
        return stmts;
    }
};

struct MakeBluesteinRCVisitor : public MakeBluesteinVisitor
{
    MakeBluesteinRCVisitor()
        : MakeBluesteinVisitor()
    {
    }

    Function visit_Function(const Function& x) override
    {
        Function y{x};

        return MakeBluesteinVisitor::visit_Function(y);
    }

    Expression visit_LoadGlobal(const LoadGlobal& x) override
    {
        return blueFunction.get_load_op(load_op, x, false);
    }

    Expression visit_LoadGlobalPlanar(const LoadGlobalPlanar& x) override
    {
        return blueFunction.get_load_op(load_op, x, true);
    }

    StatementList visit_StoreGlobal(const StoreGlobal& x) override
    {
        auto stmts = StatementList();
        stmts += blueFunction.get_store_op(store_op, x, false);
        return stmts;
    }

    StatementList visit_StoreGlobalPlanar(const StoreGlobalPlanar& x) override
    {
        auto stmts = StatementList();
        stmts += blueFunction.get_store_op(store_op, x, true);
        return stmts;
    }
};

struct MakeBluesteinFwdChirpCCVisitor : public MakeBluesteinCCVisitor
{
    MakeBluesteinFwdChirpCCVisitor()
        : MakeBluesteinCCVisitor()
    {
        load_op  = BFN_LOAD_CC_FWD_CHIRP;
        store_op = BFN_STORE_CC_FWD_CHIRP;
    }

    Function visit_Function(const Function& x) override
    {
        Function y{x};
        y.arguments.append(blueData.chirp);

        return MakeBluesteinCCVisitor::visit_Function(y);
    }
};

Function make_bluestein_fwd_chirp_cc(const Function& f)
{
    auto visitor = MakeBluesteinFwdChirpCCVisitor();
    return visitor(f);
}

struct MakeBluesteinFwdChirpRCVisitor : public MakeBluesteinRCVisitor
{
    MakeBluesteinFwdChirpRCVisitor()
        : MakeBluesteinRCVisitor()
    {
        load_op  = BFN_LOAD_RC_FWD_CHIRP;
        store_op = BFN_STORE_RC_FWD_CHIRP;
    }
};

Function make_bluestein_fwd_chirp_rc(const Function& f)
{
    auto visitor = MakeBluesteinFwdChirpRCVisitor();
    return visitor(f);
}

struct MakeBluesteinFwdChirpMulCCVisitor : public MakeBluesteinCCVisitor
{
    MakeBluesteinFwdChirpMulCCVisitor()
        : MakeBluesteinCCVisitor()
    {
        load_op  = BFN_LOAD_CC_FWD_CHIRP_MUL;
        store_op = BFN_STORE_CC_FWD_CHIRP_MUL;
    }

    Function visit_Function(const Function& x) override
    {
        Function y{x};
        y.arguments.append(blueData.chirp);

        return MakeBluesteinCCVisitor::visit_Function(y);
    }
};

Function make_bluestein_fwd_chirp_mul_cc(const Function& f)
{
    auto visitor = MakeBluesteinFwdChirpMulCCVisitor();
    return visitor(f);
}

struct MakeBluesteinFwdChirpMulRCVisitor : public MakeBluesteinRCVisitor
{
    MakeBluesteinFwdChirpMulRCVisitor()
        : MakeBluesteinRCVisitor()
    {
        load_op  = BFN_LOAD_RC_FWD_CHIRP_MUL;
        store_op = BFN_STORE_RC_FWD_CHIRP_MUL;
    }
};

Function make_bluestein_fwd_chirp_mul_rc(const Function& f)
{
    auto visitor = MakeBluesteinFwdChirpMulRCVisitor();
    return visitor(f);
}

struct MakeBluesteinInvChirpMulCCVisitor : public MakeBluesteinCCVisitor
{
    MakeBluesteinInvChirpMulCCVisitor()
        : MakeBluesteinCCVisitor()
    {
        load_op  = BFN_LOAD_CC_INV_CHIRP_MUL;
        store_op = BFN_STORE_CC_INV_CHIRP_MUL;
    }
};

Function make_bluestein_inv_chirp_mul_cc(const Function& f)
{
    auto visitor = MakeBluesteinInvChirpMulCCVisitor();
    return visitor(f);
}

struct MakeBluesteinInvChirpMulRCVisitor : public MakeBluesteinRCVisitor
{
    MakeBluesteinInvChirpMulRCVisitor()
        : MakeBluesteinRCVisitor()
    {
        load_op  = BFN_LOAD_RC_INV_CHIRP_MUL;
        store_op = BFN_STORE_RC_INV_CHIRP_MUL;
    }

    Function visit_Function(const Function& x) override
    {
        Function y{x};
        y.arguments.append(blueData.chirp);

        return MakeBluesteinRCVisitor::visit_Function(y);
    }
};

Function make_bluestein_inv_chirp_mul_rc(const Function& f)
{
    auto visitor = MakeBluesteinInvChirpMulRCVisitor();
    return visitor(f);
}

static Function
    make_bluestein(const ComputeScheme scheme, const BluesteinFuseType type, const Function& f)
{
    switch(type)
    {
    case BFT_NONE:
        break;
    case BFT_FWD_CHIRP:
        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
            return make_bluestein_fwd_chirp_cc(f);

        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
            return make_bluestein_fwd_chirp_rc(f);

        break;
    case BFT_FWD_CHIRP_MUL:
        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
            return make_bluestein_fwd_chirp_mul_cc(f);

        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
            return make_bluestein_fwd_chirp_mul_rc(f);

        break;
    case BFT_INV_CHIRP_MUL:
        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
            return make_bluestein_inv_chirp_mul_cc(f);

        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
            return make_bluestein_inv_chirp_mul_rc(f);

        break;
    }

    throw std::runtime_error("unsupported bluestein fuse scheme");
}
