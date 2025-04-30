// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

//
// High level FFT AST
//
// Here we express FFT algorithms as ASTs, using "FFT operators" as
// building blocks.
//
// Each "FFT operator" node in a tree knows how to "lower" itself into
// a StatementList (ie, our HIP AST).
//

#pragma once
#include "../../../../shared/arithmetic.h"
#include "generator.h"

#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <variant>

#ifdef WIN32
#include <sys/types.h>
#endif

//
// Helpers
//

template <typename T>
StatementList vlower(const T& x)
{
    return std::visit([](const auto a) { return a.lower(); }, x);
}

enum class Guard
{
    NONE,
    WRITE,
    THREAD,
    BOTH,
};

//
// Context
//

struct Context
{
    void add_local(Variable const& v)
    {
        if(locals_names.count(v.name) == 0)
        {
            locals_names.insert(v.name);
            locals.push_back(v);
        }
    }

    void add_argument(Variable const& v)
    {
        if(arguments_names.count(v.name) == 0)
        {
            arguments_names.insert(v.name);
            arguments.push_back(v);
        }
    }

    std::vector<Variable> get_locals()
    {
        return locals;
    }

    std::vector<Variable> get_arguments()
    {
        return arguments;
    }

private:
    std::set<std::string> locals_names;
    std::vector<Variable> locals;
    std::set<std::string> arguments_names;
    std::vector<Variable> arguments;
};

//
// FFT buffer variable
//

struct FFTBuffer : public Variable
{
    Expression offset, stride;

    FFTBuffer(std::string const& name, Expression const& offset, Expression const& stride)
        : Variable(name, "scalar_type", true)
        , offset(offset)
        , stride(stride)
    {
    }

    FFTBuffer(std::string const& name,
              Expression const&  offset,
              Expression const&  stride,
              unsigned int       size)
        : Variable(name, "scalar_type", false, false, size)
        , offset(offset)
        , stride(stride)
    {
    }

    Variable operator[](const Expression& index) const;

    Expression load_global(const Expression& index) const;
    Statement  store_global(const Expression& index, const Expression& value) const;

    Variable variable() const
    {
        return *this;
    }
};

//
// FFT distributed GPU work helper
//

// FFTGPUWorkParams: info about the work that needs to be done
struct FFTGPUWorkParams
{
    std::vector<unsigned int> factors;
    unsigned int              length, width, nheight, pass, threads_per_transform;
    double                    height;

    Variable write, thread;

    FFTGPUWorkParams() = delete;
    FFTGPUWorkParams(std::vector<unsigned int> factors,
                     unsigned int              threads_per_transform,
                     Variable                  write,
                     Variable                  thread)
        : factors(factors)
        , threads_per_transform(threads_per_transform)
        , write(write)
        , thread(thread)
    {
        length = product(factors.cbegin(), factors.cend());
        set_pass(0);
    }

    virtual ~FFTGPUWorkParams() = default;

    void set_pass(unsigned int const _pass)
    {
        pass    = _pass;
        width   = factors[pass];
        height  = double(length) / width / threads_per_transform;
        nheight = product(factors.cbegin(), factors.cbegin() + pass);
    }
};

// FFTGPUWork: lowering calls subclasses "generate" according to work parameters
struct FFTGPUWork
{
    FFTGPUWorkParams params;

    FFTGPUWork() = delete;
    FFTGPUWork(FFTGPUWorkParams const& params)
        : params(params)
    {
    }

    virtual ~FFTGPUWork() = default;

    StatementList         lower() const;
    virtual StatementList generate(unsigned int h) const = 0;

    virtual Guard guard() const
    {
        return Guard::BOTH;
    }
};

//
// FFT operators
//

struct FFTApplyTwiddle;
struct FFTApplyTwiddleInline;
struct FFTApplyTwiddleTable;
struct FFTButterfly;
struct FFTComputeOffsets;
struct FFTExchange;
struct FFTExchangeHalf;
struct FFTExchangeDual;
struct FFTLoadStockham;
struct FFTLoadStockhamDual;
struct FFTStoreStockham;
struct FFTBluesteinPadMul;
struct FFTBluesteinFFTMul;
struct FFTBluesteinResMul;
struct FFTSyncThreads;
struct FFTZero;

using FFTOperation = std::variant<FFTApplyTwiddle,
                                  FFTApplyTwiddleInline,
                                  FFTApplyTwiddleTable,
                                  FFTButterfly,
                                  FFTComputeOffsets,
                                  FFTExchange,
                                  FFTExchangeHalf,
                                  FFTExchangeDual,
                                  FFTLoadStockham,
                                  FFTLoadStockhamDual,
                                  FFTStoreStockham,
                                  FFTBluesteinPadMul,
                                  FFTBluesteinFFTMul,
                                  FFTBluesteinResMul,
                                  FFTSyncThreads,
                                  FFTZero>;

struct FFTComputeOffsets
{
    unsigned int length0;
    Variable     dim, lengths, offset, stride, batch, nbatch, thread, write;
    FFTBuffer    lds, x;
    unsigned int threads_per_block, threads_per_transform;

    std::shared_ptr<Context> context;

    Variable transform{"transform", "size_t"};
    Variable remaining{"remaining", "size_t"};
    Variable index_along_d{"index_along_d", "size_t"};
    Variable d{"d", "int"};

    FFTComputeOffsets() = delete;
    FFTComputeOffsets(unsigned int             length0,
                      Variable const&          dim,
                      Variable const&          lengths,
                      Variable const&          offset,
                      Variable const&          stride,
                      Variable const&          batch,
                      Variable const&          nbatch,
                      Variable const&          thread,
                      Variable const&          write,
                      FFTBuffer const&         lds,
                      FFTBuffer const&         x,
                      unsigned int const       threads_per_block,
                      unsigned int const       threads_per_transform,
                      std::shared_ptr<Context> context)
        : length0(length0)
        , dim(dim)
        , lengths(lengths)
        , offset(offset)
        , stride(stride)
        , batch(batch)
        , nbatch(nbatch)
        , thread(thread)
        , write(write)
        , lds(lds)
        , x(x)
        , threads_per_block(threads_per_block)
        , threads_per_transform(threads_per_transform)
        , context(context)
    {
        context->add_local(transform);
        context->add_local(remaining);
        context->add_local(index_along_d);
        context->add_local(d);
        context->add_local(std::get<Variable>(lds.offset));
        context->add_local(std::get<Variable>(lds.stride));
        context->add_local(std::get<Variable>(x.stride));
    }

    StatementList lower() const
    {
        Variable block_id{"blockIdx.x", ""};
        Variable thread_id{"threadIdx.x", ""};

        auto offset_lds = std::get<Variable>(lds.offset);
        auto stride_lds = std::get<Variable>(lds.stride);
        auto stride_x   = std::get<Variable>(x.stride);

        auto transforms_per_block = threads_per_block / threads_per_transform;

        auto stmts = StatementList();
        stmts += Assign(thread, thread_id % threads_per_transform);
        stmts += Assign(offset, 0);
        stmts += Assign(stride_x, stride[0]);
        stmts += Assign(stride_lds, 1);
        stmts += Assign(transform,
                        block_id * transforms_per_block + thread_id / threads_per_transform);
        stmts += Assign(remaining, transform);

        For offset_for{d, 1, d < dim, 1};
        offset_for.body += Assign(index_along_d, remaining % lengths[d]);
        offset_for.body += Assign(remaining, remaining / lengths[d]);
        offset_for.body += Assign(offset, offset + index_along_d * stride[d]);
        stmts += offset_for;

        stmts += Assign(batch, remaining);
        stmts += Assign(offset, offset + batch * stride[dim]);
        stmts += Assign(write, Literal{"true"});
        stmts += Assign(offset_lds, length0 * (transform % transforms_per_block));
        stmts += If(batch >= nbatch, {Return()});

        return stmts;
    }
};

struct FFTApplyTwiddle : public FFTGPUWork
{
    FFTBuffer                R;
    std::shared_ptr<Context> context;
    int                      direction;

    FFTApplyTwiddle() = delete;
    FFTApplyTwiddle(FFTBuffer const&         R,
                    FFTGPUWorkParams const&  params,
                    std::shared_ptr<Context> context,
                    int const                direction = -1)
        : FFTGPUWork(params)
        , R(R)
        , context(context)
        , direction(direction)
    {
    }

    StatementList generate(unsigned int h) const override
    {
        // this should be transformed into a table or inline twiddle
        throw std::runtime_error("Unable to lower a raw FFTApplyTwiddle node.");
    }
};

struct FFTApplyTwiddleInline : public FFTGPUWork
{
    Variable                 t{"t", "scalar_type"};
    FFTBuffer                R;
    int                      direction;
    std::shared_ptr<Context> context;

    FFTApplyTwiddleInline() = delete;
    FFTApplyTwiddleInline(FFTBuffer const&         R,
                          int const                direction,
                          FFTGPUWorkParams const&  params,
                          std::shared_ptr<Context> context)
        : FFTGPUWork(params)
        , R(R)
        , direction(direction)
        , context(context)
    {
        context->add_local(t);
    }

    FFTApplyTwiddleInline(FFTApplyTwiddle const x)
        : FFTGPUWork(x.params)
        , R(x.R)
    {
    }

    StatementList generate(unsigned int const h) const override
    {
        StatementList stmts;
        for(unsigned int w = 1; w < params.width; ++w)
        {
            // auto tid  = params.thread + h * params.threads_per_transform;
            // auto tidx = params.nheight - 1 + w - 1 + (params.width - 1) * (tid % params.nheight);
            auto ridx  = h * params.width + w;
            auto theta = Literal(params.nheight); // XXX
            stmts += Call("sincospi", {theta, t.y(), t.x()}); // XXX need address of
            stmts += Assign(t, TwiddleMultiply(t, R[ridx]));
            stmts += Assign(R[ridx], t);
        }
        return stmts;
    }

    Guard guard() const override
    {
        return Guard::NONE;
    }
};

struct FFTApplyTwiddleTable : public FFTGPUWork
{
    FFTBuffer                R;
    Variable                 W{"W", "scalar_type"};
    Variable                 t{"t", "scalar_type"};
    Variable                 twiddles;
    int                      direction;
    std::shared_ptr<Context> context;

    FFTApplyTwiddleTable() = delete;
    FFTApplyTwiddleTable(FFTBuffer const&         R,
                         Variable const&          twiddles,
                         int const                direction,
                         FFTGPUWorkParams const&  params,
                         std::shared_ptr<Context> context)
        : FFTGPUWork(params)
        , R(R)
        , twiddles(twiddles)
        , direction(direction)
        , context(context)
    {
        context->add_local(t);
        context->add_local(W);
    }

    StatementList generate(unsigned int const h) const override
    {
        StatementList stmts;
        for(unsigned int w = 1; w < params.width; ++w)
        {
            auto tid  = params.thread + h * params.threads_per_transform;
            auto tidx = params.nheight - params.factors.front() + w - 1
                        + (params.width - 1) * (tid % params.nheight);
            auto ridx = h * params.width + w;
            stmts += Assign(W, twiddles[tidx]);
            if(direction == -1)
                stmts += Assign(t, TwiddleMultiply(R[ridx], W));
            else
                stmts += Assign(t, TwiddleMultiplyConjugate(R[ridx], W));
            stmts += Assign(R[ridx], t);
        }
        return stmts;
    }

    Guard guard() const override
    {
        return Guard::NONE;
    }
};

struct FFTButterfly : public FFTGPUWork
{
    FFTBuffer R;
    int       direction;

    FFTButterfly(FFTBuffer const& R, int const direction, FFTGPUWorkParams const& params)
        : FFTGPUWork(params)
        , R(R)
        , direction(direction)
    {
    }

    StatementList generate(unsigned int const h) const override
    {
        StatementList           stmts;
        std::vector<Expression> args;
        for(unsigned int w = 0; w < params.width; ++w)
            args.push_back(R + (h * params.width + w));

        stmts += CommentLines{
            "pass " + std::to_string(params.pass) + ", width " + std::to_string(params.width),
            "using " + std::to_string(params.threads_per_transform) + " threads we need to do "
                + std::to_string(params.length / params.width) + " radix-"
                + std::to_string(params.width) + " butterflies",
            "therefore each thread will do " + std::to_string(params.height) + " butterflies"};

        if(direction == -1)
            stmts += Call("FwdRad" + std::to_string(params.width) + "B1", args);
        else
            stmts += Call("InvRad" + std::to_string(params.width) + "B1", args);
        return stmts;
    }

    Guard guard() const override
    {
        return Guard::NONE;
    }
};

struct FFTStoreStockham : public FFTGPUWork
{
    FFTBuffer src, dst;
    Component component;

    FFTStoreStockham() = delete;
    FFTStoreStockham(FFTBuffer const&        src,
                     FFTBuffer const&        dst,
                     FFTGPUWorkParams const& params,
                     Component const&        component = Component::BOTH)
        : FFTGPUWork(params)
        , src(src)
        , dst(dst)
        , component(component)
    {
    }

    StatementList generate(unsigned int const h) const override
    {
        StatementList stmts;

        // store
        for(unsigned int w = 0; w < params.width; ++w)
        {
            auto tid = params.thread + h * params.threads_per_transform;
            auto idx = (tid / params.nheight) * (params.width * params.nheight)
                       + tid % params.nheight + w * params.nheight;
            if(component == Component::BOTH)
                stmts += Assign(dst[idx], src[h * params.width + w]);
            else if(component == Component::REAL)
                stmts += Assign(dst[idx], src[h * params.width + w].x());
            else if(component == Component::IMAG)
                stmts += Assign(dst[idx], src[h * params.width + w].y());
        }

        return stmts;
    }
};

struct FFTLoadStockham : public FFTGPUWork
{
    FFTBuffer dst, src;
    Component component;

    FFTLoadStockham() = delete;
    FFTLoadStockham(FFTBuffer const&        dst,
                    FFTBuffer const&        src,
                    FFTGPUWorkParams const& params,
                    Component const&        component = Component::BOTH)
        : FFTGPUWork(params)
        , dst(dst)
        , src(src)
        , component(component)
    {
    }

    StatementList generate(unsigned int const h) const override
    {
        StatementList stmts;

        for(unsigned int w = 0; w < params.width; ++w)
        {
            // XXX callbacks
            auto tid = params.thread + h * params.threads_per_transform;
            auto idx = tid + w * (params.length / params.width);

            if(component == Component::BOTH)
                stmts += Assign(dst[h * params.width + w], src[idx]);
            else if(component == Component::REAL)
                stmts += Assign(dst[h * params.width + w].x(), src[idx]);
            else if(component == Component::IMAG)
                stmts += Assign(dst[h * params.width + w].y(), src[idx]);
        }

        return stmts;
    }
};

struct FFTLoadStockhamDual : public FFTGPUWork
{
    FFTBuffer dst, src;
    Component component;

    FFTLoadStockhamDual() = delete;
    FFTLoadStockhamDual(FFTBuffer const&        dst,
                        FFTBuffer const&        src,
                        FFTGPUWorkParams const& params,
                        Component const&        component = Component::BOTH)
        : FFTGPUWork(params)
        , dst(dst)
        , src(src)
        , component(component)
    {
    }

    StatementList generate(unsigned int const h) const override
    {
        StatementList stmts;

        for(unsigned int w = 0; w < params.width; ++w)
        {
            // XXX callbacks
            auto tid = params.thread + h * params.threads_per_transform;
            auto idx = tid + w * (params.length / params.width);
            if(component == Component::BOTH)
                stmts += Assign(dst[h * params.width + w], src[idx]);
            else if(component == Component::REAL)
                stmts += Assign(dst[h * params.width + w].x(), src[idx]);
            else if(component == Component::IMAG)
                stmts += Assign(dst[h * params.width + w].y(), src[idx]);
        }

        return stmts;
    }
};

struct FFTExchange
{
    FFTGPUWorkParams params;
    FFTBuffer        R;
    FFTBuffer        lds;

    FFTExchange() = delete;
    FFTExchange(FFTBuffer const& R, FFTBuffer const& lds, FFTGPUWorkParams const& params)
        : params(params)
        , R(R)
        , lds(lds)
    {
    }

    StatementList lower() const
    {
        auto store_params = FFTGPUWorkParams(params);
        auto load_params  = FFTGPUWorkParams(params);
        load_params.set_pass(params.pass + 1);

        auto store = FFTStoreStockham(R, lds, store_params);
        auto load  = FFTLoadStockham(R, lds, load_params);

        StatementList stmts;
        stmts += SyncThreads();
        stmts += store.lower();
        stmts += SyncThreads();
        stmts += load.lower();
        return stmts;
    }
};

struct FFTExchangeHalf
{
    FFTGPUWorkParams params;
    FFTBuffer        R;
    FFTBuffer        lds;

    FFTExchangeHalf() = delete;
    FFTExchangeHalf(FFTBuffer const& R, FFTBuffer const& lds, FFTGPUWorkParams const& params)
        : params(params)
        , R(R)
        , lds(lds)
    {
    }

    StatementList lower() const
    {
        auto store_params = FFTGPUWorkParams(params);
        auto load_params  = FFTGPUWorkParams(params);
        load_params.set_pass(params.pass + 1);

        auto rstore = FFTStoreStockham(R, lds, store_params, Component::REAL);
        auto rload  = FFTLoadStockham(R, lds, load_params, Component::REAL);
        auto istore = FFTStoreStockham(R, lds, store_params, Component::IMAG);
        auto iload  = FFTLoadStockham(R, lds, load_params, Component::IMAG);

        StatementList stmts;
        stmts += SyncThreads();
        stmts += rstore.lower();
        stmts += SyncThreads();
        stmts += rload.lower();
        stmts += SyncThreads();
        stmts += istore.lower();
        stmts += SyncThreads();
        stmts += iload.lower();
        return stmts;
    }
};

struct FFTExchangeDual
{
    FFTExchangeHalf exch_real;
    FFTExchange     exch_cmplx;
    Variable        lds_is_real{"lds_is_real", "bool"}; //  XXX

    FFTExchangeDual() = delete;
    FFTExchangeDual(FFTBuffer const&        R,
                    FFTBuffer const&        lds_real,
                    FFTBuffer const&        lds_complex,
                    FFTGPUWorkParams const& params)
        : exch_real(R, lds_real, params)
        , exch_cmplx(R, lds_complex, params)
    {
    }

    StatementList lower() const
    {
        StatementList stmts;
        stmts += If(Not{lds_is_real}, exch_cmplx.lower());
        stmts += Else(exch_real.lower());
        return stmts;
    }
};

struct FFTBluesteinPadMul : public FFTGPUWork
{
    unsigned int length;
    FFTBuffer    dst, src, chirp;

    FFTBluesteinPadMul() = delete;
    FFTBluesteinPadMul(unsigned int            length,
                       FFTBuffer const&        dst,
                       FFTBuffer const&        src,
                       FFTBuffer const&        chirp,
                       FFTGPUWorkParams const& params)
        : FFTGPUWork(params)
        , length(length)
        , dst(dst)
        , src(src)
        , chirp(chirp)
    {
    }

    Guard guard() const override
    {
        return Guard::THREAD;
    }

    StatementList generate(unsigned int h) const override
    {
        StatementList stmts;

        Variable in_elem{"in_elem", "scalar_type"};

        for(unsigned int w = 0; w < params.width; ++w)
        {
            auto tid = params.thread + h * params.threads_per_transform;
            auto idx = tid + w * (params.length / params.width);

            stmts += If{idx < length,
                        {
                            Declaration{in_elem},
                            Assign{in_elem, src.load_global(idx)},
                            Assign{dst[idx].x(),
                                   in_elem.x() * chirp[idx].x() + in_elem.y() * chirp[idx].y()},
                            Assign{dst[idx].y(),
                                   -in_elem.x() * chirp[idx].y() + in_elem.y() * chirp[idx].x()},
                        }};
            stmts += Else{{
                Assign{dst[idx], CallExpr{"scalar_type", {0, 0}}},
            }};
        }

        return stmts;
    }
};

struct FFTBluesteinFFTMul : public FFTGPUWork
{
    unsigned int lengthBlue;
    FFTBuffer    dst, srcA, srcB;
    Variable     elem;

    FFTBluesteinFFTMul() = delete;
    FFTBluesteinFFTMul(unsigned int            lengthBlue,
                       FFTBuffer const&        dst,
                       FFTBuffer const&        srcA,
                       FFTBuffer const&        srcB,
                       Variable&               elem,
                       FFTGPUWorkParams const& params)
        : FFTGPUWork(params)
        , lengthBlue(lengthBlue)
        , dst(dst)
        , srcA(srcA)
        , srcB(srcB)
        , elem(elem)
    {
    }

    Guard guard() const override
    {
        return Guard::THREAD;
    }

    StatementList generate(unsigned int h) const override
    {
        StatementList stmts;

        for(unsigned int w = 0; w < params.width; ++w)
        {
            auto tid = params.thread + h * params.threads_per_transform;
            auto idx = tid + w * (params.length / params.width);

            stmts += Assign{elem, srcA[idx]};
            stmts += Assign{dst[idx].x(), srcB[idx].x() * elem.x() - srcB[idx].y() * elem.y()};
            stmts += Assign{dst[idx].y(), srcB[idx].x() * elem.y() + srcB[idx].y() * elem.x()};
        }
        return stmts;
    }
};

struct FFTBluesteinResMul : public FFTGPUWork
{
    unsigned int length;
    unsigned int lengthBlue;
    FFTBuffer    dst, src, chirp;
    Variable     elem;

    FFTBluesteinResMul() = delete;
    FFTBluesteinResMul(unsigned int            length,
                       unsigned int            lengthBlue,
                       FFTBuffer const&        dst,
                       FFTBuffer const&        src,
                       FFTBuffer const&        chirp,
                       Variable&               elem,
                       FFTGPUWorkParams const& params)
        : FFTGPUWork(params)
        , length(length)
        , lengthBlue(lengthBlue)
        , dst(dst)
        , src(src)
        , chirp(chirp)
        , elem(elem)
    {
    }

    Guard guard() const override
    {
        return Guard::THREAD;
    }

    StatementList generate(unsigned int h) const override
    {
        StatementList stmts;

        Literal MI{"(1.0 / real_type_t<scalar_type>(" + std::to_string(lengthBlue) + "))"};

        for(unsigned int w = 0; w < params.width; ++w)
        {
            auto tid = params.thread + h * params.threads_per_transform;
            auto idx = tid + w * (params.length / params.width);

            If write_cond{idx < length, {}};
            write_cond.body += Assign{
                elem.x(), MI * (src[idx].x() * chirp[idx].x() + src[idx].y() * chirp[idx].y())};
            write_cond.body += Assign{
                elem.y(), MI * (-src[idx].x() * chirp[idx].y() + src[idx].y() * chirp[idx].x())};
            write_cond.body += dst.store_global(idx, elem);
            stmts += write_cond;
        }

        return stmts;
    }
};

struct FFTSyncThreads : public FFTGPUWork
{
    FFTSyncThreads() = delete;
    FFTSyncThreads(FFTGPUWorkParams const& params)
        : FFTGPUWork(params)
    {
    }

    Guard guard() const override
    {
        return Guard::NONE;
    }

    StatementList generate(unsigned int h) const override
    {
        StatementList stmts;
        if(h == 0)
            stmts += SyncThreads{};
        return stmts;
    }
};

struct FFTZero : public FFTGPUWork
{
    FFTBuffer dst;

    FFTZero() = delete;
    FFTZero(FFTBuffer const& dst, FFTGPUWorkParams const& params)
        : FFTGPUWork(params)
        , dst(dst)
    {
    }

    Guard guard() const override
    {
        return Guard::THREAD;
    }

    StatementList generate(unsigned int const h) const override
    {
        StatementList stmts;
        auto          idx = params.thread + h * params.threads_per_transform;
        stmts += Assign(dst[idx], ComplexLiteral{Literal{"0.0"}, Literal{"0.0"}});
        return stmts;
    }
};

//
// FFT operation list (sequential)
//

class FFTOperationList
{
public:
    std::vector<FFTOperation> operations;
    FFTOperationList(){};
    FFTOperationList(std::initializer_list<FFTOperation> il)
        : operations(il){};

    StatementList lower()
    {
        StatementList stmts;
        for(auto& op : operations)
            stmts += vlower(op);
        return stmts;
    }
};

void operator+=(FFTOperationList& opers, const FFTOperation& op);

void operator+=(FFTOperationList& opers, const FFTOperationList& ops);

//
// Visitors
//

struct FFTBaseVisitor
{
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTComputeOffsets);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTLoadStockham);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTLoadStockhamDual);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTApplyTwiddle);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTApplyTwiddleInline);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTApplyTwiddleTable);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTButterfly);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTExchange);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTExchangeHalf);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTExchangeDual);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTStoreStockham);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTBluesteinPadMul);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTBluesteinFFTMul);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTBluesteinResMul);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTSyncThreads);
    MAKE_VISITOR_OPERATOR(FFTOperation, FFTZero);

    MAKE_TRIVIAL_VISIT(FFTOperation, FFTComputeOffsets);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTLoadStockham);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTLoadStockhamDual);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTApplyTwiddle);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTApplyTwiddleInline);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTApplyTwiddleTable);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTButterfly);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTExchangeHalf);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTExchangeDual);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTStoreStockham);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTBluesteinPadMul);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTBluesteinResMul);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTBluesteinFFTMul);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTSyncThreads);
    MAKE_TRIVIAL_VISIT(FFTOperation, FFTZero);

    MAKE_VISITOR_OPERATOR(FFTBuffer, FFTBuffer);
    MAKE_TRIVIAL_VISIT(FFTBuffer, FFTBuffer);

    virtual FFTOperation visit_FFTExchange(const FFTExchange& x)
    {
        return FFTExchange(visit_FFTBuffer(x.R), visit_FFTBuffer(x.lds), x.params);
    }

    // XXX need to make others do buffer visiting...

    FFTOperationList operator()(const FFTOperationList& x)
    {
        FFTOperationList ret;
        for(const auto& op : x.operations)
        {
            ret += std::visit(*this, op);
        }
        return ret;
    }
};

//
// Twiddle transforms
//

struct TableTwiddleVisitor : public FFTBaseVisitor
{
    Variable twiddles;
    TableTwiddleVisitor(Variable twiddles)
        : twiddles(twiddles)
    {
    }

    FFTOperation visit_FFTApplyTwiddle(FFTApplyTwiddle const& x) override
    {
        auto y = FFTApplyTwiddleTable(x.R, twiddles, x.direction, x.params, x.context);
        return FFTBaseVisitor::visit_FFTApplyTwiddleTable(y);
    }
};

static FFTOperationList make_table_twiddle(const FFTOperationList& t, Variable twiddles)
{
    auto visitor = TableTwiddleVisitor(twiddles);
    return visitor(t);
}

//
// Inverse transform
//

struct FFTInverseVisitor : public FFTBaseVisitor
{
    FFTOperation visit_FFTButterfly(FFTButterfly const& x) override
    {
        auto y = FFTButterfly(x.R, 1, x.params);
        return FFTBaseVisitor::visit_FFTButterfly(y);
    }

    FFTOperation visit_FFTApplyTwiddle(FFTApplyTwiddle const& x) override
    {
        auto y = FFTApplyTwiddle(x.R, x.params, x.context, 1);
        return FFTBaseVisitor::visit_FFTApplyTwiddle(y);
    }
};

static FFTOperationList make_inverse(const FFTOperationList& t)
{
    auto visitor = FFTInverseVisitor();
    return visitor(t);
}

//
// Stockham generator
//

static unsigned int compute_nregisters(const unsigned int               length,
                                       std::vector<unsigned int> const& factors,
                                       const unsigned int               threads_per_transform)
{
    unsigned int max_registers = 0;
    for(auto width : factors)
    {
        unsigned int n = std::ceil(double(length) / width / threads_per_transform) * width;
        if(n > max_registers)
            max_registers = n;
    }
    return max_registers;
}

struct StockhamTransform
{
    unsigned int              length;
    std::vector<unsigned int> factors;

    unsigned int threads_per_block;
    unsigned int threads_per_transform;

    std::shared_ptr<Context> context;

    FFTBuffer R{"R", Literal{0}, Literal{1}, 0};
    FFTBuffer lds{"lds", Variable{"offset_lds", "int"}, Variable{"stride_lds", "int"}};
    FFTBuffer X{"X", Variable{"offset_lds", "size_t"}, Literal{1}};

    Variable dim{"dim", "unsigned int"};
    Variable nbatch{"nbatch", "size_t"};
    Variable lengths{"lengths", "size_t", true, true};
    Variable stride{"stride", "size_t", true, true};
    Variable offset{"offset", "size_t"};

    Variable write{"write", "bool"};
    Variable thread{"thread", "size_t"};
    Variable batch{"batch", "size_t"};

    Variable twiddles{"twiddles", "const scalar_type", true, true};
    Variable load_cb_fn{"load_cb_fn", "void*"};
    Variable load_cb_data{"load_cb_data", "void*"};
    Variable load_cb_lds_bytes{"load_cb_lds_bytes", "unsigned int"};
    Variable store_cb_fn{"store_cb_fn", "void*"};
    Variable store_cb_data{"store_cb_data", "void*"};

    StockhamTransform(std::vector<unsigned int> const& factors,
                      const unsigned int               threads_per_block,
                      const unsigned int               threads_per_transform,
                      std::shared_ptr<Context>         context)
        : factors(factors)
        , threads_per_block(threads_per_block)
        , threads_per_transform(threads_per_transform)
        , context(context)
    {
        length = product(factors.cbegin(), factors.cend());
        context->add_argument(lengths);
        context->add_argument(stride);
        context->add_argument(nbatch);
        context->add_argument(load_cb_fn);
        context->add_argument(load_cb_data);
        context->add_argument(load_cb_lds_bytes);
        context->add_argument(store_cb_fn);
        context->add_argument(store_cb_data);
        context->add_argument(X.variable());
        context->add_local(std::get<Variable>(lds.offset));
        context->add_local(std::get<Variable>(lds.stride));

        context->add_local(batch);
        context->add_local(thread);
        context->add_local(thread);
        context->add_local(write);
        R.size = compute_nregisters(length, factors, threads_per_transform);
        context->add_local(R.variable());
    }

    FFTOperationList generate()
    {
        FFTOperationList ops;
        FFTGPUWorkParams work(factors, threads_per_transform, write, thread);

        ops += FFTLoadStockham(R, X, work);

        for(unsigned int pass = 0; pass < factors.size(); ++pass)
        {
            bool first_pass = pass == 0;
            bool last_pass  = pass == factors.size() - 1;

            work.set_pass(pass);

            if(!first_pass)
                ops += FFTApplyTwiddle(R, work, context);

            ops += FFTButterfly(R, -1, work);

            if(!last_pass)
                ops += FFTExchange(R, lds, work);
        }

        ops += FFTStoreStockham(R, X, work);

        return ops;
    }
};

struct BluesteinTransform
{
    unsigned int              length, lengthBlue;
    int                       direction;
    std::vector<unsigned int> factors;

    unsigned int threads_per_block;
    unsigned int threads_per_transform;

    std::shared_ptr<Context> context;

    // FFT registers
    FFTBuffer R{"R", Literal{0}, Literal{1}, 0};
    // LDS buffer
    FFTBuffer A{"A", Variable{"offset_lds", "int"}, Variable{"stride_lds", "int"}};
    // FFTed chirp signal (second half of chirp buffer)
    FFTBuffer B{"B", Literal{0}, Literal{1}};
    // chirp signal (first half of chirp buffer)
    FFTBuffer a{"a", Literal{0}, Literal{1}};
    // user data
    FFTBuffer X{"X", Variable{"offset", "size_t"}, Variable{"stride0", "size_t"}};

    Variable dim{"dim", "unsigned int"};
    Variable nbatch{"nbatch", "size_t"};
    Variable lengths{"lengths", "size_t", true, true};
    Variable stride{"stride", "size_t", true, true};
    Variable offset{"offset", "size_t"};

    Variable write{"write", "bool"};
    Variable thread{"thread", "size_t"};
    Variable batch{"batch", "size_t"};
    Variable val{"val", "scalar_type"};

    Variable buf_temp{"buf_temp", "scalar_type", true, true};
    Variable twiddles{"twiddles", "const scalar_type", true, true};

    BluesteinTransform(unsigned int              length,
                       unsigned int              lengthBlue,
                       int                       direction,
                       std::vector<unsigned int> factors,
                       unsigned int              threads_per_block,
                       unsigned int              threads_per_transform,
                       std::shared_ptr<Context>  context)
        : length(length)
        , lengthBlue(lengthBlue)
        , direction(direction)
        , factors(factors)
        , threads_per_block(threads_per_block)
        , threads_per_transform(threads_per_transform)
        , context(context)
    {
        context->add_argument(buf_temp);
        context->add_argument(twiddles);
        context->add_argument(lengths);
        context->add_argument(stride);
        context->add_argument(nbatch);
        context->add_argument(X.variable());
        R.size = compute_nregisters(lengthBlue, factors, threads_per_transform);

        context->add_local(thread);
        context->add_local(offset);
        context->add_local(batch);
        context->add_local(write);
        context->add_local(a.variable());
        context->add_local(B.variable());
        context->add_local(A.variable());
        context->add_local(val);
    }

    FFTOperationList generate()
    {
        FFTOperationList  bluestein;
        StockhamTransform stockham(factors, threads_per_block, threads_per_transform, context);

        FFTGPUWorkParams work_full(factors, threads_per_transform, write, thread);

        // compute offsets into user data
        FFTComputeOffsets offsets{lengthBlue,
                                  dim,
                                  lengths,
                                  offset,
                                  stride,
                                  batch,
                                  nbatch,
                                  thread,
                                  write,
                                  A,
                                  X,
                                  threads_per_block,
                                  threads_per_transform,
                                  context};

        bluestein += offsets;

        // pad X to lengthBlue, store X * a -> A
        bluestein += FFTBluesteinPadMul(length, A, X, a, work_full);

        bluestein += FFTSyncThreads{work_full};

        // forward FFT on A
        stockham.X.name = "A"; // bit of a hack
        bluestein += stockham.generate();

        bluestein += FFTSyncThreads{work_full};

        // multiply A * B -> A
        bluestein += FFTBluesteinFFTMul(lengthBlue, A, A, B, val, work_full);
        bluestein += FFTSyncThreads{work_full};

        // inverse FFT on A
        stockham.X.name = "A"; // bit of a hack
        bluestein += make_inverse(stockham.generate());

        bluestein += FFTSyncThreads{work_full};
        // multiply a * A -> X
        bluestein += FFTBluesteinResMul(length, lengthBlue, X, A, a, val, work_full);

        return make_table_twiddle(bluestein, stockham.twiddles);
    }
};
