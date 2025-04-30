// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "fftgenerator.h"

Variable FFTBuffer::operator[](const Expression& index) const
{
    return Variable(*this, offset + index * stride);
}

Expression FFTBuffer::load_global(const Expression& index) const
{
    return LoadGlobal{*this, offset + index * stride};
}

Statement FFTBuffer::store_global(const Expression& index, const Expression& value) const
{
    return StoreGlobal{*this, offset + index * stride, value};
}

void operator+=(FFTOperationList& opers, const FFTOperation& op)
{
    opers.operations.push_back(op);
}

void operator+=(FFTOperationList& opers, const FFTOperationList& ops)
{
    for(auto op : ops.operations)
        opers.operations.push_back(op);
}

StatementList FFTGPUWork::lower() const
{
    unsigned int iheight = floor(params.height);

    if(params.height > iheight && params.threads_per_transform > params.length / params.width)
        iheight += 1;

    auto work = StatementList();
    for(unsigned int h = 0; h < iheight; ++h)
        work += generate(h);

    auto g = guard();

    auto stmts = StatementList();
    if(g != Guard::NONE)
    {
        if(params.threads_per_transform > params.length / params.width)
        {
            if(g == Guard::BOTH)
                stmts += If(params.write && (params.thread < params.length / params.width), work);
            if(g == Guard::THREAD)
                stmts += If(params.thread < params.length / params.width, work);
            if(g == Guard::WRITE)
                stmts += If(params.write, work);
        }
        else
        {
            if(g == Guard::WRITE || g == Guard::BOTH)
                stmts += If(params.write, work);
            else
                stmts += work;
        }
    }
    else
    {
        stmts += work;
    }

    if(params.height > iheight && params.threads_per_transform < params.length / params.width)
    {
        work = generate(iheight);
        if(g == Guard::NONE)
            stmts += work;
        if(g == Guard::BOTH)
            stmts += If(params.write
                            && (params.thread + iheight * params.threads_per_transform
                                < params.length / params.width),
                        work);
        if(g == Guard::THREAD)
            stmts += If(params.thread + iheight * params.threads_per_transform
                            < params.length / params.width,
                        work);
        if(g == Guard::WRITE)
            stmts += If(params.write, work);
    }

    return stmts;
}
