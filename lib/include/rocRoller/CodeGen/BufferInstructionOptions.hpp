/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

/**
 * BufferOptions is a struct that contains the options and optional values for MUBUF instructions.
 * OFFEN: 1 = Supply an offset from VGPR (VADDR), 0 Do not -> offset = 0. 1 bit
 * GLC: global coherent. Controls how reads and writes are handled by L1 cache. 1 bit
 * SLC: System Level Coherent, when set, streaming mode in L2 cache is set. 1 bit
 * LDS:  0 = Return read data to VGPR, 1 = Return read data to LDS. 1 bit
 */

namespace rocRoller
{
    struct BufferInstructionOptions
    {
        bool offen = false;
        bool glc   = false;
        bool slc   = false;
        bool sc1   = false;
        bool lds   = false;
    };
}
