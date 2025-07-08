/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <string>

namespace rocRoller
{
    /**
   * @brief Returns number of bits X each ds_read_bX_tr_bY loads for a given variable type of bit-width Y.
   * It returns 64 if elementBits in {16, 8, 4}, 96 if elementBits == 6, and 0 otherwise.
   *
   *
   * @param elementBits number of bits of variable type to load.
   */
    uint bitsPerTransposeLoad(uint elementBits);

    /**
   * @brief Returns extra number of bytes required to fulfill 128b alignment requirement of 6-bit transpose loads.
   * Zero is returned for 16, 8, and 4 bit datatypes.
   *
   *
   * @param elementBits number of bits of variable type to load.
   */
    uint extraLDSBytesPerElementBlock(uint elementBits);

    std::string transposeLoadMnemonic(uint elementBits);
} // rocRoller
