// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>

namespace rocRoller
{
    /**
   * @brief Returns number of bits X each ds_read_bX_tr_bY loads for a given variable type of bit-width Y.
   * It returns 64 if elementBits in {16, 8, 4}, 96 if elementBits == 6, and 0 otherwise.
   *
   *
   * @param elementBits number of bits of variable type to load.
   */
    uint bitsPerTransposeLoad(GPUArchitecture const& arch, uint elementBits);

    /**
   * @brief Returns extra number of bytes required to fulfill 128b alignment requirement of 6-bit transpose loads.
   * Zero is returned for 16, 8, and 4 bit datatypes.
   *
   *
   * @param elementBits number of bits of variable type to load.
   */
    uint extraLDSBytesPerElementBlock(GPUArchitecture const& arch, uint elementBits);

    std::string transposeLoadMnemonic(GPUArchitecture const& arch, uint elementBits);
} // rocRoller
