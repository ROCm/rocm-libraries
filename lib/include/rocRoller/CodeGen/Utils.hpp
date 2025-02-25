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

    std::string transposeLoadMnemonic(uint elementBits);
} // rocRoller
