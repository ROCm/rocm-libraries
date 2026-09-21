/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************
 * Standalone API for validating solution parameters against TensileLite's
 * selection rules. Used by callers (e.g. hipBLASLt library YAML tests) to
 * check that solution specs would be valid for selection without loading
 * full libraries or requiring a GPU. See docs/library-yaml-consistency-check-design.md.
 ******************************************************************************/

#ifndef TENSILE_SOLUTION_VALIDATION_HPP
#define TENSILE_SOLUTION_VALIDATION_HPP

#include <array>
#include <cstdint>
#include <string>

namespace TensileLite
{
    /** Parameters extracted from a solution block (e.g. from library logic YAML). */
    struct SolutionParamsForValidation
    {
        int cuCount = 0;
        int workGroupMappingXCC = -1;
        std::array<int, 3> workGroup = {0, 0, 0};
        int wavefrontSize = 0;
    };

    /**
     * Returns true if the given solution parameters satisfy TensileLite's
     * consistency rules (same rules as WorkgroupMappingXCCCheck and ValidParameters).
     * If outReason is non-null and validation fails, a short reason is written.
     */
    inline bool validateSolutionParamsForSelection(SolutionParamsForValidation const& params,
                                                   std::string*                     outReason = nullptr)
    {
        auto setReason = [outReason](const char* msg) {
            if(outReason)
                *outReason = msg;
        };

        // WorkGroupMappingXCC: must be -1 or (power-of-two and divisor of cuCount)
        if(params.workGroupMappingXCC != -1)
        {
            if(params.workGroupMappingXCC <= 0 || params.cuCount <= 0)
            {
                setReason("WorkGroupMappingXCC and cuCount must be positive when XCC != -1");
                return false;
            }
            bool powerOfTwo = (params.workGroupMappingXCC & (params.workGroupMappingXCC - 1)) == 0;
            if(!powerOfTwo)
            {
                setReason("WorkGroupMappingXCC must be -1 or a power of two");
                return false;
            }
            if(params.cuCount % params.workGroupMappingXCC != 0)
            {
                setReason("WorkGroupMappingXCC must divide cuCount");
                return false;
            }
        }

        // WorkGroup [a,b,c]: positive and product in [32, 1024]
        int const* wg = params.workGroup.data();
        if(wg[0] <= 0 || wg[1] <= 0 || wg[2] <= 0)
        {
            setReason("WorkGroup dimensions must be positive");
            return false;
        }
        int64_t product = static_cast<int64_t>(wg[0]) * wg[1] * wg[2];
        if(product < 32 || product > 1024)
        {
            setReason("WorkGroup product must be in [32, 1024]");
            return false;
        }

        // WavefrontSize: 32 or 64
        if(params.wavefrontSize != 32 && params.wavefrontSize != 64)
        {
            setReason("WavefrontSize must be 32 or 64");
            return false;
        }

        return true;
    }
} // namespace TensileLite

#endif // TENSILE_SOLUTION_VALIDATION_HPP
