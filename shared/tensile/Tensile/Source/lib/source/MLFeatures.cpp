/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/MLFeatures.hpp>

namespace Tensile
{
    namespace MLFeatures
    {
        float tilesPerCU(ContractionProblem const&        problem,
                         CUGranularityScaleFactors const& cuFactors)
        {
            float numBatches = 1; // TODO: Higher batch sizes
            float numTilesM  = problem.freeSizeA(0) * cuFactors.mt0Scale; // M / MT0
            float numTilesN  = problem.freeSizeB(0) * cuFactors.mt1Scale; // N / MT1
            float totalTiles = numBatches * ceil(numTilesM) * ceil(numTilesN);
            return totalTiles * cuFactors.cuScale;
        };

        std::ostream& operator<<(std::ostream& stream, CUGranularityScaleFactors const& cugsf)
        {
            return stream << " mt0=" << cugsf.mt0Scale << " mt1=" << cugsf.mt1Scale
                          << " cus=" << cugsf.cuScale;
        };

        std::ostream& operator<<(std::ostream& stream, WaveGranularityScaleFactors const& wgsf)
        {
            return stream << wgsf.cuFactors << " ws=" << wgsf.waveScale;
        };
    } // namespace Tensile
}
