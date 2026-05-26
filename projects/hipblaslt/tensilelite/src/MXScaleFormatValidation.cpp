/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/MXScaleFormatValidation.hpp>

#include <sstream>

namespace TensileLite
{
    namespace
    {
        // Stable, human-readable label that maps tensilelite's enum back to
        // the spelling used in the ISA / valid-combinations table.
        std::string mxScaleLabel(rocisa::DataType dt)
        {
            switch(dt)
            {
            case rocisa::DataType::E8:
                return "E8";
            case rocisa::DataType::E5M3:
                return "E5M3";
            case rocisa::DataType::Float8:
                return "E4M3";
            case rocisa::DataType::None:
                return "None";
            default:
                // Anything that isn't a legal MX scale - render the raw enum
                // string so log output is unambiguous.
                return rocisa::toString(dt);
            }
        }

        std::string mxMatrixLabel(rocisa::DataType dt)
        {
            switch(dt)
            {
            case rocisa::DataType::Float8:
                return "FP8";
            case rocisa::DataType::BFloat8:
                return "BF8";
            case rocisa::DataType::Float6:
                return "FP6";
            case rocisa::DataType::BFloat6:
                return "BF6";
            case rocisa::DataType::Float4:
                return "FP4";
            default:
                return rocisa::toString(dt);
            }
        }
    } // namespace

    std::string formatMXScaleFormatCombination(rocisa::DataType aType,
                                               rocisa::DataType scaleAType,
                                               rocisa::DataType bType,
                                               rocisa::DataType scaleBType)
    {
        std::ostringstream os;
        os << "(A=" << mxMatrixLabel(aType) << ", AScale=" << mxScaleLabel(scaleAType)
           << ", B=" << mxMatrixLabel(bType) << ", BScale=" << mxScaleLabel(scaleBType) << ")";
        return os.str();
    }

    std::string mxScaleFormatCombinationError(rocisa::DataType aType,
                                              rocisa::DataType scaleAType,
                                              rocisa::DataType bType,
                                              rocisa::DataType scaleBType)
    {
        if(isValidMXScaleFormatCombination(aType, scaleAType, bType, scaleBType))
            return std::string();

        std::ostringstream os;
        os << "Invalid MX scale-format combination "
           << formatMXScaleFormatCombination(aType, scaleAType, bType, scaleBType) << ": ";

        if(!isValidMXScaleFormatForDataType(aType, scaleAType))
        {
            os << "matrix A class " << mxMatrixLabel(aType)
               << " does not accept scale format " << mxScaleLabel(scaleAType) << "; ";
        }
        if(!isValidMXScaleFormatForDataType(bType, scaleBType))
        {
            os << "matrix B class " << mxMatrixLabel(bType)
               << " does not accept scale format " << mxScaleLabel(scaleBType) << "; ";
        }
        if(isFP4MatrixDataType(aType) && isFP4MatrixDataType(bType)
           && scaleAType != scaleBType
           && isValidMXScaleFormatForDataType(aType, scaleAType)
           && isValidMXScaleFormatForDataType(bType, scaleBType))
        {
            os << "FP4 x FP4 requires AScale (" << mxScaleLabel(scaleAType)
               << ") == BScale (" << mxScaleLabel(scaleBType) << "); ";
        }

        os << "see table-valid-combinations.txt / ROCm/llvm-project#2634.";
        return os.str();
    }
} // namespace TensileLite
