/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
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
#include <rocRoller/Parameters/Solution/StoreOption.hpp>
#include <rocRoller/Utilities/Error.hpp>

#include <string>

namespace rocRoller
{
    namespace Parameters
    {
        namespace Solution
        {
            MemoryType GetMemoryType(StorePath const& mode)
            {
                switch(mode)
                {
                case StorePath::VGPRToBuffer:
                    return MemoryType::WAVE;
                case StorePath::VGPRToGlobal:
                    return MemoryType::WAVE_FROM_GLOBAL;
                case StorePath::LDSViaVGPRToBuffer:
                    return MemoryType::WAVE_LDS;
                case StorePath::LDSViaVGPRToGlobal:
                    return MemoryType::WAVE_LDS_FROM_GLOBAL;
                case StorePath::LDSToBuffer:
                    return MemoryType::WAVE_Direct2LDS;
                case StorePath::Count:
                    Throw<FatalError>(
                        fmt::format("No valid MemoryType available for mode {}\n", toString(mode)));
                }
            }

            bool IsLDSStore(StorePath const& mode)
            {
                switch(mode)
                {
                case StorePath::LDSViaVGPRToBuffer:
                case StorePath::LDSViaVGPRToGlobal:
                case StorePath::LDSToBuffer:
                    return true;
                default:
                    break;
                }
                return false;
            }

            std::string toString(StorePath mode)
            {
                switch(mode)
                {
                case StorePath::VGPRToBuffer:
                    return "VGPRToBuffer";
                case StorePath::VGPRToGlobal:
                    return "VGPRToGlobal";
                case StorePath::LDSViaVGPRToBuffer:
                    return "LDSViaVGPRToBuffer";
                case StorePath::LDSViaVGPRToGlobal:
                    return "LDSViaVGPRToGlobal";
                case StorePath::LDSToBuffer:
                    return "LDSToBuffer";
                default:
                    break;
                }
                return "Invalid";
            }

            std::ostream& operator<<(std::ostream& stream, StorePath const& mode)
            {
                return stream << toString(mode);
            }

            StorePath storePathFromString(std::string const& str)
            {
                if(str == "VGPRToBuffer")
                    return StorePath::VGPRToBuffer;
                if(str == "VGPRToGlobal")
                    return StorePath::VGPRToGlobal;
                if(str == "LDSViaVGPRToBuffer")
                    return StorePath::LDSViaVGPRToBuffer;
                if(str == "LDSViaVGPRToGlobal")
                    return StorePath::LDSViaVGPRToGlobal;
                if(str == "LDSToBuffer")
                    return StorePath::LDSToBuffer;
                Throw<FatalError>(fmt::format("Invalid StorePath: {}", str));
            }

            std::istream& operator>>(std::istream& stream, StorePath& path)
            {
                std::string str;
                stream >> str;
                path = storePathFromString(str);
                return stream;
            }
        } // namespace Solution
    } // namespace Parameters
} // namespace rocRoller
