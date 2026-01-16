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
                case StorePath::VGPRToGlobalMemoryWithBuffer:
                    return MemoryType::WAVE;
                case StorePath::VGPRToGlobalMemoryWithGlobal:
                    return MemoryType::WAVE_FROM_GLOBAL;
                case StorePath::VGPRToGlobalMemoryViaLDSWithBuffer:
                    return MemoryType::WAVE_LDS;
                case StorePath::VGPRToGlobalMemoryViaLDSWithGlobal:
                    return MemoryType::WAVE_LDS_FROM_GLOBAL;
                case StorePath::LDSToGlobalMemoryWithBuffer:
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
                case StorePath::VGPRToGlobalMemoryViaLDSWithBuffer:
                case StorePath::VGPRToGlobalMemoryViaLDSWithGlobal:
                case StorePath::LDSToGlobalMemoryWithBuffer:
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
                case StorePath::VGPRToGlobalMemoryWithBuffer:
                    return "VGPRToGlobalMemoryWithBuffer";
                case StorePath::VGPRToGlobalMemoryWithGlobal:
                    return "VGPRToGlobalMemoryWithGlobal";
                case StorePath::VGPRToGlobalMemoryViaLDSWithBuffer:
                    return "VGPRToGlobalMemoryViaLDSWithBuffer";
                case StorePath::VGPRToGlobalMemoryViaLDSWithGlobal:
                    return "VGPRToGlobalMemoryViaLDSWithGlobal";
                case StorePath::LDSToGlobalMemoryWithBuffer:
                    return "LDSToGlobalMemoryWithBuffer";
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
                if(str == "VGPRToGlobalMemoryWithBuffer")
                    return StorePath::VGPRToGlobalMemoryWithBuffer;
                if(str == "VGPRToGlobalMemoryWithGlobal")
                    return StorePath::VGPRToGlobalMemoryWithGlobal;
                if(str == "VGPRToGlobalMemoryViaLDSWithBuffer")
                    return StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;
                if(str == "VGPRToGlobalMemoryViaLDSWithGlobal")
                    return StorePath::VGPRToGlobalMemoryViaLDSWithGlobal;
                if(str == "LDSToGlobalMemoryWithBuffer")
                    return StorePath::LDSToGlobalMemoryWithBuffer;
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
