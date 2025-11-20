/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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
#include <rocRoller/Parameters/Solution/LoadOption.hpp>
#include <rocRoller/Utilities/Error.hpp>

#include <string>

namespace rocRoller
{
    namespace Parameters
    {
        namespace Solution
        {
            MemoryType GetMemoryType(LoadPath const& mode)
            {
                switch(mode)
                {
                case LoadPath::BufferToVGPR:
                    return MemoryType::WAVE;
                case LoadPath::BufferToLDSViaVGPR:
                    return MemoryType::WAVE_LDS;
                case LoadPath::BufferToLDS:
                    return MemoryType::WAVE_Direct2LDS;
                case LoadPath::Count:
                    Throw<FatalError>(fmt::format("No valid MemoryType available for LDS mode {}\n",
                                                  toString(mode)));
                }
            }

            bool IsBufferToLDS(LoadPath const& mode)
            {
                return mode == LoadPath::BufferToLDS;
            }

            std::string toString(LoadPath mode)
            {
                switch(mode)
                {
                case LoadPath::BufferToVGPR:
                    return "BufferToVGPR";
                case LoadPath::BufferToLDSViaVGPR:
                    return "BufferToLDSViaVGPR";
                case LoadPath::BufferToLDS:
                    return "BufferToLDS";
                default:
                    break;
                }
                return "Invalid";
            }

            std::ostream& operator<<(std::ostream& stream, LoadPath const& mode)
            {
                return stream << toString(mode);
            }
        } // namespace Solution
    } // namespace Parameters
} // namespace rocRoller
