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

#include <Tensile/Utils.hpp>

namespace Tensile
{
    StreamRead::StreamRead(std::string const& value, bool except)
        : m_value(value)
        , m_except(except)
        , m_success(false)
    {
    }

    StreamRead::~StreamRead() = default;

    // bool StreamRead::operator bool() const { return m_success; }

    bool StreamRead::read(std::istream& stream)
    {
        m_success = false;
        char ch;

        for(int i = 0; i < m_value.size(); i++)
        {
            if((ch = stream.get()) != m_value[i])
            {
                for(int j = 0; j <= i; j++)
                    stream.unget();

                if(m_except)
                    throw std::runtime_error(
                        concatenate("Expected '", m_value[i], "', found '", ch, "'."));

                return false;
            }
        }

        m_success = true;
        return true;
    }

} // namespace Tensile
