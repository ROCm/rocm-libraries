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

#pragma once

#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <boost/lexical_cast.hpp>

namespace Tensile
{
    namespace Client
    {
        class CSVStackFile
        {
        public:
            CSVStackFile(std::string const& filename, std::string const& separator = ", ");
            CSVStackFile(std::ostream& stream, std::string const& separator = ", ");
            CSVStackFile(std::shared_ptr<std::ostream> stream, std::string const& separator = ", ");

            ~CSVStackFile();

            void setHeaderForKey(std::string const& key, std::string const& header);

            void setValueForKey(std::string const& key, std::string const& value);
            void setValueForKey(std::string const& key, double const& value);

            template <typename T>
            typename std::enable_if<!std::is_same<T, std::string>::value, void>::type
                setValueForKey(std::string const& key, T const& value)
            {
                setValueForKey(key, boost::lexical_cast<std::string>(value));
            }

            void push();
            void pop();

            void writeCurrentRow();
            void readCurrentRow(std::unordered_map<std::string, std::string>& outMap);
            void clearCurrentRow();

        private:
            std::string escape(std::string const& value);
            std::string escapeQuote(std::string const& value);

            void writeRow(std::unordered_map<std::string, std::string> const& row);

            std::shared_ptr<std::ostream> m_stream;

            std::string m_separator;

            bool                                         m_firstRow = true;
            std::vector<std::string>                     m_keyOrder;
            std::unordered_map<std::string, std::string> m_headers;

            std::unordered_map<std::string, std::string> m_currentRow;

            std::vector<std::unordered_map<std::string, std::string>> m_stack;
        };
    } // namespace Client
} // namespace Tensile
