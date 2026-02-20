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

#include "CSVStackFile.hpp"
#include "ResultReporter.hpp"

#include <boost/program_options.hpp>

#include <cstddef>

namespace Tensile
{
    namespace Client
    {
        namespace po = boost::program_options;

        class LibraryUpdateReporter : public ResultReporter
        {
        public:
            static std::shared_ptr<LibraryUpdateReporter> Default(po::variables_map const& args);

            static std::shared_ptr<LibraryUpdateReporter> FromFilename(std::string const& filename,
                                                                       bool addComment);

            /// This one will not close the stream.  Useful for writing to cout.
            LibraryUpdateReporter(std::ostream& stream, bool addComment);
            /// This one has shared ownership of the stream.
            LibraryUpdateReporter(std::shared_ptr<std::ostream> stream, bool addComment);

            virtual void reportValue_string(std::string const& key,
                                            std::string const& value) override;
            virtual void reportValue_uint(std::string const& key, uint64_t value) override;
            virtual void reportValue_int(std::string const& key, int64_t value) override;
            virtual void reportValue_double(std::string const& key, double value) override;
            virtual void reportValue_sizes(std::string const&         key,
                                           std::vector<size_t> const& value) override;

            virtual void postProblem() override;
            virtual void postSolution() override;

            void finalizeReport() override;

        private:
            template <typename T>
            void reportValue(std::string const& key, T const& value);

            std::ostream&                 m_stream;
            std::shared_ptr<std::ostream> m_ownedStream;

            std::vector<size_t> m_problemSizes;

            bool m_addComment = false;

            int64_t     m_curSolutionIdx = -1;
            std::string m_curSolutionName;
            double      m_curSolutionSpeed  = -1.0;
            bool        m_curSolutionPassed = false;

            int64_t     m_fastestSolutionIdx = -1;
            std::string m_fastestSolutionName;
            double      m_fastestSolutionSpeed = -1.0;
        };
    } // namespace Client
} // namespace Tensile
