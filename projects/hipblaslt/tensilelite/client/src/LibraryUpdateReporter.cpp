/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <LibraryUpdateReporter.hpp>

#include <cstddef>

namespace TensileLite
{
    namespace Client
    {
        std::shared_ptr<LibraryUpdateReporter>
            LibraryUpdateReporter::Default(po::variables_map const& args)
        {
            auto filename = args["library-update-file"].as<std::string>();
            auto comment  = args["library-update-comment"].as<bool>();
            if(filename != "")
            {
                return FromFilename(filename, comment);
            }
            return std::shared_ptr<LibraryUpdateReporter>();
        }

        std::shared_ptr<LibraryUpdateReporter>
            LibraryUpdateReporter::FromFilename(std::string const& filename, bool addComment)
        {
            auto file = std::make_shared<std::ofstream>(filename);
            return std::make_shared<LibraryUpdateReporter>(file, addComment);
        }

        LibraryUpdateReporter::LibraryUpdateReporter(std::ostream& stream, bool addComment)
            : m_stream(stream)
            , m_addComment(addComment)
        {
        }

        LibraryUpdateReporter::LibraryUpdateReporter(std::shared_ptr<std::ostream> stream,
                                                     bool                          addComment)
            : m_stream(*stream.get())
            , m_ownedStream(stream)
            , m_addComment(addComment)
        {
            if(!stream)
                throw std::runtime_error("Invalid stream! nullptr is not allowed.");
        }

        template <typename T>
        void LibraryUpdateReporter::reportValue(std::string const& key, T const& value)
        {
            std::string valueStr = boost::lexical_cast<std::string>(value);
            //m_stream << key << " = " << valueStr << std::endl;

            if(key == ResultKey::Validation)
            {
                m_curSolutionPassed = (valueStr == "PASSED" || valueStr == "NO_CHECK");
            }
            else if(key == ResultKey::SolutionLibraryIndex)
            {
                m_curSolutionIdx = std::stoi(valueStr);
            }
            else if(key == ResultKey::SolutionName)
            {
                m_curSolutionName = valueStr;
            }
            else if(key == ResultKey::SpeedGFlops)
            {
                try
                {
                    double speed       = std::stod(valueStr);
                    m_curSolutionSpeed = speed;
                    //m_stream << "Fastest: " << m_fastestSolutionSpeed << std::endl;
                }
                catch(std::out_of_range const& exc)
                {
                }
            }
        }

        void LibraryUpdateReporter::reportValue_string(std::string const& key,
                                                       std::string const& value)
        {
            reportValue(key, value);
        }

        void LibraryUpdateReporter::reportValue_uint(std::string const& key, uint64_t value)
        {
            reportValue(key, value);
        }

        void LibraryUpdateReporter::reportValue_int(std::string const& key, int64_t value)
        {
            reportValue(key, value);
        }

        void LibraryUpdateReporter::reportValue_double(std::string const& key, double value)
        {
            reportValue(key, value);
        }

        void LibraryUpdateReporter::reportValue_sizes(std::string const&         key,
                                                      std::vector<size_t> const& value)
        {
            if(key == ResultKey::ProblemSizes)
            {
                m_problemSizes = value;
            }
        }

        void LibraryUpdateReporter::reportValue_vecOfSizes(
            std::string const& key, std::vector<std::vector<size_t>> const& value)
        {
            if(key == ResultKey::ProblemSizes)
            {
                m_problemSizes = value[0];
            }
        }

        void LibraryUpdateReporter::postProblem()
        {
            if(m_fastestSolutionIdx < 0)
            {
                m_stream << "# [";
                streamJoin(m_stream, m_problemSizes, ", ");
                m_stream << "] no valid solutions." << std::endl;
            }
            else
            {
                //  - - [1024, 4096, 1, 6336]
                //    - [289, 4853.07]
                m_stream << "  - - [";
                streamJoin(m_stream, m_problemSizes, ", ");
                m_stream << "]" << std::endl;
                m_stream << "    - [" << m_fastestSolutionIdx << ", " << m_fastestSolutionSpeed
                         << "]";
                if(m_addComment)
                    m_stream << " # " << m_fastestSolutionName;
                m_stream << std::endl;
            }

            // reset
            m_fastestSolutionIdx   = -1;
            m_fastestSolutionName  = "";
            m_fastestSolutionSpeed = -1.0;
        }

        void LibraryUpdateReporter::postSolution()
        {
            // cascade from BenchmarkTimer, SpeedGFlops second
            if(m_curSolutionPassed && m_curSolutionSpeed > m_fastestSolutionSpeed)
            {
                m_fastestSolutionIdx   = m_curSolutionIdx;
                m_fastestSolutionName  = m_curSolutionName;
                m_fastestSolutionSpeed = m_curSolutionSpeed;
            }

            m_curSolutionName   = "";
            m_curSolutionIdx    = -1;
            m_curSolutionSpeed  = -1.0;
            m_curSolutionPassed = false;
        }

        void LibraryUpdateReporter::finalizeReport()
        {
            // Close file if we're the last owner.
            m_ownedStream.reset();
        }
    } // namespace Client
} // namespace TensileLite
