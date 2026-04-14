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

#ifdef TENSILE_DEFAULT_SERIALIZATION

#include <Tensile/Singleton.hpp>
#include <Tensile/Tensile.hpp>

namespace Tensile
{
    /**
 * \ingroup Tensile
 * \ingroup Embedding
 *
 * @brief Interface for retrieving a SolutionLibrary object which as been
 * stored in the executable via EmbedData/EmbeddedData.
 */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    class TENSILE_API EmbeddedLibrary : public LazySingleton<EmbeddedLibrary<MyProblem, MySolution>>
    {
    public:
        using Singleton = LazySingleton<EmbeddedLibrary<MyProblem, MySolution>>;

        /**
   * Constructs and returns a new SolutionLibrary instance from the static data.
   */
        static std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> NewLibrary();

        /**
   * Constructs and returns a new SolutionLibrary instance from the static data
   * for the specified key.
   */
        static std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
            NewLibrary(std::string const& key);

        /**
   * Constructs (if necessary) and returns the shared SolutionLibrary for this
   * problem type.
   */
        static std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> Get()
        {
            return Get("");
        }

        /**
   * Constructs (if necessary) and returns the shared SolutionLibrary for this
   * problem type and key.
   */
        static std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> Get(std::string const& key)
        {
            return Singleton::Instance().Library(key);
        }

        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> Library(std::string const& key)
        {
            std::lock_guard<std::mutex> lock(m_mutex);

            auto& ptr = m_libraries[key];

            if(!ptr)
                ptr = NewLibrary(key);

            return ptr;
        }

    private:
        friend Singleton;
        EmbeddedLibrary() = default;

        std::mutex m_mutex;
        std::unordered_map<std::string, std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>>
            m_libraries;
    };

} // namespace Tensile

#endif
