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

#include <memory>
#include <set>
#include <string>

#include <Tensile/Tensile.hpp>

namespace Tensile
{
    /**
 * \ingroup Tensile
 * \defgroup SolutionLibrary Solution Library Classes
 *
 * \copydoc Tensile::SolutionLibrary
 */

    template <typename MySolution>
    using SolutionSet = std::set<std::shared_ptr<MySolution>>;

    /**
 * \ingroup SolutionLibrary
 *
 * @brief Abstract base class for Library objects that can provide a
 * mapping from `Problem` and `Hardware` objects to Solution objects.
 *
 * A complete SolutionLibrary is a tree of objects which each handles a
 * single aspect of selecting a solution for a given problem. Each node in
 * the tree will handle an aspect such as:
 * - Compatibility with a particular model of GPU
 * - Selecting kernels that solve a particular type of problem (transpose,
 *   data type, etc.)
 * - Selecting the fastest kernel based on benchmark results or other logic
 * - Ensuring that a problem is compatible with any assumptions made by a
 *   particular kernel (e.g. size or stride requirements)
 *
 * A particular complete library might look like:
 * - Master library which manages serialization
 *   - GPU selection
 *     - Problem type selection
 *        - Predicated logic for specific sizes
 *          - Matching library based on benchmarks
 *            - Individual kernels
 *
 */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct TENSILE_API SolutionLibrary
    {
        virtual ~SolutionLibrary() = default;

        /**
   * Returns the single `Solution` object that best solves this
   * particular `problem` on this particular piece of `hardware`.
   *
   * May return `nullptr` if no such object exists.
   */
        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double* fitness = nullptr) const = 0;

        /**
   * Returns all `Solution` objects that are capable of correctly solving this
   * `problem` on this `hardware`.
   *
   * May return an empty set if no such object exists.
   */
        virtual SolutionSet<MySolution> findAllSolutions(MyProblem const& problem,
                                                         Hardware const&  hardware) const = 0;

        /**
   * Returns all `Solution` objects that are matched to the 'problem''s type
   * on this `hardware`.
   *
   * May return an empty set if no such object exists.
   */
        virtual SolutionSet<MySolution>
            findAllSolutionsMatchingType(MyProblem const& problem,
                                         Hardware const&  hardware) const = 0;

        virtual std::string type() const        = 0;
        virtual std::string description() const = 0;
    };

} // namespace Tensile
