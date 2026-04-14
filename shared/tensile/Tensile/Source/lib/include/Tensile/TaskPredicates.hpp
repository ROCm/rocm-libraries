/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Predicates.hpp>
#include <Tensile/Task.hpp>

#include <vector>

namespace Tensile
{
    namespace Predicates
    {
        /**
 * \addtogroup Predicates
 * @{
 */
        /**
 * @brief Complex Predicates
 */
        namespace Complex
        {
            struct WorkspaceCheck : public Predicate_CRTP<WorkspaceCheck, Task>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                WorkspaceCheck() = default;

                static std::string Type()
                {
                    return "WorkspaceCheck";
                }

                virtual bool operator()(Task const& task) const override
                {
                    auto required
                        = task.solution.requiredWorkspaceSize(task.problem, task.hardware);
                    return required <= task.problem.workspaceSize();
                }

                virtual bool debugEval(Task const& task, std::ostream& stream) const override
                {
                    bool rv = (*this)(task);

                    auto required
                        = task.solution.requiredWorkspaceSize(task.problem, task.hardware);
                    stream << *this << ": (" << required << " <= " << task.problem.workspaceSize()
                           << ") == " << rv;

                    return rv;
                }
            };
        } // namespace Complex

        /**
 * @}
 */
    } // namespace Predicates
} // namespace Tensile
