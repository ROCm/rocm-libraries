/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

namespace TensileLite
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
        namespace Contraction
        {
            struct WorkspaceCheck : public Predicate_CRTP<WorkspaceCheck, Task>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

#define MAX_GSU_WORKSPACE_SIZE 128 * 1024 * 1024

                WorkspaceCheck() = default;

                static std::string Type()
                {
                    return "WorkspaceCheck";
                }

                virtual bool operator()(Task const& task) const override
                {
                    size_t gsu = task.problem.getParams().gsu() != 0
                                     ? task.problem.getParams().gsu()
                                     : task.solution.calculateAutoGSU(task.problem, &task.hardware);

                    size_t gsuMultiplier = gsu > 1 ? gsu : 0;

                    if(task.problem.d().totalLogicalElements()
                           * task.solution.sizeMapping.workspaceSizePerElemC * gsuMultiplier
                       > MAX_GSU_WORKSPACE_SIZE)
                        return 0;

                    if(task.problem.groupedGemm())
                        return task.problem.workspaceSizeGroupedGemm()
                               <= task.problem.workspaceSize();
                    else
                        return task.solution.requiredWorkspaceSize(task.problem, task.hardware)
                               <= task.problem.workspaceSize();
                }

                virtual bool debugEval(Task const& task, std::ostream& stream) const override
                {
                    size_t gsu = task.problem.getParams().gsu() != 0
                                     ? task.problem.getParams().gsu()
                                     : task.solution.calculateAutoGSU(task.problem, &task.hardware);

                    size_t gsuMultiplier = gsu > 1 ? gsu : 0;

                    if(task.problem.d().totalLogicalElements() * gsuMultiplier
                       > MAX_GSU_WORKSPACE_SIZE)
                        return debugEvalCmp(task,
                                            stream,
                                            "prob",
                                            task.problem.d().totalLogicalElements() * gsuMultiplier,
                                            "<=",
                                            "max gsu workspace size",
                                            MAX_GSU_WORKSPACE_SIZE);

                    if(task.problem.groupedGemm())
                        return debugEvalCmp(task,
                                            stream,
                                            "prob",
                                            task.problem.workspaceSizeGroupedGemm(),
                                            "<=",
                                            "max",
                                            task.problem.workspaceSize());

                    return debugEvalCmp(
                        task,
                        stream,
                        "prob",
                        task.solution.requiredWorkspaceSize(task.problem, task.hardware),
                        "<=",
                        "max",
                        task.problem.workspaceSize());
                }
            };
        } // namespace Contraction

        /**
 * @}
 */
    } // namespace Predicates
} // namespace TensileLite
