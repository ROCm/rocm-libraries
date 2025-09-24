/*! \file */
/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*********************************************************
 * Declaration of the rocblaslt<->rocRoller interface layer. *
 *********************************************************/

#pragma once

/*****************************************************************************
 * WARNING: rocRoller-specific data types, functions and macros should only  *
 * be referenced from other files within the rocroller directory. They       *
 * should not be used in this header file.                                    *
 *****************************************************************************/

#include "rocblaslt.h"

void rocroller_destroy_handle(void* handle);

void rocroller_create_handle(void** handle);

rocblaslt_status
    getRocRollerBestSolutions(rocblaslt_handle                   handle,
                              const RocblasltContractionProblem& prob,
                              int                                requestedAlgoCount,
                              rocblaslt_matmul_heuristic_result  heuristicResultsArray[],
                              size_t                             maxWorkSpaceBytes,
                              int*                               returnAlgoCount);

rocblaslt_status
    getAllSolutionsRocRoller(RocblasltContractionProblem&                    prob,
                             rocblaslt_handle                                handle,
                             std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                             size_t                                          maxWorkSpaceBytes);

void getRocRollerSolutionsFromIndex(
    rocblaslt_handle                                handle,
    int                                             solutionIndex,
    std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
    size_t                                          maxWorkSpaceBytes);

rocblaslt_status isRocRollerSolutionSupported(rocblaslt_handle             handle,
                                              RocblasltContractionProblem& prob,
                                              rocblaslt_matmul_algo*       algo,
                                              size_t*                      workspaceSizeInBytes);

rocblaslt_status runRocRollerContractionProblem(rocblaslt_handle                   handle,
                                                const rocblaslt_matmul_algo*       algo,
                                                const RocblasltContractionProblem& prob);
