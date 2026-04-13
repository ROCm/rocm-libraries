// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

#include <string>

/** rr_… label from packed rocRoller solution index (negative int); uses shortRocRollerKernelNameFromSolutionIndex. */
std::string rocRollerShortKernelNameFromEncodedSolutionIndex(int encodedSolutionIndex);

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
