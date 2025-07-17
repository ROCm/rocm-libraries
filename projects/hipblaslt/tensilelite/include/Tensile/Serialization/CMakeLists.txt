# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(tensilelite-host
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Base.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Containers.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionPredicates.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ExactLogicLibrary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FreeSizeLibrary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/GranularitySelectionLibrary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/HasTraits.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MLFeatures.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MLPClassificationLibrary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MapLibrary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MatchingLibrary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PlaceholderLibrary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Predicates.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Properties.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibrary.hpp"
)
