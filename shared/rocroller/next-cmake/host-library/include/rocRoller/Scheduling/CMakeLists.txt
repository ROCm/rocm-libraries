################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the Software), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################


set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_LIB_DIR}/include/rocRoller/Scheduling")
target_sources(rocroller
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/CooperativeScheduler_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CooperativeScheduler.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MetaObserver.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MetaObserver_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/PriorityScheduler_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/PriorityScheduler.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RandomScheduler_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RandomScheduler.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RoundRobinScheduler_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RoundRobinScheduler.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Scheduler_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Scheduler.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Scheduler_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Scheduling_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Scheduling.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Scheduling_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/SequentialScheduler_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/SequentialScheduler.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WaitStateHazardCounter.hpp"
)

add_subdirectory(Costs)
