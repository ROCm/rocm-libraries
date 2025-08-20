/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

/*! \file
 *  \brief rocsparse-debugging.h provides debugging functions in rocsparse
 */

#ifndef ROCSPARSE_DEBUGGING_H
#define ROCSPARSE_DEBUGGING_H

#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 * \brief Enable debug kernel launch.
 * \details If the debug kernel launch is enabled then hip errors are checked before and
 *          after every kernel launch.
 * \note This routine ignores the environment variable ROCSPARSE_DEBUG_KERNEL_LAUNCH.
 */
ROCSPARSE_EXPORT
void rocsparse_enable_debug_kernel_launch();

/*! \ingroup aux_module
 * \brief Disable debug kernel launch.
 *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_KERNEL_LAUNCH.
 */
ROCSPARSE_EXPORT void rocsparse_disable_debug_kernel_launch();

/*! \ingroup aux_module
 * \details Query whether debugging kernel launch has been enabled. See \ref rocsparse_enable_debug_kernel_launch.
 * \return 1 if enabled, 0 otherwise.
 */
ROCSPARSE_EXPORT int rocsparse_state_debug_kernel_launch();

/*! \ingroup aux_module
 *  \brief Enable debug arguments.
 * \details If the debug arguments is enabled then messages are displayed when errors occur during argument checking.
 *          It provide information to the user depending of the setup of the verbosity
 * \ref rocsparse_enable_debug_arguments_verbose, \ref rocsparse_disable_debug_arguments_verbose and \ref rocsparse_state_debug_arguments_verbose.
 * \note This routine ignores the environment variable ROCSPARSE_DEBUG_ARGUMENTS.
 * \note This routine enables debug arguments verbose with \ref rocsparse_enable_debug_arguments_verbose.
 */
ROCSPARSE_EXPORT
void rocsparse_enable_debug_arguments();

/*! \ingroup aux_module
 *  \brief Disable debug arguments.
 *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_ARGUMENTS.
 *  \note This routines disables debug arguments verbose.
 */
ROCSPARSE_EXPORT
void rocsparse_disable_debug_arguments();

/*! \ingroup aux_module
 *  \details Query whether debugging arguments has been enabled. See \ref rocsparse_enable_debug_arguments.
 * \return 1 if enabled, 0 otherwise.
 */
ROCSPARSE_EXPORT
int rocsparse_state_debug_arguments();

/*! \ingroup aux_module
 *  \brief Enable debug arguments verbose.
 *  \details If the debug arguments (verbose) is enabled then messages are displayed when errors occur during argument checking.
 *           It provide information to the user depending of the setup of the verbosity
 *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_ARGUMENTS_VERBOSE)
 */
ROCSPARSE_EXPORT
void rocsparse_enable_debug_arguments_verbose();

/*! \ingroup aux_module
 *  \brief Disable debug arguments verbose.
 *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_ARGUMENTS_VERBOSE)
 */
ROCSPARSE_EXPORT
void rocsparse_disable_debug_arguments_verbose();

/*! \ingroup aux_module
 * \details Query whether debugging arguments (verbose) has been enabled. See \ref rocsparse_enable_debug_arguments_verbose.
 * \return 1 if enabled, 0 otherwise.
 */
ROCSPARSE_EXPORT
int rocsparse_state_debug_arguments_verbose();

/*! \ingroup aux_module
 *  \brief Enable debug.
 * \details If the debug is enabled then code traces are generated when unsuccessful status returns occur. It provides information to the user depending of the set of the verbosity
 * (\ref rocsparse_enable_debug_verbose, \ref rocsparse_disable_debug_verbose and \ref rocsparse_state_debug_verbose).
 *  \note This routine ignores the environment variable ROCSPARSE_DEBUG.
 * \note \ref rocsparse_enable_debug_verbose and \ref rocsparse_enable_debug_arguments are called.
 */
ROCSPARSE_EXPORT
void rocsparse_enable_debug();

/*! \ingroup aux_module
 *  \brief Disable debug.
 *  \note This routine also disables debug arguments with \ref rocsparse_disable_debug_arguments.
 *  \note This routine ignores the environment variable ROCSPARSE_DEBUG.
 */
ROCSPARSE_EXPORT
void rocsparse_disable_debug();

/*! \ingroup aux_module
 * \details Query whether debug has been enabled. See \ref rocsparse_enable_debug.
 * \return 1 if enabled, 0 otherwise.
 */
ROCSPARSE_EXPORT
int rocsparse_state_debug();

/*! \ingroup aux_module
 *  \brief Enable debug warnings
 * \details If the debug warnings are enabled, then some specific warnings could be printed during the execution.
 *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_WARNINGS.
 */
ROCSPARSE_EXPORT
void rocsparse_enable_debug_warnings();

/*! \ingroup aux_module
 *  \brief Disable debug warnings
 *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_WARNINGS.
 */
ROCSPARSE_EXPORT
void rocsparse_disable_debug_warnings();

/*! \ingroup aux_module
 *  \brief Enable debug verbose.
 *  \details The debug verbose displays a stack of code traces showing where the code is handling a unsuccessful status.
 *  \note This routine enables debug arguments verbose with \ref rocsparse_enable_debug_arguments_verbose.
 *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_VERBOSE.
 */
ROCSPARSE_EXPORT
void rocsparse_enable_debug_verbose();

/*! \ingroup aux_module
 *  \brief Disable debug verbose.
 *  \note This routine disables debug arguments verbose with  \ref rocsparse_disable_debug_arguments.
 *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_VERBOSE.
 */
ROCSPARSE_EXPORT
void rocsparse_disable_debug_verbose();
/*! \ingroup aux_module
 * \details Query whether debug (verbose) has been enabled. See \ref rocsparse_enable_debug_verbose.
 * \return 1 if enabled, 0 otherwise.
 */
ROCSPARSE_EXPORT
int rocsparse_state_debug_verbose();

/*! \ingroup aux_module
 *  \brief Enable debug force host assert.
 *  \details The debug force host assert forces the evaluation of assert on host when the compiler directive NDEBUG is used.
 */
ROCSPARSE_EXPORT
void rocsparse_enable_debug_force_host_assert();

/*! \ingroup aux_module
 *  \brief Disable debug force host assert.
 */
ROCSPARSE_EXPORT
void rocsparse_disable_debug_force_host_assert();

/*! \ingroup aux_module
 * \details Query whether debug force host assert has been enabled. See \ref rocsparse_enable_debug_force_host_assert.
 * \return 1 if enabled, 0 otherwise.
 */
ROCSPARSE_EXPORT
int rocsparse_state_debug_force_host_assert();

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_DEBUGGING_H */
