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
 *  \brief rocsparse-roctx.h provides rocTX instrumentation functions in rocsparse
 */

#ifndef ROCSPARSE_ROCTX_H
#define ROCSPARSE_ROCTX_H

#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 * \details Enable rocTX instrumentation.
 * \note This routine ignores the environment variable ROCSPARSE_ROCTX.
 */
ROCSPARSE_EXPORT
void rocsparse_enable_roctx();

/*! \ingroup aux_module
 * \details Disable rocTX instrumentation.
 * \note This routine ignores the environment variable ROCSPARSE_ROCTX.
 */
ROCSPARSE_EXPORT void rocsparse_disable_roctx();

/*! \ingroup aux_module
 * \details Query whether rocTX instrumentation has been enabled. See \ref rocsparse_enable_roctx.
 * \return 1 if enabled, 0 otherwise.
 */
ROCSPARSE_EXPORT int rocsparse_state_roctx();

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_ROCTX_H */
