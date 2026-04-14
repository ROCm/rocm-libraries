/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HIPSOLVER_COMPAT_H
#define HIPSOLVER_COMPAT_H

#if defined(_MSC_VER)
#pragma message(": warning: This file is deprecated. Use hipsolver-dense.h instead.")
#elif defined(__GNUC__)
#warning "This file is deprecated. Use hipsolver-dense.h instead."
#endif

#include "hipsolver-dense.h"

#endif // HIPSOLVER_COMPAT_H
