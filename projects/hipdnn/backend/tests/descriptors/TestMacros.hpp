// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipdnnException.hpp"

// NOLINTBEGIN
#define ASSERT_THROW_HIPDNN_STATUS(x, status)                      \
    do                                                             \
    {                                                              \
        try                                                        \
        {                                                          \
            (x);                                                   \
            FAIL() << "Expected exception not thrown";             \
        }                                                          \
        catch(const hipdnn_backend::HipdnnException& e)            \
        {                                                          \
            ASSERT_EQ(e.getStatus(), status);                      \
        }                                                          \
        catch(...)                                                 \
        {                                                          \
            FAIL() << "Expected hipdnn_backend::Hipdnn_exception"; \
        }                                                          \
    } while(0)

// NOLINTEND
