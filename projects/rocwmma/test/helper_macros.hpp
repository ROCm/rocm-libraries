/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_TEST_HELPER_MACROS_HPP
#define ROCWMMA_TEST_HELPER_MACROS_HPP

#define ROCWMMA_CASE_BODY_ARG0(CASE_LABEL, CASE_IMPL) \
    case CASE_LABEL:                                  \
    {                                                 \
        CASE_IMPL                                     \
    }                                                 \
    break;

#define ROCWMMA_CASE_BODY_ARG1(CASE_LABEL, CASE_IMPL, CASE_IMPL_ARG0) \
    case CASE_LABEL:                                                  \
    {                                                                 \
        CASE_IMPL(CASE_IMPL_ARG0)                                     \
    }                                                                 \
    break;

#define ROCWMMA_CASE_BODY_ARG2(CASE_LABEL, CASE_IMPL, CASE_IMPL_ARG0, CASE_IMPL_ARG1) \
    case CASE_LABEL:                                                                  \
    {                                                                                 \
        CASE_IMPL(CASE_IMPL_ARG0, CASE_IMPL_ARG1)                                     \
    }                                                                                 \
    break;

#define ROCWMMA_CASE_BODY_ARG3(                                            \
    CASE_LABEL, CASE_IMPL, CASE_IMPL_ARG0, CASE_IMPL_ARG1, CASE_IMPL_ARG2) \
    case CASE_LABEL:                                                       \
    {                                                                      \
        CASE_IMPL(CASE_IMPL_ARG0, CASE_IMPL_ARG1, CASE_IMPL_ARG2)          \
    }                                                                      \
    break;

#define ROCWMMA_CASE_BODY_ARG4(                                                            \
    CASE_LABEL, CASE_IMPL, CASE_IMPL_ARG0, CASE_IMPL_ARG1, CASE_IMPL_ARG2, CASE_IMPL_ARG3) \
    case CASE_LABEL:                                                                       \
    {                                                                                      \
        CASE_IMPL(CASE_IMPL_ARG0, CASE_IMPL_ARG1, CASE_IMPL_ARG2, CASE_IMPL_ARG3)          \
    }                                                                                      \
    break;

#define ROCWMMA_SWITCH_BODY2_ARG0(SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1) \
    switch(SWITCH_ARG)                                                             \
    {                                                                              \
        ROCWMMA_CASE_BODY_ARG0(CASE_LABEL0, CASE_IMPL)                             \
        ROCWMMA_CASE_BODY_ARG0(CASE_LABEL1, CASE_IMPL)                             \
    default:;                                                                      \
    }

#define ROCWMMA_SWITCH_BODY3_ARG0(SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, CASE_LABEL2) \
    switch(SWITCH_ARG)                                                                          \
    {                                                                                           \
        ROCWMMA_CASE_BODY_ARG0(CASE_LABEL0, CASE_IMPL)                                          \
        ROCWMMA_CASE_BODY_ARG0(CASE_LABEL1, CASE_IMPL)                                          \
        ROCWMMA_CASE_BODY_ARG0(CASE_LABEL2, CASE_IMPL)                                          \
    default:;                                                                                   \
    }

#define ROCWMMA_SWITCH_BODY4_ARG0(                                             \
    SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, CASE_LABEL2, CASE_LABEL3) \
    switch(SWITCH_ARG)                                                         \
    {                                                                          \
        ROCWMMA_CASE_BODY_ARG0(CASE_LABEL0, CASE_IMPL)                         \
        ROCWMMA_CASE_BODY_ARG0(CASE_LABEL1, CASE_IMPL)                         \
        ROCWMMA_CASE_BODY_ARG0(CASE_LABEL2, CASE_IMPL)                         \
        ROCWMMA_CASE_BODY_ARG0(CASE_LABEL3, CASE_IMPL)                         \
    default:;                                                                  \
    }

#define ROCWMMA_SWITCH_BODY5_ARG0(                                                          \
    SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, CASE_LABEL2, CASE_LABEL3, CASE_LABEL4) \
    switch(SWITCH_ARG)                                                                      \
    {                                                                                       \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL0, CASE_IMPL)                                      \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL1, CASE_IMPL)                                      \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL2, CASE_IMPL)                                      \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL3, CASE_IMPL)                                      \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL4, CASE_IMPL)                                      \
    default:;                                                                               \
    }

// First argument of the case_body is ALWAYS the case label
#define ROCWMMA_SWITCH_BODY2_ARG1(SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1) \
    switch(SWITCH_ARG)                                                             \
    {                                                                              \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL0, CASE_IMPL, CASE_LABEL0)                \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL1, CASE_IMPL, CASE_LABEL1)                \
    default:;                                                                      \
    }

#define ROCWMMA_SWITCH_BODY3_ARG1(SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, CASE_LABEL2) \
    switch(SWITCH_ARG)                                                                          \
    {                                                                                           \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL0, CASE_IMPL, CASE_LABEL0)                             \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL1, CASE_IMPL, CASE_LABEL1)                             \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL2, CASE_IMPL, CASE_LABEL2)                             \
    default:;                                                                                   \
    }

#define ROCWMMA_SWITCH_BODY4_ARG1(                                             \
    SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, CASE_LABEL2, CASE_LABEL3) \
    switch(SWITCH_ARG)                                                         \
    {                                                                          \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL0, CASE_IMPL, CASE_LABEL0)            \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL1, CASE_IMPL, CASE_LABEL1)            \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL2, CASE_IMPL, CASE_LABEL2)            \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL3, CASE_IMPL, CASE_LABEL3)            \
    default:;                                                                  \
    }

#define ROCWMMA_SWITCH_BODY5_ARG1(                                                          \
    SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, CASE_LABEL2, CASE_LABEL3, CASE_LABEL4) \
    switch(SWITCH_ARG)                                                                      \
    {                                                                                       \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL0, CASE_IMPL, CASE_LABEL0)                         \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL1, CASE_IMPL, CASE_LABEL1)                         \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL2, CASE_IMPL, CASE_LABEL2)                         \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL3, CASE_IMPL, CASE_LABEL3)                         \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL4, CASE_IMPL, CASE_LABEL4)                         \
    default:;                                                                               \
    }

#define ROCWMMA_SWITCH_BODY6_ARG1(SWITCH_ARG,                       \
                                  CASE_IMPL,                        \
                                  CASE_LABEL0,                      \
                                  CASE_LABEL1,                      \
                                  CASE_LABEL2,                      \
                                  CASE_LABEL3,                      \
                                  CASE_LABEL4,                      \
                                  CASE_LABEL5)                      \
    switch(SWITCH_ARG)                                              \
    {                                                               \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL0, CASE_IMPL, CASE_LABEL0) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL1, CASE_IMPL, CASE_LABEL1) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL2, CASE_IMPL, CASE_LABEL2) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL3, CASE_IMPL, CASE_LABEL3) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL4, CASE_IMPL, CASE_LABEL4) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL5, CASE_IMPL, CASE_LABEL5) \
    default:;                                                       \
    }

#define ROCWMMA_SWITCH_BODY7_ARG1(SWITCH_ARG,                       \
                                  CASE_IMPL,                        \
                                  CASE_LABEL0,                      \
                                  CASE_LABEL1,                      \
                                  CASE_LABEL2,                      \
                                  CASE_LABEL3,                      \
                                  CASE_LABEL4,                      \
                                  CASE_LABEL5,                      \
                                  CASE_LABEL6)                      \
    switch(SWITCH_ARG)                                              \
    {                                                               \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL0, CASE_IMPL, CASE_LABEL0) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL1, CASE_IMPL, CASE_LABEL1) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL2, CASE_IMPL, CASE_LABEL2) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL3, CASE_IMPL, CASE_LABEL3) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL4, CASE_IMPL, CASE_LABEL4) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL5, CASE_IMPL, CASE_LABEL5) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL6, CASE_IMPL, CASE_LABEL6) \
    default:;                                                       \
    }

#define ROCWMMA_SWITCH_BODY8_ARG1(SWITCH_ARG,                       \
                                  CASE_IMPL,                        \
                                  CASE_LABEL0,                      \
                                  CASE_LABEL1,                      \
                                  CASE_LABEL2,                      \
                                  CASE_LABEL3,                      \
                                  CASE_LABEL4,                      \
                                  CASE_LABEL5,                      \
                                  CASE_LABEL6,                      \
                                  CASE_LABEL7)                      \
    switch(SWITCH_ARG)                                              \
    {                                                               \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL0, CASE_IMPL, CASE_LABEL0) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL1, CASE_IMPL, CASE_LABEL1) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL2, CASE_IMPL, CASE_LABEL2) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL3, CASE_IMPL, CASE_LABEL3) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL4, CASE_IMPL, CASE_LABEL4) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL5, CASE_IMPL, CASE_LABEL5) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL6, CASE_IMPL, CASE_LABEL6) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL7, CASE_IMPL, CASE_LABEL7) \
    default:;                                                       \
    }

#define ROCWMMA_SWITCH_BODY9_ARG1(SWITCH_ARG,                       \
                                  CASE_IMPL,                        \
                                  CASE_LABEL0,                      \
                                  CASE_LABEL1,                      \
                                  CASE_LABEL2,                      \
                                  CASE_LABEL3,                      \
                                  CASE_LABEL4,                      \
                                  CASE_LABEL5,                      \
                                  CASE_LABEL6,                      \
                                  CASE_LABEL7,                      \
                                  CASE_LABEL8)                      \
    switch(SWITCH_ARG)                                              \
    {                                                               \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL0, CASE_IMPL, CASE_LABEL0) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL1, CASE_IMPL, CASE_LABEL1) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL2, CASE_IMPL, CASE_LABEL2) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL3, CASE_IMPL, CASE_LABEL3) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL4, CASE_IMPL, CASE_LABEL4) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL5, CASE_IMPL, CASE_LABEL5) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL6, CASE_IMPL, CASE_LABEL6) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL7, CASE_IMPL, CASE_LABEL7) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL8, CASE_IMPL, CASE_LABEL8) \
    default:;                                                       \
    }

#define ROCWMMA_SWITCH_BODY10_ARG1(SWITCH_ARG,                      \
                                   CASE_IMPL,                       \
                                   CASE_LABEL0,                     \
                                   CASE_LABEL1,                     \
                                   CASE_LABEL2,                     \
                                   CASE_LABEL3,                     \
                                   CASE_LABEL4,                     \
                                   CASE_LABEL5,                     \
                                   CASE_LABEL6,                     \
                                   CASE_LABEL7,                     \
                                   CASE_LABEL8,                     \
                                   CASE_LABEL9)                     \
    switch(SWITCH_ARG)                                              \
    {                                                               \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL0, CASE_IMPL, CASE_LABEL0) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL1, CASE_IMPL, CASE_LABEL1) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL2, CASE_IMPL, CASE_LABEL2) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL3, CASE_IMPL, CASE_LABEL3) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL4, CASE_IMPL, CASE_LABEL4) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL5, CASE_IMPL, CASE_LABEL5) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL6, CASE_IMPL, CASE_LABEL6) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL7, CASE_IMPL, CASE_LABEL7) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL8, CASE_IMPL, CASE_LABEL8) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL9, CASE_IMPL, CASE_LABEL9) \
    default:;                                                       \
    }

#define ROCWMMA_SWITCH_BODY11_ARG1(SWITCH_ARG,                        \
                                   CASE_IMPL,                         \
                                   CASE_LABEL0,                       \
                                   CASE_LABEL1,                       \
                                   CASE_LABEL2,                       \
                                   CASE_LABEL3,                       \
                                   CASE_LABEL4,                       \
                                   CASE_LABEL5,                       \
                                   CASE_LABEL6,                       \
                                   CASE_LABEL7,                       \
                                   CASE_LABEL8,                       \
                                   CASE_LABEL9,                       \
                                   CASE_LABEL10)                      \
    switch(SWITCH_ARG)                                                \
    {                                                                 \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL0, CASE_IMPL, CASE_LABEL0)   \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL1, CASE_IMPL, CASE_LABEL1)   \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL2, CASE_IMPL, CASE_LABEL2)   \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL3, CASE_IMPL, CASE_LABEL3)   \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL4, CASE_IMPL, CASE_LABEL4)   \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL5, CASE_IMPL, CASE_LABEL5)   \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL6, CASE_IMPL, CASE_LABEL6)   \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL7, CASE_IMPL, CASE_LABEL7)   \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL8, CASE_IMPL, CASE_LABEL8)   \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL9, CASE_IMPL, CASE_LABEL9)   \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL10, CASE_IMPL, CASE_LABEL10) \
    default:;                                                         \
    }

// First arg is always case label, second is a constant
#define ROCWMMA_SWITCH_BODY2_ARG2(SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, FWD_ARG_0) \
    switch(SWITCH_ARG)                                                                        \
    {                                                                                         \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL0, CASE_IMPL, CASE_LABEL0, FWD_ARG_0)                \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL1, CASE_IMPL, CASE_LABEL1, FWD_ARG_0)                \
    default:;                                                                                 \
    }

#define ROCWMMA_SWITCH_BODY3_ARG2(                                             \
    SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, CASE_LABEL2, FWD_ARG_0)   \
    switch(SWITCH_ARG)                                                         \
    {                                                                          \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL0, CASE_IMPL, CASE_LABEL0, FWD_ARG_0) \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL1, CASE_IMPL, CASE_LABEL1, FWD_ARG_0) \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL2, CASE_IMPL, CASE_LABEL2, FWD_ARG_0) \
    default:;                                                                  \
    }

#define ROCWMMA_SWITCH_BODY4_ARG2(                                                        \
    SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, CASE_LABEL2, CASE_LABEL3, FWD_ARG_0) \
    switch(SWITCH_ARG)                                                                    \
    {                                                                                     \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL0, CASE_IMPL, CASE_LABEL0, FWD_ARG_0)            \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL1, CASE_IMPL, CASE_LABEL1, FWD_ARG_0)            \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL2, CASE_IMPL, CASE_LABEL2, FWD_ARG_0)            \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL3, CASE_IMPL, CASE_LABEL3, FWD_ARG_0)            \
    default:;                                                                             \
    }

#define ROCWMMA_SWITCH_BODY5_ARG2(SWITCH_ARG,                                  \
                                  CASE_IMPL,                                   \
                                  CASE_LABEL0,                                 \
                                  CASE_LABEL1,                                 \
                                  CASE_LABEL2,                                 \
                                  CASE_LABEL3,                                 \
                                  CASE_LABEL4,                                 \
                                  FWD_ARG_0)                                   \
    switch(SWITCH_ARG)                                                         \
    {                                                                          \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL0, CASE_IMPL, CASE_LABEL0, FWD_ARG_0) \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL1, CASE_IMPL, CASE_LABEL1, FWD_ARG_0) \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL2, CASE_IMPL, CASE_LABEL2, FWD_ARG_0) \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL3, CASE_IMPL, CASE_LABEL3, FWD_ARG_0) \
        ROCWMMA_CASE_BODY_ARG2(CASE_LABEL4, CASE_IMPL, CASE_LABEL4, FWD_ARG_0) \
    default:;                                                                  \
    }

// First arg is always case label, second and third are constants
#define ROCWMMA_SWITCH_BODY2_ARG3(                                                        \
    SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, FWD_ARG_0, FWD_ARG_1)                \
    switch(SWITCH_ARG)                                                                    \
    {                                                                                     \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL0, CASE_IMPL, CASE_LABEL0, FWD_ARG_0, FWD_ARG_1) \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL1, CASE_IMPL, CASE_LABEL1, FWD_ARG_0, FWD_ARG_1) \
    default:;                                                                             \
    }

#define ROCWMMA_SWITCH_BODY3_ARG3(                                                        \
    SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, CASE_LABEL2, FWD_ARG_0, FWD_ARG_1)   \
    switch(SWITCH_ARG)                                                                    \
    {                                                                                     \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL0, CASE_IMPL, CASE_LABEL0, FWD_ARG_0, FWD_ARG_1) \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL1, CASE_IMPL, CASE_LABEL1, FWD_ARG_0, FWD_ARG_1) \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL2, CASE_IMPL, CASE_LABEL2, FWD_ARG_0, FWD_ARG_1) \
    default:;                                                                             \
    }

#define ROCWMMA_SWITCH_BODY4_ARG3(SWITCH_ARG,                                             \
                                  CASE_IMPL,                                              \
                                  CASE_LABEL0,                                            \
                                  CASE_LABEL1,                                            \
                                  CASE_LABEL2,                                            \
                                  CASE_LABEL3,                                            \
                                  FWD_ARG_0,                                              \
                                  FWD_ARG_1)                                              \
    switch(SWITCH_ARG)                                                                    \
    {                                                                                     \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL0, CASE_IMPL, CASE_LABEL0, FWD_ARG_0, FWD_ARG_1) \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL1, CASE_IMPL, CASE_LABEL1, FWD_ARG_0, FWD_ARG_1) \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL2, CASE_IMPL, CASE_LABEL2, FWD_ARG_0, FWD_ARG_1) \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL3, CASE_IMPL, CASE_LABEL3, FWD_ARG_0, FWD_ARG_1) \
    default:;                                                                             \
    }

#define ROCWMMA_SWITCH_BODY5_ARG3(SWITCH_ARG,                                             \
                                  CASE_IMPL,                                              \
                                  CASE_LABEL0,                                            \
                                  CASE_LABEL1,                                            \
                                  CASE_LABEL2,                                            \
                                  CASE_LABEL3,                                            \
                                  CASE_LABEL4,                                            \
                                  FWD_ARG_0,                                              \
                                  FWD_ARG_1)                                              \
    switch(SWITCH_ARG)                                                                    \
    {                                                                                     \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL0, CASE_IMPL, CASE_LABEL0, FWD_ARG_0, FWD_ARG_1) \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL1, CASE_IMPL, CASE_LABEL1, FWD_ARG_0, FWD_ARG_1) \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL2, CASE_IMPL, CASE_LABEL2, FWD_ARG_0, FWD_ARG_1) \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL3, CASE_IMPL, CASE_LABEL3, FWD_ARG_0, FWD_ARG_1) \
        ROCWMMA_CASE_BODY_ARG3(CASE_LABEL4, CASE_IMPL, CASE_LABEL4, FWD_ARG_0, FWD_ARG_1) \
    default:;                                                                             \
    }

// First arg is always case label, second third and fourth are constants
#define ROCWMMA_SWITCH_BODY2_ARG4(                                                    \
    SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2) \
    switch(SWITCH_ARG)                                                                \
    {                                                                                 \
        ROCWMMA_CASE_BODY_ARG4(                                                       \
            CASE_LABEL0, CASE_IMPL, CASE_LABEL0, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2)     \
        ROCWMMA_CASE_BODY_ARG4(                                                       \
            CASE_LABEL1, CASE_IMPL, CASE_LABEL1, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2)     \
    default:;                                                                         \
    }

#define ROCWMMA_SWITCH_BODY3_ARG4(                                                                 \
    SWITCH_ARG, CASE_IMPL, CASE_LABEL0, CASE_LABEL1, CASE_LABEL2, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2) \
    switch(SWITCH_ARG)                                                                             \
    {                                                                                              \
        ROCWMMA_CASE_BODY_ARG4(                                                                    \
            CASE_LABEL0, CASE_IMPL, CASE_LABEL0, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2)                  \
        ROCWMMA_CASE_BODY_ARG4(                                                                    \
            CASE_LABEL1, CASE_IMPL, CASE_LABEL1, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2)                  \
        ROCWMMA_CASE_BODY_ARG4(                                                                    \
            CASE_LABEL2, CASE_IMPL, CASE_LABEL2, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2)                  \
    default:;                                                                                      \
    }

#define ROCWMMA_SWITCH_BODY4_ARG4(SWITCH_ARG,                                     \
                                  CASE_IMPL,                                      \
                                  CASE_LABEL0,                                    \
                                  CASE_LABEL1,                                    \
                                  CASE_LABEL2,                                    \
                                  CASE_LABEL3,                                    \
                                  FWD_ARG_0,                                      \
                                  FWD_ARG_1,                                      \
                                  FWD_ARG_2)                                      \
    switch(SWITCH_ARG)                                                            \
    {                                                                             \
        ROCWMMA_CASE_BODY_ARG4(                                                   \
            CASE_LABEL0, CASE_IMPL, CASE_LABEL0, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2) \
        ROCWMMA_CASE_BODY_ARG4(                                                   \
            CASE_LABEL1, CASE_IMPL, CASE_LABEL1, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2) \
        ROCWMMA_CASE_BODY_ARG4(                                                   \
            CASE_LABEL2, CASE_IMPL, CASE_LABEL2, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2) \
        ROCWMMA_CASE_BODY_ARG4(                                                   \
            CASE_LABEL3, CASE_IMPL, CASE_LABEL3, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2) \
    default:;                                                                     \
    }

#define ROCWMMA_SWITCH_BODY5_ARG4(SWITCH_ARG,                                     \
                                  CASE_IMPL,                                      \
                                  CASE_LABEL0,                                    \
                                  CASE_LABEL1,                                    \
                                  CASE_LABEL2,                                    \
                                  CASE_LABEL3,                                    \
                                  CASE_LABEL4,                                    \
                                  FWD_ARG_0,                                      \
                                  FWD_ARG_1,                                      \
                                  FWD_ARG_2)                                      \
    switch(SWITCH_ARG)                                                            \
    {                                                                             \
        ROCWMMA_CASE_BODY_ARG4(                                                   \
            CASE_LABEL0, CASE_IMPL, CASE_LABEL0, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2) \
        ROCWMMA_CASE_BODY_ARG4(                                                   \
            CASE_LABEL1, CASE_IMPL, CASE_LABEL1, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2) \
        ROCWMMA_CASE_BODY_ARG4(                                                   \
            CASE_LABEL2, CASE_IMPL, CASE_LABEL2, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2) \
        ROCWMMA_CASE_BODY_ARG4(                                                   \
            CASE_LABEL3, CASE_IMPL, CASE_LABEL3, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2) \
        ROCWMMA_CASE_BODY_ARG4(                                                   \
            CASE_LABEL4, CASE_IMPL, CASE_LABEL4, FWD_ARG_0, FWD_ARG_1, FWD_ARG_2) \
    default:;                                                                     \
    }

#endif // ROCWMMA_TEST_HELPER_MACROS_HPP
