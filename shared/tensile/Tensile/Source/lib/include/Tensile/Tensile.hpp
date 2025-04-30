/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <vector>

#include <Tensile/Macros.hpp>
#include <Tensile/SolutionLibrary_fwd.hpp>
#include <Tensile/Tensile_fwd.hpp>

#include <Tensile/KernelArguments.hpp>
#include <Tensile/geom.hpp>

/**
 * \mainpage
 *
 * Tensile is a tool for creating a library of tensor contractions, including
 * matrix multiplications in a benchmark-driven manner.
 *
 * The host library contains classes and functions for selecting one or more
 * kernels to launch to solve a particular problem, and then launching those
 * kernels.  Kernels are selected based on the performance on a particular GPU
 * (from the benchmark runs) as well as the limitations and applicability of
 * the kernels to particular problem sizes.
 *
 * Design goals for the host library include:
 *  - A library that requires no generated code on the host
 *  - Kernel selection configured via a file
 *      - No recompilation needed to enable a different set of kernels.
 *  - Kernel selection & determination of kernel launches decoupled from host
 *  runtime language
 *      - Eliminate as much of the Hip and OpenCL-specific code as possible
 *  - Support future work including implicit GEMMs
 *  - Use modern C++
 *  - Eliminate the need for client code to call intricately named functions
 *    such as `tensile_Cijk_Ailk_Bljk_SB()`.
 */

/**
 * \addtogroup Tensile
 * @{
 */

/**
 * @brief Primary namespace for Tensile host code.
 */
namespace Tensile
{
    /**
     * \ingroup Tensile
     * \defgroup  Problem Problem Definition
     *
     * @brief Classes for defining problems
     */

    /**
 * \ingroup Problem
 * Base Problem class. A Problem object generically describes a problem to
 * be solved including the type of problem, all sizes and strides, but not
 * including actual pointers to data.
 */
    class TENSILE_API Problem
    {
    public:
        virtual ~Problem();
        virtual std::string description() const = 0;
    };

    /**
 * \ingroup Problem
 * Base class for problem inputs. This stores the actual pointers to the
 * data.
 */
    class TENSILE_API ProblemInputs
    {
    public:
        virtual ~ProblemInputs();
    };

    /**
 * \ingroup Tensile
 * \defgroup Launching Kernel Launching
 */

    /**
 * \ingroup Launching
 * Describes a single kernel invocation including kernel name, launch
 * bounds, and arguments.
 */
    struct TENSILE_API KernelInvocation
    {
    public:
        std::string kernelName;
        std::string codeObjectFile; //Code object file kernel is located in

        dim3   workGroupSize;
        dim3   numWorkGroups;
        dim3   numWorkItems;
        size_t sharedMemBytes = 0;

        KernelArguments args;
    };

    /**
 * \ingroup Tensile
 * \defgroup Hardware Hardware Description Classes
 */

    /**
 * \ingroup Hardware
 * Abstract base class for describing hardware capabilities and properties.
 */
    class TENSILE_API Hardware
    {
    public:
        Hardware();
        virtual ~Hardware();

        virtual size_t      id() const          = 0;
        virtual std::string description() const = 0;
        virtual std::string archName() const    = 0;
    };

    /**
 * \ingroup Solution
 * Generally encapsulates a single kernel or set of kernels that can be
 * used to solve a particular problem.
 */
    class TENSILE_API Solution
    {
    public:
        virtual ~Solution();

        virtual std::string name() const        = 0;
        virtual std::string description() const = 0;
    };

    /**
 * \ingroup Launching
 * Base class for objects capable of launching kernels based on
 * KernelArguments objects.
 */
    class TENSILE_API SolutionAdapter
    {
    public:
        virtual ~SolutionAdapter();

        virtual std::string name() const = 0;
    };

#ifdef TENSILE_DEFAULT_SERIALIZATION
    /**
 * Interface for deserializing a library file.
 */
    enum class LazyLoadingInit;

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    TENSILE_API std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
                LoadLibraryFile(std::string const& filename);

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    TENSILE_API std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
                LoadLibraryFilePreload(std::string const&                  filename,
                                       const std::vector<LazyLoadingInit>& preload);

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        LoadLibraryData(std::vector<uint8_t> const& data);
#endif
} // namespace Tensile

/** @} */
