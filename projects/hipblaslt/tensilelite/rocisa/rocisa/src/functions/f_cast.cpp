/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
#include "functions/f_cast.hpp"
#include "instruction/common.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;

namespace rocisa
{
    std::shared_ptr<Module> VSaturateCastInt(const std::shared_ptr<RegisterContainer>& SumIdxVgpr,
                                             const int                                 tmpVgprIdx,
                                             const int                                 tmpSgprIdx,
                                             const int                                 lowerBound,
                                             const int                                 upperBound,
                                             const SaturateCastType                    type,
                                             const bool                                initGpr)
    {
        // SaturateCastType = 0, normal case
        // SaturateCastType = 1, do nothing
        // SaturateCastType = 2, upperbound only
        // SaturateCastType = 3, lowerbound only
        std::string initGprStr = initGpr ? "with init gpr" : "without init gpr";
        auto        module     = std::make_shared<Module>("SaturateCastInt " + initGprStr);

        if(type == SaturateCastType::NORMAL)
        {
            auto tmpLowerBoundSgpr = sgpr(tmpSgprIdx);
            auto tmpUpperBoundVgpr = vgpr(tmpVgprIdx);

            if(initGpr)
            {
                module->addT<SMovkI32>(tmpLowerBoundSgpr, lowerBound, std::to_string(lowerBound));
                module->addT<VMovB32>(
                    tmpUpperBoundVgpr, upperBound, std::nullopt, std::to_string(upperBound));
            }

            module->addT<VMed3I32>(SumIdxVgpr,
                                   SumIdxVgpr,
                                   tmpLowerBoundSgpr,
                                   tmpUpperBoundVgpr,
                                   "x= min(" + std::to_string(upperBound) + ", max("
                                       + std::to_string(lowerBound) + ", x))");
        }
        else if(type == SaturateCastType::DO_NOTHING)
        {
            // Do nothing
        }
        else if(type == SaturateCastType::UPPER)
        {
            module->addT<VMinI32>(SumIdxVgpr,
                                  upperBound,
                                  SumIdxVgpr,
                                  std::nullopt,
                                  "x = min(" + std::to_string(upperBound) + ", x)");
        }
        else if(type == SaturateCastType::LOWER)
        {
            module->addT<VMaxI32>(SumIdxVgpr,
                                  lowerBound,
                                  SumIdxVgpr,
                                  std::nullopt,
                                  "x = max(" + std::to_string(lowerBound) + ", x)");
        }

        return module;
    }
} // namespace rocisa

void cast_func(nb::module_ m)
{
    m.def("VSaturateCastInt",
          &rocisa::VSaturateCastInt,
          "Saturate cast int",
          nb::arg("SumIdxVgpr"),
          nb::arg("tmpVgprIdx"),
          nb::arg("tmpSgprIdx"),
          nb::arg("lowerBound"),
          nb::arg("upperBound"),
          nb::arg("type")    = rocisa::SaturateCastType::NORMAL,
          nb::arg("initGpr") = true);
}
