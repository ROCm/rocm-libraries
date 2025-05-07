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
#include "functions/f_branch.hpp"
#include "base.hpp"
#include "code.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace rocisa
{
    std::shared_ptr<Module> BranchIfZero(const std::string& sgprName,
                                         const DataType     computeDataType,
                                         const int          tmpSgprIdx,
                                         const int          laneSC,
                                         const Label&       label,
                                         const int          waveFrontSize)
    {
        auto        module  = std::make_shared<Module>("BranchIfZero");
        std::string sgprStr = "s[" + sgprName + "]";

        auto pVCC = MAKE(VCC);
        if(computeDataType == DataType::ComplexDouble)
        {
            auto tmpSgpr = sgpr(tmpSgprIdx, laneSC);
            module->addT<VCmpEQF64>(
                tmpSgpr, sgpr(sgprName, 2), 0.0, std::nullopt, sgprStr + ".real == 0.0 ?");
            std::string sgprVar = (std::is_same<decltype(sgprName), std::string>::value)
                                      ? sgprName + "+2"
                                      : std::to_string(std::stoi(sgprName) + 2);
            module->addT<VCmpEQF64>(
                pVCC, sgpr(sgprVar, 2), 0.0, std::nullopt, sgprStr + ".imag == 0.0 ?");
            if(waveFrontSize == 32)
            {
                module->addT<SAndB32>(tmpSgpr, pVCC, tmpSgpr, sgprStr + " == 0 ?");
                module->addT<SCmpEQU32>(tmpSgpr, 0, "branch if " + sgprStr + " == 0");
            }
            else
            {
                module->addT<SAndB64>(tmpSgpr, pVCC, tmpSgpr, sgprStr + " == 0 ?");
                module->addT<SCmpEQU64>(tmpSgpr, 0, "branch if " + sgprStr + " == 0");
            }
            module->addT<SCBranchSCC0>(label.getLabelName(), "branch if " + sgprStr + " == 0");
        }
        else if(computeDataType == DataType::Double)
        {
            module->addT<VCmpEQF64>(
                pVCC, sgpr(sgprName, 2), 0.0, std::nullopt, sgprStr + " == 0.0 ?");
            module->addT<SCBranchVCCNZ>(label.getLabelName(), "branch if " + sgprStr + " == 0");
        }
        else if(computeDataType == DataType::ComplexFloat)
        {
            auto tmpSgpr = sgpr(tmpSgprIdx, laneSC);
            module->addT<VCmpEQF32>(
                tmpSgpr, sgpr(sgprName), 0.0, std::nullopt, sgprStr + ".real == 0.0f ?");
            std::string sgprVar = (std::is_same<decltype(sgprName), std::string>::value)
                                      ? sgprName + "+1"
                                      : std::to_string(std::stoi(sgprName) + 1);
            module->addT<VCmpEQF32>(
                pVCC, sgpr(sgprVar), 0.0, std::nullopt, sgprStr + ".imag == 0.0f ?");
            if(waveFrontSize == 32)
            {
                module->addT<SAndB32>(tmpSgpr, pVCC, tmpSgpr, sgprStr + " == 0 ?");
                module->addT<SCmpEQU32>(tmpSgpr, 0, "branch if " + sgprStr + " == 0");
            }
            else
            {
                module->addT<SAndB64>(tmpSgpr, pVCC, tmpSgpr, sgprStr + " == 0 ?");
                module->addT<SCmpEQU64>(tmpSgpr, 0, "branch if " + sgprStr + " == 0");
            }
            module->addT<SCBranchSCC0>(label.getLabelName(), "branch if " + sgprStr + " == 0");
        }
        else if(computeDataType == DataType::Float || computeDataType == DataType::Half
                || computeDataType == DataType::BFloat16)
        {
            module->addT<VCmpEQF32>(
                pVCC, sgpr(sgprName), 0.0, std::nullopt, sgprStr + " == 0.0f ?");
            module->addT<SCBranchVCCNZ>(label.getLabelName(), "branch if " + sgprStr + " == 0");
        }
        else if(computeDataType == DataType::Int32)
        {
            module->addT<SCmpEQU32>(sgpr(sgprName), 0, sgprStr + " == 0 ?");
            module->addT<SCBranchSCC1>(label.getLabelName(), "branch if " + sgprStr + " == 0");
        }
        else if(computeDataType == DataType::Int64)
        {
            module->addT<SCmpEQU64>(sgpr(sgprName, 2), 0, sgprStr + " == 0 ?");
            module->addT<SCBranchSCC1>(label.getLabelName(), "branch if " + sgprStr + " == 0");
        }
        else
        {
            throw std::runtime_error("Unsupported compute data type: " + toString(computeDataType));
        }

        return module;
    }

    std::shared_ptr<Module> BranchIfNotZero(const std::string& sgprName,
                                            const DataType     computeDataType,
                                            const Label&       label)
    {
        auto        module  = std::make_shared<Module>("BranchIfNotZero");
        std::string sgprStr = "s[" + sgprName + "]";

        auto pVCC = MAKE(VCC);
        if(computeDataType == DataType::ComplexDouble)
        {
            module->addT<VCmpEQF64>(
                pVCC, sgpr(sgprName, 2), 0.0, std::nullopt, sgprStr + ".real == 0.0 ?");
            module->addT<SCBranchVCCZ>(label.getLabelName(), "branch if " + sgprStr + ".real != 0");
            std::string sgprVar = (std::is_same<decltype(sgprName), std::string>::value)
                                      ? sgprName + "+2"
                                      : std::to_string(std::stoi(sgprName) + 2);
            module->addT<VCmpEQF64>(
                pVCC, sgpr(sgprVar, 2), 0.0, std::nullopt, sgprStr + ".imag == 0.0 ?");
            module->addT<SCBranchVCCZ>(label.getLabelName(), "branch if " + sgprStr + ".imag != 0");
        }
        else if(computeDataType == DataType::Double)
        {
            module->addT<VCmpEQF64>(
                pVCC, sgpr(sgprName, 2), 0.0, std::nullopt, sgprStr + " == 0.0 ?");
            module->addT<SCBranchVCCZ>(label.getLabelName(), "branch if " + sgprStr + " != 0");
        }
        else if(computeDataType == DataType::ComplexFloat)
        {
            module->addT<VCmpEQF32>(
                pVCC, sgpr(sgprName), 0.0, std::nullopt, sgprStr + ".real == 0.0f ?");
            module->addT<SCBranchVCCZ>(label.getLabelName(), "branch if " + sgprStr + ".real != 0");
            std::string sgprVar = (std::is_same<decltype(sgprName), std::string>::value)
                                      ? sgprName + "+1"
                                      : std::to_string(std::stoi(sgprName) + 1);
            module->addT<VCmpEQF32>(
                pVCC, sgpr(sgprVar), 0.0, std::nullopt, sgprStr + ".imag == 0.0f ?");
            module->addT<SCBranchVCCZ>(label.getLabelName(), "branch if " + sgprStr + ".imag != 0");
        }
        else if(computeDataType == DataType::Float || computeDataType == DataType::Half
                || computeDataType == DataType::BFloat16)
        {
            module->addT<VCmpEQF32>(
                pVCC, sgpr(sgprName), 0.0, std::nullopt, sgprStr + " == 0.0f ?");
            module->addT<SCBranchVCCZ>(label.getLabelName(), "branch if " + sgprStr + " != 0");
        }
        else if(computeDataType == DataType::Int64)
        {
            module->addT<SCmpEQU64>(sgpr(sgprName, 2), 0, sgprStr + " == 0 ?");
            module->addT<SCBranchSCC0>(label.getLabelName(), "branch if " + sgprStr + " != 0");
        }
        else
        {
            module->addT<SCmpEQU32>(sgpr(sgprName), 0, sgprStr + " == 0 ?");
            module->addT<SCBranchSCC0>(label.getLabelName(), "branch if " + sgprStr + " != 0");
        }
        return module;
    }
} // namespace rocisa

void branch_func(nb::module_ m)
{
    m.def("BranchIfZero",
          &rocisa::BranchIfZero,
          "Branch if the given SGPR is zero",
          nb::arg("sgprName"),
          nb::arg("computeDataType"),
          nb::arg("tmpSgprIdx"),
          nb::arg("laneSC"),
          nb::arg("label"),
          nb::arg("waveFrontSize"));
    m.def("BranchIfNotZero",
          &rocisa::BranchIfNotZero,
          "Branch if the given SGPR is not zero",
          nb::arg("sgprName"),
          nb::arg("computeDataType"),
          nb::arg("label"));
}
