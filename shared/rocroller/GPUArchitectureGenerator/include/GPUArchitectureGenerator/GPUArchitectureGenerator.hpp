/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <amdisa_structures.h>
#include <isa_xml_reader.h>

#include "rocRoller/GPUArchitecture/GPUArchitecture.hpp"
#include "rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp"
#include "rocRoller/GPUArchitecture/GPUCapability.hpp"
#include "rocRoller/GPUArchitecture/GPUInstructionInfo.hpp"

namespace GPUArchitectureGenerator
{
    // This is temporary storage for the architectures while generating.
    std::map<rocRoller::GPUArchitectureTarget, rocRoller::GPUArchitecture> GPUArchitectures;

    std::tuple<int, std::string> Execute(std::string const&);

    bool CheckAssembler();
    bool CheckAssembler(std::string const&);
    bool TryAssembler(std::string const&,
                      rocRoller::GPUArchitectureTarget const&,
                      std::string const&,
                      std::string const&);

    /**
     * @brief This function fills the GPUArchitectures map by querying the provided hipcc assembler
     * for functionality and loading the MRISA xml files located in xmlDir.
     *
     * @param hipcc Path to the hipcc to use to test for architecture capabilities.
     * @param xmlDir Path to the directory containing MRISA xml files.
     */
    void FillArchitectures(std::string const& hipcc, std::string const& xmlDir);

    void FillArchitectures();

    void LoadYamls(std::vector<std::string> const& yamlIns);

    void GenerateFile(std::string const&, bool asYAML = false, bool splitYAML = false);

    void AddCapability(rocRoller::GPUArchitectureTarget const&,
                       rocRoller::GPUCapability const&,
                       int);

    void AddInstructionInfo(rocRoller::GPUArchitectureTarget const&,
                            rocRoller::GPUInstructionInfo const&,
                            std::shared_ptr<amdisa::IsaSpec> const&,
                            std::map<std::string, std::string> const&);

    rocRoller::GPUInstructionInfo ConvertSpecInstruction(const amdisa::Instruction& instruction,
                                                         const std::string&         name = "");
}

#include "GPUArchitectureGenerator_impl.hpp"
