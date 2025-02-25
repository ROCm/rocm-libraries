
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

    void GenerateFile(std::string const&, bool asYAML = false);

    void AddCapability(rocRoller::GPUArchitectureTarget const&,
                       rocRoller::GPUCapability const&,
                       int);

    void AddInstructionInfo(rocRoller::GPUArchitectureTarget const&,
                            rocRoller::GPUInstructionInfo const&,
                            std::shared_ptr<amdisa::IsaSpec> const&,
                            std::map<std::string, std::string> const&);

    rocRoller::GPUInstructionInfo ConvertSpecInstruction(const amdisa::Instruction& instruction);
}

#include "GPUArchitectureGenerator_impl.hpp"
