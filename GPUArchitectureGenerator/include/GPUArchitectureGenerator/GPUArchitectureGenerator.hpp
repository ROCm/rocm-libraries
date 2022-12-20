
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

    void FillArchitectures(std::string const&);

    void FillArchitectures();

    void GenerateFile(std::string const&, bool asYAML = false);

    void AddCapability(rocRoller::GPUArchitectureTarget const&,
                       rocRoller::GPUCapability const&,
                       int);

    void AddInstructionInfo(rocRoller::GPUArchitectureTarget const&,
                            rocRoller::GPUInstructionInfo const&);
}

#include "GPUArchitectureGenerator_impl.hpp"
