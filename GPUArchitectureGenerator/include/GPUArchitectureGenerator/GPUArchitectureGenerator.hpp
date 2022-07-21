
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

    std::tuple<int, std::string> Execute(std::string);

    bool CheckAssembler();
    bool CheckAssembler(std::string);
    bool TryAssembler(std::string, rocRoller::GPUArchitectureTarget, std::string, std::string);

    void FillArchitectures(std::string);

    void FillArchitectures();

    void GenerateFile(std::string const&, bool asYAML = false);

    void AddCapability(rocRoller::GPUArchitectureTarget, rocRoller::GPUCapability, int);

    void AddInstructionInfo(rocRoller::GPUArchitectureTarget, rocRoller::GPUInstructionInfo);
}

#include "GPUArchitectureGenerator_impl.hpp"
