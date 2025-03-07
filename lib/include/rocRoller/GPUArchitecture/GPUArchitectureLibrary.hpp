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

#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/GPUArchitecture/GPUCapability.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Utilities/LazySingleton.hpp>

namespace rocRoller
{
    class GPUArchitectureLibrary : public LazySingleton<GPUArchitectureLibrary>
    {
    public:
        bool               HasCapability(GPUArchitectureTarget const&, GPUCapability const&);
        bool               HasCapability(GPUArchitectureTarget const&, std::string const&);
        int                GetCapability(GPUArchitectureTarget const&, GPUCapability const&);
        GPUInstructionInfo GetInstructionInfo(GPUArchitectureTarget const&, std::string const&);

        GPUArchitecture GetArch(GPUArchitectureTarget const& target);

        GPUArchitecture GetHipDeviceArch(int deviceIdx);
        GPUArchitecture GetDefaultHipDeviceArch(int& deviceIdx);
        GPUArchitecture GetDefaultHipDeviceArch();
        bool            HasHipDevice();

        void GetCurrentDevices(std::vector<GPUArchitecture>&, int&);

        std::vector<GPUArchitectureTarget> getAllSupportedISAs();
        std::vector<GPUArchitectureTarget> getCDNAISAs();
        std::vector<GPUArchitectureTarget> getMFMASupportedISAs();

        std::map<GPUArchitectureTarget, GPUArchitecture> LoadLibrary();

        GPUArchitectureLibrary()
            : m_gpuArchitectures(LoadLibrary())
        {
        }

    private:
        std::map<GPUArchitectureTarget, GPUArchitecture> m_gpuArchitectures;
    };
}

#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary_impl.hpp>
