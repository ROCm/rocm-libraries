#include "GPUArchitecture/GPUArchitectureLibrary.hpp"
#include "GPUArchitecture/GPUArchitecture.hpp"

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif

namespace rocRoller
{
    std::map<GPUArchitectureTarget, GPUArchitecture> GPUArchitectureLibrary::GPUArchitectures
        = GPUArchitectureLibrary::LoadLibrary();

    void GPUArchitectureLibrary::GetCurrentDevices(std::vector<GPUArchitecture>& retval,
                                                   int&                          default_device)
    {
        retval.clear();
        default_device = -1;
#ifdef ROCROLLER_USE_HIP
        int        count;
        hipError_t error = hipGetDeviceCount(&count);
        if(error != hipSuccess)
        {
            count = 0;
        }

        for(int i = 0; i < count; i++)
        {
            hipDeviceProp_t deviceProps;

            error = hipGetDeviceProperties(&deviceProps, i);
            if(error != hipSuccess)
            {
                throw new std::runtime_error(hipGetErrorString(error));
            }

            retval.push_back(GPUArchitectureLibrary::GPUArchitectures.at(
                GPUArchitectureTarget(deviceProps.gcnArchName)));
        }

        if(count > 0)
        {
            error = hipGetDevice(&default_device);
            if(error != hipSuccess)
            {
                throw new std::runtime_error(hipGetErrorString(error));
            }
        }
#else
        //TODO: Add a way to get specific GPUs. Maybe through env vars.
        throw new std::runtime_error("Non-HIP Path Not Implemented");
#endif
    }

    GPUArchitecture GPUArchitectureLibrary::GetArch(GPUArchitectureTarget const& target)
    {
        auto iter = GPUArchitectures.find(target);

        if(iter == GPUArchitectures.end())
        {
            throw std::runtime_error(concatenate("Could not find info for GPU target ", target));
        }

        return iter->second;
    }

    GPUArchitecture GPUArchitectureLibrary::GetArch(std::string const& archName)
    {
        GPUArchitectureTarget target(archName);

        return GetArch(target);
    }

    GPUArchitecture GPUArchitectureLibrary::GetHipDeviceArch(int deviceIdx)
    {
        hipDeviceProp_t deviceProps;

        hipError_t error = hipGetDeviceProperties(&deviceProps, deviceIdx);

        if(error != hipSuccess)
        {
            throw new std::runtime_error(hipGetErrorString(error));
        }

        return GetArch(deviceProps.gcnArchName);
    }

    GPUArchitecture GPUArchitectureLibrary::GetDefaultHipDeviceArch(int& deviceIdx)
    {
        hipError_t error = hipGetDevice(&deviceIdx);

        return GetHipDeviceArch(deviceIdx);
    }

    GPUArchitecture GPUArchitectureLibrary::GetDefaultHipDeviceArch()
    {
        int idx;
        return GetDefaultHipDeviceArch(idx);
    }
}
