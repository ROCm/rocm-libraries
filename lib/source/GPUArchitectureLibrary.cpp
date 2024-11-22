#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#include <rocRoller/Utilities/HipUtils.hpp>
#endif

namespace rocRoller
{

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

            HIP_CHECK(hipGetDeviceProperties(&deviceProps, i));

            retval.push_back(
                m_gpuArchitectures.at(GPUArchitectureTarget::fromString(deviceProps.gcnArchName)));
        }

        if(count > 0)
        {
            HIP_CHECK(hipGetDevice(&default_device));
        }
#else
        //TODO: Add a way to get specific GPUs. Maybe through env vars.
        AssertFatal(false, "Non-HIP Path Not Implemented");
#endif
    }

    GPUArchitecture GPUArchitectureLibrary::GetArch(GPUArchitectureTarget const& target)
    {
        auto iter = m_gpuArchitectures.find(target);

        AssertFatal(iter != m_gpuArchitectures.end(),
                    concatenate("Could not find info for GPU target ", target));

        return iter->second;
    }

    GPUArchitecture GPUArchitectureLibrary::GetHipDeviceArch(int deviceIdx)
    {
        hipDeviceProp_t deviceProps;

        HIP_CHECK(hipGetDeviceProperties(&deviceProps, deviceIdx));

        return GetArch(GPUArchitectureTarget::fromString(deviceProps.gcnArchName));
    }

    GPUArchitecture GPUArchitectureLibrary::GetDefaultHipDeviceArch(int& deviceIdx)
    {
        HIP_CHECK(hipGetDevice(&deviceIdx));
        return GetHipDeviceArch(deviceIdx);
    }

    GPUArchitecture GPUArchitectureLibrary::GetDefaultHipDeviceArch()
    {
        int idx;
        return GetDefaultHipDeviceArch(idx);
    }

    bool GPUArchitectureLibrary::HasHipDevice()
    {
        int idx;
        return hipGetDevice(&idx) == hipSuccess;
    }
}
