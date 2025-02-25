

#include <rocRoller/AssemblyKernel.hpp>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>

#include <rocRoller/Serialization/AssemblyKernel.hpp>
#include <rocRoller/Serialization/YAML.hpp>

namespace rocRoller
{
    std::string AssemblyKernel::amdgpu_metadata_yaml()
    {
        AssemblyKernels tmp;
        tmp.kernels = {*this};

        return Serialization::toYAML(tmp);
    }

    std::string AssemblyKernel::args_string()
    {
        return Serialization::toYAML(m_arguments);
    }

    AssemblyKernels AssemblyKernels::fromYAML(std::string const& str)
    {
        return Serialization::fromYAML<AssemblyKernels>(str);
    }
}
