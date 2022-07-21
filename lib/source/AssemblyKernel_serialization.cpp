

#include <rocRoller/AssemblyKernel.hpp>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>

#ifdef ROCROLLER_USE_LLVM
#include <rocRoller/Serialization/llvm/YAML.hpp>
#endif
#ifdef ROCROLLER_USE_YAML_CPP
#include <rocRoller/Serialization/yaml-cpp/YAML.hpp>
#endif

#include <rocRoller/Serialization/AssemblyKernel.hpp>

namespace rocRoller
{
    std::string AssemblyKernel::amdgpu_metadata_yaml()
    {
        AssemblyKernels tmp;
        tmp.kernels = {*this};

#ifdef ROCROLLER_USE_LLVM
        std::string              rv;
        llvm::raw_string_ostream sout(rv);
        std::error_code          err;
        llvm::yaml::Output       yout(sout);
        yout << tmp;
        return rv;
#elif ROCROLLER_USE_YAML_CPP
        YAML::Emitter                emitter;
        Serialization::EmitterOutput output(&emitter);
        output.outputDoc(tmp);
        return emitter.c_str();
#endif
    }
}
