#include "GPUArchitecture/GPUArchitecture.hpp"

#include "InstructionValues/Register.hpp"

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include <rocRoller/Utilities/Utils.hpp>

#ifdef ROCROLLER_USE_LLVM
#include <rocRoller/Serialization/llvm/YAML.hpp>
#endif
#ifdef ROCROLLER_USE_YAML_CPP
#include <rocRoller/Serialization/yaml-cpp/YAML.hpp>
#endif

#include <rocRoller/Serialization/GPUArchitecture.hpp>

namespace rocRoller
{

    std::string
        GPUArchitecture::writeYaml(std::map<GPUArchitectureTarget, GPUArchitecture> const& input)
    {
        rocRoller::GPUArchitecturesStruct tmp;
        tmp.architectures = input;
        std::string rv;
#ifdef ROCROLLER_USE_LLVM
        llvm::raw_string_ostream sout(rv);
        llvm::yaml::Output       yout(sout);
        yout << tmp;
#elif ROCROLLER_USE_YAML_CPP
        YAML::Emitter                emitter;
        Serialization::EmitterOutput yout(&emitter);
        yout.outputDoc(tmp);
        rv       = emitter.c_str();
#endif
        return rv;
    }

    std::map<GPUArchitectureTarget, GPUArchitecture>
        GPUArchitecture::readYaml(std::string const& input)
    {
        GPUArchitecturesStruct rv;
#ifdef ROCROLLER_USE_LLVM
        auto              reader = llvm::MemoryBuffer::getFile(input);
        llvm::yaml::Input yin(**reader);
#elif ROCROLLER_USE_YAML_CPP
        auto yin = YAML::LoadFile(input);
#endif
        yin >> rv;
        return rv.architectures;
    }

    std::string
        GPUArchitecture::writeMsgpack(std::map<GPUArchitectureTarget, GPUArchitecture> const& input)
    {
        std::stringstream buffer;
        msgpack::pack(buffer, input);

        return buffer.str();
    }

    std::map<GPUArchitectureTarget, GPUArchitecture>
        GPUArchitecture::readMsgpack(std::string const& input)
    {
        std::ifstream ifs(input);
        std::string buffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

        msgpack::object_handle oh  = msgpack::unpack(buffer.data(), buffer.size());
        msgpack::object const& obj = oh.get();

        auto rv = obj.as<std::map<GPUArchitectureTarget, GPUArchitecture>>();
        return rv;
    }

    bool GPUArchitecture::isSupportedConstantValue(Register::ValuePtr value) const
    {
        if(value->regType() != Register::Type::Literal)
            return false;

        return std::visit(
            [this](auto val) -> bool {
                using T = std::decay_t<decltype(val)>;
                if constexpr(std::is_pointer<T>::value || std::is_same<bool, T>::value)
                {
                    return false;
                }
                else
                {
                    return this->isSupportedConstantValue(val);
                }
            },
            value->getLiteralValue());
    }

}
