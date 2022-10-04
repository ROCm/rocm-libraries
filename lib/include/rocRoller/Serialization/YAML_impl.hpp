
#pragma once

#include "YAML.hpp"

#ifdef ROCROLLER_USE_LLVM
#include <rocRoller/Serialization/llvm/YAML.hpp>

#include <llvm/Support/raw_os_ostream.h>
#endif
#ifdef ROCROLLER_USE_YAML_CPP
#include <rocRoller/Serialization/yaml-cpp/YAML.hpp>
#endif

namespace rocRoller
{
    namespace Serialization
    {
        template <typename T>
        std::string toYAML(T obj)
        {
            std::string rv;

#ifdef ROCROLLER_USE_LLVM
            llvm::raw_string_ostream sout(rv);
            llvm::yaml::Output       yout(sout);
            yout << obj;
#elif ROCROLLER_USE_YAML_CPP
            YAML::Emitter                emitter;
            Serialization::EmitterOutput yout(&emitter);
            yout.outputDoc(obj);
            rv                            = emitter.c_str();
#endif

            return rv;
        }

        template <typename T>
        T fromYAML(std::string const& yaml)
        {
            T rv;

#ifdef ROCROLLER_USE_LLVM
            llvm::yaml::Input yin(yaml);
            yin >> rv;

#elif ROCROLLER_USE_YAML_CPP
            auto                     node = YAML::Load(yaml);
            Serialization::NodeInput nodeInput(&node, nullptr);
            nodeInput.input(node, rv);
#endif

            return rv;
        }

        template <typename T>
        void writeYAML(std::ostream& stream, T obj)
        {
#ifdef ROCROLLER_USE_LLVM
            llvm::raw_os_ostream ostr(stream);
            llvm::yaml::Output   yout(ostr);
            yout << obj;

#elif ROCROLLER_USE_YAML_CPP
            YAML::Emitter                emitter(stream);
            Serialization::EmitterOutput yout(&emitter);
            yout.outputDoc(obj);
#endif
        }

        template <typename T>
        T readYAMLFile(std::string const& filename)
        {
            T rv;

#ifdef ROCROLLER_USE_LLVM
            auto              reader = llvm::MemoryBuffer::getFile(filename);
            llvm::yaml::Input yin(**reader);

            yin >> rv;
#elif ROCROLLER_USE_YAML_CPP
            auto                     node = YAML::LoadFile(filename);
            Serialization::NodeInput nodeInput(&node, nullptr);
            nodeInput.input(node, rv);
#endif
            return rv;
        }

    }
}
