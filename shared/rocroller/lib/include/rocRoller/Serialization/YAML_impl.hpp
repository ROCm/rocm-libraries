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

#include <rocRoller/Serialization/YAML.hpp>

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
            // Don't hard-wrap until we get to 5000 lines.
            llvm::yaml::Output yout(sout, nullptr, 5000);
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
            auto reader = llvm::MemoryBuffer::getFile(filename);

            if(!reader || !*reader)
            {
                Throw<FatalError>("Can not read YAML file!", ShowValue(filename));
            }

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
