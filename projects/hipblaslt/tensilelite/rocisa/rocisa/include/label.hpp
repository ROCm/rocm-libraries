/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace rocisa
{
    class LabelManager
    {
    public:
        LabelManager() {}
        LabelManager(std::map<std::string, int> data, uint64_t counter)
            : m_labels(std::move(data))
            , m_counter(counter)
        {
        }

        void addName(const std::string& name)
        {
            auto it = m_labels.find(name);
            if(it != m_labels.end())
            {
                it->second += 1;
            }
            else
            {
                m_labels[name] = 0;
            }
        }

        std::string getName(const std::string& name)
        {
            if(m_labels.find(name) == m_labels.end())
                m_labels[name] = 0;
            if(m_labels[name] == 0)
                return name;
            return name + "_" + std::to_string(m_labels[name]);
        }

        std::string getNameInc(const std::string& name)
        {
            addName(name);
            if(m_labels[name] == 0)
                return name;
            return name + "_" + std::to_string(m_labels[name]);
        }

        std::string getNameIndex(const std::string& name, int index)
        {
            auto it = m_labels.find(name);
            if(it == m_labels.end())
            {
                throw std::runtime_error(
                    "You have to add a label first to get a label name with specific index.");
            }
            if(index > it->second)
            {
                throw std::runtime_error("The index " + std::to_string(index) + " exceeded. (> "
                                         + std::to_string(it->second) + ")");
            }
            if(index == 0)
            {
                return name;
            }
            return name + "_" + std::to_string(index);
        }

        std::string getUniqueName()
        {
            return getUniqueNamePrefix("label");
        }

        std::string getUniqueNamePrefix(const std::string& prefix)
        {
            std::string name;
            do
            {
                name = prefix + "_" + std::to_string(m_counter++);
            } while(m_labels.find(name) != m_labels.end());
            return getName(name);
        }

        std::map<std::string, int> getData() const
        {
            return m_labels;
        }

        uint64_t getCounter() const
        {
            return m_counter;
        }

    private:
        std::map<std::string, int> m_labels;
        uint64_t                   m_counter = 0;
    };
}
