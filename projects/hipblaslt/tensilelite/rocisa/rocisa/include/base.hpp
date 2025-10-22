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
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <typeinfo>

#include "hardware_caps.hpp"
#include "helper.hpp"

namespace nb = nanobind;

#define MAKE(FUNCTION, ...) std::make_shared<FUNCTION>(__VA_ARGS__)

namespace rocisa
{
    struct KernelInfo
    {
        IsaVersion isaVersion;
        int        wavefront;
        KernelInfo() = default;
        KernelInfo(const IsaVersion& isaVersion, int wavefront)
            : isaVersion(isaVersion)
            , wavefront(wavefront)
        {
        }
    };

    struct OutputOptions
    {
        bool outputNoComment = false;
    };

    struct IsaInfo
    {
        std::map<std::string, int>  asm_caps;
        std::map<std::string, int>  arch_caps;
        std::map<std::string, int>  reg_caps;
        std::map<std::string, bool> asm_bugs;
    };

    class rocIsa
    {
    public:
        // Delete copy constructor and assignment operator
        rocIsa(const rocIsa&)             = delete;
        rocIsa& operator=(const rocIsa&)  = delete;
        rocIsa(const rocIsa&&)            = delete;
        rocIsa& operator=(const rocIsa&&) = delete;

        // Static method to get the single instance of the class
        static rocIsa& getInstance()
        {
            static rocIsa instance;
            return instance;
        }

        void init(const nb::tuple& arch, const std::string& assemblerPath, bool debug = false)
        {
            IsaVersion isaVersion
                = {nb::cast<int>(arch[0]), nb::cast<int>(arch[1]), nb::cast<int>(arch[2])};
            if(m_isainfo.find(isaVersion) != m_isainfo.end())
                return;
            // Init ISA
            IsaInfo isainfo;
            isainfo.asm_caps      = initAsmCaps(isaVersion, assemblerPath, debug);
            isainfo.arch_caps     = initArchCaps(isaVersion);
            isainfo.reg_caps      = initRegisterCaps(isaVersion, isainfo.arch_caps);
            isainfo.asm_bugs      = initAsmBugs(isainfo.asm_caps);
            m_isainfo[isaVersion] = isainfo;
        }

        bool isInit()
        {
            return (m_isainfo.size() > 0);
        }

        void setKernel(const nb::tuple& arch, const int wavefrontSize)
        {
            std::thread::id id = std::this_thread::get_id();
            IsaVersion      isaVersion
                = {nb::cast<int>(arch[0]), nb::cast<int>(arch[1]), nb::cast<int>(arch[2])};
            m_mutex.lock();
            m_threads[id] = std::move(KernelInfo(isaVersion, wavefrontSize));
            m_mutex.unlock();
        }

        KernelInfo getKernel()
        {
            return m_threads[std::this_thread::get_id()];
        }

        IsaInfo getIsaInfo(const nb::tuple& arch)
        {
            IsaVersion isaVersion
                = {nb::cast<int>(arch[0]), nb::cast<int>(arch[1]), nb::cast<int>(arch[2])};
            return m_isainfo[isaVersion];
        }

        IsaInfo getIsaInfo(const IsaVersion& isaVersion)
        {
            return m_isainfo[isaVersion];
        }

        std::map<std::string, int> getAsmCaps()
        {
            return m_isainfo[m_threads[std::this_thread::get_id()].isaVersion].asm_caps;
        }

        std::map<std::string, int> getRegCaps()
        {
            return m_isainfo[m_threads[std::this_thread::get_id()].isaVersion].reg_caps;
        }

        std::map<std::string, int> getArchCaps()
        {
            return m_isainfo[m_threads[std::this_thread::get_id()].isaVersion].arch_caps;
        }

        std::map<std::string, bool> getAsmBugs()
        {
            return m_isainfo[m_threads[std::this_thread::get_id()].isaVersion].asm_bugs;
        }

        std::map<IsaVersion, IsaInfo> getData() const
        {
            return m_isainfo;
        }

        void setData(const std::map<IsaVersion, IsaInfo>& data)
        {
            m_isainfo = data;
        }

        void setOutputOptions(const OutputOptions& options)
        {
            std::thread::id id  = std::this_thread::get_id();
            m_outputOptions[id] = options;
        }

        OutputOptions getOutputOptions()
        {
            std::thread::id id = std::this_thread::get_id();
            if(m_outputOptions.find(id) == m_outputOptions.end())
                m_outputOptions[id] = OutputOptions();
            return m_outputOptions[id];
        }

    private:
        rocIsa() = default;

        std::mutex                            m_mutex;
        std::map<std::thread::id, KernelInfo> m_threads;
        std::map<IsaVersion, IsaInfo>         m_isainfo;

        std::map<std::thread::id, OutputOptions> m_outputOptions;
    };

    struct Item
    {
    public:
        Item(const std::string& name = "")
            : name(name)
        {
        }

        virtual ~Item() = default;

        virtual std::shared_ptr<Item> clone() const
        {
            throw std::runtime_error("clone() not implemented");
            return nullptr;
        }

        std::map<std::string, int> getAsmCaps() const
        {
            return rocIsa::getInstance().getAsmCaps();
        }

        std::map<std::string, int> getRegCaps() const
        {
            return rocIsa::getInstance().getRegCaps();
        }

        std::map<std::string, int> getArchCaps() const
        {
            return rocIsa::getInstance().getArchCaps();
        }

        std::map<std::string, bool> getAsmBugs() const
        {
            return rocIsa::getInstance().getAsmBugs();
        }

        KernelInfo kernel() const
        {
            return rocIsa::getInstance().getKernel();
        }

        virtual int countType(const nb::object& obj) const
        {
            nb::object self = nb::cast(*this);
            return static_cast<int>(nb::isinstance(self, obj));
        }

        virtual int countExactType(const std::type_info& targetType) const
        {
            return static_cast<int>(typeid(*this) == targetType);
        }

        virtual std::string toString() const
        {
            return name;
        }

        virtual std::string prettyPrint(const std::string& indent = "") const
        {
            nb::object  self = nb::cast(*this);
            std::string s
                = indent + nb::cast<std::string>(self.type().attr("__name__")) + " " + toString();
            return s;
        }

        Item*       parent = nullptr;
        std::string name;
    };

    struct DummyItem : public Item
    {
        DummyItem()
            : Item("")
        {
        }

        int countType(const nb::object& obj) const override
        {
            return 0;
        }
    };

    std::string isaToGfx(const nb::tuple& arch);
    std::string isaToGfx(const IsaVersion& arch);
    std::string getGlcBitName(bool hasGLCModifier);
    std::string getSlcBitName(bool hasGLCModifier);
} // namespace rocisa
