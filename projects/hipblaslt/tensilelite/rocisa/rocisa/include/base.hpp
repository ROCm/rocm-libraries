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
#include <exception>
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

    // Read a cap value, returning `defaultValue` for an absent key; some caps are
    // absent (e.g. counter caps on archs without the matching wait op, or before
    // init()), so std::map::at() would throw here.
    inline int capOrDefault(const std::map<std::string, int>& caps,
                            const std::string&                key,
                            int                                defaultValue = 0)
    {
        auto it = caps.find(key);
        return it != caps.end() ? it->second : defaultValue;
    }

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
            addIsaInfo(isaVersion, initAsmCaps(isaVersion, assemblerPath, debug));
        }

        // Same as calling init() for each arch, but probes the assembler
        // capabilities of all not-yet-initialized ISAs concurrently. Each ISA
        // has its own probe session, so batching does not mix their results.
        void initMany(const nb::iterable& archs,
                      const std::string&  assemblerPath,
                      bool                debug = false)
        {
            std::vector<IsaVersion> isaVersions;
            for(nb::handle arch : archs)
            {
                auto       t = nb::cast<nb::tuple>(arch);
                IsaVersion isaVersion
                    = {nb::cast<int>(t[0]), nb::cast<int>(t[1]), nb::cast<int>(t[2])};
                if(m_isainfo.find(isaVersion) == m_isainfo.end()
                   && std::find(isaVersions.begin(), isaVersions.end(), isaVersion)
                          == isaVersions.end())
                    isaVersions.push_back(isaVersion);
            }
            std::vector<std::map<std::string, int>> asmCaps(isaVersions.size());
            std::vector<std::exception_ptr>         errors(isaVersions.size());
            {
                nb::gil_scoped_release release;
                // jthread joins on destruction, so the probes already started are
                // joined even if starting a later thread throws.
                std::vector<std::jthread> threads;
                for(size_t i = 0; i < isaVersions.size(); ++i)
                    threads.emplace_back([&, i]() {
                        try
                        {
                            asmCaps[i] = initAsmCaps(isaVersions[i], assemblerPath, debug);
                        }
                        catch(...)
                        {
                            errors[i] = std::current_exception();
                        }
                    });
            }
            for(auto& e : errors)
                if(e)
                    std::rethrow_exception(e);
            for(size_t i = 0; i < isaVersions.size(); ++i)
                addIsaInfo(isaVersions[i], std::move(asmCaps[i]));
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
            m_vgpridx[id] = std::move(std::map<std::string, int>());
            m_vgprmsb[id] = 0;
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

        const std::map<std::string, int>& getAsmCaps()
        {
            return m_isainfo[m_threads[std::this_thread::get_id()].isaVersion].asm_caps;
        }

        const std::map<std::string, int>& getRegCaps()
        {
            return m_isainfo[m_threads[std::this_thread::get_id()].isaVersion].reg_caps;
        }

        const std::map<std::string, int>& getArchCaps()
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

        std::map<std::string, int> getVgprIdx()
        {
            return m_vgpridx[std::this_thread::get_id()];
        }

        int getVgprMsb()
        {
            return m_vgprmsb[std::this_thread::get_id()];
        }

        // gfx1250 low-precision WMMA scaled-form workaround toggle. Set per build
        // from the concrete arch name (true only for gfx1250-strict, whose V0
        // silicon needs v_wmma_scale_* with scale=0). Deliberately NOT
        // reset by setKernel: forceScaledWMMA() is evaluated at toString() time,
        // and intervening setKernel calls (e.g. activation codegen) must not clear
        // it. Defaults false, so base gfx1250 emits plain v_wmma_*.
        bool getForceScaledWMMA()
        {
            // Lock for the lookup too: setForceScaledWMMA() may be inserting a new
            // per-thread entry on another thread, which rebalances the std::map and
            // would race this traversal (UB) even though entries are keyed by id.
            std::lock_guard<std::mutex> guard(m_mutex);
            auto it = m_forceScaledWMMA.find(std::this_thread::get_id());
            return it != m_forceScaledWMMA.end() ? it->second : false;
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

        void setVgprIdx(const std::string& s, const int idx)
        {
            std::thread::id id = std::this_thread::get_id();
            // need lock here?
            m_mutex.lock();
            m_vgpridx[id][s] = idx;
            m_mutex.unlock();
        }

        void setVgprMsb(const int msb)
        {
            std::thread::id id = std::this_thread::get_id();
            // need lock here?
            m_mutex.lock();
            m_vgprmsb[id] = msb;
            m_mutex.unlock();
        }

        void setForceScaledWMMA(const bool v)
        {
            std::thread::id id = std::this_thread::get_id();
            m_mutex.lock();
            m_forceScaledWMMA[id] = v;
            m_mutex.unlock();
        }

    private:
        rocIsa() = default;

        void addIsaInfo(const IsaVersion& isaVersion, std::map<std::string, int> asmCaps)
        {
            IsaInfo isainfo;
            isainfo.asm_caps      = std::move(asmCaps);
            isainfo.arch_caps     = initArchCaps(isaVersion);
            isainfo.reg_caps      = initRegisterCaps(isaVersion, isainfo.arch_caps);
            isainfo.asm_bugs      = initAsmBugs(isainfo.asm_caps);
            m_isainfo[isaVersion] = std::move(isainfo);
        }

        std::mutex                            m_mutex;
        std::map<std::thread::id, KernelInfo> m_threads;
        std::map<IsaVersion, IsaInfo>         m_isainfo;
        std::map<std::thread::id, OutputOptions> m_outputOptions;
        std::map<std::thread::id, std::map<std::string, int>> m_vgpridx;
        std::map<std::thread::id, int>        m_vgprmsb;
        std::map<std::thread::id, bool>       m_forceScaledWMMA;
    };

    // Scheduler-introspection seam (logical-IR migration, Phase 1).
    //
    // The TensileLite scheduler classifies instructions by concrete type (C++
    // countX<T> dynamic_cast, Python isinstance). A LogicalLeaf wrapping a
    // neutral logical instruction matches none of those, so it is miscounted
    // and scheduling diverges. These categories let the scheduler ask an Item
    // "what are you, for scheduling purposes?" instead of testing its type, so
    // a logical leaf can answer the same way the rocisa instruction it replaces
    // would. In Phase 1 only rocisa answers (reporting the exact classification
    // that today's dynamic_cast/typeid produce -> byte-identical); in Phase 2
    // LogicalLeaf overrides these to report from the logical instruction.
    enum class SchedCategory
    {
        None = 0,   // not a schedulable leaf (Module, TextBlock, ...)
        Other,      // a leaf instruction with no special scheduling category
        GlobalRead, // GlobalReadInstruction (incl. SMemLoad subtree)
        SMemLoad,   // reserved; countSMemLoad stays type-based for now
        LocalRead,  // LocalReadInstruction (incl. DSLoad subtree)
        LocalWrite, // LocalWriteInstruction (incl. DSStore subtree)
        MFMA,       // MFMA / SMFMA / MXMFMA
    };

    // Exact-identity counts the scheduler still needs (DS write width, metadata
    // v_mov_b32). Kept separate from SchedCategory because they are exact-type
    // (typeid) matches today, not inheritance categories.
    enum class SchedExactKind
    {
        None = 0,
        DSStoreB128,
        DSStoreB192,
        VMovB32,
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

        const std::map<std::string, int>& getAsmCaps() const
        {
            return rocIsa::getInstance().getAsmCaps();
        }

        const std::map<std::string, int>& getRegCaps() const
        {
            return rocIsa::getInstance().getRegCaps();
        }

        const std::map<std::string, int>& getArchCaps() const
        {
            return rocIsa::getInstance().getArchCaps();
        }

        std::map<std::string, bool> getAsmBugs() const
        {
            return rocIsa::getInstance().getAsmBugs();
        }

        std::map<std::string, int> getVgprIdx() const
        {
            return rocIsa::getInstance().getVgprIdx();
        }

        int getVgprMsb() const
        {
            return rocIsa::getInstance().getVgprMsb();
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

        // Scheduler-introspection seam (see SchedCategory above). Default: not a
        // schedulable leaf. rocisa instruction families override these.
        virtual SchedCategory schedCategory() const
        {
            return SchedCategory::None;
        }

        virtual SchedExactKind schedExactKind() const
        {
            return SchedExactKind::None;
        }

        // Physical-instruction weight for weighted local-read/write counts
        // (e.g. ds_*_b192 lowers to two physical ops). Default 1.
        virtual int schedWeight() const
        {
            return 1;
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
    std::string getGlcBitName();
    std::string getSlcBitName();
} // namespace rocisa
