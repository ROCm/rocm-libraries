// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// ALMIOPEN-2812 POC: phase-separated, device-free timing of hip-kernel-provider descriptor
// loading. One process per measurement: discovery is memoized per process.
//   dlopen           -> library load only (no discovery)
//   GetAllEngineIds  -> discoverDescriptorSets(): parse + validate + resolve
//   GetEngineName    -> the admitted engine list, printed so a step cannot silently drop one
//   Create           -> Container ctor: per-engine state-manager construction
// Peak RSS is VmHWM of this process; ru_maxrss would survive exec and report the parent's.
// Run: HIPDNN_LOG_LEVEL=off HIPDNN_DESCRIPTOR_DIR=<root> hipdnn_poc_plugin_load_timing <plugin.so>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <string>
#include <sys/prctl.h>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

double ms(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

long vmHwmKb()
{
    FILE* f = std::fopen("/proc/self/status", "r");
    char line[256];
    long kb = -1;
    while(f != nullptr && std::fgets(line, sizeof line, f) != nullptr)
    {
        if(std::sscanf(line, "VmHWM: %ld kB", &kb) == 1)
        {
            break;
        }
    }
    if(f != nullptr)
    {
        std::fclose(f);
    }
    return kb;
}

} // namespace

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "usage: %s <plugin.so>\n", argv[0]);
        return 2;
    }
    // yama ptrace_scope=1 only lets ancestors attach; allow a sampling gdb that is not one.
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
    using GetIds = int (*)(int64_t*, uint32_t, uint32_t*);
    using GetName = int (*)(int64_t, const char**);
    using Create = int (*)(void**);
    using Destroy = int (*)(void*);

    const auto t0 = Clock::now();
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    const auto t1 = Clock::now();
    if(lib == nullptr)
    {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }
    auto getIds = reinterpret_cast<GetIds>(dlsym(lib, "hipdnnEnginePluginGetAllEngineIds"));
    auto getName = reinterpret_cast<GetName>(dlsym(lib, "hipdnnEnginePluginGetEngineName"));
    auto create = reinterpret_cast<Create>(dlsym(lib, "hipdnnEnginePluginCreate"));
    auto destroy = reinterpret_cast<Destroy>(dlsym(lib, "hipdnnEnginePluginDestroy"));
    if(getIds == nullptr || getName == nullptr || create == nullptr || destroy == nullptr)
    {
        std::fprintf(stderr, "missing plugin symbol\n");
        return 1;
    }
    const long hwmAfterDlopen = vmHwmKb();

    uint32_t count = 0;
    const auto t2 = Clock::now();
    const int rcIds = getIds(nullptr, 0, &count);
    const auto t3 = Clock::now();
    const long hwmAfterDiscover = vmHwmKb();

    std::vector<int64_t> ids(count);
    uint32_t got = 0;
    getIds(ids.data(), count, &got);
    std::string names;
    for(const auto id : ids)
    {
        const char* name = nullptr;
        if(getName(id, &name) == 0 && name != nullptr)
        {
            names += (names.empty() ? "\"" : ", \"") + std::string(name) + "\"";
        }
    }

    void* handle = nullptr;
    const auto t4 = Clock::now();
    const int rcCreate = create(&handle);
    const auto t5 = Clock::now();
    const long hwmAfterCreate = vmHwmKb();
    if(rcCreate == 0)
    {
        destroy(handle);
    }

    std::printf("{\"dlopen_ms\": %.2f, \"discover_ms\": %.2f, \"create_ms\": %.2f, "
                "\"engines\": %u, \"rc_ids\": %d, \"rc_create\": %d, \"hwm_dlopen_mib\": %.1f, "
                "\"hwm_discover_mib\": %.1f, \"hwm_create_mib\": %.1f, \"engine_names\": [%s]}\n",
                ms(t0, t1),
                ms(t2, t3),
                ms(t4, t5),
                count,
                rcIds,
                rcCreate,
                static_cast<double>(hwmAfterDlopen) / 1024.0,
                static_cast<double>(hwmAfterDiscover) / 1024.0,
                static_cast<double>(hwmAfterCreate) / 1024.0,
                names.c_str());
    return 0;
}
