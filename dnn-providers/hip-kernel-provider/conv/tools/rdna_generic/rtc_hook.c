// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// LD_PRELOAD hook around HIPRTC for the conv compile tooling (check_generic.py, the
// generic-vs-chip timing pilot, and the code-size gate's census, ../code_gate). Not part of the
// provider; test tooling only.
//
//   HKP_RTC_LOG=<file>          append one JSON line per hiprtcCompileProgram:
//                               {"name": <program name>, "options": [<options as passed>],
//                                "arch": <offload arch compiled for>, "result": <hiprtcResult>}
//   HKP_RTC_OFFLOAD_ARCH=<arch> compile for <arch> instead of the --offload-arch the caller
//                               passed (e.g. gfx11-generic on a gfx1100 device). The caller's
//                               in-memory and disk caches still key on its own options, so give
//                               each arch its own HIPDNN_CACHE_DIR.
//   HKP_RTC_EXTRA_OPTIONS=<opts> with an override, options (space-separated) appended to the
//                               caller's. Comgr (ROCm 7.13) links no device libraries for a
//                               generic target, so a generic HIPRTC compile needs them named:
//                               -Xclang -mlink-builtin-bitcode -Xclang <rocm>/amdgcn/bitcode/X.bc
//                               for ocml, ockl, oclc_unsafe_math_off, oclc_finite_only_off,
//                               oclc_wavefrontsize64_off, oclc_isa_version_11-generic (or 12-)
//                               and oclc_abi_version_600.
//   HKP_RTC_DRY=1               record without compiling: hiprtcCompileProgram logs (each
//                               distinct program and option list once) and succeeds with an
//                               empty code object, and hipModuleLoadData, hipModuleGetFunction
//                               and hipModuleUnload accept it. Plans then build on any host, for
//                               any --offload-arch, with nothing compiled or loaded; nothing may
//                               launch. Set HIPDNN_DISABLE_CACHE=1 so no empty object is cached.
//
// Build: cc -shared -fPIC -O2 -o librtc_hook.so rtc_hook.c -ldl -lpthread

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int hiprtcResult;
typedef struct _hiprtcProgram* hiprtcProgram;

typedef hiprtcResult (*CreateFn)(
    hiprtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
typedef hiprtcResult (*CompileFn)(hiprtcProgram, int, const char* const*);

#define MAX_PROGRAMS 4096
#define NAME_LEN 256

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static hiprtcProgram g_progs[MAX_PROGRAMS];
static char g_names[MAX_PROGRAMS][NAME_LEN];
static int g_next = 0;

static void* nextSymbol(const char* name)
{
    void* fn = dlsym(RTLD_NEXT, name);
    if(fn == NULL)
    {
        fprintf(stderr, "rtc_hook: cannot resolve %s\n", name);
        abort();
    }
    return fn;
}

hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog,
                                 const char* src,
                                 const char* name,
                                 int numHeaders,
                                 const char* const* headers,
                                 const char* const* includeNames)
{
    static CreateFn real = NULL;
    if(real == NULL)
    {
        real = (CreateFn)nextSymbol("hiprtcCreateProgram");
    }
    const hiprtcResult result = real(prog, src, name, numHeaders, headers, includeNames);
    if(result == 0 && prog != NULL)
    {
        pthread_mutex_lock(&g_lock);
        const int slot = g_next;
        g_next = (g_next + 1) % MAX_PROGRAMS;
        g_progs[slot] = *prog;
        snprintf(g_names[slot], NAME_LEN, "%s", name != NULL ? name : "");
        pthread_mutex_unlock(&g_lock);
    }
    return result;
}

// The newest slot holding @p prog: a destroyed program's address is reused by a later one, so
// an older slot can hold a stale name for the same handle.
static void nameOf(hiprtcProgram prog, char* out)
{
    out[0] = '\0';
    pthread_mutex_lock(&g_lock);
    for(int k = 1; k <= MAX_PROGRAMS; ++k)
    {
        const int i = (g_next - k + MAX_PROGRAMS) % MAX_PROGRAMS;
        if(g_progs[i] == prog)
        {
            memcpy(out, g_names[i], NAME_LEN);
            break;
        }
    }
    pthread_mutex_unlock(&g_lock);
}

typedef hiprtcResult (*DestroyFn)(hiprtcProgram*);

hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog)
{
    static DestroyFn real = NULL;
    if(real == NULL)
    {
        real = (DestroyFn)nextSymbol("hiprtcDestroyProgram");
    }
    if(prog != NULL)
    {
        pthread_mutex_lock(&g_lock);
        for(int i = 0; i < MAX_PROGRAMS; ++i)
        {
            if(g_progs[i] == *prog)
            {
                g_progs[i] = NULL;
            }
        }
        pthread_mutex_unlock(&g_lock);
    }
    return real(prog);
}

static void writeJsonString(FILE* f, const char* s)
{
    fputc('"', f);
    for(; *s != '\0'; ++s)
    {
        if(*s == '"' || *s == '\\')
        {
            fputc('\\', f);
            fputc(*s, f);
        }
        else if((unsigned char)*s < 0x20)
        {
            fprintf(f, "\\u%04x", (unsigned char)*s);
        }
        else
        {
            fputc(*s, f);
        }
    }
    fputc('"', f);
}

static const char ARCH_PREFIX[] = "--offload-arch=";

static int dryRun(void)
{
    const char* dry = getenv("HKP_RTC_DRY");
    return dry != NULL && dry[0] == '1';
}

// Distinct (name, options) hashes already logged in a dry run.
#define SEEN_SLOTS (1u << 22)
static unsigned long long* g_seen = NULL;

static unsigned long long hashString(unsigned long long h, const char* s)
{
    for(; *s != '\0'; ++s)
    {
        h = (h ^ (unsigned char)*s) * 1099511628211ULL;
    }
    return (h ^ 0xffu) * 1099511628211ULL;
}

// 1 when (name, options) is new (and now remembered); the caller holds g_lock.
static int firstSighting(const char* name, int numOptions, const char* const* options)
{
    if(g_seen == NULL)
    {
        g_seen = (unsigned long long*)calloc(SEEN_SLOTS, sizeof(unsigned long long));
        if(g_seen == NULL)
        {
            return 1;
        }
    }
    unsigned long long h = hashString(14695981039346656037ULL, name);
    for(int i = 0; i < numOptions; ++i)
    {
        h = hashString(h, options[i]);
    }
    h |= 1; // 0 marks an empty slot
    for(unsigned long long i = h % SEEN_SLOTS;; i = (i + 1) % SEEN_SLOTS)
    {
        if(g_seen[i] == h)
        {
            return 0;
        }
        if(g_seen[i] == 0)
        {
            g_seen[i] = h;
            return 1;
        }
    }
}

hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions, const char* const* options)
{
    static CompileFn real = NULL;
    if(real == NULL)
    {
        real = (CompileFn)nextSymbol("hiprtcCompileProgram");
    }
    const char* override = getenv("HKP_RTC_OFFLOAD_ARCH");
    // Extra options, space-separated, appended after the caller's (only with an override).
    const char* extraEnv = getenv("HKP_RTC_EXTRA_OPTIONS");
    char* extra = NULL;
    int extraCount = 0;
    if(override != NULL && override[0] != '\0' && extraEnv != NULL && extraEnv[0] != '\0')
    {
        extra = strdup(extraEnv);
        for(const char* c = extra; *c != '\0'; ++c)
        {
            extraCount += (*c != ' ' && (c == extra || c[-1] == ' ')) ? 1 : 0;
        }
    }
    const char** used = (const char**)malloc(sizeof(char*) * (size_t)(numOptions + extraCount + 1));
    char* replaced = NULL;
    const char* arch = "";
    for(int i = 0; i < numOptions; ++i)
    {
        used[i] = options[i];
        if(strncmp(options[i], ARCH_PREFIX, sizeof(ARCH_PREFIX) - 1) == 0)
        {
            if(override != NULL && override[0] != '\0')
            {
                const size_t n = sizeof(ARCH_PREFIX) + strlen(override);
                replaced = (char*)malloc(n);
                snprintf(replaced, n, "%s%s", ARCH_PREFIX, override);
                used[i] = replaced;
            }
            arch = used[i] + sizeof(ARCH_PREFIX) - 1;
        }
    }
    int usedCount = numOptions;
    if(extra != NULL)
    {
        char* save = NULL;
        for(char* tok = strtok_r(extra, " ", &save); tok != NULL; tok = strtok_r(NULL, " ", &save))
        {
            used[usedCount++] = tok;
        }
    }
    const int dry = dryRun();
    const hiprtcResult result = dry ? 0 : real(prog, usedCount, used);

    const char* logPath = getenv("HKP_RTC_LOG");
    if(logPath != NULL && logPath[0] != '\0')
    {
        char name[NAME_LEN];
        nameOf(prog, name);
        pthread_mutex_lock(&g_lock);
        FILE* f = !dry || firstSighting(name, numOptions, options) ? fopen(logPath, "a") : NULL;
        if(f != NULL)
        {
            fputs("{\"name\": ", f);
            writeJsonString(f, name);
            fputs(", \"options\": [", f);
            for(int i = 0; i < numOptions; ++i)
            {
                if(i != 0)
                {
                    fputs(", ", f);
                }
                writeJsonString(f, options[i]);
            }
            fputs("], \"arch\": ", f);
            writeJsonString(f, arch);
            fprintf(f, ", \"result\": %d}\n", result);
            fclose(f);
        }
        pthread_mutex_unlock(&g_lock);
    }
    free(replaced);
    free(extra);
    free((void*)used);
    return result;
}

// The dry run's stand-ins: an empty code object, and a module and function that are never
// launched.
typedef hiprtcResult (*CodeSizeFn)(hiprtcProgram, size_t*);
typedef hiprtcResult (*CodeFn)(hiprtcProgram, char*);
typedef int (*ModuleLoadFn)(void**, const void*);
typedef int (*ModuleFunctionFn)(void**, void*, const char*);
typedef int (*ModuleUnloadFn)(void*);

static char g_dryModule;
static char g_dryFunction;

hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t* size)
{
    static CodeSizeFn real = NULL;
    if(dryRun())
    {
        *size = 8;
        return 0;
    }
    if(real == NULL)
    {
        real = (CodeSizeFn)nextSymbol("hiprtcGetCodeSize");
    }
    return real(prog, size);
}

hiprtcResult hiprtcGetCode(hiprtcProgram prog, char* code)
{
    static CodeFn real = NULL;
    if(dryRun())
    {
        memset(code, 0, 8);
        return 0;
    }
    if(real == NULL)
    {
        real = (CodeFn)nextSymbol("hiprtcGetCode");
    }
    return real(prog, code);
}

int hipModuleLoadData(void** module, const void* image)
{
    static ModuleLoadFn real = NULL;
    if(dryRun())
    {
        *module = &g_dryModule;
        return 0;
    }
    if(real == NULL)
    {
        real = (ModuleLoadFn)nextSymbol("hipModuleLoadData");
    }
    return real(module, image);
}

int hipModuleGetFunction(void** function, void* module, const char* name)
{
    static ModuleFunctionFn real = NULL;
    if(module == &g_dryModule)
    {
        *function = &g_dryFunction;
        return 0;
    }
    if(real == NULL)
    {
        real = (ModuleFunctionFn)nextSymbol("hipModuleGetFunction");
    }
    return real(function, module, name);
}

int hipModuleUnload(void* module)
{
    static ModuleUnloadFn real = NULL;
    if(module == &g_dryModule)
    {
        return 0;
    }
    if(real == NULL)
    {
        real = (ModuleUnloadFn)nextSymbol("hipModuleUnload");
    }
    return real(module);
}
