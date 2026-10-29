/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

#ifdef _WIN32
#include <process.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "rocke/abi.h"
#include "rocke/instance_hiptensor_aot.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"
#include "rocke/rocke_build_id.h"

#include "data_types.hpp"
#include "hiptensor_options.hpp"
#include "logger.hpp"
#include "platform.hpp"
#include "rocke_backend.hpp"
#include "rocke_codeobject.hpp"
#include "rocke_comgr.hpp"
#include "rocke_ew_rows.hpp"
#include "rocke_jit.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        namespace
        {
            using Digest = std::array<uint8_t, 32>;

            // The kernel family of the specialized kernels (rocKE's jit_ew.EW_JIT_ABI) and the
            // pack kernels they replace.
            constexpr char kEwJitFamily[] = ROCKE_HTAOT_EW_JIT_ABI;
            constexpr char kEwAbi[]       = ROCKE_HTAOT_EW_ABI;
            constexpr char kEwvAbi[]      = ROCKE_HTAOT_EWV_ABI;
            // The scheme of the spec keys: "<prefix>|<arch>|<class key>|<rocKE build id>|<values>".
            // v2: the kernels come from rocKE's C++ builder, named by the engine's build id (v1
            // named a recipe of a recipe file). The build id covers rocKE's builders, not this
            // file: bump the version when the values change meaning for the same class and
            // build (the axis loop of specializeStep: the vector magic numbers of m0 / s0,
            // n_in / e0 counted in vectors, the order of the class's axes), or when the lowering
            // or compile options of build() change without changing the LLVM IR. Keys of another
            // version are dropped when a cache file is read.
            constexpr char     kSpecKeyPrefix[] = "hiptensor-rocke-jit/v2";
            constexpr char     kCacheMagic[8]   = {'H', 'T', 'K', 'C', 'A', 'C', 'H', '\0'};
            constexpr uint32_t kCacheFormat     = 1u;
            constexpr uint32_t kCacheReader     = 1u;
            // The geometry kind of an entry: a kernel class (its key) specialized to constant
            // values (the only kind of format 1)
            constexpr uint8_t  kGeometryClass      = 0;
            constexpr uint8_t  kArgPointer         = 0;
            constexpr uint8_t  kArgValue           = 1;
            constexpr uint64_t kMaxCacheEntryBytes = uint64_t(1) << 30;
            constexpr uint32_t kMaxListLength      = 1u << 16;
            constexpr int      kMaxRank            = 8;

            std::string hex(uint8_t const* data, size_t size)
            {
                static constexpr char digits[] = "0123456789abcdef";
                std::string           out;
                for(size_t i = 0; i < size; i++)
                {
                    out += digits[data[i] >> 4];
                    out += digits[data[i] & 15];
                }
                return out;
            }

            std::string hex(Digest const& d)
            {
                return hex(d.data(), d.size());
            }

            bool envFlag(char const* name)
            {
                auto value = getEnvironmentVariable(name);
                return value && !value->empty() && *value != "0";
            }

            void logJit(char const* api, std::string const& msg)
            {
                Logger::instance()->logHeuristics(api, msg.c_str());
            }

            // ------------------------------------------------------------------ byte streams

            struct Writer
            {
                std::vector<uint8_t> b;

                void raw(void const* p, size_t n)
                {
                    auto const* c = static_cast<uint8_t const*>(p);
                    b.insert(b.end(), c, c + n);
                }
                void u8(uint8_t v)
                {
                    b.push_back(v);
                }
                void u32(uint32_t v)
                {
                    for(int i = 0; i < 4; i++)
                    {
                        b.push_back(uint8_t(v >> (8 * i)));
                    }
                }
                void u64(uint64_t v)
                {
                    for(int i = 0; i < 8; i++)
                    {
                        b.push_back(uint8_t(v >> (8 * i)));
                    }
                }
                void str(std::string const& s)
                {
                    u32(static_cast<uint32_t>(s.size()));
                    raw(s.data(), s.size());
                }
                void strs(std::vector<std::string> const& v)
                {
                    u32(static_cast<uint32_t>(v.size()));
                    for(auto const& s : v)
                    {
                        str(s);
                    }
                }
            };

            struct Reader
            {
                uint8_t const* p  = nullptr;
                size_t         n  = 0;
                size_t         at = 0;
                bool           ok = true;

                bool need(size_t k)
                {
                    ok = ok && k <= n - at;
                    return ok;
                }
                uint8_t u8()
                {
                    return need(1) ? p[at++] : 0;
                }
                uint32_t u32()
                {
                    if(!need(4))
                    {
                        return 0;
                    }
                    uint32_t v = 0;
                    for(int i = 0; i < 4; i++)
                    {
                        v |= uint32_t(p[at + i]) << (8 * i);
                    }
                    at += 4;
                    return v;
                }
                uint64_t u64()
                {
                    if(!need(8))
                    {
                        return 0;
                    }
                    uint64_t v = 0;
                    for(int i = 0; i < 8; i++)
                    {
                        v |= uint64_t(p[at + i]) << (8 * i);
                    }
                    at += 8;
                    return v;
                }
                std::string str()
                {
                    uint32_t len = u32();
                    if(!need(len))
                    {
                        return "";
                    }
                    std::string s(reinterpret_cast<char const*>(p + at), len);
                    at += len;
                    return s;
                }
                std::vector<std::string> strs()
                {
                    uint32_t count = u32();
                    if(count > kMaxListLength)
                    {
                        ok = false;
                        return {};
                    }
                    std::vector<std::string> out;
                    for(uint32_t i = 0; i < count && ok; i++)
                    {
                        out.push_back(str());
                    }
                    return out;
                }
                void digest(Digest& d)
                {
                    if(need(32))
                    {
                        std::memcpy(d.data(), p + at, 32);
                        at += 32;
                    }
                }
            };

            // ------------------------------------------------------------------ kernel cache

            struct DeviceClass
            {
                std::string baseArch;
                std::string features;
                uint32_t    cuCount          = 0;
                uint8_t     computePartition = 0;
                uint8_t     memoryPartition  = 0;

                bool operator==(DeviceClass const& o) const
                {
                    return baseArch == o.baseArch && features == o.features && cuCount == o.cuCount
                           && computePartition == o.computePartition
                           && memoryPartition == o.memoryPartition;
                }
            };

            DeviceClass deviceClassOf(backend::DeviceKey const& device)
            {
                DeviceClass c;
                c.baseArch = device.baseArch;
                auto colon = device.archName.find(':');
                c.features = colon == std::string::npos ? "" : device.archName.substr(colon + 1);
                c.cuCount  = static_cast<uint32_t>(std::max(device.cuCount, 0));
                return c;
            }

            // A compiled kernel of the cache.
            struct Entry
            {
                Digest                   entryId{};
                std::string              isa;
                std::string              compilerId;
                std::vector<std::string> options;
                bool                     freshProcess = false;
                Digest                   llvmSha{};
                std::vector<std::string> specKeys;
                std::string              family;
                std::string              classKey;
                std::vector<uint8_t>     code;
                Digest                   codeSha{};
                PackKernel               kernel;
                bool                     fromFile = false;
                bool                     dormant  = false;

                mutable std::mutex                   mutex;
                mutable std::map<int, hipFunction_t> functions;
                mutable std::vector<hipModule_t>     modules;

                // The kernel's function on the current device (the code object is loaded once
                // per device).
                hiptensorStatus_t function(hipFunction_t& out) const
                {
                    int device = 0;
                    if(hipGetDevice(&device) != hipSuccess)
                    {
                        return HIPTENSOR_STATUS_HIP_ERROR;
                    }
                    std::scoped_lock lock(mutex);
                    if(auto it = functions.find(device); it != functions.end())
                    {
                        out = it->second;
                        return HIPTENSOR_STATUS_SUCCESS;
                    }
                    hipModule_t module = nullptr;
                    if(hipModuleLoadData(&module, code.data()) != hipSuccess)
                    {
                        return HIPTENSOR_STATUS_HIP_ERROR;
                    }
                    if(hipModuleGetFunction(&out, module, kernel.name.c_str()) != hipSuccess)
                    {
                        static_cast<void>(hipModuleUnload(module));
                        return HIPTENSOR_STATUS_HIP_ERROR;
                    }
                    modules.push_back(module);
                    functions[device] = out;
                    return HIPTENSOR_STATUS_SUCCESS;
                }
            };

            // The entry identity: SHA-256 of a length-prefixed encoding of the fields the code
            // object is a function of.
            Digest entryIdOf(std::string const&              isa,
                             Digest const&                   llvmSha,
                             std::vector<std::string> const& options,
                             bool                            freshProcess,
                             std::string const&              compilerId)
            {
                Writer w;
                w.str("hiptensor-kernel-cache-entry/v1");
                w.str(isa);
                w.raw(llvmSha.data(), llvmSha.size());
                w.strs(options);
                w.u8(freshProcess ? 1 : 0);
                w.str(compilerId);
                return sha256(w.b.data(), w.b.size());
            }

            // The kernel argument layout an entry records.
            void writeLayout(Writer& w, PackKernel const& k)
            {
                w.u32(static_cast<uint32_t>(k.args.size()));
                for(auto const& a : k.args)
                {
                    w.str(a.name);
                    w.u8(a.pointer ? kArgPointer : kArgValue);
                    w.u32(a.size);
                    w.u32(a.offset);
                }
                w.u32(k.kernargSize);
            }

            bool sameLayout(PackKernel const& a, PackKernel const& b)
            {
                if(a.args.size() != b.args.size() || a.kernargSize != b.kernargSize)
                {
                    return false;
                }
                for(size_t i = 0; i < a.args.size(); i++)
                {
                    auto const &x = a.args[i], &y = b.args[i];
                    if(x.name != y.name || x.pointer != y.pointer || x.size != y.size
                       || x.offset != y.offset)
                    {
                        return false;
                    }
                }
                return true;
            }

#ifndef _WIN32
            // The user id unmapped ids show as inside a user namespace (a container's view of the
            // host's root-owned directories).
            uid_t overflowUid()
            {
                static uid_t const uid = [] {
                    unsigned long value = 65534;
                    if(FILE* f = std::fopen("/proc/sys/kernel/overflowuid", "r"))
                    {
                        if(std::fscanf(f, "%lu", &value) != 1)
                        {
                            value = 65534;
                        }
                        std::fclose(f);
                    }
                    return static_cast<uid_t>(value);
                }();
                return uid;
            }

            // The trust rule of the cache files: a regular file owned by the user (or root) that
            // no one else may write, in a directory of the user or root (or of an id unmapped in
            // the user namespace) that no one else may write unless it is sticky (/tmp).
            bool trustedDirectory(std::string const& path, std::string& reason)
            {
                auto        slash = path.find_last_of('/');
                std::string dir   = slash == std::string::npos ? std::string(".")
                                    : slash == 0               ? std::string("/")
                                                               : path.substr(0, slash);
                struct stat st;
                if(stat(dir.c_str(), &st) != 0)
                {
                    reason = "cannot stat the directory of " + path;
                    return false;
                }
                bool const owned
                    = st.st_uid == getuid() || st.st_uid == 0 || st.st_uid == overflowUid();
                bool const writable = (st.st_mode & (S_IWGRP | S_IWOTH)) != 0;
                if(!owned || (writable && (st.st_mode & S_ISVTX) == 0))
                {
                    reason = "the directory of " + path
                             + " is not owned by the user or is writable by others";
                    return false;
                }
                return true;
            }

            bool trustedFile(int fd, std::string const& path, std::string& reason)
            {
                struct stat st;
                if(fstat(fd, &st) != 0 || !S_ISREG(st.st_mode))
                {
                    reason = path + " is not a regular file";
                    return false;
                }
                if((st.st_uid != getuid() && st.st_uid != 0)
                   || (st.st_mode & (S_IWGRP | S_IWOTH)) != 0)
                {
                    reason = path + " is not owned by the user or is writable by others";
                    return false;
                }
                return true;
            }
#endif

            // Reads a whole file. 0: ok, 1: cannot open / not trusted (IO_ERROR).
            int readFile(std::string const& path, std::vector<uint8_t>& out, std::string& reason)
            {
#ifdef _WIN32
                std::ifstream file(path, std::ios::binary);
                if(!file)
                {
                    reason = "cannot open " + path;
                    return 1;
                }
                out.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
                return 0;
#else
                if(!trustedDirectory(path, reason))
                {
                    return 1;
                }
                int fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
                if(fd < 0)
                {
                    reason = "cannot open " + path;
                    return 1;
                }
                if(!trustedFile(fd, path, reason))
                {
                    close(fd);
                    return 1;
                }
                out.clear();
                uint8_t buf[1 << 16];
                for(;;)
                {
                    ssize_t got = read(fd, buf, sizeof(buf));
                    if(got < 0)
                    {
                        reason = "cannot read " + path;
                        close(fd);
                        return 1;
                    }
                    if(got == 0)
                    {
                        break;
                    }
                    out.insert(out.end(), buf, buf + got);
                }
                close(fd);
                return 0;
#endif
            }

            // Writes a whole file atomically (temporary file, flush, rename).
            bool writeFile(std::string const&          path,
                           std::vector<uint8_t> const& bytes,
                           std::string&                reason)
            {
                std::string tmp
                    = path + ".tmp."
#ifdef _WIN32
                      + std::to_string(_getpid())
#else
                      + std::to_string(getpid())
#endif
                      + "."
                      + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
#ifdef _WIN32
                {
                    std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
                    if(!file
                       || !file.write(reinterpret_cast<char const*>(bytes.data()),
                                      static_cast<std::streamsize>(bytes.size())))
                    {
                        reason = "cannot write " + tmp;
                        std::remove(tmp.c_str());
                        return false;
                    }
                }
                std::remove(path.c_str());
#else
                if(!trustedDirectory(path, reason))
                {
                    return false;
                }
                int fd
                    = open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644);
                if(fd < 0)
                {
                    reason = "cannot create " + tmp;
                    return false;
                }
                size_t done = 0;
                while(done < bytes.size())
                {
                    ssize_t put = write(fd, bytes.data() + done, bytes.size() - done);
                    if(put <= 0)
                    {
                        reason = "cannot write " + tmp;
                        close(fd);
                        unlink(tmp.c_str());
                        return false;
                    }
                    done += size_t(put);
                }
                if(fsync(fd) != 0 || close(fd) != 0)
                {
                    reason = "cannot flush " + tmp;
                    unlink(tmp.c_str());
                    return false;
                }
#endif
                if(std::rename(tmp.c_str(), path.c_str()) != 0)
                {
                    reason = "cannot rename " + tmp + " to " + path;
                    std::remove(tmp.c_str());
                    return false;
                }
                return true;
            }

            // The process-wide kernel cache.
            class KernelCache
            {
            public:
                static KernelCache& instance()
                {
                    static KernelCache cache;
                    return cache;
                }

                // A served entry under a spec key, or nullptr.
                Entry const* bySpecKey(std::string const& key)
                {
                    std::scoped_lock lock(mMutex);
                    auto             it = mBySpecKey.find(key);
                    return it == mBySpecKey.end() || it->second->dormant ? nullptr : it->second;
                }

                // A served entry of an identity, made reachable under `key` too, or nullptr.
                Entry const* byEntryId(Digest const& id, std::string const& key)
                {
                    std::scoped_lock lock(mMutex);
                    auto             it = mById.find(id);
                    if(it == mById.end() || it->second->dormant)
                    {
                        return nullptr;
                    }
                    alias(*it->second, key);
                    return it->second;
                }

                Entry const* insert(std::unique_ptr<Entry> entry, std::string const& key)
                {
                    std::scoped_lock lock(mMutex);
                    if(auto it = mById.find(entry->entryId); it != mById.end())
                    {
                        alias(*it->second, key);
                        return it->second;
                    }
                    mEntries.push_back(std::move(entry));
                    auto* e           = mEntries.back().get();
                    mById[e->entryId] = e;
                    alias(*e, key);
                    return e;
                }

                // Single flight: the first caller of a key returns true and must call finish();
                // the others wait for its result.
                bool begin(std::string const& key, std::shared_future<Entry const*>& wait)
                {
                    std::scoped_lock lock(mMutex);
                    if(auto it = mInFlight.find(key); it != mInFlight.end())
                    {
                        wait = it->second.future;
                        return false;
                    }
                    auto& slot  = mInFlight[key];
                    slot.future = slot.promise.get_future().share();
                    return true;
                }

                // Ends a single flight; a failure is remembered for the process when `remember`.
                void finish(std::string const& key, Entry const* entry, bool remember)
                {
                    std::promise<Entry const*> promise;
                    {
                        std::scoped_lock lock(mMutex);
                        auto             it = mInFlight.find(key);
                        if(it == mInFlight.end())
                        {
                            return;
                        }
                        promise = std::move(it->second.promise);
                        mInFlight.erase(it);
                        if(entry == nullptr && remember)
                        {
                            mRefused.insert(key);
                        }
                    }
                    promise.set_value(entry);
                }

                bool refused(std::string const& key)
                {
                    std::scoped_lock lock(mMutex);
                    return mRefused.count(key) != 0;
                }

                // Writes the entries of the device class (only `one` when it is set). `api`
                // names the caller in the log.
                hiptensorStatus_t write(std::string const& path,
                                        DeviceClass const& dc,
                                        Entry const*       one = nullptr,
                                        char const*        api = "hiptensorWriteKernelCacheToFile");
                hiptensorStatus_t read(std::string const& path,
                                       DeviceClass const& dc,
                                       char const*        api = "hiptensorReadKernelCacheFromFile");

            private:
                // Drops the spec keys of another key scheme (kSpecKeyPrefix): nothing looks them
                // up any more. The entry stays reachable by its identity.
                static void dropForeignKeys(std::vector<std::string>& keys)
                {
                    std::string const prefix = std::string(kSpecKeyPrefix) + "|";
                    keys.erase(std::remove_if(keys.begin(),
                                              keys.end(),
                                              [&](std::string const& k) {
                                                  return k.compare(0, prefix.size(), prefix) != 0;
                                              }),
                               keys.end());
                }

                void alias(Entry& e, std::string const& key)
                {
                    if(key.empty())
                    {
                        return;
                    }
                    if(std::find(e.specKeys.begin(), e.specKeys.end(), key) == e.specKeys.end())
                    {
                        e.specKeys.push_back(key);
                    }
                    mBySpecKey.emplace(key, &e);
                }

                struct Flight
                {
                    std::promise<Entry const*>       promise;
                    std::shared_future<Entry const*> future;
                };

                std::mutex                        mMutex;
                std::list<std::unique_ptr<Entry>> mEntries;
                std::map<Digest, Entry*>          mById;
                std::map<std::string, Entry*>     mBySpecKey;
                std::map<std::string, Flight>     mInFlight;
                std::set<std::string>             mRefused;
            };

            void writeHeader(Writer& w, DeviceClass const& dc, uint32_t count)
            {
                w.raw(kCacheMagic, 8);
                w.u32(kCacheFormat);
                w.u32(kCacheReader);
                w.u32(0); // flags: code objects not compressed
                w.u32(count);
                w.str(dc.baseArch);
                w.str(dc.features);
                w.u32(dc.cuCount);
                w.u8(dc.computePartition);
                w.u8(dc.memoryPartition);
                w.u32(kKernelAbi);
                w.str(backend::kRockeBackendName);
                // provenance, never compared
                auto const& ci = compilerInfo();
                w.str(std::to_string(HIPTENSOR_MAJOR_VERSION) + "."
                      + std::to_string(HIPTENSOR_MINOR_VERSION) + "."
                      + std::to_string(HIPTENSOR_PATCH_VERSION));
                w.u32(ROCKE_ABI_VERSION);
                w.u32(0); // recipe ABI: none (the kernels come from rocKE's C++ builders)
                w.str(ci.id);
                auto digest = sha256(w.b.data(), w.b.size());
                w.raw(digest.data(), digest.size());
            }

            hiptensorStatus_t KernelCache::write(std::string const& path,
                                                 DeviceClass const& dc,
                                                 Entry const*       one,
                                                 char const*        api)
            {
                std::string const isa = isaName(dc.baseArch);
                Writer            body;
                uint32_t          count = 0;
                {
                    std::scoped_lock lock(mMutex);
                    for(auto const& e : mEntries)
                    {
                        // An entry without a spec key is one read from a file of an earlier key
                        // scheme and not used since (using it adds the key): no lookup reaches
                        // it, so it is not written.
                        if(e->isa != isa || (one != nullptr && e.get() != one)
                           || e->specKeys.empty())
                        {
                            continue;
                        }
                        Writer w;
                        w.raw(e->entryId.data(), 32);
                        w.str(e->isa);
                        w.u8(0); // not a generic target
                        w.str(e->compilerId);
                        w.strs(e->options);
                        w.u8(e->freshProcess ? 1 : 0);
                        w.raw(e->llvmSha.data(), 32);
                        w.strs(e->specKeys);
                        w.strs({e->kernel.name});
                        writeLayout(w, e->kernel);
                        w.str(e->family);
                        w.u8(kGeometryClass);
                        w.str(e->classKey);
                        w.u32(0); // dynamic LDS
                        w.u64(e->code.size());
                        w.raw(e->codeSha.data(), 32);
                        w.raw(e->code.data(), e->code.size());
                        Writer framed;
                        framed.u64(8 + w.b.size() + 32);
                        framed.raw(w.b.data(), w.b.size());
                        auto digest = sha256(framed.b.data(), framed.b.size());
                        framed.raw(digest.data(), 32);
                        body.raw(framed.b.data(), framed.b.size());
                        count++;
                    }
                }
                Writer out;
                writeHeader(out, dc, count);
                out.raw(body.b.data(), body.b.size());
                std::string reason;
                if(!writeFile(path, out.b, reason))
                {
                    logJit(api, "rocKE kernel cache: " + reason);
                    return HIPTENSOR_STATUS_IO_ERROR;
                }
                logJit(api,
                       "rocKE kernel cache: wrote " + std::to_string(count) + " entries for "
                           + dc.baseArch + " to " + path);
                return HIPTENSOR_STATUS_SUCCESS;
            }

            hiptensorStatus_t
                KernelCache::read(std::string const& path, DeviceClass const& dc, char const* api)
            {
                std::vector<uint8_t> bytes;
                std::string          reason;
                if(readFile(path, bytes, reason) != 0)
                {
                    logJit(api, "rocKE kernel cache: " + reason);
                    return HIPTENSOR_STATUS_IO_ERROR;
                }
                if(bytes.empty())
                {
                    return HIPTENSOR_STATUS_SUCCESS;
                }
                Reader r{bytes.data(), bytes.size()};
                if(!r.need(8) || std::memcmp(bytes.data(), kCacheMagic, 8) != 0)
                {
                    logJit(api, "rocKE kernel cache: " + path + " is not a kernel cache file");
                    return HIPTENSOR_STATUS_IO_ERROR;
                }
                r.at                   = 8;
                uint32_t const format  = r.u32();
                uint32_t const minRead = r.u32();
                uint32_t const flags   = r.u32();
                uint32_t const count   = r.u32();
                DeviceClass    fileDc;
                fileDc.baseArch         = r.str();
                fileDc.features         = r.str();
                fileDc.cuCount          = r.u32();
                fileDc.computePartition = r.u8();
                fileDc.memoryPartition  = r.u8();
                uint32_t const abi      = r.u32();
                std::string    backend  = r.str();
                r.str(); // hipTensor version
                r.u32(); // rocKE ABI
                r.u32(); // recipe ABI (written by earlier versions)
                std::string  writerCompiler = r.str();
                size_t const headerEnd      = r.at;
                Digest       headerSha{};
                r.digest(headerSha);
                if(!r.ok || sha256(bytes.data(), headerEnd) != headerSha)
                {
                    logJit(api, "rocKE kernel cache: " + path + " has a damaged header");
                    return HIPTENSOR_STATUS_IO_ERROR;
                }
                if(minRead > kCacheReader || flags != 0)
                {
                    logJit(api,
                           "rocKE kernel cache: " + path + " was written in format "
                               + std::to_string(format) + ", which needs a reader of version "
                               + std::to_string(minRead));
                    return HIPTENSOR_STATUS_INVALID_VALUE;
                }
                if(abi != kKernelAbi || backend != backend::kRockeBackendName || !(fileDc == dc))
                {
                    logJit(api,
                           "rocKE kernel cache: " + path + " was written for another device class ("
                               + fileDc.baseArch + ":" + fileDc.features + ", "
                               + std::to_string(fileDc.cuCount) + " CUs) or kernel ABI");
                    return HIPTENSOR_STATUS_INVALID_VALUE;
                }

                auto const& ci          = compilerInfo();
                bool const  acceptStale = envFlag("HIPTENSOR_KERNEL_CACHE_ACCEPT_STALE");
                int         loaded = 0, merged = 0, skipped = 0, stale = 0;
                for(uint32_t i = 0; i < count; i++)
                {
                    size_t const   start = r.at;
                    uint64_t const size  = r.u64();
                    if(!r.ok || size < 8 + 32 || size > kMaxCacheEntryBytes
                       || size > bytes.size() - start)
                    {
                        skipped += int(count - i);
                        break;
                    }
                    size_t const end = start + size_t(size);
                    Digest       stored{};
                    std::memcpy(stored.data(), bytes.data() + end - 32, 32);
                    r.at = end;
                    if(sha256(bytes.data() + start, size_t(size) - 32) != stored)
                    {
                        skipped++;
                        continue;
                    }
                    Reader e{bytes.data(), end - 32, start + 8};
                    auto   entry = std::make_unique<Entry>();
                    e.digest(entry->entryId);
                    entry->isa = e.str();
                    e.u8(); // generic
                    entry->compilerId   = e.str();
                    entry->options      = e.strs();
                    entry->freshProcess = e.u8() != 0;
                    e.digest(entry->llvmSha);
                    entry->specKeys = e.strs();
                    dropForeignKeys(entry->specKeys);
                    auto       symbols = e.strs();
                    PackKernel stored_layout;
                    uint32_t   nargs = e.u32();
                    for(uint32_t a = 0; a < nargs && e.ok && a < kMaxListLength; a++)
                    {
                        KernelArg arg;
                        arg.name    = e.str();
                        arg.pointer = e.u8() == kArgPointer;
                        arg.size    = e.u32();
                        arg.offset  = e.u32();
                        stored_layout.args.push_back(arg);
                    }
                    stored_layout.kernargSize = e.u32();
                    entry->family             = e.str();
                    uint8_t const geometry    = e.u8();
                    entry->classKey           = e.str();
                    e.u32(); // dynamic LDS
                    uint64_t const coSize = e.u64();
                    e.digest(entry->codeSha);
                    if(!e.ok || geometry != kGeometryClass || symbols.size() != 1
                       || coSize != e.n - e.at)
                    {
                        skipped++;
                        continue;
                    }
                    entry->code.assign(bytes.data() + e.at, bytes.data() + e.at + coSize);
                    std::string why;
                    if(sha256(entry->code.data(), entry->code.size()) != entry->codeSha
                       || entryIdOf(entry->isa,
                                    entry->llvmSha,
                                    entry->options,
                                    entry->freshProcess,
                                    entry->compilerId)
                              != entry->entryId
                       || entry->isa != isaName(dc.baseArch)
                       || !readKernelMetadata(
                           entry->code.data(), entry->code.size(), symbols[0], entry->kernel, why)
                       || !sameLayout(entry->kernel, stored_layout))
                    {
                        skipped++;
                        continue;
                    }
                    entry->kernel.family  = kEwJitFamily;
                    entry->kernel.specKey = entry->classKey;
                    entry->fromFile       = true;
                    stale += entry->compilerId != ci.id ? 1 : 0;
                    entry->dormant = entry->compilerId != ci.id && !acceptStale;
                    std::scoped_lock lock(mMutex);
                    if(auto it = mById.find(entry->entryId); it != mById.end())
                    {
                        for(auto const& k : entry->specKeys)
                        {
                            alias(*it->second, k);
                        }
                        merged++;
                        continue;
                    }
                    auto keys = entry->specKeys;
                    mEntries.push_back(std::move(entry));
                    auto* added           = mEntries.back().get();
                    mById[added->entryId] = added;
                    for(auto const& k : keys)
                    {
                        if(!added->dormant)
                        {
                            mBySpecKey.emplace(k, added);
                        }
                    }
                    loaded++;
                }
                logJit(api,
                       "rocKE kernel cache: read " + path + ": " + std::to_string(loaded)
                           + " new entries, " + std::to_string(merged) + " already present, "
                           + std::to_string(stale) + " of another compiler ("
                           + (acceptStale ? "used: HIPTENSOR_KERNEL_CACHE_ACCEPT_STALE"
                                          : "kept, not used")
                           + "; the file's writer: " + writerCompiler + ", this library: " + ci.id
                           + "), " + std::to_string(skipped) + " damaged (skipped)");
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // ------------------------------------------------------------------ specialization

            size_t elementBytes(std::string const& dt)
            {
                return dt == "f64" ? 8 : dt == "f32" ? 4 : (dt == "f16" || dt == "bf16") ? 2 : 0;
            }

            bool readArg(LaunchStep const& step, std::string const& name, int64_t& value)
            {
                int const index = step.kernel->argIndex(name);
                if(index < 0)
                {
                    return false;
                }
                auto const& a = step.kernel->args[index];
                if(a.pointer || a.offset + uint64_t(a.size) > step.args.size())
                {
                    return false;
                }
                if(a.size == 4)
                {
                    int32_t v;
                    std::memcpy(&v, step.args.data() + a.offset, 4);
                    value = v;
                    return true;
                }
                if(a.size == 8)
                {
                    std::memcpy(&value, step.args.data() + a.offset, 8);
                    return true;
                }
                return false;
            }

            std::string argNameAt(PackKernel const& kernel, uint32_t offset)
            {
                for(auto const& a : kernel.args)
                {
                    if(a.offset == offset)
                    {
                        return a.name;
                    }
                }
                return "";
            }

            // The dtype names of rocKE's builders (static storage): nullptr for "" (an absent B or
            // C); an unknown name sets `bad`.
            char const* dtypeName(std::string const& dt, bool& bad)
            {
                for(char const* name : {"f16", "bf16", "f32", "f64"})
                {
                    if(dt == name)
                    {
                        return name;
                    }
                }
                bad = bad || !dt.empty();
                return nullptr;
            }

            // The row of a pack kernel of the generic elementwise kernel or of one of its vector
            // classes (its attributes), checked by rocKE's builder.
            bool rowOf(PackKernel const& k, rocke_htaot_ew_spec_t& s)
            {
                s             = rocke_htaot_ew_spec_default();
                bool bad      = false;
                s.arity       = static_cast<int>(k.attr("arity"));
                s.a           = dtypeName(k.attrStr("a"), bad);
                s.b           = dtypeName(k.attrStr("b"), bad);
                s.c           = dtypeName(k.attrStr("c"), bad);
                s.d           = dtypeName(k.attrStr("d"), bad);
                s.cplx        = k.attr("cplx") == 1;
                s.idx64       = k.attr("idx64") == 1;
                s.math        = k.attr("math") == 1;
                s.block       = static_cast<int>(k.attr("block"));
                char why[160] = {0};
                return !bad && rocke_htaot_ew_check(&s, why, sizeof(why));
            }

            // A name written by one of rocKE's name functions.
            template <typename Spec>
            std::string builderName(rocke_status_t (*fn)(Spec const*, char*, size_t),
                                    Spec const& spec)
            {
                char name[256] = {0};
                return fn(&spec, name, sizeof(name)) == ROCKE_OK ? std::string(name)
                                                                 : std::string();
            }

            // The JIT class of a generic elementwise launch: rocKE's kernel class (the pack
            // kernel's row, the vector width, the inputs read with vector loads) and its key.
            struct EwClass
            {
                rocke_htaot_ew_jit_spec_t spec{};
                std::string               key;
                int                       vec = 1;
                std::vector<std::string>  tensors; // inputs in kernel order, then D
                std::vector<bool>         vectorTensor;
                std::vector<size_t>       bytes;
            };

            // Decides the class of `step` (a launch of the pack's generic elementwise kernel, or of
            // one of its vector classes, family "ewv", whose arguments are the class geometry).
            // The pack kernel must be the one rocKE's builder names for its row (a pack of
            // another rocKE version keeps its kernels).
            bool classify(LaunchStep const& step, EwClass& out, std::string& reason)
            {
                auto const&           k = *step.kernel;
                rocke_htaot_ew_spec_t base{};
                if(!rowOf(k, base))
                {
                    reason = "not a generic elementwise kernel";
                    return false;
                }
                int const  arity = base.arity;
                bool const cplx  = base.cplx;
                out.spec.base    = base;
                if(k.family == "ewv")
                {
                    auto const unit = k.attrStr("unit");
                    out.spec.vec    = static_cast<int>(k.attr("vec"));
                    out.spec.n_unit = arity;
                    for(int t = 0; t < arity && t < int(unit.size()); t++)
                    {
                        out.spec.unit[t] = unit[t] == 'u';
                    }
                    if(k.attrStr("abi") != kEwvAbi || int(unit.size()) != arity
                       || builderName(rocke_htaot_ewv_kernel_name, out.spec) != k.name)
                    {
                        reason = "not a vector class of the elementwise kernel";
                        return false;
                    }
                    out.tensors = arity == 1   ? std::vector<std::string>{"A", "D"}
                                  : arity == 2 ? std::vector<std::string>{"A", "C", "D"}
                                               : std::vector<std::string>{"A", "B", "C", "D"};
                    for(auto const& t : out.tensors)
                    {
                        std::string lower(1, char(std::tolower(t[0])));
                        out.bytes.push_back(elementBytes(k.attrStr(lower)));
                    }
                    out.vec = out.spec.vec;
                    out.vectorTensor.assign(out.tensors.size(), true);
                    for(size_t t = 0; t + 1 < out.tensors.size(); t++)
                    {
                        out.vectorTensor[t] = out.spec.unit[t];
                    }
                    out.key = "ew:" + builderName(rocke_htaot_ew_jit_kernel_name, out.spec);
                    return true;
                }
                if(k.attrStr("abi") != kEwAbi
                   || builderName(rocke_htaot_ew_kernel_name, base) != k.name)
                {
                    reason = "not a generic elementwise kernel";
                    return false;
                }
                out.tensors = arity == 1   ? std::vector<std::string>{"A", "D"}
                              : arity == 2 ? std::vector<std::string>{"A", "C", "D"}
                                           : std::vector<std::string>{"A", "B", "C", "D"};
                for(auto const& t : out.tensors)
                {
                    std::string lower(1, char(std::tolower(t[0])));
                    out.bytes.push_back(elementBytes(k.attrStr(lower)) * (cplx ? 2 : 1));
                }
                out.vec = 1;
                out.vectorTensor.assign(out.tensors.size(), false);
                if(!cplx)
                {
                    int const vec = static_cast<int>(16 / out.bytes.back());
                    int64_t   e0 = 0, lim0 = 0, sD0 = 0;
                    bool      ok = readArg(step, "e0", e0) && readArg(step, "sD0", sD0) && sD0 == 1
                              && e0 % vec == 0 && e0 >= vec;
                    if(ok && arity == 1)
                    {
                        ok = readArg(step, "lim0", lim0) && lim0 % vec == 0;
                    }
                    std::vector<bool> vector(out.tensors.size(), false);
                    vector.back() = true;
                    for(size_t t = 0; ok && t < out.tensors.size(); t++)
                    {
                        auto const& n  = out.tensors[t];
                        int64_t     s0 = 0;
                        ok             = readArg(step, "s" + n + "0", s0);
                        if(t + 1 < out.tensors.size())
                        {
                            vector[t] = s0 == 1;
                        }
                        for(int i = 1; ok && vector[t] && i < kMaxRank; i++)
                        {
                            int64_t s = 0;
                            ok = readArg(step, "s" + n + std::to_string(i), s) && s % vec == 0;
                        }
                    }
                    // pointer offsets the plan knows (workspace slices) must keep the vectors
                    // aligned; the buffers themselves are checked at execution
                    for(auto const& p : step.pointers)
                    {
                        auto name = argNameAt(k, p.offset);
                        for(size_t t = 0; ok && t < out.tensors.size(); t++)
                        {
                            if(name == out.tensors[t] && vector[t])
                            {
                                ok = p.bytes % (out.bytes[t] * vec) == 0;
                            }
                        }
                    }
                    // 64-bit offset rows and the rows with the math operators have vector
                    // classes only with every input a vector (their other layouts take the
                    // vec 1 class)
                    if(ok && (base.idx64 || base.math))
                    {
                        ok = std::all_of(vector.begin(), vector.end(), [](bool v) { return v; });
                    }
                    if(ok)
                    {
                        out.vec          = vec;
                        out.vectorTensor = vector;
                    }
                }
                out.spec.vec    = out.vec;
                out.spec.n_unit = out.vec > 1 ? arity : 0;
                for(int t = 0; out.vec > 1 && t < arity; t++)
                {
                    out.spec.unit[t] = out.vectorTensor[t];
                }
                out.key = "ew:" + builderName(rocke_htaot_ew_jit_kernel_name, out.spec);
                return true;
            }

            // Builds the kernel of class `cls` for the geometry `values` (in the order of the
            // class's axes) with rocKE's C++ builder, lowers it for the linked compiler's flavor
            // and compiles it (no lock held). nullptr with `reason`; nullptr without a reason when
            // the kernel cache holds the kernel's identity already (the caller looks it up).
            std::unique_ptr<Entry> build(EwClass const&              cls,
                                         std::vector<int64_t> const& values,
                                         std::string const&          arch,
                                         std::string const&          specKey,
                                         std::string&                reason,
                                         Digest&                     id,
                                         double&                     frontMs,
                                         double&                     compileMs)
            {
                using Clock            = std::chrono::steady_clock;
                auto const&         ci = compilerInfo();
                auto const          t0 = Clock::now();
                rocke_ir_builder_t  builder{};
                rocke_kernel_def_t* kernel = rocke_build_htaot_ew_jit(
                    &builder, &cls.spec, values.data(), static_cast<int>(values.size()));
                if(kernel == nullptr)
                {
                    char const* error = rocke_ir_builder_error(&builder);
                    reason = std::string("rocKE builder: ") + (error != nullptr ? error : "");
                    rocke_ir_builder_free(&builder);
                    return nullptr;
                }
                char           err[1024] = {0};
                char*          text      = nullptr;
                rocke_status_t st
                    = rocke_lower_kernel_to_llvm_ex(kernel,
                                                    rocke_llvm_flavor_from_name(ci.flavor.c_str()),
                                                    arch.c_str(),
                                                    &text,
                                                    err,
                                                    sizeof(err));
                std::string name = kernel->name != nullptr ? kernel->name : "";
                rocke_ir_builder_free(&builder);
                if(st != ROCKE_OK || text == nullptr)
                {
                    reason = std::string("lowering: ") + err;
                    std::free(text);
                    return nullptr;
                }
                std::string ll(text);
                std::free(text);
                auto const  llSha = sha256(reinterpret_cast<uint8_t const*>(ll.data()), ll.size());
                auto const  isa   = isaName(arch);
                auto const& opts  = jitCompileOptions();
                id                = entryIdOf(isa, llSha, opts, false, ci.id);
                auto const t1     = Clock::now();
                frontMs           = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if(auto const* hit = KernelCache::instance().byEntryId(id, specKey))
                {
                    static_cast<void>(hit);
                    return nullptr; // the caller looks the entry up again
                }
                if(envFlag("HIPTENSOR_ROCKE_JIT_NO_COMPILE"))
                {
                    reason = "compiling is disabled (HIPTENSOR_ROCKE_JIT_NO_COMPILE)";
                    return nullptr;
                }
                auto        entry = std::make_unique<Entry>();
                std::string log;
                if(!compileLlvmIr(ll, isa, opts, entry->code, log))
                {
                    reason = log;
                    return nullptr;
                }
                compileMs = std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
                if(!readKernelMetadata(
                       entry->code.data(), entry->code.size(), name, entry->kernel, reason))
                {
                    return nullptr;
                }
                entry->entryId        = id;
                entry->isa            = isa;
                entry->compilerId     = ci.id;
                entry->options        = opts;
                entry->llvmSha        = llSha;
                entry->family         = kEwJitFamily;
                entry->classKey       = cls.key;
                entry->codeSha        = sha256(entry->code.data(), entry->code.size());
                entry->kernel.family  = kEwJitFamily;
                entry->kernel.specKey = cls.key;
                return entry;
            }

            // HIPTENSOR_KERNEL_CACHE_DIR: a directory of kernel cache files, one per spec key, that
            // plans read before they compile and write after (shared by processes; files are
            // written to a temporary name and renamed).
            std::string persistPath(std::string const& specKey)
            {
                auto dir = getEnvironmentVariable("HIPTENSOR_KERNEL_CACHE_DIR");
                if(!dir || dir->empty())
                {
                    return "";
                }
                auto digest
                    = sha256(reinterpret_cast<uint8_t const*>(specKey.data()), specKey.size());
                return *dir + "/" + hex(digest.data(), 16) + ".htk";
            }

            bool fileExists(std::string const& path)
            {
                std::ifstream f(path, std::ios::binary);
                return f.good();
            }

            // The specialized launch of `step`, or false (the launch keeps its pack kernel).
            bool specializeStep(LaunchStep&        step,
                                std::string const& arch,
                                DeviceClass const& dc,
                                bool               promoteAll,
                                bool               allowCompile,
                                JitReport&         report,
                                std::string&       reason)
            {
                EwClass cls;
                if(!classify(step, cls, reason))
                {
                    return false;
                }
                // Promotion: a vector form with at least one input read with vector loads. A
                // vector form whose inputs are all gathered or broadcast along the fastest mode
                // of D (a non-unit stride there) is not promoted by default;
                // HIPTENSOR_ROCKE_JIT=all promotes every launch.
                bool const vectorInput
                    = std::find(cls.vectorTensor.begin(), cls.vectorTensor.end() - 1, true)
                      != cls.vectorTensor.end() - 1;
                if(!promoteAll && (cls.vec == 1 || !vectorInput))
                {
                    reason = cls.vec == 1
                                 ? "no vector form for this layout"
                                 : "no input is read with vector loads (gathered or broadcast)";
                    return false;
                }
                // the JIT's kernel classes (rocke_ew_rows.hpp): every row of the pack's generic
                // kernel; this fails only for a pack of another rocKE version
                if(!jitClassAdmitted(cls.spec))
                {
                    reason = "no JIT class " + cls.key;
                    return false;
                }
                // the values of the class's axes: the launch's arguments, with the thread space
                // of a vector class counted in vectors (a vector class launch counts in vectors
                // already)
                bool const classLaunch = step.kernel->family == "ewv";
                int64_t    nIn = 0, e0 = 0;
                if(!readArg(step, "n_in", nIn) || !readArg(step, "e0", e0))
                {
                    reason = "the launch has no thread space";
                    return false;
                }
                char      axes[ROCKE_HTAOT_EW_JIT_MAX_AXES][16];
                int const nAxes = rocke_htaot_ew_jit_axes(
                    &cls.spec, axes, nullptr, ROCKE_HTAOT_EW_JIT_MAX_AXES);
                if(nAxes <= 0 || nAxes > ROCKE_HTAOT_EW_JIT_MAX_AXES)
                {
                    reason = "no geometry of class " + cls.key;
                    return false;
                }
                // the spec key: the class, the builder (rocKE's engine build) and the values
                std::vector<int64_t> values;
                std::string specKey = std::string(kSpecKeyPrefix) + "|" + arch + "|" + cls.key + "|"
                                      + rocke_build_id() + "|";
                auto const vecMagic = magicU31(static_cast<uint32_t>(e0 / cls.vec));
                for(int a = 0; a < nAxes; a++)
                {
                    std::string const axis = axes[a];
                    int64_t           v    = 0;
                    if(classLaunch)
                    {
                        if(!readArg(step, axis, v))
                        {
                            reason = "the launch has no value for axis " + axis;
                            return false;
                        }
                    }
                    else if(cls.vec > 1 && axis == "n_in")
                    {
                        v = nIn / cls.vec;
                    }
                    else if(cls.vec > 1 && axis == "e0")
                    {
                        v = e0 / cls.vec;
                    }
                    else if(cls.vec > 1 && axis == "m0")
                    {
                        v = static_cast<int32_t>(vecMagic.first);
                    }
                    else if(cls.vec > 1 && axis == "s0")
                    {
                        v = vecMagic.second;
                    }
                    else if(!readArg(step, axis, v))
                    {
                        reason = "the launch has no value for axis " + axis;
                        return false;
                    }
                    values.push_back(v);
                    specKey += std::to_string(v) + ",";
                }

                auto&        cache = KernelCache::instance();
                Entry const* entry = cache.bySpecKey(specKey);
                bool         fresh = false;
                if(entry == nullptr)
                {
                    if(cache.refused(specKey))
                    {
                        reason = "refused before in this process";
                        return false;
                    }
                    if(!allowCompile)
                    {
                        reason = "the plan's JIT time budget is spent";
                        return false;
                    }
                    std::shared_future<Entry const*> wait;
                    if(cache.begin(specKey, wait))
                    {
                        std::string why;
                        Digest      id{};
                        double      frontMs = 0.0, compileMs = 0.0;
                        auto const  persisted = persistPath(specKey);
                        if(!persisted.empty() && fileExists(persisted))
                        {
                            cache.read(persisted, dc, "hiptensorCreatePlan");
                            entry = cache.bySpecKey(specKey);
                        }
                        std::unique_ptr<Entry> built;
                        if(entry == nullptr)
                        {
                            built = build(cls, values, arch, specKey, why, id, frontMs, compileMs);
                        }
                        if(built != nullptr)
                        {
                            entry = cache.insert(std::move(built), specKey);
                            fresh = true;
                            if(!persisted.empty())
                            {
                                cache.write(persisted, dc, entry, "hiptensorCreatePlan");
                            }
                            report.compiled++;
                            char msg[160];
                            snprintf(msg,
                                     sizeof(msg),
                                     " in %.2f ms (build and lowering %.2f ms, COMGR %.2f ms)",
                                     frontMs + compileMs,
                                     frontMs,
                                     compileMs);
                            logJit("hiptensorCreatePlan", "rocKE JIT: compiled " + cls.key + msg);
                        }
                        else if(entry == nullptr && why.empty())
                        {
                            entry = cache.bySpecKey(specKey); // found by identity
                        }
                        bool remember = true;
                        if(entry == nullptr)
                        {
                            reason = why;
                            logJit("hiptensorCreatePlan",
                                   "rocKE JIT: " + cls.key + " not specialized: " + why);
                            // a disabled compile is a test setting, not a property of the kernel
                            remember = !envFlag("HIPTENSOR_ROCKE_JIT_NO_COMPILE");
                        }
                        cache.finish(specKey, entry, remember);
                    }
                    else
                    {
                        entry = wait.get();
                    }
                    if(entry == nullptr)
                    {
                        return false;
                    }
                }
                if(!fresh)
                {
                    report.memoryHits++;
                    report.fileHits += entry->fromFile ? 1 : 0;
                }

                LaunchStep js;
                js.kernel = &entry->kernel;
                if(entry->function(js.function) != HIPTENSOR_STATUS_SUCCESS)
                {
                    reason = "the specialized code object does not load";
                    return false;
                }
                ArgWriter w(js);
                bool      ok = true;
                for(auto const& p : step.pointers)
                {
                    auto name = argNameAt(*step.kernel, p.offset);
                    ok        = ok && w.pointer(name, p.role, p.bytes);
                    for(size_t t = 0; ok && t < cls.tensors.size(); t++)
                    {
                        if(name == cls.tensors[t] && cls.vectorTensor[t] && cls.vec > 1)
                        {
                            js.pointers.back().align
                                = static_cast<uint32_t>(cls.bytes[t] * cls.vec);
                        }
                    }
                }
                for(auto const& s : step.scalars)
                {
                    ok = ok && w.scalar(argNameAt(*step.kernel, s.offset), s.role, s.imag);
                }
                std::string missing;
                if(!ok || !w.complete(missing))
                {
                    reason = "the specialized kernel's arguments do not match (" + missing + ")";
                    return false;
                }
                uint32_t const block = step.block.x;
                int64_t const  vecIn = classLaunch ? nIn : nIn / cls.vec;
                js.block             = step.block;
                js.grid = dim3(static_cast<uint32_t>((vecIn + block - 1) / block), step.grid.y, 1);
                if(entry->kernel.maxWorkgroup != 0 && block > entry->kernel.maxWorkgroup)
                {
                    reason = "the specialized kernel takes smaller workgroups";
                    return false;
                }
                js.fallback    = std::make_shared<std::vector<LaunchStep> const>(1, step);
                js.deviceBound = true;
                step           = std::move(js);
                return true;
            }
        } // namespace

        void
            specializeProgram(Program& program, backend::DeviceKey const& device, JitReport& report)
        {
            using Clock     = std::chrono::steady_clock;
            auto const t0   = Clock::now();
            auto       mode = getEnvironmentVariable("HIPTENSOR_ROCKE_JIT");
            if(mode && (*mode == "0" || *mode == "off"))
            {
                report.note = "JIT disabled by HIPTENSOR_ROCKE_JIT";
                return;
            }
            bool const promoteAll = mode && *mode == "all";
            for(auto const& step : program.steps)
            {
                report.launches
                    += step.kernel != nullptr
                               && (step.kernel->family == "ew" || step.kernel->family == "ewv")
                           ? 1
                           : 0;
            }
            if(report.launches == 0)
            {
                return;
            }
            auto const& ci = compilerInfo();
            if(!ci.available)
            {
                report.note = "no compiler: " + ci.reason;
                return;
            }
            std::string reason;
            // HIPTENSOR_JIT_MAX_MS: compile budget of one plan; past it, the remaining launches
            // keep their pack kernels (cache hits are still taken)
            double budget = 0.0;
            if(auto v = getEnvironmentVariable("HIPTENSOR_JIT_MAX_MS"); v && !v->empty())
            {
                budget = std::atof(v->c_str());
            }
            auto const  dc = deviceClassOf(device);
            std::string firstReason;
            for(auto& step : program.steps)
            {
                if(step.kernel == nullptr
                   || (step.kernel->family != "ew" && step.kernel->family != "ewv"))
                {
                    continue;
                }
                double const spent
                    = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                bool const allow = budget <= 0.0 || spent < budget;
                if(specializeStep(step, device.baseArch, dc, promoteAll, allow, report, reason))
                {
                    report.specialized++;
                }
                else
                {
                    report.declined++;
                    if(firstReason.empty())
                    {
                        firstReason = reason;
                    }
                }
            }
            report.milliseconds
                = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            if(report.declined > 0)
            {
                report.note = firstReason;
            }
        }

        hiptensorStatus_t writeKernelCacheFile(char const*               fileName,
                                               backend::DeviceKey const& device)
        {
            return KernelCache::instance().write(fileName, deviceClassOf(device));
        }

        hiptensorStatus_t readKernelCacheFile(char const*               fileName,
                                              backend::DeviceKey const& device)
        {
            return KernelCache::instance().read(fileName, deviceClassOf(device));
        }
    } // namespace rockebackend
} // namespace hiptensor
