/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <type_traits>

#include <Tensile/Debug.hpp>
#include <Tensile/EmbeddedData.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <regex>

namespace TensileLite
{
    namespace hip
    {
        SolutionAdapter::SolutionAdapter()
            : m_debug(Debug::Instance().printKernelArguments())
            , m_debugSkipLaunch(Debug::Instance().skipKernelLaunch())
        {
        }

        SolutionAdapter::SolutionAdapter(ModuleApi moduleApi)
            : SolutionAdapter()
        {
            m_moduleApi = moduleApi;
        }

        SolutionAdapter::SolutionAdapter(bool debug)
            : m_debug(debug)
        {
            m_debug = debug || Debug::Instance().printKernelArguments();
        }

        SolutionAdapter::SolutionAdapter(bool debug, std::string const& name)
            : m_debug(debug)
            , m_name(name)
        {
            m_debug = debug || Debug::Instance().printKernelArguments();
        }

        SolutionAdapter::~SolutionAdapter()
        {
            Debug::Instance().markerStart("UnloadCodeObjectFiles");
            for(auto module : m_modules)
                HIP_CHECK_PRINT(m_moduleApi.unload(module), [&](hipError_t error) {
                    std::cerr << "hipModuleUnload failed: " << std::endl
                              << " error: " << hipGetErrorString(error) << std::endl;
                });
            // Extra rotation copies are independent hipModule_t handles loaded
            // by loadCodeObjectFileExtraCopies(); they own their own device
            // memory and must be unloaded too or we leak it per copy.
            for(auto const& copyModules : m_extraModuleCopies)
                for(auto module : copyModules)
                    HIP_CHECK_PRINT(m_moduleApi.unload(module), [&](hipError_t error) {
                        std::cerr << "hipModuleUnload failed: " << std::endl
                                  << " error: " << hipGetErrorString(error) << std::endl;
                    });
#ifdef TENSILELITE_ENABLE_LOCAL_KERNELS
            for(auto const& entry : m_localModules)
                HIP_CHECK_PRINT(m_moduleApi.unload(entry.second), [&](hipError_t error) {
                    std::cerr << "hipModuleUnload failed: " << entry.first << std::endl
                              << " error: " << hipGetErrorString(error) << std::endl;
                });
#endif
            Debug::Instance().markerStop();
        }

        std::string removeXnack(std::string coFilename)
        {
            std::string xnackVersion = "xnack"; //Extra character before and after xnack
            size_t      loc          = coFilename.find(xnackVersion);
            if(loc != std::string::npos)
                coFilename.replace(loc - 1, xnackVersion.length() + 2, "");

            return coFilename;
        }

        static void logCodeObjectLoadError(std::string const& path, hipError_t error)
        {
            std::cerr << "hipModuleLoad failed: " << path << std::endl
                      << " error: " << hipGetErrorString(error) << std::endl;
        }

        hipError_t SolutionAdapter::loadCodeObjectFile(std::string const& path)
        {
            hipError_t error = loadCodeObjectFileOnce(path);
            if(error == hipSuccess)
                return error;

            if(error != hipErrorLaunchFailure && error != hipErrorNoBinaryForGpu)
            {
                logCodeObjectLoadError(path, error);
                return error;
            }

            std::string lazyArch;
            std::string lazyDir;
            {
                std::lock_guard<std::mutex> guard(m_access);
                lazyArch = m_lazyLoadArchitecture;
                lazyDir  = m_codeObjectDirectory;
            }

            // Without an architecture, helper recovery would probe an invalid
            // name such as Kernels.so-000-.hsaco and hide the primary error.
            if(lazyArch.empty())
            {
                logCodeObjectLoadError(path, error);
                return error;
            }

            // Reset the error code from the failed hipModuleLoad.
            (void)m_moduleApi.getLastError();
            std::cout << "Clearing modules and retrying hipModuleLoad" << std::endl;
            for(auto module : m_modules)
            {
                HIP_CHECK_PRINT(m_moduleApi.unload(module), [&](hipError_t error_t) {
                    std::cerr << "hipModuleUnload failed: " << std::endl
                              << " error: " << hipGetErrorString(error_t) << std::endl;
                });
            }

            // Extra rotation copies are not recreated by recovery.
            if(!m_extraModuleCopies.empty())
            {
                std::cerr << "[icache-rotate] WARNING: out-of-memory retry is dropping "
                          << m_extraModuleCopies.size()
                          << " rotation copy set(s); I-cache rotation is now disabled "
                          << "until loadCodeObjectFileExtraCopies() is called again." << std::endl;
            }
            for(auto const& copyModules : m_extraModuleCopies)
            {
                for(auto module : copyModules)
                {
                    HIP_CHECK_PRINT(m_moduleApi.unload(module), [&](hipError_t error_t) {
                        std::cerr << "hipModuleUnload failed: " << std::endl
                                  << " error: " << hipGetErrorString(error_t) << std::endl;
                    });
                }
            }

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.clear();
                m_loadedModuleNames.clear();
                m_loadedCOFiles.clear();
                m_kernels.clear();
                m_extraModuleCopies.clear();
                m_extraKernels.clear();
                m_currentRotationCopy.store(0);
            }

            hipError_t lazyLoadingError = initializeLazyLoading(lazyArch, lazyDir);
            if(lazyLoadingError != hipSuccess)
            {
                std::cerr << "initializeLazyLoading after module clear failed; preserving "
                             "the original hipModuleLoad error: "
                          << hipGetErrorString(error) << std::endl;
                return error;
            }

            error = loadCodeObjectFileOnce(path);
            if(error != hipSuccess)
                logCodeObjectLoadError(path, error);
            return error;
        }

        hipError_t SolutionAdapter::loadCodeObjectFileOnce(std::string const& path)
        {
            Debug::Instance().markerStart("loadCodeObjectFile", path);
            hipModule_t module;

            hipError_t error = m_moduleApi.load(&module, path.c_str());
            if(error != hipSuccess)
            {
                Debug::Instance().markerStop();
                return error;
            }

            if(m_debug)
                std::cout << "loaded code object " << path << std::endl;

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.push_back(module);
                m_loadedModuleNames.push_back(concatenate("File ", path));

                //Isolate filename
                size_t start = path.rfind('/');
                start        = (start == std::string::npos) ? 0 : start + 1;
                m_loadedCOFiles.insert(removeXnack(std::string(path.begin() + start, path.end())));
            }
            Debug::Instance().markerStop();
            return hipSuccess;
        }

        hipError_t SolutionAdapter::loadCodeObjectBytes(std::vector<uint8_t> const& bytes)
        {
            return loadCodeObject(bytes.data());
        }

        hipError_t SolutionAdapter::loadCodeObject(const void* image)
        {
            hipModule_t module;

            HIP_CHECK_RETURN_WITH_LOG(hipModuleLoadData(&module, image),
                [&](hipError_t error) {
                    std::cerr << "hipModuleLoadData failed: " << std::endl
                            << " error: " << hipGetErrorString(error) << std::endl;
                }
            );

            if(m_debug)
                std::cout << "loaded code object data." << std::endl;

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.push_back(module);
                m_loadedModuleNames.push_back("Module from bytes");
            }
            return hipSuccess;
        }

        void SolutionAdapter::loadEmbeddedCodeObjects()
        {
            loadEmbeddedCodeObjects("");
        }

        void SolutionAdapter::loadEmbeddedCodeObjects(std::string const& key)
        {
            auto const& embeddedData = EmbeddedData<TensileLite::SolutionAdapter>::Get(key);

            if(embeddedData.size() == 0)
            {
                if(m_debug || Debug::Instance().printCodeObjectInfo())
                {
                    std::cerr << "Found no embedded code objects";
                    if(key != "")
                        std::cerr << " with the key " << key;

                    std::cerr << "." << std::endl;
                }
                return;
            }

            std::vector<hipModule_t> newModules;
            newModules.reserve(embeddedData.size());

            for(size_t i = 0; i < embeddedData.size(); i++)
            {
                hipModule_t nextModule;
                try
                {
                    auto error = hipModuleLoadData(&nextModule, embeddedData[i].data());

                    if(error == hipErrorUnknown || error == hipErrorSharedObjectInitFailed)
                        continue;
                    newModules.push_back(nextModule);
                    HIP_CHECK_EXC(error);

                    if(m_debug)
                        std::cout << "Loaded code object for key " << key << std::endl;
                }
                catch(std::runtime_error const& exc)
                {
                    std::cout << exc.what() << std::endl;
                }
            }

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.insert(m_modules.end(), newModules.begin(), newModules.end());
                m_loadedModuleNames.push_back(
                    concatenate("Embedded code object ", key, " (", newModules.size(), ")"));
            }
        }

#ifdef TENSILELITE_ENABLE_LOCAL_KERNELS
        namespace
        {
            uint64_t readLE(unsigned char const* p, int bytes)
            {
                uint64_t v = 0;
                for(int i = bytes - 1; i >= 0; --i)
                    v = (v << 8) | p[i];
                return v;
            }

            // hipModuleLoad crashes inside the runtime (ROCm 7.2) on a code
            // object whose contents run past the end of the file, as a copy
            // cut short leaves it, so those are rejected before it sees them.
            // This can go once hipModuleLoad returns an error for such files.
            // Only the headers of the ELF, offload bundle and compressed
            // offload bundle layouts are read; any other file is left for
            // hipModuleLoad to judge. It catches a damaged file, not one
            // replaced between this check and the load.
            bool codeObjectFitsInFile(std::string const& path)
            {
                std::ifstream file(path, std::ios::binary);
                if(!file)
                    return true;
                file.seekg(0, std::ios::end);
                uint64_t const size = static_cast<uint64_t>(file.tellg());
                file.seekg(0);

                auto read = [&](uint64_t offset, uint64_t length) {
                    std::vector<unsigned char> bytes(length);
                    file.seekg(static_cast<std::streamoff>(offset));
                    file.read(reinterpret_cast<char*>(bytes.data()),
                              static_cast<std::streamsize>(length));
                    return bytes;
                };
                auto fits = [size](uint64_t offset, uint64_t count, uint64_t entrySize) {
                    return offset <= size
                           && (entrySize == 0 || count <= (size - offset) / entrySize);
                };
                auto startsWith
                    = [](std::vector<unsigned char> const& bytes, std::string const& magic) {
                          return bytes.size() >= magic.size()
                                 && std::equal(magic.begin(), magic.end(), bytes.begin());
                      };

                constexpr uint64_t headerSize = 64;
                if(size < 4)
                    return true;
                auto header = read(0, std::min(size, headerSize));

                // Compressed offload bundle: the header records the bundle's
                // size from version 2 on.
                if(startsWith(header, "CCOB"))
                {
                    if(size < 8)
                        return false;
                    uint64_t const version = readLE(&header[4], 2);
                    if(version == 2)
                        return size >= 12 && readLE(&header[8], 4) <= size;
                    if(version >= 3)
                        return size >= 16 && readLE(&header[8], 8) <= size;
                    return true;
                }

                if(startsWith(header, "__CLANG_OFFLOAD_BUNDLE__"))
                {
                    if(size < 32)
                        return false;
                    uint64_t const entries = readLE(&header[24], 8);
                    uint64_t       offset  = 32;
                    for(uint64_t i = 0; i < entries; ++i)
                    {
                        if(!fits(offset, 3, 8))
                            return false;
                        auto     entry   = read(offset, 24);
                        uint64_t idBytes = readLE(&entry[16], 8);
                        if(!fits(readLE(&entry[0], 8), readLE(&entry[8], 8), 1)
                           || !fits(offset + 24, idBytes, 1))
                            return false;
                        offset += 24 + idBytes;
                    }
                    return static_cast<bool>(file);
                }

                if(header[0] != 0x7f || header[1] != 'E' || header[2] != 'L' || header[3] != 'F')
                    return true;
                // ELFCLASS64, little-endian: what every AMDGPU code object is.
                if(size < headerSize || header[4] != 2 || header[5] != 1)
                    return false;

                uint64_t const phoff     = readLE(&header[0x20], 8);
                uint64_t const shoff     = readLE(&header[0x28], 8);
                uint64_t const phentsize = readLE(&header[0x36], 2);
                uint64_t const phnum     = readLE(&header[0x38], 2);
                uint64_t const shentsize = readLE(&header[0x3A], 2);
                uint64_t const shnum     = readLE(&header[0x3C], 2);
                if((phnum && phentsize < 56) || (shnum && shentsize < 64)
                   || !fits(phoff, phnum, phentsize) || !fits(shoff, shnum, shentsize))
                    return false;

                for(uint64_t i = 0; i < phnum; ++i)
                {
                    auto ph = read(phoff + i * phentsize, 56);
                    if(!fits(readLE(&ph[8], 8), readLE(&ph[32], 8), 1))
                        return false;
                }
                constexpr uint64_t SHT_NOBITS = 8;
                for(uint64_t i = 0; i < shnum; ++i)
                {
                    auto sh = read(shoff + i * shentsize, 64);
                    if(readLE(&sh[4], 4) != SHT_NOBITS
                       && !fits(readLE(&sh[24], 8), readLE(&sh[32], 8), 1))
                        return false;
                }
                return static_cast<bool>(file);
            }
        } // namespace

        bool SolutionAdapter::isLocalCodeObject(std::string const& codeObjectFile)
        {
            // Runs on every launch, so it avoids building a path where the
            // first character decides.
#ifdef _WIN32
            return !codeObjectFile.empty() && std::filesystem::path(codeObjectFile).is_absolute();
#else
            return !codeObjectFile.empty() && codeObjectFile.front() == '/';
#endif
        }

        hipError_t SolutionAdapter::loadLocalCodeObject(std::string const& path)
        {
            // Loads of one path are serialized so it is loaded once; loads of
            // different paths do not wait for each other.
            std::shared_ptr<std::mutex> pathMutex;
            {
                std::lock_guard<std::mutex> guard(m_access);
                if(m_localModules.count(path) != 0)
                    return hipSuccess;
                auto& slot = m_localLoadMutexes[path];
                if(!slot)
                    slot = std::make_shared<std::mutex>();
                pathMutex = slot;
            }
            std::lock_guard<std::mutex> loadGuard(*pathMutex);
            {
                std::lock_guard<std::mutex> guard(m_access);
                if(m_localModules.count(path) != 0)
                    return hipSuccess;
            }

            if(!codeObjectFitsInFile(path))
            {
                std::cerr << "hipModuleLoad skipped: " << path << std::endl
                          << " error: code object is truncated or malformed" << std::endl;
                return hipErrorInvalidImage;
            }

            hipModule_t module;
            hipError_t  error;
            try
            {
                error = m_moduleApi.load(&module, path.c_str());
            }
            catch(std::exception const& e)
            {
                std::cerr << "hipModuleLoad threw: " << e.what() << std::endl;
                error = hipErrorInvalidImage;
            }
            if(error != hipSuccess)
            {
                (void)m_moduleApi.getLastError();
                logCodeObjectLoadError(path, error);
                return error;
            }

            // Keep ownership until insertion succeeds, so an allocation failure
            // cannot leave a loaded module outside the adapter's maps.
            std::unique_ptr<std::remove_pointer_t<hipModule_t>, decltype(m_moduleApi.unload)>
                ownedModule(module, m_moduleApi.unload);

            if(m_debug)
                std::cout << "loaded local code object " << path << std::endl;

            std::lock_guard<std::mutex> guard(m_access);
            m_localModules.emplace(path, module);
            ownedModule.release();
            m_loadedModuleNames.push_back(concatenate("Local file ", path));
            return hipSuccess;
        }

        hipError_t SolutionAdapter::getLocalKernel(hipFunction_t&     rv,
                                                   std::string const& path,
                                                   std::string const& name)
        {
            auto key = std::make_pair(path, name);
            {
                std::lock_guard<std::mutex> guard(m_access);
                auto                        it = m_localKernels.find(key);
                if(it != m_localKernels.end())
                {
                    rv = it->second;
                    return hipSuccess;
                }
            }

            hipError_t err = loadLocalCodeObject(path);
            if(err != hipSuccess)
                return err == hipErrorNotFound ? hipErrorFileNotFound : err;

            std::lock_guard<std::mutex> guard(m_access);
            err = hipModuleGetFunction(&rv, m_localModules.at(path), name.c_str());
            if(err == hipSuccess)
                m_localKernels.emplace(std::move(key), rv);
            else
                (void)m_moduleApi.getLastError();
            return err;
        }

        size_t SolutionAdapter::localCodeObjectCount()
        {
            std::lock_guard<std::mutex> guard(m_access);
            return m_localModules.size();
        }
#endif

        bool SolutionAdapter::FindCodeObject(std::string const& codeObjectFile)
        {
#ifdef TENSILELITE_ENABLE_LOCAL_KERNELS
            if(isLocalCodeObject(codeObjectFile))
            {
                {
                    std::lock_guard<std::mutex> guard(m_access);
                    if(m_localModules.count(codeObjectFile) != 0)
                        return true;
                }
                static_cast<void>(loadLocalCodeObject(codeObjectFile));
                return false;
            }
#endif
            //If required code object file hasn't yet been loaded, load it now
            m_access.lock();
            bool loaded = m_loadedCOFiles.find(removeXnack(codeObjectFile))
                          != m_loadedCOFiles.end();
            std::string codeObjectDir = m_codeObjectDirectory;
            m_access.unlock();

            if(!loaded)
            {
                //Try other xnack versions
                size_t     loc = codeObjectFile.rfind('.');
                hipError_t err;

                for(auto ver : {"", "-xnack-", "-xnack+"})
                {
                    std::string modifiedCOName = codeObjectFile;
                    modifiedCOName.insert(loc, ver);
                    err = loadCodeObjectFile(codeObjectDir + modifiedCOName);

                    if(err == hipSuccess)
                        break;
                }
                return false;
            }
            return true;
        }

        hipError_t SolutionAdapter::initKernel(std::string const& name)
        {
            hipFunction_t function;
            return getKernel(function, name);
        }

        hipError_t SolutionAdapter::initKernels(std::vector<std::string> const& kernelNames)
        {
            hipFunction_t rv;

            for(auto name : kernelNames)
            {
                hipFunction_t function;
                auto result = getKernel(function, name);
                if(result != hipSuccess)
                    return result;
            }

            return hipSuccess;
        }

        hipError_t SolutionAdapter::getKernel(hipFunction_t& rv, std::string const& name)
        {
            int copyIdx = m_currentRotationCopy.load();

            std::unique_lock<std::mutex> guard(m_access);
            hipError_t                   err = hipErrorNotFound;

            // Defensive: clamp to a loaded slot. The size read happens under
            // m_access so it cannot race with loadCodeObjectFileExtraCopies'
            // resize/push_back. In practice the benchmark client is single-
            // threaded and all rotation copies are loaded before any launch,
            // but the lock keeps this correct regardless.
            if(copyIdx <= 0 || copyIdx > (int)m_extraModuleCopies.size())
                copyIdx = 0;

            auto&       cache   = copyIdx ? m_extraKernels[copyIdx - 1] : m_kernels;
            auto const& modules = copyIdx ? m_extraModuleCopies[copyIdx - 1] : m_modules;

            auto it = cache.find(name);
            if(it != cache.end())
            {
                rv = it->second;
                return hipSuccess;
            }

            for(auto module : modules)
            {
                err = hipModuleGetFunction(&rv, module, name.c_str());

                if(err == hipSuccess)
                {
                    cache[name] = rv;
                    return err;
                }
                else if(err != hipErrorNotFound)
                {
                    return err;
                }
                else
                {
                    (void)m_moduleApi.getLastError(); // clear hipErrorNotFound
                }
            }

            return err;
        }

        hipError_t
            SolutionAdapter::loadCodeObjectFileExtraCopies(std::string const& path,
                                                           int                extraCopies)
        {
            if(extraCopies <= 0)
                return hipSuccess;

            // Pre-grow the parallel arrays
            {
                std::lock_guard<std::mutex> guard(m_access);
                if((int)m_extraModuleCopies.size() < extraCopies)
                    m_extraModuleCopies.resize(extraCopies);
                if((int)m_extraKernels.size() < extraCopies)
                    m_extraKernels.resize(extraCopies);
            }

            // Re-load the same .co N times. Each hipModuleLoad allocates its
            // own device memory for the code object, so the N hipModule_t
            // handles map to N distinct device PCs - the prerequisite for
            // I-cache cold misses on rotation.
            for(int i = 0; i < extraCopies; ++i)
            {
                hipModule_t module;
                hipError_t  error = m_moduleApi.load(&module, path.c_str());
                if(error != hipSuccess)
                {
                    std::cerr << "loadCodeObjectFileExtraCopies hipModuleLoad failed: " << path
                              << " (rotation copy " << (i + 1) << ")"
                              << " error: " << hipGetErrorString(error) << std::endl;
                    return error;
                }
                // Print so the user can verify HIP didn't dedup: different
                // hipModule_t handles indicate independent module objects;
                // identical handles would silently break rotation.
                if(m_debug)
                    std::cout << std::endl << "[icache-rotate] loaded rotation copy " << (i + 1) << "/"
                            << extraCopies << " of " << path << " hipModule_t="
                            << static_cast<void*>(module) << std::endl << std::endl;

                // Record the module in the i-th rotation slot, plus a debug
                // name (printed by operator<<).
                std::lock_guard<std::mutex> guard(m_access);
                m_extraModuleCopies[i].push_back(module);
                m_loadedModuleNames.push_back(
                    concatenate("File ", path, " (rotation copy ", i + 1, ")"));
            }
            return hipSuccess;
        }

        void SolutionAdapter::selectRotationCopy(int idx)
        {
            // Atomic store; no lock needed. Read by getKernel.
            m_currentRotationCopy.store(idx);
        }

        int SolutionAdapter::numRotationModules()
        {
            // Total module sets available = 1 (original m_modules) + extras.
            // m_extraModuleCopies is normally guarded by m_access, but this read
            // is safe without locking because the benchmark client is single-
            // threaded and only calls this after all rotation copies have been
            // loaded.
            std::lock_guard<std::mutex> guard(m_access);
            return 1 + (int)m_extraModuleCopies.size();
        }

        // We should update the constructor to set this to avoid
        // avoid separating construction and initialization
        void SolutionAdapter::codeObjectDir(std::string codeObjDir)
        {
            if(!codeObjDir.empty() && codeObjDir.back() != '/')
                codeObjDir += '/';

            std::lock_guard<std::mutex> guard(m_access);
            m_codeObjectDirectory = std::move(codeObjDir);
        }

        void SolutionAdapter::setLazyLoadingContext(std::string arch, std::string codeObjDir)
        {
            if(!codeObjDir.empty() && codeObjDir.back() != '/')
                codeObjDir += '/';

            // Remove xnack and sramecc qualifiers.
            size_t loc = arch.find(":");
            if(loc != std::string::npos)
                arch.resize(loc);

            std::lock_guard<std::mutex> guard(m_access);
            m_lazyLoadArchitecture = std::move(arch);
            m_codeObjectDirectory  = std::move(codeObjDir);
        }

        hipError_t SolutionAdapter::initializeLazyLoading(std::string arch, std::string codeObjDir)
        {
            setLazyLoadingContext(std::move(arch), std::move(codeObjDir));

            std::string lazyArch;
            std::string lazyDir;
            bool        loaded;
            {
                std::lock_guard<std::mutex> guard(m_access);
                lazyArch                     = m_lazyLoadArchitecture;
                lazyDir                      = m_codeObjectDirectory;
                std::string helperKernelName = std::string("Kernels.so-000-") + lazyArch;
                // If required code object file hasn't yet been loaded, load it now.
                loaded = m_loadedCOFiles.find(removeXnack(helperKernelName) + ".hsaco")
                         != m_loadedCOFiles.end();
            }

            if(!loaded)
            {
                hipError_t err;
                //Try xnack variations
                for(auto ver : {"", "-xnack-", "-xnack+"})
                {
                    std::string modifiedCOName = "Kernels.so-000-" + lazyArch + ver + ".hsaco";
                    err                        = loadCodeObjectFileOnce(lazyDir + modifiedCOName);

                    if(err == hipSuccess)
                    {
                        return err;
                    }

                    logCodeObjectLoadError(lazyDir + modifiedCOName, err);
                    if(err == hipErrorFileNotFound)
                    {
                        // We expect that we could fail for cases when we have xnack variations
                        // so clear hipErrorFileNotFound between iterations.
                        (void)m_moduleApi.getLastError();
                    }
                }

                return err;
            }

            return hipSuccess;
        }

        namespace
        {
            // Both launch APIs below take 32-bit grid parameters, but the values
            // handed to them come from dim3, which is vector3<size_t>. A grid that
            // does not fit is narrowed silently rather than rejected: measured on
            // gfx950, a 256-thread kernel at 2^24 + 16 workgroups wraps to a
            // 16-workgroup launch and leaves the other 16,777,216 workgroups' worth
            // of D holding whatever was there before, with no error raised.
            //
            // Solution selection does not rule this out for Stream-K: LaunchLimits
            // exempts it, yet several Stream-K paths still size the grid at one
            // workgroup per output tile (the insufficient-workspace fallback in
            // resolveStreamKSettings(), K == 0, the data-parallel debug knob), and
            // skFixedGrid / skGridMultiplier override it outright. Refusing the
            // launch is what makes those paths loud instead of silent.
            //
            // This covers launches that go through SolutionAdapter, which is not
            // every launch in the library. rocblaslt's rocRoller custom kernels
            // call hipExtModuleLaunchKernel directly (see
            // library/src/amd_detail/rocblaslt/src/rocroller/custom_kernels.cpp)
            // and never reach this function. That path builds its grid from
            // uint32_t tile counts rather than narrowing a 64-bit dim3, so it has
            // a different failure mode than the one guarded here, but it is not
            // protected by this check.
            bool fitsLaunchDim(TensileLite::dim3 const& dim)
            {
                constexpr size_t limit = std::numeric_limits<unsigned int>::max();
                return dim.x <= limit && dim.y <= limit && dim.z <= limit;
            }
        }

        hipError_t SolutionAdapter::launchKernel(KernelInvocation const& kernel)
        {
            return launchKernel(kernel, nullptr, nullptr, nullptr);
        }

        hipError_t SolutionAdapter::launchKernel(KernelInvocation const& kernel,
                                                 hipStream_t             stream,
                                                 hipEvent_t              startEvent,
                                                 hipEvent_t              stopEvent,
                                                 bool                    isKernelLoaded)
        {
#ifdef HIP_HAS_CLUSTER_LAUNCH
            const bool enableCluster = (kernel.clusterDim.x > 1 || kernel.clusterDim.y > 1);
#else
            const bool enableCluster = false;
#endif

            // First thing in the function, ahead of loading the code object: a grid
            // that cannot be expressed is impossible whether or not the kernel
            // loads, so there is no reason to pay for the module load, and this is
            // the last point where the 64-bit values are still intact. It also
            // precedes the m_debugSkipLaunch early return, so that debug knob
            // reports the invalid grid rather than hiding it behind success.
            // The two launch APIs below narrow different quantities, so each is
            // bounded against the one it actually passes.
            if(enableCluster)
            {
                if(kernel.clusterDim.x == 0 || kernel.clusterDim.y == 0)
                {
                    std::cerr << "hipDrvLaunchKernelEx: clusterDim.x and clusterDim.y must be non-zero "
                              << "(got " << kernel.clusterDim.x << ", " << kernel.clusterDim.y
                              << ") for kernel: " << kernel.kernelName << std::endl;
                    return hipErrorInvalidValue;
                }

                // Incomplete clusters can hang in hipDrvLaunchKernelEx. Do not pad
                // here: the work partition and reduction state already use this grid.
                if(kernel.numWorkGroups.x % kernel.clusterDim.x != 0
                   || kernel.numWorkGroups.y % kernel.clusterDim.y != 0)
                {
                    std::cerr << "hipDrvLaunchKernelEx: workgroup grid must be divisible by cluster "
                              << "dimensions (numWorkGroups " << kernel.numWorkGroups
                              << ", clusterDim " << kernel.clusterDim
                              << ") for kernel: " << kernel.kernelName << std::endl;
                    return hipErrorInvalidValue;
                }

                // hipDrvLaunchKernelEx enumerates the grid in workgroups.
                if(!fitsLaunchDim(kernel.numWorkGroups))
                {
                    std::cerr << "hipDrvLaunchKernelEx: workgroup count exceeds the 32-bit grid "
                              << "dimensions (numWorkGroups " << kernel.numWorkGroups
                              << ") for kernel: " << kernel.kernelName << std::endl;
                    return hipErrorInvalidValue;
                }
            }
            else
            {
                // hipExtModuleLaunchKernel's globalWorkSize is in work items, so
                // the workGroupSize * numWorkGroups product is what has to fit.
                if(!fitsLaunchDim(kernel.numWorkItems))
                {
                    std::cerr << "hipExtModuleLaunchKernel: work-item count exceeds the 32-bit "
                              << "globalWorkSize parameters (numWorkItems " << kernel.numWorkItems
                              << ", from numWorkGroups " << kernel.numWorkGroups
                              << " of workgroup size " << kernel.workGroupSize
                              << ") for kernel: " << kernel.kernelName << std::endl;
                    return hipErrorInvalidValue;
                }
            }

            if(!isKernelLoaded && !kernel.codeObjectFile.empty())
            {
                FindCodeObject(kernel.codeObjectFile);
            }

            if(m_debug)
            {
                std::cout << "Kernel " << kernel.kernelName << std::endl;
                std::cout << " l" << kernel.workGroupSize << " x g" << kernel.numWorkGroups << " = "
                          << kernel.numWorkItems << std::endl;
                std::cout << kernel.args;
            }
            if(m_debugSkipLaunch)
            {
                std::cout << "DEBUG: Skip kernel execution" << std::endl;
                if(startEvent != nullptr)
                    HIP_CHECK_RETURN(hipEventRecord(startEvent, stream));
                if(stopEvent != nullptr)
                    HIP_CHECK_RETURN(hipEventRecord(stopEvent, stream));
                return hipSuccess;
            }

            hipFunction_t function;
            auto          findKernel = [&]() {
#ifdef TENSILELITE_ENABLE_LOCAL_KERNELS
                // Helper kernels (beta-only, output conversion) carry their
                // solution's code object name but live in the shipped helpers.
                if(isLocalCodeObject(kernel.codeObjectFile))
                {
                    hipError_t err
                        = getLocalKernel(function, kernel.codeObjectFile, kernel.kernelName);
                    if(err != hipErrorNotFound)
                        return err;
                }
#endif
                return getKernel(function, kernel.kernelName);
            };
            HIP_CHECK_RETURN_WITH_LOG(
                findKernel(), [&](hipError_t error) {
                    std::cerr << "getKernel failed: " << kernel.kernelName << std::endl
                            << " with workgroup size: " << kernel.workGroupSize << std::endl
                            << " with numWorkGroups : " << kernel.numWorkGroups << std::endl
                            << " with numWorkItems : " << kernel.numWorkItems << std::endl
                            << " error: " << hipGetErrorString(error) << std::endl;
                });

            void*  kernelArgs = const_cast<void*>(kernel.args.data());
            size_t argsSize   = kernel.args.size();

            void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                       kernelArgs,
                                       HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                       &argsSize,
                                       HIP_LAUNCH_PARAM_END};

            if(startEvent != nullptr)
                HIP_CHECK_RETURN(hipEventRecord(startEvent, stream));

#ifdef HIP_HAS_CLUSTER_LAUNCH
            if(enableCluster)
            {
                HIP_LAUNCH_CONFIG config = {0};
                // The grid dimension is not affected by cluster launch, and is still enumerated
                // using number of blocks.
                // The grid dimension should be a multiple of cluster size.
                config.gridDimX = kernel.numWorkGroups.x;
                config.gridDimY = kernel.numWorkGroups.y;
                config.gridDimZ = kernel.numWorkGroups.z;
                config.blockDimX = kernel.workGroupSize.x;
                config.blockDimY = kernel.workGroupSize.y;
                config.blockDimZ = kernel.workGroupSize.z;

                hipLaunchAttribute attribute[1];
                attribute[0].id                 = hipLaunchAttributeClusterDimension;
                attribute[0].val.clusterDim.x = kernel.clusterDim.x;
                attribute[0].val.clusterDim.y = kernel.clusterDim.y;
                attribute[0].val.clusterDim.z = 1;
                config.attrs = attribute;
                config.numAttrs = 1;
                config.sharedMemBytes = kernel.sharedMemBytes;
                config.hStream = stream;

                const HIP_LAUNCH_CONFIG *pConfig = &config;
                HIP_CHECK_RETURN_WITH_LOG(hipDrvLaunchKernelEx(pConfig,
                                                               function,
                                                               nullptr,
                                                               (void**)&hipLaunchParams),
                    [&](hipError_t error) {
                        std::cerr << "hipDrvLaunchKernelEx failed: " << kernel.kernelName << std::endl
                                  << " with workgroup size: " << kernel.workGroupSize << std::endl
                                  << " with numWorkGroups : " << kernel.numWorkGroups << std::endl
                                  << " with numWorkItems : " << kernel.numWorkItems << std::endl
                                  << " error: " << hipGetErrorString(error) << std::endl;
                    }
                );
            }
            else
#endif
            {
                HIP_CHECK_RETURN_WITH_LOG(hipExtModuleLaunchKernel(function,
                                                          kernel.numWorkItems.x,
                                                          kernel.numWorkItems.y,
                                                          kernel.numWorkItems.z,
                                                          kernel.workGroupSize.x,
                                                          kernel.workGroupSize.y,
                                                          kernel.workGroupSize.z,
                                                          kernel.sharedMemBytes,
                                                          stream,
                                                          nullptr,
                                                          (void**)&hipLaunchParams,
                                                          nullptr,
                                                          nullptr
                                                          ),
                    [&](hipError_t error) {
                        std::cerr << "hipExtModuleLaunchKernel failed: " << kernel.kernelName << std::endl
                                  << " with workgroup size: " << kernel.workGroupSize << std::endl
                                  << " with numWorkGroups : " << kernel.numWorkGroups << std::endl
                                  << " with numWorkItems : " << kernel.numWorkItems << std::endl
                                  << " error: " << hipGetErrorString(error) << std::endl;
                    }
                );
            }

            if(stopEvent != nullptr)
                HIP_CHECK_RETURN(hipEventRecord(stopEvent, stream));
            return hipSuccess;
        }

        hipError_t SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels)
        {
            for(auto const& k : kernels)
            {
                HIP_CHECK_RETURN_WITH_LOG(launchKernel(k),
                    [&](hipError_t error) {
                        std::cerr << "launchKernel failed: " << k.kernelName << std::endl
                                << " with workgroup size: " << k.workGroupSize << std::endl
                                << " with numWorkGroups : " << k.numWorkGroups << std::endl
                                << " with numWorkItems : " << k.numWorkItems << std::endl
                                << " error: " << hipGetErrorString(error) << std::endl;
                    }
                );
            }
            return hipSuccess;
        }

        hipError_t SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels,
                                                  hipStream_t                          stream,
                                                  hipEvent_t                           startEvent,
                                                  hipEvent_t                           stopEvent,
                                                  bool                                 isKernelLoaded)
        {
            auto first = kernels.begin();
            auto last  = kernels.end() - 1;

            for(auto iter = kernels.begin(); iter != kernels.end(); iter++)
            {
                hipEvent_t kStart = nullptr;
                hipEvent_t kStop  = nullptr;

                if(iter == first)
                    kStart = startEvent;
                if(iter == last)
                    kStop = stopEvent;

                HIP_CHECK_RETURN_WITH_LOG(launchKernel(*iter, stream, kStart, kStop, isKernelLoaded),
                    [&](hipError_t error) {
                        std::cerr << "launchKernel failed: " << iter->kernelName << std::endl
                                << " with workgroup size: " << iter->workGroupSize << std::endl
                                << " with numWorkGroups : " << iter->numWorkGroups << std::endl
                                << " with numWorkItems : " << iter->numWorkItems << std::endl
                                << " error: " << hipGetErrorString(error) << std::endl;
                    }
                );
            }
            return hipSuccess;
        }

        hipError_t SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels,
                                                  hipStream_t                          stream,
                                                  std::vector<hipEvent_t> const&       startEvents,
                                                  std::vector<hipEvent_t> const&       stopEvents)
        {
            if(kernels.size() != startEvents.size() || kernels.size() != stopEvents.size())
                throw std::runtime_error(concatenate("Must have an equal number of kernels (",
                                                     kernels.size(),
                                                     "), start events (",
                                                     startEvents.size(),
                                                     "), and stop events. (",
                                                     stopEvents.size(),
                                                     ")"));

            for(size_t i = 0; i < kernels.size(); i++)
            {
                HIP_CHECK_RETURN_WITH_LOG(launchKernel(kernels[i], stream, startEvents[i], stopEvents[i]),
                    [&](hipError_t error) {
                        std::cerr << "launchKernel failed: " << kernels[i].kernelName << std::endl
                                << " with workgroup size: " << kernels[i].workGroupSize << std::endl
                                << " with numWorkGroups : " << kernels[i].numWorkGroups << std::endl
                                << " with numWorkItems : " << kernels[i].numWorkItems << std::endl
                                << " error: " << hipGetErrorString(error) << std::endl;
                    }
                );
            }
            return hipSuccess;
        }

        std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter)
        {
            stream << "hip::SolutionAdapter";

            if(adapter.m_debug)
            {
                stream << "[" << std::endl;
                for(auto const& name : adapter.m_loadedModuleNames)
                    stream << name << std::endl;

                stream << "]";
            }

            stream << " (" << adapter.name() << ", " << adapter.m_modules.size()
                   << " total modules)" << std::endl;

            return stream;
        }

        std::ostream& operator<<(std::ostream& stream, std::shared_ptr<SolutionAdapter> const& ptr)
        {
            if(ptr)
            {
                return stream << "*" << *ptr;
            }
            else
            {
                return stream << "(nullptr)";
            }
        }
    } // namespace hip
} // namespace TensileLite
