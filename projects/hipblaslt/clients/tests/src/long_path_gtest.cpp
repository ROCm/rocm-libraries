// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The fixture creates long paths independently of the loader's conversion.
// Windows runs exercise the MAX_PATH workaround; other platforms open these
// paths without conversion.

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace
{
    // The verbatim form of `path`, built here rather than with the code under
    // test, so the fixture can create files the library is then asked to open.
    fs::path verbatim(fs::path const& path)
    {
#ifdef _WIN32
        return fs::path(L"\\\\?\\" + fs::absolute(path).lexically_normal().native());
#else
        return path;
#endif
    }

    void writeFile(fs::path const& path, std::vector<uint8_t> const& bytes)
    {
        std::ofstream out(verbatim(path), std::ios::binary);
        ASSERT_TRUE(out.good()) << "cannot create " << path.string();
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        ASSERT_TRUE(out.good()) << "cannot write " << path.string();
    }

#ifdef TENSILE_MSGPACK
    // The smallest library the loader accepts, in the indexed layout: a solution
    // table with one entry and a tree that is a single leaf for it. MessagePack
    // for {format_version: 2, solutions_index: [0, 0, 1],
    // solutions_blob: bin(0x01), library: {type: "Single", index: 0}}.
    std::vector<uint8_t> minimalLibrary()
    {
        std::vector<uint8_t> out;
        auto                 str = [&](std::string const& s) {
            out.push_back(static_cast<uint8_t>(0xa0 | s.size()));
            out.insert(out.end(), s.begin(), s.end());
        };
        out.push_back(0x84);
        str("format_version");
        out.push_back(0x02);
        str("solutions_index");
        out.insert(out.end(), {0x93, 0x00, 0x00, 0x01});
        str("solutions_blob");
        out.insert(out.end(), {0xc4, 0x01, 0x01});
        str("library");
        out.push_back(0x82);
        str("type");
        str("Single");
        str("index");
        out.push_back(0x00);
        return out;
    }

    // minimalLibrary() deflated with zlib at level 9, as shipped .dat.zlib files are.
    const std::vector<uint8_t> minimalLibraryZlib
        = {0x78, 0xda, 0x6b, 0x59, 0x97, 0x96, 0x5f, 0x94, 0x9b, 0x58, 0x12, 0x5f, 0x96, 0x5a,
           0x54, 0x9c, 0x99, 0x9f, 0xc7, 0xb4, 0xbe, 0x38, 0x3f, 0xa7, 0xb4, 0x04, 0xc8, 0x2a,
           0x8e, 0xcf, 0xcc, 0x4b, 0x49, 0xad, 0x98, 0xcc, 0xc0, 0xc0, 0xb8, 0x0e, 0x21, 0x96,
           0x94, 0x93, 0x9f, 0x74, 0x84, 0x91, 0x71, 0x79, 0x4e, 0x66, 0x52, 0x51, 0x62, 0x51,
           0x65, 0xd3, 0x92, 0x92, 0xca, 0x82, 0xd4, 0x65, 0xc1, 0x99, 0x79, 0xe9, 0x39, 0xa9,
           0x4b, 0xc1, 0x1a, 0x18, 0x00, 0xb2, 0xa0, 0x22, 0x78};
#endif

    class LongPathLoading : public ::testing::Test
    {
    protected:
        fs::path root;
        fs::path dir;

        void SetUp() override
        {
#ifdef _WIN32
            auto const pid = _getpid();
#else
            auto const pid = getpid();
#endif
            root = fs::temp_directory_path()
                   / ("hipblaslt-long-path-" + std::to_string(pid) + "-"
                      + ::testing::UnitTest::GetInstance()->current_test_info()->name());
            // Five 60-character levels put every file well past MAX_PATH,
            // whatever the length of the temp directory.
            dir = root;
            for(int level = 0; level < 5; ++level)
                dir /= std::string(59, 'd') + std::to_string(level);
            std::error_code ec;
            fs::create_directories(verbatim(dir), ec);
            ASSERT_FALSE(ec) << "cannot create " << dir.string() << ": " << ec.message();
        }

        void TearDown() override
        {
            std::error_code ec;
            fs::remove_all(verbatim(root), ec);
        }

        std::string pathOf(std::string const& name) const
        {
            auto const path = (dir / name).string();
            EXPECT_GE(fs::absolute(path).string().size(), 300u)
                << "the fixture path is too short to test MAX_PATH: " << path;
            return path;
        }
    };

#ifdef TENSILE_MSGPACK
    std::shared_ptr<TensileLite::SolutionLibrary<TensileLite::ContractionProblemGemm,
                                                 TensileLite::ContractionSolution>>
        loadLibrary(std::string const& path)
    {
        return TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm,
                                            TensileLite::ContractionSolution>(path);
    }

    TEST_F(LongPathLoading, smoke_LoadsUncompressedLibrary)
    {
        auto const path = pathOf("TensileLibrary_long_path.dat");
        ASSERT_NO_FATAL_FAILURE(writeFile(path, minimalLibrary()));

        EXPECT_NE(loadLibrary(path), nullptr) << "could not load " << path;
    }

    // Shipped libraries are compressed; the loader is given the .dat name and
    // probes for the .dat.zlib beside it.
    TEST_F(LongPathLoading, smoke_LoadsCompressedLibrary)
    {
        auto const path = pathOf("TensileLibrary_long_path.dat");
        ASSERT_NO_FATAL_FAILURE(writeFile(path + ".zlib", minimalLibraryZlib));

        EXPECT_NE(loadLibrary(path), nullptr) << "could not load " << path << ".zlib";
    }
#endif

    // Stands in for the HIP runtime: hipModuleLoad opens the narrow path it is
    // given, as the runtime does, and hipModuleLoadData records the image.
    struct FakeModules
    {
        std::map<hipModule_t, std::unique_ptr<int>> live;
        std::vector<char>                           loadedImage;
        unsigned                                    pathLoads = 0;
        unsigned                                    dataLoads = 0;
    };

    FakeModules* fakeModules       = nullptr;
    const char   codeObjectBytes[] = "a stand-in for a code object";

    hipModule_t newModule()
    {
        auto storage = std::make_unique<int>(0);
        auto module  = reinterpret_cast<hipModule_t>(storage.get());
        fakeModules->live.emplace(module, std::move(storage));
        return module;
    }

    hipError_t fakeLoad(hipModule_t* module, const char* path)
    {
        ++fakeModules->pathLoads;
        std::ifstream file(path, std::ios::binary);
        if(!file)
            return hipErrorFileNotFound;
        *module = newModule();
        return hipSuccess;
    }

    hipError_t fakeLoadData(hipModule_t* module, const void* image)
    {
        ++fakeModules->dataLoads;
        auto const* bytes = static_cast<const char*>(image);
        fakeModules->loadedImage.assign(bytes, bytes + sizeof(codeObjectBytes));
        *module = newModule();
        return hipSuccess;
    }

    hipError_t fakeUnload(hipModule_t module)
    {
        EXPECT_EQ(fakeModules->live.erase(module), 1u) << "double or unowned module unload";
        return hipSuccess;
    }

    hipError_t fakeGetLastError()
    {
        return hipSuccess;
    }

    TEST_F(LongPathLoading, smoke_LoadsCodeObject)
    {
        auto const path = pathOf("TensileLibrary_long_path.co");
        ASSERT_NO_FATAL_FAILURE(writeFile(
            path,
            std::vector<uint8_t>(codeObjectBytes, codeObjectBytes + sizeof(codeObjectBytes))));

        FakeModules modules;
        fakeModules = &modules;
        {
            TensileLite::hip::SolutionAdapter adapter(TensileLite::hip::SolutionAdapter::ModuleApi{
                fakeLoad, fakeUnload, fakeGetLastError, fakeLoadData});

            EXPECT_EQ(adapter.loadCodeObjectFile(path), hipSuccess) << "could not load " << path;
            EXPECT_EQ(modules.live.size(), 1u);
#ifdef _WIN32
            EXPECT_EQ(modules.pathLoads, 0u)
                << "the runtime was handed a path Windows will not open";
            ASSERT_EQ(modules.dataLoads, 1u);
            EXPECT_EQ(
                std::memcmp(modules.loadedImage.data(), codeObjectBytes, sizeof(codeObjectBytes)),
                0)
                << "the image loaded is not the file's contents";
#endif
        }
        EXPECT_TRUE(modules.live.empty()) << "the adapter leaked module handles";
        fakeModules = nullptr;
    }
} // namespace
