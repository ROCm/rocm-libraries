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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/PlaceholderLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include "TestData.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

// TENSILE_MSGPACK is private to TensileHost, so detect the msgpack header directly.
#if __has_include(<msgpack.hpp>)
#ifndef MSGPACK_NO_BOOST
#define MSGPACK_NO_BOOST
#endif
#include <msgpack.hpp>
#define PLACEHOLDER_TEST_MSGPACK 1
#endif

using namespace Tensile;

namespace
{
    using Placeholder = PlaceholderLibrary<ContractionProblem>;

    const std::string asmPrefix = "TensileLibrary_Type_SS_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx90a";
    const std::string fallbackPrefix
        = "TensileLibrary_Type_SS_Contraction_l_Ailk_Bljk_Cijk_Dijk_fallback";

    ContractionSolution makeSolution(bool sourceKernel)
    {
        ContractionSolution solution;
        solution.sizeMapping.sourceKernel = sourceKernel;
        return solution;
    }
}

// The architecture-independent overload is used when placeholder libraries are
// preloaded without hardware. Names that do not depend on the architecture must
// be returned; names that do must not be guessed.
TEST(PlaceholderLibraryCodeObject, AssemblyKernelNameIsArchitectureIndependent)
{
    Placeholder lib;
    lib.filePrefix = asmPrefix;
    auto solution  = makeSolution(false);

    EXPECT_EQ(lib.getCodeObjectFileName(std::string(), solution), asmPrefix + ".co");
    EXPECT_EQ(lib.getCodeObjectFileName(std::string("gfx90a"), solution), asmPrefix + ".co");
}

TEST(PlaceholderLibraryCodeObject, SourceKernelNameWithoutFallback)
{
    Placeholder lib;
    lib.filePrefix = asmPrefix;
    auto solution  = makeSolution(true);

    EXPECT_EQ(lib.getCodeObjectFileName(std::string(), solution), asmPrefix + ".hsaco");
}

TEST(PlaceholderLibraryCodeObject, FallbackNameNeedsArchitecture)
{
    Placeholder lib;
    lib.filePrefix = fallbackPrefix;
    auto solution  = makeSolution(true);

    EXPECT_EQ(lib.getCodeObjectFileName(std::string(), solution), std::string());
    EXPECT_EQ(lib.getCodeObjectFileName(std::string("gfx90a"), solution),
              fallbackPrefix + "_gfx90a.hsaco");
    EXPECT_EQ(lib.getCodeObjectFileName(std::string("gfx90a:xnack-"), solution),
              fallbackPrefix + "_gfx90a.hsaco");
}

TEST(PlaceholderLibraryCodeObject, HardwareOverloadMatchesArchitectureString)
{
    Placeholder lib;
    lib.filePrefix  = fallbackPrefix;
    auto   solution = makeSolution(true);
    AMDGPU hardware(AMDGPU::Processor::gfx90a, 104, 0, "test");

    EXPECT_EQ(lib.getCodeObjectFileName(hardware, solution),
              lib.getCodeObjectFileName(hardware.archName(), solution));
}

#ifdef PLACEHOLDER_TEST_MSGPACK
// Solutions of a placeholder library that is preloaded at load time (as rocBLAS does
// for the device architecture) can be reached by index, e.g. through a problem
// override file, without going through findBestSolution/findAllSolutions. They must
// already know their code object file, otherwise launching them fails.
TEST(PlaceholderLibraryCodeObject, PreloadedSolutionsKnowTheirCodeObject)
{
    auto source = TestData::Instance().file("KernelsLite", "dat");
    if(!std::filesystem::is_regular_file(source))
        GTEST_SKIP() << "KernelsLite.dat not available (Tensile built without msgpack)";

    auto dir = std::filesystem::temp_directory_path()
               / ("tensile_placeholder_preload_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);

    // The placeholder file is a complete master library: reuse KernelsLite.
    std::filesystem::copy_file(
        source, dir / (asmPrefix + ".dat"), std::filesystem::copy_options::overwrite_existing);

    // Lazy master library: no solutions of its own, one placeholder.
    auto masterFile = dir / "TensileLibrary_lazy_gfx90a.dat";
    {
        msgpack::sbuffer                  buffer;
        msgpack::packer<msgpack::sbuffer> pk(buffer);
        pk.pack_map(2);
        pk.pack(std::string("solutions"));
        pk.pack_array(0);
        pk.pack(std::string("library"));
        pk.pack_map(2);
        pk.pack(std::string("type"));
        pk.pack(std::string("Placeholder"));
        pk.pack(std::string("value"));
        pk.pack(asmPrefix);
        std::ofstream out(masterFile, std::ios::binary);
        out.write(buffer.data(), buffer.size());
    }

    auto library = LoadLibraryFilePreload<ContractionProblem>(masterFile.native(),
                                                              {LazyLoadingInit::gfx90a});
    ASSERT_NE(library, nullptr);
    auto master = std::dynamic_pointer_cast<MasterSolutionLibrary<ContractionProblem>>(library);
    ASSERT_NE(master, nullptr);
    ASSERT_FALSE(master->solutions.empty());

    int asmIndex = -1;
    for(auto const& entry : master->solutions)
    {
        if(entry.second->isSourceKernel())
            continue;
        EXPECT_EQ(entry.second->codeObjectFilename.load(), asmPrefix + ".co")
            << entry.second->name();
        asmIndex = entry.first;
    }
    ASSERT_GE(asmIndex, 0) << "KernelsLite contains no assembly kernels";

    auto byIndex = master->getSolutionByIndex(asmIndex);
    ASSERT_NE(byIndex, nullptr);
    EXPECT_EQ(byIndex->codeObjectFilename.load(), asmPrefix + ".co");

    std::filesystem::remove_all(dir);
}
#endif
