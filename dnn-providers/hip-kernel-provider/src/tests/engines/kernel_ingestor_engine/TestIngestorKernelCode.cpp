// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "compilation/ICompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestIngestorKernelCode.cpp
 * @brief The two checks buildIngestorKernelCode makes before loading anything: path
 *        confinement, and the packaged-versus-marshalled argument signature. Called rather
 *        than copied.
 *
 * TestPackedDescriptorLoad.cpp reproduces the confinement rule inline, so deleting the
 * guard leaves that suite green; these cases turn red. No device and no archive on disk are
 * needed -- both checks throw before the first HIP call and before kpackLoader.load.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using hip_kernel_provider::kernel_ingestor_engine::testing::buildPointwiseGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::GraphFixture;
using hipdnn_plugin_sdk::HipdnnPluginException;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;

constexpr const char* SCRATCH_LABEL = "ingestorkernelcode";

/// buildIngestorKernelCode takes an IKernelCompiler by reference, so the KPACK path needs
/// an object for a compiler it never reaches. A call to it means the kind switch took the
/// wrong arm, so it throws rather than returning a silently empty program.
class UnreachableCompiler : public compilation::IKernelCompiler
{
public:
    std::unique_ptr<compilation::ICompiledProgram>
        compile(const std::string& kernelFileName,
                const std::vector<std::string>& /*options*/) const override
    {
        throw std::runtime_error("the KPACK path must not reach the source compiler; asked for '"
                                 + kernelFileName + "'");
    }
};

/// One unnamed device pointer, the shape clang records for a HIP kernel parameter.
KernelArgument buffer(uint32_t offset)
{
    return KernelArgument{"global_buffer", 8, offset, ""};
}

/// What both sides of the signature comparison say unless a case overrides one of them:
/// three device pointers, as the pointwise pack marshals.
std::vector<KernelArgument> threeBuffers()
{
    return {buffer(0), buffer(8), buffer(16)};
}

/// Nothing it names has to exist on disk.
KernelDefinition makeKpackKernel(const std::filesystem::path& originDirectory,
                                 const std::filesystem::path& treeRoot,
                                 const std::string& library,
                                 const std::vector<KernelArgument>& signature = threeBuffers())
{
    KernelDefinition kernel;
    kernel.kernelId.fill(0x21);
    kernel.packId.fill(0x22);
    kernel.dispatchId.fill(0x23);
    kernel.name = "pointwise_add_f32_kpack";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.source.library = library;
    kernel.source.tocKey = "PointwiseAdd/block64";
    kernel.source.symbol = "PointwiseAdd";
    kernel.source.signature = signature;
    kernel.originDirectory = originDirectory;
    kernel.treeRoot = treeRoot;
    return kernel;
}

/// Everything buildIngestorKernelCode needs except the kernel, held together so each case
/// is the one KernelDefinition it is about.
class GuardHarness
{
public:
    /// KernelCompileOptions has no default constructor and reads its tensor argument
    /// eagerly, so it is built from the fixture's graph rather than stubbed.
    GuardHarness()
        : _options(&firstTensorOf(_fixture), _fixture.deviceProperties().gcnArchName)
    {
    }

    IngestorKernelCode build(const KernelDefinition& kernel,
                             const std::vector<KernelArgument>& expected = threeBuffers())
    {
        return buildIngestorKernelCode(
            _compiler, _loader, _fixture.context(), kernel, _options, expected);
    }

private:
    static const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes&
        firstTensorOf(const GraphFixture& fixture)
    {
        const auto& tensors = fixture.context().graph.getTensorMap();
        for(const auto& [uid, attributes] : tensors)
        {
            static_cast<void>(uid);
            if(attributes != nullptr)
            {
                return *attributes;
            }
        }
        throw std::runtime_error("the pointwise fixture graph carries no tensor to compile for");
    }

    UnreachableCompiler _compiler;
    compilation::KpackModuleCache _cache;
    compilation::KpackKernelLoader _loader{_cache};
    GraphFixture _fixture{buildPointwiseGraph()};
    compilation::KernelCompileOptions _options;
};

constexpr const char* OUTSIDE_THE_TREE = "outside the descriptor tree";

// ---------------------------------------------------------------------------
// Path confinement
// ---------------------------------------------------------------------------

TEST(TestIngestorKernelCode, RejectsALibraryThatEscapesTheDescriptorTree)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const std::filesystem::path tree = scratch.path() / "tree";
    ASSERT_TRUE(std::filesystem::create_directory(tree));

    GuardHarness harness;
    const auto kernel = makeKpackKernel(tree, tree, "../outside/x.kpack");

    try
    {
        harness.build(kernel);
        FAIL() << "expected a library outside the descriptor tree to be rejected";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        // Both halves: the resolved path says what was asked for, the boundary says what it
        // was measured against, and a reader needs the pair to place the fault.
        const std::string what = error.what();
        EXPECT_NE(what.find(OUTSIDE_THE_TREE), std::string::npos) << what;
        EXPECT_NE(what.find("x.kpack"), std::string::npos) << what;
        EXPECT_NE(what.find(tree.filename().string()), std::string::npos) << what;
    }
}

TEST(TestIngestorKernelCode, RejectsAnAbsoluteLibraryOutsideTheTree)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const std::filesystem::path tree = scratch.path() / "tree";
    const std::filesystem::path elsewhere = scratch.path() / "elsewhere";
    ASSERT_TRUE(std::filesystem::create_directory(tree));
    ASSERT_TRUE(std::filesystem::create_directory(elsewhere));

    GuardHarness harness;
    // An absolute path bypasses originDirectory entirely, and weakly_canonical normalises it
    // rather than rejecting it.
    const auto kernel = makeKpackKernel(tree, tree, (elsewhere / "x.kpack").generic_string());

    try
    {
        harness.build(kernel);
        FAIL() << "expected an absolute library outside the tree to be rejected";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find(OUTSIDE_THE_TREE), std::string::npos)
            << error.what();
    }
}

TEST(TestIngestorKernelCode, AcceptsALibraryThatClimbsOutOfItsOwnDirectoryButStaysInTheTree)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const std::filesystem::path tree = scratch.path() / "tree";
    const std::filesystem::path nested = tree / "pointwise";
    ASSERT_TRUE(std::filesystem::create_directory(tree));
    ASSERT_TRUE(std::filesystem::create_directory(nested));

    GuardHarness harness;
    // Packing preserves the authored subpath, so a boundary at originDirectory rather than
    // the tree root would reject this while flat fixture trees stayed green.
    const auto kernel = makeKpackKernel(nested, tree, "../absent.kpack");

    try
    {
        harness.build(kernel);
        FAIL() << "expected the absent archive to be reported";
    }
    catch(const HipdnnPluginException& error)
    {
        const std::string what = error.what();
        EXPECT_EQ(what.find(OUTSIDE_THE_TREE), std::string::npos)
            << "the guard rejected a path that stays inside the tree: " << what;
        EXPECT_NE(what.find("does not exist"), std::string::npos) << what;
    }
}

TEST(TestIngestorKernelCode, DoesNotConfuseASiblingPrefixWithTheTree)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const std::filesystem::path tree = scratch.path() / "base";
    const std::filesystem::path sibling = scratch.path() / "base-evil";
    ASSERT_TRUE(std::filesystem::create_directory(tree));
    ASSERT_TRUE(std::filesystem::create_directory(sibling));

    GuardHarness harness;
    // A string-prefix containment test would read `/base-evil` as inside `/base`;
    // lexically_relative walks components instead.
    const auto kernel = makeKpackKernel(tree, tree, "../base-evil/x.kpack");

    try
    {
        harness.build(kernel);
        FAIL() << "expected a sibling directory sharing the tree's prefix to be rejected";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find(OUTSIDE_THE_TREE), std::string::npos)
            << error.what();
    }
}

// ---------------------------------------------------------------------------
// Link substitution -- a symlink or junction inside the tree
// ---------------------------------------------------------------------------

/// One mechanism everywhere: Windows needs Developer Mode or SeCreateSymbolicLinkPrivilege,
/// and without either the case skips rather than silently falling back to a junction.
bool createDirectoryLink(const std::filesystem::path& link,
                         const std::filesystem::path& target,
                         std::string& failure)
{
    std::error_code error;
    std::filesystem::create_directory_symlink(target, link, error);
    if(error)
    {
        failure = "create_directory_symlink: " + error.message();
        return false;
    }

    // Confirmed rather than inferred from the absent error_code: a creation that silently
    // did nothing and a platform that cannot see links both leave is_symlink(link) false
    // below, and those two want opposite responses.
    if(!std::filesystem::exists(std::filesystem::symlink_status(link)))
    {
        failure
            = "create_directory_symlink reported success but created nothing at " + link.string();
        return false;
    }
    return true;
}

TEST(TestIngestorKernelCode, RejectsALibraryReachedThroughALinkInsideTheTree)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const std::filesystem::path tree = scratch.path() / "tree";
    const std::filesystem::path real = tree / "real";
    const std::filesystem::path link = tree / "link";
    ASSERT_TRUE(std::filesystem::create_directory(tree));
    ASSERT_TRUE(std::filesystem::create_directory(real));

    std::string failure;
    if(!createDirectoryLink(link, real, failure))
    {
        GTEST_SKIP() << "could not create a directory link, so link rejection has no executed "
                        "evidence on this machine: "
                     << failure;
    }

    // Asserted rather than skipped on: the guard reads the same standard library's
    // symlink_status, so a link this does not see is a link the guard cannot see either --
    // a real gap in the mitigation. Creation is confirmed above, so that is all this can
    // mean.
    ASSERT_TRUE(std::filesystem::is_symlink(link))
        << "the platform does not report the created component as a link, which means the "
           "production guard cannot see it either: "
        << link;

    GuardHarness harness;
    // Both paths stay inside the tree, so the containment check passes and only the symlink
    // walk can reject this.
    const auto kernel = makeKpackKernel(tree, tree, "link/x.kpack");

    try
    {
        harness.build(kernel);
        FAIL() << "expected a library reached through a link inside the tree to be rejected";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        const std::string what = error.what();
        EXPECT_NE(what.find("through the link"), std::string::npos) << what;
        EXPECT_NE(what.find("link"), std::string::npos) << what;
    }
}

#ifdef _WIN32
TEST(TestIngestorKernelCode, RejectsALibraryReachedThroughAJunctionInsideTheTree)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const std::filesystem::path tree = scratch.path() / "tree";
    const std::filesystem::path real = tree / "real";
    const std::filesystem::path junction = tree / "junction";
    ASSERT_TRUE(std::filesystem::create_directory(tree));
    ASSERT_TRUE(std::filesystem::create_directory(real));

    // mklink /J needs no privilege, so it is the substitution a Windows caller can actually
    // perform on the machines where the symlink case skips. It is a cmd builtin, hence system().
    const std::string command
        = "mklink /J \"" + junction.string() + "\" \"" + real.string() + "\" >nul 2>&1";
    if(std::system(command.c_str()) != 0)
    {
        GTEST_SKIP() << "mklink /J failed, so the junction half of the guard has no executed "
                        "evidence on this machine";
    }

    // MSVC gives a junction its own file_type, so is_symlink answers false -- which is why the
    // guard tests the kind rather than asking is_symlink.
    const std::filesystem::file_type kind = std::filesystem::symlink_status(junction).type();
    ASSERT_EQ(kind, std::filesystem::file_type::junction) << junction;
    ASSERT_FALSE(std::filesystem::is_symlink(junction)) << junction;

    GuardHarness harness;
    const auto kernel = makeKpackKernel(tree, tree, "junction/x.kpack");

    try
    {
        harness.build(kernel);
        FAIL() << "expected a library reached through a junction inside the tree to be rejected";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        const std::string what = error.what();
        EXPECT_NE(what.find("through the link"), std::string::npos) << what;
    }
}
#endif // _WIN32

// ---------------------------------------------------------------------------
// Signature comparison
// ---------------------------------------------------------------------------

/// A library that does not exist is enough for these: the comparison runs before the load,
/// so a case that reaches the loader has already failed to be rejected.
constexpr const char* ABSENT_LIBRARY = "absent.kpack";

TEST(TestIngestorKernelCode, RejectsAPackagedKernelTakingMoreArgumentsThanThePackMarshals)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    GuardHarness harness;
    auto signature = threeBuffers();
    signature.push_back(KernelArgument{"by_value", 4, 24, ""});
    const auto kernel = makeKpackKernel(scratch.path(), scratch.path(), ABSENT_LIBRARY, signature);

    try
    {
        harness.build(kernel);
        FAIL() << "expected a fourth packaged argument to be rejected";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        // Both lists, not just the verdict: the reader has to see which side gained the
        // argument without opening the archive.
        const std::string what = error.what();
        EXPECT_NE(what.find("by_value:4@24"), std::string::npos) << what;
        EXPECT_NE(what.find("this pack launches it with"), std::string::npos) << what;
        EXPECT_NE(what.find("PointwiseAdd"), std::string::npos) << what;
    }
}

TEST(TestIngestorKernelCode, RejectsAByValueArgumentWhoseWidthDiffers)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    GuardHarness harness;
    const auto kernel = makeKpackKernel(scratch.path(),
                                        scratch.path(),
                                        ABSENT_LIBRARY,
                                        {buffer(0), KernelArgument{"by_value", 8, 8, ""}});

    try
    {
        harness.build(kernel, {buffer(0), KernelArgument{"by_value", 4, 8, ""}});
        FAIL() << "expected a widened by-value argument to be rejected";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find("by_value:8@8"), std::string::npos)
            << error.what();
    }
}

TEST(TestIngestorKernelCode, RejectsAnEqualArityPermutationWhenBothSidesRecordNames)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    GuardHarness harness;
    // Both sides must keep their names: stripping them to HIP shape would leave this case
    // passing for having nothing to disagree about rather than for detecting the swap.
    const std::vector<KernelArgument> marshalled = {{"global_buffer", 8, 0, "inputA"},
                                                    {"global_buffer", 8, 8, "inputB"},
                                                    {"global_buffer", 8, 16, "output"}};
    const std::vector<KernelArgument> packaged = {{"global_buffer", 8, 0, "output"},
                                                  {"global_buffer", 8, 8, "inputA"},
                                                  {"global_buffer", 8, 16, "inputB"}};
    const auto kernel = makeKpackKernel(scratch.path(), scratch.path(), ABSENT_LIBRARY, packaged);

    try
    {
        harness.build(kernel, marshalled);
        FAIL() << "expected a permuted argument list to be rejected";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        const std::string what = error.what();
        EXPECT_NE(what.find("'output', global_buffer:8@8 'inputA'"), std::string::npos) << what;
        EXPECT_NE(what.find("'inputA', global_buffer:8@8 'inputB'"), std::string::npos) << what;
    }
}

TEST(TestIngestorKernelCode, AcceptsAMatchWhereOnlyOneSideRecordsNames)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    GuardHarness harness;
    const auto kernel = makeKpackKernel(scratch.path(), scratch.path(), ABSENT_LIBRARY);

    try
    {
        harness.build(kernel,
                      {{"global_buffer", 8, 0, "inputA"},
                       {"global_buffer", 8, 8, "inputB"},
                       {"global_buffer", 8, 16, "output"}});
        FAIL() << "expected the absent archive to be what fails, not the signature";
    }
    catch(const HipdnnPluginException& error)
    {
        // Reaching the loader IS the pass: the comparison let this through and the missing
        // file is the next thing to go wrong.
        EXPECT_NE(std::string(error.what()).find("does not exist"), std::string::npos)
            << error.what();
    }
}

TEST(TestIngestorKernelCode, DoesNotCompareOffsets)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    GuardHarness harness;
    const auto kernel = makeKpackKernel(scratch.path(), scratch.path(), ABSENT_LIBRARY);

    try
    {
        harness.build(kernel, {buffer(0), buffer(0), buffer(0)});
        FAIL() << "expected the absent archive to be what fails, not the signature";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_NE(std::string(error.what()).find("does not exist"), std::string::npos)
            << error.what();
    }
}

// ---------------------------------------------------------------------------
// Per-device resolution
// ---------------------------------------------------------------------------
//
// A plan is re-usable across handles, and a handle names its device only through its
// stream, so which device launches is not known until execute(). These cases pin which
// ordinal is admitted, which is refused, and which is answered without touching hardware
// at all.
//
// Faked rather than measured: proving that a second ordinal of ANOTHER architecture is
// refused needs a host holding two architectures at once, and no such host exists. The
// subclass below stands in for the two steps a new device needs -- reading its
// architecture, and loading its module -- so the decision logic above them is exercised
// on any machine, including one with no GPU.

/// Counts launches so a test can tell which device's kernel it was handed back.
class CountingKernel : public compilation::IRunnableKernel
{
public:
    explicit CountingKernel(int ordinal)
        : _ordinal(ordinal)
    {
    }

    void setBlockSize(unsigned int x, unsigned int y, unsigned int z) override
    {
        blockX = x;
        blockY = y;
        blockZ = z;
    }

    void setGridSize(unsigned int x, unsigned int y, unsigned int z) override
    {
        gridX = x;
        gridY = y;
        gridZ = z;
    }

    void setSharedMemBytes(unsigned int bytes) override
    {
        sharedMemBytes = bytes;
    }

    int ordinal() const
    {
        return _ordinal;
    }

    unsigned int blockX = 0;
    unsigned int blockY = 0;
    unsigned int blockZ = 0;
    unsigned int gridX = 0;
    unsigned int gridY = 0;
    unsigned int gridZ = 0;
    unsigned int sharedMemBytes = 0;

private:
    // Never reached: no case here launches. Present because the interface demands it.
    void launchImpl(hipStream_t /*stream*/, void** /*kernelParams*/) const override
    {
        FAIL() << "no case in this fixture launches";
    }

    int _ordinal;
};

class CountingProgram : public compilation::ICompiledProgram
{
public:
    explicit CountingProgram(int ordinal)
        : _ordinal(ordinal)
    {
    }

    std::unique_ptr<compilation::IRunnableKernel>
        getKernel(const std::string& /*kernelName*/) const override
    {
        return std::make_unique<CountingKernel>(_ordinal);
    }

private:
    int _ordinal;
};

/// Stands in for a machine: every ordinal's architecture is dictated, and loading a
/// module is recorded rather than performed.
class FakeDeviceCode : public IngestorKernelCode
{
public:
    FakeDeviceCode(std::map<int, std::string> architectures, int firstOrdinal)
        : IngestorKernelCode(std::make_unique<CountingProgram>(firstOrdinal),
                             std::make_unique<CountingKernel>(firstOrdinal),
                             firstOrdinal,
                             architectures.at(firstOrdinal))
        , _architectures(std::move(architectures))
    {
    }

    // Mutated from the const resolution path, which is what the object under test calls.
    mutable std::vector<int> archQueries;
    mutable std::vector<int> resolves;

protected:
    hipError_t queryDeviceArch(int deviceOrdinal, std::string& reportedArch) const override
    {
        archQueries.push_back(deviceOrdinal);
        const auto found = _architectures.find(deviceOrdinal);
        if(found == _architectures.end())
        {
            return hipErrorInvalidDevice;
        }
        reportedArch = found->second;
        return hipSuccess;
    }

    Resolved resolveForDevice(int deviceOrdinal, const std::string& /*reportedArch*/) const override
    {
        resolves.push_back(deviceOrdinal);
        return Resolved{std::make_unique<CountingProgram>(deviceOrdinal),
                        std::make_unique<CountingKernel>(deviceOrdinal)};
    }

private:
    std::map<int, std::string> _architectures;
};

TEST(TestIngestorKernelCodeDevice, ResolvesASecondOrdinalOfTheSameArchitecture)
{
    FakeDeviceCode code({{0, "gfx942:sramecc+:xnack-"}, {1, "gfx942:sramecc+:xnack-"}}, 0);
    code.setBlockSize(64, 1, 1);
    code.setGridSize(7, 1, 1);

    const auto& first = dynamic_cast<const CountingKernel&>(code.kernelFor(0));
    const auto& second = dynamic_cast<const CountingKernel&>(code.kernelFor(1));

    // Device 1 gets its own module rather than device 0's: a hipModule_t belongs to the
    // device it was loaded on.
    EXPECT_EQ(first.ordinal(), 0);
    EXPECT_EQ(second.ordinal(), 1);
    EXPECT_EQ(code.resolves, std::vector<int>{1});

    // Configured identically. Without the recorded geometry the second device would
    // launch 1x1x1 and quietly compute a fraction of the output.
    EXPECT_EQ(second.blockX, 64U);
    EXPECT_EQ(second.gridX, 7U);
}

TEST(TestIngestorKernelCodeDevice, RefusesAnOrdinalOfAnotherArchitecture)
{
    const FakeDeviceCode code({{0, "gfx942:sramecc+:xnack-"}, {1, "gfx950"}}, 0);

    try
    {
        code.kernelFor(1);
        FAIL() << "expected a device of another architecture to be refused";
    }
    catch(const HipdnnPluginException& error)
    {
        // Author-facing, not an internal inconsistency: the caller paired a plan with a
        // handle on an architecture it was never matched for.
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find("gfx950"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("gfx942"), std::string::npos) << error.what();
    }

    // Refused before anything was loaded for it.
    EXPECT_TRUE(code.resolves.empty());
}

TEST(TestIngestorKernelCodeDevice, AnswersASeenOrdinalWithoutQueryingOrLoading)
{
    const FakeDeviceCode code({{0, "gfx942:sramecc+:xnack-"}, {1, "gfx942:sramecc+:xnack-"}}, 0);

    code.kernelFor(1);
    const size_t queriesAfterFirst = code.archQueries.size();
    const size_t resolvesAfterFirst = code.resolves.size();

    const auto& repeat = dynamic_cast<const CountingKernel&>(code.kernelFor(1));

    // The fast path this exists for: a dispatch on a device already seen touches neither
    // HIP nor the archive.
    EXPECT_EQ(repeat.ordinal(), 1);
    EXPECT_EQ(code.archQueries.size(), queriesAfterFirst);
    EXPECT_EQ(code.resolves.size(), resolvesAfterFirst);

    // The ordinal the object was built for is likewise never re-queried.
    code.kernelFor(0);
    EXPECT_EQ(code.archQueries.size(), queriesAfterFirst);
}

/// A program that runs anywhere is answered without resolving a device at all. The
/// non-kpack path could not fail here before and must not start.
class AnyDeviceCode : public IngestorKernelCode
{
public:
    AnyDeviceCode()
        : IngestorKernelCode(std::make_unique<CountingProgram>(99),
                             std::make_unique<CountingKernel>(99))
    {
    }

    mutable int ordinalResolutions = 0;

protected:
    int resolveLaunchOrdinal(hipStream_t stream) const override
    {
        ++ordinalResolutions;
        return IngestorKernelCode::resolveLaunchOrdinal(stream);
    }
};

TEST(TestIngestorKernelCodeDevice, AnswersANonDeviceBoundProgramWithoutResolvingADevice)
{
    const AnyDeviceCode code;

    // A stream token that names no device: were the ordinal resolved eagerly, this would
    // query HIP and could throw. It must not be reached at all.
    const auto& kernel
        = dynamic_cast<const CountingKernel&>(code.kernelForStream(hipStreamPerThread));

    EXPECT_EQ(kernel.ordinal(), 99);

    // The discriminating assertion: resolving eagerly would consult HIP on a path that
    // has no device to resolve and previously could not fail here.
    EXPECT_EQ(code.ordinalResolutions, 0);
}

TEST(TestIngestorKernelCodeDevice, ReportsADeviceItCannotQuery)
{
    const FakeDeviceCode code({{0, "gfx942:sramecc+:xnack-"}}, 0);

    try
    {
        code.kernelFor(4);
        FAIL() << "expected an unqueryable device to be reported";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    }
    EXPECT_TRUE(code.resolves.empty());
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
