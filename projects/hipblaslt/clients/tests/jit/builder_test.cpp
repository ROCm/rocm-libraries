// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-source-bundle.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

// Builds the hand-written HIP kernel with comgr for each written bundle's
// target, then every written bundle with that kernel linked into its code
// object, and checks that the code object defines every expected kernel. Needs
// no GPU.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;
    namespace co = hj::code_object;
    using hipblaslt_jit_test::require;

    std::uint16_t le16(const unsigned char* p)
    {
        return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
    }

    std::uint32_t le32(const unsigned char* p)
    {
        return static_cast<std::uint32_t>(le16(p)) | (static_cast<std::uint32_t>(le16(p + 2)) << 16);
    }

    std::uint64_t le64(const unsigned char* p)
    {
        return static_cast<std::uint64_t>(le32(p)) | (static_cast<std::uint64_t>(le32(p + 4)) << 32);
    }

    // One kernel. The note's .wavefront_size is the requested size. There is no
    // .amdhsa_wavefront_size32 directive, so on a processor that has both sizes
    // the assembler flag is what sets the kernel descriptor.
    std::string waveKernel(int wave)
    {
        std::ostringstream source;
        source
            << ".amdgcn_target \"amdgcn-amd-amdhsa--gfx1030\"\n"
               ".text\n.globl k\n.p2align 8\n.type k,@function\nk:\n  s_endpgm\n"
               ".Lend:\n  .size k, .Lend-k\n\n.rodata\n.p2align 6\n.amdhsa_kernel k\n"
               "  .amdhsa_next_free_vgpr 1\n  .amdhsa_next_free_sgpr 1\n.end_amdhsa_kernel\n\n"
               ".amdgpu_metadata\n---\namdhsa.version:\n  - 1\n  - 1\namdhsa.kernels:\n"
               "  - .name: k\n    .symbol: k.kd\n    .kernarg_segment_size: 0\n"
               "    .kernarg_segment_align: 4\n    .group_segment_fixed_size: 0\n"
               "    .private_segment_fixed_size: 0\n    .wavefront_size: "
            << wave
            << "\n    .sgpr_count: 1\n    .vgpr_count: 1\n    .max_flat_workgroup_size: 256\n"
               "...\n.end_amdgpu_metadata\n";
        return source.str();
    }

    // ENABLE_WAVEFRONT_SIZE32 is bit 10 of kernel_code_properties (byte 56).
    std::uint32_t descriptorWavefront(const std::vector<char>& elf)
    {
        require(elf.size() >= 64, "The code object is too small to hold a kernel descriptor");
        const auto* bytes = reinterpret_cast<const unsigned char*>(elf.data());
        require(std::memcmp(bytes, "\x7f" "ELF", 4) == 0, "The code object is not an ELF file");
        const auto shoff     = le64(bytes + 40);
        const auto shentsize = le16(bytes + 58);
        const auto shnum     = le16(bytes + 60);
        const auto shstrndx  = le16(bytes + 62);
        auto       at       = [&](std::uint64_t off, std::uint64_t size) {
            require(off + size <= elf.size(), "An ELF header is outside the code object");
            return bytes + off;
        };
        const auto shstr    = at(shoff + static_cast<std::uint64_t>(shstrndx) * shentsize, shentsize);
        const auto strOff   = le64(shstr + 24);
        const auto strSize  = le64(shstr + 32);
        const auto strings  = at(strOff, strSize);
        for(std::uint16_t i = 0; i < shnum; ++i)
        {
            const auto header  = at(shoff + static_cast<std::uint64_t>(i) * shentsize, shentsize);
            const auto nameOff = le32(header);
            require(nameOff < strSize, "A section name is outside the string table");
            const auto* name = reinterpret_cast<const char*>(strings + nameOff);
            const auto  end  = static_cast<std::uint64_t>(strSize - nameOff);
            require(std::memchr(name, '\0', end) != nullptr, "A section name is not terminated");
            if(std::strcmp(name, ".rodata") != 0)
                continue;
            const auto off  = le64(header + 24);
            const auto size = le64(header + 32);
            require(size >= 64, "The code object has no kernel descriptor");
            const auto properties = le16(at(off + 56, 2));
            constexpr std::uint16_t wave32 = 1u << 10;
            return (properties & wave32) != 0 ? 32u : 64u;
        }
        require(false, "The code object has no .rodata");
        return 0;
    }

    // gfx1030 has both wavefront sizes, so the flag the builder passes to comgr
    // is visible in the descriptor. A gfx9 processor stays at 64 either way.
    void wavefrontSizeMatchesRequest()
    {
        co::Options options;
        options.codeObjectVersion = 4;
        for(const int wave : {64, 32})
        {
            const auto target = co::Target::fromTargetId("gfx1030", wave);
            const auto result = co::assemble({{"k.s", waveKernel(wave)}}, target, options);
            require(result.ok(),
                    "Assembling wavefront " + std::to_string(wave) + " failed: " + result.log);
            const auto metadata = co::readMetadata(result.bytes);
            require(metadata.ok(), "Cannot read the assembled code object: " + metadata.log);
            require(metadata.metadata.wavefrontSizes.size() == 1
                        && metadata.metadata.wavefrontSizes[0]
                               == static_cast<std::uint32_t>(wave),
                    "Metadata wavefront size for request " + std::to_string(wave) + " is "
                        + (metadata.metadata.wavefrontSizes.empty()
                               ? std::string("missing")
                               : std::to_string(metadata.metadata.wavefrontSizes[0])));
            require(descriptorWavefront(result.bytes) == static_cast<std::uint32_t>(wave),
                    "The kernel descriptor wavefront size is not " + std::to_string(wave));
        }
        std::cout << "PASS wavefrontSizeMatchesRequest\n";
    }

    // An empty source name is rejected before comgr runs.
    void emptySourceNameIsInvalidArgument()
    {
        const auto target = co::Target::fromTargetId("gfx1030");
        const auto result = co::assemble({{"", "s_endpgm\n"}}, target, {});
        require(result.status == co::Status::InvalidArgument,
                "An empty source name returned " + std::string(co::toString(result.status))
                    + ", not an invalid argument: " + result.log);
        require(result.status != co::Status::ComgrError,
                "An empty source name was reported as a comgr error");
        std::cout << "PASS emptySourceNameIsInvalidArgument\n";
    }

    const char* const kernelName = "hipblaslt_jit_builder_test_scale";

    // The string value of key in the manifest's object named owner.
    std::string field(const std::string& manifest, const std::string& owner, const std::string& key)
    {
        std::smatch      match;
        const std::regex pattern("\"" + owner + "\": *\\{[^}]*\"" + key + "\": *\"([^\"]*)\"");
        require(std::regex_search(manifest, match, pattern), "The manifest has no " + owner + "." + key);
        return match[1].str();
    }

    // The strings of the manifest's array named key.
    std::vector<std::string> strings(const std::string& manifest, const std::string& key)
    {
        std::smatch match;
        require(std::regex_search(manifest, match, std::regex("\"" + key + "\": *\\[([^\\]]*)\\]")),
                "The manifest has no " + key);
        const auto               list = match[1].str();
        std::vector<std::string> result;
        const std::regex         item("\"([^\"]*)\"");
        for(auto at = std::sregex_iterator(list.begin(), list.end(), item);
            at != std::sregex_iterator();
            ++at)
            result.push_back((*at)[1].str());
        require(!result.empty(), "The manifest's " + key + " is empty");
        return result;
    }

    // Builds solution for targetId and checks that the code object defines its
    // kernels and kernelName.
    void build(hj::GeneratedSolution& solution,
               const std::string&     targetId,
               const fs::path&        scratch,
               hj::BuiltSolution&     built)
    {
        const hj::BuildRequest request{targetId, hj::jitCodeObjectVersion, scratch};
        const auto             status = hj::makeComgrBuilder()->build(solution, request, built);
        require(status.ok(), "The build for " + targetId + " failed: " + status.message);

        const auto metadata
            = hj::code_object::readMetadata(built.object.bytes.data(), built.object.bytes.size());
        require(metadata.ok(), "Cannot read the built code object: " + metadata.log);
        const auto& kernels  = metadata.metadata.kernelNames;
        auto        expected = solution.kernelNames;
        expected.push_back(kernelName);
        for(const auto& name : expected)
            require(std::find(kernels.begin(), kernels.end(), name) != kernels.end(),
                    "The code object does not define " + name);
        require(metadata.metadata.codeObjectVersion == hj::jitCodeObjectVersion,
                "The code object is not version " + std::to_string(hj::jitCodeObjectVersion));
    }

    void buildKernel(const std::string&   targetId,
                     const hj::BuildUnit& kernel,
                     const fs::path&      scratch)
    {
        hj::GeneratedSolution solution;
        solution.kernelNames = {kernelName};
        solution.units       = {kernel};
        hj::BuiltSolution built;
        build(solution, targetId, scratch, built);
        std::cout << "PASS comgr built the HIP kernel for " << targetId << '\n';
    }

    void buildBundle(const fs::path& bundle, const hj::BuildUnit& kernel, const fs::path& scratch)
    {
        const auto bytes    = hj::source_bundle::readArtifact(bundle / "manifest.json");
        const auto manifest = std::string(bytes.begin(), bytes.end());
        auto       sources  = hj::source_bundle::readSourceBundle(bundle);
        require(sources.hip.empty(), "Expected only assembly in " + bundle.u8string());

        hj::GeneratedSolution solution;
        solution.entry       = std::move(sources.library);
        solution.kernelNames = strings(manifest, "main_kernels");
        for(auto& file : sources.assembly)
            solution.units.push_back(
                {std::move(file.name), std::move(file.bytes), hj::BuildUnit::Kind::Assembly, {}});
        solution.units.push_back(kernel);

        const auto        targetId = field(manifest, "architecture", "compiler_target");
        hj::BuiltSolution built;
        build(solution, targetId, scratch, built);
        std::cout << "PASS comgr built " << bundle.filename().u8string() << ", "
                  << solution.kernelNames.size() << " main kernel(s) and the HIP kernel, for "
                  << targetId << '\n';

        solution.kernelNames.push_back("hipblaslt_jit_builder_test_missing");
        const hj::BuildRequest request{targetId, hj::jitCodeObjectVersion, scratch};
        const auto             missing = hj::makeComgrBuilder()->build(solution, request, built);
        require(!missing.ok() && missing.stage == hj::Stage::Build
                    && missing.message.find("hipblaslt_jit_builder_test_missing")
                           != std::string::npos,
                "The build accepted a kernel name the code object does not define");
    }
}

int main(int argc, char** argv)
try
{
    require(argc == 4, "Usage: hipblaslt-jit-builder-test BUNDLES KERNEL SCRATCH");
    wavefrontSizeMatchesRequest();
    emptySourceNameIsInvalidArgument();
    const auto bundles = fs::u8path(argv[1]);
    const auto source  = hipblaslt_jit_test::readFile(fs::u8path(argv[2]));
    const auto scratch = fs::u8path(argv[3]);
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    const hj::BuildUnit kernel{"builder_test_kernel.hip",
                               std::vector<uint8_t>(source.begin(), source.end()),
                               hj::BuildUnit::Kind::Hip,
                               {}};
    int built = 0;
    for(const auto& target : fs::directory_iterator(bundles))
    {
        buildKernel(target.path().filename().u8string(), kernel, scratch);
        for(const auto& bundle : fs::directory_iterator(target))
        {
            buildBundle(bundle.path(), kernel, scratch);
            ++built;
        }
    }
    require(built > 0, "No bundle in " + bundles.u8string());
    std::cout << "PASS the build of every bundle fails when a kernel name is not defined\n";
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
