// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "compilation/ICompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "compilation/IRunnableKernel.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The ordinal a launch on @p stream will run on.
///
/// Deliberately the same rule HandleDeviceResolver::deviceId applies at plan build: a
/// concrete stream names its own device, and anything else -- a default token, a query
/// that fails, a runtime that reports hipSuccess without writing the out-parameter --
/// falls through to the current device. Resolving a launch by a stricter rule than the
/// one that built the plan would let a plan build and then refuse every execute.
///
/// Seeded to -1 and negatives rejected: at 0 an unwritten out-parameter reads as device
/// zero and goes unseen on a single-device host.
inline int launchDeviceOrdinal(hipStream_t stream)
{
    int deviceOrdinal = -1;
    if(!hipdnn_plugin_sdk::isDefaultStream(stream)
       && hipdnn_plugin_sdk::getDeviceFromStream(stream, &deviceOrdinal) == hipSuccess
       && deviceOrdinal >= 0)
    {
        return deviceOrdinal;
    }

    deviceOrdinal = -1;
    const hipError_t status = hipGetDevice(&deviceOrdinal);
    if(status != hipSuccess || deviceOrdinal < 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "cannot resolve the device to launch on from the handle's stream: "
                + std::string(hipGetErrorString(status)) + ", ordinal "
                + std::to_string(deviceOrdinal));
    }
    return deviceOrdinal;
}

/// The kpack module caches each pack's dispatch handler loads through,
/// process-lifetime. Declared here rather than in IngestorPacks.hpp because they belong
/// to the kernel-code path, and exposed at all so a test can assert that two dispatches
/// over one (archive, toc_key, arch) produced a single hipModule_t -- otherwise
/// unobservable. Each is defined beside the handler it serves.
///
/// One cache per pack, not one shared: a key is (archive, toc_key, arch), so two packs
/// reading one archive would answer each other's lookups.
compilation::KpackModuleCache& pointwiseKpackModuleCache();
compilation::KpackModuleCache& convFwdKpackModuleCache();
compilation::KpackModuleCache& gfx942AttentionDenseKpackModuleCache();

/// What a kpack kernel needs to be loaded again for another device: the archive it was
/// resolved to, the entry inside it, and the declared digest the loader verifies. Held
/// because `buildIngestorKernelCode` derives them from a descriptor that a prepared
/// dispatch does not keep.
///
/// `strippedArch` is the feature-stripped architecture the plan was matched against,
/// which is what a second device must agree with. `label` and `symbol` are carried so a
/// later failure names the descriptor as precisely as the first load would have.
struct KpackSource
{
    std::filesystem::path archive;
    std::string tocKey;
    std::string symbol;
    std::string sha256;
    std::string label;
    std::string strippedArch;
};

/// The program plus the kernel resolved out of it, answered per device.
///
/// A hipModule_t belongs to the device it was loaded on, and a hipFunction_t is a
/// non-owning view into one, so a dispatch running on a second device needs that
/// device's own module. Which device that is cannot be known before execute(): a plan is
/// immutable and re-usable across handles and threads, and a handle names its device
/// only through its stream, which a default token leaves to whatever is current.
///
/// KPACK kernels therefore retain what a second resolution needs and memoise per
/// ordinal. Every other source kind compiles against an architecture rather than a
/// device and answers with the one kernel it was built with.
///
/// Movable rather than copyable: built by value and stored in a PreparedDispatch. The
/// memo lives behind a pointer so moving one does not move a locked mutex.
class IngestorKernelCode
{
public:
    /// Source kinds whose program is not tied to one device.
    IngestorKernelCode(std::unique_ptr<compilation::ICompiledProgram> program,
                       std::unique_ptr<compilation::IRunnableKernel> kernel)
        : _memo(std::make_unique<Memo>())
    {
        _memo->byOrdinal.emplace(ANY_DEVICE, Resolved{std::move(program), std::move(kernel)});
    }

    /// KPACK: the first device's result, plus the coordinates and the architecture that
    /// let a second device be answered without returning to the descriptor.
    IngestorKernelCode(const compilation::KpackKernelLoader& loader,
                       KpackSource source,
                       int deviceOrdinal,
                       std::unique_ptr<compilation::ICompiledProgram> program,
                       std::unique_ptr<compilation::IRunnableKernel> kernel)
        : _memo(std::make_unique<Memo>())
        , _loader(&loader)
        , _source(std::move(source))
    {
        requireRealOrdinal(deviceOrdinal);

        // Stripped here rather than trusted from the caller, so the field is what its
        // name says whichever constructor ran. Stripping twice is idempotent.
        _source->strippedArch
            = std::string(hipdnn_plugin_sdk::stripArchFeatures(_source->strippedArch));
        _memo->byOrdinal.emplace(deviceOrdinal, Resolved{std::move(program), std::move(kernel)});
    }

    /// Recorded as well as applied: every kernel already resolved is updated, and a
    /// kernel created later for another device is configured from the same record rather
    /// than launching with the 1x1x1 default. Recording means the result does not depend
    /// on geometry being set before any second device is resolved.
    void setBlockSize(unsigned int x, unsigned int y = 1, unsigned int z = 1)
    {
        _geometry.blockX = x;
        _geometry.blockY = y;
        _geometry.blockZ = z;
        applyGeometryToResolved();
    }

    void setGridSize(unsigned int x, unsigned int y = 1, unsigned int z = 1)
    {
        _geometry.gridX = x;
        _geometry.gridY = y;
        _geometry.gridZ = z;
        applyGeometryToResolved();
    }

    void setSharedMemBytes(unsigned int bytes)
    {
        _geometry.sharedMemBytes = bytes;
        applyGeometryToResolved();
    }

    /// The kernel a dispatch on @p stream must launch. The entry point a pack calls.
    ///
    /// The device is resolved only when this code is bound to one. A program that runs
    /// anywhere -- anything but KPACK -- is answered without a HIP query, so that path
    /// keeps costing what it did and keeps being unable to fail here.
    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        {
            const std::lock_guard<std::mutex> lock(_memo->mutex);
            const auto any = _memo->byOrdinal.find(ANY_DEVICE);
            if(any != _memo->byOrdinal.end())
            {
                return *any->second.kernel;
            }
        }
        return kernelFor(resolveLaunchOrdinal(stream));
    }

    /// The kernel to launch on @p deviceOrdinal.
    ///
    /// An ordinal already resolved is answered from the memo without a HIP query or a
    /// module load. A new one is admitted only if its feature-stripped architecture is
    /// the one the plan was matched against: the ordinal identifies a slot, the
    /// architecture is what makes the selected kernel the right one.
    ///
    /// @throws HipdnnPluginException when the architecture differs, or when a kpack
    ///         kernel is asked for a device and no coordinates were retained.
    compilation::IRunnableKernel& kernelFor(int deviceOrdinal) const;

    virtual ~IngestorKernelCode() = default;

    IngestorKernelCode(const IngestorKernelCode&) = delete;
    IngestorKernelCode& operator=(const IngestorKernelCode&) = delete;

    // Public because moving is how buildIngestorKernelCode's result reaches a
    // PreparedDispatch, which stores it by value.
    //
    // That by-value store means a subclass moved through this type slices, and deleting
    // the move on the subclass does NOT prevent it: a Derived&& binds to this base
    // overload directly. Nothing production subclasses this, and the by-value parameter
    // is only ever handed the result of buildIngestorKernelCode, so the hole is latent.
    // Closing it would take private inheritance on the subclass, which is the test
    // fixture's business rather than this type's.
    IngestorKernelCode(IngestorKernelCode&&) = default;
    IngestorKernelCode& operator=(IngestorKernelCode&&) = default;

protected:
    /// One device's program and the kernel viewing into it. The program is held because
    /// a hipFunction_t does not own the module it came from.
    struct Resolved
    {
        std::unique_ptr<compilation::ICompiledProgram> program;
        std::unique_ptr<compilation::IRunnableKernel> kernel;
    };

    /// For a subclass that supplies its own resolution and therefore needs no loader or
    /// archive coordinates -- only the architecture the plan was matched against, which
    /// is what the admission rule compares.
    IngestorKernelCode(std::unique_ptr<compilation::ICompiledProgram> program,
                       std::unique_ptr<compilation::IRunnableKernel> kernel,
                       int deviceOrdinal,
                       const std::string& strippedArch)
        : _memo(std::make_unique<Memo>())
    {
        requireRealOrdinal(deviceOrdinal);
        _source = KpackSource{};
        _source->strippedArch = std::string(hipdnn_plugin_sdk::stripArchFeatures(strippedArch));
        _memo->byOrdinal.emplace(deviceOrdinal, Resolved{std::move(program), std::move(kernel)});
    }

    /// The two steps a device this object has not seen before requires, as overridable
    /// units so a subclass can stand in for hardware.
    ///
    /// Neither is a hook for production: nothing overrides them outside tests, and a
    /// default build resolves both against HIP and the archive. They exist because the
    /// behaviour above them -- which ordinal is admitted, which is refused, and which is
    /// answered without touching either step -- is otherwise reachable only on a host
    /// holding two architectures at once, and no such host exists.

    /// Which ordinal @p stream launches on. Overridable alongside the two below so a
    /// test can prove this is not consulted for a program that runs anywhere.
    virtual int resolveLaunchOrdinal(hipStream_t stream) const
    {
        return launchDeviceOrdinal(stream);
    }

    /// The architecture @p deviceOrdinal reports, undecorated as HIP gives it.
    virtual hipError_t queryDeviceArch(int deviceOrdinal, std::string& reportedArch) const
    {
        hipDeviceProp_t properties{};
        const hipError_t status = hipGetDeviceProperties(&properties, deviceOrdinal);
        if(status == hipSuccess)
        {
            reportedArch = properties.gcnArchName;
        }
        return status;
    }

    /// Loads @p deviceOrdinal's own module and resolves the symbol out of it. Reached
    /// only once the architecture has been accepted.
    virtual Resolved resolveForDevice(int deviceOrdinal, const std::string& reportedArch) const
    {
        if(_loader == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "kpack kernel source for " + _source->label + ": no loader was retained, so device "
                    + std::to_string(deviceOrdinal) + " cannot be resolved");
        }

        // The raw name: the cache strips it for the key, and the loader's own arch check
        // reads the archive's decoration.
        auto program = _loader->load(_source->archive,
                                     _source->tocKey,
                                     reportedArch,
                                     deviceOrdinal,
                                     _source->symbol,
                                     _source->sha256,
                                     _source->label);
        auto kernel = program->getKernel(_source->symbol);
        return Resolved{std::move(program), std::move(kernel)};
    }

    /// A device-bound program must name a real device. ANY_DEVICE is what says "runs
    /// anywhere", so filing one under it would answer every later ordinal without ever
    /// reaching the architecture gate.
    static void requireRealOrdinal(int deviceOrdinal)
    {
        if(deviceOrdinal < 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "a device-bound kpack kernel was given device ordinal "
                    + std::to_string(deviceOrdinal) + ", which names no device");
        }
    }

    /// True when the architecture this plan was matched against was retained.
    bool hasKpackCoordinates() const
    {
        // The loader is not part of the test: only the default resolveForDevice() needs
        // one, and production sets it whenever it sets the source.
        return _source.has_value();
    }

private:
    /// Key for a program every device can run. Deliberately not -1: that is both
    /// ingestor::NO_DEVICE and Kernel::NO_DEVICE, and a -1 reaching the device-bound
    /// constructor would file a device-specific program under the runs-anywhere key and
    /// silently bypass the architecture gate for every later ordinal.
    static constexpr int ANY_DEVICE = std::numeric_limits<int>::min();

    struct Geometry
    {
        unsigned int blockX = 1;
        unsigned int blockY = 1;
        unsigned int blockZ = 1;
        unsigned int gridX = 1;
        unsigned int gridY = 1;
        unsigned int gridZ = 1;
        unsigned int sharedMemBytes = 0;
    };

    struct Memo
    {
        std::mutex mutex;
        std::unordered_map<int, Resolved> byOrdinal;
    };

    void applyGeometry(compilation::IRunnableKernel& kernel) const
    {
        kernel.setBlockSize(_geometry.blockX, _geometry.blockY, _geometry.blockZ);
        kernel.setGridSize(_geometry.gridX, _geometry.gridY, _geometry.gridZ);
        kernel.setSharedMemBytes(_geometry.sharedMemBytes);
    }

    /// Under the lock: a second device can be resolving while this runs.
    void applyGeometryToResolved() const
    {
        const std::lock_guard<std::mutex> lock(_memo->mutex);
        for(auto& entry : _memo->byOrdinal)
        {
            applyGeometry(*entry.second.kernel);
        }
    }

    std::unique_ptr<Memo> _memo;
    const compilation::KpackKernelLoader* _loader = nullptr;
    std::optional<KpackSource> _source;
    Geometry _geometry;
};

inline compilation::IRunnableKernel& IngestorKernelCode::kernelFor(int deviceOrdinal) const
{
    const std::lock_guard<std::mutex> lock(_memo->mutex);

    // A program that is not tied to a device answers for every ordinal, including one
    // that could not be resolved at all.
    const auto any = _memo->byOrdinal.find(ANY_DEVICE);
    if(any != _memo->byOrdinal.end())
    {
        return *any->second.kernel;
    }

    const auto seen = _memo->byOrdinal.find(deviceOrdinal);
    if(seen != _memo->byOrdinal.end())
    {
        return *seen->second.kernel;
    }

    if(!hasKpackCoordinates())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "kpack kernel is bound to a device but retained no archive coordinates, so it "
            "cannot be resolved for device "
                + std::to_string(deviceOrdinal));
    }

    // Queried rather than taken from the plan: the plan's properties describe the device
    // it was built under, which is the one device this branch has already ruled out.
    std::string reportedArch;
    const hipError_t status = queryDeviceArch(deviceOrdinal, reportedArch);
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "kpack kernel source for " + _source->label + ": cannot query device "
                + std::to_string(deviceOrdinal) + " to launch symbol '" + _source->symbol
                + "': " + hipGetErrorString(status));
    }

    const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(reportedArch));

    if(arch != _source->strippedArch)
    {
        // The selected kernel is the one the catalog matched for an architecture, so a
        // device of another architecture is not a slot this plan can be moved to. Reported
        // rather than re-matched: choosing a different kernel here would substitute one
        // the caller never asked for.
        //
        // INVALID_VALUE, not INTERNAL_ERROR: nothing inside the provider is inconsistent.
        // The caller paired a plan with a handle on an architecture it was never built
        // for, and that pairing is the caller's to correct.
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "kpack kernel source for " + _source->label + ": symbol '" + _source->symbol
                + "' was matched for architecture '" + _source->strippedArch
                + "' but is being launched on device " + std::to_string(deviceOrdinal)
                + ", which reports '" + arch
                + "'; a plan is being executed under a handle from a device of another "
                  "architecture");
    }

    Resolved resolved = resolveForDevice(deviceOrdinal, reportedArch);
    applyGeometry(*resolved.kernel);

    auto& stored = _memo->byOrdinal.emplace(deviceOrdinal, std::move(resolved)).first->second;
    return *stored.kernel;
}

/// Fails unless the argument list a descriptor records matches the one its pack marshals.
///
/// A prebuilt archive is compiled out of band from the pack that launches it, and
/// `hipModuleGetFunction` confirms only that a symbol of that name exists, so a drifted
/// signature is otherwise undefined behaviour at launch rather than a diagnostic. Embedded
/// source needs none -- HIPRTC compiles it against the declaration the host marshals.
///
/// `kind` and `size` always. `name` only where both sides carry one: it is the one thing
/// that catches an operand permutation, but clang omits it for HIP `extern "C" __global__`
/// kernels, and requiring it would make every such kernel undispatchable. `offset` is
/// printed and never compared -- the kernarg layout is the driver's, not the pack's to
/// assert.
///
/// Throwing rather than warning because every way this fires is a static disagreement
/// between two authored artifacts: no configuration reaches it with a correct launch.
inline void
    requireSignatureMatch(const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& recorded,
                          const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& expected,
                          const std::string& symbol,
                          const std::string& label)
{
    const auto agrees = [](const hipdnn_plugin_sdk::ingestor::KernelArgument& lhs,
                           const hipdnn_plugin_sdk::ingestor::KernelArgument& rhs) {
        if(lhs.kind != rhs.kind || lhs.size != rhs.size)
        {
            return false;
        }
        return lhs.name.empty() || rhs.name.empty() || lhs.name == rhs.name;
    };

    if(recorded.size() != expected.size()
       || !std::equal(recorded.begin(), recorded.end(), expected.begin(), agrees))
    {
        // Both sides printed: a mismatch diagnostic naming only one of them sends the
        // reader to the archive by hand to find out what the other was.
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "kpack kernel source for " + label + ": symbol '" + symbol
                + "' is packaged with arguments "
                + hipdnn_plugin_sdk::ingestor::describeKernelSignature(recorded)
                + ", but this pack launches it with "
                + hipdnn_plugin_sdk::ingestor::describeKernelSignature(expected));
    }
}

/// The single place a KernelSource's `kind` decides where the code object comes from.
///
/// @param compiler   Used only on the EMBEDDED_SOURCE path.
/// @param kpackLoader Used only on the KPACK path.
/// @param options    HIPRTC build options. A kpack blob's build defines were baked at
///                   pack time, so there is nothing left for them to affect. Ignoring
///                   them on the KPACK path is the correct behaviour, not an oversight.
/// @param expectedSignature The list the calling pack declares beside its own launch.
///                   Used only on the KPACK path, and deliberately without a default:
///                   a pack that omits it should not compile into one that silently
///                   skips the check.
inline IngestorKernelCode buildIngestorKernelCode(
    const compilation::IKernelCompiler& compiler,
    const compilation::KpackKernelLoader& kpackLoader,
    const hipdnn_plugin_sdk::ingestor::MatchContext& context,
    const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
    const compilation::KernelCompileOptions& options,
    const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& expectedSignature)
{
    using hipdnn_plugin_sdk::ingestor::KernelSourceKind;

    switch(kernel.source.kind)
    {
    case KernelSourceKind::EMBEDDED_SOURCE:
    {
        auto program = compiler.compile(kernel.source.sourceFile, options);
        auto runnableKernel = program->getKernel(kernel.source.entryPoint);
        return IngestorKernelCode{std::move(program), std::move(runnableKernel)};
    }
    case KernelSourceKind::KPACK:
    {
        // `library` is authored relative to the descriptor that declared it;
        // originDirectory is the loader-supplied anchor that makes it nameable.
        // weakly_canonical because the target need not exist -- when it does not, the
        // archive-open failure below is the diagnostic, not a filesystem exception.
        std::error_code ignored;
        const std::filesystem::path origin
            = std::filesystem::weakly_canonical(kernel.originDirectory, ignored);
        const std::filesystem::path resolved
            = std::filesystem::weakly_canonical(origin / kernel.source.library, ignored);

        const std::string label = hipdnn_plugin_sdk::ingestor::describeDescriptor(
            "kernel", kernel.name, kernel.kernelId);

        // A descriptor names an archive shipped inside the tree it was loaded from, never
        // one elsewhere on the filesystem. weakly_canonical normalises `..` and absolute
        // paths rather than rejecting them, so without this a descriptor could name any
        // readable file and have it loaded as executable code. Compare canonical forms:
        // the lexical check alone would miss a symlink out of the tree.
        //
        // The boundary is the TREE, not the descriptor's own directory. One archive ships
        // per arch shard, at the shard root, so a descriptor authored in a child folder --
        // which is every production layout, since packing preserves the authored subpath --
        // has to climb out of its own directory to reach it.
        //
        // treeRoot rather than a derived arch-shard root: it is what the loader actually
        // walked, so it needs no filesystem probing and assumes nothing about how deep a
        // shard sits under it. A kernel built in memory carries neither path and is not
        // reachable here -- KPACK requires a file -- but an empty treeRoot narrows the
        // boundary to the descriptor's own directory rather than opening a hole, so fall
        // back to origin.
        const std::filesystem::path boundary
            = kernel.treeRoot.empty() ? origin
                                      : std::filesystem::weakly_canonical(kernel.treeRoot, ignored);
        const std::string relative = resolved.lexically_relative(boundary).generic_string();
        if(resolved != boundary && (relative.empty() || relative.rfind("..", 0) == 0))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "kpack kernel source for " + label + ": library '" + kernel.source.library
                    + "' resolves to '" + resolved.string()
                    + "', which is outside the descriptor tree '" + boundary.string() + "'");
        }

        // Reject a `library` that reaches its archive through a link -- a POSIX symlink or
        // a Windows junction -- inside the tree.
        //
        // Walked over the UN-canonicalised join: weakly_canonical has already resolved the
        // symlinks in whatever prefix exists, so testing `resolved` would find none. Scoped
        // strictly below `boundary` because a tree under a symlinked prefix is ordinary.
        //
        // Each component must BE an ordinary file or directory rather than merely not be one
        // named kind. MSVC reports a junction as file_type::junction, so is_symlink answers
        // false for one -- and mklink /J needs no privilege, unlike a symlink.
        //
        // This refuses a path that IS a link at validation time and does not close the
        // time-of-check/time-of-use race: kpack_open is path-only, with no fd or handle
        // overload anywhere in the kpack C API.
        std::filesystem::path prefix;
        for(const auto& component : origin / kernel.source.library)
        {
            prefix /= component;
            if(component == "." || component == "..")
            {
                continue;
            }
            const std::string below = prefix.lexically_relative(boundary).generic_string();
            if(below.empty() || below == "." || below.rfind("..", 0) == 0)
            {
                continue;
            }
            // symlink_status, not status: the latter follows the link and reports the
            // target's kind, which is exactly the answer that must not be trusted here.
            const std::filesystem::file_type kind
                = std::filesystem::symlink_status(prefix, ignored).type();

            // not_found is allowed through so the archive's own absence is reported by the
            // loader, which names it, rather than as a link that is not there either.
            if(kind != std::filesystem::file_type::regular
               && kind != std::filesystem::file_type::directory
               && kind != std::filesystem::file_type::not_found)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                    "kpack kernel source for " + label + ": library '" + kernel.source.library
                        + "' reaches its archive through the link '" + prefix.string()
                        + "' inside the descriptor tree '" + boundary.string() + "'");
            }
        }

        // Ahead of the load: the descriptor and the pack already disagree, and opening an
        // archive to confirm it would only delay the same diagnostic.
        requireSignatureMatch(
            kernel.source.signature, expectedSignature, kernel.source.symbol, label);

        auto program = kpackLoader.load(resolved,
                                        kernel.source.tocKey,
                                        context.deviceProperties.gcnArchName,
                                        context.deviceId,
                                        kernel.source.symbol,
                                        kernel.source.sha256,
                                        label);
        auto runnableKernel = program->getKernel(kernel.source.symbol);
        return IngestorKernelCode{kpackLoader,
                                  KpackSource{resolved,
                                              kernel.source.tocKey,
                                              kernel.source.symbol,
                                              kernel.source.sha256,
                                              label,
                                              std::string(hipdnn_plugin_sdk::stripArchFeatures(
                                                  context.deviceProperties.gcnArchName))},
                                  context.deviceId,
                                  std::move(program),
                                  std::move(runnableKernel)};
    }
    case KernelSourceKind::HSACO_FILE:
    case KernelSourceKind::ROCKE_BUILDER:
    // A kind added after this adapter was written lands here too, and gets the same
    // named diagnostic rather than falling off the end of the function.
    default:
        break;
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
        "no kernel source adapter for "
            + hipdnn_plugin_sdk::ingestor::describeDescriptor(
                "kernel", kernel.name, kernel.kernelId)
            + ": its source kind is not one this provider can load");
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
