// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "SynchronizerValidator.hpp"
#include "TimingInstrumentation.hpp"

#include "ResultReporter.hpp"

#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace TensileLite
{
    namespace Client
    {
        namespace
        {
            constexpr unsigned orReduceBlockSize = 256;
            // Enough blocks to stream the client's 1.6 MB buffer at full
            // bandwidth; larger buffers loop in the grid stride.
            constexpr size_t orReduceMaxBlocks = 1024;

            // ORs every 32-bit word of the buffer into *any. Only threads that
            // saw residue touch *any, so a clean buffer costs no atomics.
            __global__ void orReduceSynchronizer(uint32_t const* buffer, size_t ints, uint32_t* any)
            {
                uint32_t     v      = 0;
                size_t const stride = size_t(gridDim.x) * blockDim.x;
                for(size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < ints; i += stride)
                    v |= buffer[i];
                if(v != 0)
                    atomicOr(any, v);
            }
        }

        bool scanSynchronizerResidue(uint8_t const* host, size_t bytes, SynchronizerResidue& out)
        {
            // The buffer is declared with the alpha type but every consumer
            // writes 32-bit counters, so an int offset names the counter left
            // set. Callers guarantee an int-sized-or-wider alpha, so the byte
            // count always divides evenly.
            assert(bytes % sizeof(uint32_t) == 0);
            size_t const    ints = bytes / sizeof(uint32_t);
            uint32_t const* v    = reinterpret_cast<uint32_t const*>(host);

            // Clean is the common case and only needs "anything nonzero?": a
            // branch-free OR over the buffer, which vectorizes. The count and
            // first offset are only needed for the dirty-buffer diagnostic.
            uint32_t any = 0;
            for(size_t i = 0; i < ints; i++)
                any |= v[i];

            if(any == 0)
                return false;

            size_t nonzero = 0;
            size_t first   = ints;
            for(size_t i = 0; i < ints; i++)
            {
                if(v[i] != 0)
                {
                    if(nonzero == 0)
                        first = i;
                    nonzero++;
                }
            }

            out.totalInts   = ints;
            out.nonzeroInts = nonzero;
            out.firstInt    = first;

            return true;
        }

        SynchronizerValidator::SynchronizerValidator(po::variables_map const& args)
            : m_enabled(args["check-synchronizer"].as<bool>())
            , m_validateAfterEnqueues(args.count("validate-after-enqueues")
                                      && args["validate-after-enqueues"].as<bool>())
        {
            // Under the FFM emulator the reduction kernel's cost is unknown;
            // keep the host reduction there.
            char const* memfile = std::getenv("HSA_MODEL_MEMFILE");
            m_hostReduction     = memfile != nullptr && memfile[0] != '\0';
        }

        void SynchronizerValidator::preProblem(ContractionProblem* const problem)
        {
            m_problem = problem;
        }

        void SynchronizerValidator::preSolution(ContractionSolution* const solution)
        {
            m_failedInSolution = false;
            // Covers the possible uses of the pointer: Flags in singleCallArgs
            // for StreamK, the dstD/Synchronizer block for MBSK, AmaxSync for
            // amaxD, and the AddressFlags/AddressSynchronizer/AmaxSync args of
            // handwritten custom kernels (custom StreamK workspace types are
            // scanned as well). A custom Synchronizer arg gets the buffer only
            // when the resolved accumulation is MBSK; getAccumulation() returns
            // either the static globalAccumulation or MultipleBuffer, so the
            // MBSK check covers it.
            // AdaptiveGemmGSUA kernels pick MBSK at run time by checking the
            // pointer for null, and the grouped user-args launch always passes
            // it, so they are scanned regardless of the static mode.
            // Generated kernels never take the custom-arg dispatch, so their
            // declared args are not consulted.
            // Parallel Stream-K reduction passes Flags=nullptr, but excluding it
            // requires the problem- and hardware-dependent reduction decision.
            // We conservatively scan its buffer too; it normally remains zero.
            //
            // An unknown solution is scanned rather than skipped.
            if(solution == nullptr)
            {
                m_mayUseSynchronizer = true;
                return;
            }

            auto const& sm               = solution->sizeMapping;
            auto const& ck               = solution->customKernel;
            bool const  partialReduction = sm.requiresPartialReduction();
            bool const  customStreamK    = ck.workspaceType == CustomWorkspaceType::StreamK
                                        || ck.workspaceType == CustomWorkspaceType::StreamKWithReduction;
            bool const  mbsk             = sm.globalAccumulation == 3 || sm.adaptiveGemmGSUA == 1;
            bool const  handwritten      = !ck.name.empty() && !ck.generated;
            bool const  customArg
                = handwritten
                  && std::any_of(ck.args.begin(), ck.args.end(), [](CustomArgDefinition const& a) {
                         return a.semantic == CustomArgSemantic::AddressFlags
                                || a.semantic == CustomArgSemantic::AddressSynchronizer
                                || a.semantic == CustomArgSemantic::AmaxSync;
                     });
            m_mayUseSynchronizer = partialReduction || customStreamK || mbsk || customArg
                                   || solution->problemType.outputAmaxD
                                   || solution->problemType.fusedGemmA2A;
        }

        void SynchronizerValidator::postSolution()
        {
            // Guard 2: an exception ended the solution's last run iteration
            // before its check. A usable device is checked as normal; a lost
            // one fails the readback, which only fails the solution: nothing
            // may throw out of postSolution.
            auto guarded = [&](auto&& check) {
                try
                {
                    check();
                }
                catch(std::exception const& err)
                {
                    m_reporter->log(LogLevel::Error,
                                    concatenate("Synchronizer check failed: ", err.what(), "\n"));
                    m_failedInSolution = true;
                }
                catch(...)
                {
                    m_reporter->log(LogLevel::Error, "Synchronizer check failed.\n");
                    m_failedInSolution = true;
                }
            };
            guarded([&] { checkLaunches("interrupted run iteration", postSolutionTimers); });
            // The fused-A2A flag blocks are not tied to a launch: checked here regardless.
            guarded([&] {
                for(auto const& device : *m_devices)
                    if(active() && device->commFlags != nullptr)
                    {
                        ScopedTimer timer(flagBlockTimers.check);
                        if(!checkRegion(device->deviceId,
                                        device->commFlags,
                                        device->commFlagsBytes,
                                        concatenate("fused-A2A flag block left dirty (device ",
                                                    device->deviceId,
                                                    ")"),
                                        flagBlockTimers))
                            m_failedInSolution = true;
                    }
            });

            if(m_failedInSolution)
            {
                m_errorsReported++;
                // Overrides the reference verdict so the CSV and library logic
                // fed from it cannot crown a solution that corrupts the buffer
                // (also main's INVALID: the exception is still logged and counted).
                m_reporter->report(ResultKey::Validation, "FAILED");
            }

            m_failedInSolution = false;
            m_iterationInputs.clear();
        }

        void SynchronizerValidator::beginRunIteration()
        {
            m_iterationInputs.clear();
            for(auto const& device : *m_devices)
                m_iterationInputs.push_back({device->deviceId, device->inputArr});
        }

        void SynchronizerValidator::checkRunIteration()
        {
            checkLaunches("run iteration", runIterationTimers);
        }

        void SynchronizerValidator::validateEnqueues(std::vector<size_t> const& slots,
                                                     TimingEvents const&        startEvents,
                                                     TimingEvents const&        stopEvents)
        {
            // Opt-in check right after the group of enqueues, never between its launches;
            // the run iteration's check covers them in any case.
            if(!m_validateAfterEnqueues)
                return;
            m_launchedSinceCheck = false;
            checkAfterLaunch("enqueues", afterEnqueuesTimers);
        }

        void SynchronizerValidator::checkLaunches(char const* stage, CheckTimers const& timers)
        {
            // Guard 1: nothing launched, nothing to check (postSolution also
            // runs for rejected and never-run solutions).
            if(!m_launchedSinceCheck)
                return;
            m_launchedSinceCheck = false;
            checkAfterLaunch(stage, timers);
        }

        void SynchronizerValidator::checkAfterLaunch(char const* stage, CheckTimers const& timers)
        {
            try
            {
                ScopedTimer timer(timers.check);
                // A launch with no captured inputs means beginRunIteration() was not called.
                // Skipping the check there would turn the protection off silently.
                if(active() && m_problem != nullptr && m_iterationInputs.empty())
                    throw std::runtime_error(
                        "Synchronizer check: kernels launched but no run iteration inputs were "
                        "captured (beginRunIteration() was not called)");
                checkInputs(stage, timers);
            }
            catch(...)
            {
                // Preserve failure even when the buffer's contents could not be checked.
                m_failedInSolution = true;
                throw;
            }
        }

        void SynchronizerValidator::checkInputs(char const* stage, CheckTimers const& timers)
        {
            if(!active() || m_problem == nullptr)
            {
                // Solution cannot use the Synchronizer (or the check is off): nothing to read.
                ScopedTimer timer(timers.inactive);
                return;
            }

            // Rotating-buffer inputs usually share the Synchronizer, but they are
            // built once per problem, so under GuardPageAll they keep the first
            // iteration's address while the reset inputs move: read each buffer once.
            std::vector<void const*> checked;

            auto checkOnce = [&](int                           deviceId,
                                 ContractionProblemGemm const& problem,
                                 void*                         buffer,
                                 size_t                        gemmIdx) {
                if(std::find(checked.begin(), checked.end(), buffer) != checked.end())
                    return;
                checked.push_back(buffer);
                if(!checkBuffer(deviceId, problem, buffer, stage, gemmIdx, timers))
                    m_failedInSolution = true;
            };

            for(auto const& device : m_iterationInputs)
            {
                for(auto const& inputs : device.inputs)
                {
                    if(inputs == nullptr)
                        continue;

                    if(auto problems = dynamic_cast<ContractionProblemGroupedGemm*>(m_problem))
                    {
                        auto const& result = dynamic_cast<ContractionGroupedInputs const&>(*inputs);
                        for(size_t j = 0; j < problems->gemms.size(); j++)
                            checkOnce(device.deviceId,
                                      problems->gemms[j],
                                      result.grouped[j].Synchronizer,
                                      j);
                    }
                    else if(auto problem = dynamic_cast<ContractionProblemGemm*>(m_problem))
                    {
                        auto const& result = dynamic_cast<ContractionInputs const&>(*inputs);
                        checkOnce(device.deviceId, *problem, result.Synchronizer, 0);
                    }
                    else
                    {
                        throw std::runtime_error("Failed to cast to any ContractionProblem.");
                    }
                }
            }
        }

        SynchronizerValidator::~SynchronizerValidator()
        {
            if(m_staging != nullptr)
                static_cast<void>(hipHostFree(m_staging));
            if(m_hostAny != nullptr)
                static_cast<void>(hipHostFree(m_hostAny));
            for(auto const& deviceAny : m_deviceAny)
                static_cast<void>(hipFree(deviceAny.second));
        }

        hipStream_t SynchronizerValidator::streamOf(int deviceId) const
        {
            for(auto const& device : *m_devices)
                if(device->deviceId == deviceId)
                    return device->stream;
            return nullptr;
        }

        uint8_t* SynchronizerValidator::stagingBuffer(size_t bytes)
        {
            if(m_stagingBytes >= bytes)
                return m_staging;

            uint8_t* old   = m_staging;
            m_staging      = nullptr;
            m_stagingBytes = 0;
            if(old != nullptr)
                HIP_CHECK_EXC(hipHostFree(old));
            HIP_CHECK_EXC(hipHostMalloc(&m_staging, bytes));
            m_stagingBytes = bytes;

            return m_staging;
        }

        uint8_t* SynchronizerValidator::readBuffer(int deviceId, void* device, size_t bytes)
        {
            ScopedDevice guard(deviceId);
            auto         host   = stagingBuffer(bytes);
            auto         stream = streamOf(deviceId);
            HIP_CHECK_EXC(hipMemcpyAsync(host, device, bytes, hipMemcpyDeviceToHost, stream));
            HIP_CHECK_EXC(hipStreamSynchronize(stream));
            return host;
        }

        void SynchronizerValidator::clearBuffer(int deviceId, void* device, size_t bytes)
        {
            ScopedDevice guard(deviceId);
            auto         stream = streamOf(deviceId);
            HIP_CHECK_EXC(hipMemsetAsync(device, 0, bytes, stream));
            HIP_CHECK_EXC(hipStreamSynchronize(stream));
        }

        bool SynchronizerValidator::reduceBuffer(int         deviceId,
                                                 void const* device,
                                                 size_t      bytes,
                                                 uint32_t&   any)
        {
            ScopedDevice guard(deviceId);
            auto         stream = streamOf(deviceId);

            auto deviceAny = m_deviceAny.find(deviceId);
            if(deviceAny == m_deviceAny.end())
            {
                // Ask before the first launch on a device: launching a kernel that has
                // no code object for the device crashes inside HIP (ROCm 7.2.1)
                // instead of returning an error.
                hipFunction_t    function;
                hipError_t const err = hipGetFuncBySymbol(
                    &function, reinterpret_cast<void const*>(&orReduceSynchronizer));
                if(err == hipErrorInvalidDeviceFunction || err == hipErrorNoBinaryForGpu)
                {
                    // Clear the expected probe failure: it is not an error of the run.
                    static_cast<void>(hipGetLastError());
                    return false;
                }
                HIP_CHECK_EXC(err);

                uint32_t* deviceWord = nullptr;
                HIP_CHECK_EXC(hipMalloc(&deviceWord, sizeof(uint32_t)));
                deviceAny = m_deviceAny.emplace(deviceId, deviceWord).first;
                if(m_hostAny == nullptr)
                    HIP_CHECK_EXC(hipHostMalloc(&m_hostAny, sizeof(uint32_t)));
            }
            uint32_t* deviceWord = deviceAny->second;

            assert(bytes % sizeof(uint32_t) == 0);
            auto   buffer = static_cast<uint32_t const*>(device);
            size_t ints   = bytes / sizeof(uint32_t);
            auto   blocks
                = std::min((ints + orReduceBlockSize - 1) / orReduceBlockSize, orReduceMaxBlocks);
            void* args[] = {&buffer, &ints, &deviceWord};

            HIP_CHECK_EXC(hipMemsetAsync(deviceWord, 0, sizeof(uint32_t), stream));
            HIP_CHECK_EXC(hipLaunchKernel(reinterpret_cast<void const*>(&orReduceSynchronizer),
                                          ::dim3(blocks),
                                          ::dim3(orReduceBlockSize),
                                          args,
                                          0,
                                          stream));

            HIP_CHECK_EXC(hipMemcpyAsync(
                m_hostAny, deviceWord, sizeof(uint32_t), hipMemcpyDeviceToHost, stream));
            HIP_CHECK_EXC(hipStreamSynchronize(stream));
            any = *m_hostAny;
            return true;
        }

        bool SynchronizerValidator::checkBuffer(int                           deviceId,
                                                ContractionProblemGemm const& problem,
                                                void*                         deviceSynchronizer,
                                                char const*                   stage,
                                                size_t                        gemmIdx,
                                                CheckTimers const&            timers)
        {
            if(deviceSynchronizer == nullptr)
            {
                ScopedTimer timer(timers.noBuffer);
                return true;
            }

            auto const&  tensor = problem.tensors()[ContractionProblemGemm::TENSOR::Synchronizer];
            size_t const bytes  = tensor.totalAllocatedBytes();
            if(bytes == 0)
                return true;

            // A narrower-than-int alpha leaves the tail of the kernel's range
            // outside the allocation, where residue is unreadable rather than
            // merely unreported.
            if(bytes < tensor.totalAllocatedElements() * sizeof(int))
            {
                // Skipped, not failed: a limit of the checker, not a kernel that
                // failed to self-clean. Warned once so it cannot pass unnoticed.
                if(!m_narrowAlphaWarned)
                {
                    m_narrowAlphaWarned = true;
                    std::ostringstream msg;
                    msg << "Warning: Synchronizer is declared with a type narrower than int (" << bytes
                        << " bytes for " << tensor.totalAllocatedElements()
                        << " elements); --check-synchronizer cannot cover the range the kernel "
                           "uses and is skipped for these problems.\n";
                    m_reporter->log(LogLevel::Error, msg.str());
                }
                return true;
            }

            return checkRegion(
                deviceId,
                deviceSynchronizer,
                bytes,
                concatenate("Synchronizer left dirty after ", stage, " (gemm ", gemmIdx, ")"),
                timers);
        }

        bool SynchronizerValidator::checkRegion(int                deviceId,
                                                void*              device,
                                                size_t             bytes,
                                                std::string const& what,
                                                CheckTimers const& timers)
        {
            if(!m_hostReduction && bytes % sizeof(uint32_t) == 0)
            {
                uint32_t any = 0;
                bool     reduced;
                {
                    ScopedTimer timer(timers.reduce);
                    reduced = reduceBuffer(deviceId, device, bytes, any);
                }
                if(reduced)
                {
                    if(any == 0)
                        return true;
                }
                else
                {
                    // No code object for this device: reduce on the host from now on.
                    m_hostReduction = true;
                    m_reporter->log(LogLevel::Verbose,
                                    "Synchronizer check: the reduction kernel cannot launch on "
                                    "this device; reducing on the host.\n");
                }
            }

            // The host reduction, or the diagnostic for residue the GPU found.
            // Pinned staging avoids the pageable bounce buffer.
            uint8_t* host;
            {
                ScopedTimer timer(timers.readback);
                host = readBuffer(deviceId, device, bytes);
            }

            SynchronizerResidue residue;
            bool                dirty;
            {
                ScopedTimer timer(timers.scan);
                dirty = scanSynchronizerResidue(host, bytes, residue);
            }
            if(!dirty)
                return true;

            std::ostringstream msg;
            msg << what << ": " << residue.nonzeroInts << "/" << residue.totalInts
                << " ints nonzero, first at int offset " << residue.firstInt
                << " -- the kernel did not reset it on exit.\n";
            m_reporter->log(LogLevel::Error, msg.str());

            // resetOutput skips the Synchronizer (it is not an output tensor),
            // so clear the residue here to report it once rather than for every
            // solution that follows.
            clearBuffer(deviceId, device, bytes);

            return false;
        }
    } // namespace Client
} // namespace TensileLite
