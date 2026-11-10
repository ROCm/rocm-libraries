// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TriggeredGateValidator.hpp"

#include "DeviceContext.hpp"
#include "ResultReporter.hpp"

#include <Tensile/GateKernArg.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <cstring>

namespace TensileLite
{
    namespace Client
    {
        namespace
        {
            constexpr size_t kGateStride  = 64;
            constexpr size_t kMaxReported = 4;

            size_t ceilDiv(size_t a, size_t b)
            {
                return (a + b - 1) / b;
            }
        }

        GateTableLayout gateTableLayout(size_t bands, uint32_t publishers)
        {
            GateTableLayout layout;
            layout.bands      = bands;
            layout.publishers = publishers;
            layout.gatesOffset
                = ceilDiv(sizeof(GateTableHeader) + bands * sizeof(GateTileRecord), kGateStride)
                  * kGateStride;
            layout.bytes = layout.gatesOffset + bands * kGateStride;
            return layout;
        }

        void fillGateTable(std::vector<uint8_t>&  host,
                           GateTableLayout const& layout,
                           uint64_t               deviceBase)
        {
            host.assign(layout.bytes, 0);

            GateTableHeader header{};
            header.abiVersion     = GATE_ABI_VERSION;
            header.structSize     = sizeof(GateTableHeader);
            header.tileRecordSize = sizeof(GateTileRecord);
            header.tileCount      = static_cast<uint32_t>(layout.bands);
            header.heapBase       = deviceBase;
            header.tiles          = deviceBase + sizeof(GateTableHeader);
            std::memcpy(host.data(), &header, sizeof(header));

            for(size_t b = 0; b < layout.bands; b++)
            {
                GateTileRecord record{};
                record.tileId         = static_cast<uint32_t>(b);
                record.roleFlags      = GATE_ROLE_PUBLISH;
                record.publisherCount = layout.publishers;
                record.readyOffset    = layout.gatesOffset + b * kGateStride;
                std::memcpy(host.data() + sizeof(GateTableHeader) + b * sizeof(GateTileRecord),
                            &record,
                            sizeof(record));
            }
        }

        TriggeredGateValidator::TriggeredGateValidator(po::variables_map const& args)
            : m_enabled(args.count("triggered-gate-epilogue")
                        && args["triggered-gate-epilogue"].as<bool>())
        {
        }

        void TriggeredGateValidator::preSolution(ContractionSolution* const solution)
        {
            m_failedInSolution = false;
            m_checked          = false;
            m_active
                = m_enabled && solution != nullptr && solution->problemType.triggeredGateEpilogue;
            if(!m_active)
                return;

            // Runs outside the client's per-solution try block. On failure the
            // table pointer stays null, so solve() rejects the launch.
            try
            {
                setUpTables(*solution);
            }
            catch(std::exception const& err)
            {
                m_reporter->log(LogLevel::Error, concatenate("[gate] ", err.what(), "\n"));
                m_failedInSolution = true;
                m_checked          = true;
            }
        }

        void TriggeredGateValidator::setUpTables(ContractionSolution const& solution)
        {
            auto const* gemm = dynamic_cast<ContractionProblemGemm const*>(m_problem);
            if(gemm == nullptr)
                throw std::runtime_error("TriggeredGateEpilogue needs a GEMM problem");

            size_t batch = 1;
            for(size_t i = 0; i < gemm->batchIndices().size(); i++)
                batch *= gemm->batchSize(i);
            auto const& macroTile = solution.sizeMapping.macroTile;
            m_layout              = gateTableLayout(
                ceilDiv(gemm->freeSizeB(0), macroTile.y),
                static_cast<uint32_t>(ceilDiv(gemm->freeSizeA(0), macroTile.x) * batch));

            for(auto const& device : *m_devices)
            {
                ScopedDevice guard(device->deviceId);
                if(device->gateTableBytes < m_layout.bytes)
                {
                    if(device->gateTable != nullptr)
                        HIP_CHECK_EXC(hipFree(device->gateTable));
                    device->gateTable      = nullptr;
                    device->gateTableBytes = 0;
                    HIP_CHECK_EXC(hipMalloc(&device->gateTable, m_layout.bytes));
                    device->gateTableBytes = m_layout.bytes;
                }
                fillGateTable(m_host, m_layout, reinterpret_cast<uint64_t>(device->gateTable));
                HIP_CHECK_EXC(hipMemcpy(
                    device->gateTable, m_host.data(), m_layout.bytes, hipMemcpyHostToDevice));
            }
        }

        void TriggeredGateValidator::preWarmup()
        {
            if(!m_active || m_checked)
                return;
            for(auto const& device : *m_devices)
            {
                ScopedDevice guard(device->deviceId);
                HIP_CHECK_EXC(
                    hipMemsetAsync(static_cast<uint8_t*>(device->gateTable) + m_layout.gatesOffset,
                                   0,
                                   m_layout.bands * kGateStride,
                                   device->stream));
            }
        }

        void TriggeredGateValidator::validateWarmups(std::vector<size_t> const& slots,
                                                     TimingEvents const&        startEvents,
                                                     TimingEvents const&        stopEvents)
        {
            if(!m_active || m_checked)
                return;
            m_checked = true;
            try
            {
                checkGates();
            }
            catch(...)
            {
                m_failedInSolution = true;
                throw;
            }
        }

        void TriggeredGateValidator::checkGates()
        {
            std::vector<uint8_t> gates(m_layout.bands * kGateStride);
            for(auto const& device : *m_devices)
            {
                ScopedDevice guard(device->deviceId);
                HIP_CHECK_EXC(hipStreamSynchronize(device->stream));
                HIP_CHECK_EXC(
                    hipMemcpy(gates.data(),
                              static_cast<uint8_t*>(device->gateTable) + m_layout.gatesOffset,
                              gates.size(),
                              hipMemcpyDeviceToHost));
                size_t mismatches = 0;
                for(size_t b = 0; b < m_layout.bands; b++)
                {
                    uint64_t got;
                    std::memcpy(&got, gates.data() + b * kGateStride, sizeof(got));
                    if(got == m_layout.publishers)
                        continue;
                    if(mismatches++ < kMaxReported)
                        m_reporter->log(LogLevel::Error,
                                        concatenate("[gate] device ",
                                                    device->deviceId,
                                                    " band ",
                                                    b,
                                                    ": gate ",
                                                    got,
                                                    ", expected ",
                                                    m_layout.publishers,
                                                    "\n"));
                }
                if(mismatches > 0)
                    m_failedInSolution = true;
            }
        }

        void TriggeredGateValidator::postSolution()
        {
            if(m_failedInSolution)
            {
                m_errorsReported++;
                m_reporter->report(ResultKey::Validation, "FAILED");
            }
            m_failedInSolution = false;
            m_active           = false;
        }
    } // namespace Client
} // namespace TensileLite
