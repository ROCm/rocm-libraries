/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************/
#pragma once

#include <vector>
#include <memory>

#ifndef _WIN32
#include <amd_smi/amdsmi.h>

// True when AMD-SMI is answering correctly but the platform can't provide
// the requested telemetry (e.g. some paravirtualized GPU setups don't expose
// a real PCI BDF), as opposed to a genuine AMD-SMI failure.
inline bool isAmdsmiTelemetryUnavailable(amdsmi_status_t status)
{
    return status == AMDSMI_STATUS_NOT_SUPPORTED;
}

// Best-effort AMD-SMI processor index for `hipDeviceIndex` when BDF matching
// isn't available, clamped to a valid index.
inline uint32_t selectFallbackAmdsmiIndex(int hipDeviceIndex, uint32_t amdsmiDeviceCount)
{
    if(amdsmiDeviceCount == 0)
        return 0;
    if(hipDeviceIndex <= 0)
        return 0;

    uint32_t index = static_cast<uint32_t>(hipDeviceIndex);
    return index < amdsmiDeviceCount ? index : amdsmiDeviceCount - 1;
}

// Outcome of one step of GetAMDSMIIndex()'s BDF-matching loop.
enum class BdfMatchAction
{
    ContinueSearch, // no match yet; check the next processor
    ReturnIndex,     // done - use BdfMatchDecision::index
    Throw            // genuine AMD-SMI failure
};

struct BdfMatchDecision
{
    BdfMatchAction action;
    uint32_t       index = 0;
};

// Decision logic for one iteration of GetAMDSMIIndex(), factored out so it's
// unit testable without a GPU/AMD-SMI session.
inline BdfMatchDecision decideBdfMatch(amdsmi_status_t status,
                                        uint32_t        smiIndex,
                                        uint64_t        amdSmiPciId,
                                        uint64_t        hipPciId,
                                        int             hipDeviceIndex,
                                        uint32_t        amdsmiDeviceCount)
{
    if(isAmdsmiTelemetryUnavailable(status))
        return {BdfMatchAction::ReturnIndex,
                selectFallbackAmdsmiIndex(hipDeviceIndex, amdsmiDeviceCount)};

    if(status != AMDSMI_STATUS_SUCCESS)
        return {BdfMatchAction::Throw, 0};

    if(amdSmiPciId == hipPciId)
        return {BdfMatchAction::ReturnIndex, smiIndex};

    return {BdfMatchAction::ContinueSearch, 0};
}
#endif

class EfficiencyMonitor
{
public:
    virtual ~EfficiencyMonitor()    = default;
    virtual bool enabled()          = 0;
    virtual bool detailedReport()   = 0;
    virtual bool efficiencyReport() = 0;

    virtual void setDeviceId(int deviceId) = 0;

    virtual void start() = 0;
    virtual void stop()  = 0;

    virtual double              getLowestAverageSYSCLK()   = 0;
    virtual double              getLowestMedianSYSCLK()    = 0;
    virtual std::vector<double> getAllAverageSYSCLK()      = 0;
    virtual std::vector<double> getAllMedianSYSCLK()       = 0;
    virtual double              getAverageMEMCLK()         = 0;
    virtual double              getMedianMEMCLK()          = 0;
    virtual double              getTotalGranularityValue() = 0;
    virtual double              getTilesPerCuValue()       = 0;
    virtual double              getTile0Granularity()      = 0;
    virtual double              getTile1Granularity()      = 0;
    virtual double              getCuGranularity()         = 0;
    virtual double              getWaveGranularity()       = 0;
    virtual int                 getCUs()                   = 0;
    virtual size_t              getMemWriteBytesD()        = 0;
    virtual size_t              getMemReadBytes()          = 0;
    virtual uint16_t            getCuCount()               = 0;
    virtual std::string         getDeviceString()          = 0;

    static std::shared_ptr<EfficiencyMonitor> create();
};
