// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

struct HipdnnHipKernelSettings
{
    void setBenchmarkingEnabled(bool enabled)
    {
        _benchmarkingEnabled = enabled;
    }

    bool benchmarkingEnabled() const
    {
        return _benchmarkingEnabled;
    }

private:
    bool _benchmarkingEnabled = false;
};
