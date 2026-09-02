// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "driver.hpp"
#include "registry_driver_maker.hpp"

#include "sys_info.hpp"
#include <miopen/config.h>
#include <miopen/errors.hpp>
#include <miopen/stringutils.hpp>

#include <cstdio>
#include <iostream>

int main(int argc, char* argv[])
{

    std::string base_arg = ParseBaseArg(argc, argv);

    if(base_arg == "--version")
    {
        size_t major, minor, patch;
        miopenGetVersion(&major, &minor, &patch);
        std::cout << "MIOpen (version: " << major << "." << minor << "." << patch << ")"
                  << std::endl;
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    const bool json_mode = miopen::IsPerformanceLoggingEnabled();
    try
    {

        // show command
        if(!json_mode)
        {
            std::cout << "MIOpenDriver";
            for(int i = 1; i < argc; i++)
                std::cout << " " << argv[i];
            std::cout << std::endl;
        }
        else
        {
            std::cout << "{\"command\":\"MIOpenDriver";
            for(int i = 1; i < argc; i++)
                std::cout << " " << argv[i];
            std::cout << "\"}" << std::endl;
        }

        std::shared_ptr<Driver> drv;
        for(auto f : rdm::GetRegistry())
        {
            drv.reset(f(base_arg));
            if(drv != nullptr)
                break;
        }
        if(drv == nullptr)
        {
            if(!json_mode)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Incorrect BaseArg");
            }
            else
            {
                std::cout << "{\"error\":\"Incorrect BaseArg\"}" << std::endl;
                exit(0); // NOLINT (concurrency-mt-unsafe)
            }
        }

        drv->name = base_arg;

        drv->AddCmdLineArgs();
        int rc = drv->ParseCmdLineArgs(argc, argv);
        if(rc != 0)
        {
            if(!json_mode)
            {
                std::cout << "ParseCmdLineArgs() FAILED, rc = " << rc << std::endl;
            }
            else
            {
                std::cout << "{\"error\":\"ParseCmdLineArgs() FAILED, rc = " << rc << "\"}"
                          << std::endl;
            }
            return rc;
        }

        drv->GetandSetData();

        rc = drv->AllocateBuffersAndCopy();

        if(rc != 0)
        {
            if(!json_mode)
            {
                std::cout << "AllocateBuffersAndCopy() FAILED, rc = " << rc << std::endl;
            }
            else
            {
                std::cout << "{\"error\":\"AllocateBuffersAndCopy() FAILED, rc = " << rc << "\"}"
                          << std::endl;
            }
            return rc;
        }

        if(drv->GetInputFlags().GetValueInt("time") == 1)
        {
            // Print system information for ROCmPerf analysis.
            // The ROCmPerf is a performance analysis tool based on MIOpenDirver logs.
            size_t major, minor, patch;
            miopenGetVersion(&major, &minor, &patch);
            RocmPerf::SysInfo sysInfo(major, minor, patch);
            sysInfo.ShowSysInfo();
        }

        int fargval =
            !(miopen::StartsWith(base_arg, "CBAInfer") || miopen::StartsWith(base_arg, "CAInfer"))
                ? drv->GetInputFlags().GetValueInt("forw")
                : 1;
        bool bnFwdInVer   = (fargval == 2 && miopen::StartsWith(base_arg, "bnorm"));
        bool verifyarg    = (drv->GetInputFlags().GetValueInt("verify") == 1);
        int cumulative_rc = 0; // Do not stop running tests in case of errors.

        if(fargval & 1 || fargval == 0 || bnFwdInVer)
        {
            rc = drv->RunForwardGPU();
            cumulative_rc |= rc;
            if(rc != 0)
            {
                if(!json_mode)
                {
                    std::cout << "RunForwardGPU() FAILED, rc = " << "0x" << std::hex << rc
                              << std::dec << std::endl;
                }
                else
                {
                    std::cout << "{\"error\":\"RunForwardGPU() FAILED, rc = 0x" << std::hex << rc
                              << std::dec << "\"}" << std::endl;
                }
            }
            if(verifyarg) // Verify even if Run() failed.
                cumulative_rc |= drv->VerifyForward();
        }

        if(fargval != 1)
        {
            rc = drv->RunBackwardGPU();
            cumulative_rc |= rc;
            if(rc != 0)
            {
                if(!json_mode)
                {
                    std::cout << "RunBackwardGPU() FAILED, rc = " << "0x" << std::hex << rc
                              << std::dec << std::endl;
                }
                else
                {
                    std::cout << "{\"error\":\"RunBackwardGPU() FAILED, rc = 0x" << std::hex << rc
                              << std::dec << "\"}" << std::endl;
                }
            }
            if(verifyarg) // Verify even if Run() failed.
                cumulative_rc |= drv->VerifyBackward();
        }

        // Flush any accumulated performance logging data before exiting
        miopen::FinalizeJsonLogging();

        return cumulative_rc;
    }
    catch(const miopen::Exception& ex)
    {
        // Flush any accumulated performance logging data before exiting
        miopen::FinalizeJsonLogging();

        if(!json_mode)
        {
            std::cerr << "Error: " << ex.what() << std::endl;
        }
        else
        {
            std::cout << "{\"error\":\"" << ex.what() << "\"}" << std::endl;
        }
        return EXIT_FAILURE;
    }
    catch(const std::exception& ex)
    {
        // Flush any accumulated performance logging data before exiting
        miopen::FinalizeJsonLogging();

        if(!json_mode)
        {
            std::cerr << "Error: " << ex.what() << std::endl;
        }
        else
        {
            std::cout << "{\"error\":\"" << ex.what() << "\"}" << std::endl;
        }
        return EXIT_FAILURE;
    }
}
