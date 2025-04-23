/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/
#include <iostream>

// hiptensor includes
#include "logger.hpp"
#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
#include <hiptensor/internal/hiptensor_utility.hpp>

void printBool(bool in)
{
    std::cout << (in ? "PASSED" : "FAILED") << std::endl;
}

bool loggerSingletonTest()
{
    auto& loggerInit = hiptensor::Logger::instance();
    auto& logger     = hiptensor::Logger::instance();

    return (loggerInit == logger);
}

bool hiptensorLoggerSetCallbackTest()
{
    static bool checkEntry   = false;
    auto        callBackFunc = [](int32_t logContext, const char* funcName, const char* msg) {
        checkEntry = ((logContext
                       == static_cast<int32_t>(hiptensor::Logger::LogLevel_t::LOG_LEVEL_API_TRACE))
                      && !strcmp(funcName, "TestFunction") && (strstr(msg, "TestMessage") != NULL));
    };

    if(hiptensorLoggerSetCallback(callBackFunc) != HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }

    using hiptensor::Logger;
    auto& logger = Logger::instance();

    if(logger->logAPITrace("TestFunction", "TestMessage") != hiptensor::Logger::Status_t::SUCCESS
       || checkEntry == false)
    {
        return false;
    }

    if(hiptensorLoggerSetCallback(nullptr) != HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }

    return true;
}

bool hiptensorLoggerSetFileTest()
{
    std::string fname = std::tmpnam(nullptr);

    // hiptensorLoggerSetFile should have filestream open in write mode.
    FILE* fp = fopen(fname.c_str(), "r");
    if(hiptensorLoggerSetFile(fp) == HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }

    long fileSizeOrig = 0, fileSizeAfter = 0;

    // hiptensorLoggerSetFile with file in write mode.
    fp = fopen(fname.c_str(), "w");
    if(fp == NULL)
    {
        std::cout << " Failed to Open File. Check Permissions!";
        return false;
    }
    else
    {
        fseek(fp, 0, SEEK_END);
        fileSizeOrig = ftell(fp);
        fseek(fp, 0, SEEK_SET);
    }

    //Only Sets Filestream to fp. Doesn't write logs yet.
    if(hiptensorLoggerSetFile(fp) != HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }

    //Call API again to write logs to fp
    if(hiptensorLoggerSetFile(fp) != HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }

    //Check size after API call
    fseek(fp, 0, SEEK_END);
    fileSizeAfter = ftell(fp);
    fclose(fp);
    std::remove(fname.c_str());

    if(fileSizeAfter <= fileSizeOrig)
    {
        return false;
    }

    return true;
}

bool hiptensorLoggerOpenFileTest()
{
    std::string fname   = std::tmpnam(nullptr);
    const char* fname_c = fname.c_str();

    long fileSizeOrig = 0, fileSizeAfter = 0;

    FILE* fp = fopen(fname_c, "w");
    if(fp == NULL)
    {
        std::cout << " Failed to Open File. Check Permissions!";
        return false;
    }
    else
    {
        fseek(fp, 0, SEEK_END);
        fileSizeOrig = ftell(fp);
        fseek(fp, 0, SEEK_SET);
    }

    //Write logs to fname_c.
    if(hiptensorLoggerOpenFile(fname_c) != HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }

    //Check size after API call
    fseek(fp, 0, SEEK_END);
    fileSizeAfter = ftell(fp);
    fclose(fp);
    std::remove(fname.c_str());

    if(fileSizeAfter <= fileSizeOrig)
    {
        return false;
    }

    return true;
}

bool hiptensorLoggerSetLevelTest()
{
    // Test all log levels enumerated in hiptensorLogLevel_t
    if(hiptensorLoggerSetLevel(HIPTENSOR_LOG_LEVEL_OFF) != HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }
    if(hiptensorLoggerSetLevel(HIPTENSOR_LOG_LEVEL_ERROR) != HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }
    if(hiptensorLoggerSetLevel(HIPTENSOR_LOG_LEVEL_PERF_TRACE) != HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }
    if(hiptensorLoggerSetLevel(HIPTENSOR_LOG_LEVEL_PERF_HINT) != HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }
    if(hiptensorLoggerSetLevel(HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE) != HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }
    if(hiptensorLoggerSetLevel(HIPTENSOR_LOG_LEVEL_API_TRACE) != HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }

    // Test out-of-range input value
    if(hiptensorLoggerSetLevel(hiptensorLogLevel_t(3)) == HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }

    return true;
}

bool hiptensorLoggerSetMaskTest()
{
    // Test all bitmask values in range
    for(int i = 0x00; i <= 0x1F; i++)
    {
        if(hiptensorLoggerSetMask(i) != HIPTENSOR_STATUS_SUCCESS)
        {
            return false;
        }
    }

    // Test out-of-range input value
    if(hiptensorLoggerSetMask(0x20) == HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }

    return true;
}

bool hiptensorLoggerForceDisableTest()
{
    if(hiptensorLoggerForceDisable() != HIPTENSOR_STATUS_SUCCESS)
    {
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    bool totalPass = true;
    bool testPass  = false;

    hiptensorLoggerOpenFile("test.log");
    hiptensorLoggerSetLevel(HIPTENSOR_LOG_LEVEL_API_TRACE);

    testPass = loggerSingletonTest();
    totalPass &= testPass;
    std::cout << "Logger Singleton: ";
    printBool(testPass);

    testPass = hiptensorLoggerSetCallbackTest();
    totalPass &= testPass;
    std::cout << "hiptensorLoggerSetCallback: ";
    printBool(testPass);

    testPass = hiptensorLoggerSetFileTest();
    totalPass &= testPass;
    std::cout << "hiptensorLoggerSetFile: ";
    printBool(testPass);

    testPass = hiptensorLoggerOpenFileTest();
    totalPass &= testPass;
    std::cout << "hiptensorLoggerOpenFile: ";
    printBool(testPass);

    //As the above function call sets to a temporary file, need to call OpenFile again.
    hiptensorLoggerOpenFile("test.log");

    testPass = hiptensorLoggerSetLevelTest();
    totalPass &= testPass;
    std::cout << "hiptensorLoggerSetLevel: ";
    printBool(testPass);

    testPass = hiptensorLoggerSetMaskTest();
    totalPass &= testPass;
    std::cout << "hiptensorLoggerSetMask: ";
    printBool(testPass);

    // This test must be performed last as hiptensorLoggerForceDisable() cannot be undone
    testPass = hiptensorLoggerForceDisableTest();
    totalPass &= testPass;
    std::cout << "hiptensorLoggerForceDisableTest: ";
    printBool(testPass);

    if(!totalPass)
        return -1;
    return 0;
}
