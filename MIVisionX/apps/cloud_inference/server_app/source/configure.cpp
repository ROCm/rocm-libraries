/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "configure.h"
#include "netutil.h"
#include "common.h"

int runConfigure(int sock, Arguments * args, std::string& clientName, InfComCommand * cmd)
{
    //////
    /// \brief send the configuration info
    ///
    InfComCommand reply;

    // send: INFCOM_CMD_CONFIG_INFO { modelCount, numGpus, shadowFolderAvailable }
    int modelCount = args->getNumConfigureddModels();
    InfComCommand config_info = {
        INFCOM_MAGIC, INFCOM_CMD_CONFIG_INFO,
        { modelCount, args->getNumGPUs(), !args->getlocalShadowRootDir().empty() },
        { 0 }
    };
    ERRCHK(sendCommand(sock, config_info, clientName));
    info("number of pre-configured models: %d", modelCount);
    ERRCHK(recvCommand(sock, reply, clientName, INFCOM_CMD_CONFIG_INFO));
    for(size_t i = 0; i < modelCount; i++) {
        // send: INFCOM_CMD_MODEL_INFO { iw, ih, ic, ow, oh, oc } "modelName"
        std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float,std::string> model_config = args->getConfiguredModelInfo(i);
        auto float_as_int = [](float v) -> int { return *(int *)&v; };
        InfComCommand model_info = {
            INFCOM_MAGIC, INFCOM_CMD_MODEL_INFO,
            { std::get<1>(model_config), std::get<2>(model_config), std::get<3>(model_config),
              std::get<4>(model_config), std::get<5>(model_config), std::get<6>(model_config),
              std::get<7>(model_config),
              float_as_int(std::get<8>(model_config)), float_as_int(std::get<9>(model_config)), float_as_int(std::get<10>(model_config)),
              float_as_int(std::get<11>(model_config)), float_as_int(std::get<12>(model_config)), float_as_int(std::get<13>(model_config))
            },
            { 0 }
        };
        strncpy(model_info.message, std::get<0>(model_config).c_str(), sizeof(model_info.message));
        ERRCHK(sendCommand(sock, model_info, clientName));
        info("pre-configured model#%d: %s [input %dx%dx%d] [output %dx%dx%d] [reverseInputChannelOrder %d] [mpy %g %g %g] [add %g %g %g]", i, model_info.message,
             model_info.data[2], model_info.data[1], model_info.data[0],
             model_info.data[5], model_info.data[4], model_info.data[3],
             model_info.data[6],
             std::get<8>(model_config), std::get<9>(model_config), std::get<10>(model_config),
             std::get<11>(model_config), std::get<12>(model_config), std::get<13>(model_config));
        ERRCHK(recvCommand(sock, reply, clientName, INFCOM_CMD_MODEL_INFO));
    }

    // wait for INFCOM_CMD_DONE message
    ERRCHK(recvCommand(sock, reply, clientName, INFCOM_CMD_DONE));

    return 0;
}
