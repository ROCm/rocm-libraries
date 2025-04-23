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

#include "compiler.h"
#include "netutil.h"
#include "common.h"
#include <sstream>

int runCompiler(int sock, Arguments * args, std::string& clientName, InfComCommand * cmdMode)
{
    //////
    /// \brief get and check parameters
    ///
    int dimInput[3] = { cmdMode->data[1], cmdMode->data[2], cmdMode->data[3] };
    int modelType = cmdMode->data[4];
    int reverseInputChannelOrder = cmdMode->data[5];
    float preprocessMpy[3] = { *(float *)&cmdMode->data[6], *(float *)&cmdMode->data[7], *(float *)&cmdMode->data[8] };
    float preprocessAdd[3] = { *(float *)&cmdMode->data[9], *(float *)&cmdMode->data[10], *(float *)&cmdMode->data[11] };
    bool overrideModel = false;
    std::string saveModelAs;
    std::string password;
    if(dimInput[0] <= 0 || dimInput[1] <= 0 || dimInput[2] != 3) {
        dumpCommand("X", *cmdMode);
        return error_close(sock, "unsupported input dimensions %dx%dx%d", dimInput[2], dimInput[1], dimInput[0]);
    }
    if(modelType != 0) {
        dumpCommand("X", *cmdMode);
        return error_close(sock, "unsupported compiler model type = %d", modelType);
    }
    if(strlen(cmdMode->message) > 0) {
        std::stringstream ss(cmdMode->message);
        std::string option;
        while (std::getline(ss, option, ',')) {
            info("option %s", option.c_str());
            if(option == "override") {
                overrideModel = true;
            }
            else if(option.size() > 5 && option.substr(0, 5) == "save=") {
                saveModelAs = option.substr(5);
            }
            else if(option.size() > 7 && option.substr(0, 7) == "passwd=") {
                password = option.substr(7);
            }
            else {
                return error_close(sock, "unsupported compiler options [%s]", option.c_str());
            }
        }
    }
    if(saveModelAs.length() > 0) {
        for(size_t i = 0; i < saveModelAs.length(); i++) {
            if((i >= 32) ||
               !((saveModelAs[i] >= 'a' && saveModelAs[i] <= 'z') ||
                 (saveModelAs[i] >= 'A' && saveModelAs[i] <= 'Z') ||
                 (saveModelAs[i] >= '0' && saveModelAs[i] <= '9') ||
                 (saveModelAs[i] == '-') ||
                 (saveModelAs[i] == '_')))
            {
                return error_close(sock, "invalid options: modelName is not valid [%s]", saveModelAs.c_str());
            }
        }
        bool found = false;
        for(size_t i = 0; i < args->getNumConfigureddModels(); i++) {
            if(std::get<0>(args->getConfiguredModelInfo(i)) == saveModelAs ||
               std::get<14>(args->getConfiguredModelInfo(i)) == saveModelAs)
            {
                found = true;
                break;
            }
        }
        if(!found) {
            for(size_t i = 0; i < args->getNumUploadedModels(); i++) {
                if(std::get<0>(args->getUploadedModelInfo(i)) == saveModelAs) {
                    found = true;
                    break;
                }
            }
        }
        if(found && !overrideModel) {
            return error_close(sock, "modelName already in use [%s]", saveModelAs.c_str());
        }
        if(!args->checkPassword(password)) {
            return error_close(sock, "invalid password");
        }
    }

    //////
    /// \brief generate new model name for this download and create model folders
    ///
    char modelName[64];
    if(saveModelAs.length() > 0) {
        sprintf(modelName, "%s", saveModelAs.c_str());
    }
    else {
        sprintf(modelName, "upload/model-%08d", args->getNextModelUploadCounter());
    }
    std::string modelFolder = args->getConfigurationDir() + "/" + modelName;
    std::string buildFolder = modelFolder + "/build";
    // remove modelFolder if exists
    std::string command = "/bin/rm -rf ";
    command += modelFolder;
    info("executing: %% %s", command.c_str());
    if(system(command.c_str()) < 0) {
        return error_close(sock, "unable to remove folder %s", modelName);
    }
    // make folders
    if(mkdir(modelFolder.c_str(), 0700) < 0 || mkdir(buildFolder.c_str(), 0700) < 0) {
        fatal("unable to create folders: %s and %s", modelFolder.c_str(), buildFolder.c_str());
    }

    //////
    /// \brief download model files
    ///
    int modelFileCommand[2] = {
        INFCOM_CMD_SEND_MODELFILE1, INFCOM_CMD_SEND_MODELFILE2
    };
    for(int i = 0; i < 2; i++) {
        // send INFCOM_CMD_SEND_MODELFILE1 or INFCOM_CMD_SEND_MODELFILE2
        InfComCommand cmd = {
            INFCOM_MAGIC, modelFileCommand[i], { 0 }, { 0 }
        };
        ERRCHK(sendCommand(sock, cmd, clientName));
        // wait for reply with same command and fileSize in bytes
        ERRCHK(recvCommand(sock, cmd, clientName, modelFileCommand[i]));
        // receive the modelFile byte stream
        int size = cmd.data[0];
        char * byteStream = nullptr;
        if(size > 0) {
            byteStream = new char [size];
            int remaining = size;
            while(remaining > 0) {
                int n = recv(sock, byteStream + size - remaining, remaining, 0);
                if(n < 1)
                    break;
                remaining -= n;
            }
            if(remaining > 0) {
                delete[] byteStream;
                return error_close(sock, "INFCOM_CMD_SEND_MODELFILE%d: could only received %d bytes out of %d bytes from %s", i + 1, size - remaining, size, clientName.c_str());
            }
            int eofMarker = 0;
            recv(sock, &eofMarker, sizeof(eofMarker), 0);
            if(eofMarker != INFCOM_EOF_MARKER) {
                delete[] byteStream;
                return error_close(sock, "INFCOM_CMD_SEND_MODELFILE%d: eofMarker 0x%08x (incorrect) from %s", i + 1, eofMarker, clientName.c_str());
            }
        }
        if(byteStream) {
            std::string fileName = modelFolder + ((i == 0) ? "/deploy.prototxt" : "/weights.caffemodel");
            info("saving INFCOM_CMD_SEND_MODELFILE%d with %d bytes from %s into %s", i + 1, size, clientName.c_str(), fileName.c_str());
            FILE * fp = fopen(fileName.c_str(), "wb");
            if(fp) {
                fwrite(byteStream, 1, size, fp);
                fclose(fp);
            }
            else {
                fatal("unable to create: %s", fileName.c_str());
            }
            delete[] byteStream;
        }
    }

    //////
    /// \brief start inference generator
    ///
    InfComCommand cmdUpdate = {
        INFCOM_MAGIC, INFCOM_CMD_COMPILER_STATUS, { 0 }, { 0 }
    };
    int status = chdir(modelFolder.c_str());
    if(status < 0) {
        cmdUpdate.data[0] = -1;
        sprintf(cmdUpdate.message, "model folder not found");
        ERRCHK(sendCommand(sock, cmdUpdate, clientName));
        ERRCHK(recvCommand(sock, cmdUpdate, clientName, INFCOM_CMD_COMPILER_STATUS));
        return error_close(sock, "chdir('%s') failed", modelFolder.c_str());
    }
    // step-1: run inference generator
    cmdUpdate.data[0] = 0;
    cmdUpdate.data[1] = 1;
    sprintf(cmdUpdate.message, "inference_generator started ...");
    ERRCHK(sendCommand(sock, cmdUpdate, clientName));
    ERRCHK(recvCommand(sock, cmdUpdate, clientName, INFCOM_CMD_COMPILER_STATUS));

    if (args->getModelCompilerPath().empty())
    {
        // step-1.1: caffe2openvx on caffemodel for weights
        command = "caffe2openvx weights.caffemodel";
        command += " " + std::to_string(args->getBatchSize())
                +  " " + std::to_string(dimInput[2])
                +  " " + std::to_string(dimInput[1])
                +  " " + std::to_string(dimInput[0])
                +  " >caffe2openvx.log";
        info("executing: %% %s", command.c_str());
        status = system(command.c_str());
        cmdUpdate.data[0] = (status != 0) ? -2 : 0;
        cmdUpdate.data[1] = 25;
        sprintf(cmdUpdate.message, "caffe2openvx weights.caffemodel completed (%d)", status);
        ERRCHK(sendCommand(sock, cmdUpdate, clientName));
        ERRCHK(recvCommand(sock, cmdUpdate, clientName, INFCOM_CMD_COMPILER_STATUS));
        if(status) {
            return error_close(sock, "command-failed(%d): %s", status, command.c_str());
        }
        // step-1.2: caffe2openvx on prototxt for network structure
        command = "caffe2openvx deploy.prototxt";
        command += " " + std::to_string(args->getBatchSize())
                +  " " + std::to_string(dimInput[2])
                +  " " + std::to_string(dimInput[1])
                +  " " + std::to_string(dimInput[0])
                +  " >>caffe2openvx.log";
        info("executing: %% %s", command.c_str());
        status = system(command.c_str());
        cmdUpdate.data[0] = (status != 0) ? -3 : 0;
        cmdUpdate.data[1] = 50;
        sprintf(cmdUpdate.message, "caffe2openvx deploy.prototxt completed (%d)", status);
        ERRCHK(sendCommand(sock, cmdUpdate, clientName));
        ERRCHK(recvCommand(sock, cmdUpdate, clientName, INFCOM_CMD_COMPILER_STATUS));
        if(status) {
            return error_close(sock, "command-failed(%d): %s", status, command.c_str());
        }
    } else
    {
        // run nnir model_compiler
        // step-1.1: run python3 caffe_to_nnir <.caffemodel> nnir_output --input-dims <args->getBatchSize(),dimOutput[2], dimOutput[1], dimOutput[0]>
        command = "python3 ";
        command += args->getModelCompilerPath() + "/" + "caffe_to_nnir.py weights.caffemodel nnir-output --input-dims";
        command += " " + std::to_string(args->getBatchSize())
                +  "," + std::to_string(dimInput[2])
                +  "," + std::to_string(dimInput[1])
                +  "," + std::to_string(dimInput[0]);
        info("executing: %% %s", command.c_str());
        status = system(command.c_str());
        sprintf(cmdUpdate.message, "python caffe_to_nnir weights.caffemodel completed (%d)", status);
        cmdUpdate.data[0] = (status != 0) ? -2 : 0;
        cmdUpdate.data[1] = 25;
        ERRCHK(sendCommand(sock, cmdUpdate, clientName));
        ERRCHK(recvCommand(sock, cmdUpdate, clientName, INFCOM_CMD_COMPILER_STATUS));
        if(status) {
            return error_close(sock, "command-failed(%d): %s", status, command.c_str());
        }
        // steo-1.2: todo:: nnir_update
        command = "python3 ";
        command += args->getModelCompilerPath() + "/" + "nnir_update.py --fuse-ops 1";  // --fuse-ops is required to fuse batch-norm at NNIR. Workaround for FP16 MIOPen bug with batchnorm
        if (args->fp16Inference())
        {
            command += " --convert-fp16 1";
        }
        command += " nnir-output nnir-output_1";
        info("executing: %% %s", command.c_str());
        status = system(command.c_str());
        sprintf(cmdUpdate.message, "python nnir_update.py fuse-ops 1 <--convert-fp16 1> nnir-output nnir-output_1 completed (%d)", status);
        cmdUpdate.data[0] = (status != 0) ? -3 : 0;
        cmdUpdate.data[1] = 50;
        ERRCHK(sendCommand(sock, cmdUpdate, clientName));
        ERRCHK(recvCommand(sock, cmdUpdate, clientName, INFCOM_CMD_COMPILER_STATUS));
        if(status) {
            return error_close(sock, "command-failed(%d): %s", status, command.c_str());
        }

        // step-1.3: nnir_to_openvx
        command = "python3 ";
        command += args->getModelCompilerPath() + "/" + "nnir_to_openvx.py nnir-output_1 ."
                +  " >>caffe2openvx.log";
        info("executing: %% %s", command.c_str());
        status = system(command.c_str());
        cmdUpdate.data[0] = (status != 0) ? -4 : 0;
        cmdUpdate.data[1] = 75;
        sprintf(cmdUpdate.message, "python nnir_to_openvx completed (%d)", status);
        ERRCHK(sendCommand(sock, cmdUpdate, clientName));
        ERRCHK(recvCommand(sock, cmdUpdate, clientName, INFCOM_CMD_COMPILER_STATUS));
        if(status) {
            return error_close(sock, "command-failed(%d): %s", status, command.c_str());
        }
    }
    // step-2: get output dimensions
    int dimOutput[3] = { 0 };
    FILE * fp = fopen("caffe2openvx.log", "r");
    if(!fp) {
        return error_close(sock, "unable to open: caffe2openvx.log");
    }
    char line[1024];
    while(fgets(line, sizeof(line), fp) == line) {
        if(!strncmp(line, "#OUTPUT-TENSOR: ", 16)) {
            sscanf(line, "%*s%*s%*s%d%d%d", &dimOutput[2], &dimOutput[1], &dimOutput[0]);
        }
    }
    fclose(fp);
    info("found output tensor dimensions %dx%dx%d for %s", dimOutput[2], dimOutput[1], dimOutput[0], modelName);
    // step-3: build the module
    status = chdir(buildFolder.c_str());
    if(status < 0) {
        cmdUpdate.data[0] = -1;
        sprintf(cmdUpdate.message, "build folder not found");
        ERRCHK(sendCommand(sock, cmdUpdate, clientName));
        ERRCHK(recvCommand(sock, cmdUpdate, clientName, INFCOM_CMD_COMPILER_STATUS));
        return error_close(sock, "chdir('%s') failed", buildFolder.c_str());
    }
    command = "cmake .. >../cmake.log";
    info("executing: %% %s", command.c_str());
    status = system(command.c_str());
    std::string makefilePath = buildFolder + "/Makefile";
    struct stat sbufMakefile = { 0 };
    if(stat(makefilePath.c_str(), &sbufMakefile) != 0) {
        status = 1;
        warning("could not locate cmake output: %s", makefilePath.c_str());
    }
    cmdUpdate.data[0] = (status != 0) ? -4 : 0;
    cmdUpdate.data[1] = 75;
    sprintf(cmdUpdate.message, "cmake completed (status = %d)", status);
    ERRCHK(sendCommand(sock, cmdUpdate, clientName));
    ERRCHK(recvCommand(sock, cmdUpdate, clientName, INFCOM_CMD_COMPILER_STATUS));
    if(status) {
        return error_close(sock, "command-failed(%d): %s", status, command.c_str());
    }
    command = "make >../make.log";
    info("executing: %% %s", command.c_str());
    status = system(command.c_str());
    cmdUpdate.data[0] = (status != 0) ? -5 : 0;
    cmdUpdate.data[1] = 99;
    sprintf(cmdUpdate.message, "make completed (status = %d)", status);
    ERRCHK(sendCommand(sock, cmdUpdate, clientName));
    ERRCHK(recvCommand(sock, cmdUpdate, clientName, INFCOM_CMD_COMPILER_STATUS));
    if(status) {
        return error_close(sock, "command-failed(%d): %s", status, command.c_str());
    }
    std::string modulePath = buildFolder + "/" + MODULE_LIBNAME;
    struct stat sbufModule = { 0 };
    if(stat(modulePath.c_str(), &sbufModule) != 0) {
        cmdUpdate.data[0] = -6;
        sprintf(cmdUpdate.message, "couldn't locate generated module");
        ERRCHK(sendCommand(sock, cmdUpdate, clientName));
        ERRCHK(recvCommand(sock, cmdUpdate, clientName, INFCOM_CMD_COMPILER_STATUS));
        return error_close(sock, "could not locate built module: %s", modulePath.c_str());
    }

    // step-final: send completion status message
    cmdUpdate.data[0] = 1;
    cmdUpdate.data[1] = 100;
    cmdUpdate.data[2] = dimOutput[0];
    cmdUpdate.data[3] = dimOutput[1];
    cmdUpdate.data[4] = dimOutput[2];
    sprintf(cmdUpdate.message, "%s", modelName);
    ERRCHK(sendCommand(sock, cmdUpdate, clientName));
    ERRCHK(recvCommand(sock, cmdUpdate, clientName, INFCOM_CMD_COMPILER_STATUS));

    // create module configuration file
    std::string annModuleConfigFile = args->getConfigurationDir() + "/" + modelName + "/" + MODULE_CONFIG;
    fp = fopen(annModuleConfigFile.c_str(), "w");
    if(fp) {
        fprintf(fp, "%s\n%d %d %d\n%d %d %d\n%d\n%g %g %g %g %g %g\n", modelName,
                       dimInput[0], dimInput[1], dimInput[2],
                       dimOutput[0], dimOutput[1], dimOutput[2],
                       reverseInputChannelOrder,
                       preprocessMpy[0], preprocessMpy[1], preprocessMpy[2],
                       preprocessAdd[0], preprocessAdd[1], preprocessAdd[2]);
        fclose(fp);
    }
    else {
        fatal("unable to create: %s", annModuleConfigFile.c_str());
    }

    // add uploaded model to args
    if(saveModelAs.length() > 0) {
        std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float,std::string>
                ann(modelName, dimInput[0], dimInput[1], dimInput[2], dimOutput[0], dimOutput[1], dimOutput[2],
                reverseInputChannelOrder,
                preprocessMpy[0], preprocessMpy[1], preprocessMpy[2],
                preprocessAdd[0], preprocessAdd[1], preprocessAdd[2],
                modelName);
        args->addConfigToPreconfiguredList(ann);
    }
    else {
        std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float>
                ann(modelName, dimInput[0], dimInput[1], dimInput[2], dimOutput[0], dimOutput[1], dimOutput[2],
                reverseInputChannelOrder,
                preprocessMpy[0], preprocessMpy[1], preprocessMpy[2],
                preprocessAdd[0], preprocessAdd[1], preprocessAdd[2]);
        args->addConfigToUploadedList(ann);
    }
    info("added uploaded model name:%s input:%dx%dx%d output:%dx%dx%d reverseInputChannelOrder:%d mpy:[%g %g %g] add:[%g %g %g]",
            modelName, dimInput[2], dimInput[1], dimInput[0], dimOutput[2], dimOutput[1], dimOutput[0],
            reverseInputChannelOrder,
            preprocessMpy[0], preprocessMpy[1], preprocessMpy[2],
            preprocessAdd[0], preprocessAdd[1], preprocessAdd[2]);

    // send and wait for INFCOM_CMD_DONE message
    InfComCommand reply = {
        INFCOM_MAGIC, INFCOM_CMD_DONE, { 0 }, { 0 }
    };
    ERRCHK(sendCommand(sock, reply, clientName));
    ERRCHK(recvCommand(sock, reply, clientName, INFCOM_CMD_DONE));

    return 0;
}
