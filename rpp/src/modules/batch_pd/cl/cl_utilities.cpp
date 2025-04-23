/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "rpp_cl_common.hpp"
#include "cl_declarations.hpp"

#ifndef MOD_CL_PATH
#error Kernel files base path not defined; undefined `MOD_CL_PATH`
#endif //MOD_CL_PATH

// Note: gcc throws stray issue without this preprocessor derivative being used
//       for reading MOD_CL_PATH used for reading kernel file
//#define TO_STRING(x) #x

cl_int
cl_kernel_initializer(rpp::Handle &handle,
                      std::string kernelFile, std::string kernelName,
                      cl_program &theProgram, cl_kernel &theKernel)
{
    cl_int err;
    // File Handling
    char *sourceStr;
    size_t sourceSize;
    std::string kernelFile_cl = MOD_CL_PATH + kernelFile;
    FILE *filePtr = fopen(kernelFile_cl.c_str(), "rb");
    if (!filePtr)
    {
        fprintf(stderr, "Failed to load kernel.\n");
        return 1;
    }
    fseek(filePtr, 0, SEEK_END);
    size_t fileSize = ftell(filePtr);
    rewind(filePtr);
    sourceStr = (char *)malloc(fileSize + 1);
    sourceStr[fileSize] = '\0';
    fread(sourceStr, sizeof(char), fileSize, filePtr);
    fclose(filePtr);

    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(),
                          CL_QUEUE_CONTEXT,
                          sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(),
                          CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);

    theProgram = clCreateProgramWithSource(theContext, 1,
                                           (const char **)&sourceStr, NULL, &err);

    clBuildProgram(theProgram, 0, NULL, NULL, NULL, NULL);
    // theKernel = clCreateKernel(theProgram, kernelName.c_str(), &err);

    return err;
}

cl_int
cl_kernel_implementer(rpp::Handle &handle, size_t *globalDim3, size_t *localDim3,
                      cl_program &theProgram, cl_kernel &theKernel)
{
    cl_int err;
    // std::cerr<<"\n coming here in cl_utilities.cpp";
    err = clEnqueueNDRangeKernel(handle.GetStream(), theKernel, 3, NULL, globalDim3, localDim3,
                                 0, NULL, NULL);
    //clFinish(handle.GetStream());
    //clReleaseProgram(theProgram);
    //clReleaseKernel(theKernel);
    return err;
}

//
///
//  Retreive program binary for all of the devices attached to the
//  program an and store the one for the device passed in
//
bool SaveProgramBinary(cl_program program, cl_device_id device, const std::string fileName)
{
    cl_uint numDevices = 0;
    cl_int errNum;

    // 1 - Query for number of devices attached to program
    errNum = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint),
                              &numDevices, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for number of devices." << std::endl;
        return false;
    }

    // 2 - Get all of the Device IDs
    cl_device_id *devices = new cl_device_id[numDevices];
    errNum = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                              sizeof(cl_device_id) * numDevices,
                              devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for devices." << std::endl;
        delete[] devices;
        return false;
    }

    // 3 - Determine the size of each program binary
    size_t *programBinarySizes = new size_t[numDevices];
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                              sizeof(size_t) * numDevices,
                              programBinarySizes, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for program binary sizes." << std::endl;
        delete[] devices;
        delete[] programBinarySizes;
        return false;
    }

    unsigned char **programBinaries = new unsigned char *[numDevices];
    for (cl_uint i = 0; i < numDevices; i++)
    {
        programBinaries[i] = new unsigned char[programBinarySizes[i]];
    }

    // 4 - Get all of the program binaries
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *) * numDevices,
                              programBinaries, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for program binaries" << std::endl;

        delete[] devices;
        delete[] programBinarySizes;
        for (cl_uint i = 0; i < numDevices; i++)
        {
            delete[] programBinaries[i];
        }
        delete[] programBinaries;
        return false;
    }

    // 5 - Finally store the binaries for the device requested out to disk for future reading.
    for (cl_uint i = 0; i < numDevices; i++)
    {
        // Store the binary just for the device requested.  In a scenario where
        // multiple devices were being used you would save all of the binaries out here.
        if (devices[i] == device)
        {
            FILE *fp = fopen(fileName.c_str(), "wb");
            fwrite(programBinaries[i], 1, programBinarySizes[i], fp);
            fclose(fp);
            break;
        }
    }

    // Cleanup
    delete[] devices;
    delete[] programBinarySizes;
    for (cl_uint i = 0; i < numDevices; i++)
    {
        delete[] programBinaries[i];
    }
    delete[] programBinaries;
    return true;
}

///
//  Attempt to create the program object from a cached binary.  Note that
//  on first run this will fail because the binary has not yet been created.
//
cl_int CreateProgramFromBinary(rpp::Handle &handle, const std::string kernelFile,
                               const std::string binaryFile, std::string kernelName,
                               cl_program &theProgram, cl_kernel &theKernel)
{
    cl_int err;
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(),
                          CL_QUEUE_CONTEXT,
                          sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(),
                          CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    FILE *fp = fopen(binaryFile.c_str(), "rb");
    if (fp == NULL)
    {
        theProgram = NULL;
    }
    else
    {
        // Determine the size of the binary
        size_t binarySize;
        fseek(fp, 0, SEEK_END);
        binarySize = ftell(fp);
        rewind(fp);

        unsigned char *programBinary = new unsigned char[binarySize];
        fread(programBinary, 1, binarySize, fp);
        fclose(fp);

        cl_int errNum = 0;
        cl_int binaryStatus;
        theProgram = clCreateProgramWithBinary(theContext,
                                               1,
                                               &theDevice,
                                               &binarySize,
                                               (const unsigned char **)&programBinary,
                                               &binaryStatus,
                                               &errNum);
        delete[] programBinary;
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error loading program binary." << std::endl;
            return 0;
        }

        if (binaryStatus != CL_SUCCESS)
        {
            std::cerr << "Invalid binary for device" << std::endl;
            return 0;
        }

        errNum = clBuildProgram(theProgram, 0, NULL, NULL, NULL, NULL);
        if (errNum != CL_SUCCESS)
        {
            // Determine the reason for the error
            char buildLog[16384];
            clGetProgramBuildInfo(theProgram, theDevice, CL_PROGRAM_BUILD_LOG,
                                  sizeof(buildLog), buildLog, NULL);

            std::cerr << "Error in program: " << std::endl;
            std::cerr << buildLog << std::endl;
            clReleaseProgram(theProgram);
            return 0;
        }
    }
    if (theProgram == NULL)
    {
        //std::cout << "Binary not loaded, create from source..." << kernelFile << " "  << kernelName << std::endl;
        err = cl_kernel_initializer(handle,
                                    kernelFile,
                                    kernelName,
                                    theProgram, theKernel);

        //std::cout << "Save program binary for future run..." << std::endl;
        if (SaveProgramBinary(theProgram, theDevice, binaryFile) == false)
        {
            std::cerr << "Failed to write program binary" << std::endl;
            clFinish(handle.GetStream());
            clReleaseProgram(theProgram);
            clReleaseKernel(theKernel);
            return 1;
        }
    }
    else
    {
        //std::cout << "Read program from binary." << std::endl;
    }
    theKernel = clCreateKernel(theProgram, kernelName.c_str(), &err);
    return err;
}

void max_size(Rpp32u *height, Rpp32u *width, unsigned int batch_size, unsigned int *max_height, unsigned int *max_width)
{
    int i;
    *max_height = 0;
    *max_width = 0;
    for (i = 0; i < batch_size; i++)
    {
        if (*max_height < height[i])
            *max_height = height[i];
        if (*max_width < width[i])
            *max_width = width[i];
    }
}

void get_kernel_name(std::string &kernel_name, const RPPTensorFunctionMetaData &tensor_info)
{
    switch (tensor_info._in_type)
    {
    case RPPTensorDataType::U8:
        switch (tensor_info._out_type)
        {
        case RPPTensorDataType::U8:
            break;
        case RPPTensorDataType::FP32:
            kernel_name = kernel_name + "_u8_fp32";
            break;
        case RPPTensorDataType::FP16:
            kernel_name = kernel_name + "_u8_fp16";
            break;
        case RPPTensorDataType::I8:
            kernel_name = kernel_name + "_u8_int8";
            break;
        default:
            break;
        }
        break;
    case RPPTensorDataType::FP32:
        kernel_name = kernel_name + "_fp32";
        break;
    case RPPTensorDataType::FP16:
        kernel_name = kernel_name + "_fp16";
        break;
    case RPPTensorDataType::I8:
        kernel_name = kernel_name + "_int8";
        break;
    default:
        break;
    }
}
