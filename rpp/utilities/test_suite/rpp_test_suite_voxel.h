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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <unistd.h>
#include <dirent.h>
#include <map>
#include <iomanip>
#include "rpp.h"
#include "nifti1.h"

using namespace std;
typedef int16_t NIFTI_DATATYPE;

#define MIN_HEADER_SIZE 348
#define RPPRANGECHECK(value)     (value < -32768) ? -32768 : ((value < 32767) ? value : 32767)
#define DEBUG_MODE 0
#define CUTOFF 1
#define TOLERANCE 0.01
#define MAX_IMAGE_DUMP 100
#define MAX_BATCH_SIZE 512

#define CHECK(x) do { \
  int retval = (x); \
  if (retval != 0) { \
    fprintf(stderr, "Runtime error: %s returned %d at %s:%d", #x, retval, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)

std::map<int, string> augmentationMap =
{
    {0, "fused_multiply_add_scalar"},
    {1, "slice"},
    {2, "add_scalar"},
    {3, "subtract_scalar"},
    {4, "flip_voxel"},
    {5, "multiply_scalar"},
    {6, "gaussian_noise_voxel"}
};

enum Augmentation {
    FUSED_MULTIPLY_ADD_SCALAR = 0,
    SLICE = 1,
    ADD_SCALAR = 2,
    SUBTRACT_SCALAR = 3,
    FLIP_VOXEL = 4,
    MULTIPLY_SCALAR = 5,
    GAUSSIAN_NOISE_VOXEL = 6
};

void replicate_last_file_to_fill_batch(const string& lastFilePath, vector<string>& filePathVector, vector<string>& fileNamesVector, const string& lastFileName, int noOfFiles, int batchCount)
{
    int remainingFiles = batchCount - (noOfFiles % batchCount);
    std::string filePath = lastFilePath;
    std::string fileName = lastFileName;
    if (noOfFiles > 0 && ( noOfFiles < batchCount || noOfFiles % batchCount != 0 ))
    {
        for (int i = 0; i < remainingFiles; i++)
        {
            filePathVector.push_back(filePath);
            fileNamesVector.push_back(fileName);
        }
    }
}

// Opens a folder and recursively search for .nii files
void open_folder(const string& folderPath, vector<string>& niiFileNames, vector<string>& niiFilePath)
{
    auto src_dir = opendir(folderPath.c_str());
    struct dirent* entity;
    std::string fileName = " ";

    if (src_dir == nullptr)
        std::cout << "\n ERROR: Failed opening the directory at " <<folderPath;

    while((entity = readdir(src_dir)) != nullptr)
    {
        string entry_name(entity->d_name);
        if (entry_name == "." || entry_name == "..")
            continue;
        fileName = entity->d_name;
        std::string filePath = folderPath;
        filePath.append("/");
        filePath.append(entity->d_name);
        fs::path pathObj(filePath);
        if(fs::exists(pathObj) && fs::is_directory(pathObj))
            open_folder(filePath, niiFileNames, niiFilePath);

        if (fileName.size() > 4 && fileName.substr(fileName.size() - 4) == ".nii")
        {
            niiFilePath.push_back(filePath);
            niiFileNames.push_back(entity->d_name);
        }
    }
    if(niiFileNames.empty())
        std::cout << "\n Did not load any file from " << folderPath;

    closedir(src_dir);
}

// Searches for .nii files in input folders
void search_nii_files(const string& folder_path, vector<string>& niiFileNames, vector<string>& niiFilePath)
{
    vector<string> entry_list;
    string full_path = folder_path;
    auto sub_dir = opendir(folder_path.c_str());
    if (!sub_dir)
    {
        std::cout << "ERROR: Failed opening the directory at "<< folder_path << std::endl;
        exit(0);
    }

    struct dirent* entity;
    while ((entity = readdir(sub_dir)) != nullptr)
    {
        string entry_name(entity->d_name);
        if (entry_name == "." || entry_name == "..")
            continue;
        entry_list.push_back(entry_name);
    }
    closedir(sub_dir);
    sort(entry_list.begin(), entry_list.end());

    for (unsigned dir_count = 0; dir_count < entry_list.size(); ++dir_count)
    {
        string subfolder_path = full_path + "/" + entry_list[dir_count];
        fs::path pathObj(subfolder_path);
        if (fs::exists(pathObj) && fs::is_regular_file(pathObj))
        {
            // ignore files with extensions .tar, .zip, .7z
            auto file_extension_idx = subfolder_path.find_last_of(".");
            if (file_extension_idx != std::string::npos)
            {
                std::string file_extension = subfolder_path.substr(file_extension_idx+1);
                if ((file_extension == "tar") || (file_extension == "zip") || (file_extension == "7z") || (file_extension == "rar"))
                    continue;
            }
            if (entry_list[dir_count].size() > 4 && entry_list[dir_count].substr(entry_list[dir_count].size() - 4) == ".nii")
            {
                niiFileNames.push_back(entry_list[dir_count]);
                niiFilePath.push_back(subfolder_path);
            }
        }
        else if (fs::exists(pathObj) && fs::is_directory(pathObj))
            open_folder(subfolder_path, niiFileNames, niiFilePath);
    }
}

// sets generic descriptor dimensions and strides of src/dst
inline void set_generic_descriptor(RpptGenericDescPtr descriptorPtr3D, int noOfImages, int maxX, int maxY, int maxZ,
                                  int numChannels, int offsetInBytes, int layoutType, int inputBitDepth)
{
    descriptorPtr3D->numDims = 5;
    descriptorPtr3D->offsetInBytes = offsetInBytes;
    if(inputBitDepth == 0)
        descriptorPtr3D->dataType = RpptDataType::U8;
    else if(inputBitDepth == 2)
        descriptorPtr3D->dataType = RpptDataType::F32;

    if (layoutType == 0)
    {
        descriptorPtr3D->layout = RpptLayout::NDHWC;
        descriptorPtr3D->dims[0] = noOfImages;
        descriptorPtr3D->dims[1] = maxZ;
        descriptorPtr3D->dims[2] = maxY;
        descriptorPtr3D->dims[3] = maxX;
        descriptorPtr3D->dims[4] = numChannels;
    }
    else if (layoutType == 1 || layoutType == 2)
    {
        descriptorPtr3D->layout = RpptLayout::NCDHW;
        descriptorPtr3D->dims[0] = noOfImages;
        descriptorPtr3D->dims[1] = numChannels;
        descriptorPtr3D->dims[2] = maxZ;
        descriptorPtr3D->dims[3] = maxY;
        descriptorPtr3D->dims[4] = maxX;
    }

    descriptorPtr3D->strides[0] = descriptorPtr3D->dims[1] * descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[1] = descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[2] = descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[3] = descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[4] = 1;
}

//returns function type
inline string set_function_type(int layoutType, int pln1OutTypeCase, int outputFormatToggle, string backend)
{
    string funcType;
    if(layoutType == 0)
    {
        funcType = "Tensor_" + backend + "_PKD3";
        if (pln1OutTypeCase)
            funcType += "_toPLN1";
        else
        {
            if (outputFormatToggle)
                funcType += "_toPLN3";
            else
                funcType += "_toPKD3";
        }
    }
    else if (layoutType == 1)
    {
        funcType = "Tensor_" + backend + "_PLN3";
        if (pln1OutTypeCase)
            funcType += "_toPLN1";
        else
        {
            if (outputFormatToggle)
                funcType += "_toPKD3";
            else
                funcType += "_toPLN3";
        }
    }
    else
    {
       funcType = "Tensor_" + backend + "_PLN1";
       funcType += "_toPLN1";
    }

    return funcType;
}

// initialize the roi, anchor and shape values required for slice
void init_slice_voxel(RpptGenericDescPtr descriptorPtr3D, RpptROI3D *roiGenericSrcPtr, Rpp32u *roiTensor, Rpp32s *anchorTensor, Rpp32s *shapeTensor)
{
    if (descriptorPtr3D->layout == RpptLayout::NCDHW)
    {
        for(int i = 0; i < descriptorPtr3D->dims[0]; i++)
        {
            int idx1 = i * 4;
            int idx2 = i * 8;
            roiTensor[idx2] = anchorTensor[idx1] = 0;
            roiTensor[idx2 + 1] = anchorTensor[idx1 + 1] = roiGenericSrcPtr[i].xyzwhdROI.xyz.z;
            roiTensor[idx2 + 2] = anchorTensor[idx1 + 2] = roiGenericSrcPtr[i].xyzwhdROI.xyz.y;
            roiTensor[idx2 + 3] = anchorTensor[idx1 + 3] = roiGenericSrcPtr[i].xyzwhdROI.xyz.x;
            roiTensor[idx2 + 4] = descriptorPtr3D->dims[1];
            roiTensor[idx2 + 5] = roiGenericSrcPtr[i].xyzwhdROI.roiDepth;
            roiTensor[idx2 + 6] = roiGenericSrcPtr[i].xyzwhdROI.roiHeight;
            roiTensor[idx2 + 7] = roiGenericSrcPtr[i].xyzwhdROI.roiWidth;
            shapeTensor[idx1] = roiTensor[idx2 + 4];
            shapeTensor[idx1 + 1] = roiTensor[idx2 + 5] / 2;
            shapeTensor[idx1 + 2] = roiTensor[idx2 + 6] / 2;
            shapeTensor[idx1 + 3] = roiTensor[idx2 + 7] / 2;
        }
    }
    else if(descriptorPtr3D->layout == RpptLayout::NDHWC)
    {
        for(int i = 0; i < descriptorPtr3D->dims[0]; i++)
        {
            int idx1 = i * 4;
            int idx2 = i * 8;
            roiTensor[idx2] = anchorTensor[idx1] = roiGenericSrcPtr[i].xyzwhdROI.xyz.z;
            roiTensor[idx2 + 1] = anchorTensor[idx1 + 1] = roiGenericSrcPtr[i].xyzwhdROI.xyz.y;
            roiTensor[idx2 + 2] = anchorTensor[idx1 + 2] = roiGenericSrcPtr[i].xyzwhdROI.xyz.x;
            roiTensor[idx2 + 3] = anchorTensor[idx1 + 3] = 0;
            roiTensor[idx2 + 4] = roiGenericSrcPtr[i].xyzwhdROI.roiDepth;
            roiTensor[idx2 + 5] = roiGenericSrcPtr[i].xyzwhdROI.roiHeight;
            roiTensor[idx2 + 6] = roiGenericSrcPtr[i].xyzwhdROI.roiWidth;
            roiTensor[idx2 + 7] = descriptorPtr3D->dims[4];
            shapeTensor[idx1] = roiTensor[idx2 + 4] / 2;
            shapeTensor[idx1 + 1] = roiTensor[idx2 + 5] / 2;
            shapeTensor[idx1 + 2] = roiTensor[idx2 + 6] / 2;
            shapeTensor[idx1 + 3] = roiTensor[idx2 + 7];
        }
    }
}

// reads nifti-1 header file
static int read_nifti_header_file(char* const header_file, nifti_1_header &niftiHeader)
{
    nifti_1_header hdr;

    // open and read header
    FILE *fp = fopen(header_file,"r");
    if (fp == NULL)
    {
        fprintf(stdout, "\nError opening header file %s\n", header_file);
        exit(1);
    }
    int ret = fread(&hdr, MIN_HEADER_SIZE, 1, fp);
    if (ret != 1)
    {
        fprintf(stdout, "\nError reading header file %s\n", header_file);
        exit(1);
    }
    fclose(fp);

    // print header information
    fprintf(stdout, "\n%s header information:", header_file);
    fprintf(stdout, "\nNIFTI1 XYZT dimensions: %d %d %d %d", hdr.dim[1], hdr.dim[2], hdr.dim[3], hdr.dim[4]);
    fprintf(stdout, "\nNIFTI1 Datatype code and bits/pixel: %d %d", hdr.datatype, hdr.bitpix);
    fprintf(stdout, "\nNIFTI1 Scaling slope and intercept: %.6f %.6f", hdr.scl_slope, hdr.scl_inter);
    fprintf(stdout, "\nNIFTI1 Byte offset to data in datafile: %ld", (long)(hdr.vox_offset));
    fprintf(stdout, "\n");

    niftiHeader = hdr;

    return(0);
}

// reads nifti-1 data file
inline void read_nifti_data_file(char* const data_file, nifti_1_header *niftiHeader, NIFTI_DATATYPE *data)
{
    nifti_1_header hdr = *niftiHeader;
    int ret;

    // open the datafile, jump to data offset
    FILE *fp = fopen(data_file, "r");
    if (fp == NULL)
    {
        fprintf(stdout, "\nError opening data file %s\n", data_file);
        exit(1);
    }
    ret = fseek(fp, (long)(hdr.vox_offset), SEEK_SET);
    if (ret != 0)
    {
        fprintf(stdout, "\nError doing fseek() to %ld in data file %s\n", (long)(hdr.vox_offset), data_file);
        exit(1);
    }

    ret = fread(data, sizeof(NIFTI_DATATYPE), hdr.dim[1] * hdr.dim[2] * hdr.dim[3], fp);
    if (ret != hdr.dim[1] * hdr.dim[2] * hdr.dim[3])
    {
        fprintf(stdout, "\nError reading volume 1 from %s (%d)\n", data_file, ret);
        exit(1);
    }
    fclose(fp);
}

inline void write_nifti_file(nifti_1_header *niftiHeader, NIFTI_DATATYPE *niftiData, int batchCount, int chn, string dstPath, string func)
{
    nifti_1_header hdr = *niftiHeader;
    //nifti1_extender pad = {0,0,0,0};
    FILE *fp;
    int ret, i;

    // write first hdr.vox_offset bytes of header
    string niiOutputString = dstPath + "/" + std::to_string(batchCount) + "_" + func + "_chn_" + std::to_string(chn)+"_nifti_output.nii";
    const char *niiOutputFile = niiOutputString.c_str();
    fp = fopen(niiOutputFile,"w");
    if (fp == NULL)
    {
        fprintf(stdout, "\nError opening header file %s for write\n",niiOutputFile);
        exit(1);
    }
    ret = fwrite(&hdr, hdr.vox_offset, 1, fp);
    if (ret != 1)
    {
        fprintf(stdout, "\nError writing header file %s\n",niiOutputFile);
        exit(1);
    }

    // for nii files, write extender pad and image data
    //ret = fwrite(&pad, 4, 1, fp);
    if (ret != 1)
    {
        fprintf(stdout, "\nError writing header file extension pad %s\n",niiOutputFile);
        exit(1);
    }

    ret = fwrite(niftiData, (size_t)(hdr.bitpix/8), hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4], fp);
    if (ret != hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4])
    {
        fprintf(stdout, "\nError writing data to %s\n",niiOutputFile);
        exit(1);
    }

    fclose(fp);
}

inline void write_image_from_nifti_opencv(uchar *niftiDataXYFrameU8, int niftiHeaderImageWidth, RpptRoiXyzwhd *roiGenericSrcPtr, uchar *outputBufferOpenCV, int zPlane, int Channel, int batchCount, string dst_path, string func, int index)
{
    static int imageCount = 0;
    if (imageCount > MAX_IMAGE_DUMP)
        exit(0);
    uchar *outputBufferOpenCVRow = outputBufferOpenCV;
    uchar *niftiDataXYFrameU8Row = niftiDataXYFrameU8;
    for(int i = 0; i < roiGenericSrcPtr[batchCount].roiHeight; i++)
    {
        memcpy(outputBufferOpenCVRow, niftiDataXYFrameU8Row, roiGenericSrcPtr[batchCount].roiWidth);
        outputBufferOpenCVRow += roiGenericSrcPtr[batchCount].roiWidth;
        niftiDataXYFrameU8Row += niftiHeaderImageWidth;
    }
    cv::Mat matOutputImage = cv::Mat(roiGenericSrcPtr[batchCount].roiHeight, roiGenericSrcPtr[batchCount].roiWidth, CV_8UC1, outputBufferOpenCV);
    string fileName = dst_path + "/" + func +"_nifti_" + std::to_string(index) + "_zPlane_chn_"+ std::to_string(Channel) + "_" + std::to_string(zPlane) + ".jpg";
    cv::imwrite(fileName, matOutputImage);
    imageCount++;
}

// Convert default NIFTI_DATATYPE unstrided buffer to RpptDataType::F32 strided buffer
template<typename T>
inline void convert_input_niftitype_to_Rpp32f_generic(T **niftyInput, nifti_1_header headerData[], Rpp32f *inputF32, RpptGenericDescPtr descriptorPtr3D)
{
    bool replicateToAllChannels = false;
    Rpp32u depthStride, rowStride, channelStride, channelIncrement;
    if (descriptorPtr3D->layout == RpptLayout::NCDHW)
    {
        depthStride = descriptorPtr3D->strides[2];
        rowStride = descriptorPtr3D->strides[3];
        channelStride = descriptorPtr3D->strides[1];
        channelIncrement = 1;
        if(descriptorPtr3D->dims[1] == 3)
            replicateToAllChannels = true;                            //temporary chnage to replicate the data for pln3 using pln1 data
    }
    else if (descriptorPtr3D->layout == RpptLayout::NDHWC)
    {
        depthStride = descriptorPtr3D->strides[1];
        rowStride = descriptorPtr3D->strides[2];
        channelStride = 1;
        channelIncrement = 3;
        replicateToAllChannels = true;
    }
    if (replicateToAllChannels)
    {
        for (int batchcount = 0; batchcount < descriptorPtr3D->dims[0]; batchcount++)
        {
            T *niftyInputTemp = niftyInput[batchcount];
            Rpp32f *outputF32Temp = inputF32 + batchcount * descriptorPtr3D->strides[0];
            Rpp32f *outputChannelR = outputF32Temp;
            Rpp32f *outputChannelG = outputChannelR + channelStride;
            Rpp32f *outputChannelB = outputChannelG + channelStride;
            for (int d = 0; d < headerData[batchcount].dim[3]; d++)
            {
                Rpp32f *outputDepthR = outputChannelR;
                Rpp32f *outputDepthG = outputChannelG;
                Rpp32f *outputDepthB = outputChannelB;
                for (int h = 0; h < headerData[batchcount].dim[2]; h++)
                {
                    Rpp32f *outputRowR = outputDepthR;
                    Rpp32f *outputRowG = outputDepthG;
                    Rpp32f *outputRowB = outputDepthB;
                    for (int w = 0; w < headerData[batchcount].dim[1]; w++)
                    {
                        *outputRowR = static_cast<Rpp32f>(*niftyInputTemp);
                        *outputRowG = static_cast<Rpp32f>(*niftyInputTemp);
                        *outputRowB = static_cast<Rpp32f>(*niftyInputTemp);

                        niftyInputTemp++;
                        outputRowR += channelIncrement;
                        outputRowG += channelIncrement;
                        outputRowB += channelIncrement;
                    }
                    outputDepthR += rowStride;
                    outputDepthG += rowStride;
                    outputDepthB += rowStride;
                }
                outputChannelR += depthStride;
                outputChannelG += depthStride;
                outputChannelB += depthStride;
            }
        }
    }
    else
    {
        for (int batchcount = 0; batchcount < descriptorPtr3D->dims[0]; batchcount++)
        {
            T *niftyInputTemp = niftyInput[batchcount];
            Rpp32f *outputTemp = inputF32 + batchcount * descriptorPtr3D->strides[0];
            for (int c = 0; c < headerData[batchcount].dim[4]; c++)
            {
                Rpp32f *outputChannel = outputTemp;
                for (int d = 0; d < headerData[batchcount].dim[3]; d++)
                {
                    Rpp32f *outputDepth = outputChannel;
                    for (int h = 0; h < headerData[batchcount].dim[2]; h++)
                    {
                        Rpp32f *outputRow = outputDepth;
                        for (int w = 0; w < headerData[batchcount].dim[1]; w++)
                        {
                            *outputRow++ = static_cast<Rpp32f>(*niftyInputTemp++);
                        }
                        outputDepth += rowStride;
                    }
                    outputChannel += depthStride;
                }
                outputTemp += channelStride;
            }
        }
    }
}

// Convert RpptDataType::F32 strided buffer to default NIFTI_DATATYPE unstrided buffer
template<typename T>
inline void convert_output_Rpp32f_to_niftitype_generic(Rpp32f *input, RpptGenericDescPtr descriptorPtr3D, T *niftyOutput, nifti_1_header *niftiHeader)
{
    nifti_1_header headerData = *niftiHeader;
    Rpp32u niftyStride = headerData.dim[1] * headerData.dim[2] * headerData.dim[3];
    if (descriptorPtr3D->layout == RpptLayout::NCDHW)
    {
        niftyStride = niftyStride * descriptorPtr3D->dims[1];
        Rpp32f *inputTemp = input;
        T *niftyOutputTemp = niftyOutput;
        for (int d = 0; d < headerData.dim[3]; d++)
        {
            Rpp32f *inputDepth = inputTemp;
            for (int h = 0; h < headerData.dim[2]; h++)
            {
                Rpp32f *inputRow = inputDepth;
                for (int w = 0; w < headerData.dim[1]; w++)
                {
                    *inputRow = RPPRANGECHECK(*inputRow);
                    *niftyOutputTemp++ = (T)*inputRow++;
                }
                inputDepth += descriptorPtr3D->strides[3];
            }
            inputTemp += descriptorPtr3D->strides[2];
        }
    }
    else if (descriptorPtr3D->layout == RpptLayout::NDHWC)
    {
        niftyStride = niftyStride * descriptorPtr3D->dims[4];
        Rpp32f *inputTemp = input;
        T *niftyOutputTemp = niftyOutput;
        for (int d = 0; d < headerData.dim[3]; d++)
        {
            Rpp32f *inputDepth = inputTemp;
            for (int h = 0; h < headerData.dim[2]; h++)
            {
                Rpp32f *inputRow = inputDepth;
                for (int w = 0; w < headerData.dim[1]; w++)
                {
                    *inputRow = RPPRANGECHECK(*inputRow);
                    *niftyOutputTemp = (T)*inputRow;

                    inputRow += 3;
                    niftyOutputTemp++;
                }
                inputDepth += descriptorPtr3D->strides[2];
            }
            inputTemp += descriptorPtr3D->strides[1];
        }
    }
}

// read nifti data file
inline void read_nifti_data(vector<string>::const_iterator dataFilePathStart, vector<string>::const_iterator dataFilePathEnd, NIFTI_DATATYPE** niftiDataArray, nifti_1_header* niftiHeader)
{
    int i = 0;
    for ( ; dataFilePathStart != dataFilePathEnd; ++dataFilePathStart, i++)
    {
        const string& dataFilePath = *dataFilePathStart;
        uint dataSize = niftiHeader[i].dim[1] * niftiHeader[i].dim[2] * niftiHeader[i].dim[3];
        uint dataSizeInBytes = dataSize * sizeof(NIFTI_DATATYPE);
        niftiDataArray[i] = (NIFTI_DATATYPE *) calloc(dataSizeInBytes, 1);
        if (niftiDataArray[i] == NULL)
        {
            fprintf(stdout, "\nError allocating data buffer for %s\n", dataFilePath.c_str());
            exit(1);
        }
        // read nifti data file
        read_nifti_data_file((char *)dataFilePath.c_str(), &niftiHeader[i], niftiDataArray[i]);
    }
}

// compares the output of PKD3-PKD3 variants
void compare_outputs_pkd(Rpp32f* output, Rpp32f* refOutput, int &fileMatch, RpptGenericDescPtr descriptorPtr3D, RpptRoiXyzwhd *roiGenericSrcPtr)
{
    Rpp32f *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef, *depthTemp, *depthTempRef;
    for(int cnt = 0; cnt < descriptorPtr3D->dims[0]; cnt++)
    {
        outputTemp = output + cnt * descriptorPtr3D->strides[0];
        outputTempRef = refOutput + cnt * descriptorPtr3D->strides[0];
        int height = roiGenericSrcPtr[cnt].roiHeight;
        int width = roiGenericSrcPtr[cnt].roiWidth * descriptorPtr3D->dims[4];
        int depth = roiGenericSrcPtr[cnt].roiDepth;
        int depthStride = descriptorPtr3D->strides[1];
        int rowStride = descriptorPtr3D->strides[2];
        int matchedIdx = 0;
        for(int d = 0; d < depth; d++)
        {
            depthTemp = outputTemp + d * depthStride;
            depthTempRef = outputTempRef + d * depthStride;
            for(int i = 0; i < height; i++)
            {
                rowTemp = depthTemp + i * rowStride;
                rowTempRef = depthTempRef + i * rowStride;
                for(int j = 0; j < width; j++)
                {
                    outVal = rowTemp + j;
                    outRefVal = rowTempRef + j;
                    int diff = abs(*outVal - *outRefVal);
                    if(diff <= CUTOFF)
                        matchedIdx++;
                }
            }
        }
        if(matchedIdx >= (1 - TOLERANCE) * (height * width * depth) && matchedIdx !=0)
            fileMatch++;
    }
}

// compares the output of PLN3-PLN3 variants
void compare_outputs_pln3(Rpp32f* output, Rpp32f* refOutput, int &fileMatch, RpptGenericDescPtr descriptorPtr3D, RpptRoiXyzwhd *roiGenericSrcPtr)
{
    Rpp32f *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef, *outputTempChn, *outputTempRefChn, *depthTemp, *depthTempRef;
    for(int cnt = 0; cnt < descriptorPtr3D->dims[0]; cnt++)
    {
        outputTemp = output + cnt * descriptorPtr3D->strides[0];
        outputTempRef = refOutput + cnt * descriptorPtr3D->strides[0];
        int height = roiGenericSrcPtr[cnt].roiHeight;
        int width = roiGenericSrcPtr[cnt].roiWidth;
        int depth = roiGenericSrcPtr[cnt].roiDepth;
        int depthStride = descriptorPtr3D->strides[2];
        int refDepthStride = descriptorPtr3D->strides[2] * descriptorPtr3D->dims[1];
        int rowStride = descriptorPtr3D->strides[3];
        int refRowStride = descriptorPtr3D->strides[3] * 3;
        int channelStride = descriptorPtr3D->strides[1];
        int matchedIdx = 0;

        for(int c = 0; c < descriptorPtr3D->dims[1]; c++)
        {
            outputTempChn = outputTemp + c * channelStride;
            outputTempRefChn = outputTempRef + c;
            for(int d = 0; d < depth; d++)
            {
                depthTemp = outputTempChn + d * depthStride;
                depthTempRef = outputTempRefChn + d * refDepthStride;
                for(int i = 0; i < height; i++)
                {
                    rowTemp = depthTemp + i * rowStride;
                    rowTempRef = depthTempRef + i * refRowStride;
                    for(int j = 0; j < width; j++)
                    {
                        outVal = rowTemp + j;
                        outRefVal = rowTempRef + j * 3 ;
                        int diff = abs(*outVal - *outRefVal);
                        if(diff <= CUTOFF)
                            matchedIdx++;
                    }
                }
            }
        }
        if(matchedIdx >= (1 - TOLERANCE) * (height * width * descriptorPtr3D->dims[1] * depth) && matchedIdx !=0)
            fileMatch++;
    }
}

// compares the output of PLN1-PLN1 variants
void compare_outputs_pln1(Rpp32f* output, Rpp32f* refOutput, int &fileMatch, RpptGenericDescPtr descriptorPtr3D, RpptRoiXyzwhd *roiGenericSrcPtr)
{
    Rpp32f *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef, *outputTempChn, *outputTempRefChn, *depthTemp, *depthTempRef;
    for(int cnt = 0; cnt < descriptorPtr3D->dims[0]; cnt++)
    {
        outputTemp = output + cnt * descriptorPtr3D->strides[0];
        outputTempRef = refOutput + cnt * descriptorPtr3D->strides[0];
        int height = roiGenericSrcPtr[cnt].roiHeight;
        int width = roiGenericSrcPtr[cnt].roiWidth;
        int depth = roiGenericSrcPtr[cnt].roiDepth;
        int depthStride = descriptorPtr3D->strides[2];
        int rowStride = descriptorPtr3D->strides[3];
        int channelStride = descriptorPtr3D->strides[1];
        int matchedIdx = 0;

        outputTempChn = outputTemp;
        outputTempRefChn = outputTempRef;
        for(int d = 0; d < depth; d++)
        {
            depthTemp = outputTempChn + d * depthStride;
            depthTempRef = outputTempRefChn + d * depthStride;
            for(int i = 0; i < height; i++)
            {
                rowTemp = depthTemp + i * rowStride;
                rowTempRef = depthTempRef + i * rowStride;
                for(int j = 0; j < width; j++)
                {
                    outVal = rowTemp + j;
                    outRefVal = rowTempRef + j ;
                    int diff = abs(*outVal - *outRefVal);
                    if(diff <= CUTOFF)
                        matchedIdx++;
                }
            }
        }
        if(matchedIdx >= (1 - TOLERANCE) * (height * width * descriptorPtr3D->dims[1] * depth) && matchedIdx !=0)
            fileMatch++;
    }
}

inline void compare_output(Rpp32f* output, Rpp64u oBufferSize, string func, int layoutType, RpptGenericDescPtr descriptorPtr3D, RpptRoiXyzwhd *roiGenericSrcPtr, string dst, string scriptPath)
{
    string binName = "";
    binName = func + "_nifti_output.bin";
    int pln1RefStride = descriptorPtr3D->strides[1] * descriptorPtr3D->dims[0] * 3;
    int binOutputSize = descriptorPtr3D->strides[0] * 4;
    binOutputSize = (layoutType == 2)? binOutputSize * 3 : binOutputSize;

    string refFile = scriptPath + "/../REFERENCE_OUTPUT_VOXEL/"+ func + "/" + binName;

    string line,word;
    int index = 0;
    int mismatches = 0;
    float *refOutput = (float *)malloc(binOutputSize * sizeof(float));

    FILE *fp;
    fp = fopen(refFile.c_str(), "rb");
    if (fp == NULL)
        std::cout << "Error opening file";

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    if (fsize == 0)
        std::cerr << "File is empty";

    fseek(fp, 0, SEEK_SET);
    fread(refOutput, fsize, 1, fp);
    fclose(fp);

    int fileMatch = 0;
    if(layoutType == 0)
        compare_outputs_pkd(output, refOutput, fileMatch, descriptorPtr3D, roiGenericSrcPtr);
    else if(layoutType == 1)
        compare_outputs_pln3(output, refOutput, fileMatch, descriptorPtr3D, roiGenericSrcPtr);
    else
        compare_outputs_pln1(output, refOutput + pln1RefStride, fileMatch, descriptorPtr3D, roiGenericSrcPtr);

    std::cout << std::endl << "Results for " << func << " :" << std::endl;
    if(descriptorPtr3D->layout == RpptLayout::NDHWC)
        func += "_Tensor_PKD3";
    else
    {
        if (descriptorPtr3D->dims[1] == 3)
            func += "_Tensor_PLN3";
        else
            func += "_Tensor_PLN1";
    }

    std::string status = func + ": ";
    if(fileMatch == descriptorPtr3D->dims[0])
    {
        std::cout << "PASSED!" << std::endl;
        status += "PASSED";
    }
    else
    {
        std::cout << "FAILED! " << fileMatch << "/" << descriptorPtr3D->dims[0] << " outputs are matching with reference outputs" << std::endl;
        status += "FAILED";
    }

    // Append the QA results to file
    std::string qaResultsPath = dst + "/QA_results.txt";
    std::ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open())
    {
        qaResults << status << std::endl;
        qaResults.close();
    }
}
