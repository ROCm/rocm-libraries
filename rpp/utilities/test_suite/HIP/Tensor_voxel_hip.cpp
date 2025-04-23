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

#include "../rpp_test_suite_voxel.h"

int main(int argc, char * argv[])
{
    // Handle inputs
    const int MIN_ARG_COUNT = 11;

    int layoutType, testCase, testType, qaFlag, numRuns, batchSize, inputBitDepth;
    char *headerFile, *dataFile, *dstPath;

    if (argc < MIN_ARG_COUNT)
    {
        cout << "\nImproper Usage! Needs all arguments!\n";
        exit(1);
    }

    headerFile = argv[1];
    dataFile = argv[2];
    dstPath = argv[3];
    layoutType = atoi(argv[4]); // 0 for PKD3 // 1 for PLN3 // 2 for PLN1
    testCase = atoi(argv[5]); // 0 to 1
    numRuns = atoi(argv[6]);
    testType = atoi(argv[7]); // 0 - unit test / 1 - performance test
    qaFlag = atoi(argv[8]); //0 - QA disabled / 1 - QA enabled
    batchSize = atoi(argv[9]);
    inputBitDepth = atoi(argv[10]);
    string scriptPath = argv[11];

    bool nonQACase = (testCase == 6);

    if ((layoutType < 0) || (layoutType > 2))
    {
        fprintf(stdout, "\nUsage: %s <header file> <data file> <layoutType = 0 - PKD3/ 1 - PLN3/ 2 - PLN1>\n", argv[0]);
        exit(1);
    }
    if(batchSize > MAX_BATCH_SIZE)
    {
        cout << "\n Batchsize should be less than or equal to "<< MAX_BATCH_SIZE << " Aborting!";
        exit(0);
    }

    string funcName = augmentationMap[testCase];
    if (funcName.empty())
    {
        if (testType == 0)
            cout << "\ncase " << testCase << " is not supported\n";

        return -1;
    }

    int numChannels, offsetInBytes;
    int noOfFiles = 0, maxX = 0, maxY = 0, maxZ = 0;
    vector<string> headerNames, headerPath, dataFileNames, dataFilePath;
    search_nii_files(headerFile, headerNames, headerPath);
    search_nii_files(dataFile, dataFileNames, dataFilePath);
    noOfFiles = dataFileNames.size();
    if(noOfFiles < batchSize || ((noOfFiles % batchSize) != 0))
    {
        replicate_last_file_to_fill_batch(headerPath[noOfFiles - 1], headerPath, headerNames, headerNames[noOfFiles - 1], noOfFiles, batchSize);
        replicate_last_file_to_fill_batch(dataFilePath[noOfFiles - 1], dataFilePath, dataFileNames, dataFileNames[noOfFiles - 1], noOfFiles, batchSize);
        noOfFiles = dataFileNames.size();
    }

    // NIFTI_DATATYPE *niftiData = NULL;
    NIFTI_DATATYPE** niftiDataArray = (NIFTI_DATATYPE**)malloc(batchSize * sizeof(NIFTI_DATATYPE*));
    nifti_1_header* niftiHeader = (nifti_1_header*)malloc(noOfFiles * sizeof(nifti_1_header));

    // read nifti header file
    for(int i = 0; i < noOfFiles; i++)
    {
        read_nifti_header_file((char *)headerPath[i].c_str(), niftiHeader[i]);
        maxX = max(static_cast<int>(niftiHeader[i].dim[1]), maxX);
        maxY = max(static_cast<int>(niftiHeader[i].dim[2]), maxY);
        maxZ = max(static_cast<int>(niftiHeader[i].dim[3]), maxZ);
    }

    // Set ROI tensors types for src
    RpptRoi3DType roiTypeSrc;
    roiTypeSrc = RpptRoi3DType::XYZWHD;

    numChannels = (layoutType == 2) ? 1: 3;                    //Temporary value set to 3 for running pln3, the actual value should be obtained from niftiHeader.dim[4].
    offsetInBytes = 0;

    // optionally set maxX as a multiple of 8 for RPP optimal CPU/GPU processing
    maxX = ((maxX / 8) * 8) + 8;

    // set src/dst generic tensor descriptors
    RpptGenericDesc descriptor3D;
    RpptGenericDescPtr descriptorPtr3D = &descriptor3D;
    set_generic_descriptor(descriptorPtr3D, batchSize, maxX, maxY, maxZ, numChannels, offsetInBytes, layoutType, inputBitDepth);

    // update funcName based on bitdepth and layout
    if(inputBitDepth == 0)
        funcName += "_u8_";
    else if(inputBitDepth == 2)
        funcName += "_f32_";
    int pln1OutTypeCase = 0, outputFormatToggle = 0;
    string funcType = set_function_type(layoutType, pln1OutTypeCase, outputFormatToggle, "HIP");
    funcName += funcType;

    // set src/dst xyzwhd ROI tensors
    void *pinnedMemROI;
    CHECK_RETURN_STATUS(hipHostMalloc(&pinnedMemROI, noOfFiles * sizeof(RpptROI3D)));
    RpptROI3D *roiGenericSrcPtr = reinterpret_cast<RpptROI3D *>(pinnedMemROI);

    // Set buffer sizes in pixels for src/dst
    Rpp64u iBufferSize = (Rpp64u)descriptorPtr3D->strides[0] * (Rpp64u)descriptorPtr3D->dims[0]; //  (d x h x w x c) x (n)
    Rpp64u oBufferSize = iBufferSize;   // User can provide a different oBufferSize

    // Set buffer sizes in bytes for src/dst (including offsets)
    Rpp64u iBufferSizeInBytes = iBufferSize * sizeof(Rpp32f) + descriptorPtr3D->offsetInBytes;
    Rpp64u oBufferSizeInBytes = iBufferSizeInBytes;

    // Allocate host memory in Rpp32f for RPP strided buffer
    Rpp32f *inputF32 = static_cast<Rpp32f *>(calloc(iBufferSizeInBytes, 1));
    Rpp32f *outputF32 = static_cast<Rpp32f *>(calloc(oBufferSizeInBytes, 1));

    // Allocate hip memory in float for RPP strided buffer
    void *d_inputF32, *d_outputF32;
    CHECK_RETURN_STATUS(hipMalloc(&d_inputF32, iBufferSizeInBytes));
    CHECK_RETURN_STATUS(hipMalloc(&d_outputF32, oBufferSizeInBytes));

    // set argument tensors
    void *pinnedMemArgs;
    CHECK_RETURN_STATUS(hipHostMalloc(&pinnedMemArgs, 2 * noOfFiles * sizeof(Rpp32f)));

    // arguments required for slice
    Rpp32s *anchorTensor = NULL, *shapeTensor = NULL;
    Rpp32u *roiTensor = NULL;

    rppHandle_t handle;
    hipStream_t stream;
    CHECK_RETURN_STATUS(hipStreamCreate(&stream));
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
    rppCreate(&handle, batchSize, 0, stream, backend);

    // Run case-wise RPP API and measure time
    int missingFuncFlag = 0;
    double maxWallTime = 0, minWallTime = 5000, avgWallTime = 0, wallTime = 0;
    int noOfIterations = (int)noOfFiles / batchSize;
    string testCaseName;

    Rpp8u *inputU8 = NULL;
    Rpp8u *outputU8 = NULL;
    void *d_inputU8 = NULL, *d_outputU8 = NULL;
    Rpp64u iBufferSizeU8 = iBufferSize * sizeof(Rpp8u) + descriptorPtr3D->offsetInBytes;
    if(inputBitDepth == 0)
    {
        inputU8 = static_cast<Rpp8u *>(calloc(iBufferSizeU8, 1));
        outputU8 = static_cast<Rpp8u *>(calloc(iBufferSizeU8, 1));

        CHECK_RETURN_STATUS(hipMalloc(&d_inputU8, iBufferSizeU8));
        CHECK_RETURN_STATUS(hipMalloc(&d_outputU8, iBufferSizeU8));
    }

    cout << "\nRunning " << funcName << " " << numRuns << " times (each time with a batch size of " << batchSize << " images) and computing mean statistics...";
    for(int iterCount = 0; iterCount < noOfIterations; iterCount++)
    {
        vector<string>::const_iterator dataFilePathStart = dataFilePath.begin() + (iterCount * batchSize);
        vector<string>::const_iterator dataFilePathEnd = dataFilePathStart + batchSize;
        nifti_1_header *niftiHeaderTemp = niftiHeader + batchSize * iterCount;

        read_nifti_data(dataFilePathStart, dataFilePathEnd, niftiDataArray, niftiHeaderTemp);

        // optionally pick full image as ROI or a smaller slice of the 3D tensor in X/Y/Z dimensions
        for(int i = 0; i < batchSize; i++)
        {
            // option 1 - test using roi as the whole 3D image - not sliced (example for 240 x 240 x 155 x 1)
            roiGenericSrcPtr[i].xyzwhdROI.xyz.x = 0;                                    // start X dim = 0
            roiGenericSrcPtr[i].xyzwhdROI.xyz.y = 0;                                    // start Y dim = 0
            roiGenericSrcPtr[i].xyzwhdROI.xyz.z = 0;                                    // start Z dim = 0
            roiGenericSrcPtr[i].xyzwhdROI.roiWidth = niftiHeaderTemp[i].dim[1];         // length in X dim
            roiGenericSrcPtr[i].xyzwhdROI.roiHeight = niftiHeaderTemp[i].dim[2];        // length in Y dim
            roiGenericSrcPtr[i].xyzwhdROI.roiDepth = niftiHeaderTemp[i].dim[3];         // length in Z dim
            // option 2 - test using roi as a smaller 3D tensor slice - sliced in X, Y and Z dims (example for 240 x 240 x 155 x 1)
            // roiGenericSrcPtr[i].xyzwhdROI.xyz.x = niftiHeader.dim[1] / 4;            // start X dim = 60
            // roiGenericSrcPtr[i].xyzwhdROI.xyz.y = niftiHeader[i].dim[2] / 4;         // start Y dim = 60
            // roiGenericSrcPtr[i].xyzwhdROI.xyz.z = niftiHeader[i].dim[3] / 3;         // start Z dim = 51
            // roiGenericSrcPtr[i].xyzwhdROI.roiWidth = niftiHeader[i].dim[1] / 2;      // length in X dim = 120
            // roiGenericSrcPtr[i].xyzwhdROI.roiHeight = niftiHeader[i].dim[2] / 2;     // length in Y dim = 120
            // roiGenericSrcPtr[i].xyzwhdROI.roiDepth = niftiHeader[i].dim[3] / 3;      // length in Z dim = 51
            // option 3 - test using roi as a smaller 3D tensor slice - sliced in only Z dim (example for 240 x 240 x 155 x 1)
            // roiGenericSrcPtr[i].xyzwhdROI.xyz.x = 0;                                 // start X dim = 0
            // roiGenericSrcPtr[i].xyzwhdROI.xyz.y = 0;                                 // start Y dim = 0
            // roiGenericSrcPtr[i].xyzwhdROI.xyz.z = niftiHeader[i].dim[3] / 3;         // start Z dim = 51
            // roiGenericSrcPtr[i].xyzwhdROI.roiWidth = niftiHeader[i].dim[1];          // length in X dim = 240
            // roiGenericSrcPtr[i].xyzwhdROI.roiHeight = niftiHeader[i].dim[2];         // length in Y dim = 240
            // roiGenericSrcPtr[i].xyzwhdROI.roiDepth = niftiHeader[i].dim[3] / 3;      // length in Z dim = 51
            // option 4 - test using roi as a smaller 3D tensor slice - sliced in only X and Z dim (example for 240 x 240 x 155 x 1)
            // roiGenericSrcPtr[i].xyzwhdROI.xyz.x = niftiHeader[i].dim[1] / 5;         // start X dim = 48
            // roiGenericSrcPtr[i].xyzwhdROI.xyz.y = 0;                                 // start Y dim = 0
            // roiGenericSrcPtr[i].xyzwhdROI.xyz.z = niftiHeader[i].dim[3] / 3;         // start Z dim = 51
            // roiGenericSrcPtr[i].xyzwhdROI.roiWidth = niftiHeader[i].dim[1] * 3 / 5;  // length in X dim = 144
            // roiGenericSrcPtr[i].xyzwhdROI.roiHeight = niftiHeader[i].dim[2];         // length in Y dim = 240
            // roiGenericSrcPtr[i].xyzwhdROI.roiDepth = niftiHeader[i].dim[3] / 3;      // length in Z dim = 51
        }

        // Convert default NIFTI_DATATYPE unstrided buffer to RpptDataType::F32 strided buffer
        convert_input_niftitype_to_Rpp32f_generic(niftiDataArray, niftiHeaderTemp, inputF32 , descriptorPtr3D);

        // Typecast input from F32 to U8 if input bitdepth requested is U8
        if (inputBitDepth == 0)
        {
            for(int i = 0; i < iBufferSizeU8; i++)
                inputU8[i] = std::min(std::max(static_cast<unsigned char>(inputF32[i]), static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
            CHECK_RETURN_STATUS(hipMemcpy(d_inputU8, inputU8, iBufferSizeU8, hipMemcpyHostToDevice));
        }

        //Copy input buffer to hip
        CHECK_RETURN_STATUS(hipMemcpy(d_inputF32, inputF32, iBufferSizeInBytes, hipMemcpyHostToDevice));

        for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
        {
            double startWallTime, endWallTime;
            switch (testCase)
            {
                case FUSED_MULTIPLY_ADD_SCALAR:
                {
                    testCaseName = "fused_multiply_add_scalar";
                    Rpp32f *mulTensor = reinterpret_cast<Rpp32f *>(pinnedMemArgs);
                    Rpp32f *addTensor = mulTensor + batchSize;

                    for (int i = 0; i < batchSize; i++)
                    {
                        mulTensor[i] = 80;
                        addTensor[i] = 5;
                    }

                    startWallTime = omp_get_wtime();
                    if(inputBitDepth == 2)
                        rppt_fused_multiply_add_scalar_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, mulTensor, addTensor, roiGenericSrcPtr, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SLICE:
                {
                    testCaseName = "slice";
                    if(anchorTensor == NULL)
                        CHECK_RETURN_STATUS(hipHostMalloc(&anchorTensor, batchSize * 4 * sizeof(Rpp32s)));
                    if(shapeTensor == NULL)
                        CHECK_RETURN_STATUS(hipHostMalloc(&shapeTensor, batchSize * 4 * sizeof(Rpp32s)));
                    if(roiTensor == NULL)
                        CHECK_RETURN_STATUS(hipHostMalloc(&roiTensor, batchSize * 8 * sizeof(Rpp32u)));
                    bool enablePadding = false;
                    auto fillValue = 0;
                    init_slice_voxel(descriptorPtr3D, roiGenericSrcPtr, roiTensor, anchorTensor, shapeTensor);

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0)
                        rppt_slice_gpu(d_inputU8, descriptorPtr3D, d_outputU8, descriptorPtr3D, anchorTensor, shapeTensor, &fillValue, enablePadding, roiTensor, handle);
                    else if(inputBitDepth == 2)
                        rppt_slice_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, anchorTensor, shapeTensor, &fillValue, enablePadding, roiTensor, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case ADD_SCALAR:
                {
                    testCaseName = "add_scalar";
                    Rpp32f addTensor[batchSize];

                    for (int i = 0; i < batchSize; i++)
                        addTensor[i] = 40;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 2)
                        rppt_add_scalar_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, addTensor, roiGenericSrcPtr, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SUBTRACT_SCALAR:
                {
                    testCaseName = "subtract_scalar";
                    Rpp32f subtractTensor[batchSize];

                    for (int i = 0; i < batchSize; i++)
                        subtractTensor[i] = 40;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 2)
                        rppt_subtract_scalar_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, subtractTensor, roiGenericSrcPtr, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case FLIP_VOXEL:
                {
                    testCaseName = "flip_voxel";
                    Rpp32u horizontalTensor[batchSize];
                    Rpp32u verticalTensor[batchSize];
                    Rpp32u depthTensor[batchSize];

                    for (int i = 0; i < batchSize; i++)
                    {
                        horizontalTensor[i] = 1;
                        verticalTensor[i] = 0;
                        depthTensor[i] = 0;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0)
                        rppt_flip_voxel_gpu(d_inputU8, descriptorPtr3D, d_outputU8, descriptorPtr3D, horizontalTensor, verticalTensor, depthTensor, roiGenericSrcPtr, roiTypeSrc, handle);
                    else if(inputBitDepth == 2)
                        rppt_flip_voxel_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, horizontalTensor, verticalTensor, depthTensor, roiGenericSrcPtr, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case MULTIPLY_SCALAR:
                {
                    testCaseName = "multiply_scalar";
                    Rpp32f mulTensor[batchSize];

                    for (int i = 0; i < batchSize; i++)
                        mulTensor[i] = 80;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 2)
                        rppt_multiply_scalar_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, mulTensor, roiGenericSrcPtr, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case GAUSSIAN_NOISE_VOXEL:
                {
                    testCaseName = "gaussian_noise_voxel";
                    Rpp32f *meanTensor = reinterpret_cast<Rpp32f *>(pinnedMemArgs);
                    Rpp32f *stdDevTensor = meanTensor + batchSize;

                    Rpp32u seed = 1255459;
                    for (int i = 0; i < batchSize; i++)
                    {
                        meanTensor[i] = 1.4;
                        stdDevTensor[i] = 0.6;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0)
                        rppt_gaussian_noise_voxel_gpu(d_inputU8, descriptorPtr3D, d_outputU8, descriptorPtr3D, meanTensor, stdDevTensor, seed, roiGenericSrcPtr, roiTypeSrc, handle);
                    else if (inputBitDepth == 2)
                        rppt_gaussian_noise_voxel_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, meanTensor, stdDevTensor, seed, roiGenericSrcPtr, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                default:
                {
                    missingFuncFlag = 1;
                    break;
                }
            }

            CHECK_RETURN_STATUS(hipDeviceSynchronize());
            endWallTime = omp_get_wtime();
            wallTime = endWallTime - startWallTime;
            maxWallTime = std::max(maxWallTime, wallTime);
            minWallTime = std::min(minWallTime, wallTime);
            avgWallTime += wallTime;
        }

        wallTime *= 1000;
        if (missingFuncFlag == 1)
        {
            cout << "\nThe functionality doesn't yet exist in RPP\n";
            return -1;
        }

        // Copy output buffer to host
        CHECK_RETURN_STATUS(hipMemcpy(outputF32, d_outputF32, oBufferSizeInBytes, hipMemcpyDeviceToHost));
        if(testType == 0)
        {
            cout <<"\n\n";
            if(noOfIterations > 1)
                cout <<"Execution Timings for Iteration "<< iterCount+1 <<":"<<endl;
            cout << "GPU Backend Wall Time: " << wallTime <<" ms per batch";
            if(DEBUG_MODE)
            {
                std::ofstream refFile;
                std::string refFileName;
                if(layoutType == 0)
                    refFileName = testCaseName + "_nifti_hip_pkd3.csv";
                else if(layoutType == 1)
                    refFileName = testCaseName + "_nifti_hip_pln3.csv";
                else
                    refFileName = testCaseName + "_nifti_hip_pln1.csv";
                refFile.open(refFileName);
                for (int i = 0; i < oBufferSize; i++)
                    refFile << *(outputF32 + i) << ",";
                refFile.close();
            }

            if(inputBitDepth == 0)
            {
                Rpp64u bufferLength = iBufferSize * sizeof(Rpp8u) + descriptorPtr3D->offsetInBytes;
                CHECK_RETURN_STATUS(hipMemcpy(outputU8, d_outputU8, bufferLength, hipMemcpyDeviceToHost));

                // Copy U8 buffer to F32 buffer for display purposes
                for(int i = 0; i < bufferLength; i++)
                    outputF32[i] = static_cast<float>(outputU8[i]);
            }

            // if test case is slice and qaFlag is set, update the ROI with shapeTensor values
            // for output display and comparison purposes
            if(testCase == SLICE)
            {
                // update the roi for comparision with the shapeTensor values
                if (descriptorPtr3D->layout == RpptLayout::NCDHW)
                {
                    for(int i = 0; i < batchSize; i++)
                    {
                        int idx1 = i * 4;
                        roiGenericSrcPtr[i].xyzwhdROI.xyz.x = 0;
                        roiGenericSrcPtr[i].xyzwhdROI.xyz.y = 0;
                        roiGenericSrcPtr[i].xyzwhdROI.xyz.z = 0;
                        roiGenericSrcPtr[i].xyzwhdROI.roiDepth = shapeTensor[idx1 + 1];
                        roiGenericSrcPtr[i].xyzwhdROI.roiHeight = shapeTensor[idx1 + 2];
                        roiGenericSrcPtr[i].xyzwhdROI.roiWidth = shapeTensor[idx1 + 3];
                    }
                }
                else if(descriptorPtr3D->layout == RpptLayout::NDHWC)
                {
                    for(int i = 0; i < batchSize; i++)
                    {
                        int idx1 = i * 4;
                        roiGenericSrcPtr[i].xyzwhdROI.xyz.x = 0;
                        roiGenericSrcPtr[i].xyzwhdROI.xyz.y = 0;
                        roiGenericSrcPtr[i].xyzwhdROI.xyz.z = 0;
                        roiGenericSrcPtr[i].xyzwhdROI.roiDepth = shapeTensor[idx1];
                        roiGenericSrcPtr[i].xyzwhdROI.roiHeight = shapeTensor[idx1 + 1];
                        roiGenericSrcPtr[i].xyzwhdROI.roiWidth = shapeTensor[idx1 + 2];
                    }
                }
            }

            /*Compare the output of the function with golden outputs only if
            1.QA Flag is set
            2.input bit depth 2 (F32)*/
            if(qaFlag && inputBitDepth == 2  && !(nonQACase))
                compare_output(outputF32, oBufferSize, testCaseName, layoutType, descriptorPtr3D, (RpptRoiXyzwhd *)roiGenericSrcPtr, dstPath, scriptPath);
            else
            {
                for(int batchCount = 0; batchCount < batchSize; batchCount++)
                {
                    int index = iterCount * batchSize + batchCount;
                    Rpp32f *outputTemp = outputF32 + batchCount * descriptorPtr3D->strides[0];
                    for(int i = 0; i < numChannels; i++) // temporary changes to process pln3
                    {
                        int xyFrameSize = niftiHeaderTemp[batchCount].dim[1] * niftiHeaderTemp[batchCount].dim[2];
                        int xyFrameSizeROI = roiGenericSrcPtr[batchCount].xyzwhdROI.roiWidth * roiGenericSrcPtr[batchCount].xyzwhdROI.roiHeight;

                        uint dataSize = niftiHeaderTemp[batchCount].dim[1] * niftiHeaderTemp[batchCount].dim[2] * niftiHeaderTemp[batchCount].dim[3];
                        uchar *niftiDataU8 = (uchar *) malloc(dataSize * sizeof(uchar));
                        uchar *outputBufferOpenCV = (uchar *)calloc(xyFrameSizeROI, sizeof(uchar));

                        // Convert RpptDataType::F32 strided buffer to default NIFTI_DATATYPE unstrided buffer
                        Rpp64u increment;
                        if (descriptorPtr3D->layout == RpptLayout::NCDHW)
                            increment = (Rpp64u)descriptorPtr3D->strides[1];
                        else
                            increment = 1;
                        convert_output_Rpp32f_to_niftitype_generic(outputTemp + i * increment, descriptorPtr3D, niftiDataArray[batchCount], &niftiHeaderTemp[batchCount]);
                        NIFTI_DATATYPE min = niftiDataArray[batchCount][0];
                        NIFTI_DATATYPE max = niftiDataArray[batchCount][0];
                        for (int i = 0; i < dataSize; i++)
                        {
                            min = std::min(min, niftiDataArray[batchCount][i]);
                            max = std::max(max, niftiDataArray[batchCount][i]);
                        }
                        Rpp32f multiplier = 255.0f / (max - min);
                        for (int i = 0; i < dataSize; i++)
                            niftiDataU8[i] = (uchar)((niftiDataArray[batchCount][i] - min) * multiplier);

                        uchar *niftiDataU8Temp = niftiDataU8;
                        for (int zPlane = roiGenericSrcPtr[batchCount].xyzwhdROI.xyz.z; zPlane < roiGenericSrcPtr[batchCount].xyzwhdROI.xyz.z + roiGenericSrcPtr[batchCount].xyzwhdROI.roiDepth; zPlane++)
                        {
                            write_image_from_nifti_opencv(niftiDataU8Temp, niftiHeaderTemp[batchCount].dim[1], (RpptRoiXyzwhd *)roiGenericSrcPtr, outputBufferOpenCV, zPlane, i, batchCount, dstPath, testCaseName, index);
                            niftiDataU8Temp += xyFrameSize;
                        }

                        write_nifti_file(&niftiHeaderTemp[batchCount], niftiDataArray[batchCount], index, i, dstPath, testCaseName);

                        if(i == 0)
                        {
                            std::string command = "convert -delay 10 -loop 0 " + std::string(dstPath) + "/" + testCaseName + "_nifti_" + std::to_string(index) + "_zPlane_chn_0_*.jpg " + std::string(dstPath) + "/" + testCaseName + "_niftiOutput_" + std::to_string(index) + "_chn_" + std::to_string(i) + ".gif";
                            system(command.c_str());
                        }
                        if(i == 1)
                        {
                            std::string command = "convert -delay 10 -loop 0 " + std::string(dstPath) + "/" + testCaseName + "_nifti_" + std::to_string(index) + "_zPlane_chn_1_*.jpg " + std::string(dstPath) + "/" + testCaseName + "_niftiOutput_" + std::to_string(index) + "_chn_" + std::to_string(i) + ".gif";
                            system(command.c_str());
                        }
                        if(i == 2)
                        {
                            std::string command = "convert -delay 10 -loop 0 " + std::string(dstPath) + "/" + testCaseName + "_nifti_" + std::to_string(index) + "_zPlane_chn_2_*.jpg " + std::string(dstPath) + "/" + testCaseName + "_niftiOutput_" + std::to_string(index) + "_chn_" + std::to_string(i) + ".gif";
                            system(command.c_str());
                        }
                        free(niftiDataU8);
                        free(outputBufferOpenCV);
                    }
                }
            }
        }
    }

    if(testType == 1)
    {
        // Display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns * noOfIterations);
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }
    rppDestroy(handle, backend);

    // Free memory
    free(niftiDataArray);
    free(inputF32);
    free(outputF32);
    CHECK_RETURN_STATUS(hipHostFree(pinnedMemROI));
    CHECK_RETURN_STATUS(hipHostFree(pinnedMemArgs));
    CHECK_RETURN_STATUS(hipFree(d_inputF32));
    CHECK_RETURN_STATUS(hipFree(d_outputF32));
    if(anchorTensor != NULL)
        CHECK_RETURN_STATUS(hipHostFree(anchorTensor));
    if(shapeTensor != NULL)
        CHECK_RETURN_STATUS(hipHostFree(shapeTensor));
    if(roiTensor != NULL)
        CHECK_RETURN_STATUS(hipHostFree(roiTensor));
    if(inputBitDepth == 0)
    {
        if(inputU8 != NULL)
            free(inputU8);
        if(outputU8 != NULL)
            free(outputU8);
        if(d_inputU8 != NULL)
            CHECK_RETURN_STATUS(hipFree(d_inputU8));
        if(d_outputU8 != NULL)
            CHECK_RETURN_STATUS(hipFree(d_outputU8));
    }

    return(0);
}