/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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

#include "benchmarks_common.h"

// Global configuration variables (defined here, declared extern in header)
int NUM_RUNS = 100;   // Default number of runs, can be overridden via command line
int NUM_THREADS = 0;  // Will be set at runtime
string GRAY_IMAGE_PATH = DEFAULT_GRAY_IMAGE_PATH;
string RGB_IMAGE_PATH = DEFAULT_RGB_IMAGE_PATH;

// Global vectors to store results (defined here, declared extern in header)
vector<BenchmarkResult> grayscaleResults;
vector<BenchmarkResult> rgbResults;

// Global variables for image metadata
string grayImageSize;
string grayImageDtype;
string rgbImageSize;
string rgbImageDtype;
int grayBatchSize = 0;
int rgbBatchSize = 0;

vector<Mat> loadBatchImages(const string& directory, int& batchSize, int& maxWidth, int& maxHeight,
                            bool isColor) {
    vector<Mat> images;
    DIR* dir;
    struct dirent* entry;

    maxWidth = 0;
    maxHeight = 0;

    if ((dir = opendir(directory.c_str())) == NULL) {
        cerr << "Could not open directory: " << directory << endl;
        return images;
    }

    while ((entry = readdir(dir)) != NULL) {
        string filename = entry->d_name;
        if (filename == "." || filename == "..") continue;

        string ext = filename.substr(filename.find_last_of(".") + 1);
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != "jpg" && ext != "jpeg" && ext != "png" && ext != "bmp" && ext != "tiff")
            continue;

        string filePath = directory + "/" + filename;
        Mat img = imread(filePath, isColor ? IMREAD_COLOR : IMREAD_GRAYSCALE);

        if (!img.empty()) {
            maxWidth = max(maxWidth, img.cols);
            maxHeight = max(maxHeight, img.rows);
            images.push_back(move(img));
        } else {
            cerr << "Warning: Could not read image " << filePath << endl;
        }
    }

    closedir(dir);
    batchSize = images.size();
    return images;
}

// Map to store benchmark times and parameters by operation name
struct BenchmarkData {
    double rppTime;
    double opencvTime;
    string parameters;
    bool rppCalled;
    bool opencvCalled;

    BenchmarkData()
        : rppTime(0), opencvTime(0), parameters(""), rppCalled(false), opencvCalled(false) {}
};

static map<string, BenchmarkData> benchmarkTimes;  // operationName -> data
static string currentOperation;
static bool currentIsColor;

void printResult(const string& name, int batchSize, bool isColor, double totalMs,
                 const string& params) {
    double avgTime = totalMs / NUM_RUNS;
    cout << name << " (Avg per run, " << batchSize << " images, "
         << (isColor ? "RGB" : "Grayscale");
    if (!params.empty()) cout << ", " << params;
    cout << "): " << avgTime << " ms" << endl;

    // Extract operation name
    string opName = name;

    if (opName.find("RPP HOST ") == 0) {
        opName = opName.substr(9);  // Remove "RPP HOST "
        currentOperation = opName;
        currentIsColor = isColor;

        // For operations with variations (Resize, Flip), append parameter suffix to operation name
        string displayName = opName;
        if ((opName == "Resize" || opName == "Flip") && !params.empty()) {
            // Extract type= from params
            size_t typePos = params.find("type=");
            if (typePos != string::npos) {
                size_t endPos = params.find(",", typePos);
                string typeValue = (endPos != string::npos)
                                       ? params.substr(typePos + 5, endPos - typePos - 5)
                                       : params.substr(typePos + 5);
                displayName = opName + "_" + typeValue;
            }
        }

        // Create unique key using operation name + color mode (for matching)
        // For operations like Exposure/SobelFilter that have different param names, we use
        // operation name only
        string key = displayName + "|" + (isColor ? "rgb" : "gray");
        benchmarkTimes[key].rppTime = avgTime;
        benchmarkTimes[key].parameters = params;
        benchmarkTimes[key].rppCalled = true;
    } else if (opName.find("OpenCV ") == 0) {
        opName = opName.substr(7);  // Remove "OpenCV "

        // For operations with variations (Resize, Flip), append parameter suffix to operation name
        string displayName = opName;
        if ((opName == "Resize" || opName == "Flip") && !params.empty()) {
            // Extract type= from params
            size_t typePos = params.find("type=");
            if (typePos != string::npos) {
                size_t endPos = params.find(",", typePos);
                string typeValue = (endPos != string::npos)
                                       ? params.substr(typePos + 5, endPos - typePos - 5)
                                       : params.substr(typePos + 5);
                displayName = opName + "_" + typeValue;
            }
        }

        // Create unique key using operation name + color mode (for matching)
        string key = displayName + "|" + (isColor ? "rgb" : "gray");
        benchmarkTimes[key].opencvTime = avgTime;
        benchmarkTimes[key].opencvCalled = true;

        // If parameters weren't set by RPP (e.g., SobelFilter OpenCV has no params), use OpenCV
        // params
        if (benchmarkTimes[key].parameters.empty() && !params.empty())
            benchmarkTimes[key].parameters = params;

        // After OpenCV result, record the pair
        auto& data = benchmarkTimes[key];
        // Only add if both RPP and OpenCV have been called
        if (data.rppCalled && data.opencvCalled) {
            if (isColor)
                rgbResults.emplace_back(displayName, data.parameters, data.opencvTime, data.rppTime,
                                        rgbImageSize, rgbImageDtype, rgbBatchSize, NUM_RUNS);
            else
                grayscaleResults.emplace_back(displayName, data.parameters, data.opencvTime,
                                              data.rppTime, grayImageSize, grayImageDtype,
                                              grayBatchSize, NUM_RUNS);
        }
    }
}

// Helper to get CPU model information
string getCPUInfo() {
    ifstream cpuinfo("/proc/cpuinfo");
    string line;
    while (getline(cpuinfo, line)) {
        if (line.find("model name") != string::npos) {
            size_t pos = line.find(":");
            if (pos != string::npos) return line.substr(pos + 2);
        }
    }
    return "Unknown CPU";
}

// Helper to get memory information (in GB)
string getMemoryInfo() {
    ifstream meminfo("/proc/meminfo");
    string line;
    while (getline(meminfo, line)) {
        if (line.find("MemTotal") != string::npos) {
            istringstream iss(line);
            string label;
            long memKB;
            iss >> label >> memKB;
            double memGB = memKB / 1024.0 / 1024.0;
            ostringstream oss;
            oss.precision(2);
            oss << fixed << memGB << " GB";
            return oss.str();
        }
    }
    return "Unknown";
}

// Helper to get OS information
string getOSInfo() {
    struct utsname unameData;
    if (uname(&unameData) == 0) {
        ostringstream oss;
        oss << unameData.sysname << " " << unameData.release;
        return oss.str();
    }
    return "Unknown OS";
}

// Helper to get RPP version
string getRPPVersion() {
    ostringstream oss;
    oss << RPP_VERSION_MAJOR << "." << RPP_VERSION_MINOR << "." << RPP_VERSION_PATCH;
    return oss.str();
}

// Helper to get current date and time
string getCurrentDateTime() {
    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    tm local_tm = *localtime(&now_time);

    ostringstream oss;
    oss << put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// Helper to get data type string from OpenCV type
string getDtypeString(int cvType) {
    int depth = cvType & CV_MAT_DEPTH_MASK;

    switch (depth) {
        case CV_8U:
            return "U8";
        case CV_8S:
            return "S8";
        case CV_16U:
            return "U16";
        case CV_16S:
            return "S16";
        case CV_32S:
            return "S32";
        case CV_32F:
            return "F32";
        case CV_64F:
            return "F64";
        default:
            return "Unknown";
    }
}

// Helper to create RPP descriptor from OpenCV Mat
RpptDesc createRppDescriptor(const Mat& img, RpptLayout layout) {
    RpptDesc desc;
    desc.n = 1;
    desc.h = img.rows;
    desc.w = img.cols;
    desc.c = img.channels();
    desc.layout = layout;
    desc.dataType = (img.depth() == CV_32F) ? RpptDataType::F32 : RpptDataType::U8;
    desc.offsetInBytes = 0;
    desc.strides.nStride = desc.h * desc.w * desc.c;
    if (layout == RpptLayout::NHWC) {
        desc.strides.hStride = desc.w * desc.c;
        desc.strides.wStride = desc.c;
        desc.strides.cStride = 1;
    } else {  // NCHW
        desc.strides.cStride = desc.h * desc.w;
        desc.strides.hStride = desc.w;
        desc.strides.wStride = 1;
    }
    return desc;
}

// Helper to create full image ROI
RpptROI createFullImageROI(const Mat& img) {
    RpptROI roi;
    roi.xywhROI.xy.x = 0;
    roi.xywhROI.xy.y = 0;
    roi.xywhROI.roiWidth = img.cols;
    roi.xywhROI.roiHeight = img.rows;
    return roi;
}

// Helper to convert RpptDesc to RpptGenericDesc
RpptGenericDesc toGenericDesc(const RpptDesc& desc) {
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 4;
    genericDesc.offsetInBytes = desc.offsetInBytes;
    genericDesc.dataType = desc.dataType;
    genericDesc.layout = desc.layout;

    // Set dims and strides based on layout
    if (desc.layout == RpptLayout::NHWC) {
        // NHWC: dims = [N, H, W, C]
        genericDesc.dims[0] = desc.n;
        genericDesc.dims[1] = desc.h;
        genericDesc.dims[2] = desc.w;
        genericDesc.dims[3] = desc.c;
        genericDesc.strides[0] = desc.strides.nStride;
        genericDesc.strides[1] = desc.strides.hStride;
        genericDesc.strides[2] = desc.strides.wStride;
        genericDesc.strides[3] = desc.strides.cStride;
    } else {  // NCHW
        // NCHW: dims = [N, C, H, W]
        genericDesc.dims[0] = desc.n;
        genericDesc.dims[1] = desc.c;
        genericDesc.dims[2] = desc.h;
        genericDesc.dims[3] = desc.w;
        genericDesc.strides[0] = desc.strides.nStride;
        genericDesc.strides[1] = desc.strides.cStride;
        genericDesc.strides[2] = desc.strides.hStride;
        genericDesc.strides[3] = desc.strides.wStride;
    }

    return genericDesc;
}

// Helper to create full image ROI3D
RpptROI3D createFullImageROI3D(const Mat& img) {
    RpptROI3D roi3d;
    roi3d.xyzwhdROI.xyz.x = 0;
    roi3d.xyzwhdROI.xyz.y = 0;
    roi3d.xyzwhdROI.xyz.z = 0;
    roi3d.xyzwhdROI.roiWidth = img.cols;
    roi3d.xyzwhdROI.roiHeight = img.rows;
    roi3d.xyzwhdROI.roiDepth = 1;
    return roi3d;
}

// ==================== RPP COLOR AUGMENTATIONS ====================
bool writeResultsToExcel(const string& filename, const vector<BenchmarkResult>& grayResults,
                         const vector<BenchmarkResult>& colorResults) {
    lxw_workbook* workbook = workbook_new(filename.c_str());
    if (!workbook) {
        cerr << "Error: Failed to create Excel workbook: " << filename << endl;
        cerr << "       Possible causes: insufficient permissions, invalid path, or disk full"
             << endl;
        return false;
    }

    // Create formats
    lxw_format* header_format = workbook_add_format(workbook);
    format_set_bold(header_format);
    format_set_bg_color(header_format, 0x4472C4);
    format_set_font_color(header_format, LXW_COLOR_WHITE);
    format_set_align(header_format, LXW_ALIGN_CENTER);

    lxw_format* info_label_format = workbook_add_format(workbook);
    format_set_bold(info_label_format);

    lxw_format* speedup_format = workbook_add_format(workbook);
    format_set_num_format(speedup_format, "0.00\"x\"");

    lxw_format* time_format = workbook_add_format(workbook);
    format_set_num_format(time_format, "0.00");

    // Sheet 1: System Information
    lxw_worksheet* info_sheet = workbook_add_worksheet(workbook, "System Information");

    worksheet_set_column(info_sheet, 0, 0, 25, NULL);
    worksheet_set_column(info_sheet, 1, 1, 40, NULL);

    int row = 0;
    worksheet_write_string(info_sheet, row++, 0, "Parameter", header_format);
    worksheet_write_string(info_sheet, row - 1, 1, "Value", header_format);

    worksheet_write_string(info_sheet, row, 0, "Benchmark Date & Time", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getCurrentDateTime().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "RPP Version", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getRPPVersion().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "OpenCV Version", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, CV_VERSION, NULL);

    worksheet_write_string(info_sheet, row, 0, "Operating System", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getOSInfo().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "CPU", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getCPUInfo().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "Memory", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getMemoryInfo().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "Number of Threads", info_label_format);
    worksheet_write_number(info_sheet, row++, 1, NUM_THREADS, NULL);

    worksheet_write_string(info_sheet, row, 0, "Number of Runs", info_label_format);
    worksheet_write_number(info_sheet, row++, 1, NUM_RUNS, NULL);

    // Sheet 2: Grayscale Results
    lxw_worksheet* gray_sheet = workbook_add_worksheet(workbook, "Grayscale Benchmarks");

    worksheet_set_column(gray_sheet, 0, 0, 30, NULL);
    worksheet_set_column(gray_sheet, 1, 1, 40, NULL);
    worksheet_set_column(gray_sheet, 2, 2, 15, NULL);
    worksheet_set_column(gray_sheet, 3, 3, 12, NULL);
    worksheet_set_column(gray_sheet, 4, 5, 12, NULL);
    worksheet_set_column(gray_sheet, 6, 8, 15, NULL);

    row = 0;
    worksheet_write_string(gray_sheet, row, 0, "Operation", header_format);
    worksheet_write_string(gray_sheet, row, 1, "Parameters", header_format);
    worksheet_write_string(gray_sheet, row, 2, "Image Size", header_format);
    worksheet_write_string(gray_sheet, row, 3, "DType", header_format);
    worksheet_write_string(gray_sheet, row, 4, "Batch Size", header_format);
    worksheet_write_string(gray_sheet, row, 5, "Runs", header_format);
    worksheet_write_string(gray_sheet, row, 6, "OpenCV (avg ms)", header_format);
    worksheet_write_string(gray_sheet, row, 7, "RPP HOST (avg ms)", header_format);
    worksheet_write_string(gray_sheet, row++, 8, "Speedup", header_format);

    for (const auto& result : grayResults) {
        worksheet_write_string(gray_sheet, row, 0, result.operationName.c_str(), NULL);
        worksheet_write_string(gray_sheet, row, 1, result.parameters.c_str(), NULL);
        worksheet_write_string(gray_sheet, row, 2, result.imageSize.c_str(), NULL);
        worksheet_write_string(gray_sheet, row, 3, result.dtype.c_str(), NULL);
        worksheet_write_number(gray_sheet, row, 4, result.batchSize, NULL);
        worksheet_write_number(gray_sheet, row, 5, result.numRuns, NULL);
        worksheet_write_number(gray_sheet, row, 6, result.opencvTime, time_format);
        worksheet_write_number(gray_sheet, row, 7, result.rppTime, time_format);
        worksheet_write_number(gray_sheet, row, 8, result.speedup, speedup_format);
        row++;
    }

    // Sheet 3: RGB Results
    lxw_worksheet* rgb_sheet = workbook_add_worksheet(workbook, "RGB Benchmarks");

    worksheet_set_column(rgb_sheet, 0, 0, 30, NULL);
    worksheet_set_column(rgb_sheet, 1, 1, 40, NULL);
    worksheet_set_column(rgb_sheet, 2, 2, 15, NULL);
    worksheet_set_column(rgb_sheet, 3, 3, 12, NULL);
    worksheet_set_column(rgb_sheet, 4, 5, 12, NULL);
    worksheet_set_column(rgb_sheet, 6, 8, 15, NULL);

    row = 0;
    worksheet_write_string(rgb_sheet, row, 0, "Operation", header_format);
    worksheet_write_string(rgb_sheet, row, 1, "Parameters", header_format);
    worksheet_write_string(rgb_sheet, row, 2, "Image Size", header_format);
    worksheet_write_string(rgb_sheet, row, 3, "DType", header_format);
    worksheet_write_string(rgb_sheet, row, 4, "Batch Size", header_format);
    worksheet_write_string(rgb_sheet, row, 5, "Runs", header_format);
    worksheet_write_string(rgb_sheet, row, 6, "OpenCV (avg ms)", header_format);
    worksheet_write_string(rgb_sheet, row, 7, "RPP HOST (avg ms)", header_format);
    worksheet_write_string(rgb_sheet, row++, 8, "Speedup", header_format);

    for (const auto& result : colorResults) {
        worksheet_write_string(rgb_sheet, row, 0, result.operationName.c_str(), NULL);
        worksheet_write_string(rgb_sheet, row, 1, result.parameters.c_str(), NULL);
        worksheet_write_string(rgb_sheet, row, 2, result.imageSize.c_str(), NULL);
        worksheet_write_string(rgb_sheet, row, 3, result.dtype.c_str(), NULL);
        worksheet_write_number(rgb_sheet, row, 4, result.batchSize, NULL);
        worksheet_write_number(rgb_sheet, row, 5, result.numRuns, NULL);
        worksheet_write_number(rgb_sheet, row, 6, result.opencvTime, time_format);
        worksheet_write_number(rgb_sheet, row, 7, result.rppTime, time_format);
        worksheet_write_number(rgb_sheet, row, 8, result.speedup, speedup_format);
        row++;
    }

    lxw_error error = workbook_close(workbook);
    if (error != LXW_NO_ERROR) {
        cerr << "Error: Failed to close Excel workbook: " << filename << " (Error code: " << error
             << ")" << endl;
        cerr << "       The file may be corrupted or incomplete" << endl;
        return false;
    }

    cout << "\nResults exported successfully to: " << filename << endl;
    return true;
}

// ==================== MAIN ====================

// Helper to initialize RICAP boxes for 4-way cutmix
void init_ricap_boxes(int maxWidth, int maxHeight, int batchSize, Rpp32u* permutationTensor,
                      RpptROI* roiPtrInputCropRegion) {
    // Simple RICAP: divide output into 4 quadrants
    int halfW = maxWidth / 2;
    int halfH = maxHeight / 2;

    roiPtrInputCropRegion[0].xywhROI.xy.x = 0;
    roiPtrInputCropRegion[0].xywhROI.xy.y = 0;
    roiPtrInputCropRegion[0].xywhROI.roiWidth = halfW;
    roiPtrInputCropRegion[0].xywhROI.roiHeight = halfH;

    roiPtrInputCropRegion[1].xywhROI.xy.x = halfW;
    roiPtrInputCropRegion[1].xywhROI.xy.y = 0;
    roiPtrInputCropRegion[1].xywhROI.roiWidth = halfW;
    roiPtrInputCropRegion[1].xywhROI.roiHeight = halfH;

    roiPtrInputCropRegion[2].xywhROI.xy.x = 0;
    roiPtrInputCropRegion[2].xywhROI.xy.y = halfH;
    roiPtrInputCropRegion[2].xywhROI.roiWidth = halfW;
    roiPtrInputCropRegion[2].xywhROI.roiHeight = halfH;

    roiPtrInputCropRegion[3].xywhROI.xy.x = halfW;
    roiPtrInputCropRegion[3].xywhROI.xy.y = halfH;
    roiPtrInputCropRegion[3].xywhROI.roiWidth = halfW;
    roiPtrInputCropRegion[3].xywhROI.roiHeight = halfH;

    // Permutation: which source image for each quadrant
    for (int i = 0; i < batchSize; i++) {
        permutationTensor[i * 4 + 0] = (i + 0) % batchSize;
        permutationTensor[i * 4 + 1] = (i + 1) % batchSize;
        permutationTensor[i * 4 + 2] = (i + 2) % batchSize;
        permutationTensor[i * 4 + 3] = (i + 3) % batchSize;
    }
}

// Helper to initialize grid dropout boxes
void init_grid_dropout_boxes(int batchCount, RpptRoiLtrb* anchorBoxInfoTensor,
                             RpptROI* roiTensorPtrSrc, Rpp32u gridH, Rpp32u gridW, Rpp32u& maxHoleW,
                             Rpp32u& maxHoleH, Rpp32f holeRatio, int seed) {
    std::mt19937 rng(seed);

    for (int i = 0; i < batchCount; i++) {
        Rpp32u roiW = roiTensorPtrSrc[i].xywhROI.roiWidth;
        Rpp32u roiH = roiTensorPtrSrc[i].xywhROI.roiHeight;
        Rpp32s x_base = roiTensorPtrSrc[i].xywhROI.xy.x;
        Rpp32s y_base = roiTensorPtrSrc[i].xywhROI.xy.y;

        Rpp32u cellW = std::max(1u, roiW / gridW);
        Rpp32u cellH = std::max(1u, roiH / gridH);
        Rpp32u holeW = std::max(1u, static_cast<Rpp32u>(cellW * holeRatio));
        Rpp32u holeH = std::max(1u, static_cast<Rpp32u>(cellH * holeRatio));
        if (holeW > maxHoleW) maxHoleW = holeW;
        if (holeH > maxHoleH) maxHoleH = holeH;

        std::uniform_int_distribution<int> distX(0, (cellW > holeW) ? cellW - holeW : 0);
        std::uniform_int_distribution<int> distY(0, (cellH > holeH) ? cellH - holeH : 0);

        int boxOffset = i * gridH * gridW;
        for (Rpp32u row = 0; row < gridH; ++row) {
            for (Rpp32u col = 0; col < gridW; ++col) {
                Rpp32s cellX = x_base + col * cellW;
                Rpp32s cellY = y_base + row * cellH;

                Rpp32s offsetX = 0, offsetY = 0;
                if (cellW > holeW && cellH > holeH) {
                    offsetX = distX(rng);
                    offsetY = distY(rng);
                }

                Rpp32s x1 = std::min(cellX + offsetX, x_base + (Rpp32s)roiW - 1);
                Rpp32s y1 = std::min(cellY + offsetY, y_base + (Rpp32s)roiH - 1);
                Rpp32s x2 = std::min(x1 + (Rpp32s)holeW - 1, x_base + (Rpp32s)roiW - 1);
                Rpp32s y2 = std::min(y1 + (Rpp32s)holeH - 1, y_base + (Rpp32s)roiH - 1);

                int boxIdx = boxOffset + (row * gridW + col);
                anchorBoxInfoTensor[boxIdx].lt.x = x1;
                anchorBoxInfoTensor[boxIdx].lt.y = y1;
                anchorBoxInfoTensor[boxIdx].rb.x = x2;
                anchorBoxInfoTensor[boxIdx].rb.y = y2;
            }
        }
    }
}

// Dropout helper function for channel dropout
void generate_channel_dropout_mask(Rpp8u* dropoutTensor, Rpp32f* dropoutProbability, int batchSize,
                                   int channels, int seed) {
    int numThreads = NUM_THREADS;
    omp_set_dynamic(0);

#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < batchSize; batchCount++) {
        std::mt19937 rng(seed + batchCount);
        std::bernoulli_distribution keepDist(1.0f - dropoutProbability[batchCount]);
        Rpp8u* maskPtrTemp = dropoutTensor + (batchCount * channels);
        bool atLeastOne = false;

        for (int channel = 0; channel < channels; channel++) {
            maskPtrTemp[channel] = keepDist(rng);
            atLeastOne |= maskPtrTemp[channel];
        }

        if (!atLeastOne) maskPtrTemp[rng() % channels] = 1;
    }
}

// Dropout helper function for cutout dropout
void init_cutout_dropout(int batchSize, int maxBoxesPerImage, Rpp32u* numOfBoxes,
                         RpptRoiLtrb* anchorBoxInfoTensor, RpptROIPtr roiTensorPtrSrc, int channels,
                         int BitDepthTestMode, int seed, int dropoutType, void* colorBuffer) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos_ratio(0.1f, 0.9f);
    std::uniform_real_distribution<float> wh_ratio_cutout(0.4f, 0.6f);

    Rpp8u* colors8u = reinterpret_cast<Rpp8u*>(colorBuffer);
    Rpp16f* colors16f = reinterpret_cast<Rpp16f*>(colorBuffer);
    Rpp32f* colors32f = reinterpret_cast<Rpp32f*>(colorBuffer);
    Rpp8s* colors8s = reinterpret_cast<Rpp8s*>(colorBuffer);

    for (int i = 0; i < batchSize; i++) {
        numOfBoxes[i] = maxBoxesPerImage;
        for (int j = 0; j < maxBoxesPerImage; j++) {
            int idx = i * maxBoxesPerImage + j;

            // Get ROI dimensions
            Rpp32f roiWidth = static_cast<Rpp32f>(roiTensorPtrSrc[i].xywhROI.roiWidth);
            Rpp32f roiHeight = static_cast<Rpp32f>(roiTensorPtrSrc[i].xywhROI.roiHeight);
            Rpp32f roiX = static_cast<Rpp32f>(roiTensorPtrSrc[i].xywhROI.xy.x);
            Rpp32f roiY = static_cast<Rpp32f>(roiTensorPtrSrc[i].xywhROI.xy.y);

            // Random box dimensions (40-60% of ROI)
            Rpp32f boxWidth = roiWidth * wh_ratio_cutout(rng);
            Rpp32f boxHeight = roiHeight * wh_ratio_cutout(rng);

            // Random position within ROI
            Rpp32f maxX = roiX + roiWidth - boxWidth;
            Rpp32f maxY = roiY + roiHeight - boxHeight;
            Rpp32f boxX = roiX + (maxX - roiX) * pos_ratio(rng);
            Rpp32f boxY = roiY + (maxY - roiY) * pos_ratio(rng);

            // Set anchor box in LTRB format
            anchorBoxInfoTensor[idx].lt.x = static_cast<Rpp32u>(boxX);
            anchorBoxInfoTensor[idx].lt.y = static_cast<Rpp32u>(boxY);
            anchorBoxInfoTensor[idx].rb.x = static_cast<Rpp32u>(boxX + boxWidth);
            anchorBoxInfoTensor[idx].rb.y = static_cast<Rpp32u>(boxY + boxHeight);

            // Set random color for the box
            for (int c = 0; c < channels; c++) {
                int colorIdx = idx * channels + c;
                if (BitDepthTestMode == 0)  // U8
                    colors8u[colorIdx] = static_cast<Rpp8u>(rng() % 256);
                else if (BitDepthTestMode == 2)  // F32
                    colors32f[colorIdx] = static_cast<Rpp32f>(rng() % 256) / 255.0f;
                else if (BitDepthTestMode == 1)  // F16
                    colors16f[colorIdx] = static_cast<Rpp16f>(rng() % 256) / 255.0f;
                else if (BitDepthTestMode == 6)  // I8
                    colors8s[colorIdx] = static_cast<Rpp8s>((rng() % 256) - 128);
            }
        }
    }
}
