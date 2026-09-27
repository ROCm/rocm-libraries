// -*- C++ -*-

// Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <memory>
#include <fstream>
#include <iostream>
#include <tuple>
#include <limits>
#include <cstdint>
#include <vector>
#include <iomanip>
#include <thread>
#include <cstring>

struct COOMatrix {
    uint32_t n_rows, n_cols;
    std::vector<uint32_t> row, col;
    std::vector<double> values;
};
struct CSRMatrix {
    const uint32_t n_rows, n_cols, entry_count;
    std::unique_ptr<uint32_t[]> row_ptr, col;
    std::unique_ptr<double[]> values;
};
struct CSCMatrix {
    const uint32_t n_rows, n_cols, entry_count;
    std::unique_ptr<uint32_t[]> row, col_ptr;
    std::unique_ptr<double[]> values;
};


void sortMatrix(CSRMatrix &matrix) {
    std::vector<std::thread> threads(4);
    for (uint32_t t = 0; t < threads.size(); ++t) {
        uint32_t startRow =                                           t      * (matrix.n_rows / threads.size());
        uint32_t endRow = (t + 1 == threads.size() ? matrix.n_rows : (t + 1) * (matrix.n_rows / threads.size()));

        threads[t] = std::thread([&](uint32_t startRow, uint32_t endRow) {
            auto &[n_rows, n_cols, entry_count, row_ptr, col, values] = matrix;
            for (uint32_t curRow = startRow; curRow < endRow; ++curRow) {
                // Insertion sort
                for (uint32_t i = row_ptr[curRow] + 1; i < row_ptr[curRow+1]; ++i) {
                    uint32_t curCol = col[i];
                    double curVal = values[i];
                    uint32_t j;
                    for (j = i; j > row_ptr[curRow] && col[j-1] > curCol; --j) {
                        col[j] = col[j-1];
                        values[j] = values[j-1];
                    }
                    col[j] = curCol;
                    values[j] = curVal;
                }
            }
        }, startRow, endRow);
    }
    for (auto &t : threads) {
        t.join();
    }
}

void printMatrix(CSRMatrix &matrix) {
    auto &[n_rows, n_cols, entry_count, row_ptr, col, values] = matrix;
    std::cout << std::showpoint;
    for (uint32_t idx = 0, curRow = 0; curRow < n_rows; ++curRow) {
        for (uint32_t curCol = 0; curCol < n_cols; ++curCol) {
            double val = 0;
            if (row_ptr[curRow] <= idx && idx < row_ptr[curRow + 1] &&
                col[idx] == curCol) {
                val = values[idx++];
            }
            std::cout << std::setw(25) << val << " ";
        }
        std::cout << "\n";
    }
}

void readHeader(std::ifstream &fileStream, bool &symmetric) {
    // Read first line 
    for (int i = 0; i < 4; ++i) {
        std::string word;
        fileStream >> word;
        if (word != (const char*[]){"%%MatrixMarket", "matrix", "coordinate", "real"}[i]) {
            throw std::logic_error("Matrix file header didn't match expected format");
        }
    }
    {
        std::string word;
        fileStream >> word;
        if (word != "symmetric" && word != "general") {
            throw std::logic_error("Matrix file header didn't match expected format");
        }
        symmetric = (word == "symmetric");
    }
    if (fileStream.get() != '\n') {
        throw std::logic_error("Matrix file header didn't match expected format");
    }

    while (fileStream.peek() == '%') {
        fileStream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

COOMatrix readMatrix(std::ifstream &fileStream, bool symmetric) {
    COOMatrix matrix;
    auto &[n_rows, n_cols, row, col, values] = matrix;
    uint32_t entry_count;
    fileStream >> n_rows >> n_cols >> entry_count;
    if (symmetric) {
        entry_count *= 2;
    }

    row.resize(entry_count);
    col.resize(entry_count);
    values.resize(entry_count);

    for (uint32_t i = 0; i < entry_count; i += (symmetric ? 2 : 1)) {
        fileStream >> row[i];
        fileStream >> col[i];
        fileStream >> values[i];
        // Skip explicit zeroes
        if (values[i] == 0.0) {
            i -= (symmetric ? 2 : 1);
            entry_count -= (symmetric ? 2 : 1);
            continue;
        }
        // switch from 1 indexed to 0 indexed
        --row[i];
        --col[i];

        if (symmetric) {
            row[i+1] = col[i];
            col[i+1] = row[i];
            values[i+1] = values[i];
        }
    }
    row.resize(entry_count);
    col.resize(entry_count);
    values.resize(entry_count);
    return matrix;
}

CSRMatrix convertToCSRMatrix(const COOMatrix &cooMatrix) {
    CSRMatrix csrMatrix{cooMatrix.n_rows, cooMatrix.n_cols, static_cast<uint32_t>(cooMatrix.values.size()), nullptr, nullptr, nullptr};
    auto &[n_rows, n_cols, entry_count, row_ptr, col, values] = csrMatrix;

    row_ptr = std::make_unique<uint32_t[]>(n_rows + 1);
    col = std::make_unique<uint32_t[]>(entry_count);
    values = std::make_unique<double[]>(entry_count);

    std::vector<uint32_t> row_widths(n_rows);
    for (uint32_t i = 0; i < entry_count; ++i) {
        ++row_widths[cooMatrix.row[i]];
    }

    row_ptr[0] = 0;
    for (uint32_t i = 0, total = 0; i < n_rows; ++i) {
        total += row_widths[i];
        row_ptr[i+1] = total;
    }

    std::vector<uint32_t> filled_row_width(n_rows);
    for (uint32_t i = 0; i < entry_count; ++i) {
        uint32_t curRow = cooMatrix.row[i], curCol = cooMatrix.col[i];
        double curVal = cooMatrix.values[i];
        uint32_t newIdx = row_ptr[curRow] + filled_row_width[curRow];
        ++filled_row_width[curRow];
        col[newIdx] = curCol;
        values[newIdx] = curVal;
    }

    return csrMatrix;
}

CSCMatrix convertToCSCMatrix(const CSRMatrix &csrMatrix) {
    CSCMatrix cscMatrix{csrMatrix.n_rows, csrMatrix.n_cols, csrMatrix.entry_count, nullptr, nullptr, nullptr};
    auto &[n_rows, n_cols, entry_count, row, col_ptr, values] = cscMatrix;

    row = std::make_unique<uint32_t[]>(entry_count);
    col_ptr = std::make_unique<uint32_t[]>(n_cols + 1);
    values = std::make_unique<double[]>(entry_count);

    std::vector<uint32_t> col_widths(n_cols);
    for (uint32_t i = 0; i < entry_count; ++i) {
        ++col_widths[csrMatrix.col[i]];
    }

    col_ptr[0] = 0;
    for (uint32_t i = 0, total = 0; i < n_cols; ++i) {
        total += col_widths[i];
        col_ptr[i+1] = total;
    }

    std::vector<uint32_t> filled_col_width(n_cols);
    for (uint32_t curRow = 0; curRow < n_rows; ++curRow) {
        for (uint32_t i = csrMatrix.row_ptr[curRow]; i < csrMatrix.row_ptr[curRow+1]; ++i) {
            uint32_t curCol = csrMatrix.col[i];
            double curVal = csrMatrix.values[i];
            uint32_t newIdx = col_ptr[curCol] + filled_col_width[curCol];
            ++filled_col_width[curCol];
            row[newIdx] = curRow;
            values[newIdx] = curVal;
        }
    }

    return cscMatrix;
}

CSRMatrix parseFile(std::string filePath) {
    std::ifstream fileStream(filePath);

    bool symmetric;
    readHeader(fileStream, symmetric);
    COOMatrix cooMatrix = readMatrix(fileStream, symmetric);
    CSRMatrix csrMatrix = convertToCSRMatrix(cooMatrix);
    return csrMatrix;
}

uint32_t findCol(const CSRMatrix &matrix, uint32_t row, uint32_t col) {
    uint32_t start = matrix.row_ptr[row], end = matrix.row_ptr[row + 1] - 1;
    while (start <= end) {
        int mid = start + (end - start) / 2;
        if (matrix.col[mid] == col) {
            return mid;
        }

        if (matrix.col[mid] < col) {
            start = mid + 1;
        } else if (mid == 0) {
            break;
        } else {
            end = mid - 1;
        }
    }
    return matrix.row_ptr[row + 1];
}

double computeEntry(const CSRMatrix &matrixA, const CSCMatrix &matrixB, uint32_t rowA, uint32_t colB) {
    const uint32_t rowA_start = matrixA.row_ptr[rowA];
    const uint32_t rowA_end = matrixA.row_ptr[rowA+1];
    const uint32_t colB_start = matrixB.col_ptr[colB];
    const uint32_t colB_end = matrixB.col_ptr[colB+1];
    double sum = 0;
    for (uint32_t entryA = rowA_start, entryB = colB_start; entryA < rowA_end && entryB < colB_end; ) {
        const uint32_t colA = matrixA.col[entryA];
        const uint32_t rowB = matrixB.row[entryB];
        if (colA == rowB)
            sum += matrixA.values[entryA] * matrixB.values[entryB];
        if (colA <= rowB)
            ++entryB;
        if (colA >= rowB)
            ++entryA;
    }
    return sum;
}

CSRMatrix multiply(const CSRMatrix &matrixA, const CSRMatrix &matrixB_csr) {
    const uint32_t n_rows = matrixA.n_rows, n_cols = matrixB_csr.n_cols;
    CSCMatrix matrixB_csc = convertToCSCMatrix(matrixB_csr);

    std::vector<std::vector<uint32_t>> col(n_rows);
    std::vector<std::vector<double>> values(n_rows);

    std::vector<std::thread> threads(64 * std::thread::hardware_concurrency());
    for (uint32_t t = 0; t < threads.size(); ++t) {
        const uint32_t chunk_size = (t < n_rows % threads.size()) ? (n_rows / threads.size() + 1) : (n_rows / threads.size());
        const uint32_t startRow =   (t < n_rows % threads.size()) ? (t * chunk_size)              : (t * chunk_size + n_rows % threads.size());
        threads[t] = std::thread([&](uint32_t startRow, uint32_t endRow) {
            for (uint32_t curRow = startRow; curRow < endRow; ++curRow) {
                std::vector<uint32_t> myCol;
                std::vector<double> myValues;
                for (uint32_t curCol = 0; curCol < n_cols; ++curCol) {
                    double val = computeEntry(matrixA, matrixB_csc, curRow, curCol);
                    if (val != 0.0) {
                        myCol.push_back(curCol);
                        myValues.push_back(val);
                    }
                }
                col[curRow] = std::move(myCol);
                values[curRow] = std::move(myValues);
            }
        }, startRow, startRow + chunk_size);
    }
    for (auto &t : threads) {
        t.join();
    }

    auto row_ptr = std::make_unique<uint32_t[]>(n_rows + 1);
    row_ptr[0] = 0;
    uint32_t entry_count = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        entry_count += values[i].size();
        row_ptr[i+1] = entry_count;
    }

    CSRMatrix result{n_rows,
                     n_cols,
                     entry_count,
                     std::move(row_ptr),
                     std::make_unique<uint32_t[]>(entry_count),
                     std::make_unique<double[]>(entry_count)};
    for (uint32_t i = 0; i < n_rows; ++i) {
        const uint32_t row_start = result.row_ptr[i];
        std::memcpy(&result.col[row_start], col[i].data(), col[i].size() * sizeof(uint32_t));
        std::memcpy(&result.values[row_start], values[i].data(), values[i].size() * sizeof(double));
    }
    return result;
}

void test(std::string fileName) {
    CSRMatrix matrix = parseFile(fileName);
    sortMatrix(matrix);
    auto start = std::chrono::steady_clock::now();
    CSRMatrix squared = multiply(matrix, matrix);
    auto finish = std::chrono::steady_clock::now();
    std::chrono::nanoseconds total = finish - start;
    std::cout.imbue(std::locale(""));
    std::cout << "Time(" << fileName << ") = " << total.count() << "ns\n";
}

int main() {
    test("../data/test_general.mtx");
    test("../data/shallow_water1.mtx");
    // Other matrices are in data/matrices.tgz
    // test("../data/pdb1HYS.mtx");
    // test("../data/2cubes_sphere.mtx");
    // test("../data/cant.mtx");
    // test("../data/scircuit.mtx");
    // test("../data/cop20k_A.mtx");
    // test("../data/mac_econ_fwd500.mtx");
    // test("../data/crankseg_2.mtx");
    // test("../data/nd24k.mtx");
    // test("../data/pwtk.mtx");
    // test("../data/webbase-1M.mtx");
    // test("../data/F1.mtx");
    // test("../data/atmosmodd.mtx");
    // test("../data/cage14.mtx");
    // test("../data/ldoor.mtx");
    return 0;
}
