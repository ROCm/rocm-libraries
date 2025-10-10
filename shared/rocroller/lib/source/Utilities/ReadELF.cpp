/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Serialization/comgr/comgr.hpp>
#include <rocRoller/Serialization/ELF.hpp>
#include <rocRoller/Serialization/YAML.hpp>
#include <amd_comgr/amd_comgr.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <stdarg.h>

using namespace rocRoller;

void fail(const char *format, ...) {
  va_list ap;
  va_start(ap, format);

  printf("FAILED: ");
  vprintf(format, ap);
  printf("\n");

  va_end(ap);

  exit(1);
}

void checkStatus(amd_comgr_status_t status, amd_comgr_status_t expected,
                 const char *str) {
  if (status != expected) {
    const char *statusStr;
    printf("FAILED: %s\n", str);
    status = amd_comgr_status_string(status, &statusStr);
    if (status == AMD_COMGR_STATUS_SUCCESS)
      printf(" REASON: %s\n", statusStr);
    exit(1);
  }
}

void checkError(amd_comgr_status_t status, const char *str) {
  checkStatus(status, AMD_COMGR_STATUS_SUCCESS, str);
}

amd_comgr_status_t printEntry(amd_comgr_metadata_node_t key,
                              amd_comgr_metadata_node_t value, void *data) {
  amd_comgr_metadata_kind_t kind;
  amd_comgr_metadata_node_t son;
  amd_comgr_status_t status;
  size_t size;
  char *keybuf;
  char *valbuf;
  int *indent = (int *)data;

  // assume key to be string in this test function
  status = amd_comgr_get_metadata_kind(key, &kind);
  checkError(status, "amd_comgr_get_metadata_kind");
  if (kind != AMD_COMGR_METADATA_KIND_STRING)
    return AMD_COMGR_STATUS_ERROR;
  status = amd_comgr_get_metadata_string(key, &size, NULL);
  checkError(status, "amd_comgr_get_metadata_string");
  keybuf = (char *)calloc(size, sizeof(char));
  if (!keybuf)
    fail("calloc");
  status = amd_comgr_get_metadata_string(key, &size, keybuf);
  checkError(status, "amd_comgr_get_metadata_string");

  status = amd_comgr_get_metadata_kind(value, &kind);
  checkError(status, "amd_comgr_get_metadata_kind");
  for (int i = 0; i < *indent; i++)
    printf("  ");

  switch (kind) {
  case AMD_COMGR_METADATA_KIND_STRING: {
    printf("%s  :  ", size ? keybuf : "");
    status = amd_comgr_get_metadata_string(value, &size, NULL);
    checkError(status, "amd_comgr_get_metadata_string");
    valbuf = (char *)calloc(size, sizeof(char));
    if (!valbuf)
      fail("calloc");
    status = amd_comgr_get_metadata_string(value, &size, valbuf);
    checkError(status, "amd_comgr_get_metadata_string");
    printf(" %s\n", valbuf);
    free(valbuf);
    break;
  }
  case AMD_COMGR_METADATA_KIND_LIST: {
    *indent += 1;
    status = amd_comgr_get_metadata_list_size(value, &size);
    checkError(status, "amd_comgr_get_metadata_list_size");
    printf("LIST %s %zd entries = \n", keybuf, size);
    for (size_t i = 0; i < size; i++) {
      status = amd_comgr_index_list_metadata(value, i, &son);
      checkError(status, "amd_comgr_index_list_metadata");
      status = printEntry(key, son, data);
      checkError(status, "printEntry");
      status = amd_comgr_destroy_metadata(son);
      checkError(status, "amd_comgr_destroy_metadata");
    }
    *indent = *indent > 0 ? *indent - 1 : 0;
    break;
  }
  case AMD_COMGR_METADATA_KIND_MAP: {
    *indent += 1;
    status = amd_comgr_get_metadata_map_size(value, &size);
    checkError(status, "amd_comgr_get_metadata_map_size");
    printf("MAP %zd entries = \n", size);
    status = amd_comgr_iterate_map_metadata(value, printEntry, data);
    checkError(status, "amd_comgr_iterate_map_metadata");
    *indent = *indent > 0 ? *indent - 1 : 0;
    break;
  }
  default:
    free(keybuf);
    return AMD_COMGR_STATUS_ERROR;
  } // switch

  free(keybuf);
  return AMD_COMGR_STATUS_SUCCESS;
}

const unsigned char ELF_MAGIC[] = {0x7f, 'E', 'L', 'F'};
const unsigned char ELF_CLASS_64 = 2;      // 64-bit
const unsigned char ELF_LITTLE_ENDIAN = 1;      // Little endian

bool isELF64LE(const std::vector<char>& buffer) {
    if (buffer.size() < 16) {
        return false;
    }
    
    if (std::memcmp(buffer.data(), ELF_MAGIC, 4) != 0) {
        std::cout << "Not an ELF file - invalid magic number" << std::endl;
        return false;
    }
    
    if (static_cast<unsigned char>(buffer[4]) != ELF_CLASS_64) {
        std::cout << "Not a 64-bit ELF file" << std::endl;
        return false;
    }
    
    if (static_cast<unsigned char>(buffer[5]) != ELF_LITTLE_ENDIAN) {
        std::cout << "Not little endian ELF file" << std::endl;
        return false;
    }
    
    std::cout << "File is ELF64LE" << std::endl;
    return true;
}

std::string rocRoller::readMetaDataFromCodeObject(std::string const& fileName)
{
    std::string yaml;

    amd_comgr_data_t elfData;
    
    auto status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &elfData);
    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to create COMGR data object");
    
    std::ifstream file(fileName, std::ios::binary);
    AssertFatal(file.is_open(), "Failed to open file: " + fileName);
    
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(fileSize);
    file.read(buffer.data(), fileSize);
    file.close();

    if (!isELF64LE(buffer)) {
        amd_comgr_release_data(elfData);
        throw std::runtime_error("File is not a valid ELF64LE file: " + fileName);
    }
    
    // Debug: Print buffer contents (first 256 bytes or less)
    std::cout << "File size:" << fileSize << " bytes" << std::endl;
    std::cout << "file name: " << fileName << std::endl;
    std::cout << "Buffer size: " << buffer.size() << " bytes" << std::endl;

    // Debug: Print first 256 bytes as hex
    size_t bytesToPrint = std::min(size_t(256), buffer.size());
    std::cout << "First " << bytesToPrint << " bytes (hex):" << std::endl;
    
    for (size_t i = 0; i < bytesToPrint; ++i) {
        if (i % 16 == 0) {
            std::cout << std::endl << std::setw(4) << std::setfill('0') << std::hex << i << ": ";
        }
        std::cout << std::setw(2) << std::setfill('0') << std::hex 
                  << (static_cast<unsigned int>(static_cast<unsigned char>(buffer[i]))) << " ";
    }
    std::cout << std::dec << std::endl; // Reset to decimal
    
    // Debug Alternative: Print as ASCII with dots for non-printable chars
    std::cout << "\nFirst " << bytesToPrint << " bytes (ASCII):" << std::endl;
    for (size_t i = 0; i < bytesToPrint; ++i) {
        if (i % 64 == 0) {
            std::cout << std::endl;
        }
        unsigned char c = static_cast<unsigned char>(buffer[i]);
        if (c >= 32 && c <= 126) {
            std::cout << c;
        } else {
            std::cout << '.';
        }
    }
    std::cout << std::endl;
    /*
    status = amd_comgr_set_data(elfData, buffer.size(), buffer.data());
    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to set ELF data");

    amd_comgr_metadata_node_t metadataNode;
    status = amd_comgr_get_data_metadata(elfData, &metadataNode);
    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to extract metadata from ELF");

    std::cout << "Metadata extraction status: " << status << std::endl;
    std::cout << "Metadata node handle: " << metadataNode.handle << std::endl;

    amd_comgr_metadata_kind_t metadataKind;
    status = amd_comgr_get_metadata_kind(metadataNode, &metadataKind);
    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get metadata kind");

    std::cout << "Metadata kind: " << metadataKind << std::endl;

    size_t yamlSize;
    status = amd_comgr_get_metadata_string(metadataNode, &yamlSize, nullptr);
    std::cout << "Get metadata string size status: " << status << std::endl;
    std::cout << "Expected YAML size: " << yamlSize << std::endl;
    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get metadata string size");
    
    std::vector<char> yamlBuffer(yamlSize);
    status = amd_comgr_get_metadata_string(metadataNode, &yamlSize, yamlBuffer.data());
    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get metadata string");
    
    yaml = std::string(yamlBuffer.data(), yamlSize - 1);
    amd_comgr_destroy_metadata(metadataNode);
    amd_comgr_release_data(elfData);
    */

    status = amd_comgr_set_data(elfData, buffer.size(), buffer.data());
    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to set ELF data");

    // Extract metadata node from ELF data
    amd_comgr_metadata_node_t metadataNode;
    status = amd_comgr_get_data_metadata(elfData, &metadataNode);
    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to extract metadata from ELF");

    int Indent = 0;
    status = amd_comgr_iterate_map_metadata(metadataNode, printEntry, (void *)&Indent);
    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "iterate_map_metadata");

    status = amd_comgr_destroy_metadata(metadataNode);
    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "destroy_metadata");

    return yaml;
}
