/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifdef ENABLE_WDS
#include <fstream>
#include <memory>
#include <string>
#include <libtar.h>

constexpr size_t kBlockSize = T_BLOCKSIZE;
// Refactor TarArchive to use std::ifstream
class TarArchive {
public:
    TarArchive() = default;
    explicit TarArchive(std::unique_ptr<std::ifstream> stream);
    TarArchive(TarArchive &&);
    ~TarArchive();
    TarArchive &operator=(TarArchive &&);
    bool advance_to_next_file_in_tar();
    bool at_end_of_archive() const;
    void seek_to_offset_in_archive(int64_t offset);
    int64_t get_current_archive_offset() const;
    int64_t get_current_header_size() const;
    std::ifstream* get_stream();

    enum EntryType {
        ENTRY_NONE = 0,
        ENTRY_FILE,
        ENTRY_DIR,
        ENTRY_HARDLINK,
        ENTRY_SYMLINK,
        ENTRY_CHARDEV,
        ENTRY_BLOCKDEV,
        ENTRY_FIFO,
        ENTRY_NOT_DEFINED
    };

    const std::string& get_current_file_name() const;
    size_t get_current_file_size() const;
    EntryType get_current_file_type() const;
    std::shared_ptr<void> read_current_file();
    size_t read_into_buffer(void *buffer, size_t count);
    bool is_end_of_file() const;
    std::unique_ptr<std::ifstream> release_file_stream();

private:
    std::unique_ptr<std::ifstream> _stream; // Using std::ifstream directly
    int _instance_handle = -1;
    void* _handle = nullptr;
    friend ssize_t read_tar_archive(int, void *, size_t);
    std::string _filename;
    size_t _filesize = 0;
    EntryType _filetype = ENTRY_NONE;
    size_t _readoffset = 0;
    int64_t _current_header = 0;
    bool _eof = true;
    void mark_end_of_file();
    void parse_current_header();
};
#endif
