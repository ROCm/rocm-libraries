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
#include <mutex>
#include <vector>
#include <libtar.h>
#include <string>
#include <list>
#include <cassert>
#include <cstdarg>
#include <algorithm>
#include <cstring>
#include "helpers/tar_helper_functions.h"
#include "pipeline/exception.h"

template <typename value, typename alignment>
constexpr value align_up(value v, alignment a) {
  return v + ((a - 1) & -v);
}

constexpr uint64_t TARARCHIVEBUFFERINITSIZE = 1;

std::mutex instances_mutex;
std::list<std::vector<TarArchive*>> instances_registry = {
    std::vector<TarArchive*>(TARARCHIVEBUFFERINITSIZE)
};
TarArchive** instances = instances_registry.back().data();

int add_to_register(TarArchive* archive) {
  std::lock_guard<std::mutex> instances_lock(instances_mutex);
  for (auto& instances_entry : instances_registry.back()) {
    if (instances_entry != nullptr) {
      continue;
    }
    instances_entry = archive;
    return &instances_entry - instances;
  }
  std::vector<TarArchive*>& old = instances_registry.back();
  instances_registry.emplace_back();
  std::vector<TarArchive*>& curr = instances_registry.back();
  curr.reserve(old.size() * 2);
  curr = old;
  curr.push_back(archive);
  curr.resize(old.size() * 2, nullptr);
  instances = curr.data();
  return old.size();
}

inline void remove_from_register(int instance_handle) {
  instances[instance_handle] = nullptr;
}

inline TAR* void_ptr_to_tar_ptr(void* handle) {
  return reinterpret_cast<TAR*>(handle);
}

inline TAR** void_ptr_to_tar_ptr(void** handle) {
  return reinterpret_cast<TAR**>(handle);
}

ssize_t read_tar_archive(int instance_handle, void* buf, size_t count) {
  const auto current_archive = instances[instance_handle];
  current_archive->get_stream()->read(reinterpret_cast<char*>(buf), count);
  const ssize_t num_read = current_archive->get_stream()->gcount();
  return num_read;
}

int open_tar_archive(const char*, int oflags, ...) {
  va_list args;
  va_start(args, oflags);
  const int _instance_handle = va_arg(args, int);
  va_end(args);
  return _instance_handle;
}

static tartype_t kTarArchiveType = {open_tar_archive, [](int) -> int { return 0; },
                                    read_tar_archive,
                                    [](int, const void*, size_t) -> ssize_t { return 0; }};

TarArchive::TarArchive(std::unique_ptr<std::ifstream> stream)
    : _stream(std::move(stream)), _instance_handle(add_to_register(this)) {
  tar_open(void_ptr_to_tar_ptr(&_handle), "", &kTarArchiveType, 0, _instance_handle, TAR_GNU);
   _stream->seekg(0, std::ios_base::end);
  _eof = _stream->tellg() == 0;
  _stream->seekg(0, std::ios_base::beg);
  parse_current_header();
}

TarArchive::TarArchive(TarArchive&& other) {
  *this = std::move(other);
}

TarArchive::~TarArchive() {
  release_file_stream();
}

std::ifstream* TarArchive::get_stream() {
    return _stream.get();
}

TarArchive& TarArchive::operator=(TarArchive&& other) {
  if (&other != this) {
    _stream = std::move(other._stream);
    std::swap(_handle, other._handle);
    std::swap(_filename, other._filename);
    std::swap(_filesize, other._filesize);
    std::swap(_filetype, other._filetype);
    std::swap(_readoffset, other._readoffset);
    std::swap(_current_header, other._current_header);
    std::swap(_eof, other._eof);
    std::swap(_instance_handle, other._instance_handle);
    if (_instance_handle >= 0) {
      std::lock_guard<std::mutex> instances_lock(instances_mutex);
      instances[_instance_handle] = this;
    }
    other.release_file_stream();
  }
  return *this;
}

constexpr size_t round_it_to_given_block_size(size_t count) {
  return align_up(count, kBlockSize);
}

bool TarArchive::advance_to_next_file_in_tar() {
  if (_eof) { 
    return false;
  }
//   const int64_t offset = _stream->tellg() + round_it_to_given_block_size(_filesize) - _readoffset;
  const int64_t offset = static_cast<int64_t>(_stream->tellg()) + round_it_to_given_block_size(_filesize) - _readoffset;
  _current_header = offset;
  _stream->seekg(offset, std::ios::beg);
  parse_current_header();
  return !_eof;
}

bool TarArchive::at_end_of_archive() const {
  return _eof;
}

void TarArchive::seek_to_offset_in_archive(int64_t offset) {
  if (offset == _current_header) return;
  assert(offset % T_BLOCKSIZE == 0);
  _eof = false;
  _readoffset = 0;
  _stream->seekg(offset, std::ios::beg);
  _current_header = offset;
  parse_current_header();
}

int64_t TarArchive::get_current_archive_offset() const {
  return _current_header;
}

int64_t TarArchive::get_current_header_size() const {
  return static_cast<int64_t>(_stream->tellg()) - _readoffset - _current_header;

}

const std::string& TarArchive::get_current_file_name() const {
  return _filename;
}

size_t TarArchive::get_current_file_size() const {
  return _filesize;
}

TarArchive::EntryType TarArchive::get_current_file_type() const {
  return _filetype;
}

std::shared_ptr<void> TarArchive::read_current_file() {
  _stream->seekg(static_cast<size_t>(_stream->tellg()) - static_cast<size_t>(_readoffset), std::ios::beg);

  std::shared_ptr<void> out;
  if (out != nullptr) {
    _readoffset = _filesize;
  }
  return out;
}

size_t TarArchive::read_into_buffer(void *buffer, size_t count) {
  if (_eof) return 0;
  count = std::clamp(_filesize - _readoffset, static_cast<uint64_t>(0), count);
  _stream->read(reinterpret_cast<char*>(buffer), count);
  size_t num_read_bytes = _stream->gcount();
  _readoffset += num_read_bytes;
  return num_read_bytes;
}

bool TarArchive::is_end_of_file() const {
  return _readoffset >= _filesize;
}

inline void TarArchive::mark_end_of_file() {
  _eof = true;
  _filename = "";
  _filesize = 0;
  _filetype = ENTRY_NONE;
}

inline void TarArchive::parse_current_header() {
  if (_eof) return;
  int errorcode = th_read(void_ptr_to_tar_ptr(_handle));
  if (errorcode) {
    if (errorcode == -1)
      THROW("Corrupted tar file at " + void_ptr_to_tar_ptr(_handle)->pathname);
    mark_end_of_file();
  } else {
    _filename = th_get_pathname(void_ptr_to_tar_ptr(_handle));
    _filesize = th_get_size(void_ptr_to_tar_ptr(_handle));

    if (TH_ISREG(void_ptr_to_tar_ptr(_handle))) {
      _filetype = ENTRY_FILE;
    } else if (TH_ISDIR(void_ptr_to_tar_ptr(_handle))) {
      _filetype = ENTRY_DIR;
    } else if (TH_ISLNK(void_ptr_to_tar_ptr(_handle))) {
      _filetype = ENTRY_HARDLINK;
    } else if (TH_ISSYM(void_ptr_to_tar_ptr(_handle))) {
      _filetype = ENTRY_SYMLINK;
    } else if (TH_ISCHR(void_ptr_to_tar_ptr(_handle))) {
      _filetype = ENTRY_CHARDEV;
    } else if (TH_ISBLK(void_ptr_to_tar_ptr(_handle))) {
      _filetype = ENTRY_BLOCKDEV;
    } else if (TH_ISFIFO(void_ptr_to_tar_ptr(_handle))) {
      _filetype = ENTRY_FIFO;
    } else {
       _filetype = ENTRY_NOT_DEFINED;
    }
  }
  _readoffset = 0;
}

std::unique_ptr<std::ifstream> TarArchive::release_file_stream() {
  auto out = std::move(_stream);
  if (_handle != nullptr) {
    tar_close(void_ptr_to_tar_ptr(_handle));
    _handle = nullptr;
  }
  _readoffset = 0;
  _current_header = 0;
  mark_end_of_file();
  if (_instance_handle >= 0) {
    remove_from_register(_instance_handle);
  }
  _instance_handle = -1;
  return out;
}
#endif
