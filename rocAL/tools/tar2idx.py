# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import tarfile
import time
import os
import subprocess
import sys
from shutil import which


class CreateIndexFiles:

    index_file_version = "v1.0"

    def __init__(self, path_to_tar_archive, index_path):
        self.path_to_tar_archive = path_to_tar_archive
        self.index_path = index_path
        self.file_index = open(self.index_path, "w")

    def open(self):
        if self.file_index.closed:
            self.file_index = open(self.index_path, "w")
        else:
            self.file_index.seek(0)

    def close(self):
        if not self.file_index.closed:
            self.file_index.close()

    def reset(self):
        self.close()
        self.open()

    def split_filepath_name(path):
        dot_pos = path.find(".", path.rfind("/") + 1)
        return path[:dot_pos], path[dot_pos + 1:]

    def get_tar_files_data(self):
        tar_blocks_process = subprocess.Popen(
            ["tar", "--list", "--block-num", "--file", self.path_to_tar_archive],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        tar_types_sizes_proc = subprocess.Popen(
            ["tar", "--verbose", "--list", "--file", self.path_to_tar_archive],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        tar_blocks = tar_blocks_process.communicate()[0].split(b"\n")
        tar_types_sizes = tar_types_sizes_proc.communicate()[0].split(
            b"\n"
        )

        for blocks_line, types_sizes_line in zip(tar_blocks, tar_types_sizes):
            if not blocks_line or not types_sizes_line:
                continue

            name = str(blocks_line[blocks_line.find(b":") + 2:], "ascii")
            entry_type = types_sizes_line[0:1]

            if entry_type != b"-":
                continue

            offset = int(blocks_line[blocks_line.find(b"block") + 6: blocks_line.find(b":")])
            offset = (offset + 1) * 512

            size = types_sizes_line[: -len(name)]
            size = size[: size.rfind(b"-") - 8]
            size = int(size[size.rfind(b" "):])

            yield offset, name, size

    def get_data_from_tar_files(self):

        file_arch = tarfile.open(self.path_to_tar_archive)
        for member_file_arch in iter(file_arch):
            if member_file_arch.type != tarfile.REGTYPE:
                continue
            offset = file_arch.fileobj.tell()
            yield offset, member_file_arch.name, member_file_arch.size

    def create_index_from_tar(self):
        self.reset()

        # Aggregates extensions in samples
        aggregated_data = []
        last_basename = None

        for offset, name, size in (
            self.get_tar_files_data() if which("tar") is not None else self.get_data_from_tar_files()
        ):

            basename, extension = CreateIndexFiles.split_filepath_name(name)

            if not basename or basename.endswith("/"):
                continue

            if last_basename != basename:
                aggregated_data.append([(extension, offset, size, name)])
                last_basename = basename
            else:
                aggregated_data[-1].append((extension, offset, size, name))

        if not aggregated_data:
            raise ValueError("Webdataset Tar File empty - Please verify the tar files passed as input")

        self.file_index.write(f"{CreateIndexFiles.index_file_version} {len(aggregated_data)}\n")
        for data_item in aggregated_data:
            self.file_index.write(" ".join(map(lambda component: " ".join(map(str, component)), data_item)))
            self.file_index.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Creates index files for all the tar files in the specified directory.",
    )
    parser.add_argument(
        "directory",
        help="path to the directory containing .tar files.",
    )
    args = parser.parse_args()
    args.directory = os.path.abspath(args.directory)
    return args


def main():
    args = parse_args()
    tar_files = [f for f in os.listdir(args.directory) if f.endswith('.tar')]
    if not tar_files:
        print(f"No .tar files found in directory, please check the directory: {args.directory}", file=sys.stderr)
        return

    for tar_file in tar_files:
        tar_path = os.path.join(args.directory, tar_file)
        index_path = os.path.join(args.directory, os.path.splitext(tar_file)[0] + ".idx")
        print(f"Processing tar files {tar_path} to index files {index_path}")
        creator = CreateIndexFiles(tar_path, index_path)
        creator.create_index_from_tar()


if __name__ == "__main__":
    main()
