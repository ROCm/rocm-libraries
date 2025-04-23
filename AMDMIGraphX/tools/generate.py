#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#####################################################################################
import api, argparse, os, runpy, subprocess, sys, te
from pathlib import Path

clang_format_path = Path('clang-format.exe' if os.name ==
                         'nt' else '/opt/rocm/llvm/bin/clang-format')
work_dir = Path().cwd()
src_dir = (work_dir / '../src').absolute()
migraphx_py_path = src_dir / 'api/migraphx.py'


def clang_format(buffer, **kwargs):
    return subprocess.run(f'{clang_format_path} -style=file',
                          capture_output=True,
                          shell=True,
                          check=True,
                          input=buffer.encode('utf-8'),
                          cwd=work_dir,
                          **kwargs).stdout.decode('utf-8')


def api_generate(input_path: Path, output_path: Path):
    with open(output_path, 'w') as f:
        f.write(clang_format(api.run(input_path)))


def te_generate(input_path: Path, output_path: Path):
    with open(output_path, 'w') as f:
        f.write(clang_format(te.run(input_path)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--clang-format', type=Path)
    args = parser.parse_args()

    global clang_format_path
    if args.clang_format:
        clang_format_path = args.clang_format

    if not clang_format_path.is_file():
        print(f"{clang_format_path}: invalid path or not installed",
              file=sys.stderr)
        return

    try:
        files = Path('include').absolute().iterdir()
        for f in [f for f in files if f.is_file()]:
            te_generate(f, src_dir / f'include/migraphx/{f.name}')
        runpy.run_path(str(migraphx_py_path))
        api_generate(work_dir / 'api/migraphx.h',
                     src_dir / 'api/include/migraphx/migraphx.h')
        print('Finished generating header migraphx.h')
        api_generate(work_dir / 'api/api.cpp', src_dir / 'api/api.cpp')
        print('Finished generating source api.cpp')
    except subprocess.CalledProcessError as ex:
        if ex.stdout:
            print(ex.stdout.decode('utf-8'))
        if ex.stderr:
            print(ex.stdout.decode('utf-8'))
        print(f"Command '{ex.cmd}' returned {ex.returncode}")
        raise


if __name__ == "__main__":
    main()
