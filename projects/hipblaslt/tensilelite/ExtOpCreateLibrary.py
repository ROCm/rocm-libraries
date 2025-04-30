################################################################################
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from argparse import ArgumentParser
from collections import defaultdict
import os
import glob
import msgpack
import yaml
import json

if __name__ == '__main__':
    ap = ArgumentParser(description='Parse op YAMLs and create library for hipBLASLt')
    ap.add_argument('--src', type=str, required=True, help='Folder that contains op meta files')
    ap.add_argument('--co', type=str, required=True, help='Path to code object file')
    ap.add_argument('--input-format', type=str, default='yaml', choices=('yaml', 'json'), help='Input kernel meta format')
    ap.add_argument('--format', type=str, default='dat', choices=('yaml', 'json', 'dat'), help='Library format, default is dat')
    ap.add_argument('--output', type=str, default='./', help='Output folder')
    ap.add_argument('--arch', type=str, required=True, help='GPU Architecture, e.g. gfx90a')
    args = ap.parse_args()
    src_folder: str = args.src
    lib_format: str = args.format
    input_format: str = args.input_format
    co_path: str = args.co
    output: str = args.output
    opt_arch: str = args.arch

    src_folder = os.path.expandvars(os.path.expanduser(src_folder))

    lib_meta = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for p in glob.glob(f'{src_folder}/*{opt_arch}.{input_format}'):
        meta_dict = {}

        with open(p) as f:
            if input_format == 'yaml':
                meta_dict = yaml.load(f, yaml.SafeLoader)
            elif input_format == 'json':
                meta_dict = json.load(f)

        meta_dict['co_path'] = os.path.basename(co_path)
        arch = meta_dict.pop('arch')
        op = meta_dict.pop('op')
        datatype = meta_dict['io_type']
        lib_meta[arch][op][datatype].append(meta_dict)

    output_open_foramt = 'wb' if lib_format == 'dat' else 'w'
    output_format_2_writer = {
        'dat': msgpack,
        'yaml': yaml,
        'json': json
    }

    output_lib_path = os.path.join(output, f'hipblasltExtOpLibrary.{lib_format}')
    
    if os.path.exists(output_lib_path):
        update_open_foramt = 'rb' if lib_format == 'dat' else 'r'
        with open(output_lib_path, update_open_foramt) as f:
            org_content = output_format_2_writer[lib_format].load(f)

        lib_meta = {**org_content, **lib_meta}

    with open(output_lib_path, output_open_foramt) as f:
        output_format_2_writer[lib_format].dump(lib_meta, f)
