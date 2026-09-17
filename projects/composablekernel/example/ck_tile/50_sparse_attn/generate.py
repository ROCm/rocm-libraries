# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import sys
import importlib.util
from enum import IntEnum
from pathlib import Path
import pkgutil
from typing import List, Optional

import codegen.ops


class HandlerId(IntEnum):
    LIST_BLOBS = 0
    WRITE_BLOBS = 1


def _load_ops_module(importer, module_name: str):
    """Load a submodule from codegen.ops, matching Loader.load_module behavior."""
    spec = importer.find_spec(module_name)
    loader = spec.loader
    if hasattr(loader, "load_module"):
        return loader.load_module(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


# Load all modules under codegen.ops and register their API handlers.
ops = []
for importer, module_name, _ in pkgutil.iter_modules(codegen.ops.__path__):
    full_module_name = "%s.%s" % (codegen.ops.__name__, module_name)
    ops.append(_load_ops_module(importer, module_name))
unwanted_prefix = "fmha_"
handlers = dict(
    [
        (
            op.__name__[len(unwanted_prefix) :]
            if op.__name__.startswith(unwanted_prefix)
            else op.__name__,
            (op.list_blobs, op.write_blobs),
        )
        for op in ops
    ]
)
assert 0 < len(handlers)


def write_blobs(
    output_dir: Optional[str],
    api_list: List[str],
    filters_list: List[str],
    optdim_list: List[int],
    receipt,
    mask_impl,
) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for api, kernel_filter in zip(api_list, filters_list):
        handler = handlers[api][HandlerId.WRITE_BLOBS]
        handler(output_dir, kernel_filter, receipt, optdim_list, mask_impl)


def list_blobs(
    output_file: Optional[str],
    api_list: List[str],
    filters_list: List[str],
    optdim_list: List[int],
    receipt,
    mask_impl,
) -> None:
    assert output_file is not None
    file_path = Path(output_file)

    open(file_path, "w").close()

    for api, kernel_filter in zip(api_list, filters_list):
        handler = handlers[api][HandlerId.LIST_BLOBS]
        handler(file_path, kernel_filter, receipt, optdim_list, mask_impl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK fmha kernel",
    )
    parser.add_argument(
        "-a",
        "--api",
        default="fwd_jenga",
        required=False,
        help="supply API(s) to generate (default: fwd_jenga). separated by comma.\n"
        "  fwd_jenga / fwd_vsa / fwd_sparge",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="write all the blobs into a directory",
    )
    parser.add_argument(
        "-l", "--list_blobs", required=False, help="list all the kernels to a file"
    )
    parser.add_argument(
        "-f",
        "--filter",
        default="",
        required=False,
        help="filter out kernels that need to generate, using fnmatch module",
    )

    parser.add_argument(
        "-m",
        "--mask",
        default="simplified",
        required=False,
        help="mask implementation, simplified/generic",
    )

    parser.add_argument(
        "-r",
        "--receipt",
        default=0,
        required=False,
        help="codegen receipt (sparse_attn build uses 600). Recognized values:\n"
        "  0   : default coverage\n"
        "  2/3 : Flash attention integration subset\n"
        "  4   : PyTorch integration subset (jenga / vsa only)\n"
        "  100 : Aiter mha_fwd subset (jenga / vsa only)\n"
        "  200 : Aiter mha_varlen_fwd subset (group mode for all 3 APIs)\n"
        "  600 : Aiter C++ API integration subset (used by CMake)",
    )

    parser.add_argument(
        "--optdim",
        default="-1",
        required=False,
        help="only optimize the hdim in the list. separated by comma. -1 is the default choice"
        + "eg. --optdim=32,64,128,256",
    )

    args = parser.parse_args()
    api_list = args.api.split(",")
    filter_list = args.filter.split(",")
    filter_list.extend([""] * (len(api_list) - len(filter_list)))
    optdim_list = [int(hdim) for hdim in args.optdim.split(",")]

    if args.list_blobs is not None:
        list_blobs(
            args.list_blobs,
            api_list,
            filter_list,
            optdim_list,
            int(args.receipt),
            mask_impl=args.mask,
        )
    else:
        write_blobs(
            args.output_dir,
            api_list,
            filter_list,
            optdim_list,
            int(args.receipt),
            mask_impl=args.mask,
        )
