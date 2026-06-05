# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""MIOpen argument parsing and flag alias normalization."""

import warnings
from typing import Dict, List


def is_flag(token: str) -> bool:
    """Return True if token looks like a CLI flag (e.g. -n, --layout).

    Negative numbers like -1 or -0.5 are values, not flags.
    """
    if not token.startswith("-"):
        return False
    rest = token[1:]
    if rest and (rest[0].isdigit() or rest[0] == "."):
        return False
    return True


def parse_args(tokens: List[str]) -> Dict[str, str]:
    """Parse a flat list of flag/value tokens into a dict."""
    result: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if is_flag(tok):
            # Check if next token is a value (not a flag)
            if i + 1 < len(tokens) and not is_flag(tokens[i + 1]):
                result[tok] = tokens[i + 1]
                i += 2
            else:
                result[tok] = "1"  # boolean flag
                i += 1
        else:
            warnings.warn(f"Stray value ignored: {tok!r}", stacklevel=2)
            i += 1
    return result


def get_int_arg(args: Dict[str, str], key: str, default: int = 0) -> int:
    return int(args.get(key, default))


def normalize_args(args: Dict[str, str], aliases: Dict[str, str]) -> Dict[str, str]:
    """Merge flag aliases so both short and long forms are recognized.

    The canonical key (value in *aliases*) takes precedence when both the
    alternative and canonical forms appear in *args*.
    """
    normalized = dict(args)
    for alt, canonical in aliases.items():
        if alt in normalized and canonical not in normalized:
            normalized[canonical] = normalized.pop(alt)
        elif alt in normalized:
            del normalized[alt]
    return normalized


# MIOpenDriver accepts both short and long flag forms.  The mappings below
# convert the *alternative* form to the *canonical* form already used by
# ConvParams.from_args() / build_bnorm_json() so that either style works.

CONV_FLAG_ALIASES: Dict[str, str] = {
    # Long → short (2D parameters)
    "--batchsize": "-n",
    "--in_channels": "-c",
    "--in_h": "-H",
    "--in_w": "-W",
    "--out_channels": "-k",
    "--fil_h": "-y",
    "--fil_w": "-x",
    "--pad_h": "-p",
    "--pad_w": "-q",
    "--conv_stride_h": "-u",
    "--conv_stride_w": "-v",
    "--dilation_h": "-l",
    "--dilation_w": "-j",
    "--group_count": "-g",
    "--forw": "-F",
    "--mode": "-m",
    # Short → long (3D / layout parameters)
    "-_": "--spatial_dim",
    "-!": "--in_d",
    "-@": "--fil_d",
    "-$": "--pad_d",
    "-#": "--conv_stride_d",
    "-^": "--dilation_d",
    "-I": "--in_layout",
    "-f": "--fil_layout",
    "-O": "--out_layout",
    "--pad_mode": "-z",
    "--tensor_vect": "-Z",
    "--trans_output_pad_h": "-Y",
    "--trans_output_pad_w": "-X",
    "--trans_output_pad_d": "-%",
    "--bias": "-b",
    "--vector_length": "-L",
    "--pad_val": "-r",
    "--wei_cast_type": "-R",
    "--out_cast_type": "-T",
    "--in_cast_type": "-U",
}

BNORM_FLAG_ALIASES: Dict[str, str] = {
    # Long → short
    "--batchsize": "-n",
    "--in_channels": "-c",
    "--in_h": "-H",
    "--in_w": "-W",
    "--in_d": "-D",
    "--layout": "-L",
    "--mode": "-m",
    "--alpha": "-A",
    "--beta": "-B",
    "--activ_mode": "-f",
    # Short → long
    "-F": "--forw",
    "-b": "--back",
    "--run": "-r",
    "--save": "-s",
    "--inverse_variance": "-I",
}
