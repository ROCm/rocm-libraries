# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A kernel descriptor pack (KDP) whose kernels carry numeric tile metadata."""


def kdp(identity: str, engine: str, tiles) -> dict:
    """A KDP of `engine` with one kernel per entry of `tiles`, each with that `tile`."""
    return {
        "version": "1.0",
        "id": identity,
        "name": "kernels",
        "engine": engine,
        "kernelDescriptors": [
            {
                "id": f"{identity[:-1]}{i}",
                "metadata": {"tile": tile, "dtype": "bf16"},
            }
            for i, tile in enumerate(tiles)
        ],
    }
