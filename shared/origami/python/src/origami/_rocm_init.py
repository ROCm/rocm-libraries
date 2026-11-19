# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Load native dependencies from ROCm Python packages when installed."""


def initialize():
    try:
        import rocm_sdk
    except ModuleNotFoundError as error:
        if error.name != "rocm_sdk":
            raise
        return

    rocm_sdk.initialize_process(
        preload_shortnames=["amdhip64", "origami"],
        env_override=False,
    )
