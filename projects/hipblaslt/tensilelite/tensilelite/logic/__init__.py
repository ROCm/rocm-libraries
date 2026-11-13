# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

def main():
    """Run TensileLogic without importing its heavy dependencies eagerly."""
    from .run import main as run

    return run()
