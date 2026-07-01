# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Concurrency module.

This module provides tools to manage and manipulate concurrent tasks across multiple devices, including
load balancing, task scheduling, and parallel execution.

It defines a Runner class that can execute tasks in parallel across multiple GPU devices, 
handling task distribution, progress tracking, and error handling.

Modules:
    runner: Defines the Runner and Worker classes.
    utils: Utility functions for parallel execution.
"""

from .runner import Runner, Worker
from .utils import parallel_for
