#!/bin/bash

# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

nohup python gemm_tuning.py --input_file example/Qwen3-32B_hipblaslt.log --output_path test_tuning --requested_solution 128 > output.log 2>&1 &
