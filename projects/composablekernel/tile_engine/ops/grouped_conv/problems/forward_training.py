#!/usr/bin/env python3

# Training problem set for forward grouped convolution
# 200 diverse MIOpen production shapes
# Real-world workloads from MIOpen benchmark suite

from grouped_conv_utils import GroupedConvProblem

# Import all 300 MIOpen shapes
from forward_training_miopen import TRAINING_PROBLEMS_FORWARD_MIOPEN

# Use first 200 shapes for training
TRAINING_PROBLEMS_FORWARD = TRAINING_PROBLEMS_FORWARD_MIOPEN[:200]

assert len(TRAINING_PROBLEMS_FORWARD) == 200, f"Expected 200 problems, got {len(TRAINING_PROBLEMS_FORWARD)}"
