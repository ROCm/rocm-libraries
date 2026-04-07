#!/usr/bin/env python3
"""
Training problem set for forward grouped convolution.

Combines:
1. MIOpen real-world shapes (300 problems)
2. Synthetic shapes with diverse G and N (2165 problems)

Total: 2465 training problems

Distribution:
- Balanced G coverage (G=1,2,4,8,16,32)
- Balanced N coverage (N=1,2,4,8,16,32,64)
- Diverse spatial sizes (7x7 to 112x112)
- Various filter sizes (1x1, 3x3, stride 1 and 2)
"""

import sys
from pathlib import Path

# Add dispatcher/python to path for grouped_conv_utils import
dispatcher_python = Path(__file__).resolve().parents[4] / "dispatcher" / "python"
sys.path.insert(0, str(dispatcher_python))


# Import MIOpen real-world shapes
from forward_training_miopen import TRAINING_PROBLEMS_FORWARD_MIOPEN

# Import synthetic shapes with diverse G and N
from forward_synthetic_extended import TRAINING_PROBLEMS_FORWARD_SYNTHETIC

# Combine both datasets
TRAINING_PROBLEMS_FORWARD = TRAINING_PROBLEMS_FORWARD_MIOPEN + TRAINING_PROBLEMS_FORWARD_SYNTHETIC

assert len(TRAINING_PROBLEMS_FORWARD) == 2465, f"Expected 2465 problems, got {len(TRAINING_PROBLEMS_FORWARD)}"
