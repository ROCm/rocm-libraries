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
from forward_training_miopen import TRAINING_PROBLEMS_FORWARD_MIOPEN  # noqa: E402

# Import synthetic shapes with diverse G and N
from forward_synthetic_extended import TRAINING_PROBLEMS_FORWARD_SYNTHETIC  # noqa: E402

# Combine both datasets
TRAINING_PROBLEMS_FORWARD = (
    TRAINING_PROBLEMS_FORWARD_MIOPEN + TRAINING_PROBLEMS_FORWARD_SYNTHETIC
)

# Validate count
assert len(TRAINING_PROBLEMS_FORWARD) > 1000, (
    f"Expected >1000 problems, got {len(TRAINING_PROBLEMS_FORWARD)}"
)

if __name__ == "__main__":
    # Note: Count may vary as synthetic set is tuned for C%8==0 and C%G==0 constraints
    print(
        f"Total training problems: {len(TRAINING_PROBLEMS_FORWARD)} "
        + f"({len(TRAINING_PROBLEMS_FORWARD_MIOPEN)} MIOpen + {len(TRAINING_PROBLEMS_FORWARD_SYNTHETIC)} synthetic)"
    )
