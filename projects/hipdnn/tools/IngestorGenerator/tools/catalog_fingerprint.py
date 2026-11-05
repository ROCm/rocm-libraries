"""Print the catalog fingerprint of a KDP, the value an engine config records as
``engine.trained_rankers.<arch>.trained_catalog_sha256`` when a ranker is trained on it.

    python3 tools/catalog_fingerprint.py <bundle>/artifacts/<pack>.kdp.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codegen.catalog_fingerprint import catalog_fingerprint  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kdp", type=Path, help="the KDP the ranker was trained on")
    args = parser.parse_args(argv)
    print(catalog_fingerprint(json.loads(args.kdp.read_text())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
