# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only validation and discovery tests; no HIP or third-party Python modules."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import rocm_arch

spec = importlib.util.spec_from_file_location("generate", ROOT / "tools/generate.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.definitions = self.root / "definitions"
        self.definitions.mkdir()

    def write_record(self, name="gfx942", **overrides):
        record = {"schema_version": 1, "id": name, "kind": "processor"}
        record.update(overrides)
        path = self.definitions / f"{name}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_public_lookup(self):
        names = rocm_arch.architectures()
        self.assertIsInstance(names, tuple)
        self.assertEqual(names, tuple(sorted(names)))
        self.assertIn("gfx1250", names)
        self.assertIn("gfx1250-strict", names)
        self.assertIn("gfx11-generic", rocm_arch.generic_targets())
        self.assertNotIn("gfx11-generic", names)
        for name in ("gfx94", "gfx942:xnack+", "GFX942", "all", None):
            with self.subTest(name=name):
                self.assertFalse(rocm_arch.is_known(name))

    def test_reject_invalid_records(self):
        for overrides in (
            {"schema_version": 2},
            {"schema_version": True},
            {"id": "gfx942:xnack+"},
            {"id": "gfx950"},
            {"kind": "unknown"},
            {"kind": "generic"},
            {"lds": 65536},
        ):
            with self.subTest(overrides=overrides):
                self.write_record(**overrides)
                with self.assertRaises(ValueError):
                    rocm_arch.load_catalog(self.definitions)

    def test_reject_empty_and_duplicate_json_fields(self):
        with self.assertRaises(ValueError):
            rocm_arch.load_catalog(self.definitions)
        path = self.write_record()
        path.write_text(
            '{"schema_version":1,"id":"gfx942","id":"gfx950","kind":"processor"}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
            rocm_arch.load_catalog(self.definitions)

    def test_discovery_removal_and_reproducibility(self):
        self.write_record("gfx942")
        output = self.root / "generated"
        generator.generate(self.definitions, output)
        original = {
            p.relative_to(output): p.read_bytes()
            for p in output.rglob("*")
            if p.is_file()
        }
        added = self.write_record("gfx9999")
        generator.generate(self.definitions, output)
        self.assertIn(
            ("gfx9999", "processor"), rocm_arch.load_catalog(self.definitions)
        )
        for p in original:
            self.assertIn(b"gfx9999", (output / p).read_bytes())
        added.unlink()
        generator.generate(self.definitions, output)
        self.assertEqual(original, {p: (output / p).read_bytes() for p in original})

    def test_cross_language_parity(self):
        # Compile an actual consumer and ask CMake for its public list. Compare
        # both outputs to Python, so a generator dropping a name cannot pass.
        output = self.root / "generated"
        generator.generate(None, output)
        names = rocm_arch.architectures()
        generics = rocm_arch.generic_targets()
        expected = "\n".join(
            (*names, "--generic--", *generics, rocm_arch.catalog_digest())
        )
        source = self.root / "consumer.cpp"
        source.write_text(
            "#include <rocm/arch/catalog.hpp>\n#include <iostream>\n"
            'int main() { for(auto n: rocm::arch::architectures) std::cout << n << "\\n";\n'
            'std::cout << "--generic--\\n";\n'
            'for(auto n: rocm::arch::generic_targets) std::cout << n << "\\n";\n'
            'std::cout << rocm::arch::catalog_digest << "\\n"; }\n',
            encoding="utf-8",
        )
        executable = self.root / "consumer"
        subprocess.run(
            [
                shutil.which("c++") or "c++",
                "-std=c++14",
                "-I",
                str(output / "include"),
                str(source),
                "-o",
                str(executable),
            ],
            check=True,
        )
        self.assertEqual(
            subprocess.check_output([str(executable)], text=True).strip(), expected
        )
        script = self.root / "consumer.cmake"
        result = self.root / "cmake.txt"
        script.write_text(
            f"cmake_minimum_required(VERSION 3.25.2)\n"
            f'include("{output}/rocm-arch-catalog.cmake")\n'
            f'include("{ROOT}/cmake/rocm-arch-functions.cmake")\n'
            "rocm_arch_list(processors KIND PROCESSOR)\n"
            "rocm_arch_list(generics KIND GENERIC)\n"
            "set(all ${processors} --generic-- ${generics} ${ROCM_ARCH_CATALOG_DIGEST})\n"
            'string(REPLACE ";" "\\n" contents "${all}")\n'
            f'file(WRITE "{result}" "${{contents}}")\n',
            encoding="utf-8",
        )
        subprocess.run(["cmake", "-P", str(script)], check=True)
        self.assertEqual(result.read_text(encoding="utf-8"), expected)


if __name__ == "__main__":
    unittest.main()
