# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocjitsu_shape_plan import benchmark_group_plan, digest, whole_yaml_plan


class BenchmarkGroupPlanTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.jobs = []
        for index, n in enumerate((4, 12, 8)):
            source = self.root / f"shape{index}.yaml"
            source.write_text(
                yaml.safe_dump(
                    dict(
                        GlobalParameters=dict(NumWarmups=3),
                        BenchmarkProblems=[
                            [
                                dict(OperationType="GEMM"),
                                dict(
                                    ForkParameters=[dict(DepthU=[16, 32])],
                                    BenchmarkFinalParameters=[
                                        dict(
                                            ProblemSizes=[
                                                dict(Exact=[n, 8, 1, 16]),
                                                dict(MinStride=[1, 2, 0, 0]),
                                            ]
                                        )
                                    ],
                                ),
                            ]
                        ],
                    )
                )
            )
            self.jobs.append(
                dict(
                    id=str(index),
                    nodeid=f"sample/p0_g1_s{index}",
                    source="sample.yaml",
                    source_sha256="frozen",
                    problem_type_index=0,
                    group_index=0,
                    grouped=False,
                    markers=[],
                    work=n * 8 * 16,
                    problems=[dict(sizes=[n, 8, 1, 16])],
                    yaml=str(source),
                    yaml_sha256=digest(source),
                )
            )
        # Composition tests isolate the existing domain parser dependency.
        # Full-corpus qualification uses the real parser before numerical trials.
        parser = SimpleNamespace(
            ProblemType=lambda *args: {},
            ProblemSizes=lambda problem_type, sizes: SimpleNamespace(
                problems=[
                    SimpleNamespace(sizes=entry["Exact"])
                    for entry in sizes
                    if "Exact" in entry
                ]
            ),
        )
        patcher = patch.dict(sys.modules, {"tensilelite.SolutionStructs": parser})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_admission_order_controls_and_atomic_domains_survive_bundling(self):
        atomic = dict(
            self.jobs[0], id="atomic", grouped=True, problems=[dict(sizes=[2, 3, 1, 4])]
        )
        plan = dict(
            ceiling=2000,
            jobs=[self.jobs[2], atomic, self.jobs[0], self.jobs[1]],
            excluded=[dict(id="excluded", work=9999)],
        )
        result = benchmark_group_plan(plan, self.root / "bundles")
        ordinary = next(job for job in result["jobs"] if not job["grouped"])
        self.assertEqual(len(result["jobs"]), 2)
        self.assertEqual(result["excluded"], plan["excluded"])
        self.assertEqual(next(job for job in result["jobs"] if job["grouped"]), atomic)
        self.assertEqual(ordinary["admitted_case_ids"], ["0", "1", "2"])
        self.assertEqual(
            ordinary["problems"], [job["problems"][0] for job in self.jobs]
        )
        self.assertEqual(ordinary["work"], sum(job["work"] for job in self.jobs))
        self.assertGreater(ordinary["work"], plan["ceiling"])
        document = yaml.safe_load(Path(ordinary["yaml"]).read_text())
        self.assertEqual(document["GlobalParameters"], dict(NumWarmups=3))
        group = document["BenchmarkProblems"][0][1]
        self.assertEqual(group["ForkParameters"], [dict(DepthU=[16, 32])])
        self.assertEqual(
            group["BenchmarkFinalParameters"][0]["ProblemSizes"],
            [
                *[dict(Exact=job["problems"][0]["sizes"]) for job in self.jobs],
                dict(MinStride=[1, 2, 0, 0]),
            ],
        )
        self.assertEqual(plan["jobs"][0], self.jobs[2])

    def test_incompatible_controls_and_changed_inputs_are_rejected(self):
        for change in ("controls", "hash", "source_hash", "markers"):
            with self.subTest(change=change):
                jobs = copy.deepcopy(self.jobs)
                if change == "controls":
                    source = Path(jobs[1]["yaml"])
                    document = yaml.safe_load(source.read_text())
                    document["GlobalParameters"]["NumWarmups"] = 7
                    changed = self.root / "changed.yaml"
                    changed.write_text(yaml.safe_dump(document))
                    jobs[1].update(yaml=str(changed), yaml_sha256=digest(changed))
                elif change == "hash":
                    jobs[1]["yaml_sha256"] = "wrong"
                elif change == "source_hash":
                    jobs[1]["source_sha256"] = "different"
                else:
                    jobs[1]["markers"] = [dict(name="skip")]
                with self.assertRaises(ValueError):
                    benchmark_group_plan(
                        dict(jobs=jobs, excluded=[]), self.root / change
                    )

    def test_whole_yaml_keeps_original_group_identities_after_removal(self):
        source = self.root / "original.yaml"
        document = yaml.safe_load(Path(self.jobs[0]["yaml"]).read_text())
        document["TestParameters"] = dict(skip=False)
        source.write_text(yaml.safe_dump(document))
        jobs = copy.deepcopy(self.jobs)
        for index, job in enumerate(jobs):
            job.update(
                source=str(source),
                source_sha256=digest(source),
                problem_type_index=2 if index == 2 else 0,
                group_index=4 if index == 2 else 2,
                nodeid=f"sample/p{2 if index == 2 else 0}_g{5 if index == 2 else 3}_s{index}",
            )
        plan = dict(
            jobs=jobs,
            excluded=[dict(nodeid="excluded", work=99999)],
            sources=[
                dict(
                    path=str(source), sha256=digest(source), nodeid="sample", markers=[]
                )
            ],
        )
        result = whole_yaml_plan(plan, self.root / "whole")
        self.assertEqual(result["excluded"], plan["excluded"])
        self.assertEqual(len(result["jobs"]), 1)
        job = result["jobs"][0]
        self.assertEqual(job["group_mapping"], {"0:0": [0, 2], "1:0": [2, 4]})
        self.assertEqual(job["problems"], [j["problems"][0] for j in jobs])
        retained = yaml.safe_load(Path(job["yaml"]).read_text())
        self.assertEqual(retained["GlobalParameters"], document["GlobalParameters"])
        self.assertEqual(retained["TestParameters"], document["TestParameters"])
        self.assertEqual([len(p) for p in retained["BenchmarkProblems"]], [2, 2])
        self.assertEqual(
            [
                (g["problem_type_index"], g["group_index"])
                for g in job["benchmark_groups"]
            ],
            [(0, 2), (2, 4)],
        )


if __name__ == "__main__":
    unittest.main()
