# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocjitsu_shape_queue import Queue


CHILD = """
import argparse, json, time
from pathlib import Path
import xml.etree.ElementTree as ET
p = argparse.ArgumentParser()
for flag in ('rocjitsu-shape-job', 'rocjitsu-report-dir', 'basetemp', 'junit-xml'):
    p.add_argument('--' + flag)
a = p.parse_args()
job = json.loads(Path(a.rocjitsu_shape_job).read_text())
if job['id'] == 'hang':
    time.sleep(30)
if job['id'] == 'missing':
    raise SystemExit(0)
if job['id'] == 'orphan':
    import subprocess, sys
    orphan = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    Path(a.rocjitsu_report_dir, 'orphan.pid').write_text(str(orphan.pid))
time.sleep(0.02)
outcome = 'failed' if job['id'] == 'fail' else 'skipped' if job['id'] == 'skip' else 'passed'
progress = {'tests': {job['nodeid']: {'outcome': outcome, 'phases': {
    'call': {'outcome': outcome, 'seconds': 0.02},
    'teardown': {'outcome': 'passed', 'seconds': 0}}}}}
Path(a.rocjitsu_report_dir, 'progress.json').write_text(json.dumps(progress))
root = ET.Element('testsuites'); suite = ET.SubElement(root, 'testsuite', tests='1')
case = ET.SubElement(suite, 'testcase', name=job['nodeid'], time='0.02')
if outcome != 'passed': ET.SubElement(case, 'failure' if outcome == 'failed' else 'skipped')
ET.ElementTree(root).write(a.junit_xml)
raise SystemExit(1 if outcome == 'failed' else 0)
"""


class ShapeQueueTest(unittest.TestCase):
    def run_queue(self, identifiers, workers=2, timeout=5):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        child = root / "child.py"
        child.write_text(CHILD)
        plan = dict(
            jobs=[
                dict(id=name, nodeid=name, work=len(identifiers) - i)
                for i, name in enumerate(identifiers)
            ],
            excluded=[dict(id="excluded", nodeid="excluded", reason="size ceiling")],
        )
        queue = Queue(plan, root, [sys.executable, str(child)], timeout)
        return queue, root, queue.run(workers)

    def test_pull_queue_preserves_failures_skips_and_missing_reports(self):
        queue, root, status = self.run_queue(["ok", "fail", "skip", "missing"])
        self.assertEqual(status, 1)
        progress = json.loads((root / "progress.json").read_text())
        self.assertEqual((progress["selected"], progress["finished"]), (5, 5))
        self.assertEqual(
            (progress["passed"], progress["failed"], progress["skipped"]), (1, 2, 2)
        )
        self.assertEqual(queue.peak_active, 2)
        self.assertFalse((root / "jobs/excluded").exists())
        starts = sorted(
            queue.records.items(),
            key=lambda pair: pair[1].get("queue_position", float("inf")),
        )
        self.assertEqual(
            [name for name, _ in starts[:4]], ["ok", "fail", "skip", "missing"]
        )
        cases = list(ET.parse(root / "junit/tensilelite.xml").iter("testcase"))
        self.assertEqual(len(cases), 5)
        self.assertEqual(sum(case.find("failure") is not None for case in cases), 2)

    def test_per_job_timeout_is_a_failure(self):
        _, root, status = self.run_queue(["hang"], workers=1, timeout=0.1)
        self.assertEqual(status, 1)
        result = json.loads((root / "jobs/hang/result.json").read_text())
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["exit_code"], 124)

    def test_successful_queue_completes(self):
        _, root, status = self.run_queue(["ok", "ok2"], workers=1)
        self.assertEqual(status, 0)
        self.assertEqual(
            json.loads((root / "progress.json").read_text())["incomplete"], 0
        )

    def test_helpers_in_separate_sessions_are_reaped_before_next_job(self):
        _, root, status = self.run_queue(["orphan", "ok"], workers=1)
        self.assertEqual(status, 0)
        pid = (root / "jobs/orphan/orphan.pid").read_text()
        self.assertFalse(Path(f"/proc/{pid}").exists())


if __name__ == "__main__":
    unittest.main()
