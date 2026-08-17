import json
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import therock_multiarch_label_flags as label_flags


FAKE_MAP = {
    "ci:miopen-hipdnn-wrapper": ["-DTHEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER=ON"],
    "ci:two-flags": [
        "-DTHEROCK_FLAG_FIRST=ON",
        "-DTHEROCK_FLAG_SECOND=OFF",
    ],
    "ci:overlapping": ["-DTHEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER=ON"],
    "ci:conflicting": ["-DTHEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER=OFF"],
    # Deliberately unlike the suggested convention, and full of characters a
    # regex would treat specially: membership is an exact string test.
    "weird.label+name(1)": ["-DTHEROCK_FLAG_WEIRD=ON"],
}


def run_main(env, fake_map=FAKE_MAP):
    """Run main() with a patched map and env, returning the GHA outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        output_file = os.path.join(tmp, "github_output")
        summary_file = os.path.join(tmp, "github_step_summary")
        full_env = {
            "GITHUB_OUTPUT": output_file,
            "GITHUB_STEP_SUMMARY": summary_file,
            **env,
        }
        with mock.patch.dict(os.environ, full_env, clear=True), mock.patch.object(
            label_flags, "LABEL_GATED_THEROCK_FLAGS", fake_map
        ):
            label_flags.main()
        raw = Path(output_file).read_text()
        summary = Path(summary_file).read_text()

    outputs = {}
    for line in raw.splitlines():
        key, _, value = line.partition("=")
        outputs[key] = value
    return outputs, raw, summary


BASE_ENV = {
    "GITHUB_EVENT_NAME": "pull_request",
    "EXTERNAL_REPO_REPOSITORY": "ROCm/rocm-libraries",
    "EXTERNAL_REPO_REF": "deadbeef",
}


class ShippedMapTest(unittest.TestCase):
    def test_shipped_map_is_valid(self):
        # The map that actually ships must satisfy its own validator.
        label_flags.validate_label_gated_therock_flags()

    def test_fake_map_is_valid(self):
        label_flags.validate_label_gated_therock_flags(FAKE_MAP)


class ValidationTest(unittest.TestCase):
    def assert_rejected(self, mapping):
        with self.assertRaises(ValueError):
            label_flags.validate_label_gated_therock_flags(mapping)

    def test_unprefixed_flag_rejected(self):
        # The trap this validator exists for: an unprefixed name set at top
        # level is an unread cache variable and builds green with the flag off.
        self.assert_rejected({"ci:x": ["-DMIOPEN_ENABLE_MY_FEATURE=ON"]})

    def test_clobbered_namespace_rejected(self):
        self.assert_rejected({"ci:x": ["-DTHEROCK_ENABLE_BLAS=ON"]})

    def test_non_boolean_values_rejected(self):
        self.assert_rejected({"ci:x": ["-DTHEROCK_FLAG_FOO=1"]})
        self.assert_rejected({"ci:x": ["-DTHEROCK_FLAG_FOO=on"]})
        self.assert_rejected({"ci:x": ["-DTHEROCK_FLAG_FOO="]})
        self.assert_rejected({"ci:x": ["-DTHEROCK_FLAG_FOO"]})

    def test_shell_metacharacters_rejected(self):
        for bad in [
            "-DTHEROCK_FLAG_FOO=ON;rm -rf /",
            "-DTHEROCK_FLAG_FOO=ON -DTHEROCK_FLAG_BAR=ON",
            "-DTHEROCK_FLAG_FOO=ON'",
            '-DTHEROCK_FLAG_FOO="ON"',
            "-DTHEROCK_FLAG_FOO=ON\n",
            "-DTHEROCK_FLAG_FOO=$(id)",
            "-DTHEROCK_FLAG_FOO=ON&&id",
        ]:
            with self.subTest(bad=bad):
                self.assert_rejected({"ci:x": [bad]})

    def test_lowercase_flag_name_rejected(self):
        self.assert_rejected({"ci:x": ["-DTHEROCK_FLAG_foo=ON"]})

    def test_bad_map_shapes_rejected(self):
        # A bare string would iterate character by character.
        self.assert_rejected({"ci:x": "-DTHEROCK_FLAG_FOO=ON"})
        self.assert_rejected({"ci:x": []})
        self.assert_rejected({"ci:x": None})
        self.assert_rejected({"": ["-DTHEROCK_FLAG_FOO=ON"]})
        self.assert_rejected({1: ["-DTHEROCK_FLAG_FOO=ON"]})
        self.assert_rejected({"ci:x": [None]})
        self.assert_rejected([])

    def test_empty_map_is_valid(self):
        label_flags.validate_label_gated_therock_flags({})


class ExternalRepoTest(unittest.TestCase):
    def test_default_payload_matches_workflow_literal(self):
        # This must stay byte-identical to the literal the workflows used
        # before this script existed, or unlabeled runs change behavior.
        outputs, _, _ = run_main({**BASE_ENV, "PR_LABELS_JSON": '["test:rocblas"]'})
        self.assertEqual(
            outputs["external_repo"],
            '{"repository":"ROCm/rocm-libraries","ref":"deadbeef"}',
        )
        self.assertEqual(outputs["flags_active"], "false")
        self.assertEqual(outputs["flags"], "")
        self.assertEqual(outputs["matched_labels"], "")

    def test_single_matching_label(self):
        outputs, _, _ = run_main(
            {**BASE_ENV, "PR_LABELS_JSON": '["ci:miopen-hipdnn-wrapper"]'}
        )
        payload = json.loads(outputs["external_repo"])
        self.assertEqual(
            payload["extra_cmake_options"],
            "-DTHEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER=ON",
        )
        self.assertEqual(outputs["flags_active"], "true")
        self.assertEqual(outputs["matched_labels"], "ci:miopen-hipdnn-wrapper")

    def test_multiple_labels_and_flags(self):
        outputs, _, _ = run_main(
            {
                **BASE_ENV,
                "PR_LABELS_JSON": '["ci:two-flags","ci:miopen-hipdnn-wrapper","test:rocblas"]',
            }
        )
        payload = json.loads(outputs["external_repo"])
        # Order follows the map, not the payload.
        self.assertEqual(
            payload["extra_cmake_options"],
            "-DTHEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER=ON "
            "-DTHEROCK_FLAG_FIRST=ON -DTHEROCK_FLAG_SECOND=OFF",
        )
        self.assertEqual(
            outputs["matched_labels"], "ci:miopen-hipdnn-wrapper,ci:two-flags"
        )

    def test_duplicate_flag_with_same_value_is_emitted_once(self):
        outputs, _, _ = run_main(
            {
                **BASE_ENV,
                "PR_LABELS_JSON": '["ci:miopen-hipdnn-wrapper","ci:overlapping"]',
            }
        )
        payload = json.loads(outputs["external_repo"])
        self.assertEqual(
            payload["extra_cmake_options"],
            "-DTHEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER=ON",
        )

    def test_conflicting_values_are_fatal(self):
        with self.assertRaises(ValueError):
            run_main(
                {
                    **BASE_ENV,
                    "PR_LABELS_JSON": '["ci:miopen-hipdnn-wrapper","ci:conflicting"]',
                }
            )

    def test_unknown_labels_ignored(self):
        outputs, _, _ = run_main(
            {**BASE_ENV, "PR_LABELS_JSON": '["ci:smoke","ci:asan","not-a-label"]'}
        )
        self.assertEqual(outputs["flags_active"], "false")

    def test_non_pull_request_event_ignores_labels(self):
        outputs, _, _ = run_main(
            {
                **BASE_ENV,
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "PR_LABELS_JSON": '["ci:miopen-hipdnn-wrapper"]',
            }
        )
        self.assertEqual(outputs["flags_active"], "false")
        self.assertEqual(
            outputs["external_repo"],
            '{"repository":"ROCm/rocm-libraries","ref":"deadbeef"}',
        )

    def test_junk_labels_json_tolerated(self):
        for raw in ["", "null", "not json", "{}", '"a string"', "[1,2,null]", "[]"]:
            with self.subTest(raw=raw):
                outputs, _, _ = run_main({**BASE_ENV, "PR_LABELS_JSON": raw})
                self.assertEqual(outputs["flags_active"], "false")

    def test_outputs_are_single_line_and_shell_safe(self):
        outputs, raw, _ = run_main({**BASE_ENV, "PR_LABELS_JSON": '["ci:two-flags"]'})
        self.assertEqual(len(raw.splitlines()), 5)
        self.assertNotIn("'", outputs["external_repo"])
        self.assertNotIn('"', outputs["flags"])

    def test_summary_names_the_labels_it_acted_on(self):
        _, _, summary = run_main(
            {**BASE_ENV, "PR_LABELS_JSON": '["ci:miopen-hipdnn-wrapper","ci:smoke"]'}
        )
        self.assertIn("ci:miopen-hipdnn-wrapper", summary)
        self.assertIn("ci:smoke", summary)
        self.assertIn("-DTHEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER=ON", summary)


class PrebuiltStageGuardTest(unittest.TestCase):
    def test_prebuilt_stages_with_flags_is_fatal(self):
        with self.assertRaises(SystemExit):
            run_main(
                {
                    **BASE_ENV,
                    "PR_LABELS_JSON": '["ci:miopen-hipdnn-wrapper"]',
                    "PREBUILT_STAGES": "all",
                }
            )

    def test_baseline_run_id_with_flags_is_fatal(self):
        with self.assertRaises(SystemExit):
            run_main(
                {
                    **BASE_ENV,
                    "PR_LABELS_JSON": '["ci:miopen-hipdnn-wrapper"]',
                    "BASELINE_RUN_ID": "123456",
                }
            )

    def test_prebuilt_stages_without_flags_is_fine(self):
        outputs, _, _ = run_main(
            {
                **BASE_ENV,
                "PR_LABELS_JSON": '["ci:smoke"]',
                "PREBUILT_STAGES": "all",
                "BASELINE_RUN_ID": "123456",
            }
        )
        self.assertEqual(outputs["flags_active"], "false")


class LabelRelevantTest(unittest.TestCase):
    """`label_relevant` is what decides whether an expensive build runs."""

    def test_mapped_changed_label_is_relevant(self):
        outputs, _, _ = run_main(
            {
                **BASE_ENV,
                "PR_LABELS_JSON": '["ci:miopen-hipdnn-wrapper"]',
                "CHANGED_LABEL": "ci:miopen-hipdnn-wrapper",
            }
        )
        self.assertEqual(outputs["label_relevant"], "true")

    def test_unmapped_changed_label_is_not_relevant(self):
        for changed in ["ci:smoke", "documentation", "ci:gpu:gfx942"]:
            with self.subTest(changed=changed):
                outputs, _, _ = run_main(
                    {
                        **BASE_ENV,
                        "PR_LABELS_JSON": f'["{changed}"]',
                        "CHANGED_LABEL": changed,
                    }
                )
                self.assertEqual(outputs["label_relevant"], "false")

    def test_removing_a_mapped_label_is_relevant(self):
        # `unlabeled`: the label is gone from the payload but still names a
        # configuration change, so the flag-off build must run.
        outputs, _, _ = run_main(
            {
                **BASE_ENV,
                "PR_LABELS_JSON": "[]",
                "CHANGED_LABEL": "ci:miopen-hipdnn-wrapper",
            }
        )
        self.assertEqual(outputs["label_relevant"], "true")
        self.assertEqual(outputs["flags_active"], "false")

    def test_no_changed_label_is_not_relevant(self):
        # As on synchronize/opened/reopened. The workflow gate's first clause
        # ("action is not labeled/unlabeled") is what carries those events;
        # label_relevant can only ever suppress a label event itself.
        for env in [{}, {"CHANGED_LABEL": ""}]:
            with self.subTest(env=env):
                outputs, _, _ = run_main({**BASE_ENV, "PR_LABELS_JSON": "[]", **env})
                self.assertEqual(outputs["label_relevant"], "false")

    def test_sticky_label_still_injects_without_a_label_event(self):
        # The case most likely to regress if someone "simplifies" the two
        # inputs into one: a push to a PR that already carries the label.
        outputs, _, _ = run_main(
            {
                **BASE_ENV,
                "GITHUB_EVENT_NAME": "pull_request",
                "PR_LABELS_JSON": '["ci:miopen-hipdnn-wrapper"]',
                "CHANGED_LABEL": "",
            }
        )
        self.assertEqual(outputs["label_relevant"], "false")
        self.assertEqual(outputs["flags_active"], "true")

    def test_membership_is_exact_string_not_a_pattern(self):
        outputs, _, _ = run_main(
            {
                **BASE_ENV,
                "PR_LABELS_JSON": '["weird.label+name(1)"]',
                "CHANGED_LABEL": "weird.label+name(1)",
            }
        )
        self.assertEqual(outputs["label_relevant"], "true")
        self.assertEqual(outputs["flags"], "-DTHEROCK_FLAG_WEIRD=ON")

        # A string the key would match only if it were treated as a regex.
        outputs, _, _ = run_main(
            {
                **BASE_ENV,
                "PR_LABELS_JSON": '["weirdXlabel+name1"]',
                "CHANGED_LABEL": "weirdXlabel+name1",
            }
        )
        self.assertEqual(outputs["label_relevant"], "false")
        self.assertEqual(outputs["flags_active"], "false")


if __name__ == "__main__":
    unittest.main()
