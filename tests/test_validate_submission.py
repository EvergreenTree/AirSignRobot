#!/usr/bin/env python3
"""Focused tests for submission credential scanning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import validate_submission


class SubmissionCredentialScanTests(unittest.TestCase):
    def scan_text(self, text: str) -> tuple[list[str], list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.txt"
            candidate.write_text(text, encoding="utf-8")
            with (
                mock.patch.object(validate_submission, "ROOT", root),
                mock.patch.object(
                    validate_submission,
                    "tracked_and_untracked_files",
                    return_value=[candidate],
                ),
                mock.patch.object(validate_submission, "FAILURES", []),
                mock.patch.object(validate_submission, "PASSED", []),
            ):
                validate_submission.scan_for_secrets()
                return (
                    list(validate_submission.FAILURES),
                    list(validate_submission.PASSED),
                )

    def test_detects_fine_grained_github_token(self) -> None:
        token = "github" + "_pat_" + "11AIRSIGN_" + ("A" * 40)
        failures, passed = self.scan_text(token)
        self.assertEqual(
            failures,
            ["Potential GitHub token in candidate.txt"],
        )
        self.assertEqual(passed, [])

    def test_preserves_classic_detection_and_allows_short_placeholder(self) -> None:
        classic_token = "gh" + "p_" + ("A" * 20)
        failures, _ = self.scan_text(classic_token)
        self.assertEqual(
            failures,
            ["Potential GitHub token in candidate.txt"],
        )

        placeholder = "github" + "_pat_" + "EXAMPLE"
        failures, passed = self.scan_text(placeholder)
        self.assertEqual(failures, [])
        self.assertEqual(
            passed,
            ["Tracked and pending files contain no recognized credential patterns"],
        )


if __name__ == "__main__":
    unittest.main()
