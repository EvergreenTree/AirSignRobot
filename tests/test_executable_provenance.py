#!/usr/bin/env python3
"""CPU-only tests for participant executable-source provenance."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from participant.executable_provenance import executable_source_guard


class ExecutableProvenanceTests(unittest.TestCase):
    def guard(
        self,
        root: Path,
        *paths: Path,
    ) -> dict[str, object]:
        return executable_source_guard(
            paths,
            root=root,
            forbidden_attribute_calls=("set_world_pose",),
            forbidden_direct_calls=("eval",),
        )

    def test_hash_binds_every_source_and_is_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.py"
            second = root / "b.py"
            first.write_text("value = 1\n", encoding="utf-8")
            second.write_text("value = 2\n", encoding="utf-8")
            forward = self.guard(root, first, second)
            reverse = self.guard(root, second, first)
            self.assertTrue(forward["passed"])
            self.assertEqual(
                forward["source_set_sha256"],
                reverse["source_set_sha256"],
            )
            second.write_text("value = 3\n", encoding="utf-8")
            changed = self.guard(root, first, second)
            self.assertNotEqual(
                forward["source_set_sha256"],
                changed["source_set_sha256"],
            )

    def test_forbidden_calls_are_attributed_to_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "controller.py"
            source.write_text(
                "robot.set_world_pose()\neval('1 + 1')\n",
                encoding="utf-8",
            )
            record = self.guard(root, source)
            self.assertFalse(record["passed"])
            self.assertEqual(
                [item["call"] for item in record["violations"]],
                ["set_world_pose", "eval"],
            )
            self.assertTrue(
                all(
                    item["path"] == "controller.py"
                    for item in record["violations"]
                )
            )

    def test_outside_duplicate_and_non_python_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            source = root / "controller.py"
            source.write_text("value = 1\n", encoding="utf-8")
            outside = Path(directory) / "outside.py"
            outside.write_text("value = 2\n", encoding="utf-8")
            text = root / "notes.txt"
            text.write_text("not Python\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.guard(root, outside)
            with self.assertRaises(ValueError):
                self.guard(root, source, source)
            with self.assertRaises(ValueError):
                self.guard(root, text)


if __name__ == "__main__":
    unittest.main()
