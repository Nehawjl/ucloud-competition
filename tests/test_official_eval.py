import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_official_eval", ROOT / "scripts/verify_official_eval.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)
ENTRY_SPEC = importlib.util.spec_from_file_location("official_entry", ROOT / "scripts/run_official_performance.py")
ENTRY = importlib.util.module_from_spec(ENTRY_SPEC)
ENTRY_SPEC.loader.exec_module(ENTRY)


class OfficialToolTests(unittest.TestCase):
    def test_original_archive_and_all_extracted_files_match(self):
        result = VERIFY.verify()
        self.assertEqual(result["status"], "byte-identical")
        self.assertEqual(set(result["files"]), {"README.md", "run.sh", "runner.py"})
        self.assertEqual(result["files"]["runner.py"]["sha256"],
                         "d80df390cf6d1d50129796b31fdd7f4d1e95d0043d3b942d47ac582118702fb0")

    def test_changed_extracted_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            shutil.copytree(ROOT / "third_party", target / "third_party",
                            ignore=shutil.ignore_patterns("__pycache__"))
            (target / "third_party/official-eval/run.sh").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "differs from archive"):
                VERIFY.verify(target)

    def test_official_entry_help_is_available_without_model(self):
        result = subprocess.run(["bash", str(ROOT / "third_party/official-eval/run.sh"), "--help"],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--mode all|perf|accuracy|check", result.stdout)

    def test_adapter_changes_only_selected_concurrency(self):
        arguments = ["--base-url", "http://127.0.0.1:18108/v1", "--model", "qwen3.8-27b"]
        for concurrency in range(1, 15):
            official = SimpleNamespace(CONCURRENCIES=(1, 2, 4, 8), main=Mock(return_value=0),
                                       EvaluationError=RuntimeError)
            with patch.object(ENTRY, "load_official", return_value=official):
                self.assertEqual(ENTRY.run(concurrency, arguments), 0)
            self.assertEqual(official.CONCURRENCIES, (concurrency,))
            official.main.assert_called_once_with(["--mode", "perf", *arguments])

    def test_adapter_rejects_invalid_concurrency_or_mode(self):
        for concurrency in (0, 15):
            with self.assertRaises(ValueError):
                ENTRY.run(concurrency, [])
        for arguments in (["--mode", "all"], ["--mode=accuracy"]):
            with self.assertRaises(ValueError):
                ENTRY.run(8, arguments)

    def test_shell_rejects_c15_before_starting_model(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/run.sh"), "official-perf"],
                                env={**os.environ, "CONCURRENCY": "15"},
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("supports C1-C14", result.stderr)

    def test_official_dry_run_selects_only_c8_or_c14(self):
        for concurrency in (8, 14):
            with tempfile.TemporaryDirectory() as temporary:
                result = subprocess.run([sys.executable, str(ROOT / "scripts/run_official_performance.py"),
                                         "--concurrency", str(concurrency), "--base-url", "http://127.0.0.1:18108/v1",
                                         "--model", "qwen3.8-27b", "--env-dir", temporary, "--dry-run"],
                                        text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [line for line in result.stdout.splitlines() if " --concurrency " in line]
            self.assertEqual(len(commands), 1)
            self.assertIn(f" --concurrency {concurrency} ", commands[0])


if __name__ == "__main__":
    unittest.main()
