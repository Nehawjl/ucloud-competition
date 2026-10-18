import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_eval", ROOT / "scripts/check_eval.py")
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


class EvalSetupTests(unittest.TestCase):
    def test_perf_checks_only_perf_environment(self):
        runner = SimpleNamespace(tool_python=Mock(return_value=Path("/perf/python")),
                                 inspect_tool=Mock(return_value={"ready": True}))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertTrue(SETUP.check_stage(runner, directory, "perf")["ready"])
            runner.tool_python.assert_called_once_with(directory, "perf")
            runner.inspect_tool.assert_called_once_with(Path("/perf/python"), "perf",
                                                       directory / "nltk_data")

    def test_nltk_download_is_noninteractive(self):
        data = SimpleNamespace(path=[], find=Mock(side_effect=[LookupError(), object()]))
        nltk = SimpleNamespace(data=data, download=Mock(return_value=True))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"nltk": nltk}), \
                patch.object(SETUP.socket, "setdefaulttimeout"):
            SETUP.prepare_nltk(Path(temporary), {"punkt_tab": "tokenizers/punkt_tab/english/"})
            nltk.download.assert_called_once_with("punkt_tab", download_dir=temporary,
                                                  halt_on_error=True, raise_on_error=True)

    def test_existing_nltk_resource_needs_no_network(self):
        nltk = SimpleNamespace(data=SimpleNamespace(path=[], find=Mock(return_value=object())),
                               download=Mock())
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"nltk": nltk}), \
                patch.object(SETUP.socket, "setdefaulttimeout"):
            SETUP.prepare_nltk(Path(temporary), {"stopwords": "corpora/stopwords"})
            nltk.download.assert_not_called()

    def test_security_error_is_preserved(self):
        data = SimpleNamespace(path=[], find=Mock(side_effect=LookupError()))
        nltk = SimpleNamespace(data=data, download=Mock(side_effect=ValueError("Security Violation")))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"nltk": nltk}), \
                patch.object(SETUP.socket, "setdefaulttimeout"):
            with self.assertRaisesRegex(ValueError, "Security Violation"):
                SETUP.prepare_nltk(Path(temporary), {"stopwords": "corpora/stopwords"})
            self.assertEqual(nltk.download.call_count, 1)

    def test_false_download_result_is_failure(self):
        data = SimpleNamespace(path=[], find=Mock(side_effect=LookupError()))
        nltk = SimpleNamespace(data=data, download=Mock(return_value=False))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"nltk": nltk}), \
                patch.object(SETUP.socket, "setdefaulttimeout"):
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                SETUP.prepare_nltk(Path(temporary), {"stopwords": "corpora/stopwords"})

    def test_invalid_shell_mode_fails_before_install(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/setup_eval.sh"), "invalid"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
