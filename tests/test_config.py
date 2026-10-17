import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfigurationTests(unittest.TestCase):
    def test_graph_coverage(self):
        runtime = load("runtime_config")
        for concurrency in (1, 2, 4, 8, 9, 10, 11, 12, 13, 14, 15):
            config = runtime.graph_config(concurrency)
            self.assertEqual(config["max_cudagraph_capture_size"], 3 * concurrency)
            self.assertEqual(max(config["cudagraph_capture_sizes"]), 3 * concurrency)
        self.assertEqual(runtime.graph_config(14)["cudagraph_capture_sizes"],
                         [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 21, 24, 27, 30, 32, 33, 36, 39, 40, 42])

    def test_invalid_concurrency(self):
        runtime = load("runtime_config")
        for value in (0, -1, 16):
            with self.assertRaises(ValueError):
                runtime.graph_config(value)

    def test_groupwise_int4_only(self):
        prepare = load("prepare_model")
        source = {"quantization_config": {"config_groups": {
            "linear": {"weights": {"type": "int", "num_bits": 4}},
            "embed": {"weights": {"type": "int", "num_bits": 8}, "input_activations": None}}}}
        result = prepare.w4a4_config(source)
        activation = result["quantization_config"]["config_groups"]["linear"]["input_activations"]
        self.assertEqual((activation["num_bits"], activation["group_size"], activation["strategy"]), (4, 128, "group"))
        self.assertTrue(activation["dynamic"] and activation["symmetric"])
        self.assertIsNone(result["quantization_config"]["config_groups"]["embed"]["input_activations"])
        self.assertNotIn("input_activations", source["quantization_config"]["config_groups"]["linear"])


if __name__ == "__main__":
    unittest.main()
