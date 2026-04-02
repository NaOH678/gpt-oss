#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path("/Users/naoh/Desktop/gpt-oss/zh_structured_verl/scripts/evaluate_structured_calls.py")


spec = importlib.util.spec_from_file_location("evaluate_structured_calls", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class EvaluateStructuredCallsTest(unittest.TestCase):
    def test_parse_prediction_variants(self) -> None:
        obj, snippet = module.parse_prediction('{"name":"alarm_set","arguments":{"date":"星期五"}}')
        self.assertIsNotNone(obj)
        self.assertEqual(obj["name"], "alarm_set")
        self.assertEqual(snippet, '{"name":"alarm_set","arguments":{"date":"星期五"}}')

        obj2, snippet2 = module.parse_prediction('好的，结果如下：{"name":"alarm_set","arguments":{}}')
        self.assertIsNotNone(obj2)
        self.assertEqual(obj2["name"], "alarm_set")
        self.assertEqual(snippet2, '{"name":"alarm_set","arguments":{}}')

        obj3, snippet3 = module.parse_prediction('not a json')
        self.assertIsNone(obj3)
        self.assertIsNone(snippet3)

    def test_slot_counter_multiset_behavior(self) -> None:
        pred = module._slot_counter({"tag": ["a", "b", "b"]})
        gold = module._slot_counter({"tag": ["a", "a", "b"]})

        tp = sum((pred & gold).values())
        fp = sum(pred.values()) - tp
        fn = sum(gold.values()) - tp
        f1 = (2 * tp) / (2 * tp + fp + fn)

        self.assertEqual(tp, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)
        self.assertAlmostEqual(f1, 2 / 3, places=6)

    def test_metrics_hand_computable_and_outputs(self) -> None:
        rows = [
            {
                "prompt": [{"role": "user", "content": "星期五早上九点叫醒我"}],
                "responses": ['{"name":"alarm_set","arguments":{"date":"星期五","time":"九点"}}'],
                "reward_model": {"ground_truth": {"name": "alarm_set", "arguments": {"date": "星期五", "time": "九点"}}},
            },
            {
                "prompt": [{"role": "user", "content": "星期五早上九点叫醒我"}],
                "responses": ['答案：{"name":"alarm_set","arguments":{"date":"星期五","time":"九点"}}'],
                "reward_model": {"ground_truth": {"name": "alarm_set", "arguments": {"date": "星期五", "time": "九点"}}},
            },
            {
                "prompt": [{"role": "user", "content": "星期五早上九点叫醒我"}],
                "responses": ["这不是json"],
                "reward_model": {"ground_truth": {"name": "alarm_set", "arguments": {"date": "星期五", "time": "九点"}}},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "responses.parquet"
            out_dir = Path(tmpdir) / "metrics"
            pd.DataFrame(rows).to_parquet(input_path)

            summary_path, by_intent_path, failure_path = module.evaluate_parquet(
                responses_parquet=str(input_path),
                output_dir=str(out_dir),
                response_index=0,
            )

            self.assertTrue(Path(summary_path).exists())
            self.assertTrue(Path(by_intent_path).exists())
            self.assertTrue(Path(failure_path).exists())

            summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))

            self.assertEqual(summary["count"], 3)
            self.assertAlmostEqual(summary["json_parse_rate"], 2 / 3, places=6)
            self.assertAlmostEqual(summary["intent_accuracy"], 2 / 3, places=6)
            self.assertAlmostEqual(summary["slot_f1"], 0.8, places=6)
            self.assertAlmostEqual(summary["schema_pass_rate"], 2 / 3, places=6)
            self.assertAlmostEqual(summary["full_call_exact_match"], 1 / 3, places=6)
            self.assertAlmostEqual(summary["only_json_rate"], 1 / 3, places=6)
            self.assertAlmostEqual(summary["extra_text_rate"], 2 / 3, places=6)

            failure_lines = [line for line in Path(failure_path).read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(failure_lines), 2)


if __name__ == "__main__":
    unittest.main()
