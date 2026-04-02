#!/usr/bin/env python3
"""Evaluate structured function-calling outputs for baseline/SFT/RL comparisons."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd

JSON_KEY_SET = {"name", "arguments"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return "<null>"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def _ensure_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)

    # numpy arrays in parquet object columns may expose tolist()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        arr = tolist()
        if isinstance(arr, list):
            return arr
        return [arr]

    return [value]


def _find_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def parse_prediction(raw_output: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = (raw_output or "").strip()
    if not raw:
        return None, None

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, raw
    except Exception:
        pass

    candidate = _find_first_json_object(raw)
    if candidate is None:
        return None, None

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj, candidate
    except Exception:
        return None, candidate

    return None, candidate


def parse_ground_truth(reward_model: Any) -> dict[str, Any]:
    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth")
        if isinstance(gt, dict):
            return gt
        if isinstance(gt, str):
            try:
                parsed = json.loads(gt)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}

    if isinstance(reward_model, str):
        try:
            parsed = json.loads(reward_model)
            if isinstance(parsed, dict):
                maybe_gt = parsed.get("ground_truth")
                if isinstance(maybe_gt, dict):
                    return maybe_gt
                if isinstance(maybe_gt, str):
                    inner = json.loads(maybe_gt)
                    return inner if isinstance(inner, dict) else {}
                return parsed
        except Exception:
            return {}

    return {}


def _slot_counter(arguments: dict[str, Any]) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for slot, value in arguments.items():
        slot_name = _normalize_text(slot)
        values = _ensure_list(value)
        if not values:
            counter[(slot_name, "<empty>")] += 1
            continue
        for item in values:
            counter[(slot_name, _normalize_text(item))] += 1
    return counter


def _extract_response(cell: Any, response_index: int) -> str:
    if isinstance(cell, str):
        return cell

    values = _ensure_list(cell)
    if not values:
        return ""

    if response_index < 0:
        response_index = 0

    idx = response_index if response_index < len(values) else 0
    selected = values[idx]
    return "" if selected is None else str(selected)


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt

    prompt_list = _ensure_list(prompt)
    parts: list[str] = []
    for item in prompt_list:
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role or content:
                parts.append(f"{role}: {content}".strip())
        else:
            text = str(item).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


@dataclass
class MetricAccumulator:
    count: int = 0
    json_valid_count: int = 0
    intent_correct_count: int = 0
    schema_pass_count: int = 0
    full_call_em_count: int = 0
    only_json_count: int = 0
    extra_text_count: int = 0
    slot_tp: int = 0
    slot_fp: int = 0
    slot_fn: int = 0

    def update(self, sample: dict[str, Any]) -> None:
        self.count += 1
        self.json_valid_count += int(sample["json_valid"])
        self.intent_correct_count += int(sample["intent_correct"])
        self.schema_pass_count += int(sample["schema_pass"])
        self.full_call_em_count += int(sample["full_call_em"])
        self.only_json_count += int(sample["only_json"])
        self.extra_text_count += int(sample["extra_text"])
        self.slot_tp += int(sample["slot_tp"])
        self.slot_fp += int(sample["slot_fp"])
        self.slot_fn += int(sample["slot_fn"])

    def to_metrics(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "json_parse_rate": 0.0,
                "intent_accuracy": 0.0,
                "slot_f1": 0.0,
                "schema_pass_rate": 0.0,
                "full_call_exact_match": 0.0,
                "only_json_rate": 0.0,
                "extra_text_rate": 0.0,
                "slot_tp": 0,
                "slot_fp": 0,
                "slot_fn": 0,
            }

        slot_denom = 2 * self.slot_tp + self.slot_fp + self.slot_fn
        slot_f1 = 0.0 if slot_denom == 0 else (2 * self.slot_tp) / slot_denom

        return {
            "count": self.count,
            "json_parse_rate": self.json_valid_count / self.count,
            "intent_accuracy": self.intent_correct_count / self.count,
            "slot_f1": slot_f1,
            "schema_pass_rate": self.schema_pass_count / self.count,
            "full_call_exact_match": self.full_call_em_count / self.count,
            "only_json_rate": self.only_json_count / self.count,
            "extra_text_rate": self.extra_text_count / self.count,
            "slot_tp": self.slot_tp,
            "slot_fp": self.slot_fp,
            "slot_fn": self.slot_fn,
        }


def _schema_pass(pred_obj: dict[str, Any] | None) -> bool:
    if pred_obj is None:
        return False
    if set(pred_obj.keys()) != JSON_KEY_SET:
        return False

    name = pred_obj.get("name")
    arguments = pred_obj.get("arguments")

    if not isinstance(name, str) or not name.strip():
        return False
    if not isinstance(arguments, dict):
        return False
    return True


def _slot_value_mismatch(pred_args: dict[str, Any], gold_args: dict[str, Any]) -> bool:
    shared = set(pred_args.keys()) & set(gold_args.keys())
    for key in shared:
        pred_c = _slot_counter({key: pred_args.get(key)})
        gold_c = _slot_counter({key: gold_args.get(key)})
        if pred_c != gold_c:
            return True
    return False


def _evaluate_one(raw_output: str, gold_obj: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, list[str], str | None]:
    pred_obj, json_snippet = parse_prediction(raw_output)

    json_valid = pred_obj is not None
    schema_pass = _schema_pass(pred_obj)

    pred_name = "" if pred_obj is None else str(pred_obj.get("name", ""))
    gold_name = str(gold_obj.get("name", ""))
    intent_correct = _normalize_text(pred_name) == _normalize_text(gold_name)

    pred_args = _ensure_dict(pred_obj.get("arguments") if pred_obj else {})
    gold_args = _ensure_dict(gold_obj.get("arguments", {}))

    pred_slots = _slot_counter(pred_args)
    gold_slots = _slot_counter(gold_args)

    slot_tp = sum((pred_slots & gold_slots).values())
    slot_fp = sum(pred_slots.values()) - slot_tp
    slot_fn = sum(gold_slots.values()) - slot_tp
    slot_em = slot_fp == 0 and slot_fn == 0

    raw_stripped = (raw_output or "").strip()
    only_json = json_valid and json_snippet is not None and raw_stripped == json_snippet.strip()
    extra_text = not only_json

    full_call_em = bool(json_valid and intent_correct and schema_pass and slot_em and only_json)

    labels: list[str] = []
    if not json_valid:
        labels.append("json_parse_error")
    else:
        if not schema_pass:
            labels.append("schema_error")
        if not intent_correct:
            labels.append("intent_error")
        if slot_fn > 0:
            labels.append("slot_missing")
        if slot_fp > 0:
            labels.append("slot_extra")
        if _slot_value_mismatch(pred_args, gold_args):
            labels.append("slot_value_mismatch")
    if extra_text:
        labels.append("extra_text")

    sample_metrics = {
        "json_valid": json_valid,
        "intent_correct": intent_correct,
        "schema_pass": schema_pass,
        "full_call_em": full_call_em,
        "only_json": only_json,
        "extra_text": extra_text,
        "slot_tp": slot_tp,
        "slot_fp": slot_fp,
        "slot_fn": slot_fn,
    }
    return sample_metrics, pred_obj, labels, json_snippet


def evaluate_dataframe(
    df: pd.DataFrame,
    response_index: int,
    prompt_key: str,
    responses_key: str,
    reward_model_key: str,
    max_failure_cases: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    overall = MetricAccumulator()
    by_intent: dict[str, MetricAccumulator] = {}
    failure_cases: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        raw_output = _extract_response(row.get(responses_key), response_index=response_index)
        prompt_text = _prompt_to_text(row.get(prompt_key))

        gold_obj = parse_ground_truth(row.get(reward_model_key))
        gold_intent = str(gold_obj.get("name", "<unknown>"))

        metrics, pred_obj, labels, json_snippet = _evaluate_one(raw_output, gold_obj)

        overall.update(metrics)
        if gold_intent not in by_intent:
            by_intent[gold_intent] = MetricAccumulator()
        by_intent[gold_intent].update(metrics)

        if not metrics["full_call_em"] and len(failure_cases) < max_failure_cases:
            failure_cases.append(
                {
                    "index": int(idx),
                    "intent": gold_intent,
                    "error_labels": labels,
                    "prompt": prompt_text,
                    "gold": gold_obj,
                    "prediction_raw": raw_output,
                    "prediction_json_snippet": json_snippet,
                    "prediction_parsed": pred_obj,
                }
            )

    summary = overall.to_metrics()
    summary["response_index"] = response_index
    summary["max_failure_cases"] = max_failure_cases

    by_intent_metrics = {
        intent: acc.to_metrics() for intent, acc in sorted(by_intent.items(), key=lambda x: x[0])
    }

    return summary, by_intent_metrics, failure_cases


def evaluate_parquet(
    responses_parquet: str,
    output_dir: str,
    response_index: int = 0,
    prompt_key: str = "prompt",
    responses_key: str = "responses",
    reward_model_key: str = "reward_model",
    max_failure_cases: int = 1000,
) -> tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_parquet(responses_parquet)

    summary, by_intent, failures = evaluate_dataframe(
        df=df,
        response_index=response_index,
        prompt_key=prompt_key,
        responses_key=responses_key,
        reward_model_key=reward_model_key,
        max_failure_cases=max_failure_cases,
    )

    summary_path = os.path.join(output_dir, "metrics_summary.json")
    by_intent_path = os.path.join(output_dir, "metrics_by_intent.json")
    failure_path = os.path.join(output_dir, "failure_cases.jsonl")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(by_intent_path, "w", encoding="utf-8") as f:
        json.dump(by_intent, f, ensure_ascii=False, indent=2)

    with open(failure_path, "w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return summary_path, by_intent_path, failure_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses_parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--response_index", type=int, default=0)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--responses_key", default="responses")
    parser.add_argument("--reward_model_key", default="reward_model")
    parser.add_argument("--max_failure_cases", type=int, default=1000)
    args = parser.parse_args()

    summary_path, by_intent_path, failure_path = evaluate_parquet(
        responses_parquet=args.responses_parquet,
        output_dir=args.output_dir,
        response_index=args.response_index,
        prompt_key=args.prompt_key,
        responses_key=args.responses_key,
        reward_model_key=args.reward_model_key,
        max_failure_cases=args.max_failure_cases,
    )

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote by-intent metrics: {by_intent_path}")
    print(f"Wrote failure cases: {failure_path}")


if __name__ == "__main__":
    main()
