#!/usr/bin/env python3
"""Custom reward for Chinese structured function-calling tasks in verl."""

from __future__ import annotations

import json
import re
from typing import Any

W_JSON = 0.2
W_INTENT = 0.3
W_SLOT = 0.4
W_CONSTRAINT = 0.1


def _normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def _find_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
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
                return text[start : i + 1]

    return None


def _parse_prediction(solution_str: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = solution_str.strip()
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


def _to_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _value_match(pred_val: Any, gold_val: Any) -> bool:
    if isinstance(gold_val, list):
        gold_norm = [_normalize_text(v) for v in gold_val]
        if isinstance(pred_val, list):
            pred_norm = [_normalize_text(v) for v in pred_val]
            return pred_norm == gold_norm
        pred_norm = _normalize_text(pred_val)
        return pred_norm in gold_norm

    if isinstance(pred_val, list):
        pred_norm = [_normalize_text(v) for v in pred_val]
        gold_norm = _normalize_text(gold_val)
        return gold_norm in pred_norm

    return _normalize_text(pred_val) == _normalize_text(gold_val)


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict[str, Any] | None = None) -> dict[str, float]:
    del data_source
    del extra_info

    pred_obj, json_candidate = _parse_prediction(solution_str)
    json_valid = 1.0 if pred_obj is not None else 0.0

    gt = _to_dict(ground_truth)
    gt_name = gt.get("name", "")
    gt_args = _to_dict(gt.get("arguments", {}))

    if pred_obj is None:
        return {
            "score": 0.0,
            "r_json": 0.0,
            "r_intent": 0.0,
            "r_slot": 0.0,
            "r_constraint": 0.0,
            "intent_correct": 0.0,
            "slot_ratio": 0.0,
            "json_valid": 0.0,
            "only_json": 0.0,
            "exact_match": 0.0,
        }

    pred_name = pred_obj.get("name", "")
    pred_args = _to_dict(pred_obj.get("arguments", {}))

    intent_correct = 1.0 if _normalize_text(pred_name) == _normalize_text(gt_name) else 0.0

    required_slots = list(gt_args.keys())
    if required_slots:
        matched = sum(1 for slot in required_slots if _value_match(pred_args.get(slot), gt_args.get(slot)))
        r_slot = matched / max(1, len(required_slots))
    else:
        r_slot = 1.0 if len(pred_args) == 0 else 0.0

    missing_required = sum(1 for slot in required_slots if slot not in pred_args)
    extra_slots = sum(1 for slot in pred_args.keys() if slot not in gt_args)

    only_json = 0.0
    raw = solution_str.strip()
    if json_candidate is not None and raw == json_candidate.strip():
        only_json = 1.0

    r_constraint = 1.0
    if only_json < 1.0:
        r_constraint -= 0.4

    if required_slots:
        r_constraint -= 0.4 * (missing_required / len(required_slots))

    if pred_args:
        r_constraint -= 0.2 * (extra_slots / len(pred_args))
    elif extra_slots > 0:
        r_constraint -= 0.2

    r_constraint = max(0.0, min(1.0, r_constraint))

    score = W_JSON * json_valid + W_INTENT * intent_correct + W_SLOT * r_slot + W_CONSTRAINT * r_constraint

    exact_match = 1.0 if (
        json_valid == 1.0
        and intent_correct == 1.0
        and r_slot == 1.0
        and missing_required == 0
        and extra_slots == 0
        and only_json == 1.0
    ) else 0.0

    return {
        "score": float(score),
        "r_json": float(json_valid),
        "r_intent": float(intent_correct),
        "r_slot": float(r_slot),
        "r_constraint": float(r_constraint),
        "intent_correct": float(intent_correct),
        "slot_ratio": float(r_slot),
        "json_valid": float(json_valid),
        "only_json": float(only_json),
        "exact_match": float(exact_match),
    }
