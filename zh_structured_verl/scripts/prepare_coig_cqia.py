#!/usr/bin/env python3
"""Prepare COIG-CQIA style data into verl SFT parquet format (messages column)."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


def find_splits(dataset: DatasetDict | Dataset) -> dict[str, Dataset]:
    if isinstance(dataset, DatasetDict):
        out: dict[str, Dataset] = {}
        for key, ds in dataset.items():
            lower = key.lower()
            if lower == "validation":
                out["dev"] = ds
            elif lower in {"train", "dev", "test"}:
                out[lower] = ds
        if not out:
            first_key = next(iter(dataset.keys()))
            out["train"] = dataset[first_key]
        return out
    return {"train": dataset}


def first_non_empty(ex: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        val = ex.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if s:
            return s
    return ""


def build_user_message(instruction: str, input_text: str) -> str:
    if input_text:
        return f"{instruction}\\n\\n补充信息：{input_text}"
    return instruction


def build_rows(dataset: Dataset, instruction_keys: list[str], input_keys: list[str], output_keys: list[str], seed: int, max_samples: int | None) -> list[dict[str, Any]]:
    indices = list(range(len(dataset)))
    if max_samples is not None and max_samples > 0 and len(indices) > max_samples:
        random.Random(seed).shuffle(indices)
        indices = indices[:max_samples]

    rows: list[dict[str, Any]] = []
    for idx in indices:
        ex = dataset[int(idx)]
        instruction = first_non_empty(ex, instruction_keys)
        input_text = first_non_empty(ex, input_keys)
        answer = first_non_empty(ex, output_keys)

        if not instruction or not answer:
            continue

        user_msg = build_user_message(instruction, input_text)
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
    return rows


def write_parquet(rows: list[dict[str, Any]], path: str) -> int:
    if not rows:
        return 0
    pd.DataFrame(rows).to_parquet(path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="BAAI/COIG-CQIA")
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--output_dir", default="./zh_structured_verl/data/coig_cqia")
    parser.add_argument("--instruction_keys", default="instruction,query,prompt")
    parser.add_argument("--input_keys", default="input,context")
    parser.add_argument("--output_keys", default="output,response,answer,target")
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_dev_samples", type=int, default=-1)
    parser.add_argument("--max_test_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.local_dataset_path:
        dataset = load_dataset(args.local_dataset_path)
    else:
        dataset = load_dataset(args.dataset_name)

    instruction_keys = [k.strip() for k in args.instruction_keys.split(",") if k.strip()]
    input_keys = [k.strip() for k in args.input_keys.split(",") if k.strip()]
    output_keys = [k.strip() for k in args.output_keys.split(",") if k.strip()]

    split_map = find_splits(dataset)

    max_map = {
        "train": args.max_train_samples,
        "dev": args.max_dev_samples,
        "test": args.max_test_samples,
    }

    stats: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "splits": {},
    }

    for split_name in ["train", "dev", "test"]:
        if split_name not in split_map:
            continue
        max_samples = max_map[split_name]
        max_samples = None if max_samples < 0 else max_samples

        rows = build_rows(
            dataset=split_map[split_name],
            instruction_keys=instruction_keys,
            input_keys=input_keys,
            output_keys=output_keys,
            seed=args.seed,
            max_samples=max_samples,
        )

        out_path = os.path.join(args.output_dir, f"sft_{split_name}.parquet")
        count = write_parquet(rows, out_path)
        stats["splits"][split_name] = {
            "rows": count,
            "path": out_path if count else None,
        }

    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
