#!/usr/bin/env python3
"""Prepare MASSIVE zh-CN data for verl SFT and GRPO training."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

SLOT_PATTERN = re.compile(r"\[(?P<slot>[^\[\]:]+?)\s*:\s*(?P<value>[^\[\]]+?)\]")

PROMPT_TEMPLATE = (
    "请将下面的中文请求转换为函数调用。\\n"
    "要求：\\n"
    "1) 只输出 JSON，不要输出解释\\n"
    "2) 输出格式为 {\"name\": \"<intent>\", \"arguments\": {...}}\\n"
    "3) arguments 的 key 使用英文槽位名\\n\\n"
    "用户请求：{utt}"
)


@dataclass
class SplitRows:
    sft_rows: list[dict[str, Any]]
    rl_rows: list[dict[str, Any]]
    dropped: int


def parse_slots(annot_utt: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    if not annot_utt:
        return arguments

    for match in SLOT_PATTERN.finditer(annot_utt):
        slot = match.group("slot").strip()
        value = match.group("value").strip()
        if not slot or not value:
            continue

        if slot in arguments:
            prev = arguments[slot]
            if isinstance(prev, list):
                prev.append(value)
            else:
                arguments[slot] = [prev, value]
        else:
            arguments[slot] = value

    return arguments


def build_prompt(utt: str) -> str:
    return PROMPT_TEMPLATE.format(utt=utt)


def find_split_datasets(dataset: DatasetDict | Dataset) -> dict[str, Dataset]:
    """Support both standard split datasets and single-split with partition field."""
    split_map: dict[str, Dataset] = {}

    if isinstance(dataset, DatasetDict):
        keys = set(dataset.keys())
        if {"train", "validation", "test"}.issubset(keys) or {"train", "dev", "test"}.issubset(keys):
            for key in dataset.keys():
                lower = key.lower()
                if lower == "validation":
                    split_map["dev"] = dataset[key]
                elif lower in {"dev", "train", "test"}:
                    split_map[lower] = dataset[key]
        elif "train" in keys:
            train_ds = dataset["train"]
            if "partition" not in train_ds.column_names:
                split_map["train"] = train_ds
            else:
                groups: dict[str, list[int]] = {"train": [], "dev": [], "test": []}
                for idx, row in enumerate(train_ds):
                    p = str(row.get("partition", "train")).lower()
                    if p in {"validation", "valid", "dev"}:
                        groups["dev"].append(idx)
                    elif p == "test":
                        groups["test"].append(idx)
                    else:
                        groups["train"].append(idx)
                for split_name, indices in groups.items():
                    if indices:
                        split_map[split_name] = train_ds.select(indices)
        else:
            first_key = next(iter(dataset.keys()))
            split_map["train"] = dataset[first_key]

    else:
        split_map["train"] = dataset

    return split_map


def build_rows(split_name: str, dataset: Dataset, max_samples: int | None, seed: int) -> SplitRows:
    indices = list(range(len(dataset)))
    if max_samples is not None and max_samples > 0 and len(indices) > max_samples:
        random.Random(seed).shuffle(indices)
        indices = indices[:max_samples]

    sft_rows: list[dict[str, Any]] = []
    rl_rows: list[dict[str, Any]] = []
    dropped = 0

    for idx in indices:
        ex = dataset[int(idx)]
        utt = str(ex.get("utt") or ex.get("utterance") or "").strip()
        intent = str(ex.get("intent") or ex.get("name") or "").strip()
        annot_utt = str(ex.get("annot_utt") or ex.get("annotated_utt") or "").strip()

        if not utt or not intent:
            dropped += 1
            continue

        arguments = parse_slots(annot_utt)
        call_obj = {
            "name": intent,
            "arguments": arguments,
        }
        prompt = build_prompt(utt)
        assistant = json.dumps(call_obj, ensure_ascii=False, separators=(",", ":"))

        sft_rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )

        rl_rows.append(
            {
                "data_source": "massive_zh_structured",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "structured_call",
                "reward_model": {"style": "rule", "ground_truth": call_obj},
                "extra_info": {
                    "split": split_name,
                    "index": int(idx),
                    "utt": utt,
                    "intent": intent,
                    "scenario": ex.get("scenario"),
                },
            }
        )

    return SplitRows(sft_rows=sft_rows, rl_rows=rl_rows, dropped=dropped)


def maybe_write_parquet(rows: list[dict[str, Any]], path: str) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df.to_parquet(path)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="AmazonScience/massive")
    parser.add_argument("--dataset_config", default="zh-CN")
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--output_dir", default="./zh_structured_verl/data/massive_zh")
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_dev_samples", type=int, default=-1)
    parser.add_argument("--max_test_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.local_dataset_path:
        dataset = load_dataset(args.local_dataset_path)
    else:
        dataset = load_dataset(args.dataset_name, args.dataset_config)

    split_datasets = find_split_datasets(dataset)

    max_sample_map = {
        "train": args.max_train_samples,
        "dev": args.max_dev_samples,
        "test": args.max_test_samples,
    }

    stats: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "splits": {},
    }

    for split_name in ["train", "dev", "test"]:
        if split_name not in split_datasets:
            continue

        max_samples = max_sample_map.get(split_name, -1)
        max_samples = None if max_samples is None or max_samples < 0 else max_samples
        result = build_rows(
            split_name=split_name,
            dataset=split_datasets[split_name],
            max_samples=max_samples,
            seed=args.seed,
        )

        sft_out = os.path.join(args.output_dir, f"sft_{split_name}.parquet")
        rl_out = os.path.join(args.output_dir, f"rl_{split_name}.parquet")

        sft_count = maybe_write_parquet(result.sft_rows, sft_out)
        rl_count = maybe_write_parquet(result.rl_rows, rl_out)

        stats["splits"][split_name] = {
            "sft_rows": sft_count,
            "rl_rows": rl_count,
            "dropped": result.dropped,
            "sft_path": sft_out if sft_count else None,
            "rl_path": rl_out if rl_count else None,
        }

    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
