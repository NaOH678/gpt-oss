#!/usr/bin/env python3
"""Build a mixed SFT parquet from MASSIVE and COIG parquet files."""

from __future__ import annotations

import argparse
import json

import pandas as pd


def sample_df(df: pd.DataFrame, n: int, seed: int, allow_oversample: bool) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0]
    replace = allow_oversample and n > len(df)
    if not replace and n > len(df):
        n = len(df)
    return df.sample(n=n, replace=replace, random_state=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--massive_train", required=True)
    parser.add_argument("--coig_train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--massive_ratio", type=float, default=0.7)
    parser.add_argument("--coig_ratio", type=float, default=0.3)
    parser.add_argument("--total_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_oversample", action="store_true")
    args = parser.parse_args()

    df_massive = pd.read_parquet(args.massive_train)
    df_coig = pd.read_parquet(args.coig_train)

    ratio_sum = args.massive_ratio + args.coig_ratio
    if ratio_sum <= 0:
        raise ValueError("massive_ratio + coig_ratio must be > 0")

    massive_ratio = args.massive_ratio / ratio_sum

    if args.total_samples < 0:
        total = len(df_massive) + len(df_coig)
    else:
        total = args.total_samples

    n_massive = int(total * massive_ratio)
    n_coig = total - n_massive

    part_massive = sample_df(df_massive, n_massive, args.seed, args.allow_oversample)
    part_coig = sample_df(df_coig, n_coig, args.seed + 1, args.allow_oversample)

    mixed = pd.concat([part_massive, part_coig], axis=0, ignore_index=True)
    mixed = mixed.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    mixed.to_parquet(args.output)

    report = {
        "output": args.output,
        "rows": len(mixed),
        "massive_rows": len(part_massive),
        "coig_rows": len(part_coig),
        "massive_source_size": len(df_massive),
        "coig_source_size": len(df_coig),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
