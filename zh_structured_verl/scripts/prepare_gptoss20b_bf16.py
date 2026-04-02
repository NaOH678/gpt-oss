#!/usr/bin/env python3
"""Convert openai/gpt-oss-20b to a local bf16 checkpoint for verl training."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="openai/gpt-oss-20b")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    quantization_config = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        device_map=args.device_map,
    )

    model.config.attn_implementation = "eager"
    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved bf16 model to: {args.output_dir}")


if __name__ == "__main__":
    main()
