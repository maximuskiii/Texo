#!/usr/bin/env python3
"""
Evaluate a Texo checkpoint on all UniMER-Test splits.

Usage:
    python scripts/python/texo_eval.py \
        --ckpt ./outputs/.../checkpoints/last.ckpt \
        --test_root ./data/dataset/hf_datasets/UniMER-Test

Run from the Texo repo root.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import lightning as L
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from task import FormulaNetLit
from texo.data.dataset import MERDatasetHF
from texo.data.processor import EvalMERImageProcessor, TextProcessor
from texo.data.sampler import SortedSampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint")
    p.add_argument(
        "--test_root",
        default="./data/dataset/hf_datasets/UniMER-Test",
        help="Directory containing spe/cpe/sce/hwe subdirs",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output", default="results/texo_eval.json")
    p.add_argument("--device", default="gpu" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    test_root = Path(args.test_root)

    print(f"Loading checkpoint: {ckpt_path}")
    model = FormulaNetLit.load_from_checkpoint(str(ckpt_path))
    model.eval()

    image_size = {"width": 384, "height": 384}
    image_processor = EvalMERImageProcessor(image_size=image_size)
    text_processor = TextProcessor({
        "tokenizer_path": model.tokenizer_path,
        "tokenizer_config": {
            "add_special_tokens": True,
            "max_length": 1024,
            "padding": "longest",
            "truncation": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
            "pad_to_multiple_of": 8,
        },
    })

    trainer = L.Trainer(
        accelerator=args.device,
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
    )

    all_results = {}
    split_dirs = sorted(p for p in test_root.iterdir() if p.is_dir())

    for split_dir in split_dirs:
        split_name = split_dir.name
        print(f"\n=== Evaluating {split_name} ===")
        dataset = MERDatasetHF(
            dataset_path=str(split_dir),
            image_processor=image_processor,
            text_processor=text_processor,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            sampler=SortedSampler(dataset.labels_length, sort_key=lambda x: x),
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        )
        results = trainer.test(model, dataloaders=loader, verbose=True)
        all_results[split_name] = results[0] if results else {}

    print("\n=== Summary ===")
    for split, metrics in all_results.items():
        bleu = metrics.get("test_BLEU", float("nan"))
        ed = metrics.get("test_edit_distance", float("nan"))
        print(f"  {split:<6}  BLEU={bleu:.4f}  edit_distance={ed:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"ckpt": str(ckpt_path), "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
