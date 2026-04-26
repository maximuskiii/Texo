#!/usr/bin/env python3
"""
Evaluate a Texo checkpoint on UniMER-Test splits.

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
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

# ── Texo source ──────────────────────────────────────────────────────────────
_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))

from task import FormulaNetLit
from texo.data.dataset import MERDatasetHF
from texo.data.processor import EvalMERImageProcessor, TextProcessor
from texo.data.sampler import SortedSampler

# ── Shared metrics from got_mer ───────────────────────────────────────────────
_got_mer = Path(__file__).resolve().parents[3] / "got_mer"
sys.path.insert(0, str(_got_mer))
from utils.metrics import compute_metrics


ALL_SPLITS = ["spe", "cpe", "sce", "hwe"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt file")
    p.add_argument(
        "--test_root",
        default="./data/dataset/hf_datasets/UniMER-Test",
        help="Dir containing spe/cpe/sce/hwe subdirs saved via save_to_disk()",
    )
    p.add_argument("--splits", nargs="+", default=ALL_SPLITS)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--output", default="results/texo_eval.json")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.inference_mode()
def eval_split(model, tokenizer, dataset, batch_size, max_new_tokens, num_workers, device):
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=SortedSampler(dataset.labels_length, sort_key=lambda x: x),
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )

    all_preds, all_refs = [], []
    pad_id = tokenizer.pad_token_id

    for batch in tqdm(loader, desc="Generating"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].clone()
        labels[labels == -100] = pad_id
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

        outputs = model.generate(pixel_values, num_beams=1, do_sample=False, max_new_tokens=max_new_tokens)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        all_preds.extend(preds)
        all_refs.extend(refs)

    return all_preds, all_refs


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    test_root = Path(args.test_root)

    print(f"Loading checkpoint: {ckpt_path}")
    lit_model = FormulaNetLit.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    lit_model.eval()
    lit_model.to(args.device)

    tokenizer = lit_model.tokenizer
    image_size = {"width": 384, "height": 384}
    image_processor = EvalMERImageProcessor(image_size=image_size)
    text_processor = TextProcessor({
        "tokenizer_path": lit_model.tokenizer_path,
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

    all_split_results = {}
    all_split_preds = {}

    for split_name in args.splits:
        split_dir = test_root / split_name
        if not split_dir.is_dir():
            print(f"  WARNING: split dir '{split_dir}' not found, skipping")
            continue

        dataset = MERDatasetHF(
            dataset_path=str(split_dir),
            image_processor=image_processor,
            text_processor=text_processor,
        )
        if args.max_samples:
            dataset.dataset = dataset.dataset[: args.max_samples]
            dataset.labels_length = dataset.labels_length[: args.max_samples]

        print(f"\n=== {split_name} ({len(dataset):,} samples) ===")
        preds, refs = eval_split(
            lit_model, tokenizer, dataset,
            args.batch_size, args.max_new_tokens, args.num_workers, args.device,
        )

        preds_dump = Path(args.output).parent / f"texo_preds_{split_name}.json"
        preds_dump.parent.mkdir(parents=True, exist_ok=True)
        with open(preds_dump, "w") as f:
            json.dump([{"pred": p, "ref": r} for p, r in zip(preds, refs)], f)
        print(f"  saved {len(preds)} preds → {preds_dump}")

        metrics = compute_metrics(preds, refs)
        all_split_results[split_name] = metrics
        if args.save_predictions:
            all_split_preds[split_name] = [{"pred": p, "ref": r} for p, r in zip(preds, refs)]

    print("\n=== Summary ===")
    for split, m in all_split_results.items():
        print(
            f"  {split:<6}  NED={m.get('ned', float('nan')):.4f}"
            f"  BLEU={m.get('bleu4', float('nan')):.4f}"
            f"  EM={m.get('exact_match', float('nan')):.4f}"
            f"  CDM={m.get('cdm_f1', float('nan')):.4f}"
            f"  ExpRate={m.get('cdm_exprate', float('nan')):.4f}"
        )

    output = {
        "config": {
            "ckpt": str(ckpt_path),
            "splits": args.splits,
            "max_new_tokens": args.max_new_tokens,
        },
        "results": all_split_results,
    }
    if args.save_predictions:
        output["predictions"] = all_split_preds

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
