"""Compare two SDXL bench --capture-update directories."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

import torch


def stats(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError("capture keys differ")
    error_sq = ref_sq = candidate_sq = dot = 0.0
    max_abs = missing_mismatch = zero_mismatch = 0
    for name, left in reference.items():
        right = candidate[name]
        if left is None or right is None:
            missing_mismatch += (left is None) != (right is None)
            continue
        left, right = left.float().flatten(), right.float().flatten()
        if left.shape != right.shape or not (torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise ValueError(f"invalid tensor pair for {name}")
        delta = left - right
        error_sq += delta.dot(delta).item()
        ref_sq += left.dot(left).item()
        candidate_sq += right.dot(right).item()
        dot += left.dot(right).item()
        max_abs = max(max_abs, delta.abs().max().item())
        zero_mismatch += (left.count_nonzero().item() == 0) != (right.count_nonzero().item() == 0)
    return {
        "relative_l2": math.sqrt(error_sq / ref_sq) if ref_sq else (None if error_sq else 0.0),
        "max_absolute": max_abs,
        "cosine": dot / math.sqrt(ref_sq * candidate_sq) if ref_sq and candidate_sq else None,
        "missing_tensor_mismatches": missing_mismatch,
        "zero_tensor_mismatches": zero_mismatch,
    }


def load(directory, name):
    return torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    inputs = []
    for directory in (args.reference, args.candidate):
        forward = load(directory, "forward")
        text = forward.pop("text_conds")
        forward.update({f"text_{index}": cond for index, cond in enumerate(text)})
        forward.update(forward.pop("sizes"))
        forward["target"] = load(directory, "target")
        inputs.append(forward)
    report = {name: stats({name: left}, {name: inputs[1][name]}) for name, left in inputs[0].items()}
    for kind in ("initial", "gradients", "deltas"):
        reference, candidate = (load(directory, kind) for directory in (args.reference, args.candidate))
        report[kind] = stats(reference, candidate)
        if kind == "initial":
            continue
        blocks = defaultdict(list)
        for name in reference:
            block = re.search(r"(input_blocks_\d+|middle_block|output_blocks_\d+)", name)
            blocks[block[0] if block else "other"].append(name)
        report[f"{kind}_by_block"] = {
            block: stats({name: reference[name] for name in names}, {name: candidate[name] for name in names})
            for block, names in sorted(blocks.items())
        }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
