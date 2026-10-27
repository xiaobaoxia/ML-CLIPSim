#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Refresh supervised targets while preserving saved pair membership and order.

Records are matched by (image, p0, p1). Diagnostic task dictionaries are retained
from the input split; winner, mean_vote and num_votes come from the new labels.
"""

import os
import json
import copy
from typing import Dict, Any, List, Tuple


# Default paths and scoring parameters; command-line paths override these defaults.

OLD_SPLIT_DIR = "dataset/pcmp/split"

NEW_LABELS_PATH = "dataset/pcmp/labels.json"

# Output folder
NEW_SPLIT_DIR = "outputs/updated-split"

# Whether to drop samples if they are not found in NEW_LABELS_PATH
DROP_IF_NOT_FOUND_IN_NEW_LABELS = True

# Tie filtering rules:
# Keep tie samples here; filter later in dataset loading if needed
FILTER_TIE_BY_MEAN_VOTE = False
FILTER_INVALID_WINNER = False

# Save filenames
TRAIN_NAME = "labels_train.json"
VAL_NAME = "labels_val.json"
META_NAME = "split_meta.json"


# utils

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_key(d: Dict[str, Any]) -> Tuple[str, str, str]:
    return (str(d["image"]), str(d["p0"]), str(d["p1"]))


def build_new_label_map(new_labels: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out = {}

    for item in new_labels:
        key = make_key(item)
        if key in out:
            raise ValueError(f"Duplicate pair identity: {key}")
        for field in ("winner", "mean_vote", "num_votes"):
            if field not in item:
                raise ValueError(f"Missing {field} for pair {key}")
        out[key] = item

    print(f"[INFO] new labels total          : {len(new_labels)}")
    print(f"[INFO] usable new labels        : {len(out)}")
    return out


def update_one_split(split_items: List[Dict[str, Any]], label_map: Dict[Tuple[str, str, str], Dict[str, Any]], split_name: str):
    updated = []
    kept = 0
    dropped_not_found = 0
    changed = 0

    for item in split_items:
        key = make_key(item)
        if key not in label_map:
            if DROP_IF_NOT_FOUND_IN_NEW_LABELS:
                dropped_not_found += 1
                continue
            else:
                updated.append(item)
                kept += 1
                continue

        new_lab = label_map[key]
        new_item = copy.deepcopy(item)

        # choose which field to update
        old_label = None
        if "label" in new_item:
            old_label = new_item["label"]
            new_item["label"] = new_lab["winner"]
        elif "winner" in new_item:
            old_label = new_item["winner"]
            new_item["winner"] = new_lab["winner"]
        else:
            new_item["label"] = new_lab["winner"]

        # optionally sync some helpful metadata
        new_item["mean_vote"] = float(new_lab["mean_vote"])
        new_item["num_votes"] = int(new_lab["num_votes"])

        if old_label is not None and old_label != new_lab["winner"]:
            changed += 1

        updated.append(new_item)
        kept += 1

    print(f"[INFO] {split_name}: input                  = {len(split_items)}")
    print(f"[INFO] {split_name}: kept                   = {kept}")
    print(f"[INFO] {split_name}: dropped_not_found      = {dropped_not_found}")
    print(f"[INFO] {split_name}: changed_label_count    = {changed}")

    return updated, {
        "input": len(split_items),
        "kept": kept,
        "dropped_not_found": dropped_not_found,
        "changed_label_count": changed,
    }


# main

def main():
    import argparse
    global OLD_SPLIT_DIR, NEW_LABELS_PATH, NEW_SPLIT_DIR, DROP_IF_NOT_FOUND_IN_NEW_LABELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', default=OLD_SPLIT_DIR)
    parser.add_argument('--labels', default=NEW_LABELS_PATH)
    parser.add_argument('--output-dir', default=NEW_SPLIT_DIR)
    parser.add_argument('--keep-missing', action='store_true')
    args = parser.parse_args()
    if os.path.realpath(args.split_dir) == os.path.realpath(args.output_dir):
        parser.error('output-dir must differ from split-dir')
    OLD_SPLIT_DIR, NEW_LABELS_PATH, NEW_SPLIT_DIR = args.split_dir, args.labels, args.output_dir
    DROP_IF_NOT_FOUND_IN_NEW_LABELS = not args.keep_missing
    old_train_path = os.path.join(OLD_SPLIT_DIR, TRAIN_NAME)
    old_val_path = os.path.join(OLD_SPLIT_DIR, VAL_NAME)
    old_meta_path = os.path.join(OLD_SPLIT_DIR, META_NAME)

    new_train_path = os.path.join(NEW_SPLIT_DIR, TRAIN_NAME)
    new_val_path = os.path.join(NEW_SPLIT_DIR, VAL_NAME)
    new_meta_path = os.path.join(NEW_SPLIT_DIR, META_NAME)

    old_train = read_json(old_train_path)
    old_val = read_json(old_val_path)
    old_meta = read_json(old_meta_path) if os.path.isfile(old_meta_path) else {}
    new_labels = read_json(NEW_LABELS_PATH)

    label_map = build_new_label_map(new_labels)

    updated_train, train_stats = update_one_split(old_train, label_map, "train")
    updated_val, val_stats = update_one_split(old_val, label_map, "val")

    new_meta = {
        "source_old_split_dir": OLD_SPLIT_DIR,
        "source_new_labels_path": NEW_LABELS_PATH,
        "output_split_dir": NEW_SPLIT_DIR,
        "filter": {
            "drop_mean_vote_eq_0_5": FILTER_TIE_BY_MEAN_VOTE,
            "drop_winner_not_0_or_1": FILTER_INVALID_WINNER,
            "drop_if_not_found_in_new_labels": DROP_IF_NOT_FOUND_IN_NEW_LABELS,
        },
        "train": train_stats,
        "val": val_stats,
        "num_train": len(updated_train),
        "num_val": len(updated_val),
        "old_meta": old_meta,
    }

    write_json(new_train_path, updated_train)
    write_json(new_val_path, updated_val)
    write_json(new_meta_path, new_meta)

    print(f"[DONE] saved train -> {new_train_path}")
    print(f"[DONE] saved val   -> {new_val_path}")
    print(f"[DONE] saved meta  -> {new_meta_path}")


if __name__ == "__main__":
    main()
