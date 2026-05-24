#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import os
import pickle

from tqdm import tqdm


def _clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).replace("\t", " ").replace("\n", " ").split()).strip()


def _build_description(record):
    parts = [
        _clean_text(record.get("address")),
        _clean_text(record.get("city")),
        _clean_text(record.get("state")),
        _clean_text(record.get("categories")),
    ]
    if record.get("stars") is not None:
        parts.append(f"stars: {_clean_text(record.get('stars'))}")
    if record.get("review_count") is not None:
        parts.append(f"review_count: {_clean_text(record.get('review_count'))}")
    return ", ".join(part for part in parts if part)


def build_yelp_name_dict(meta_path: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    name_dict = {"title": {}, "description": {}}

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading Yelp item_meta JSONL"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_id = record.get("item_id")
            if item_id is None:
                continue
            try:
                item_id = int(item_id)
            except (TypeError, ValueError):
                continue

            name_dict["title"][item_id] = _clean_text(record.get("name")) or "No Title"
            name_dict["description"][item_id] = _build_description(record)

    with gzip.open(out_path, "wb") as f:
        pickle.dump(name_dict, f)
    print(f"Saved Yelp name_dict: {out_path} (items: {len(name_dict['title'])})")

    print("\nSample items:")
    for item_id in sorted(name_dict["title"])[:3]:
        print(f"- item_id={item_id}")
        print(f"  title: {name_dict['title'][item_id]}")
        print(f"  description: {name_dict['description'][item_id]}")


if __name__ == "__main__":
    META_PATH = "./data/yelp/item_meta_2018_kcore5_user_item_split_filtered.json"
    OUT_DIR = "./data/yelp"
    OUT_PATH = os.path.join(OUT_DIR, "yelp_text_name_dict.json.gz")
    build_yelp_name_dict(META_PATH, OUT_PATH)
