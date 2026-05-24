#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gzip
import json
import pickle
import ast
from tqdm import tqdm


def _safe_listify_desc(x):
    """
    description 필드가
      - 리스트인 경우: 그대로 문자열 결합
      - 문자열인데 "['..', '..']" 형태인 경우: literal_eval로 리스트 변환 시도
      - 일반 문자열: 그대로 사용
      - None/결측: 빈 문자열
    최종적으로 문자열 반환
    """
    if x is None:
        return ""
    # 이미 리스트
    if isinstance(x, list):
        return " ".join(str(s) for s in x if s)
    # 문자열인 경우
    s = str(x)
    s_stripped = s.strip()
    # 문자열로 리스트처럼 보이면 안전 파싱
    if s_stripped.startswith("[") and s_stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(s_stripped)
            if isinstance(parsed, list):
                return " ".join(str(t) for t in parsed if t)
        except Exception:
            pass
    return s

def _safe_join_category(cat):
    """
    category가 리스트면 ', '로 조인, 문자열이면 그대로, 없으면 빈 문자열
    """
    if cat is None:
        return ""
    if isinstance(cat, list):
        return ", ".join(str(c) for c in cat if c)
    return str(cat)

def build_books_name_dict(meta_path: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    name_dict = {"title": {}, "description": {}}

    # JSONL 한 줄씩 읽기
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading meta JSONL"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 필드 안전 추출
            item_id = obj.get("item_id", None)
            title   = obj.get("title", "")
            brand   = obj.get("brand", "")
            category = obj.get("category", [])
            desc    = obj.get("description", "")

            if item_id is None:
                continue  # item_id 없는 레코드는 스킵

            # 정수 캐스팅 방어
            try:
                item_id = int(item_id)
            except Exception:
                continue

            # 포맷팅
            title_str = "" if title is None else str(title).strip()
            brand_str = "" if brand is None else str(brand).strip()
            cat_str   = _safe_join_category(category).strip()
            desc_str  = _safe_listify_desc(desc).strip()

            custom_desc = f"[{brand_str}], [{cat_str}], [{desc_str}]"

            name_dict["title"][item_id] = title_str
            name_dict["description"][item_id] = custom_desc
            #print(custom_desc)
    # gzip + pickle로 저장
    
    with gzip.open(out_path, "wb") as tf:
        pickle.dump(name_dict, tf)
    print(f"✅ Saved name_dict: {out_path} (items: {len(name_dict['title'])})")

if __name__ == "__main__":
    META_PATH = "./data/books/item_meta_2017_kcore10_user_item_split_filtered.json"  # JSONL
    OUT_DIR   = "./data/books"
    OUT_PATH  = os.path.join(OUT_DIR, "books_text_name_dict.json.gz")
    build_books_name_dict(META_PATH, OUT_PATH)
