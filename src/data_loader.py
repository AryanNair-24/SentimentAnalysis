"""
data_loader.py
--------------
Loads and joins all three data sources:
  1. movie_character_texts/  → per-character dialogue (.txt files)
  2. rule_based_annotations/ → structured scene JSON
  3. character_genders.pickle → gender labels per character per movie

Usage:
    from src.data_loader import load_all
    df_chars, df_scenes = load_all(DATA_DIR)
"""

import os
import re
import json
import pickle
import pandas as pd
from pathlib import Path


# ── 1. CHARACTER TEXT FILES ───────────────────────────────────────────────────

def parse_character_file(filepath: Path) -> list[dict]:
    """
    Parse a single character text file like Alice_Evans_text.txt.
    Returns list of dicts: {segment, scene, label, text}
    """
    records = []
    pattern = re.compile(r"^(\d+)\)\s*(\d+)\)\s*(dialog|text):\s*(.+)$")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                records.append({
                    "segment": int(m.group(1)),
                    "scene":   int(m.group(2)),
                    "label":   m.group(3),
                    "text":    m.group(4).strip(),
                })
    return records


def load_character_texts(char_texts_dir: str | Path) -> pd.DataFrame:
    """
    Walk movie_character_texts/ and load all character files.
    Returns DataFrame with columns:
        movie_name, imdb_id, character, segment, scene, label, text
    """
    char_texts_dir = Path(char_texts_dir)
    rows = []

    for movie_dir in sorted(char_texts_dir.iterdir()):
        if not movie_dir.is_dir():
            continue

        # Folder name format: MovieName_tt1234567
        parts = movie_dir.name.rsplit("_", 1)
        movie_name = parts[0] if len(parts) == 2 else movie_dir.name
        imdb_id    = parts[1] if len(parts) == 2 else None

        for txt_file in sorted(movie_dir.glob("*_text.txt")):
            # Filename format: CharacterName_text.txt
            character = txt_file.stem.replace("_text", "").replace("_", " ")
            records = parse_character_file(txt_file)

            for rec in records:
                rows.append({
                    "movie_name": movie_name,
                    "imdb_id":    imdb_id,
                    "character":  character,
                    **rec,
                })

    df = pd.DataFrame(rows)
    print(f"✓ Character texts loaded: {len(df):,} lines across "
          f"{df['movie_name'].nunique()} movies, "
          f"{df['character'].nunique()} characters")
    return df


# ── 2. RULE-BASED ANNOTATIONS (JSON) ─────────────────────────────────────────

def parse_annotation_file(filepath: Path) -> list[dict]:
    """
    Parse a single rule_based_annotations JSON file.
    Returns flat list of scene elements with segment + scene index.
    """
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    # Structure: list of segments → list of scenes → dict
    for seg_idx, segment in enumerate(data):
        for scene_idx, scene in enumerate(segment):
            head_type = scene.get("head_type", "")
            head_text = scene.get("head_text", {})
            text      = scene.get("text", "")

            record = {
                "segment":   seg_idx,
                "scene":     scene_idx,
                "head_type": head_type,
                "text":      text,
            }

            # Flatten head_text fields
            if isinstance(head_text, dict):
                for k, v in head_text.items():
                    record[f"head_{k}"] = v

            records.append(record)

    return records


def load_annotations(annotations_dir: str | Path) -> pd.DataFrame:
    """
    Walk rule_based_annotations/ and load all JSON files.
    Returns DataFrame with columns:
        movie_name, imdb_id, segment, scene, head_type, text, ...
    """
    annotations_dir = Path(annotations_dir)
    rows = []

    for json_file in sorted(annotations_dir.glob("*.json")):
        # Filename: MovieName_tt1234567.json
        stem  = json_file.stem
        parts = stem.rsplit("_", 1)
        movie_name = parts[0] if len(parts) == 2 else stem
        imdb_id    = parts[1] if len(parts) == 2 else None

        records = parse_annotation_file(json_file)
        for rec in records:
            rows.append({"movie_name": movie_name, "imdb_id": imdb_id, **rec})

    df = pd.DataFrame(rows)
    print(f"✓ Annotations loaded: {len(df):,} scene elements across "
          f"{df['movie_name'].nunique()} movies")
    return df


# ── 3. GENDER PICKLE ──────────────────────────────────────────────────────────

def load_genders(pickle_path: str | Path) -> pd.DataFrame:
    """
    Load character_genders.pickle.
    Returns DataFrame with columns: imdb_id, character, gender
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    rows = []
    for imdb_id, char_list in data.items():
        for entry in char_list:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                rows.append({
                    "imdb_id":   imdb_id,
                    "character": str(entry[0]).replace("_", " "),
                    "gender":    "M" if entry[1] == "actor" else "F",
                })

    df = pd.DataFrame(rows)
    print(f"✓ Genders loaded: {len(df):,} characters "
          f"({(df.gender=='F').sum()} F, {(df.gender=='M').sum()} M)")
    return df


# ── 4. JOIN EVERYTHING ────────────────────────────────────────────────────────

def load_all(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master loader. Pass your top-level data directory.

    Expected layout:
        data_dir/
            movie_character_texts/   ← unzipped
            rule_based_annotations/  ← unzipped
            character_genders.pickle

    Returns:
        df_chars  — character-level dialogue DataFrame (with gender joined)
        df_scenes — scene-level annotation DataFrame
    """
    data_dir = Path(data_dir)

    char_dir  = data_dir / "movie_character_texts"
    anno_dir  = data_dir / "rule_based_annotations"
    pkl_path  = data_dir / "character_genders.pickle"

    # Load each source
    df_chars  = load_character_texts(char_dir)
    df_scenes = load_annotations(anno_dir)
    df_gender = load_genders(pkl_path)

    # Join gender onto character dialogue
    df_chars = df_chars.merge(
        df_gender[["imdb_id", "character", "gender"]],
        on=["imdb_id", "character"],
        how="left",
    )

    unmatched = df_chars["gender"].isna().sum()
    pct = unmatched / len(df_chars) * 100
    print(f"✓ Gender join: {pct:.1f}% unmatched (expected ~10–20% due to name variations)")

    return df_chars, df_scenes


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    df_chars, df_scenes = load_all(data_dir)

    print("\n── df_chars sample ──")
    print(df_chars.head(3).to_string())
    print(f"\nShape: {df_chars.shape}")

    print("\n── df_scenes sample ──")
    print(df_scenes.head(3).to_string())
    print(f"\nShape: {df_scenes.shape}")
