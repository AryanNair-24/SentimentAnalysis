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
import src
import json
import pickle
import pandas as pd
from pathlib import Path


# ── CURATED FILM LIST ─────────────────────────────────────────────────────────
# ~50 well-known films selected for:
#   - Genre variety (thriller, comedy, drama, action, sci-fi)
#   - Strong female leads (good for gender analysis)
#   - Decade spread (1970s–2010s)
#
# Format must match folder names exactly: "Movie Name_imdbid"
# To use ALL movies instead, set CURATED_ONLY = False

CURATED_ONLY = True

CURATED_FILMS = {
    # ── 1970s ──────────────────────────────────────────
    "Alien_0078748",                        # sci-fi, Ripley (strong female lead)
    "Annie Hall_0075686",                   # comedy/drama
    "Apocalypse Now_0078788",               # war/drama
    "Kramer vs Kramer_0079417",             # drama
    "Manhattan_0079522",                    # comedy/drama

    # ── 1980s ──────────────────────────────────────────
    "Aliens_0090605",                       # sci-fi/action, Ripley
    "Back to the Future_0088763",           # sci-fi/comedy
    "Die Hard_0095016",                     # action/thriller
    "Rain Man_0095953",                     # drama
    "Steel Magnolias_0098384",              # drama, female ensemble
    "Tootsie_0084805",                      # comedy, gender themes
    "Working Girl_0096463",                 # drama, strong female lead

    # ── 1990s ──────────────────────────────────────────
    "12 Monkeys_0114746",                   # sci-fi/thriller
    "A Few Good Men_0104257",               # drama/thriller
    "American Beauty_0169547",              # drama
    "Boogie Nights_0118749",                # drama
    "Fargo_0116282",                        # thriller, female lead
    "Fight Club_0137523",                   # thriller/drama
    "Forrest Gump_0109830",                 # drama
    "Goodfellas_0099685",                   # crime/drama
    "LA Confidential_0119488",              # noir/thriller
    "Pulp Fiction_0110912",                 # crime/drama
    "Schindler s List_0108052",             # historical drama
    "Se7en_0114369",                        # thriller
    "Silence of the Lambs_0102926",         # thriller, female lead
    "The Shawshank Redemption_0111161",     # drama
    "Thelma and Louise_0103074",            # drama, female leads
    "Toy Story_0114709",                    # animation/comedy

    # ── 2000s ──────────────────────────────────────────
    "10 Things I Hate About You_0147800",   # comedy/romance, female lead
    "12 Years a Slave_2024544",             # historical drama
    "25th Hour_0307901",                    # drama
    "28 Days Later_0289043",                # horror/thriller
    "Brokeback Mountain_0388795",           # drama
    "Crash_0375679",                        # drama, ensemble
    "Eternal Sunshine of the Spotless Mind_0338013",  # sci-fi/romance
    "Juno_0467406",                         # comedy/drama, female lead
    "Kill Bill Volume 1_0266697",           # action, female lead
    "Million Dollar Baby_0405159",          # drama, female lead
    "Mulholland Drive_0166924",             # thriller, female lead
    "No Country for Old Men_0477348",       # thriller
    "The Dark Knight_0468569",              # action/thriller
    "The Devil Wears Prada_0458352",        # comedy/drama, female lead
    "There Will Be Blood_0469494",          # drama
    "Zodiac_0443706",                       # thriller

    # ── 2010s ──────────────────────────────────────────
    "20th Century Women_4385888",           # drama, female lead
    "Black Swan_0947798",                   # thriller, female lead
    "Bridesmaids_1478338",                  # comedy, female ensemble
    "Gone Girl_2267998",                    # thriller, female lead
    "Her_1798709",                          # sci-fi/romance
    "Inception_1375666",                    # sci-fi/thriller
    "Interstellar_0816692",                 # sci-fi/drama
    "Mad Max Fury Road_1392190",            # action, female lead
    "Moonlight_4975722",                    # drama
    "The Social Network_1285016",           # drama
    "Wild_2305051",                         # drama, female lead
    "Wolf of Wall Street_0993846",          # drama/comedy
}


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

        # Skip movies not in curated list
        if CURATED_ONLY and movie_dir.name not in CURATED_FILMS:
            continue

        # Folder name format: Movie Name_imdbid
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
                    "character_normalized": (character.replace("-", " ")).replace("'", "").upper(),
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
        # Filename: Movie Name_imdbid.json
        stem  = json_file.stem
        parts = stem.rsplit("_", 1)
        movie_name = parts[0] if len(parts) == 2 else stem
        imdb_id    = parts[1] if len(parts) == 2 else None

        # Skip movies not in curated list
        if CURATED_ONLY and json_file.stem not in CURATED_FILMS:
            continue

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
                    "character_normalized": (str(entry[0]).replace("-", " ")).replace("'", "").upper()
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
        df_gender[["imdb_id", "character", "gender", "character_normalized"]],
        on=["imdb_id", "character_normalized"],
        how="left",
    )

    # Clean up character column clash from Step 1 merge
    df_chars = df_chars.drop(columns=["character_y"])
    df_chars = df_chars.rename(columns={"character_x": "character"})

    matched = df_chars[df_chars["gender"].notna()]
    unmatched_chars = df_chars[df_chars["gender"].isna()].copy()

    unmatched_chars["first_name"] = unmatched_chars["character_normalized"].str.split(" ").str[0]

    matched2 = df_gender[df_gender["gender"].notna()]
    unmatched_gender = df_gender[df_gender["gender"].isna()].copy()

    df_gender["first_name"] = df_gender["character_normalized"].str.split(" ").str[0]

    print(unmatched_chars.columns.tolist())
    unmatched_chars = unmatched_chars.drop(columns=["gender", "character"])

    unmatched_chars = unmatched_chars.merge(
        df_gender[["imdb_id", "first_name", "gender"]],
        on=["imdb_id", "first_name"],
        how="left",
    )

    print(unmatched_chars[["movie_name", "character_normalized"]].drop_duplicates().head(20))

    print(f"Matched after Step 1: {len(matched)}")
    print(f"Unmatched after Step 1: {len(unmatched_chars)}")
    print(f"Unmatched after Step 2: {unmatched_chars["gender"].isna().sum()}")

    df_chars = pd.concat([matched, unmatched_chars], ignore_index=True)

    df_chars = df_chars.drop(columns=["character_normalized"])

    # Report the number of rows still unmatched.
    unmatched = df_chars["gender"].isna().sum()
    pct = unmatched / len(df_chars) * 100
    print(f"✓ Gender join: {pct:.1f}% unmatched (expected ~10–20% due to name variations)")

    return df_chars, df_scenes


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_dir = r"C:\Users\owner\Documents\SentimentAnalysis\sentiment_analysis\SentimentAnalysis\data"
    df_chars, df_scenes = load_all(data_dir)

    print("\n── df_chars sample ──")
    print(df_chars.head(3).to_string())
    print(f"\nShape: {df_chars.shape}")

    print("\n── df_scenes sample ──")
    print(df_scenes.head(3).to_string())
    print(f"\nShape: {df_scenes.shape}")
