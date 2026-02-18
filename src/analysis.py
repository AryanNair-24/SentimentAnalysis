"""
analysis.py
-----------
NLP analysis functions. All return dicts/DataFrames ready for Plotly charts.

Functions:
    sentiment_arc(df_chars, movie_name, n_chunks=20)
    compare_sentiment_arcs(df_chars, movie_names)
    character_voice(df_chars, movie_name, top_n=5)
    gender_dialogue_comparison(df_chars)
    pacing_analysis(df_scenes, movie_name)
"""

import re
import pandas as pd
import numpy as np
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Lazy-load NLTK stopwords (downloads once)
import nltk
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

VADER = SentimentIntensityAnalyzer()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _dialog_only(df: pd.DataFrame, movie_name: str | None = None) -> pd.DataFrame:
    """Filter to dialog rows, optionally for one movie."""
    mask = df["label"] == "dialog"
    if movie_name:
        mask &= df["movie_name"] == movie_name
    return df[mask].copy()


def _score(text: str) -> float:
    """VADER compound score for a string."""
    return VADER.polarity_scores(str(text))["compound"]


def _vocab_richness(texts: list[str]) -> float:
    """Type-token ratio: unique words / total words."""
    words = " ".join(texts).lower().split()
    words = [w for w in words if w.isalpha() and w not in STOPWORDS]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _top_words(texts: list[str], n: int = 10) -> list[tuple[str, int]]:
    """Most frequent non-stopwords."""
    words = " ".join(texts).lower().split()
    words = [re.sub(r"[^a-z]", "", w) for w in words]
    words = [w for w in words if w and w not in STOPWORDS and len(w) > 2]
    return Counter(words).most_common(n)


# ── 1. SENTIMENT ARC ──────────────────────────────────────────────────────────

def sentiment_arc(df_chars: pd.DataFrame, movie_name: str, n_chunks: int = 20) -> pd.DataFrame:
    """
    Split a movie's dialogue into n_chunks and score each chunk with VADER.

    Returns DataFrame:
        chunk (int), position (0–100 float), sentiment (float), text_preview (str)
    """
    dialog = _dialog_only(df_chars, movie_name)
    if dialog.empty:
        raise ValueError(f"No dialog found for '{movie_name}'")

    texts  = dialog["text"].tolist()
    chunks = np.array_split(texts, min(n_chunks, len(texts)))

    rows = []
    for i, chunk in enumerate(chunks):
        combined = " ".join(chunk)
        rows.append({
            "chunk":        i + 1,
            "position":     round((i / len(chunks)) * 100, 1),
            "sentiment":    round(_score(combined), 4),
            "text_preview": combined[:120] + "…",
        })

    return pd.DataFrame(rows)


def compare_sentiment_arcs(df_chars: pd.DataFrame, movie_names: list[str],
                            n_chunks: int = 20) -> dict[str, pd.DataFrame]:
    """
    Run sentiment_arc for multiple movies.
    Returns dict: {movie_name: arc_df}
    """
    return {m: sentiment_arc(df_chars, m, n_chunks) for m in movie_names}


# ── 2. CHARACTER VOICE ANALYSIS ───────────────────────────────────────────────

def character_voice(df_chars: pd.DataFrame, movie_name: str,
                    top_n: int = 8) -> pd.DataFrame:
    """
    Analyse top N characters by dialogue volume in a movie.

    Returns DataFrame:
        character, gender, line_count, word_count, vocab_richness,
        avg_sentiment, top_words
    """
    dialog = _dialog_only(df_chars, movie_name)
    rows   = []

    # Rank characters by line count
    top_chars = dialog["character"].value_counts().head(top_n).index

    for char in top_chars:
        char_lines = dialog[dialog["character"] == char]["text"].tolist()
        gender     = dialog[dialog["character"] == char]["gender"].iloc[0] \
                     if "gender" in dialog.columns else "?"

        rows.append({
            "character":     char,
            "gender":        gender if pd.notna(gender) else "?",
            "line_count":    len(char_lines),
            "word_count":    sum(len(t.split()) for t in char_lines),
            "vocab_richness": round(_vocab_richness(char_lines), 4),
            "avg_sentiment": round(np.mean([_score(t) for t in char_lines]), 4),
            "top_words":     _top_words(char_lines, n=8),
        })

    return pd.DataFrame(rows).sort_values("word_count", ascending=False)


# ── 3. GENDER DIALOGUE COMPARISON ────────────────────────────────────────────

def gender_dialogue_comparison(df_chars: pd.DataFrame) -> pd.DataFrame:
    """
    Compare male vs female characters across all movies.

    Returns DataFrame per movie:
        movie_name, male_lines, female_lines, male_pct, female_pct,
        male_vocab, female_vocab, male_sentiment, female_sentiment,
        bechdel_proxy (bool: ≥2 female chars with ≥10 lines each)
    """
    dialog = df_chars[df_chars["label"] == "dialog"].copy()
    dialog = dialog[dialog["gender"].isin(["M", "F"])]

    rows = []
    for movie, grp in dialog.groupby("movie_name"):
        m = grp[grp["gender"] == "M"]
        f = grp[grp["gender"] == "F"]
        total = len(grp)

        # Bechdel proxy: ≥2 named female chars with meaningful dialogue
        female_chars_active = (
            f.groupby("character")["text"].count() >= 10
        ).sum()

        rows.append({
            "movie_name":       movie,
            "male_lines":       len(m),
            "female_lines":     len(f),
            "male_pct":         round(len(m) / total * 100, 1) if total else 0,
            "female_pct":       round(len(f) / total * 100, 1) if total else 0,
            "male_vocab":       round(_vocab_richness(m["text"].tolist()), 4),
            "female_vocab":     round(_vocab_richness(f["text"].tolist()), 4),
            "male_sentiment":   round(m["text"].apply(_score).mean(), 4) if len(m) else 0,
            "female_sentiment": round(f["text"].apply(_score).mean(), 4) if len(f) else 0,
            "bechdel_proxy":    female_chars_active >= 2,
            "total_lines":      total,
        })

    df = pd.DataFrame(rows).sort_values("total_lines", ascending=False)
    print(f"✓ Gender analysis: {len(df)} movies | "
          f"Bechdel proxy pass rate: {df['bechdel_proxy'].mean()*100:.1f}%")
    return df


# ── 4. PACING ANALYSIS ────────────────────────────────────────────────────────

def pacing_analysis(df_scenes: pd.DataFrame, movie_name: str) -> pd.DataFrame:
    """
    Measure pacing across a movie using scene annotations.

    Pacing signal: ratio of 'text' (action) lines to dialog lines per segment.
    High action ratio = fast/tense. High dialog ratio = slower/talky.

    Returns DataFrame:
        segment, action_lines, dialog_lines, action_ratio, pacing_label
    """
    movie = df_scenes[df_scenes["movie_name"] == movie_name].copy()
    if movie.empty:
        raise ValueError(f"No annotation data for '{movie_name}'")

    rows = []
    for seg_idx, seg in movie.groupby("segment"):
        action = (seg["head_type"] == "heading").sum() + \
                 (seg["text"].str.len() > 20).sum()
        dialog = (seg["head_type"] == "speaker/title").sum()

        total  = action + dialog
        ratio  = action / total if total else 0.5

        rows.append({
            "segment":      seg_idx,
            "action_lines": int(action),
            "dialog_lines": int(dialog),
            "action_ratio": round(ratio, 3),
            "pacing_label": "fast" if ratio > 0.6 else "slow" if ratio < 0.4 else "balanced",
        })

    return pd.DataFrame(rows)


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Smoke test with a fake mini-DataFrame
    fake = pd.DataFrame([
        {"movie_name": "TestFilm", "label": "dialog", "character": "ALICE",
         "gender": "F", "text": "I love this wonderful day!", "segment": 0, "scene": 0},
        {"movie_name": "TestFilm", "label": "dialog", "character": "BOB",
         "gender": "M", "text": "Everything is terrible and I hate it.", "segment": 1, "scene": 0},
        {"movie_name": "TestFilm", "label": "dialog", "character": "ALICE",
         "gender": "F", "text": "Don't be so negative, Bob.", "segment": 1, "scene": 1},
    ])

    arc = sentiment_arc(fake, "TestFilm", n_chunks=2)
    print("Sentiment arc:\n", arc)

    voice = character_voice(fake, "TestFilm")
    print("\nCharacter voice:\n", voice)

    gender = gender_dialogue_comparison(fake)
    print("\nGender comparison:\n", gender)
