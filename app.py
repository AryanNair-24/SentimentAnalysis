"""
app.py
------
Streamlit dashboard for Screenplay NLP Analysis.
Run with: streamlit run app.py
"""

import sys
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# ── PATH SETUP ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from src.data_loader import load_all
from src.analysis import (
    sentiment_arc,
    compare_sentiment_arcs,
    character_voice,
    gender_dialogue_comparison,
    pacing_analysis,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Screenplay NLP",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h1 { font-size: 2rem !important; }
    .stMetric { background: #1a1e2a; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def get_data(data_dir: str):
    return load_all(data_dir)


@st.cache_data(show_spinner="Running gender analysis…")
def get_gender_df(data_dir: str):
    df_chars, _ = get_data(data_dir)
    return gender_dialogue_comparison(df_chars)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Screenplay NLP")
    st.caption("Hollywood dialogue analysis · Cornell Movie Dialogs Corpus")

    data_dir = st.text_input("Data directory", value="./data")

    if not Path(data_dir).exists():
        st.error("Directory not found. Check the path.")
        st.stop()

    df_chars, df_scenes = get_data(data_dir)
    all_movies = sorted(df_chars["movie_name"].unique())

    st.markdown("---")
    st.markdown(f"**{len(all_movies)} movies** · "
                f"**{df_chars['character'].nunique():,} characters** · "
                f"**{len(df_chars):,} lines**")

    page = st.radio("View", [
        "🏠 Overview",
        "📈 Sentiment Arc",
        "🗣️ Character Voice",
        "⚡ Pacing",
        "♀♂ Gender Analysis",
    ])


# ── PAGES ─────────────────────────────────────────────────────────────────────

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("Hollywood Screenplay Analysis")
    st.markdown("Explore sentiment arcs, character voice patterns, pacing, "
                "and gender representation across 200+ films.")

    col1, col2, col3, col4 = st.columns(4)
    dialog_df = df_chars[df_chars["label"] == "dialog"]

    with col1:
        st.metric("Movies", f"{len(all_movies)}")
    with col2:
        st.metric("Characters", f"{df_chars['character'].nunique():,}")
    with col3:
        st.metric("Dialogue lines", f"{len(dialog_df):,}")
    with col4:
        gendered = dialog_df[dialog_df["gender"].isin(["M", "F"])]
        f_pct = (gendered["gender"] == "F").mean() * 100
        st.metric("Female dialogue %", f"{f_pct:.1f}%")

    st.markdown("---")
    st.subheader("Most dialogue-heavy movies")
    top_movies = (
        dialog_df.groupby("movie_name")
        .size()
        .reset_index(name="dialogue_lines")
        .sort_values("dialogue_lines", ascending=False)
        .head(20)
    )
    fig = px.bar(
        top_movies, x="movie_name", y="dialogue_lines",
        color="dialogue_lines", color_continuous_scale="Blues",
        labels={"movie_name": "Movie", "dialogue_lines": "Dialogue Lines"},
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45,
                      plot_bgcolor="#0c0e14", paper_bgcolor="#0c0e14",
                      font_color="#e8eaf0")
    st.plotly_chart(fig, use_container_width=True)


# ── SENTIMENT ARC ─────────────────────────────────────────────────────────────
elif page == "📈 Sentiment Arc":
    st.title("Sentiment Arc")
    st.markdown("Emotional trajectory through a film's dialogue. "
                "Watch 3-act structure emerge automatically.")

    mode = st.radio("Mode", ["Single film", "Compare films"], horizontal=True)

    if mode == "Single film":
        movie = st.selectbox("Select movie", all_movies)
        n_chunks = st.slider("Granularity (chunks)", 10, 40, 20)

        try:
            arc = sentiment_arc(df_chars, movie, n_chunks)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=arc["position"], y=arc["sentiment"],
                mode="lines+markers",
                line=dict(color="#4f8ef7", width=3),
                marker=dict(size=6),
                hovertext=arc["text_preview"],
                hoverinfo="text+y",
                name=movie,
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#5a6080",
                          annotation_text="neutral")
            fig.update_layout(
                xaxis_title="Story progress (%)",
                yaxis_title="Sentiment (VADER compound)",
                yaxis=dict(range=[-1, 1]),
                plot_bgcolor="#0c0e14", paper_bgcolor="#0c0e14",
                font_color="#e8eaf0",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw data"):
                st.dataframe(arc)
        except ValueError as e:
            st.error(str(e))

    else:
        movies = st.multiselect("Select 2–5 movies to compare", all_movies,
                                max_selections=5,
                                default=all_movies[:3] if len(all_movies) >= 3 else all_movies)
        n_chunks = st.slider("Granularity", 10, 40, 20)

        if movies:
            arcs = compare_sentiment_arcs(df_chars, movies, n_chunks)
            fig  = go.Figure()
            colors = px.colors.qualitative.Set2

            for i, (name, arc) in enumerate(arcs.items()):
                fig.add_trace(go.Scatter(
                    x=arc["position"], y=arc["sentiment"],
                    mode="lines", name=name,
                    line=dict(color=colors[i % len(colors)], width=2.5),
                ))

            fig.add_hline(y=0, line_dash="dash", line_color="#5a6080")
            fig.update_layout(
                xaxis_title="Story progress (%)",
                yaxis_title="Sentiment",
                yaxis=dict(range=[-1, 1]),
                plot_bgcolor="#0c0e14", paper_bgcolor="#0c0e14",
                font_color="#e8eaf0",
                legend=dict(bgcolor="#1a1e2a"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── CHARACTER VOICE ───────────────────────────────────────────────────────────
elif page == "🗣️ Character Voice":
    st.title("Character Voice Analysis")
    st.markdown("Vocabulary richness, sentiment, and word patterns per character.")

    movie = st.selectbox("Select movie", all_movies)
    top_n = st.slider("Top N characters", 3, 15, 8)

    try:
        voice = character_voice(df_chars, movie, top_n)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                voice, x="character", y="vocab_richness",
                color="gender",
                color_discrete_map={"M": "#4f8ef7", "F": "#f7904f", "?": "#888"},
                title="Vocabulary Richness (unique words / total words)",
                labels={"vocab_richness": "Richness", "character": ""},
            )
            fig.update_layout(plot_bgcolor="#0c0e14", paper_bgcolor="#0c0e14",
                              font_color="#e8eaf0")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                voice, x="character", y="avg_sentiment",
                color="gender",
                color_discrete_map={"M": "#4f8ef7", "F": "#f7904f", "?": "#888"},
                title="Average Dialogue Sentiment",
                labels={"avg_sentiment": "Sentiment", "character": ""},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#5a6080")
            fig.update_layout(plot_bgcolor="#0c0e14", paper_bgcolor="#0c0e14",
                              font_color="#e8eaf0")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top words per character")
        cols = st.columns(min(4, len(voice)))
        for i, (_, row) in enumerate(voice.iterrows()):
            with cols[i % len(cols)]:
                words = [w for w, _ in row["top_words"][:6]]
                st.markdown(f"**{row['character']}** ({row['gender']})")
                st.markdown(" · ".join(words))

    except ValueError as e:
        st.error(str(e))


# ── PACING ────────────────────────────────────────────────────────────────────
elif page == "⚡ Pacing":
    st.title("Pacing Analysis")
    st.markdown("Action vs dialogue ratio per segment — high action = fast, tense pacing.")

    movie = st.selectbox("Select movie", [m for m in all_movies
                                          if m in df_scenes["movie_name"].unique()])

    try:
        pacing = pacing_analysis(df_scenes, movie)

        color_map = {"fast": "#f7904f", "balanced": "#4fc97a", "slow": "#4f8ef7"}
        fig = px.bar(
            pacing, x="segment", y="action_ratio",
            color="pacing_label", color_discrete_map=color_map,
            title="Action/Dialogue Ratio by Segment",
            labels={"action_ratio": "Action ratio", "segment": "Segment"},
        )
        fig.add_hline(y=0.6, line_dash="dot", line_color="#f7904f",
                      annotation_text="fast threshold")
        fig.add_hline(y=0.4, line_dash="dot", line_color="#4f8ef7",
                      annotation_text="slow threshold")
        fig.update_layout(plot_bgcolor="#0c0e14", paper_bgcolor="#0c0e14",
                          font_color="#e8eaf0")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pacing, use_container_width=True)

    except ValueError as e:
        st.error(str(e))


# ── GENDER ANALYSIS ───────────────────────────────────────────────────────────
elif page == "♀♂ Gender Analysis":
    st.title("Gender Representation Analysis")
    st.markdown("Female vs male dialogue share, vocabulary richness, and sentiment across all films.")

    gender_df = get_gender_df(data_dir)

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_f = gender_df["female_pct"].mean()
        st.metric("Avg female dialogue %", f"{avg_f:.1f}%")
    with col2:
        bechdel_rate = gender_df["bechdel_proxy"].mean() * 100
        st.metric("Bechdel proxy pass rate", f"{bechdel_rate:.1f}%")
    with col3:
        avg_vocab_gap = (gender_df["male_vocab"] - gender_df["female_vocab"]).mean()
        st.metric("Avg vocab richness gap (M−F)", f"{avg_vocab_gap:+.4f}")

    st.markdown("---")

    # Distribution of female dialogue %
    fig = px.histogram(
        gender_df, x="female_pct", nbins=30,
        title="Distribution: Female Dialogue % Across All Films",
        labels={"female_pct": "Female dialogue (%)"},
        color_discrete_sequence=["#f7904f"],
    )
    fig.add_vline(x=50, line_dash="dash", line_color="#888",
                  annotation_text="50% parity")
    fig.update_layout(plot_bgcolor="#0c0e14", paper_bgcolor="#0c0e14",
                      font_color="#e8eaf0")
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: female % vs vocab richness
    fig2 = px.scatter(
        gender_df, x="female_pct", y="female_vocab",
        hover_name="movie_name",
        color="bechdel_proxy",
        color_discrete_map={True: "#4fc97a", False: "#f74f4f"},
        title="Female Dialogue % vs Vocabulary Richness",
        labels={"female_pct": "Female dialogue (%)",
                "female_vocab": "Female vocab richness",
                "bechdel_proxy": "Bechdel proxy"},
        size="total_lines", size_max=18,
    )
    fig2.update_layout(plot_bgcolor="#0c0e14", paper_bgcolor="#0c0e14",
                       font_color="#e8eaf0")
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Full data table"):
        st.dataframe(
            gender_df.sort_values("female_pct", ascending=False),
            use_container_width=True,
        )
