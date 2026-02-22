# Screenplay NLP Analysis Pipeline

> An end-to-end NLP pipeline analyzing sentiment arcs, character voice patterns, pacing, and gender representation across 200+ Hollywood screenplays.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![AWS](https://img.shields.io/badge/AWS-App%20Runner-orange?logo=amazon-aws)](https://aws.amazon.com/apprunner/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://docker.com)

**[Live Demo →](https://sentimentanalysis-mkmh8qvgqwpfay3yxxzeer.streamlit.app/)**

---

## What it does

This pipeline ingests structured screenplay data (Cornell Movie Dialogs Corpus) and runs four NLP analyses:

| Analysis                  | What it reveals                                                             |
| ------------------------- | --------------------------------------------------------------------------- |
| **Sentiment Arc**         | Emotional trajectory through a film — 3-act structure emerges automatically |
| **Character Voice**       | Vocabulary richness, sentiment, and word patterns per character             |
| **Pacing Analysis**       | Action vs dialogue ratio per scene segment                                  |
| **Gender Representation** | Female/male dialogue share, vocab richness gap, Bechdel proxy score         |

## Key findings

- **Female characters speak 33% of dialogue** on average across the dataset
- **Male characters have 0.1021% higher vocabulary richness** than female characters (or vice versa)
- **Only 33.3% of films pass the Bechdel proxy test** (≥2 female chars with meaningful dialogue)
<!-- - **[Interesting film]'s sentiment arc** shows the sharpest single-act drop in the dataset -->

## Architecture

```
Cornell Movie Dialogs Corpus (Kaggle)
        │
        ├── movie_character_texts/    Per-character .txt files
        ├── rule_based_annotations/   Structured scene JSON
        └── character_genders.pickle  Gender labels
        │
        ▼
   src/data_loader.py    ← Joins all 3 sources into clean DataFrames
        │
        ▼
   src/analysis.py       ← VADER sentiment · NLTK vocab · pacing metrics
        │
        ▼
   app.py (Streamlit)    ← Interactive Plotly dashboard
        │
        ▼
   Docker container → AWS App Runner → Live URL
```

## Tech stack

- **NLP:** VADER Sentiment, NLTK
- **Data:** Pandas, NumPy
- **Viz:** Plotly, Streamlit
- **Deploy:** Docker, AWS App Runner
- **Data source:** [Cornell Movie Dialogs Corpus](https://www.kaggle.com/datasets/Cornell-University/movie-dialog-corpus)

## Run locally

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/screenplay-nlp
cd screenplay-nlp
pip install -r requirements.txt

# 2. Place your data
# Unzip your Kaggle data into ./data/ so it looks like:
#   data/movie_character_texts/
#   data/rule_based_annotations/
#   data/character_genders.pickle

# 3. Run
streamlit run app.py
```

## Deploy with Docker

```bash
docker build -t screenplay-nlp .
docker run -p 8501:8501 -v $(pwd)/data:/app/data screenplay-nlp
```

## AWS Deployment

Deployed via **AWS App Runner** from an ECR container image.

```bash
# Push to ECR
aws ecr create-repository --repository-name screenplay-nlp
docker tag screenplay-nlp:latest <ECR_URI>
docker push <ECR_URI>
# Then create App Runner service pointing to ECR image
```

---

_Built as a portfolio project exploring NLP techniques on publicly available screenplay data._
