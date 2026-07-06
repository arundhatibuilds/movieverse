import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Netflix Recommender", page_icon="🎬", layout="wide")

DATA_DIR = "data"
MOVIES_CSV = os.path.join(DATA_DIR, "netflix_movies_detailed_up_to_2025.csv")
TV_CSV = os.path.join(DATA_DIR, "netflix_tv_shows_detailed_up_to_2025.csv")

# TMDB key: set via environment variable or Streamlit secrets (see README)
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", st.secrets.get("TMDB_API_KEY", "") if hasattr(st, "secrets") else "")
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"
PLACEHOLDER_POSTER = "https://placehold.co/342x513?text=No+Poster"


# ----------------------------------------------------------------------------
# DATA LOADING + MODEL BUILDING (cached so this only runs once per session)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    movies = pd.read_csv(MOVIES_CSV)
    tv = pd.read_csv(TV_CSV)

    netflix = pd.concat([movies, tv], ignore_index=True)
    netflix = netflix[["title", "genres", "director", "cast", "description"]]
    netflix.fillna("", inplace=True)
    netflix.drop_duplicates(subset="title", inplace=True)
    netflix.reset_index(drop=True, inplace=True)

    netflix["tags"] = (
        netflix["genres"] + " " + netflix["director"] + " " + netflix["cast"] + " " + netflix["description"]
    )
    return netflix


@st.cache_resource(show_spinner=False)
def build_model(tags: pd.Series):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(tags)

    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=11)
    model.fit(vectors)
    return tfidf, vectors, model


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster(title: str) -> str:
    """Look up a poster on TMDB by title, with graceful fallback."""
    if not TMDB_API_KEY:
        return PLACEHOLDER_POSTER
    try:
        resp = requests.get(
            "https://api.themoviedb.org/3/search/multi",
            params={"api_key": TMDB_API_KEY, "query": title},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        for r in results:
            poster_path = r.get("poster_path")
            if poster_path:
                return TMDB_IMG_BASE + poster_path
        return PLACEHOLDER_POSTER
    except Exception:
        return PLACEHOLDER_POSTER


# ----------------------------------------------------------------------------
# RECOMMENDATION LOGIC
# ----------------------------------------------------------------------------
def recommend(netflix: pd.DataFrame, vectors, model, title: str, n: int = 10):
    matches = netflix[netflix["title"].str.lower() == title.lower()]
    if matches.empty:
        return None, []

    idx = matches.index[0]
    distances, indices = model.kneighbors(vectors[idx], n_neighbors=n + 1)

    results = []
    for i in indices.flatten()[1:]:
        results.append(netflix.iloc[i]["title"])
    return netflix.loc[idx, "genres"], results


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🎬 Netflix Movie & TV Show Recommender")
st.caption("Content-based recommendations using TF-IDF + cosine similarity over genres, cast, director, and description.")

if not os.path.exists(MOVIES_CSV) or not os.path.exists(TV_CSV):
    st.error(
        f"Data files not found. Please make sure both CSVs are present in the `{DATA_DIR}/` folder:\n\n"
        f"- {MOVIES_CSV}\n- {TV_CSV}"
    )
    st.stop()

with st.spinner("Loading data and building model..."):
    netflix = load_data()
    tfidf, vectors, model = build_model(netflix["tags"])

all_titles = sorted(netflix["title"].unique().tolist())

query = st.text_input("Search for a movie or TV show", placeholder="e.g. Avatar, Wednesday, Stranger Things")

selected_title = None
if query:
    matching = [t for t in all_titles if query.lower() in t.lower()]
    if matching:
        selected_title = st.selectbox("Matching titles", matching)
    else:
        st.warning("No matching titles found. Try a different search term.")

n_recs = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if selected_title and st.button("Recommend", type="primary"):
    genres, recs = recommend(netflix, vectors, model, selected_title, n=n_recs)

    if not recs:
        st.error("Movie/Series not found!")
    else:
        st.subheader(f"Because you liked: {selected_title}")
        st.caption(f"Genre(s): {genres}")

        cols = st.columns(5)
        for i, rec_title in enumerate(recs):
            with cols[i % 5]:
                poster_url = fetch_poster(rec_title)
                st.image(poster_url, use_container_width=True)
                st.markdown(f"**{rec_title}**")

st.divider()
st.caption(
    "Data: Netflix movies & TV shows catalog. Posters via TMDB API (set TMDB_API_KEY to enable). "
    "Built with Streamlit, scikit-learn, and pandas."
)
