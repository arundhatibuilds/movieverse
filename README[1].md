# 🎬 Netflix Movie & TV Show Recommender

A content-based recommendation system for Netflix movies and TV shows. It combines each
title's genres, director, cast, and description into a single text "tag", vectorizes it
with **TF-IDF**, and finds similar titles using **cosine similarity via NearestNeighbors**.
Posters are fetched live from the **TMDB API**, with caching and a fallback image if a
poster isn't found.

## How it works

1. Two catalogs (movies + TV shows) are merged into one dataframe.
2. Each title gets a `tags` field: `genres + director + cast + description`.
3. `TfidfVectorizer` turns tags into vectors; `NearestNeighbors(metric="cosine")` finds the
   closest titles to whatever you search for.
4. Posters are looked up on TMDB by title and cached for 24 hours to avoid hammering the API.

## Project structure

```
.
├── app.py                # Streamlit app
├── requirements.txt
├── README.md
└── data/
    ├── netflix_movies_detailed_up_to_2025.csv
    └── netflix_tv_shows_detailed_up_to_2025.csv
```

**Important:** the two CSV files must exist in a `data/` folder next to `app.py`. They
aren't included here — copy the same files you used in the notebook into `data/` before
running or deploying.

## Setup (local)

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Add your TMDB API key (free, from https://www.themoviedb.org/settings/api) as an
environment variable, or in `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml
TMDB_API_KEY = "your_key_here"
```

Then run:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

If you skip the TMDB key, the app still works — it just shows a placeholder poster image
instead of real ones.

## Deployment options

### Option 1: Streamlit Community Cloud (recommended, free, easiest)

This is the simplest path — no server/port configuration needed at all.

1. Push this project (app.py, requirements.txt, data/ folder) to a **public GitHub repo**.
   - If your CSVs are large (>25MB combined), consider Git LFS, or host the CSVs somewhere
     and download them at startup instead of committing them.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, and set the main file path to `app.py`.
4. Under **"Advanced settings" → Secrets**, add:
   ```toml
   TMDB_API_KEY = "your_key_here"
   ```
5. Click **Deploy**. You'll get a public URL like `https://your-app.streamlit.app`.

This avoids the port-visibility issue you ran into with Codespaces entirely, since
Streamlit Cloud handles hosting and networking for you.

### Option 2: GitHub Codespaces (what you were already using)

If you want to keep using Codespaces to test before deploying elsewhere:

1. Open your repo in a Codespace.
2. Install deps: `pip install -r requirements.txt`
3. Run: `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
4. Open the **"Ports"** tab in VS Code (bottom panel).
5. Find port **8501**, right-click it → **Port Visibility → Public**.
   - This is the step that was blocking you — by default Codespaces ports are Private,
     so the forwarded URL returns a 401/blank page until you flip it to Public.
6. Click the forwarded address (or the globe icon) next to port 8501 to open the app.

Note Codespaces is meant for development, not permanent hosting — the URL only stays
alive while the Codespace is running. Use it for testing, then deploy properly via
Option 1 (or Option 3) for something you can share long-term.

### Option 3: Other free/low-cost hosts

If you outgrow Streamlit Cloud's resource limits, the same `app.py` also runs fine on:
- **Render** (Web Service, `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`)
- **Hugging Face Spaces** (choose the "Streamlit" SDK when creating a Space)
- **Railway** or **Fly.io** (Docker or buildpack deploys)

All of these just need `requirements.txt`, `app.py`, and your data files (or a way to
download them at startup).

## Notes / possible improvements

- The current matching is exact-title based after you pick from the search dropdown —
  this avoids the "movie not found" issue from typos.
- If your dataset is large, consider precomputing and saving the TF-IDF matrix (e.g. with
  `joblib`) instead of rebuilding it on every cold start, to speed up first load.
- Poster caching is in-memory per session (`st.cache_data` with a 24h TTL) — for a
  production app with many users, a persistent cache (e.g. a small SQLite file or Redis)
  would cut down TMDB API calls further.
