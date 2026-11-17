"""
Load and clean the Spotify Tracks dataset into the recommender's Song objects.

The raw Kaggle file lists the same song several times: once per genre it was
filed under, and again for each album it was released on. Cleaning collapses
those into one row per song and caches the result as parquet, so later steps
don't re-clean 114k rows on every run.

Build the cache from the project root with:  python -m src.rag.data
"""

from pathlib import Path
from typing import List

import pandas as pd

try:
    from ..recommender import Song
except ImportError:  # run as `python src/main.py`, where src/ is the top level
    from recommender import Song

DATASET_PATH = Path("data/dataset.csv")
INDEX_DIR = Path("data/index")
CATALOG_PATH = INDEX_DIR / "songs.parquet"

# Spotify column -> name used in the catalog (and by Song where they overlap)
RENAME = {
    "track_name": "title",
    "artists": "artist",
    "album_name": "album",
    "tempo": "tempo_bpm",
}

# The five audio features the Recommender scores, in Recommender.GENRE_FEATURES order.
FEATURES = ["energy", "valence", "danceability", "acousticness", "tempo_bpm"]

# Kept for later steps (descriptor words, filters); not scored yet.
EXTRAS = ["popularity", "explicit", "instrumentalness", "speechiness"]

CATALOG_COLUMNS = ["id", "track_id", "title", "artist", "album", "genre", "genres", *FEATURES, *EXTRAS]


def clean_spotify(raw: pd.DataFrame) -> pd.DataFrame:
    """
    One row per song, with a stable integer id.

    - Drops rows with no title or artist, and rows with tempo 0 (unmeasured).
    - "A;B" artist lists become "A, B".
    - A song is identified by title + artist (case-insensitive). Duplicates
      keep the most popular row's features, and `genres` collects every genre
      the song was filed under, in file order. `genre` is the first of them.
    - Rows keep their original file order, so ids are reproducible.
    """
    df = raw.rename(columns=RENAME)
    df = df.dropna(subset=["title", "artist"])
    df = df[df["tempo_bpm"] > 0]

    df = df.assign(
        title=df["title"].str.strip(),
        artist=df["artist"].str.split(";").map(lambda parts: ", ".join(p.strip() for p in parts)),
        track_genre=df["track_genre"].str.strip().str.lower(),
    )
    df = df.assign(_key=df["title"].str.lower() + "\x1f" + df["artist"].str.lower())

    genres = df.groupby("_key", sort=False)["track_genre"].agg(lambda s: list(dict.fromkeys(s)))

    df = (
        df.sort_values("popularity", ascending=False, kind="stable")
        .drop_duplicates("_key")
        .sort_index()
        .reset_index(drop=True)
    )
    df["genres"] = df["_key"].map(genres)
    df["genre"] = df["genres"].str[0]
    df["id"] = df.index
    return df[CATALOG_COLUMNS]


def build_catalog(csv_path: Path = DATASET_PATH, out_path: Path = CATALOG_PATH) -> pd.DataFrame:
    """Clean the raw CSV and write the parquet cache."""
    catalog = clean_spotify(pd.read_csv(csv_path, index_col=0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out_path, index=False)
    return catalog


def load_catalog(
    csv_path: Path = DATASET_PATH, cache_path: Path = CATALOG_PATH, rebuild: bool = False
) -> pd.DataFrame:
    """The cleaned catalog, rebuilt when the cache is missing or older than the CSV."""
    stale = not cache_path.exists() or cache_path.stat().st_mtime < csv_path.stat().st_mtime
    if rebuild or stale:
        return build_catalog(csv_path, cache_path)
    return pd.read_parquet(cache_path)


def songs_from_frame(catalog: pd.DataFrame) -> List[Song]:
    """Song objects in catalog order, so songs[i].id == i."""
    return [
        Song(
            id=int(row.id),
            title=row.title,
            artist=row.artist,
            genre=row.genre,
            energy=float(row.energy),
            tempo_bpm=float(row.tempo_bpm),
            valence=float(row.valence),
            danceability=float(row.danceability),
            acousticness=float(row.acousticness),
        )
        for row in catalog.itertuples(index=False)
    ]


def frame_from_songs(songs: List[Song]) -> pd.DataFrame:
    """
    A catalog-shaped frame from Song objects (e.g. the 20-song songs.csv), so
    later steps can run on the small catalog. Columns Song doesn't have are left out.
    """
    return pd.DataFrame(
        {
            "id": [s.id for s in songs],
            "title": [s.title for s in songs],
            "artist": [s.artist for s in songs],
            "genre": [s.genre for s in songs],
            "genres": [[s.genre] for s in songs],
            **{f: [getattr(s, f) for s in songs] for f in FEATURES},
        }
    )


if __name__ == "__main__":
    catalog = build_catalog()
    multi = (catalog["genres"].str.len() > 1).sum()
    print(f"Wrote {CATALOG_PATH}: {len(catalog):,} songs, {catalog['genre'].nunique()} genres")
    print(f"{multi:,} songs are filed under more than one genre")
