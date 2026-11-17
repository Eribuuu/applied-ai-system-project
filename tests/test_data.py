import math
from pathlib import Path

import pandas as pd
import pytest

from src.rag.data import (
    CATALOG_COLUMNS,
    DATASET_PATH,
    FEATURES,
    clean_spotify,
    frame_from_songs,
    songs_from_frame,
)
from src.recommender import Song, load_songs


def raw_row(track_id, name, artists, genre, popularity=50, tempo=120.0, energy=0.5):
    """One row shaped like the Kaggle Spotify Tracks CSV."""
    return {
        "track_id": track_id, "artists": artists, "album_name": "Album", "track_name": name,
        "popularity": popularity, "duration_ms": 200000, "explicit": False,
        "danceability": 0.5, "energy": energy, "key": 0, "loudness": -6.0, "mode": 1,
        "speechiness": 0.05, "acousticness": 0.3, "instrumentalness": 0.0,
        "liveness": 0.1, "valence": 0.5, "tempo": tempo, "time_signature": 4,
        "track_genre": genre,
    }


def make_raw() -> pd.DataFrame:
    return pd.DataFrame([
        raw_row("a", "Song A", "Artist One", "rock"),
        raw_row("a", "Song A", "Artist One", "alt-rock"),         # same track, second genre
        raw_row("b", "Song B", "Two;Three", "pop", popularity=10, energy=0.2),
        raw_row("c", "song b", "two;three", "dance", popularity=90, energy=0.9),  # same song, other album
        raw_row("d", "No Tempo", "Artist One", "rock", tempo=0.0),
        raw_row("e", None, "Artist One", "rock"),
        raw_row("f", "Song F", "Artist Four", "jazz"),
    ])


def test_clean_collapses_duplicate_songs():
    catalog = clean_spotify(make_raw())
    assert catalog["title"].tolist() == ["Song A", "song b", "Song F"]
    assert list(catalog.columns) == CATALOG_COLUMNS


def test_genres_are_merged_in_file_order():
    catalog = clean_spotify(make_raw())
    assert catalog.loc[0, "genres"] == ["rock", "alt-rock"]
    assert catalog.loc[0, "genre"] == "rock"
    assert catalog.loc[1, "genres"] == ["pop", "dance"]


def test_duplicate_keeps_most_popular_features():
    catalog = clean_spotify(make_raw())
    song_b = catalog.loc[1]
    assert song_b["track_id"] == "c"
    assert song_b["energy"] == 0.9


def test_bad_rows_dropped_and_artists_joined():
    catalog = clean_spotify(make_raw())
    assert "No Tempo" not in catalog["title"].values
    assert catalog["title"].notna().all()
    assert catalog.loc[1, "artist"] == "two, three"


def test_ids_are_sequential():
    catalog = clean_spotify(make_raw())
    assert catalog["id"].tolist() == [0, 1, 2]


def test_songs_from_frame_builds_songs():
    songs = songs_from_frame(clean_spotify(make_raw()))
    assert all(isinstance(s, Song) for s in songs)
    assert [s.id for s in songs] == [0, 1, 2]
    assert songs[0].genre == "rock"
    assert songs[0].tempo_bpm == 120.0


def test_small_catalog_round_trips():
    songs = load_songs("data/songs.csv")
    frame = frame_from_songs(songs)
    assert len(frame) == len(songs)
    assert songs_from_frame(frame) == songs


@pytest.mark.skipif(not Path(DATASET_PATH).exists(), reason="Spotify dataset not downloaded")
def test_real_dataset_is_clean():
    catalog = clean_spotify(pd.read_csv(DATASET_PATH, index_col=0))
    key = catalog["title"].str.lower() + "|" + catalog["artist"].str.lower()
    assert key.is_unique
    assert catalog["tempo_bpm"].gt(0).all()
    for f in ["energy", "valence", "danceability", "acousticness"]:
        assert catalog[f].between(0, 1).all()
    assert catalog[FEATURES].notna().all().all()
    assert not math.isnan(catalog["energy"].mean())
