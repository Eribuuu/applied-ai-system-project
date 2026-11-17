import pytest

from src.rag.genres import genres_in_query, resolve_genre

# Normalized names, as FeatureScorer.genre_vocab holds them for the Spotify catalog.
SPOTIFY = {"hip hop", "r n b", "pop", "synth pop", "study", "sad", "acoustic",
           "heavy metal", "metal", "jazz", "electronic", "drum and bass"}


@pytest.mark.parametrize("typed, expected", [
    ("rap", "hip hop"),
    ("Hip-Hop", "hip hop"),
    ("R&B", "r n b"),
    ("lofi", "study"),
    ("lo-fi", "study"),
    ("DnB", "drum and bass"),
    ("k-pop", "k-pop"),  # not in this catalog: unchanged, word-overlap fallback applies
])
def test_resolve_genre(typed, expected):
    assert resolve_genre(typed, SPOTIFY) == expected


def test_catalog_name_beats_alias():
    # songs.csv really has a "lofi" genre, so it isn't remapped to "study".
    assert resolve_genre("lofi", {"lofi", "study"}) == "lofi"


def test_alias_to_missing_genre_is_ignored():
    assert resolve_genre("rap", {"pop"}) == "rap"
    assert genres_in_query("some rap", {"pop"}) == []


@pytest.mark.parametrize("query, expected", [
    ("aggressive rap with heavy beats", ["hip hop"]),
    ("feel-good retro synth pop", ["synth pop"]),          # longest phrase wins, no extra "pop"
    ("angry heavy metal for the gym", ["heavy metal"]),
    ("sad acoustic breakup songs", ["sad", "acoustic"]),   # two genres, in query order
    ("mellow songs for late-night studying", []),          # whole words only
    ("Some R&B, please!", ["r n b"]),
    ("lofi beats", ["study"]),
    ("rap and jazz", ["hip hop", "jazz"]),
])
def test_genres_in_query(query, expected):
    assert genres_in_query(query, SPOTIFY) == expected
