import numpy as np
import pytest

from src.rag.feature_scorer import FeatureScorer
from src.recommender import Recommender, Song, UserProfile, load_songs

# The three profiles from src/main.py, plus partial ones that skip features.
PROFILES = [
    UserProfile("pop", 0.92, 0.75, 0.88, 130, 0.05),
    UserProfile("lofi", 0.38, 0.58, 0.60, 76, 0.80),
    UserProfile("rock", 0.92, 0.40, 0.60, 150, 0.08),
    UserProfile(None, 0.5, None, None, 100, None),
    UserProfile("k-pop", None, 0.9, None, None, None),  # genre not in catalog: word-overlap fallback
]


@pytest.fixture(scope="module")
def small():
    rec = Recommender(load_songs("data/songs.csv"))
    return rec, FeatureScorer(rec)


@pytest.mark.parametrize("user", PROFILES)
def test_scores_match_recommender(small, user):
    rec, scorer = small
    expected = [rec.weighted_score(rec.closeness(user, s)) for s in rec.songs]
    np.testing.assert_allclose(scorer.scores(user), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("user", PROFILES)
def test_top_matches_recommender_order(small, user):
    rec, scorer = small
    expected = rec.recommend_with_details(user, k=5, min_score=0.5)
    got = scorer.top(user, n=5, min_score=0.5)
    assert [i for i, _ in got] == [rec.songs.index(r.song) for r in expected]


def test_per_song_closeness_gives_same_reasons(small):
    rec, scorer = small
    user = PROFILES[0]
    c = scorer.closeness(user)
    for i, song in enumerate(rec.songs):
        assert rec.reasons(user, song, scorer.song_closeness(c, i)) == rec.score(user, song)[1]


def test_no_preferences_scores_zero(small):
    _, scorer = small
    empty = UserProfile(None, None, None, None, None, None)
    assert not scorer.scores(empty).any()
    assert scorer.top(empty, n=5, min_score=0.1) == []


def test_mask_limits_results(small):
    rec, scorer = small
    mask = np.array([s.genre == "lofi" for s in rec.songs])
    got = scorer.top(PROFILES[0], n=20, mask=mask)
    assert got and all(rec.songs[i].genre == "lofi" for i, _ in got)


def make_song(id, genre, energy):
    return Song(id=id, title=f"S{id}", artist="A", genre=genre, energy=energy,
                tempo_bpm=120, valence=0.5, danceability=0.5, acousticness=0.3)


def test_multi_genre_song_gets_best_genre_match():
    songs = [make_song(0, "acoustic", 0.3), make_song(1, "j-pop", 0.7), make_song(2, "acoustic", 0.35)]
    rec = Recommender(songs)
    scorer = FeatureScorer(rec, genres=[["acoustic", "j-pop"], ["j-pop"], ["acoustic"]])
    g = scorer.genre_closeness("j-pop")
    assert g[0] == 1.0  # filed under j-pop as its second genre
    assert g[1] == 1.0
    assert g[2] < 1.0
    assert scorer.matched_genre(0, "j-pop") == "j-pop"
    assert scorer.matched_genre(0, "acoustic") == "acoustic"


def test_reasons_name_the_matched_genre():
    from src.rag.retrieve import HybridRetriever

    songs = [make_song(0, "electro", 0.9), make_song(1, "pop", 0.4), make_song(2, "hip-hop", 0.6)]
    scorer = FeatureScorer(Recommender(songs), genres=[["electro", "hip-hop"], ["pop"], ["hip-hop"]])
    hits = HybridRetriever(scorer).retrieve(flags=UserProfile("rap", 0.9, None, None, None, None), k=3,
                                            min_score=0.0).hits
    electro = next(h for h in hits if h.index == 0)
    assert "same genre (hip-hop)" in electro.reasons
    assert electro.song.genre == "electro"  # the Song itself is unchanged


def test_genres_must_align_with_songs(small):
    rec, _ = small
    with pytest.raises(ValueError):
        FeatureScorer(rec, genres=[["pop"]])
