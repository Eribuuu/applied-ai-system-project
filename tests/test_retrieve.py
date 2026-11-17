import numpy as np
import pytest
from conftest import FakeEmbedder

from src.rag.data import frame_from_songs
from src.rag.feature_scorer import FeatureScorer
from src.rag.index import make_cards
from src.rag.retrieve import HybridRetriever, rrf_fuse
from src.recommender import Recommender, UserProfile, load_songs

NONE = dict(favorite_genre=None, target_energy=None, target_valence=None,
            target_danceability=None, target_tempo_bpm=None, target_acousticness=None)


def profile(**targets):
    return UserProfile(**{**NONE, **targets})


# --- RRF on hand-made rankings -------------------------------------------------

def test_rrf_item_in_both_lists_wins():
    fused = rrf_fuse({"a": [1, 2, 3], "b": [3, 4, 5]}, k=60)
    assert fused[0][0] == 3
    assert fused[0][1] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_single_list_keeps_its_order():
    assert [i for i, _ in rrf_fuse({"a": [7, 3, 9]})] == [7, 3, 9]


def test_rrf_weights_shift_the_order():
    rankings = {"a": [1, 2], "b": [2, 1]}
    assert rrf_fuse(rankings, {"a": 2.0, "b": 1.0})[0][0] == 1
    assert rrf_fuse(rankings, {"a": 1.0, "b": 2.0})[0][0] == 2


def test_rrf_ties_keep_first_seen_order():
    assert [i for i, _ in rrf_fuse({"a": [5], "b": [8]})] == [5, 8]


# --- the retriever on the 20-song catalog ---------------------------------------

@pytest.fixture(scope="module")
def retriever():
    songs = load_songs("data/songs.csv")
    scorer = FeatureScorer(Recommender(songs))
    embedder = FakeEmbedder()
    embeddings = embedder.embed_documents(make_cards(frame_from_songs(songs)))
    return HybridRetriever(scorer, embeddings, embedder, dense_n=20, prf_n=5)


def test_needs_a_query_or_a_target(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve()
    with pytest.raises(ValueError):
        retriever.retrieve(query="   ", flags=profile())


def test_flags_only_matches_recommender(retriever):
    user = UserProfile("lofi", 0.38, 0.58, 0.60, 76, 0.80)
    result = retriever.retrieve(flags=user, k=5, min_score=0.5)
    expected = retriever.rec.recommend_with_details(user, k=5, min_score=0.5)
    assert [h.song for h in result.hits] == [r.song for r in expected]
    assert [h.reasons for h in result.hits] == [r.reasons for r in expected]
    assert result.profile_source == "flags"
    assert all(h.dense_rank is None for h in result.hits)


def test_query_only_builds_profile_from_top_hits(retriever):
    result = retriever.retrieve(query="calm study music", k=5, min_score=0.0)
    assert result.profile_source == "pseudo-relevance"
    top = [i for i, _ in retriever.dense("calm study music", 5)]
    expected_energy = np.mean([retriever.rec.songs[i].energy for i in top])
    assert result.profile.target_energy == pytest.approx(expected_energy)
    assert result.hits


def test_flags_override_pseudo_relevance(retriever):
    result = retriever.retrieve(query="calm study music", flags=profile(target_energy=0.95), min_score=0.0)
    assert result.profile_source == "flags + pseudo-relevance"
    assert result.profile.target_energy == 0.95
    assert result.profile.target_valence is not None  # filled from the dense hits


def test_min_score_filters_every_hit(retriever):
    result = retriever.retrieve(query="anything", flags=profile(target_energy=0.9), k=20, min_score=0.7)
    assert result.hits
    assert all(h.feature_score >= 0.7 for h in result.hits)


def test_hits_carry_evidence_from_both_retrievers(retriever):
    result = retriever.retrieve(query="upbeat pop", flags=profile(target_energy=0.8), k=20, min_score=0.0)
    rrf = [h.rrf for h in result.hits]
    assert rrf == sorted(rrf, reverse=True)
    for h in result.hits:
        assert h.dense_rank is not None and h.feature_rank is not None  # 20-song catalog: every song is in both
        assert h.reasons


def test_mask_limits_results(retriever):
    mask = np.array([s.genre == "lofi" for s in retriever.rec.songs])
    result = retriever.retrieve(query="music", k=10, min_score=0.0, mask=mask)
    assert result.hits and all(h.song.genre == "lofi" for h in result.hits)


def test_text_query_needs_an_index():
    rec = Recommender(load_songs("data/songs.csv"))
    flags_only = HybridRetriever(FeatureScorer(rec))
    assert flags_only.retrieve(flags=profile(target_energy=0.5), min_score=0.0).hits
    with pytest.raises(RuntimeError):
        flags_only.retrieve(query="chill")
