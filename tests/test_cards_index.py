import hashlib
import json
import re

import numpy as np
import pytest

from src.rag.cards import descriptors, make_card
from src.rag.data import frame_from_songs
from src.rag.index import (
    META_FILE,
    build_index,
    index_is_current,
    load_index,
    make_cards,
    normalize,
)
from src.recommender import load_songs


def row(**overrides):
    base = {
        "title": "Test Song", "artist": "Test Artist", "genres": ["hip-hop", "pop"],
        "energy": 0.5, "valence": 0.5, "danceability": 0.5, "acousticness": 0.5, "tempo_bpm": 100,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("energy, words", [
    (0.0, "very calm, quiet"),
    (0.19, "very calm, quiet"),
    (0.2, "calm, mellow"),
    (0.59, "moderate energy"),
    (0.8, "high-energy, intense"),
    (1.0, "high-energy, intense"),
])
def test_energy_bins(energy, words):
    assert descriptors(row(energy=energy))[0] == words


@pytest.mark.parametrize("valence, words", [
    (0.1, "dark, sad"), (0.3, "melancholic, moody"), (0.5, "bittersweet"),
    (0.7, "upbeat, positive"), (0.95, "happy, cheerful"),
])
def test_valence_bins(valence, words):
    assert descriptors(row(valence=valence))[1] == words


@pytest.mark.parametrize("tempo, words", [
    (60, "slow tempo"), (80, "mid-tempo"), (125, "upbeat tempo"), (140, "fast tempo"),
])
def test_tempo_bins(tempo, words):
    assert words in descriptors(row(tempo_bpm=tempo))


def test_optional_descriptors():
    d = descriptors(row(danceability=0.8, acousticness=0.9, instrumentalness=0.9,
                        speechiness=0.4, explicit=True))
    assert {"very danceable, groovy", "acoustic", "instrumental, no vocals",
            "rap or spoken vocals", "explicit lyrics"} <= set(d)
    assert "electric, studio-produced" in descriptors(row(acousticness=0.05))
    assert "not for dancing" in descriptors(row(danceability=0.2))


def test_middle_values_add_no_extra_words():
    # Mid danceability/acousticness and no extras columns (like songs.csv): only the 3 binned features.
    assert len(descriptors(row())) == 3


def test_card_has_words_not_numbers():
    card = make_card(row(energy=0.873, valence=0.12, tempo_bpm=171.4))
    assert card.startswith("Test Song by Test Artist. Genres: hip hop, pop. Sound: ")
    assert not re.search(r"\d", card)


class FakeEmbedder:
    """Deterministic stand-in for the model: a vector from each text's hash."""
    name = "fake-embedder"

    def embed_documents(self, texts):
        return normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text):
        return normalize(self._vec(text))

    @staticmethod
    def _vec(text):
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=16)


@pytest.fixture
def small_catalog():
    return frame_from_songs(load_songs("data/songs.csv"))


def test_build_index_shapes_and_meta(small_catalog, tmp_path):
    emb = build_index(small_catalog, FakeEmbedder(), out_dir=tmp_path)
    assert emb.shape == (len(small_catalog), 16)
    assert emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-6)
    meta = json.loads((tmp_path / META_FILE).read_text())
    assert meta["model"] == "fake-embedder" and meta["n_songs"] == len(small_catalog)
    np.testing.assert_array_equal(load_index(small_catalog, "fake-embedder", tmp_path), emb)


def test_row_i_is_song_i(small_catalog, tmp_path):
    emb = build_index(small_catalog, FakeEmbedder(), out_dir=tmp_path)
    cards = make_cards(small_catalog)
    np.testing.assert_allclose(emb[3], FakeEmbedder().embed_documents([cards[3]])[0])


def test_stale_index_is_detected(small_catalog, tmp_path):
    build_index(small_catalog, FakeEmbedder(), out_dir=tmp_path)
    assert index_is_current(small_catalog, "fake-embedder", tmp_path)
    assert not index_is_current(small_catalog, "other-model", tmp_path)

    changed = small_catalog.copy()
    changed.loc[0, "energy"] = 1.0 - changed.loc[0, "energy"]  # flips its energy words
    assert not index_is_current(changed, "fake-embedder", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_index(changed, "fake-embedder", tmp_path)


def test_missing_index_is_not_current(small_catalog, tmp_path):
    assert not index_is_current(small_catalog, "fake-embedder", tmp_path)
