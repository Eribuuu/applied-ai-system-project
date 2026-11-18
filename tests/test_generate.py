import json
import os

import pytest
from conftest import FakeEmbedder

from src.rag.data import frame_from_songs
from src.rag.feature_scorer import FeatureScorer
from src.rag.generate import (
    Explainer,
    LLMUnavailable,
    OllamaClient,
    build_context,
    template_text,
)
from src.rag.index import make_cards
from src.rag.retrieve import HybridRetriever
from src.recommender import Recommender, UserProfile, load_songs

QUERY = "upbeat pop for a road trip"


class FakeClient:
    """Returns a canned reply and records what it was sent."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system, user, schema):
        self.calls.append(user)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


@pytest.fixture(scope="module")
def setup():
    songs = load_songs("data/songs.csv")
    frame = frame_from_songs(songs)
    embedder = FakeEmbedder()
    retriever = HybridRetriever(FeatureScorer(Recommender(songs)), embedder.embed_documents(make_cards(frame)),
                                embedder, dense_n=20, prf_n=5)
    result = retriever.retrieve(query=QUERY, k=3, min_score=0.0)
    return frame.to_dict("records"), result


def explain(setup, reply, query=QUERY):
    rows, result = setup
    client = FakeClient(reply)
    return Explainer(rows, client).explain(query, result), client


def good_reply(n=3):
    return {"summary": "Bright, energetic picks.", "picks": [{"id": f"S{i}", "why": f"Fits well, song {i}."} for i in range(1, n + 1)]}


def test_context_states_values_and_targets(setup):
    rows, result = setup
    ctx = build_context(QUERY, result, rows)
    assert f'Listener request: "{QUERY}"' in ctx
    assert ctx.count("[S") == 3
    first = result.hits[0].song
    assert f'"{first.title}" by {first.artist}' in ctx
    assert f"energy {first.energy:.2f} (target {result.profile.target_energy:.2f})" in ctx
    assert "within" not in ctx  # no "within X of target" phrasing for the model to misread


def test_valid_reply_is_used(setup):
    exp, _ = explain(setup, good_reply())
    assert exp.used_llm and exp.issues == []
    assert exp.summary == "Bright, energetic picks."
    assert [s.source for s in exp.songs] == ["llm"] * 3
    assert exp.songs[1].text == "Fits well, song 2."


def test_output_keeps_retrieval_order(setup):
    reply = good_reply()
    reply["picks"].reverse()
    exp, _ = explain(setup, reply)
    _, result = setup
    assert [s.hit for s in exp.songs] == result.hits
    assert exp.songs[0].text == "Fits well, song 1."


def test_not_json_falls_back_to_templates(setup):
    exp, _ = explain(setup, "Sure! Here are some great songs...")
    assert not exp.used_llm and exp.summary is None
    assert all(s.source == "template" and s.text.startswith("Because: ") for s in exp.songs)


def test_unknown_song_id_is_ignored(setup):
    reply = good_reply()
    reply["picks"].append({"id": "S9", "why": "A song that was never retrieved."})
    exp, _ = explain(setup, reply)
    assert [s.source for s in exp.songs] == ["llm"] * 3
    assert any("unknown song id 'S9'" in i for i in exp.issues)


def test_skipped_song_gets_template_text(setup):
    exp, _ = explain(setup, good_reply(n=2))
    assert [s.source for s in exp.songs] == ["llm", "llm", "template"]
    assert exp.songs[2].text == template_text(exp.songs[2].hit)
    assert any("S3 was not explained" in i for i in exp.issues)


def test_invented_number_is_replaced(setup):
    reply = good_reply()
    reply["picks"][0]["why"] = "Released in 1987 with a 300 BPM drum line."
    exp, _ = explain(setup, reply)
    assert exp.songs[0].source == "template"
    assert any("S1 numbers not in the context: 300, 1987" in i for i in exp.issues)


def test_numbers_from_context_are_allowed(setup):
    _, result = setup
    song = result.hits[0].song
    reply = good_reply()
    reply["picks"][0]["why"] = f"Energy {song.energy} sits right on target."
    exp, _ = explain(setup, reply)
    assert exp.songs[0].source == "llm"


def test_reference_to_unretrieved_song_is_replaced(setup):
    reply = good_reply()
    reply["picks"][1]["why"] = "Pairs nicely with S7."
    reply["summary"] = "S1 and S8 lead the list."
    exp, _ = explain(setup, reply)
    assert exp.songs[1].source == "template"
    assert exp.summary is None


def test_llm_unavailable_falls_back(setup):
    exp, _ = explain(setup, LLMUnavailable("connection refused"))
    assert not exp.used_llm
    assert all(s.source == "template" for s in exp.songs)
    assert "LLM unavailable" in exp.issues[0]


def test_flags_only_skips_the_llm(setup):
    exp, client = explain(setup, good_reply(), query="")
    assert client.calls == []
    assert all(s.source == "template" for s in exp.songs)


def test_no_hits_no_call(setup):
    rows, result = setup
    client = FakeClient(good_reply())
    empty = type(result)(hits=[], profile=result.profile, profile_source="flags", query_genres=[])
    exp = Explainer(rows, client).explain(QUERY, empty)
    assert exp.songs == [] and client.calls == []


def test_unreachable_server_raises_llm_unavailable():
    client = OllamaClient(url="http://127.0.0.1:9", timeout=2)
    with pytest.raises(LLMUnavailable):
        client.complete("system", "user", {"type": "object"})


@pytest.mark.skipif(not os.environ.get("RUN_OLLAMA_TESTS"), reason="set RUN_OLLAMA_TESTS=1 with Ollama running")
def test_live_qwen_reply_passes_guardrail(setup):
    rows, result = setup
    exp = Explainer(rows).explain(QUERY, result)
    assert exp.used_llm
    assert len(exp.songs) == len(result.hits)
