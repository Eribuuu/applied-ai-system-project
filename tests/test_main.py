import io

import pytest
from conftest import FakeEmbedder

from src.main import (
    build_parser,
    describe_profile,
    flags_from_args,
    genre_warning,
    main,
    print_results,
)
from src.rag.data import frame_from_songs
from src.rag.feature_scorer import FeatureScorer
from src.rag.generate import Explainer, LLMUnavailable
from src.rag.index import make_cards
from src.rag.retrieve import HybridRetriever
from src.recommender import Recommender, UserProfile, load_songs


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_no_flags_means_none():
    assert flags_from_args(parse("--query", "chill")) is None


def test_flags_become_a_profile():
    p = flags_from_args(parse("--genre", "rap", "--energy", "0.9", "--tempo", "140"))
    assert p == UserProfile("rap", 0.9, None, None, 140.0, None)


@pytest.mark.parametrize("argv", [["--energy", "1.5"], ["--valence", "-0.1"], ["--tempo", "0"]])
def test_out_of_range_flags_are_rejected(argv):
    with pytest.raises(SystemExit):
        parse(*argv)


def test_empty_query_without_flags_is_an_error():
    with pytest.raises(SystemExit):
        main(["--query", "   "])


def test_demo_flag_runs_the_original_demo(capsys):
    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "Loaded Songs: 20" in out and "Profile: Deep Intense Rock" in out


KNOWN = ["hip hop", "rock", "study", "synth pop"]


@pytest.mark.parametrize("typed", ["rock", "Hip-Hop", "rap", "lofi"])
def test_known_or_aliased_genre_has_no_warning(typed):
    assert genre_warning(typed, KNOWN) is None


def test_unknown_genre_warns_with_suggestion():
    note = genre_warning("synthpop", KNOWN)
    assert "isn't a genre" in note and "synth pop" in note


def test_describe_profile_skips_unset():
    assert describe_profile(UserProfile("rock", 0.9, None, None, 150, None)) == "genre rock, energy 0.90, tempo 150 BPM"


class DownClient:
    def complete(self, *a):
        raise LLMUnavailable("connection refused")


@pytest.fixture(scope="module")
def small_result():
    songs = load_songs("data/songs.csv")
    frame = frame_from_songs(songs)
    emb = FakeEmbedder()
    r = HybridRetriever(FeatureScorer(Recommender(songs)), emb.embed_documents(make_cards(frame)), emb,
                        dense_n=20, prf_n=5)
    return frame.to_dict("records"), r.retrieve(query="upbeat pop", k=3, min_score=0.0)


def render(small_result, client, query="upbeat pop", **kw):
    rows, result = small_result
    exp = Explainer(rows, client).explain(query if kw.get("llm_requested", True) else "", result)
    out = io.StringIO()
    print_results(query, result, exp, verbose=kw.get("verbose", False),
                  llm_requested=kw.get("llm_requested", True), out=out)
    return out.getvalue()


def test_output_lists_songs_with_explanations(small_result):
    text = render(small_result, DownClient())
    _, result = small_result
    assert 'Query: "upbeat pop"' in text
    assert f"1. {result.hits[0].song.title} — {result.hits[0].song.artist}" in text
    assert "Because: " in text


def test_ollama_down_tells_you_how_to_start_it(small_result):
    assert "ollama serve" in render(small_result, DownClient())


def test_no_llm_flag_prints_no_fallback_note(small_result):
    text = render(small_result, DownClient(), llm_requested=False)
    assert "template reasons" not in text and "Because: " in text


def test_verbose_shows_profile_and_notes(small_result):
    text = render(small_result, DownClient(), verbose=True)
    assert "Profile (pseudo-relevance" in text and "Explanation notes:" in text
